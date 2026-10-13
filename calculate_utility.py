import os
import re
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyterrier as pt

# Requires installation of bert_score: pip install bert-score
from bert_score import BERTScorer

# The prompts in prompts/ ask the model to end its answer with this marker
STOP_MARKER = "STOP"

# ==========================================
# 1. LOAD RETRIEVAL RUN (BM25 >> MonoT5 TOP-100) & QRELS
# ==========================================
def load_run(run_path, max_k):
    """Loads a *_top100.csv run, keeps the top max_k per query and adds passage text."""
    run = pd.read_csv(run_path, dtype={"qid": str, "docno": str})
    run = run.sort_values(["qid", "rank"]).groupby("qid").head(max_k)

    get_text = pt.text.get_text(pt.get_dataset("irds:msmarco-passage"), "text")
    run = get_text(run)
    return run.sort_values(["qid", "rank"]).reset_index(drop=True)


def load_gold_passages(qrels_path):
    """Returns {qid: [passage texts]} for the passages judged with relevance 2 or 3."""
    qrels = pd.read_csv(qrels_path, dtype={"qid": str, "docno": str})
    qrels = qrels[qrels["label"].isin([2, 3])]
    return qrels.groupby("qid")["text"].apply(list).to_dict()


def find_run(system, query_set):
    """Path of the context run retr_results/{system}_{query_set}_top*.csv (e.g. monot5, oracle-rel)."""
    matches = sorted(f for f in os.listdir("retr_results")
                     if f.startswith(f"{system}_{query_set}_top") and f.endswith(".csv"))
    if not matches:
        raise FileNotFoundError(f"No run retr_results/{system}_{query_set}_top*.csv")
    return os.path.join("retr_results", matches[0])


def load_ndcg_at_k(system, query_set):
    """Per-query nDCG@k (columns qid, k, ndcg_at_k) written by ndcg_at_k.py."""
    ndcg = pd.read_csv(os.path.join("retr_results", f"{system}_{query_set}_per_query_ndcg_at_k.csv"), dtype={"qid": str})
    return ndcg.rename(columns={"ndcg": "ndcg_at_k"})


def infer_query_set(run_path):
    """Finds which TREC DL query set (dl19 / dl20) a run file belongs to."""
    match = re.search(r"dl(19|20)", os.path.basename(run_path))
    if match is None:
        raise ValueError(f"Cannot infer dl19/dl20 from '{run_path}'; pass --qrels explicitly.")
    return f"dl{match.group(1)}"

# ==========================================
# 2. LLM BACKENDS (Hugging Face transformers or llama.cpp GGUF)
# ==========================================
class HFBackend:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct", batch_size=4,
                 max_length=2048, max_new_tokens=150, chat_template_kwargs=None, dtype="bfloat16"):
        """
        Hugging Face transformers backend on Apple Silicon (MPS).
        chat_template_kwargs are passed to the chat template; by default thinking mode is
        disabled (enable_thinking=False, used by Qwen3/Qwen3.5; ignored by other templates).
        dtype: "bfloat16" (default) or "float32"; models with linear-attention layers such as
        Qwen3.5 are not precise enough in bfloat16 for the logprob differences we compute.
        Note: batch_size 4 keeps an 8B model in BF16 within 24GB of unified memory.
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading Hugging Face model {model_id} on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            # Llama requires explicit pad token assignment
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            # SDPA attention on MPS gives corrupted outputs for prompts > ~1k tokens in half precision
            attn_implementation="eager" if self.device.type == "mps" and torch_dtype != torch.float32 else "sdpa"
        ).to(self.device)
        self.model.eval()

    def compute_batch_logprobs(self, contexts, targets):
        """
        Computes log P(target | context) for each pair and the number of scored target
        tokens (used for per-token normalisation). Context and target are tokenised
        separately so that exactly the target tokens are scored; sequences are right-padded
        (left padding yields NaNs on MPS) and truncated from the left so the target
        is never cut.
        """
        compiled_logprobs, compiled_token_counts = [], []
        pad_id = self.tokenizer.pad_token_id

        with torch.no_grad():
            for idx in range(0, len(contexts), self.batch_size):
                ctx_ids = self.tokenizer(contexts[idx:idx+self.batch_size]).input_ids
                tgt_ids = self.tokenizer(targets[idx:idx+self.batch_size], add_special_tokens=False).input_ids

                sequences = [(c + t)[-self.max_length:] for c, t in zip(ctx_ids, tgt_ids)]
                target_lens = [min(len(t), len(s) - 1) for t, s in zip(tgt_ids, sequences)]
                seq_len = max(len(s) for s in sequences)

                input_ids = torch.full((len(sequences), seq_len), pad_id, dtype=torch.long)
                attention_mask = torch.zeros_like(input_ids)
                for i, seq in enumerate(sequences):
                    input_ids[i, :len(seq)] = torch.tensor(seq)
                    attention_mask[i, :len(seq)] = 1

                logits = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                ).logits
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite logits from the LLM; check the model dtype/device.")

                # Score only the target tokens of each row (the logit at position t predicts token t+1)
                for i, (seq, t_len) in enumerate(zip(sequences, target_lens)):
                    start, end = len(seq) - t_len, len(seq)
                    row_logprobs = torch.log_softmax(logits[i, start-1:end-1, :].float(), dim=-1)
                    row_targets = input_ids[i, start:end].to(self.device)
                    compiled_logprobs.append(row_logprobs.gather(-1, row_targets.unsqueeze(-1)).sum().item())
                    compiled_token_counts.append(t_len)
        return compiled_logprobs, compiled_token_counts

    def generate_answer(self, instruction, prompt):
        """
        Generates text (RAG answer) from the local LLM (greedy decoding). The instruction
        (from prompts/) is the system message and the prompt (context + question) the user
        message. Generation stops at the STOP marker the instructions ask for.
        """
        if self.tokenizer.chat_template:
            messages = [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}]
            try:
                input_ids = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt", **self.chat_template_kwargs)
            except Exception:
                # Some chat templates have no system role: prepend the instruction to the user message
                input_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{instruction}\n\n{prompt}"}], add_generation_prompt=True,
                    return_tensors="pt", **self.chat_template_kwargs)
        else:
            input_ids = self.tokenizer(f"{instruction}\n\n{prompt}", return_tensors="pt").input_ids

        # transformers >= 5 returns a BatchEncoding instead of a tensor
        if hasattr(input_ids, "keys"):
            input_ids = input_ids["input_ids"]

        # Multimodal configs (e.g. Qwen3.5) keep the context length in text_config
        config = getattr(self.model.config, "text_config", self.model.config)
        max_prompt_length = config.max_position_embeddings - self.max_new_tokens
        if input_ids.shape[1] > max_prompt_length:
            print(f"Warning: prompt of {input_ids.shape[1]} tokens truncated to the last {max_prompt_length}")
            input_ids = input_ids[:, -max_prompt_length:]
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Greedy decoding for absolute stability
                stop_strings=[STOP_MARKER],
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.pad_token_id
            )
        # Keep only the newly generated tokens and cut at the STOP marker to get the clean answer
        answer = self.tokenizer.decode(generation_output[0, input_ids.shape[1]:], skip_special_tokens=True)
        return answer.split(STOP_MARKER)[0].strip()


class LlamaCppBackend:
    def __init__(self, repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                 gguf_file="Meta-Llama-3-8B-Instruct.Q8_0.gguf", n_ctx=4096, max_new_tokens=150,
                 chat_tokenizer=None, chat_template_kwargs=None):
        """
        llama.cpp backend for a quantized GGUF model, fully offloaded to the Metal GPU.
        logits_all=True keeps the logits of every position, which the target-token
        logprobs need (memory: n_ctx x vocab floats, ~2GB for Llama-3 at n_ctx=4096).
        chat_tokenizer: optional Hugging Face model id whose chat template formats the
        answer prompts (e.g. Qwen/Qwen3.5-4B, with thinking disabled by default through
        chat_template_kwargs); otherwise the chat template embedded in the GGUF is used.
        """
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download

        self.n_ctx = n_ctx
        self.max_new_tokens = max_new_tokens
        print(f"Loading GGUF model {repo_id}/{gguf_file} with llama.cpp (Metal)")
        # hf_hub_download uses the local cache when available (also with HF_HUB_OFFLINE=1)
        self.llm = Llama(
            model_path=hf_hub_download(repo_id=repo_id, filename=gguf_file),
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            logits_all=True,
            seed=0,
            verbose=False,
        )
        # Recurrent / hybrid models (e.g. Qwen3.5, with linear-attention layers) keep a running
        # state that cannot be rolled back to an earlier position, so a cached prefix cannot be
        # reused: every sequence is then evaluated from the start
        import llama_cpp
        self.reuse_prefix = not any(
            getattr(llama_cpp, f, lambda m: False)(self.llm.model)
            for f in ("llama_model_is_recurrent", "llama_model_is_hybrid"))
        if not self.reuse_prefix:
            print("Recurrent/hybrid model: prefix reuse disabled")
        self.chat_tokenizer = None
        if chat_tokenizer:
            self.chat_tokenizer = AutoTokenizer.from_pretrained(chat_tokenizer)
            self.chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs

    def compute_batch_logprobs(self, contexts, targets):
        """
        Computes log P(target | context) for each pair and the number of scored target
        tokens. Pairs sharing a prefix (same query and Background Info passage) reuse the
        evaluated prefix from llama.cpp's KV cache, so only the new tokens are evaluated.
        """
        compiled_logprobs, compiled_token_counts = [], []

        for context, target in zip(contexts, targets):
            ctx_ids = self.llm.tokenize(context.encode("utf-8"), add_bos=True)
            tgt_ids = self.llm.tokenize(target.encode("utf-8"), add_bos=False)

            sequence = (ctx_ids + tgt_ids)[-self.n_ctx:]
            t_len = min(len(tgt_ids), len(sequence) - 1)

            # Reuse the longest evaluated prefix (its logits rows are still valid in llm.scores)
            prefix = longest_common_prefix(self.llm.input_ids[:self.llm.n_tokens], sequence) if self.reuse_prefix else 0
            self.llm.n_tokens = min(prefix, len(sequence) - 1)
            self.llm.eval(sequence[self.llm.n_tokens:])

            # Score only the target tokens (the logit at position t predicts token t+1)
            start, end = len(sequence) - t_len, len(sequence)
            logits = self.llm.scores[start-1:end-1, :].astype(np.float64)
            if not np.isfinite(logits).all():
                raise FloatingPointError("Non-finite logits from the LLM.")
            max_logits = logits.max(axis=-1, keepdims=True)
            log_norm = max_logits[:, 0] + np.log(np.exp(logits - max_logits).sum(axis=-1))
            token_logprobs = logits[np.arange(t_len), sequence[start:end]] - log_norm

            compiled_logprobs.append(float(token_logprobs.sum()))
            compiled_token_counts.append(t_len)
        return compiled_logprobs, compiled_token_counts

    def generate_answer(self, instruction, prompt):
        """
        Generates text (RAG answer) with the chat template (greedy decoding:
        temperature 0). Generation stops at the STOP marker the instructions ask for.
        """
        messages = [{"role": "system", "content": instruction}, {"role": "user", "content": prompt}]
        if not self.reuse_prefix:
            # llama.cpp would otherwise reuse the prefix of the previous call
            self.llm.reset()
        if self.chat_tokenizer is not None:
            prompt_text = self.chat_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **self.chat_template_kwargs)
            response = self.llm.create_completion(
                prompt_text,
                max_tokens=self.max_new_tokens,
                temperature=0.0,  # Greedy decoding for absolute stability
                stop=[STOP_MARKER],
            )
            answer = response["choices"][0]["text"] or ""
        else:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=0.0,  # Greedy decoding for absolute stability
                stop=[STOP_MARKER],
            )
            answer = response["choices"][0]["message"]["content"] or ""
        return answer.split(STOP_MARKER)[0].strip()


def longest_common_prefix(a, b):
    """Length of the common token prefix of two token sequences."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

# ==========================================
# 3. FULL INTEGRATED EVALUATION ENGINE
# ==========================================
class UnifiedRAGDeltaEvaluator:
    def __init__(self, llm, gamma=0.6, rescale_bertscore=False,
                 prompt_0shot_path="prompts/prompt-0shot.txt", prompt_kshot_path="prompts/prompt-kshot.txt"):
        """
        Unified Evaluator computing our predictive joint utility delta
        and the empirical ECIR 2025 BERTScore delta. llm is an HFBackend or LlamaCppBackend.
        """
        self.llm = llm
        self.gamma = gamma
        with open(prompt_0shot_path) as f:
            self.instruction_0shot = f.read().strip()
        with open(prompt_kshot_path) as f:
            self.instruction_kshot = f.read().strip()

        # BERTScore model (roberta-large for lang="en") is loaded once and reused.
        # rescale_with_baseline spreads scores over ~[0, 1] instead of bunching them near 0.85
        bert_device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=rescale_bertscore, device=bert_device)

    def _compute_batch_logprobs(self, contexts, targets):
        return self.llm.compute_batch_logprobs(contexts, targets)

    def _generate_answer(self, instruction, prompt):
        return self.llm.generate_answer(instruction, prompt)

    def _calculate_empirical_ecir_p(self, generated_answer, gold_texts):
        """Calculates P(A(q)) of ECIR 2025: max BERTScore F1 against the gold (rel 2/3) passages."""
        if not generated_answer or not gold_texts:
            return 0.0
        candidates = [generated_answer] * len(gold_texts)
        _, _, f1_scores = self.bert_scorer.score(candidates, gold_texts)
        # Extraction of the maximum F1 Score (Equation 1 of ECIR 2025)
        return float(torch.max(f1_scores).item())

    def evaluate_query(self, query_string, retrieved_docs, gold_texts, ks):
        """
        Evaluates one query for every k in ks. The top-k sets are nested, so the pairwise
        logprobs and the zero-shot answer are computed once for max(ks) and reused.
        Returns one record per k.
        """
        max_k = max(ks)
        retrieved_docs = retrieved_docs[:max_k]

        # --- 1. LOGPROBS FOR OUR INTEGRATED TERMS (shared across k) ---
        # Baseline log P(doc_b | query) does not depend on doc_a: computed once per document
        baseline_contexts = [f"Query: {query_string}\n\nDocument: "] * len(retrieved_docs)
        baseline_targets = [doc["text"] for doc in retrieved_docs]
        baseline_lps, _ = self._compute_batch_logprobs(baseline_contexts, baseline_targets)

        # Pairs (a, b) with rank(a) < rank(b); pairs within the top-k are those with b < k
        pair_indices = list(itertools.combinations(range(len(retrieved_docs)), 2))
        joint_contexts, joint_targets = [], []
        for a, b in pair_indices:
            joint_contexts.append(f"Query: {query_string}\n\nBackground Info: {retrieved_docs[a]['text']}\n\nDocument: ")
            joint_targets.append(retrieved_docs[b]["text"])
        joint_lps, joint_token_counts = self._compute_batch_logprobs(joint_contexts, joint_targets)

        # --- 2. ZERO-SHOT (EMPTY CONTEXT) ANSWER (shared across k) ---
        zero_shot_prompt = f"Question: {query_string}\nAnswer:"
        answer_zero_shot = self._generate_answer(self.instruction_0shot, zero_shot_prompt)
        p_zero_shot_raw = self._calculate_empirical_ecir_p(answer_zero_shot, gold_texts)
        # Avoid division by zero if the zero-shot is completely irrelevant
        p_zero_shot = max(0.01, p_zero_shot_raw)

        records = []
        for k in ks:
            top_k_docs = retrieved_docs[:k]
            max_standalone_relevance = max(doc["score"] for doc in top_k_docs)

            total_positive_synergy = 0.0
            total_redundancy_penalty = 0.0
            # Per-token synergy (PMI in nats per target token), so long passages do not dominate
            token_positive_synergy = 0.0
            token_redundancy_penalty = 0.0
            num_pairs = 0
            for (a, b), j_lp, n_tokens in zip(pair_indices, joint_lps, joint_token_counts):
                if b >= k:
                    continue
                synergy_score = j_lp - baseline_lps[b]
                total_positive_synergy += max(0.0, synergy_score)
                total_redundancy_penalty += max(0.0, -synergy_score)
                token_positive_synergy += max(0.0, synergy_score) / n_tokens
                token_redundancy_penalty += max(0.0, -synergy_score) / n_tokens
                num_pairs += 1

            # Absolute theoretical utility: U_context (raw sums, grows with passage length and k^2)
            u_context_absolute = max_standalone_relevance + total_positive_synergy - (self.gamma * total_redundancy_penalty)

            # Normalised utility: MonoT5 log P(true) -> probability in [0, 1], synergy and
            # redundancy averaged per token and per pair, so the scale is independent of k
            relevance_prob = float(np.exp(max_standalone_relevance))
            mean_positive_synergy = token_positive_synergy / max(1, num_pairs)
            mean_redundancy_penalty = token_redundancy_penalty / max(1, num_pairs)
            u_context_norm = relevance_prob + mean_positive_synergy - (self.gamma * mean_redundancy_penalty)

            # --- 3. CALCULATION OF THE EMPIRICAL DELTA (ECIR 2025) ---
            # Context-Shot (With the top-K documents)
            context_str = "\n".join([f"[{i+1}] {d['text']}" for i, d in enumerate(top_k_docs)])
            context_shot_prompt = f"Context:\n{context_str}\n\nQuestion: {query_string}\nAnswer:"
            answer_context_shot = self._generate_answer(self.instruction_kshot, context_shot_prompt)
            p_context_shot = self._calculate_empirical_ecir_p(answer_context_shot, gold_texts)

            # Equation (2) of ECIR 2025: (P_k - P_0) / P_0
            ecir_empirical_delta = (p_context_shot - p_zero_shot) / p_zero_shot
            # Our Predictive Delta (normalized utility, same zero-shot basis)
            our_predictive_delta = (u_context_norm - p_zero_shot) / p_zero_shot

            records.append({
                "k": k,
                "monot5_max_r": max_standalone_relevance,
                "positive_synergy": total_positive_synergy,
                "redundancy_penalty": total_redundancy_penalty,
                "u_context": u_context_absolute,
                "num_pairs": num_pairs,
                "relevance_prob": relevance_prob,
                "mean_positive_synergy": mean_positive_synergy,
                "mean_redundancy_penalty": mean_redundancy_penalty,
                "u_context_norm": u_context_norm,
                "p_zero_shot": p_zero_shot_raw,
                "p_context_shot": p_context_shot,
                "Our_Predictive_Delta_pct": our_predictive_delta * 100,
                "ECIR_Empirical_Delta_pct": ecir_empirical_delta * 100,
                "num_gold_passages": len(gold_texts),
                "answer_zero_shot": answer_zero_shot,
                "answer_context_shot": answer_context_shot,
            })
        return records

# ==========================================
# 4. RUN EVALUATION PIPELINE
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint utility & ECIR 2025 BERTScore deltas over a top-100 run.")
    parser.add_argument("--run", default="retr_results/monot5_dl19_top100.csv", help="A *_top100.csv file in retr_results/")
    parser.add_argument("--qrels", default=None, help="Defaults to qrels/qrels_{dl19|dl20}_rel23.csv, matching --run")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20], help="Top-k cut-offs (subsets of the top-100)")
    parser.add_argument("--backend", choices=["llama-cpp", "hf"], default="llama-cpp",
                        help="llama-cpp: quantized GGUF via llama.cpp (Metal); hf: Hugging Face transformers")
    parser.add_argument("--gguf-repo", default="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", help="[llama-cpp] Hugging Face repo of the GGUF")
    parser.add_argument("--gguf-file", default="Meta-Llama-3-8B-Instruct.Q8_0.gguf", help="[llama-cpp] GGUF file (quantization) in --gguf-repo")
    parser.add_argument("--n-ctx", type=int, default=4096, help="[llama-cpp] Context window in tokens")
    parser.add_argument("--gguf-chat-tokenizer", default=None,
                        help="[llama-cpp] Hugging Face model id whose chat template formats the prompts (e.g. Qwen/Qwen3.5-4B)")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="[hf] Hugging Face model id")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="[hf] Model precision")
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=4, help="[hf] Batch size for the logprob scoring")
    parser.add_argument("--max-queries", type=int, default=None, help="Only evaluate the first N queries (for testing)")
    parser.add_argument("--prompt-0shot", default="prompts/prompt-0shot.txt", help="Instruction for the zero-shot answer")
    parser.add_argument("--prompt-kshot", default="prompts/prompt-kshot.txt", help="Instruction for the top-k context answer")
    parser.add_argument("--rescale-bertscore", action="store_true", help="BERTScore with rescale_with_baseline=True")
    parser.add_argument("--out-dir", default="utility_results")
    args = parser.parse_args()

    ks = sorted(set(args.ks))
    query_set = infer_query_set(args.run)
    qrels_path = args.qrels or os.path.join("qrels", f"qrels_{query_set}_rel23.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    # Each model and BERTScore scale gets its own files so resuming never mixes results
    model_name = os.path.splitext(args.gguf_file)[0] if args.backend == "llama-cpp" else args.model.split("/")[-1]
    suffix = f"_{model_name}" + ("_rescaled" if args.rescale_bertscore else "")
    out_path = os.path.join(args.out_dir, f"utility_{query_set}{suffix}.csv")

    print(f"Run: {args.run} | Qrels: {qrels_path} | k: {ks} | Backend: {args.backend} | Model: {model_name} | "
          f"BERTScore rescaled: {args.rescale_bertscore}")
    run_df = load_run(args.run, max(ks))
    gold_passages = load_gold_passages(qrels_path)

    qids = list(run_df["qid"].unique())
    if args.max_queries:
        qids = qids[:args.max_queries]

    # Resume: skip queries already written to the output file
    done_qids = set()
    if os.path.exists(out_path):
        done_qids = set(pd.read_csv(out_path, dtype={"qid": str})["qid"])
        print(f"Resuming: {len(done_qids)} queries already in {out_path}")

    if args.backend == "llama-cpp":
        llm = LlamaCppBackend(repo_id=args.gguf_repo, gguf_file=args.gguf_file, n_ctx=args.n_ctx, chat_tokenizer=args.gguf_chat_tokenizer)
    else:
        llm = HFBackend(model_id=args.model, batch_size=args.batch_size, dtype=args.dtype)
    evaluator = UnifiedRAGDeltaEvaluator(llm, gamma=args.gamma, rescale_bertscore=args.rescale_bertscore,
                                         prompt_0shot_path=args.prompt_0shot, prompt_kshot_path=args.prompt_kshot)

    for n, qid in enumerate(qids, start=1):
        if qid in done_qids:
            continue
        if qid not in gold_passages:
            print(f"[{n}/{len(qids)}] qid {qid}: no rel 2/3 passages in qrels, skipped")
            continue

        query_partition = run_df[run_df["qid"] == qid]
        query_string = query_partition["query"].iloc[0]
        retrieved_docs = query_partition[["docno", "text", "score"]].to_dict(orient="records")

        records = evaluator.evaluate_query(query_string, retrieved_docs, gold_passages[qid], ks)
        query_df = pd.DataFrame([{"qid": qid, "query": query_string, **r} for r in records])
        query_df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)

        summary = ", ".join(f"k={r['k']}: ECIR {r['ECIR_Empirical_Delta_pct']:.2f}%" for r in records)
        print(f"[{n}/{len(qids)}] qid {qid}: {summary}")

    # One CSV per k, as a subset of the combined results
    results = pd.read_csv(out_path, dtype={"qid": str})
    for k in ks:
        k_path = os.path.join(args.out_dir, f"utility_{query_set}{suffix}_k{k}.csv")
        results[results["k"] == k].to_csv(k_path, index=False)
        print(f"Saved {k_path}")

    # Agreement between predicted utility and the empirical ECIR delta (scale-free, across queries)
    print("\nRank correlation with ECIR_Empirical_Delta_pct across queries:")
    for k in ks:
        k_results = results[results["k"] == k]
        if len(k_results) < 3:
            continue
        for col in ["u_context_norm", "Our_Predictive_Delta_pct", "u_context"]:
            spearman = k_results[col].corr(k_results["ECIR_Empirical_Delta_pct"], method="spearman")
            kendall = k_results[col].corr(k_results["ECIR_Empirical_Delta_pct"], method="kendall")
            print(f"  k={k:<3} {col:<26} Spearman {spearman:+.3f}  Kendall {kendall:+.3f}")

    # Free the llama.cpp model explicitly (its destructor fails at interpreter shutdown)
    if args.backend == "llama-cpp":
        llm.llm.close()
