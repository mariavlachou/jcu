import os
import argparse
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

from calculate_utility import LlamaCppBackend, HFBackend, UnifiedRAGDeltaEvaluator, load_run, load_gold_passages, find_run

# Compares our utility against the top-1 passage as baseline (k = 1):
#   Predictive: delta_pred(k) = U(C_k) - U(C_1) = mean_positive_synergy - gamma * mean_redundancy_penalty
#               (U(C_1) = r(d_1): one passage has no pairs, and d_1 has the highest MonoT5 score)
#   Empirical:  delta_emp(k)  = P_k - P_1, where P_1 is the BERTScore of the answer given only d_1

# ==========================================
# 1. P_1: ANSWER FROM THE TOP-1 PASSAGE ONLY
# ==========================================
def compute_p1(evaluator, run_path, qrels_path, out_path):
    """Generates the top-1 answer per query and its P_1 score; resumes from out_path."""
    run_df = load_run(run_path, 1)
    gold_passages = load_gold_passages(qrels_path)

    done_qids = set()
    if os.path.exists(out_path):
        done_qids = set(pd.read_csv(out_path, dtype={"qid": str})["qid"])

    for n, row in enumerate(run_df.itertuples(), start=1):
        if row.qid in done_qids or row.qid not in gold_passages:
            continue
        # Same prompt as the top-k answers in calculate_utility.py, with k = 1
        if evaluator is None:
            raise RuntimeError(f"P_1 missing for qid {row.qid} in {out_path}; run without --tables-only first")
        prompt = f"Context:\n[1] {row.text}\n\nQuestion: {row.query}\nAnswer:"
        answer = evaluator._generate_answer(evaluator.instruction_kshot, prompt)
        p_top1 = evaluator._calculate_empirical_ecir_p(answer, gold_passages[row.qid])

        pd.DataFrame([{"qid": row.qid, "p_top1_shot": p_top1, "answer_top1_shot": answer}]).to_csv(
            out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(f"[{n}/{len(run_df)}] qid {row.qid}: P_1 = {p_top1:.4f}")

    return pd.read_csv(out_path, dtype={"qid": str})

# ==========================================
# 2. DELTAS AGAINST THE TOP-1 BASELINE
# ==========================================
def add_top1_deltas(results, p1):
    results = results.merge(p1[["qid", "p_top1_shot"]], on="qid", how="inner")
    # U(C_k) - U(C_1): the relevance term r(d_1) cancels, leaving the normalised synergy terms
    results["delta_pred_top1"] = results["u_context_norm"] - results["relevance_prob"]
    results["delta_emp_top1"] = results["p_context_shot"] - results["p_top1_shot"]
    return results


def summarise(results, name):
    """
    Means per k, Spearman correlations of both top-1 deltas with the ECIR delta
    ((P_k - P_0) / P_0) and the paired t-test (P_k vs P_1), all across queries.
    """
    rows = []
    for k, k_df in results.groupby("k"):
        ecir_delta = k_df["ECIR_Empirical_Delta_pct"] / 100
        rho_pred, rho_pred_p = spearmanr(k_df["delta_pred_top1"], ecir_delta)
        rho_emp, rho_emp_p = spearmanr(k_df["delta_emp_top1"], ecir_delta)
        rows.append({
            "query_set": name,
            "k": k,
            "delta_pred_top1": k_df["delta_pred_top1"].mean(),
            "delta_emp_top1": k_df["delta_emp_top1"].mean(),
            "p_top1_shot": k_df["p_top1_shot"].mean(),
            "p_context_shot": k_df["p_context_shot"].mean(),
            "ecir_delta": ecir_delta.mean(),
            "share_topk_beats_top1": (k_df["delta_emp_top1"] > 0).mean(),
            "spearman_pred_vs_ecir": rho_pred,
            "p_value_pred_vs_ecir": rho_pred_p,
            "spearman_emp_vs_ecir": rho_emp,
            "p_value_emp_vs_ecir": rho_emp_p,
            "ttest_pk_vs_p1_p": ttest_rel(k_df["p_context_shot"], k_df["p_top1_shot"]).pvalue,
            "n": len(k_df),
        })
    return rows

# ==========================================
# 3. RUN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Our utility vs the top-1 passage baseline (P_1).")
    parser.add_argument("--backend", choices=["llama-cpp", "hf"], default="llama-cpp")
    parser.add_argument("--gguf-repo", default="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF")
    parser.add_argument("--gguf-file", default="Meta-Llama-3-8B-Instruct.Q8_0.gguf")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--gguf-chat-tokenizer", default=None,
                        help="[llama-cpp] Hugging Face model id whose chat template formats the prompts (e.g. Qwen/Qwen3.5-4B)")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="[hf] Model precision")
    parser.add_argument("--tables-only", action="store_true",
                        help="Do not load the LLM: rebuild the outputs from the saved P_1 answers")
    parser.add_argument("--prompt-kshot", default="prompts/prompt-kshot.txt")
    parser.add_argument("--system", default="monot5", help="Context run retr_results/{system}_{set}_top*.csv")
    parser.add_argument("--results-dir", default="utility_results")
    parser.add_argument("--tables-dir", default="tables")
    args = parser.parse_args()

    model_name = os.path.splitext(args.gguf_file)[0] if args.backend == "llama-cpp" else args.model.split("/")[-1]
    os.makedirs(args.tables_dir, exist_ok=True)

    llm, evaluator = None, None
    if args.tables_only:
        pass
    elif args.backend == "llama-cpp":
        llm = LlamaCppBackend(repo_id=args.gguf_repo, gguf_file=args.gguf_file, n_ctx=args.n_ctx, chat_tokenizer=args.gguf_chat_tokenizer)
    else:
        llm = HFBackend(model_id=args.model, dtype=args.dtype)
    if llm is not None:
        evaluator = UnifiedRAGDeltaEvaluator(llm, prompt_kshot_path=args.prompt_kshot)

    all_results, table_rows = [], []
    for name in ["dl19", "dl20"]:
        print(f"\nProcessing {name}...")
        p1 = compute_p1(
            evaluator,
            run_path=find_run(args.system, name),
            qrels_path=os.path.join("qrels", f"qrels_{name}_rel23.csv"),
            out_path=os.path.join(args.results_dir, f"p1_{name}_{model_name}.csv"),
        )
        results = pd.read_csv(os.path.join(args.results_dir, f"utility_{name}_{model_name}.csv"), dtype={"qid": str})
        results = add_top1_deltas(results, p1)

        per_query_path = os.path.join(args.results_dir, f"top1_baseline_{name}_{model_name}.csv")
        results.drop(columns=["answer_zero_shot", "answer_context_shot"]).to_csv(per_query_path, index=False)
        print(f"Saved {per_query_path}")

        all_results.append(results)
        table_rows += summarise(results, name)
    table_rows += summarise(pd.concat(all_results), "both")

    # p-values are kept at full precision (rounding would show the very small ones as 0)
    table = pd.DataFrame(table_rows)
    other_cols = [c for c in table.columns if not c.startswith("p_value") and c not in ("ttest_pk_vs_p1_p", "query_set")]
    table[other_cols] = table[other_cols].round(4)
    table_path = os.path.join(args.tables_dir, "top1_baseline.csv")
    table.to_csv(table_path, index=False)
    print(f"Saved {table_path}")
    print(table.to_string(index=False))

    # Free the llama.cpp model explicitly (its destructor fails at interpreter shutdown)
    if args.backend == "llama-cpp" and llm is not None:
        llm.llm.close()
