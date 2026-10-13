import os
import argparse
import itertools
import pandas as pd
from scipy.stats import spearmanr

from calculate_utility import LlamaCppBackend, HFBackend, load_run, find_run, load_ndcg_at_k

# Two fixes for the interaction term of our utility, reported against the top-1 baseline
# (U(C_1) has no pairs, so each delta below is the change from the top-1 passage):
#
# Fix 1 - positive PMI is redundancy (flipped sign). s(a, b) = [log P(b|q,a) - log P(b|q)] / |b|
#   is positive when a makes b more predictable (b adds little new information):
#       delta_flip(k) = S^-_k - gamma * S^+_k
#   S^+_k / S^-_k are the mean positive / negative PMI over the top-k pairs, already saved by
#   calculate_utility.py (mean_positive_synergy / mean_redundancy_penalty): no LLM needed.
#
# Fix 2 - complementarity relative to the query. Using the query likelihood given passages
#   (as in unsupervised passage reranking), a pair is complementary when both passages
#   together explain the query better than either passage alone:
#       c(a, b) = [log P(q | a, b) - max(log P(q | a), log P(q | b))] / |q|
#       delta_comp(k) = C^+_k - gamma * C^-_k
#   C^+_k / C^-_k are the mean positive / negative c(a, b) over the top-k pairs.
#
# Combined - complementarity minus the PMI redundancy of Fix 1:
#       delta_comb(k) = delta_comp(k) - gamma * S^+_k

INSTRUCTION = "Please write a question based on the passage(s) above."

# ==========================================
# 1. FIX 2: QUERY-LIKELIHOOD COMPLEMENTARITY (LLM)
# ==========================================
def complementarity_for_query(llm, query, docs, ks, gamma):
    """Returns one record per k with C^+_k, C^-_k and delta_comp(k)."""
    # Pairs share the "Passage: a" prefix, so llama.cpp reuses it across consecutive pairs
    pair_indices = list(itertools.combinations(range(len(docs)), 2))
    contexts = [f"Passage: {d}\n\n{INSTRUCTION}\nQuestion: " for d in docs]
    contexts += [f"Passage: {docs[a]}\n\nPassage: {docs[b]}\n\n{INSTRUCTION}\nQuestion: " for a, b in pair_indices]
    logprobs, token_counts = llm.compute_batch_logprobs(contexts, [query] * len(contexts))
    per_token = [lp / n for lp, n in zip(logprobs, token_counts)]
    single_lps, pair_lps = per_token[:len(docs)], per_token[len(docs):]

    records = []
    for k in ks:
        comps = [pair_lp - max(single_lps[a], single_lps[b])
                 for (a, b), pair_lp in zip(pair_indices, pair_lps) if b < k]
        comp_pos = sum(max(0.0, c) for c in comps) / max(1, len(comps))
        comp_neg = sum(max(0.0, -c) for c in comps) / max(1, len(comps))
        records.append({
            "k": k,
            "mean_complementarity": comp_pos,
            "mean_anti_complementarity": comp_neg,
            "delta_comp": comp_pos - gamma * comp_neg,
        })
    return records


def compute_complementarity(llm, system, name, ks, gamma, out_path, max_queries=None):
    """Runs Fix 2 for every query of a query set; resumes from out_path."""
    run_df = load_run(find_run(system, name), max(ks))
    done_qids = set()
    if os.path.exists(out_path):
        done_qids = set(pd.read_csv(out_path, dtype={"qid": str})["qid"])

    qids = list(run_df["qid"].unique())[:max_queries]
    for n, qid in enumerate(qids, start=1):
        if qid in done_qids:
            continue
        if llm is None:
            raise RuntimeError(f"Complementarity missing for qid {qid} in {out_path}; run without --tables-only first")
        query_partition = run_df[run_df["qid"] == qid]
        records = complementarity_for_query(
            llm, query_partition["query"].iloc[0], list(query_partition["text"]), ks, gamma)
        pd.DataFrame([{"qid": qid, **r} for r in records]).to_csv(
            out_path, mode="a", header=not os.path.exists(out_path), index=False)
        summary = ", ".join(f"k={r['k']}: {r['delta_comp']:+.4f}" for r in records)
        print(f"[{n}/{len(qids)}] qid {qid}: delta_comp {summary}")

    return pd.read_csv(out_path, dtype={"qid": str})

# ==========================================
# 2. FIX 1, COMBINED DELTA & CORRELATIONS
# ==========================================
def load_query_set(system, name, model_name, complementarity, gamma, results_dir):
    """Per-query top-1 baseline results with nDCG@k, Fix 1, Fix 2 and the combined delta."""
    df = pd.read_csv(os.path.join(results_dir, f"top1_baseline_{name}_{model_name}.csv"), dtype={"qid": str})
    df = df.merge(load_ndcg_at_k(system, name), on=["qid", "k"], how="inner")
    df = df.merge(complementarity, on=["qid", "k"], how="inner")

    df["ecir_delta"] = df["ECIR_Empirical_Delta_pct"] / 100
    df["delta_flip"] = df["mean_redundancy_penalty"] - gamma * df["mean_positive_synergy"]
    df["delta_comb"] = df["delta_comp"] - gamma * df["mean_positive_synergy"]
    return df


DELTAS = ["delta_pred_top1", "delta_flip", "delta_comp", "delta_comb"]
TARGETS = ["delta_emp_top1", "ecir_delta", "ndcg_at_k"]


def correlation_rows(df, name):
    """Spearman correlation of every delta with the empirical gains and nDCG@k, per k."""
    rows = []
    for k, k_df in df.groupby("k"):
        for delta in DELTAS:
            row = {"query_set": name, "k": k, "delta": delta, "mean": k_df[delta].mean()}
            for target in TARGETS:
                rho, p_value = spearmanr(k_df[delta], k_df[target])
                row[f"spearman_vs_{target}"] = rho
                row[f"p_value_vs_{target}"] = p_value
            row["n"] = len(k_df)
            rows.append(row)
    return rows

# ==========================================
# 3. RUN
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix 1 (flipped PMI), Fix 2 (complementarity) and both combined.")
    parser.add_argument("--backend", choices=["llama-cpp", "hf"], default="llama-cpp")
    parser.add_argument("--gguf-repo", default="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF")
    parser.add_argument("--gguf-file", default="Meta-Llama-3-8B-Instruct.Q8_0.gguf")
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--gguf-chat-tokenizer", default=None,
                        help="[llama-cpp] Hugging Face model id whose chat template formats the prompts (e.g. Qwen/Qwen3.5-4B)")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="[hf] Model precision")
    parser.add_argument("--tables-only", action="store_true",
                        help="Do not load the LLM: rebuild the outputs from the saved complementarity scores")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--query-sets", nargs="+", default=["dl19", "dl20"])
    parser.add_argument("--max-queries", type=int, default=None, help="Only score the first N queries (for testing)")
    parser.add_argument("--system", default="monot5", help="Context run retr_results/{system}_{set}_top*.csv")
    parser.add_argument("--results-dir", default="utility_results")
    parser.add_argument("--tables-dir", default="tables")
    args = parser.parse_args()

    ks = sorted(set(args.ks))
    model_name = os.path.splitext(args.gguf_file)[0] if args.backend == "llama-cpp" else args.model.split("/")[-1]
    os.makedirs(args.tables_dir, exist_ok=True)

    llm = None
    if args.tables_only:
        pass
    elif args.backend == "llama-cpp":
        llm = LlamaCppBackend(repo_id=args.gguf_repo, gguf_file=args.gguf_file, n_ctx=args.n_ctx, chat_tokenizer=args.gguf_chat_tokenizer)
    else:
        llm = HFBackend(model_id=args.model, dtype=args.dtype)

    results = {}
    for name in args.query_sets:
        print(f"\nProcessing {name}...")
        complementarity = compute_complementarity(
            llm, args.system, name, ks, args.gamma,
            out_path=os.path.join(args.results_dir, f"complementarity_{name}_{model_name}.csv"),
            max_queries=args.max_queries)
        results[name] = load_query_set(args.system, name, model_name, complementarity, args.gamma, args.results_dir)

        per_query_path = os.path.join(args.results_dir, f"utility_fixes_{name}_{model_name}.csv")
        results[name].to_csv(per_query_path, index=False)
        print(f"Saved {per_query_path}")

    rows = []
    for name, df in list(results.items()) + ([("both", pd.concat(results.values()))] if len(results) > 1 else []):
        rows += correlation_rows(df, name)

    # p-values are kept at full precision (rounding would show the very small ones as 0)
    table = pd.DataFrame(rows)
    other_cols = [c for c in table.columns if not c.startswith("p_value") and c not in ("query_set", "delta")]
    table[other_cols] = table[other_cols].round(4)
    table_path = os.path.join(args.tables_dir, "utility_fixes.csv")
    table.to_csv(table_path, index=False)
    print(f"Saved {table_path}")
    print(table.to_string(index=False))

    # Free the llama.cpp model explicitly (its destructor fails at interpreter shutdown)
    if args.backend == "llama-cpp" and llm is not None:
        llm.llm.close()
