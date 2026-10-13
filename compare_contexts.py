import os
import argparse
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

# Controlled comparison of two context runs that share the top-1 passage d_1 (and hence
# P_0 and P_1) but differ in the companion passages 2..k, e.g. MonoT5 (ranks 2..k) vs
# top1-rand (random passages from MonoT5 ranks 2..20). Per k, query set and pooled:
#   - means of P_k, delta_emp = P_k - P_1 and the four predictive deltas, with paired
#     t-tests (reference vs alternative) across queries
#   - within-query: Spearman correlation across queries of the alternative-minus-reference
#     difference of each predictive delta with the difference of delta_emp
#   - pooled over both runs: Spearman correlation of each delta with delta_emp and nDCG@k
MODEL_NAME = "Meta-Llama-3-8B-Instruct.Q8_0"
DELTAS = ["delta_pred_top1", "delta_flip", "delta_comp", "delta_comb"]
COLUMNS = ["qid", "k", "p_context_shot", "delta_emp_top1", "ndcg_at_k"] + DELTAS


def load(results_dir, name):
    path = os.path.join(results_dir, f"utility_fixes_{name}_{MODEL_NAME}.csv")
    return pd.read_csv(path, dtype={"qid": str})[COLUMNS]


def summarise(ref, alt, name):
    pairs = ref.merge(alt, on=["qid", "k"], suffixes=("_ref", "_alt"))
    pooled = pd.concat([ref, alt])
    rows = []
    for k, g in pairs.groupby("k"):
        g_pooled = pooled[pooled["k"] == k]
        emp_diff = g["delta_emp_top1_alt"] - g["delta_emp_top1_ref"]
        for col in ["p_context_shot", "delta_emp_top1"] + DELTAS:
            row = {
                "query_set": name, "k": k, "measure": col,
                "mean_ref": g[f"{col}_ref"].mean(),
                "mean_alt": g[f"{col}_alt"].mean(),
                "ttest_p": ttest_rel(g[f"{col}_alt"], g[f"{col}_ref"]).pvalue,
            }
            if col in DELTAS:
                rho, p = spearmanr(g[f"{col}_alt"] - g[f"{col}_ref"], emp_diff)
                row.update({"within_query_spearman_vs_delta_emp": rho, "within_query_p_value": p})
                rho, p = spearmanr(g_pooled[col], g_pooled["delta_emp_top1"])
                row.update({"pooled_spearman_vs_delta_emp": rho, "pooled_p_value_vs_delta_emp": p})
                rho, p = spearmanr(g_pooled[col], g_pooled["ndcg_at_k"])
                row.update({"pooled_spearman_vs_ndcg_at_k": rho, "pooled_p_value_vs_ndcg_at_k": p})
            row["n_queries"] = len(g)
            rows.append(row)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two context runs with the same top-1 passage.")
    parser.add_argument("--reference-dir", default="utility_results", help="Results of the reference run (MonoT5)")
    parser.add_argument("--alternative-dir", default="utility_results/top1-rand", help="Results of the alternative run")
    parser.add_argument("--out", default="tables/top1-rand/compare_vs_monot5.csv")
    args = parser.parse_args()

    refs = {name: load(args.reference_dir, name) for name in ["dl19", "dl20"]}
    alts = {name: load(args.alternative_dir, name) for name in ["dl19", "dl20"]}
    rows = []
    for name in ["dl19", "dl20"]:
        rows += summarise(refs[name], alts[name], name)
    rows += summarise(pd.concat(refs.values()), pd.concat(alts.values()), "both")

    # p-values are kept at full precision (rounding would show the very small ones as 0)
    table = pd.DataFrame(rows)
    other_cols = [c for c in table.columns
                  if not (c.endswith("_p") or "p_value" in c) and c not in ("query_set", "measure")]
    table[other_cols] = table[other_cols].round(4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"Saved {args.out}")
