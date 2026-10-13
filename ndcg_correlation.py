import os
import argparse
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from calculate_utility import load_ndcg_at_k

# Spearman and Pearson correlations across queries of per-query retrieval quality (nDCG@k of
# the context, as in Tian et al., ECIR 2025, Eq. 3) with our predictive delta (top-1
# baseline) and with the ECIR delta ((P_k - P_0) / P_0)
DECIMALS = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlations of the deltas with nDCG@k.")
    parser.add_argument("--system", default="monot5", help="Context run retr_results/{system}_{set}_top*.csv")
    parser.add_argument("--results-dir", default="utility_results")
    parser.add_argument("--tables-dir", default="tables")
    parser.add_argument("--model-name", default="Meta-Llama-3-8B-Instruct.Q8_0", help="Model name in the result file names")
    args = parser.parse_args()
    os.makedirs(args.tables_dir, exist_ok=True)

    results = {}
    for name in ["dl19", "dl20"]:
        # Per-query deltas from compare_p1.py (includes the ECIR delta of calculate_utility.py)
        deltas = pd.read_csv(os.path.join(args.results_dir, f"top1_baseline_{name}_{args.model_name}.csv"), dtype={"qid": str})
        results[name] = deltas.merge(load_ndcg_at_k(args.system, name), on=["qid", "k"], how="inner")

    for method, corr in [("spearman", spearmanr), ("pearson", pearsonr)]:
        rows = []
        for name, df in list(results.items()) + [("both", pd.concat(results.values()))]:
            for k, k_df in df.groupby("k"):
                ecir_delta = k_df["ECIR_Empirical_Delta_pct"] / 100
                rho_pred, rho_pred_p = corr(k_df["delta_pred_top1"], k_df["ndcg_at_k"])
                rho_ecir, rho_ecir_p = corr(ecir_delta, k_df["ndcg_at_k"])
                rows.append({
                    "query_set": name,
                    "k": k,
                    "ndcg_at_k": k_df["ndcg_at_k"].mean(),
                    "delta_pred_top1": k_df["delta_pred_top1"].mean(),
                    "ecir_delta": ecir_delta.mean(),
                    f"{method}_pred_vs_ndcg_at_k": rho_pred,
                    "p_value_pred_vs_ndcg_at_k": rho_pred_p,
                    f"{method}_ecir_vs_ndcg_at_k": rho_ecir,
                    "p_value_ecir_vs_ndcg_at_k": rho_ecir_p,
                    "n": len(k_df),
                })

        # p-values are kept at full precision (rounding would show the very small ones as 0)
        table = pd.DataFrame(rows)
        other_cols = [c for c in table.columns if not c.startswith("p_value") and c != "query_set"]
        table[other_cols] = table[other_cols].round(DECIMALS)
        # The Spearman table has no method suffix
        suffix = "" if method == "spearman" else f"_{method}"
        path = os.path.join(args.tables_dir, f"ndcg_at_k_correlations{suffix}.csv")
        table.to_csv(path, index=False)
        print(f"Saved {path}")
