import os
import argparse
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

# Summary tables of the utility runs, with percentages written as decimal fractions
parser = argparse.ArgumentParser(description="Summary tables of the utility runs.")
parser.add_argument("--results-dir", default="utility_results")
parser.add_argument("--tables-dir", default="tables")
parser.add_argument("--model-name", default="Meta-Llama-3-8B-Instruct.Q8_0", help="Model name in the result file names")
args = parser.parse_args()
RESULTS_DIR, OUT_DIR, MODEL_NAME = args.results_dir, args.tables_dir, args.model_name
DECIMALS = 4
os.makedirs(OUT_DIR, exist_ok=True)

results = {
    name: pd.read_csv(os.path.join(RESULTS_DIR, f"utility_{name}_{MODEL_NAME}.csv"), dtype={"qid": str})
    for name in ["dl19", "dl20"]
}

# 1. Means per k across all queries of each query set
for name, df in results.items():
    means = df.groupby("k").agg(
        ecir_delta=("ECIR_Empirical_Delta_pct", lambda s: s.mean() / 100),
        predictive_delta=("Our_Predictive_Delta_pct", lambda s: s.mean() / 100),
        u_context_norm=("u_context_norm", "mean"),
        p_zero_shot=("p_zero_shot", "mean"),
        p_context_shot=("p_context_shot", "mean"),
        share_context_helped=("ECIR_Empirical_Delta_pct", lambda s: (s > 0).mean()),
        num_queries=("qid", "nunique"),
    ).reset_index()
    path = os.path.join(OUT_DIR, f"means_{name}.csv")
    means.round(DECIMALS).to_csv(path, index=False)
    print(f"Saved {path}")

# 2. Spearman correlation of u_context_norm with the ECIR delta across queries, plus a paired
#    t-test across queries of P_k vs P_0 (does the top-k context improve the answer?)
combined = pd.concat(results.values())
rows = []
for k in sorted(combined["k"].unique()):
    row = {"k": k}
    for name, df in list(results.items()) + [("both", combined)]:
        k_df = df[df["k"] == k]
        rho, p_value = spearmanr(k_df["u_context_norm"], k_df["ECIR_Empirical_Delta_pct"])
        row[f"spearman_{name}"] = rho
        row[f"p_value_{name}"] = p_value
        row[f"ttest_pk_vs_p0_p_{name}"] = ttest_rel(k_df["p_context_shot"], k_df["p_zero_shot"]).pvalue
        row[f"n_{name}"] = len(k_df)
    rows.append(row)
path = os.path.join(OUT_DIR, "spearman_u_context_norm_vs_ecir.csv")
# p-values are kept at full precision (rounding would show the very small ones as 0)
tests = pd.DataFrame(rows)
other_cols = [c for c in tests.columns if "_p_" not in c and not c.startswith("p_value")]
tests[other_cols] = tests[other_cols].round(DECIMALS)
tests.to_csv(path, index=False)
print(f"Saved {path}")
