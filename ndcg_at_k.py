import os
import argparse
import ir_measures
import pandas as pd
import pyterrier as pt
from ir_measures import nDCG

# Per-query nDCG@k of a context run for every k, as in Tian et al. (ECIR 2025, Eq. 3).
# Contexts are evaluated in their given order (rank), which for the oracle contexts is
# the sampled order, not the MonoT5 score order.
QUERY_SETS = {"dl19": "irds:msmarco-passage/trec-dl-2019/judged",
              "dl20": "irds:msmarco-passage/trec-dl-2020/judged"}


def ndcg_at_k(run_df, qrels_df, ks):
    """Returns a DataFrame with columns qid, k, ndcg (one row per query and k)."""
    run = pd.DataFrame({"query_id": run_df["qid"].astype(str), "doc_id": run_df["docno"].astype(str),
                        "score": -run_df["rank"].astype(float)})
    qrels = pd.DataFrame({"query_id": qrels_df["qid"].astype(str), "doc_id": qrels_df["docno"].astype(str),
                          "relevance": qrels_df["label"].astype(int)})
    measures = {nDCG @ k: k for k in ks}
    rows = [{"qid": m.query_id, "k": measures[m.measure], "ndcg": m.value}
            for m in ir_measures.iter_calc(list(measures), qrels, run)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-query nDCG@k of context runs.")
    parser.add_argument("--systems", nargs="+", default=["monot5"], help="Run files retr_results/{system}_{set}_*.csv")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20])
    args = parser.parse_args()

    for name, dataset_id in QUERY_SETS.items():
        qrels = pt.get_dataset(dataset_id).get_qrels()
        for system in args.systems:
            run_path = next(os.path.join("retr_results", f) for f in sorted(os.listdir("retr_results"))
                            if f.startswith(f"{system}_{name}_top") and f.endswith(".csv"))
            run_df = pd.read_csv(run_path, dtype={"qid": str, "docno": str})
            out = ndcg_at_k(run_df, qrels, args.ks)
            out_path = os.path.join("retr_results", f"{system}_{name}_per_query_ndcg_at_k.csv")
            out.to_csv(out_path, index=False)
            print(f"Saved {out_path}: " + ", ".join(
                f"nDCG@{k}={v:.4f}" for k, v in out.groupby("k")["ndcg"].mean().items()))
