import os
import argparse
import numpy as np
import pandas as pd

# Top-1 + random contexts: the MonoT5 top-1 passage d_1 at position 1, followed by k-1
# passages sampled at random (in random order) from MonoT5 ranks 2-20 of the same query.
# One random order per query is sampled, so the contexts are nested across k (the first
# k passages form the top-k context), as for the MonoT5 runs. d_1, and hence P_1, is the
# same as in the MonoT5 condition; at k=20 the context is the MonoT5 top-20 reordered.
# MonoT5 scores are kept from the original run, so the relevance term r(d) is defined.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build top-1 + random-companion context runs.")
    parser.add_argument("--pool-depth", type=int, default=20, help="Companions are sampled from MonoT5 ranks 2..pool_depth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system", default="top1-rand", help="Name of the new context run")
    args = parser.parse_args()

    for name in ["dl19", "dl20"]:
        run = pd.read_csv(os.path.join("retr_results", f"monot5_{name}_top100.csv"), dtype={"qid": str, "docno": str})
        run = run.sort_values(["qid", "rank"]).groupby("qid").head(args.pool_depth)

        rng = np.random.default_rng(args.seed)
        contexts = []
        for qid, query_run in run.groupby("qid", sort=True):
            top1, pool = query_run.iloc[:1], query_run.iloc[1:]
            companions = pool.iloc[rng.permutation(len(pool))]
            context = pd.concat([top1, companions]).assign(rank=range(len(query_run)))
            contexts.append(context)

        out = pd.concat(contexts)[["qid", "query", "docno", "rank", "score"]]
        out_path = os.path.join("retr_results", f"{args.system}_{name}_top{args.pool_depth}.csv")
        out.to_csv(out_path, index=False)
        print(f"Saved {out_path}: {out['qid'].nunique()} queries")
