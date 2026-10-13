import os
import argparse
import numpy as np
import pandas as pd
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker

# Oracle contexts of Tian et al. (ECIR 2025), in the same format as the MonoT5 runs:
#   oracle-rel:  up to max_k passages judged relevant (labels 2 and 3), sampled in random
#                order; if a query has fewer relevant passages, all of them are used
#   oracle-nrel: max_k passages judged non-relevant (label 0), sampled in random order
# The context order is the sampled order (rank); MonoT5 scores are added so that the
# relevance term r(d) of our utility is defined.
QUERY_SETS = {"dl19": "irds:msmarco-passage/trec-dl-2019/judged",
              "dl20": "irds:msmarco-passage/trec-dl-2020/judged"}
ORACLES = {"oracle-rel": lambda labels: labels >= 2, "oracle-nrel": lambda labels: labels == 0}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Rel / NRel oracle context runs.")
    parser.add_argument("--max-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    get_text = pt.text.get_text(pt.get_dataset("irds:msmarco-passage"), "text")
    monot5 = MonoT5ReRanker(model="castorini/monot5-base-msmarco", batch_size=64)

    for name, dataset_id in QUERY_SETS.items():
        dataset = pt.get_dataset(dataset_id)
        topics = dataset.get_topics()[["qid", "query"]]
        qrels = dataset.get_qrels()

        for oracle, is_member in ORACLES.items():
            rng = np.random.default_rng(args.seed)
            rows = []
            for qid in sorted(topics["qid"]):
                candidates = qrels[(qrels["qid"] == qid) & is_member(qrels["label"])]["docno"].tolist()
                sampled = [candidates[i] for i in rng.permutation(len(candidates))[:args.max_k]]
                rows += [{"qid": qid, "docno": docno, "rank": rank} for rank, docno in enumerate(sampled)]

            run = pd.DataFrame(rows).merge(topics, on="qid")
            run = monot5(get_text(run)).drop(columns=["text"])
            # MonoT5 re-ranking overwrites the rank: restore the sampled context order
            run = run.drop(columns=["rank"]).merge(pd.DataFrame(rows), on=["qid", "docno"])
            run = run[["qid", "query", "docno", "rank", "score"]].sort_values(["qid", "rank"])

            out_path = os.path.join("retr_results", f"{oracle}_{name}_top{args.max_k}.csv")
            run.to_csv(out_path, index=False)
            print(f"Saved {out_path}: {run['qid'].nunique()} queries, "
                  f"{run.groupby('qid').size().min()}-{run.groupby('qid').size().max()} passages per query")
