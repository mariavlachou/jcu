import os
import pyterrier as pt

# Output directory for the selected qrels
OUT_DIR = "qrels"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Corpus (for passage text)
msmarco = pt.get_dataset('irds:msmarco-passage')
get_text = pt.text.get_text(msmarco, 'text')

# 2. Load the Query Sets
query_sets = {
    "dl19": pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged"),
    "dl20": pt.get_dataset("irds:msmarco-passage/trec-dl-2020/judged"),
}

for name, dataset in query_sets.items():
    print(f"\nProcessing {name}...")
    qrels = dataset.get_qrels()
    topics = dataset.get_topics()

    # Keep only the passages judged highly or perfectly relevant (rel=2 or rel=3)
    rel23 = qrels[qrels["label"].isin([2, 3])]

    # Add query and passage text
    rel23 = rel23.merge(topics[["qid", "query"]], on="qid", how="left")
    rel23 = get_text(rel23)
    rel23 = rel23[["qid", "query", "docno", "label", "text"]].sort_values(["qid", "docno"])

    rel23.to_csv(os.path.join(OUT_DIR, f"qrels_{name}_rel23.csv"), index=False)
    print(f"Saved {len(rel23)} rel=2/3 passages for {rel23['qid'].nunique()} of "
          f"{qrels['qid'].nunique()} queries to '{OUT_DIR}/qrels_{name}_rel23.csv'")
