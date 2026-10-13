import os
import pyterrier as pt
from pyterrier.measures import nDCG
from pyterrier_t5 import MonoT5ReRanker

# Output directory for runs and per-query evaluation
OUT_DIR = "retr_results"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Corpus, BM25 Retriever & MonoT5 ReRanker
msmarco = pt.get_dataset('irds:msmarco-passage')
bm25 = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel="BM25")
monot5 = MonoT5ReRanker(model="castorini/monot5-base-msmarco", batch_size=64)

# 2. Definition of the Pipeline (Top-100 from BM25 -> MonoT5)
# tokenise() strips punctuation that breaks the Terrier query parser; reset()
# restores the original query text for MonoT5; get_text() adds passage text.
pipeline = (
    pt.rewrite.tokenise()
    >> (bm25 % 100)
    >> pt.rewrite.reset()
    >> pt.text.get_text(msmarco, 'text')
    >> monot5
)

# 3. Load the Query Sets
query_sets = {
    "dl19": pt.get_dataset("irds:msmarco-passage/trec-dl-2019/judged"),
    "dl20": pt.get_dataset("irds:msmarco-passage/trec-dl-2020/judged"),
}

for name, dataset in query_sets.items():
    print(f"\nProcessing {name}...")
    topics = dataset.get_topics()
    qrels = dataset.get_qrels()

    # Run the pipeline once and save the reranked top-100
    results = pipeline.transform(topics)
    results[["qid", "query", "docno", "rank", "score"]].to_csv(
        os.path.join(OUT_DIR, f"monot5_{name}_top100.csv"), index=False)
    print(f"Saved {name} results to '{OUT_DIR}/monot5_{name}_top100.csv'")

    # nDCG@10 per query via pt.Experiment (reusing the computed results)
    per_query = pt.Experiment(
        [results],
        topics,
        qrels,
        eval_metrics=[nDCG@10],
        names=["BM25>>MonoT5"],
        perquery=True,
    )
    per_query.to_csv(os.path.join(OUT_DIR, f"monot5_{name}_per_query_ndcg10.csv"), index=False)
    print(f"Saved {name} per-query nDCG@10 to '{OUT_DIR}/monot5_{name}_per_query_ndcg10.csv'")
    print(f"Mean nDCG@10: {per_query['value'].mean():.4f}")
