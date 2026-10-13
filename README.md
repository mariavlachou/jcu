# Joint Context Utility for RAG

Here we share our code for implementing our approach of Joint Context Utility in RAG.

Our implementation uses a retrieval pipeline with BM25 and the top-100 re-ranked passages by MonoT5 using PyTerrier as context. To produce our generated answers, we use Llama-3-8b (in particular the Q8_0 GGUF `QuantFactory/Meta-Llama-3-8B-Instruct-GGUF` run with llama.cpp) and Qwen3.5-4B. 

## Setup

### Getting Started

```sh
pip install -r requirements.txt
```


### Qwen3.5 environment (only for `run_model.sh`)

Qwen3.5 needs newer `transformers` and `llama-cpp-python` versions than the main environment. They are installed in a separate virtual environment that reuses all other packages of the main environment:

```sh
python -m venv --system-site-packages .venv-qwen
CMAKE_ARGS="-DGGML_METAL=on" .venv-qwen/bin/pip install -r requirements-qwen.txt
```

Download the models with the **main** environment (the newer download client in
`.venv-qwen` can fail with older compression libraries):

```sh
python -c "from huggingface_hub import hf_hub_download, snapshot_download; \
hf_hub_download('unsloth/Qwen3.5-4B-GGUF', 'Qwen3.5-4B-Q8_0.gguf'); \
snapshot_download('Qwen/Qwen3.5-4B', allow_patterns=['*.json', '*.jinja', '*.txt'])"
```

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Retrieval: BM25 >> MonoT5 top-100, per-query nDCG@10 | `monot5_pipeline.py` | `retr_results/` |
| 2. Gold passages (relevance 2 or 3) with text | `qrels_rel23.py` | `qrels/` |
| 3. Per-query nDCG@k (k = 5, 10, 15, 20) | `ndcg_at_k.py` | `retr_results/*_per_query_ndcg_at_k.csv` |
| 4. Answers (zero-shot and top-k), BERTScore, PMI utility | `calculate_utility.py` | `utility_results/utility_*.csv` |
| 5. Top-1 answers (P_1) and top-1 baseline deltas | `compare_p1.py` | `utility_results/p1_*`, `top1_baseline_*`, `tables/top1_baseline.csv` |
| 6. Fix 1 (flipped PMI), Fix 2 (complementarity), combined | `utility_fixes.py` | `utility_results/complementarity_*`, `utility_fixes_*`, `tables/utility_fixes.csv` |
| 7. Correlations with nDCG@k (Spearman, Pearson) | `ndcg_correlation.py` | `tables/ndcg_at_k_correlations*.csv` |
| 8. Zero-shot summary tables | `make_tables.py` | `tables/means_*.csv`, `tables/spearman_*.csv` |

Steps 4–6 load the LLM and save their results after every query, so an interrupted run resumes where it stopped. 

### Other context conditions

- **Top-1 + random companions** (controlled experiment): the MonoT5 top passage followed by
  passages sampled at random from MonoT5 ranks 2–20.
  `python build_random_runs.py`, then `./run_contexts.sh top1-rand`, then
  `python compare_contexts.py` (comparison with MonoT5 in
  `tables/top1-rand/compare_vs_monot5.csv`).


### Another LLM: Qwen3.5-4B

```sh
./run_model.sh                     # Qwen3.5-4B Q8_0 by default
```

`run_model.sh` runs steps 4–8 on the MonoT5 contexts with Qwen3.5-4B and writes to
`utility_results/qwen3.5-4b-q8_0/` and `tables/qwen3.5-4b-q8_0/`. Other GGUF models can be
passed as arguments: `./run_model.sh <gguf_repo> <gguf_file> <chat_tokenizer>`.


## Main measures

- **P(A)**: maximum BERTScore F1 of an answer against the passages judged 2 or 3. P_0: zero-shot answer; P_1: answer from the top passage only;
  P_k: answer from the top-k passages.
- **ECIR delta**: (P_k − P_0) / P_0 (corresponds to Eq. 2 the Tian et al. 2025 ECIR paper, which is why we give this notation to distinguish it from our paper's notation).
- **Empirical gain of passages 2..k**: Δ_emp = P_k − P_1.
- **Joint context utility**: U(C_k) = r(d_1) + G_k − γ L_k, with r the MonoT5 probability of
  relevance and G_k, L_k averages of a pairwise interaction score over the passage pairs of
  the top-k (PMI or query-likelihood complementarity); γ = 0.6.
- **Predictive delta**: Δ_pred(k) = U(C_k) − U(C_1) = G_k − γ L_k.


