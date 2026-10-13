#!/bin/sh
# Main results (BM25 >> MonoT5 contexts, TREC DL 2019/2020) with another LLM, by default
# Qwen3.5-4B (Q8_0 GGUF via llama.cpp, the same setup as the Llama-3-8B results; see README):
#   Phase 1: answers, P_0 / P_k and PMI utility (calculate_utility.py), top-1 answers P_1
#            (compare_p1.py), BERTScore recomputed in the main env (rescore_bertscore.py),
#            tables: top-1 baseline, nDCG@k correlations, zero-shot means/correlations
#   Phase 2: complementarity (Fix 2) and the fixes table (utility_fixes.py)
# The LLM steps run with LLM_PYTHON (an environment whose llama-cpp-python / transformers support
# the model, .venv-qwen); BERTScore and tables with MAIN_PYTHON, so that BERTScore matches the Llama
# results. Every LLM step resumes from its saved per-query results if interrupted.
#
# Usage: ./run_model.sh [gguf_repo] [gguf_file] [chat_tokenizer]
#        ./run_model.sh unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q8_0.gguf Qwen/Qwen3.5-4B
set -e
GGUF_REPO="${1:-unsloth/Qwen3.5-4B-GGUF}"
GGUF_FILE="${2:-Qwen3.5-4B-Q8_0.gguf}"
CHAT_TOKENIZER="${3:-Qwen/Qwen3.5-4B}"
LLM_PYTHON="${LLM_PYTHON:-.venv-qwen/bin/python}"
MAIN_PYTHON="${MAIN_PYTHON:-python}"

model_name="${GGUF_FILE%.gguf}"                                   # e.g. Qwen3.5-4B-Q8_0
tag=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')            # e.g. qwen3.5-4b-q8_0
results_dir="utility_results/$tag"
tables_dir="tables/$tag"
mkdir -p "$results_dir" "$tables_dir"
llm_args="--backend llama-cpp --gguf-repo $GGUF_REPO --gguf-file $GGUF_FILE --gguf-chat-tokenizer $CHAT_TOKENIZER"

# Models are read from the local Hugging Face cache (download them beforehand, see README)
export HF_HUB_OFFLINE=1

# Phase 1: answers, P_1 and PMI utility, then BERTScore rescoring and the PMI tables
for set in dl19 dl20; do
    $LLM_PYTHON -u calculate_utility.py $llm_args --run "retr_results/monot5_${set}_top100.csv" \
        --out-dir "$results_dir" >> "$results_dir/run_utility_$set.log" 2>&1
done
$LLM_PYTHON -u compare_p1.py $llm_args --results-dir "$results_dir" --tables-dir "$tables_dir" \
    >> "$results_dir/run_top1.log" 2>&1
$MAIN_PYTHON -u rescore_bertscore.py --results-dir "$results_dir" --model-name "$model_name" \
    >> "$results_dir/run_rescore.log" 2>&1
$MAIN_PYTHON -u compare_p1.py --gguf-file "$GGUF_FILE" --tables-only \
    --results-dir "$results_dir" --tables-dir "$tables_dir" >> "$results_dir/run_tables.log" 2>&1
$MAIN_PYTHON -u ndcg_correlation.py --model-name "$model_name" \
    --results-dir "$results_dir" --tables-dir "$tables_dir" >> "$results_dir/run_tables.log" 2>&1
$MAIN_PYTHON -u make_tables.py --model-name "$model_name" \
    --results-dir "$results_dir" --tables-dir "$tables_dir" >> "$results_dir/run_tables.log" 2>&1
echo "PMI results ready: $tables_dir"

# Phase 2: complementarity (Fix 2) and the fixes table (uses the rescored top-1 baseline)
$LLM_PYTHON -u utility_fixes.py $llm_args --results-dir "$results_dir" --tables-dir "$tables_dir" \
    >> "$results_dir/run_fixes.log" 2>&1
$MAIN_PYTHON -u utility_fixes.py --gguf-file "$GGUF_FILE" --tables-only \
    --results-dir "$results_dir" --tables-dir "$tables_dir" >> "$results_dir/run_tables.log" 2>&1
echo "Finished $model_name: results in $results_dir, tables in $tables_dir"
