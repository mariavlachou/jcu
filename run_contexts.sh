#!/bin/sh
# Full pipeline for alternative context runs retr_results/{system}_{set}_top20.csv
# (e.g. top1-rand from build_random_runs.py, oracle-rel / oracle-nrel from build_oracle_runs.py):
# answers and PMI utility, top-1 baseline, fixes (complementarity) and nDCG@k correlations.
# Usage: ./run_contexts.sh top1-rand [oracle-rel ...]
# Every step resumes from its saved per-query results if interrupted.
set -e
model=Meta-Llama-3-8B-Instruct.Q8_0
for system in "$@"; do
    results_dir="utility_results/$system"
    tables_dir="tables/$system"
    mkdir -p "$results_dir" "$tables_dir"
    if [ "$system" = "top1-rand" ]; then
        # Same top-1 passage as MonoT5: reuse its P_1 instead of regenerating it
        for set in dl19 dl20; do
            [ -f "$results_dir/p1_${set}_$model.csv" ] || cp "utility_results/p1_${set}_$model.csv" "$results_dir/"
        done
    fi
    for set in dl19 dl20; do
        python -u calculate_utility.py --run "retr_results/${system}_${set}_top20.csv" \
            --out-dir "$results_dir" >> "$results_dir/run_utility_$set.log" 2>&1
    done
    python -u compare_p1.py --system "$system" --results-dir "$results_dir" --tables-dir "$tables_dir" \
        >> "$results_dir/run_top1.log" 2>&1
    python -u utility_fixes.py --system "$system" --results-dir "$results_dir" --tables-dir "$tables_dir" \
        >> "$results_dir/run_fixes.log" 2>&1
    python -u ndcg_correlation.py --system "$system" --results-dir "$results_dir" --tables-dir "$tables_dir" \
        >> "$results_dir/run_ndcg.log" 2>&1
    echo "Finished $system"
done
