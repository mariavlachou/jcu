import os
import argparse
import pandas as pd
import torch
import transformers
from bert_score import BERTScorer

from calculate_utility import load_gold_passages

# Recomputes every BERTScore-based value of a run from its saved answers, in the current
# environment. Models that need a newer transformers (e.g. Qwen3.5, run in .venv-qwen) give
# different BERTScores there; rescoring in the main environment keeps P_0, P_1, P_k and the
# deltas comparable across models. Updates the files in place:
#   utility_{set}_{model}.csv: p_zero_shot, p_context_shot, ECIR_Empirical_Delta_pct,
#                              Our_Predictive_Delta_pct
#   p1_{set}_{model}.csv:      p_top1_shot


def max_f1(scorer, answer, gold_texts):
    """P(A(q)) of ECIR 2025: max BERTScore F1 against the gold (rel 2/3) passages."""
    if not isinstance(answer, str) or not answer or not gold_texts:
        return 0.0
    _, _, f1_scores = scorer.score([answer] * len(gold_texts), gold_texts)
    return float(torch.max(f1_scores).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute BERTScore values from saved answers.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-name", required=True, help="Model name in the result file names")
    parser.add_argument("--rescale-bertscore", action="store_true")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    scorer = BERTScorer(lang="en", rescale_with_baseline=args.rescale_bertscore, device=device)
    print(f"Rescoring with transformers {transformers.__version__}")

    for name in ["dl19", "dl20"]:
        gold = load_gold_passages(os.path.join("qrels", f"qrels_{name}_rel23.csv"))

        path = os.path.join(args.results_dir, f"utility_{name}_{args.model_name}.csv")
        df = pd.read_csv(path, dtype={"qid": str})
        # The zero-shot answer does not depend on k: score it once per query
        p0 = {qid: max_f1(scorer, answer, gold.get(qid, []))
              for qid, answer in df.groupby("qid")["answer_zero_shot"].first().items()}
        df["p_zero_shot"] = df["qid"].map(p0)
        df["p_context_shot"] = [max_f1(scorer, a, gold.get(q, [])) for q, a in zip(df["qid"], df["answer_context_shot"])]
        p0_clamped = df["p_zero_shot"].clip(lower=0.01)
        df["ECIR_Empirical_Delta_pct"] = (df["p_context_shot"] - p0_clamped) / p0_clamped * 100
        df["Our_Predictive_Delta_pct"] = (df["u_context_norm"] - p0_clamped) / p0_clamped * 100
        df["bertscore_transformers"] = transformers.__version__
        df.to_csv(path, index=False)
        print(f"Rescored {path}")

        path = os.path.join(args.results_dir, f"p1_{name}_{args.model_name}.csv")
        if os.path.exists(path):
            p1 = pd.read_csv(path, dtype={"qid": str})
            p1["p_top1_shot"] = [max_f1(scorer, a, gold.get(q, [])) for q, a in zip(p1["qid"], p1["answer_top1_shot"])]
            p1["bertscore_transformers"] = transformers.__version__
            p1.to_csv(path, index=False)
            print(f"Rescored {path}")
