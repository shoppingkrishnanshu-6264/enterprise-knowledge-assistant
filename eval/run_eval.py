"""
Runs the full agent against the ground-truth evaluation set and scores it.

Metrics (same methodology as RAGAS, implemented as a lightweight local LLM-as-judge
to avoid a heavy, fragile dependency chain that conflicted with this environment):
- faithfulness: does the answer stick to what the retrieved evidence actually says,
  or does it hallucinate facts not present in the evidence?
- answer_relevancy: does the answer actually address the question asked?

Everything runs locally via Ollama — no API key, no cost.

Run from the project root:
    python3 -m eval.run_eval

Output:
- Prints a summary table of scores per category
- Saves detailed results (question, answer, contexts, scores, reasons) to eval/results.csv
"""

import os
import pandas as pd

from src.agents.graph import ask
from eval.eval_dataset import EVAL_QUESTIONS
from eval.llm_judge_metrics import score_faithfulness, score_answer_relevancy

RESULTS_DIR = "eval"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")


def run_full_evaluation():
    rows = []

    for i, item in enumerate(EVAL_QUESTIONS):
        print(f"\n[{i + 1}/{len(EVAL_QUESTIONS)}] Running: {item['question']}")

        result = ask(item["question"])
        answer = result["final_answer"]

        contexts = [str(e.get("content", "")) for e in result["evidence"] if e.get("content")]
        if not contexts:
            contexts = ["No evidence retrieved."]

        print("   -> Scoring faithfulness...")
        faithfulness_result = score_faithfulness(answer, contexts)

        print("   -> Scoring answer relevancy...")
        relevancy_result = score_answer_relevancy(item["question"], answer)

        rows.append({
            "question": item["question"],
            "category": item["category"],
            "answer": answer,
            "ground_truth": item["ground_truth"],
            "retries": result["retry_count"],
            "faithfulness": faithfulness_result["score"],
            "faithfulness_reason": faithfulness_result["reason"],
            "answer_relevancy": relevancy_result["score"],
            "relevancy_reason": relevancy_result["reason"],
        })

        print(f"   Faithfulness: {faithfulness_result['score']:.2f} | Relevancy: {relevancy_result['score']:.2f}")

    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("Running evaluation: agent execution + faithfulness/relevancy scoring")
    print("=" * 70)

    rows = run_full_evaluation()

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved detailed results to: {RESULTS_CSV}")

    print("\n" + "=" * 70)
    print("SUMMARY (average scores by category)")
    print("=" * 70)
    summary = df.groupby("category")[["faithfulness", "answer_relevancy"]].mean()
    print(summary)

    print("\n" + "=" * 70)
    print("OVERALL AVERAGE SCORES")
    print("=" * 70)
    print(f"Faithfulness:     {df['faithfulness'].mean():.3f}")
    print(f"Answer Relevancy: {df['answer_relevancy'].mean():.3f}")

    low_faithfulness = df[df["faithfulness"] < 0.7]
    if not low_faithfulness.empty:
        print("\n" + "=" * 70)
        print("QUESTIONS WITH LOW FAITHFULNESS (< 0.7) — worth reviewing:")
        print("=" * 70)
        for _, row in low_faithfulness.iterrows():
            print(f"- [{row['faithfulness']:.2f}] {row['question']}")
            print(f"  Reason: {row['faithfulness_reason']}")


if __name__ == "__main__":
    main()
