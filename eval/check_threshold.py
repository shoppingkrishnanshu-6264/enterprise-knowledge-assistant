"""
CI gate for the GKE backup deploy workflow.

eval/run_eval.py always exits 0 (it just prints a warning for low-faithfulness
questions), so it can't be used directly as a pass/fail CI gate. This script
runs AFTER run_eval.py, reads the results.csv it just wrote, and exits
non-zero if the average faithfulness or answer_relevancy falls below
--threshold — which is what actually blocks build-and-push / deploy-gke.

Usage (run from repo root, after `python eval/run_eval.py`):
    python eval/check_threshold.py --threshold 0.7
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

RESULTS_CSV = Path(__file__).parent / "results.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum acceptable average score (default: 0.7, matching the "
             "low-faithfulness flag already used in run_eval.py)",
    )
    args = parser.parse_args()

    if not RESULTS_CSV.exists():
        print(f"ERROR: {RESULTS_CSV} not found — run eval/run_eval.py first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        print("ERROR: results.csv is empty — nothing to check.")
        sys.exit(1)

    faithfulness = df["faithfulness"].mean()
    answer_relevancy = df["answer_relevancy"].mean()

    print(f"Average faithfulness:     {faithfulness:.3f}")
    print(f"Average answer_relevancy: {answer_relevancy:.3f}")
    print(f"Threshold:                {args.threshold:.3f}")

    if faithfulness < args.threshold or answer_relevancy < args.threshold:
        print("\nFAILED — one or more scores are below threshold. Blocking deploy.")
        sys.exit(1)

    print("\nPASSED — scores meet threshold.")


if __name__ == "__main__":
    main()
