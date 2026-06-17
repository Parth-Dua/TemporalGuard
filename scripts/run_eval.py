"""Score a RAG pipeline version and write results/<version>/.

Usage:
  python scripts/run_eval.py --version baseline \
      --outputs cache/eval_runs/baseline_outputs.jsonl
  python scripts/run_eval.py --version baseline --run-leaderboard   # also run the bench grader
"""
import argparse
import datetime as _dt

import _bootstrap  # noqa: F401

from temporalguard.config import load_yaml
from temporalguard.eval.metrics.metrics import score_version
from temporalguard.utils.json_utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="baseline", help="Pipeline version name -> results/<version>/")
    ap.add_argument("--outputs", default=None,
                    help="Outputs JSONL to score (default: results/<version>/outputs.jsonl)")
    ap.add_argument("--run-leaderboard", action="store_true", help="Also run the bench answer-eval grader")
    args = ap.parse_args()
    if args.outputs is None:
        args.outputs = f"results/{args.version}/outputs.jsonl"

    app = load_yaml("configs/app.yaml")
    outputs = read_jsonl(args.outputs)
    questions = read_jsonl(app["paths"]["eval_questions"])
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    m = score_version(args.version, outputs, questions, timestamp=ts, run_leaderboard=args.run_leaderboard)

    rel = m["reliability"]
    print(f"\n=== {args.version} (n={m['n_questions']}, {m['category_counts']}) ===")
    print(f"  answerable_accuracy        {rel['answerable_accuracy']}")
    print(f"  not_found_accuracy         {rel['not_found_accuracy']}")
    print(f"  conflict_detection_accuracy{rel['conflict_detection_accuracy']}")
    print(f"  unsupported_answer_rate    {rel['unsupported_answer_rate']}")
    print(f"  false_abstention_rate      {rel['false_abstention_rate']}")
    print(f"  safe_decision_accuracy     {rel['safe_decision_accuracy']}")
    print(f"  recall@k {m['retrieval']['recall_at_k']}  mrr {m['retrieval']['mrr']}")
    print(f"\nresults -> results/{args.version}/")


if __name__ == "__main__":
    main()
