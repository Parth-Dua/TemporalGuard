"""Build the eval question set from HuggingFace.

The corpus itself is loaded directly from HF on Modal at index time; this script
only produces the labeled eval questions (our schema) used to score the baseline
and TemporalGuard. By default it writes the 70-question evaluation subset:
30 clear_answerable + 20 unanswerable + 20 conflicting_info.

Usage:
  python scripts/build_corpus.py                 # 70-question subset
  python scripts/build_corpus.py --all           # all supported mapped questions
"""
import argparse

import _bootstrap  # noqa: F401

from temporalguard.config import load_dotenv, load_yaml
from temporalguard.corpus.bench_import import category_breakdown
from temporalguard.corpus.hf_loader import map_hf_questions, select_subset
from temporalguard.utils.json_utils import write_jsonl

SUBSET = {"clear_answerable": 30, "unanswerable": 20, "conflicting_info": 20}


def main():
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Use all mapped questions instead of the 70-subset.")
    args = ap.parse_args()

    load_dotenv()
    app = load_yaml("configs/app.yaml")
    q_path = app["paths"]["eval_questions"]
    token = os.environ.get("HF_TOKEN")

    questions = map_hf_questions(token=token)
    if not args.all:
        questions = select_subset(questions, SUBSET)

    n = write_jsonl(q_path, [q.to_dict() for q in questions])
    print(f"Wrote {n} questions -> {q_path}")
    print("By category:", category_breakdown(questions))


if __name__ == "__main__":
    main()
