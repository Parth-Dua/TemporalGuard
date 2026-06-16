"""Run naive baseline RAG over the eval questions; cache outputs.

Usage:
  python scripts/run_baseline.py --mock          # $0 extractive smoke test over the 70-question subset
  python scripts/run_baseline.py                  # real DeepSeek baseline over the subset
  python scripts/run_baseline.py --emit-bench     # also write bench answer_evaluation format
"""
import argparse

import _bootstrap  # noqa: F401

from temporalguard.config import load_dotenv, load_yaml
from temporalguard.eval.baseline import answer_baseline
from temporalguard.eval.bench_adapter import to_bench_answers
from temporalguard.llm.cache import LLMCache
from temporalguard.llm.provider import LLMProvider
from temporalguard.retrieval.retriever import Retriever
from temporalguard.utils.json_utils import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="Force mock (no API) mode")
    ap.add_argument("--index-dir", default=None, help="Override index dir (else configs/retrieval.yaml)")
    ap.add_argument("--emit-bench", action="store_true", help="Also write bench answer format")
    args = ap.parse_args()

    load_dotenv()
    app = load_yaml("configs/app.yaml")
    retr = load_yaml("configs/retrieval.yaml")
    prompts = load_yaml("configs/prompts.yaml")

    llm_cfg = dict(app["llm"])
    if args.mock:
        llm_cfg["mock"] = True

    index_dir = args.index_dir or retr["paths"]["index_dir"]
    retriever = Retriever(index_dir=index_dir, embedding_model=retr["embedding_model"], top_k=retr.get("top_k", 5))
    provider = LLMProvider(llm_cfg, cache=LLMCache(app["cache"]["llm_dir"]))

    questions = read_jsonl(app["paths"]["eval_questions"])

    rows, total_cost = [], 0.0
    for i, q in enumerate(questions, 1):
        res = answer_baseline(q["question"], retriever, provider, prompts, top_k=retr.get("top_k", 5))
        res.update(question_id=q["question_id"], category=q["category"], expected_decision=q["expected_decision"])
        rows.append(res)
        total_cost += res["cost_estimate"]
        if res["mock"]:
            tag = "mock"
        elif res["cached"]:
            tag = "cached"
        else:
            tag = "api"
        print(f"[{i:>3}/{len(questions)}] {q['question_id']:<30} ({tag})")

    out_path = app["paths"]["baseline_outputs"]
    write_jsonl(out_path, rows)
    print(f"\nWrote {len(rows)} baseline outputs -> {out_path}")
    print(f"Estimated total cost: ${total_cost:.5f}")

    if args.emit_bench:
        bench_path = app["paths"]["baseline_bench_answers"]
        write_jsonl(bench_path, to_bench_answers(rows))
        print(f"Wrote bench-format answers -> {bench_path}")


if __name__ == "__main__":
    main()
