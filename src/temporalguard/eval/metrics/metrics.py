"""Metrics orchestrator: score one RAG pipeline version and write results/<version>/.

For a given version (e.g. "baseline"), given its per-question output rows, this:
  1. derives a decision per row (if not already set),
  2. computes reliability metrics (ours) + retrieval metrics (recall@k, MRR),
  3. optionally runs the bench leaderboard track (their correctness/completeness/recall),
  4. writes a self-describing metrics.json + per-question rows + outputs + bench answers,
  5. writes a concise REPORT.md template.

results/<version>/
  metrics.json          # combined, self-describing (reliability + retrieval + leaderboard)
  metrics_rows.jsonl    # per-question: decision vs expected, retrieval hit
  outputs.jsonl         # raw per-question outputs (answer, retrieved chunks, cost, latency)
  bench_answers.jsonl   # {question_id, answer, document_ids} for the leaderboard track
  REPORT.md             # concise interpretation + next steps
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from temporalguard.config import repo_path
from temporalguard.eval.metrics.answer_metrics import run_bench_answer_eval
from temporalguard.eval.metrics.bench_adapter import to_bench_answers
from temporalguard.eval.metrics.reliability_metrics import (
    compute_reliability_metrics,
    derive_baseline_decision,
)
from temporalguard.eval.metrics.retrieval_metrics import compute_retrieval_metrics
from temporalguard.utils.json_utils import read_jsonl, write_jsonl

RESULTS_ROOT = "results"


def _ensure_decisions(rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        if not r.get("decision"):
            r["decision"] = derive_baseline_decision(r.get("answer", "") or "")


def score_version(
    version: str,
    outputs: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    timestamp: str,
    run_leaderboard: bool = False,
    bench_root: str = "EnterpriseRAG-Bench",
) -> Dict[str, Any]:
    """Score `outputs` for a pipeline `version`; write results/<version>/; return metrics dict."""
    _ensure_decisions(outputs)
    gold_by_qid = {q["question_id"]: q.get("gold_doc_ids", []) for q in questions}

    reliability = compute_reliability_metrics(outputs)
    retrieval = compute_retrieval_metrics(outputs, gold_by_qid)

    costs = [r.get("cost_estimate", 0.0) for r in outputs]
    lats = [r.get("latency_ms", 0) for r in outputs if r.get("latency_ms")]
    engineering = {
        "total_cost_usd": round(sum(costs), 6),
        "median_latency_ms": int(statistics.median(lats)) if lats else None,
    }

    out_dir = repo_path(f"{RESULTS_ROOT}/{version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_answers = to_bench_answers(outputs)
    write_jsonl(str(out_dir / "outputs.jsonl"), outputs)
    write_jsonl(str(out_dir / "bench_answers.jsonl"), bench_answers)
    write_jsonl(str(out_dir / "metrics_rows.jsonl"), [
        {
            "question_id": r.get("question_id"),
            "category": r["category"],
            "expected_decision": r["expected_decision"],
            "decision": r["decision"],
            "decision_correct": _decision_correct(r),
            "gold_retrieved": bool(set(gold_by_qid.get(r["question_id"], [])) & set(r.get("retrieved_doc_ids", []))),
        }
        for r in outputs
    ])

    leaderboard: Dict[str, Any] = {"ran": False, "reason": "skipped (run_leaderboard=False)"}
    if run_leaderboard:
        leaderboard = run_bench_answer_eval(str(out_dir / "bench_answers.jsonl"), bench_root=bench_root)

    metrics = {
        "version": version,
        "timestamp": timestamp,
        "n_questions": len(outputs),
        "category_counts": reliability["counts"],
        "reliability": {k: v for k, v in reliability.items() if k not in ("counts",)},
        "retrieval": retrieval,
        "leaderboard": leaderboard,
        "engineering": engineering,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_report(out_dir, metrics)
    return metrics


def _decision_correct(r: Dict[str, Any]) -> bool:
    from temporalguard.eval.metrics.reliability_metrics import _is_correct_decision

    return _is_correct_decision(r["expected_decision"], r["decision"])


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _write_report(out_dir: Path, m: Dict[str, Any]) -> None:
    rel, ret = m["reliability"], m["retrieval"]
    lb = m["leaderboard"]
    lb_line = "_(leaderboard track not run)_"
    if lb.get("ran"):
        lb_line = f"See `metrics.json` → leaderboard ({lb.get('results_path','')})."

    md = f"""# Results — {m['version']}

_{m['timestamp']}_ · n={m['n_questions']} · {m['category_counts']}

## Reliability (ours)
| Metric | Value |
|---|---|
| Answerable accuracy | {_pct(rel['answerable_accuracy'])} |
| Not-found accuracy | {_pct(rel['not_found_accuracy'])} |
| Conflict-detection accuracy | {_pct(rel['conflict_detection_accuracy'])} |
| Unsupported-answer rate (lower=better) | {_pct(rel['unsupported_answer_rate'])} |
| False-abstention rate (lower=better) | {_pct(rel['false_abstention_rate'])} |
| Safe-decision accuracy | {_pct(rel['safe_decision_accuracy'])} |

## Retrieval
| Metric | Value |
|---|---|
| Recall@k | {_pct(ret['recall_at_k'])} |
| MRR | {ret['mrr'] if ret['mrr'] is None else round(ret['mrr'],3)} |

Recall by category: {ret['recall_by_category']}

## Leaderboard (their grader)
{lb_line}

## Engineering
Cost ${m['engineering']['total_cost_usd']} · median latency {m['engineering']['median_latency_ms']} ms

## Interpretation & next steps
_(fill in: what this version shows, what to change next)_
"""
    (out_dir / "REPORT.md").write_text(md)
