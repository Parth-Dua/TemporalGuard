"""Convert TemporalGuard baseline rows into EnterpriseRAG-Bench answer format.

Their answer_evaluation harness expects JSONL lines:
    {"question_id": "qst_0001", "answer": "...", "document_ids": ["dsid_abc", ...]}

We stored question ids as "bench_qst_XXXX" and our doc_ids ARE bench dsids, so
the only translation is stripping the "bench_" prefix.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _strip_prefix(question_id: str) -> str:
    return question_id[len("bench_"):] if question_id.startswith("bench_") else question_id


def to_bench_answers(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "question_id": _strip_prefix(row["question_id"]),
                "answer": row.get("answer", ""),
                "document_ids": list(dict.fromkeys(row.get("retrieved_doc_ids", []))),
            }
        )
    return out
