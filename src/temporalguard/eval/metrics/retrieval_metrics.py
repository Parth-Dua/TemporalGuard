"""Retrieval metrics: Recall@k and MRR against gold doc ids.

Operates on baseline/TemporalGuard output rows (which carry `retrieved_doc_ids`)
joined to the eval questions (which carry `gold_doc_ids`). Only questions that
*have* gold docs (answerable + conflicting) are scored.
"""
from __future__ import annotations

from typing import Any, Dict, List


def compute_retrieval_metrics(rows: List[Dict[str, Any]], gold_by_qid: Dict[str, List[str]]) -> Dict[str, Any]:
    recall_hits = 0
    rr_sum = 0.0
    scored = 0
    per_category: Dict[str, List[int]] = {}

    for r in rows:
        gold = set(gold_by_qid.get(r["question_id"], []))
        if not gold:
            continue
        scored += 1
        retrieved = list(r.get("retrieved_doc_ids", []))
        hit = bool(gold & set(retrieved))
        recall_hits += int(hit)
        # reciprocal rank of the first gold doc among retrieved
        rr = 0.0
        for i, d in enumerate(retrieved, start=1):
            if d in gold:
                rr = 1.0 / i
                break
        rr_sum += rr
        per_category.setdefault(r["category"], [0, 0])
        per_category[r["category"]][0 if hit else 1] += 1

    return {
        "scored_questions": scored,
        "recall_at_k": (recall_hits / scored) if scored else None,
        "mrr": (rr_sum / scored) if scored else None,
        "recall_by_category": {
            c: {"hit": h, "miss": m, "recall": h / (h + m) if (h + m) else None}
            for c, (h, m) in per_category.items()
        },
    }
