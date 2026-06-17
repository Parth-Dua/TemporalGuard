"""Naive baseline RAG: retrieve top-k chunks, generate an answer from them.

Deliberately naive (Phase 1): no abstention, no conflict detection, no
temporal/authority reasoning, no evidence judging. It always answers — the
contrast point TemporalGuard improves on in later phases.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from temporalguard.eval.augment_generate.provider import LLMProvider
from temporalguard.vector_db import SearchHit
from temporalguard.eval.retrieval.retriever import Retriever


def _format_context(hits: List[SearchHit]) -> str:
    return "\n\n".join(
        f"[{h.chunk.doc_id}] ({h.chunk.source_type}, {h.chunk.created_at})\n{h.chunk.text}"
        for h in hits
    )


def answer_baseline(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    prompts: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    t0 = time.time()
    hits = retriever.retrieve(question, top_k=top_k)
    context = _format_context(hits)

    system = prompts["baseline_answer"]["system"]
    user = prompts["baseline_answer"]["user"].format(context=context, question=question)
    result = provider.generate(system=system, user=user)
    latency_ms = int((time.time() - t0) * 1000)

    # Preserve doc order (first occurrence) for citation / bench document_ids.
    seen, doc_ids = set(), []
    for h in hits:
        if h.chunk.doc_id not in seen:
            seen.add(h.chunk.doc_id)
            doc_ids.append(h.chunk.doc_id)

    return {
        "question": question,
        "answer": result.text,
        "retrieved_doc_ids": doc_ids,
        "retrieved_chunks": [
            {
                "chunk_id": h.chunk.chunk_id,
                "doc_id": h.chunk.doc_id,
                "source_type": h.chunk.source_type,
                "created_at": h.chunk.created_at,
                "updated_at": h.chunk.updated_at,
                "authority_score": h.chunk.authority_score,
                "status": h.chunk.status,
                "score": h.score,
                "rank": h.rank,
                "text": h.chunk.text,
            }
            for h in hits
        ],
        "model": result.model,
        "cached": result.cached,
        "mock": result.mock,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_estimate": round(result.cost_estimate, 6),
        "latency_ms": latency_ms,
    }
