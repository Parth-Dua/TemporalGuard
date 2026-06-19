"""v2 RAG pipeline: wide retrieve -> cross-encoder rerank -> structured generate.

Accuracy-focused successor to the naive baseline. Differences from baseline.py:
  - retrieves a wide candidate set (retrieve_k) then reranks to top_k
  - uses a better, evidence-grounded prompt
  - returns an EXPLICIT structured decision (no regex derivation)

Output row shape is a superset of baseline's (so the metrics orchestrator works
unchanged) plus: decision, confidence, used_doc_ids.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from temporalguard.eval.augment_generate.structured import StructuredProvider
from temporalguard.eval.retrieval.reranker import Reranker
from temporalguard.eval.retrieval.retriever import Retriever
from temporalguard.vector_db import SearchHit


def _format_context(hits: List[SearchHit]) -> str:
    return "\n\n".join(
        f"[{h.chunk.doc_id}] ({h.chunk.source_type})\n{h.chunk.text}" for h in hits
    )


def answer_v2(
    question: str,
    retriever: Retriever,
    reranker: Reranker,
    provider: StructuredProvider,
    prompts: Dict[str, Any],
    retrieve_k: int = 50,
    top_k: int = 15,
) -> Dict[str, Any]:
    t0 = time.time()
    candidates = retriever.retrieve(question, top_k=retrieve_k)
    hits = reranker.rerank(question, candidates, top_k=top_k)
    context = _format_context(hits)

    system = prompts["v2_answer"]["system"]
    user = prompts["v2_answer"]["user"].format(context=context, question=question)
    res = provider.generate(system=system, user=user)
    latency_ms = int((time.time() - t0) * 1000)

    seen, doc_ids = set(), []
    for h in hits:
        if h.chunk.doc_id not in seen:
            seen.add(h.chunk.doc_id)
            doc_ids.append(h.chunk.doc_id)

    return {
        "question": question,
        "answer": res.answer,
        "decision": res.decision,                 # explicit (no regex)
        "confidence": res.confidence,
        "used_doc_ids": res.used_doc_ids,
        "retrieved_doc_ids": doc_ids,
        "retrieved_chunks": [
            {
                "chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id,
                "source_type": h.chunk.source_type, "rerank_score": h.score, "rank": h.rank,
                "text": h.chunk.text,
            }
            for h in hits
        ],
        "model": res.model,
        "cached": res.cached,
        "mock": res.mock,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "cost_estimate": round(res.cost_estimate, 6),
        "latency_ms": latency_ms,
    }
