"""Cross-encoder reranker for v2 retrieval.

Dense retrieval (MiniLM bi-encoder) is fast but coarse — baseline recall@5 was 50%.
A cross-encoder scores each (query, chunk) pair jointly, which is far more accurate
at ordering. v2 retrieves a wide candidate set from FAISS (retrieve_k≈50) and reranks
down to top_k≈15. Local model, $0, runs on CPU or GPU.
"""
from __future__ import annotations

from typing import List, Optional

from temporalguard.vector_db import SearchHit


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, hits: List[SearchHit], top_k: int) -> List[SearchHit]:
        """Re-score `hits` by cross-encoder relevance to `query`; return top_k, re-ranked."""
        if not hits:
            return []
        model = self._ensure_model()
        scores = model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)
        out: List[SearchHit] = []
        for rank, i in enumerate(order[:top_k]):
            h = hits[i]
            out.append(SearchHit(chunk=h.chunk, score=float(scores[i]), rank=rank))
        return out
