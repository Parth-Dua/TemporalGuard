"""Query-time retrieval: embed a question, return top-k chunk hits with metadata."""
from __future__ import annotations

from typing import List, Optional

from temporalguard.retrieval.embeddings import Embedder
from temporalguard.retrieval.faiss_index import FaissIndex, SearchHit


class Retriever:
    def __init__(self, index_dir: str, embedding_model: str, top_k: int = 5, device: Optional[str] = None):
        self.embedder = Embedder(embedding_model, device=device)
        self.index = FaissIndex(index_dir).load()
        self.top_k = top_k

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[SearchHit]:
        qvec = self.embedder.encode([question])
        return self.index.search(qvec, top_k=top_k or self.top_k)
