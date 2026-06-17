"""FAISS index over chunk embeddings, with a JSONL payload sidecar.

Layout on disk (one directory):
  index.faiss        - IndexFlatIP over L2-normalized vectors (cosine via IP)
  chunks.jsonl       - one Chunk dict per row, aligned to FAISS row order
  meta.json          - {embedding_model, dim, count}

IndexFlatIP is exact and simple; ~2M chunks at 384-dim ≈ 3 GB RAM, fine on a
Modal container. Swap to IVF/HNSW later only if query latency demands it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np

from temporalguard.schemas import Chunk
from temporalguard.utils.json_utils import iter_jsonl


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    rank: int


class FaissIndex:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.payload_path = os.path.join(index_dir, "chunks.jsonl")
        self.meta_path = os.path.join(index_dir, "meta.json")
        self._index = None
        self._chunks: List[Chunk] = []

    # ---- build / persist ----
    def build(self, chunks: List[Chunk], vectors: np.ndarray, embedding_model: str) -> None:
        import faiss

        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"chunks ({len(chunks)}) != vectors ({vectors.shape[0]})")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self._index = index
        self._chunks = list(chunks)
        self._embedding_model = embedding_model
        self._dim = int(vectors.shape[1])

    def save(self) -> None:
        import faiss

        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(self._index, self.index_path)
        with open(self.payload_path, "w") as f:
            for c in self._chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        with open(self.meta_path, "w") as f:
            json.dump(
                {"embedding_model": self._embedding_model, "dim": self._dim, "count": len(self._chunks)},
                f,
                indent=2,
            )

    # ---- load / search ----
    def load(self) -> "FaissIndex":
        import faiss

        self._index = faiss.read_index(self.index_path)
        self._chunks = [Chunk.from_dict(row) for row in iter_jsonl(self.payload_path)]
        return self

    @property
    def count(self) -> int:
        return len(self._chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[SearchHit]:
        if self._index is None:
            raise RuntimeError("index not built/loaded")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, idxs = self._index.search(query_vec.astype("float32"), top_k)
        hits: List[SearchHit] = []
        for rank, (i, s) in enumerate(zip(idxs[0], scores[0])):
            if 0 <= i < len(self._chunks):
                hits.append(SearchHit(chunk=self._chunks[i], score=float(s), rank=rank))
        return hits
