"""Local sentence-transformers embeddings.

Lazy model load (importing this module is cheap). Embeddings are L2-normalized
so a FAISS inner-product index gives cosine similarity. Works on CPU locally and
GPU on Modal — sentence-transformers picks the device automatically, or pass one.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: List[str], batch_size: int = 256, show_progress: bool = False) -> np.ndarray:
        model = self._ensure_model()
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype("float32")

    @property
    def dim(self) -> int:
        return int(self._ensure_model().get_sentence_embedding_dimension())
