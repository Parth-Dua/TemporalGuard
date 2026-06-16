"""Shared index-build pipeline used by both the local script and Modal.

docs (iterable of Document) -> chunks -> embeddings (batched) -> FAISS index.
Accumulates chunks in memory because FAISS IndexFlatIP needs all vectors to
build; for the full bench this is ~2M chunks (~3 GB) which fits on a Modal
container. ``log`` lets the caller stream progress.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional

import numpy as np

from temporalguard.data.chunking import chunk_documents
from temporalguard.retrieval.embeddings import Embedder
from temporalguard.retrieval.faiss_index import FaissIndex
from temporalguard.schemas import Chunk, Document


def build_index_from_docs(
    docs: Iterable[Document],
    index_dir: str,
    embedding_model: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    embed_batch: int = 4096,
    device: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> FaissIndex:
    embedder = Embedder(embedding_model, device=device)

    chunks: List[Chunk] = []
    vec_batches: List[np.ndarray] = []
    pending: List[str] = []

    def flush():
        if pending:
            vec_batches.append(embedder.encode(pending, batch_size=256))
            pending.clear()

    n_docs = 0
    last_doc_id = None
    for ch in chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
        if ch.doc_id != last_doc_id:           # new document boundary
            n_docs += 1
            last_doc_id = ch.doc_id
            if n_docs % 10000 == 0:
                log(f"  ingested {n_docs} docs / {len(chunks)} chunks "
                    f"({len(vec_batches)} batches embedded)")
        chunks.append(ch)
        pending.append(ch.text)
        if len(pending) >= embed_batch:
            flush()
            log(f"  embedded {len(chunks)} chunks ({n_docs} docs seen)...")

    flush()
    log(f"ingest complete: {n_docs} docs -> {len(chunks)} chunks")
    if not chunks:
        raise ValueError("no chunks produced — empty corpus?")

    vectors = np.vstack(vec_batches)
    log(f"chunks={len(chunks)} dim={vectors.shape[1]} — building FAISS index")

    index = FaissIndex(index_dir)
    index.build(chunks, vectors, embedding_model=embedding_model)
    index.save()
    log(f"saved index -> {index_dir}")
    return index
