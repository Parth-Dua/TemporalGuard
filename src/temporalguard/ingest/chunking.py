"""Chunk documents with LangChain's RecursiveCharacterTextSplitter.

Each chunk inherits all of its parent document's authority signals as scalar
fields (see schemas.Chunk), so the FAISS sidecar + any later vector store can
carry recency/source/status straight through to the reliability layers.
"""
from __future__ import annotations

from typing import Iterable, Iterator

from temporalguard.schemas import Chunk, Document
from temporalguard.utils.ids import chunk_id


def _splitter(chunk_size: int, chunk_overlap: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_document(doc: Document, splitter) -> Iterator[Chunk]:
    for i, piece in enumerate(splitter.split_text(doc.text)):
        piece = piece.strip()
        if not piece:
            continue
        yield Chunk(
            chunk_id=chunk_id(doc.doc_id, i),
            doc_id=doc.doc_id,
            chunk_index=i,
            text=piece,
            title=doc.title,
            source_type=doc.source_type,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            authority_score=doc.authority_score,
            status=doc.status,
            bench_source=doc.metadata.get("bench_source", ""),
        )


def chunk_documents(
    docs: Iterable[Document], chunk_size: int = 900, chunk_overlap: int = 120
) -> Iterator[Chunk]:
    """Stream chunks for an iterable of documents (memory-safe for the full bench)."""
    splitter = _splitter(chunk_size, chunk_overlap)
    for doc in docs:
        yield from chunk_document(doc, splitter)
