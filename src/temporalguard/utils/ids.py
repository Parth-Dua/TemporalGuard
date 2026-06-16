"""Deterministic id helpers."""
from __future__ import annotations


def chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}::chunk_{chunk_index:03d}"
