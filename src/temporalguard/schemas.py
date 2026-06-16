"""Typed dataclasses shared across every layer.

These are the data contract: corpus -> chunks -> retrieval -> baseline -> eval.
Stdlib dataclasses only, so they import anywhere with no heavy deps.

Design notes:
- ``Document.doc_id`` IS the EnterpriseRAG-Bench ``dsid`` so retrieved doc ids
  map back to the bench with zero translation (needed by the leaderboard track).
- ``updated_at`` is preserved because it is the discriminator for recency in the
  bench's conflicting_info pairs (e.g. an "applied on <date>" note supersedes an
  earlier suggestion). Phase 1 stores it; Phase 6 uses it.
- ``authority_score`` here is only a *default* derived from source type. Real
  authority is computed later from source_type + recency + LLM judging.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# The five final decision labels TemporalGuard may emit (per PRD / CLAUDE.md).
DECISION_LABELS = [
    "ANSWER",
    "NOT_FOUND",
    "LOW_CONFIDENCE_ABSTAIN",
    "CONFLICT_DETECTED",
    "STALE_INFO_RESOLVED",
]


@dataclass
class Document:
    doc_id: str
    title: str
    source_type: str
    created_at: str
    updated_at: str
    authority_score: float
    status: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        return cls(
            doc_id=d["doc_id"],
            title=d.get("title", ""),
            source_type=d["source_type"],
            created_at=d.get("created_at", "unknown"),
            updated_at=d.get("updated_at", d.get("created_at", "unknown")),
            authority_score=float(d.get("authority_score", 0.5)),
            status=d.get("status", "active"),
            text=d["text"],
            metadata=d.get("metadata", {}) or {},
        )


@dataclass
class Chunk:
    """A retrievable unit. All authority signals are scalar so they survive a
    FAISS sidecar round-trip (JSON) and any future vector-store metadata field."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    title: str
    source_type: str
    created_at: str
    updated_at: str
    authority_score: float
    status: str
    bench_source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            chunk_index=int(d.get("chunk_index", 0)),
            text=d["text"],
            title=d.get("title", ""),
            source_type=d.get("source_type", ""),
            created_at=d.get("created_at", "unknown"),
            updated_at=d.get("updated_at", "unknown"),
            authority_score=float(d.get("authority_score", 0.5)),
            status=d.get("status", "active"),
            bench_source=d.get("bench_source", ""),
        )


@dataclass
class EvalQuestion:
    question_id: str
    question: str
    category: str            # clear_answerable | unanswerable | conflicting_info | stale_info | partial_evidence
    expected_decision: str   # one of DECISION_LABELS
    gold_answer: Optional[str] = None
    gold_doc_ids: List[str] = field(default_factory=list)
    conflicting_doc_ids: List[str] = field(default_factory=list)
    answer_facts: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalQuestion":
        return cls(
            question_id=d["question_id"],
            question=d["question"],
            category=d["category"],
            expected_decision=d["expected_decision"],
            gold_answer=d.get("gold_answer"),
            gold_doc_ids=list(d.get("gold_doc_ids", [])),
            conflicting_doc_ids=list(d.get("conflicting_doc_ids", [])),
            answer_facts=list(d.get("answer_facts", [])),
            notes=d.get("notes", ""),
        )
