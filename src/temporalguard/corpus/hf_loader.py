"""Load EnterpriseRAG-Bench directly from HuggingFace.

This is the default corpus path: no local files, no consolidation, no Volume
gymnastics. The HF dataset has two configs:

  documents : {doc_id, source_type, title, content}
  questions : {question_id, question_type, source_types, question,
               expected_doc_ids, gold_answer, answer_facts}

We map each into our schema. NOTE: the HF `documents` config exposes only
doc_id/source_type/title/content — it does NOT carry per-doc created_at /
updated_at / status (those lived in the raw JSON tree). So those fields default
to "unknown"/"active" here, and authority is derived from source_type. The
temporal/authority resolver (Phase 6) still has source_type + (later) any dates
we can recover; if recency signals are needed we can join back to the raw export.
"""
from __future__ import annotations

from typing import Iterator, List

from temporalguard.corpus.bench_import import QUESTION_TYPE_MAP, SOURCE_MAP
from temporalguard.schemas import Document, EvalQuestion

HF_DATASET = "onyx-dot-app/EnterpriseRAG-Bench"


def _authority_for(source_type: str) -> float:
    # SOURCE_MAP keys are bench source dirs -> (our_source_type, authority).
    for _dir, (st, auth) in SOURCE_MAP.items():
        if st == source_type or _dir == source_type:
            return auth
    return 0.6


def iter_hf_docs(cache_dir: str | None = None, token: str | None = None) -> Iterator[Document]:
    """Stream every HF document row as a ``Document`` (memory-safe via streaming)."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, "documents", split="test", streaming=True,
                      cache_dir=cache_dir, token=token)
    for row in ds:
        text = (row.get("content") or "").strip()
        if not text:
            continue
        source_type = row.get("source_type", "wiki")
        yield Document(
            doc_id=row["doc_id"],
            title=(row.get("title") or row["doc_id"])[:200],
            source_type=source_type,
            created_at="unknown",
            updated_at="unknown",
            authority_score=_authority_for(source_type),
            status="active",
            text=text,
            metadata={"origin": "hf:" + HF_DATASET},
        )


def select_subset(questions: List[EvalQuestion], per_category: dict) -> List[EvalQuestion]:
    """Deterministic first-N-per-category subset (questions pre-sorted by id).

    ``per_category`` e.g. {"clear_answerable": 30, "unanswerable": 20, "conflicting_info": 20}.
    """
    by_cat: dict = {}
    for q in sorted(questions, key=lambda x: x.question_id):
        by_cat.setdefault(q.category, []).append(q)
    out: List[EvalQuestion] = []
    for cat, n in per_category.items():
        out.extend(by_cat.get(cat, [])[:n])
    return out


def map_hf_questions(cache_dir: str | None = None, token: str | None = None) -> List[EvalQuestion]:
    """Map supported HF questions into ``EvalQuestion`` (same category mapping as the JSON path)."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, "questions", split="test",
                      cache_dir=cache_dir, token=token)
    out: List[EvalQuestion] = []
    for q in ds:
        mapping = QUESTION_TYPE_MAP.get(q.get("question_type"))
        if not mapping:
            continue
        category, expected_decision, import_gold = mapping
        gold_ids = list(q.get("expected_doc_ids", []) or [])
        out.append(
            EvalQuestion(
                question_id=f"bench_{q['question_id']}",
                question=q["question"],
                category=category,
                expected_decision=expected_decision,
                gold_answer=q.get("gold_answer"),
                gold_doc_ids=gold_ids if import_gold else [],
                conflicting_doc_ids=gold_ids if category == "conflicting_info" else [],
                answer_facts=list(q.get("answer_facts", []) or []),
                notes=f"Imported from EnterpriseRAG-Bench HF ({q.get('question_type')}).",
            )
        )
    return out
