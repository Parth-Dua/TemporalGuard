"""Map EnterpriseRAG-Bench into the TemporalGuard schema.

The full bench (~512k docs, 500 questions) is the corpus. We index *every*
document, so every question's gold docs are present by construction — no subset
selection, no orphaned questions.

Two entry points:
  - ``iter_all_bench_docs``: stream every bench doc as a ``Document`` (generator;
    the corpus does not fit comfortably in RAM as a list).
  - ``map_questions``: convert the bench questions we support into ``EvalQuestion``.

Bench question_type -> our (category, expected_decision, import_gold):
  basic / semantic   -> clear_answerable / ANSWER            (single gold doc)
  conflicting_info   -> conflicting_info / CONFLICT_DETECTED  (exactly 2 golds; recency-resolvable,
                        but Phase 1 labels them CONFLICT only; staleness is Phase 6)
  info_not_found     -> unanswerable / NOT_FOUND             (no gold docs)

All reads are local files — no API cost. doc_id == bench dsid throughout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from temporalguard.schemas import Document, EvalQuestion

# Bench source directory -> (our source_type, default authority weight).
# Mirrors configs/source_authority.yaml; per-doc authority refined later.
SOURCE_MAP: Dict[str, Tuple[str, float]] = {
    "confluence": ("wiki", 0.70),
    "google_drive": ("wiki", 0.70),
    "github": ("github", 0.55),
    "jira": ("jira", 0.55),
    "linear": ("jira", 0.55),
    "slack": ("slack", 0.40),
    "gmail": ("email", 0.65),
    "fireflies": ("email", 0.60),
    "hubspot": ("product_doc", 0.60),
}

# Bench question_type -> (category, expected_decision, import_gold_docs?)
QUESTION_TYPE_MAP: Dict[str, Tuple[str, str, bool]] = {
    "basic": ("clear_answerable", "ANSWER", True),
    "semantic": ("clear_answerable", "ANSWER", True),
    "conflicting_info": ("conflicting_info", "CONFLICT_DETECTED", True),
    "info_not_found": ("unanswerable", "NOT_FOUND", False),
}

# Date keys to try, in priority order. Mix of bench source conventions.
_DATE_KEYS_CREATED = ("created_at", "recorded_at", "first_email_at", "first_message_ts")
_DATE_KEYS_UPDATED = ("last_updated", "updated_at", "last_modified", "last_email_at", "last_message_ts", "merged_at")


def _stringify(value: Any) -> str:
    """Flatten a bench content field (str / list / dict) into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_stringify(v) for v in value if v)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_stringify(v)}" for k, v in value.items() if v)
    return str(value)


def _pick_date(raw: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    """Return the first YYYY-MM-DD-looking value among ``keys``, else 'unknown'.

    Slack/gmail use unix timestamps in some fields; those won't match the
    YYYY check and fall through, which is fine — created_at is preferred and is
    a date string for the sources that have it."""
    for key in keys:
        val = raw.get(key)
        if isinstance(val, str) and len(val) >= 4 and val[:4].isdigit():
            return val[:10]
    return "unknown"


def normalize_bench_doc(dsid: str, rel_path: str, sources_root: Path) -> Optional[Document]:
    """Read one bench file and map it into a ``Document``. None if unreadable/empty."""
    path = sources_root / rel_path
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    source_dir = rel_path.split("/", 1)[0]
    source_type, authority = SOURCE_MAP.get(source_dir, ("wiki", 0.6))

    title_field = raw.get("title_field_name", "title")
    content_fields = raw.get("content_field_names", ["body"])
    title = (_stringify(raw.get(title_field, "")) or rel_path)[:200]
    text = "\n\n".join(_stringify(raw.get(f, "")) for f in content_fields).strip()
    if not text:
        return None

    created = _pick_date(raw, _DATE_KEYS_CREATED)
    updated = _pick_date(raw, _DATE_KEYS_UPDATED)
    if updated == "unknown":
        updated = created

    return Document(
        doc_id=dsid,
        title=title,
        source_type=source_type,
        created_at=created,
        updated_at=updated,
        authority_score=authority,
        status=str(raw.get("status", raw.get("state", "active"))).lower() or "active",
        text=text,
        metadata={"bench_source": source_dir, "bench_path": rel_path},
    )


def load_uuid_index(bench_root: str) -> Dict[str, str]:
    return json.loads((Path(bench_root) / "generated_data" / "uuid_index.json").read_text())


def iter_bench_docs_slice(
    bench_root: str, start: int = 0, end: Optional[int] = None
) -> Iterator[Document]:
    """Stream the bench docs whose uuid_index position is in ``[start, end)``.

    Used to shard consolidation across parallel processes: each process handles a
    disjoint index range, so the ~512k small-file reads happen concurrently
    instead of one slow serial pass.
    """
    sources_root = Path(bench_root) / "generated_data" / "sources"
    items = list(load_uuid_index(bench_root).items())
    for dsid, rel_path in items[start:end]:
        doc = normalize_bench_doc(dsid, rel_path, sources_root)
        if doc is not None:
            yield doc


def iter_all_bench_docs(bench_root: str) -> Iterator[Document]:
    """Stream every bench document as a ``Document`` (skips unreadable/empty)."""
    yield from iter_bench_docs_slice(bench_root, 0, None)


def bench_doc_count(bench_root: str) -> int:
    return len(load_uuid_index(bench_root))


def map_questions(bench_root: str, include_unresolvable: bool = False) -> List[EvalQuestion]:
    """Convert supported bench questions into ``EvalQuestion`` objects.

    Since the full corpus is indexed, every gold doc resolves; we still verify
    against uuid_index and skip any answerable/conflict question with a missing
    gold (defensive) unless ``include_unresolvable`` is set.
    """
    root = Path(bench_root)
    uuid_index = load_uuid_index(bench_root)

    out: List[EvalQuestion] = []
    with open(root / "questions.jsonl") as f:
        bench_questions = [json.loads(line) for line in f if line.strip()]

    for q in bench_questions:
        mapping = QUESTION_TYPE_MAP.get(q.get("question_type"))
        if not mapping:
            continue
        category, expected_decision, import_gold = mapping

        gold_ids = [e for e in q.get("expected_doc_ids", []) if e in uuid_index]
        if import_gold and not include_unresolvable:
            if len(gold_ids) != len(q.get("expected_doc_ids", [])):
                # A gold doc id not present in the index — skip to keep golds honest.
                continue

        out.append(
            EvalQuestion(
                question_id=f"bench_{q['question_id']}",
                question=q["question"],
                category=category,
                expected_decision=expected_decision,
                gold_answer=q.get("gold_answer"),
                gold_doc_ids=gold_ids if import_gold else [],
                conflicting_doc_ids=gold_ids if category == "conflicting_info" else [],
                answer_facts=list(q.get("answer_facts", [])),
                notes=f"Imported from EnterpriseRAG-Bench ({q.get('question_type')}).",
            )
        )
    return out


def category_breakdown(questions: List[EvalQuestion]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1
    return counts
