"""Build the eval question set (and, for local dev, a small documents slice).

Full corpus path: questions only here; the 512k docs are streamed straight into
the index on Modal (see modal_app.py / index_corpus). For local dev we also emit
a small documents.jsonl that is GUARANTEED to contain the gold docs of the
selected dev questions (so no question is orphaned locally).

Usage:
  python scripts/build_corpus.py                  # questions only
  python scripts/build_corpus.py --dev-docs 400   # + local dev documents.jsonl
  python scripts/build_corpus.py --include-seed    # append hand-authored demo seed
"""
import argparse
import itertools

import _bootstrap  # noqa: F401

from temporalguard.config import load_yaml
from temporalguard.corpus.bench_import import (
    QUESTION_TYPE_MAP,
    category_breakdown,
    iter_all_bench_docs,
    load_uuid_index,
    map_questions,
    normalize_bench_doc,
)
from temporalguard.schemas import Document, EvalQuestion
from temporalguard.utils.json_utils import write_jsonl


def _dev_questions(questions, per_category):
    """First N per category — deterministic dev subset."""
    buckets = {}
    for q in questions:
        buckets.setdefault(q.category, []).append(q)
    out = []
    for cat, qs in buckets.items():
        out.extend(qs[:per_category])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-docs", type=int, default=0,
                    help="If >0, also emit a local documents.jsonl: all gold docs of the "
                         "dev questions + filler up to this many total docs.")
    ap.add_argument("--dev-per-category", type=int, default=8,
                    help="Questions per category in the local dev set (used with --dev-docs).")
    ap.add_argument("--include-seed", action="store_true", help="Append hand-authored demo seed.")
    args = ap.parse_args()

    retr = load_yaml("configs/retrieval.yaml")
    app = load_yaml("configs/app.yaml")
    bench_root = retr["bench_root"]
    q_path = app["paths"]["eval_questions"]

    questions = map_questions(bench_root)

    # Determine which questions form the local dev set (so its index is self-consistent).
    dev_qs = _dev_questions(questions, args.dev_per_category) if args.dev_docs else []

    if args.include_seed:
        from temporalguard.corpus import load_seed

        _, seed_q = load_seed()
        questions = questions + [EvalQuestion.from_dict(q) for q in seed_q]

    n_q = write_jsonl(q_path, [q.to_dict() for q in questions])
    print(f"Wrote {n_q} questions -> {q_path}")
    print("By category:", category_breakdown(questions))

    if args.dev_docs:
        _build_dev_docs(bench_root, dev_qs, args.dev_docs, retr["paths"]["chunks"], app)


def _build_dev_docs(bench_root, dev_qs, target_docs, _chunks_path, app):
    """Emit data/synthetic/documents.jsonl for local dev: gold docs of dev_qs +
    filler, with a parallel dev question file so the local index is consistent."""
    uuid_index = load_uuid_index(bench_root)
    from pathlib import Path

    sources_root = Path(bench_root) / "generated_data" / "sources"

    gold_ids, docs = set(), []
    for q in dev_qs:
        gold_ids.update(q.gold_doc_ids)
    for dsid in sorted(gold_ids):
        d = normalize_bench_doc(dsid, uuid_index[dsid], sources_root)
        if d:
            docs.append(d)

    # Deterministic filler to reach target_docs, excluding golds.
    have = {d.doc_id for d in docs}
    for d in iter_all_bench_docs(bench_root):
        if len(docs) >= target_docs:
            break
        if d.doc_id not in have:
            docs.append(d)
            have.add(d.doc_id)

    docs_path = app["paths"].get("dev_documents", "data/synthetic/documents.jsonl")
    dev_q_path = app["paths"].get("dev_questions", "data/synthetic/dev_questions.jsonl")
    n = write_jsonl(docs_path, [d.to_dict() for d in docs])
    write_jsonl(dev_q_path, [q.to_dict() for q in dev_qs])
    print(f"[dev] Wrote {n} documents ({len(gold_ids)} gold + filler) -> {docs_path}")
    print(f"[dev] Wrote {len(dev_qs)} dev questions -> {dev_q_path}")


if __name__ == "__main__":
    main()
