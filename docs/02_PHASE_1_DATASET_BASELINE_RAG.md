# Phase 1 — Dataset + Baseline RAG

## Goal

Stand up the corpus, retrieval, and a deliberately **naive baseline RAG** that always
answers. The baseline will hallucinate on unanswerable questions and pick stale/conflicting
evidence — making TemporalGuard's later improvements measurable.

## Corpus: full EnterpriseRAG-Bench (not synthetic)

The corpus is the **full EnterpriseRAG-Bench** (~512k real enterprise docs across Slack,
Gmail, Linear, Google Drive, Hubspot, Fireflies, GitHub, Jira, Confluence). We index the
**entire corpus** — no subset selection — so every question's gold docs are present by
construction (no orphaned questions).

Each bench doc JSON is self-describing (`title_field_name`, `content_field_names`) and is
normalized into our `Document` schema, preserving `created_at` / `updated_at` / `status` /
`source_type` and a default `authority_score` (by source). `doc_id == bench dsid` so
retrieved ids map straight back for the leaderboard.

For the full-corpus run, all docs are consolidated into a **single Parquet** locally
(`scripts/consolidate_corpus.py`) and uploaded as one file — a classic Modal Volume caps
at ~500k inodes, which the 512k raw files exceed.

## Eval questions: map the bench's labeled types

| bench question_type | our category      | expected_decision | gold docs            |
| ------------------- | ----------------- | ----------------- | -------------------- |
| basic, semantic     | clear_answerable  | ANSWER            | 1 (imported)         |
| conflicting_info    | conflicting_info  | CONFLICT_DETECTED | 2 (recency-resolvable; staleness handled in Phase 6) |
| info_not_found      | unanswerable      | NOT_FOUND         | none (stays unanswerable) |

~340 of the 500 bench questions map cleanly (≈300 clear_answerable, 20 conflicting_info,
20 unanswerable). `answer_facts` are preserved for the Phase-2 leaderboard track.

## Retrieval setup

- embeddings: local `sentence-transformers/all-MiniLM-L6-v2` (L2-normalized)
- vector store: **FAISS** `IndexFlatIP` (cosine via inner product) + JSONL chunk sidecar
- chunking: `RecursiveCharacterTextSplitter` (chunk_size 900, overlap 120); each chunk carries
  scalar authority metadata so Phase 6 can compute recency/authority
- top_k: 5
- **full-corpus embedding/indexing runs on Modal (L4 GPU)**, writing the index to a Modal Volume;
  local dev uses a small slice (FAISS still, just fewer docs)

## Baseline RAG behavior

```text
question → retrieve top-k chunks → generate answer from chunks (DeepSeek)
```

The baseline has **no** abstention, conflict detection, temporal/authority scoring, or
evidence judging. It always answers — including unanswerable questions. Prompt lives in
`configs/prompts.yaml` (`baseline_answer`) and is deliberately naive. A `--mock` mode answers
extractively at $0 for tests/CI.

## Deliverables

- `data/synthetic/eval_questions.jsonl` (~340 mapped questions; our schema)
- `data/synthetic/documents.jsonl` + `dev_questions.jsonl` (local dev slice only)
- `src/temporalguard/`: `schemas.py`, `corpus/bench_import.py`, `data/chunking.py`,
  `retrieval/{embeddings,faiss_index,build,retriever}.py`, `llm/{cache,provider}.py`,
  `eval/{baseline,bench_adapter}.py`, `utils/{json_utils,ids}.py`, `config.py`
- `modal_app.py` (full-corpus L4 index build)
- `scripts/{build_corpus,index_corpus,run_baseline}.py`
- baseline outputs cached: `cache/eval_runs/baseline_outputs.jsonl` and bench-format
  `cache/eval_runs/baseline_bench_answers.jsonl`

## Command flow

```bash
# Local dev ($0)
python scripts/build_corpus.py --dev-docs 60 --dev-per-category 4
python scripts/index_corpus.py
python scripts/run_baseline.py --mock --questions dev --emit-bench

# Full corpus (Modal): consolidate -> upload one file -> GPU index
python scripts/consolidate_corpus.py        # 512k files -> data/processed/documents.parquet
modal secret create temporalguard-secrets DEEPSEEK_API_KEY=... HF_TOKEN=...
modal volume create temporalguard-data
modal volume put temporalguard-data data/processed/documents.parquet /corpus/documents.parquet
modal run modal_app.py::index_corpus
# then run the baseline over all ~340 questions against the full index
```

## Acceptance criteria

- `build_corpus.py` writes ~340 mapped questions; `info_not_found` have empty gold_doc_ids;
  `conflicting_info` have exactly 2 gold docs.
- Bench docs normalize with non-empty text, scalar metadata, and `doc_id == dsid`.
- `index_corpus` (local dev) builds a FAISS index; a query returns top_k chunks with metadata.
- `modal run modal_app.py::index_corpus` builds the full index on L4 and writes it to the Volume.
- `run_baseline` answers ALL questions including unanswerable ones (naive contrast), at $0 in
  `--mock` and cached otherwise.
- `baseline_bench_answers.jsonl` validates as `{question_id (qst_*), answer, document_ids (dsids)}`.

## Common pitfalls

- Do not embed the full corpus locally — it is far too slow; that pass belongs on Modal GPU.
- Do not over-optimize the baseline; it must be a realistic naive baseline.
- Do not build conflict/abstention/temporal logic yet (later phases).
- Keep `EnterpriseRAG-Bench/` present at runtime (the Phase-2 leaderboard track runs their code).
