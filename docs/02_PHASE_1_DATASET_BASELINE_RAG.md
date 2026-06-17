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

The corpus is loaded **directly from HuggingFace** (`onyx-dot-app/EnterpriseRAG-Bench`,
`documents` config) on Modal — no local files, no uploads. The `documents` config exposes
`doc_id`/`source_type`/`title`/`content` (no per-doc dates/status), which is why temporal/
staleness is de-scoped; authority defaults by source type.

## Eval questions: map the bench's labeled types

| bench question_type | our category      | expected_decision | gold docs            |
| ------------------- | ----------------- | ----------------- | -------------------- |
| basic, semantic     | clear_answerable  | ANSWER            | 1                    |
| conflicting_info    | conflicting_info  | CONFLICT_DETECTED | 2                    |
| info_not_found      | unanswerable      | NOT_FOUND         | none (stays unanswerable) |

The **eval subset = 70 questions**: 30 clear_answerable + 20 unanswerable + 20 conflicting_info,
selected deterministically (first-N per category). `answer_facts` are preserved for the
Phase-2 leaderboard track.

## Retrieval setup

- embeddings: local `sentence-transformers/all-MiniLM-L6-v2` (L2-normalized)
- vector store: **FAISS** `IndexFlatIP` (cosine via inner product) + JSONL chunk sidecar
- chunking: `RecursiveCharacterTextSplitter` (chunk_size 900, overlap 120); each chunk carries
  scalar metadata (source_type, authority_score) for the reliability layer
- top_k: 5
- **full-corpus embedding/indexing runs on Modal (L4 GPU)** from HuggingFace, writing the index
  to a Modal Volume; local dev indexes a small HF slice

## Baseline RAG behavior

```text
question → retrieve top-k chunks → generate answer from chunks (DeepSeek)
```

The baseline has **no** abstention, conflict detection, temporal/authority scoring, or
evidence judging. It always answers — including unanswerable questions. Prompt lives in
`configs/prompts.yaml` (`baseline_answer`) and is deliberately naive. A `--mock` mode answers
extractively at $0 for tests/CI.

## Deliverables

- `data/synthetic/eval_questions.jsonl` (70-question eval subset; our schema)
- `src/temporalguard/`: `schemas.py`, `corpus/hf_loader.py`, `corpus/bench_import.py`,
  `data/chunking.py`, `retrieval/{embeddings,faiss_index,build,retriever}.py`,
  `llm/{cache,provider}.py`, `eval/{baseline,bench_adapter}.py`, `utils/{json_utils,ids}.py`, `config.py`
- `modal_app.py` (full-corpus L4 index build from HuggingFace)
- `scripts/{build_corpus,index_corpus,run_baseline}.py`
- baseline outputs cached: `cache/eval_runs/baseline_outputs.jsonl` and bench-format
  `cache/eval_runs/baseline_bench_answers.jsonl`

## Command flow

```bash
# Build the 70-question eval subset (small HF questions config only)
python scripts/build_corpus.py

# Local dev ($0): index a small HF slice, mock baseline
python scripts/index_corpus.py --limit 2000
python scripts/run_baseline.py --mock --emit-bench

# Full corpus (Modal L4): downloads docs from HF, embeds on GPU -> FAISS on Volume
modal secret create temporalguard-secrets DEEPSEEK_API_KEY=... HF_TOKEN=...
modal run modal_app.py::index_corpus
# then run the baseline over the 70-question subset against the full index
```

## Acceptance criteria

- `build_corpus.py` writes the 70-question subset (30/20/20); `info_not_found` have empty
  gold_doc_ids; `conflicting_info` have exactly 2 gold docs.
- HF docs normalize with non-empty text, scalar metadata, and `doc_id` == the HF `doc_id`.
- `index_corpus.py --limit N` (local dev) builds a FAISS index; a query returns top_k chunks with metadata.
- `modal run modal_app.py::index_corpus` builds the full index on L4 and writes it to the Volume.
- `run_baseline` answers ALL questions including unanswerable ones (naive contrast), at $0 in
  `--mock` and cached otherwise.
- `baseline_bench_answers.jsonl` validates as `{question_id (qst_*), answer, document_ids}`.

## Common pitfalls

- Do not embed the full corpus locally — it is far too slow; that pass belongs on Modal GPU.
- Do not over-optimize the baseline; it must be a realistic naive baseline.
- Do not build conflict/abstention logic yet (later phases).
- The Phase-2 leaderboard track runs the bench's `answer_evaluation/` scripts; keep a clone of
  the EnterpriseRAG-Bench repo available for that (the corpus itself comes from HuggingFace).
