# TemporalGuard RAG — Architecture and Folder Structure

## 1. High-Level Architecture

```text
                   ┌─────────────────────┐
                   │   Streamlit UI       │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  FastAPI Backend     │
                   └──────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌────────────────┐     ┌────────────────┐
│ Retrieval    │      │ Reliability     │     │ Evaluation      │
│ Engine       │      │ Layer           │     │ Harness         │
└──────┬───────┘      └───────┬────────┘     └───────┬────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐      ┌────────────────┐     ┌────────────────┐
│ Vector Store │      │ LLM Judge /     │     │ Metrics +      │
│ Chroma/FAISS │      │ Heuristics      │     │ Result Cache    │
└──────────────┘      └────────────────┘     └────────────────┘
```

## 2. Core Components

### 2.1 Corpus Layer

Responsible for loading synthetic enterprise documents and optional benchmark documents.

Inputs:
- JSONL synthetic documents
- optional EnterpriseRAG-Bench subset

Outputs:
- normalized document objects
- chunks with metadata

### 2.2 Retrieval Layer

MVP retrieval:
- dense vector search using local `sentence-transformers/all-MiniLM-L6-v2` embeddings
- **FAISS** vector store (`IndexFlatIP` over normalized vectors; JSONL chunk sidecar)
- chunking via `RecursiveCharacterTextSplitter`; every chunk carries scalar authority
  metadata (source_type, created_at, updated_at, status, authority_score)

Heavy compute (embedding/indexing the full ~512k-doc corpus) runs on **Modal (L4 GPU)**;
the FAISS index + chunk sidecar are written to a **Modal Volume**.

Optional retrieval: BM25, hybrid dense + sparse, reranking.

### 2.3 Reliability Layer

This is the main differentiator.

Modules:
- evidence support judge
- answer candidate extractor
- abstention policy
- conflict detector
- temporal/source authority resolver
- final response composer

### 2.4 Evaluation Layer

Compares:
- baseline RAG
- TemporalGuard RAG

Metrics:
- answer correctness
- abstention recall
- conflict detection accuracy
- stale resolution accuracy
- unsupported answer rate
- latency/cost

### 2.5 Dashboard Layer

Streamlit app for:
- querying the system
- viewing evidence traces
- comparing metrics
- showing demo examples

## 3. Folder Structure

Actual Phase-1 layout (consolidated, FAISS-based; `(later)` = planned for a future phase):

```text
temporalguard-rag/
  README.md
  CLAUDE.md
  requirements.txt
  .env.example  .env  .gitignore
  modal_app.py                  # Modal: image + Volume + Secret + index_corpus (L4 GPU)

  EnterpriseRAG-Bench/          # cloned benchmark (full corpus + their answer_evaluation/)

  data/
    synthetic/
      eval_questions.jsonl      # ~340 mapped bench questions (our schema)
      documents.jsonl           # local dev slice only
      dev_questions.jsonl       # local dev question subset
    processed/
      chunks.jsonl              # (local dev only) chunk sidecar from a small HF slice

  cache/
    llm_calls/                  # cached LLM responses (by hash)
    index/dev/                  # local dev FAISS index (full index lives on the Modal Volume)

  results/                      # one folder per RAG pipeline version (sibling of cache/)
    baseline/
      metrics.json              # self-describing: reliability + retrieval + leaderboard + engineering
      metrics_rows.jsonl        # per-question: decision vs expected, gold_retrieved
      outputs.jsonl             # raw per-question outputs (gitignored)
      bench_answers.jsonl       # {question_id, answer, document_ids} (gitignored)
      REPORT.md                 # concise interpretation + next steps

  configs/
    app.yaml                    # LLM (DeepSeek), cache/output paths, modal section
    retrieval.yaml              # embedding model, top_k, chunking, FAISS dirs, bench_root
    source_authority.yaml       # default authority weight per source type
    prompts.yaml                # baseline prompt (later: judge/conflict prompts)

  src/temporalguard/
    __init__.py
    config.py                   # load_yaml / load_configs / load_dotenv / repo_path
    schemas.py                  # Document, Chunk, EvalQuestion (+ to/from_dict)
    embeddings.py               # shared: sentence-transformers encoder (ingest + eval)
    vector_db.py                # shared: FAISS build/save/load/search + chunk sidecar
    ingest/                     # everything to BUILD the index
      hf_loader.py              #   load EnterpriseRAG-Bench from HuggingFace; 70-question subset
      bench_import.py           #   SOURCE_MAP + QUESTION_TYPE_MAP (category mapping)
      chunking.py               #   RecursiveCharacterTextSplitter -> Chunk (metadata-carrying)
      build.py                  #   docs -> chunks -> embed -> FAISS (used by local dev + Modal)
    eval/                       # everything to RUN + SCORE a pipeline
      retrieval/
        retriever.py            #   query -> top_k SearchHit (with metadata)
      augment_generate/
        provider.py cache.py    #   DeepSeek (openai-compatible) + disk cache + mock
        baseline.py             #   naive answer_baseline()
      metrics/
        bench_adapter.py        #   our rows -> bench answers.jsonl
        reliability_metrics.py  #   our reliability track (not-found/conflict/abstention)
        retrieval_metrics.py    #   recall@k, MRR
        answer_metrics.py       #   runs the bench's leaderboard grader (their code)
        metrics.py              #   orchestrator -> results/<version>/
      reliability/              # (later) evidence_judge, abstention, conflict_detector, decision_policy
    utils/
      json_utils.py ids.py

  app/                          # (later) Streamlit dashboard
  scripts/
    build_corpus.py index_corpus.py run_baseline.py   # (later) run_eval.py
  tests/                        # (later) abstention / conflict / temporal / metrics

  docs/
    00_PRD.md
    01_ARCHITECTURE_AND_FOLDER_STRUCTURE.md
    02_PHASE_1_DATASET_BASELINE_RAG.md
    03_PHASE_2_EVALUATION_HARNESS.md
    04_PHASE_3_EVIDENCE_SUPPORT_JUDGE.md
    05_PHASE_4_ABSTENTION_LAYER.md
    06_PHASE_5_CONFLICT_DETECTION.md
    07_PHASE_6_TEMPORAL_AUTHORITY_RESOLVER.md
    08_PHASE_7_DASHBOARD_ABLATIONS_POLISH.md
```

## 4. Data Schemas

### 4.1 Document Schema

```json
{
  "doc_id": "runbook_deploy_2026_04",
  "title": "Deployment Runbook",
  "source_type": "runbook",
  "created_at": "2026-04-01",
  "authority_score": 0.95,
  "status": "active",
  "text": "Current deployment timeout is 60 seconds.",
  "metadata": {
    "department": "engineering",
    "topic": "deployment",
    "version": "2026-04"
  }
}
```

### 4.2 Chunk Schema

```json
{
  "chunk_id": "runbook_deploy_2026_04_chunk_001",
  "doc_id": "runbook_deploy_2026_04",
  "text": "Current deployment timeout is 60 seconds.",
  "source_type": "runbook",
  "created_at": "2026-04-01",
  "authority_score": 0.95,
  "metadata": {}
}
```

### 4.3 Eval Question Schema

```json
{
  "question_id": "q_deploy_timeout_current",
  "question": "What is the current deployment timeout?",
  "category": "stale_conflict",
  "expected_decision": "STALE_INFO_RESOLVED",
  "gold_answer": "60 seconds",
  "gold_doc_ids": ["runbook_deploy_2026_04"],
  "conflicting_doc_ids": ["wiki_deploy_2024_01"],
  "notes": "Old wiki says 30 seconds; newer runbook says 60 seconds."
}
```

### 4.4 Evidence Judgment Schema

```json
{
  "chunk_id": "runbook_deploy_2026_04_chunk_001",
  "label": "SUPPORTS",
  "answer_candidate": "60 seconds",
  "confidence": 0.91,
  "evidence_quote": "Current deployment timeout is 60 seconds.",
  "reasoning_summary": "The chunk directly answers the question."
}
```

### 4.5 Final Decision Schema

```json
{
  "decision": "STALE_INFO_RESOLVED",
  "answer": "The current deployment timeout is 60 seconds.",
  "trusted_sources": ["runbook_deploy_2026_04"],
  "rejected_or_conflicting_sources": ["wiki_deploy_2024_01"],
  "explanation": "A newer runbook overrides the older wiki page.",
  "confidence": 0.86
}
```

## 5. API Endpoints

### MVP FastAPI Endpoints

```text
POST /query
POST /evaluate
GET /eval-runs
GET /eval-runs/{run_id}
GET /documents/{doc_id}
GET /health
```

### `POST /query` Request

```json
{
  "question": "What is the current deployment timeout?",
  "top_k": 5,
  "use_temporalguard": true
}
```

### `POST /query` Response

```json
{
  "question": "What is the current deployment timeout?",
  "baseline_answer": "The timeout is 30 seconds.",
  "temporalguard": {
    "decision": "STALE_INFO_RESOLVED",
    "answer": "The current deployment timeout is 60 seconds.",
    "evidence_trace": [],
    "conflicts": [],
    "cost_estimate": 0.0012,
    "latency_ms": 1830
  }
}
```

## 6. Budget Controls

To keep total cost under $5:

- Use local embeddings.
- Use small eval set first: 25–50 queries.
- Cache all LLM calls by hash of prompt + input.
- Add `--dry-run` mode for eval pipeline.
- Use heuristic judges when possible.
- Avoid generating synthetic data with expensive API calls; write or script the corpus manually.

## 7. Compute Layer (Modal)

Heavy compute (embedding/indexing the full ~512k-doc corpus, ~630M tokens) runs on Modal:

- **Image**: python 3.11 + faiss-cpu, sentence-transformers, torch, numpy, openai,
  datasets, huggingface-hub, langchain-text-splitters; source shipped via `add_local_dir("src")`.
- **Volume** `temporalguard-data`: `/data/hf_cache` (HuggingFace dataset cache, downloaded once),
  `/data/index/full` (FAISS index + chunk sidecar).
- **Secret** `temporalguard-secrets`: `DEEPSEEK_API_KEY`, `HF_TOKEN`.
- **`index_corpus`** (`gpu="L4"`): stream docs from HuggingFace → chunk → embed (GPU) → FAISS
  → commit to Volume, with doc/chunk progress logging.
- Estimated: ~2 GPU-hours, within Modal credits. FAISS itself is $0.

### Why load from HuggingFace (not local files)
The raw bench corpus is **~512k loose files** — too many to upload to a classic Modal Volume
(500k-inode limit) and slow to read on a laptop. The dataset is published on HuggingFace
(`onyx-dot-app/EnterpriseRAG-Bench`, `documents` config), so Modal pulls it directly with
`datasets`, caches it on the Volume, and never touches local disk. Re-indexing (e.g. after a
chunking change) re-reads the cached dataset — no re-download.

## 8. Command Flow

```bash
# local dev (small HF slice, $0-ish)
python scripts/build_corpus.py            # 70-question eval subset from HF
python scripts/index_corpus.py --limit 2000   # index first 2k HF docs locally
python scripts/run_baseline.py --mock --emit-bench

# full corpus (Modal L4): downloads from HF, embeds on GPU -> FAISS on Volume
modal secret create temporalguard-secrets DEEPSEEK_API_KEY=... HF_TOKEN=...
modal run modal_app.py::index_corpus

# later phases
python scripts/run_eval.py
streamlit run app/streamlit_app.py
```
