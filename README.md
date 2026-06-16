# TemporalGuard RAG

A **pre-generation reliability layer** for RAG. Before answering, TemporalGuard decides
whether retrieved evidence is sufficient, missing, contradictory, or stale — emitting one of:
`ANSWER` · `NOT_FOUND` · `LOW_CONFIDENCE_ABSTAIN` · `CONFLICT_DETECTED` · `STALE_INFO_RESOLVED`.

Evaluated on the full [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
corpus (~512k enterprise docs). Two evaluation tracks: the benchmark's own leaderboard grader
(correctness / completeness / recall), and a **reliability track** (abstention, conflict, and
stale-source behavior) that the benchmark does not measure.

## Phase 1 (current): corpus + baseline RAG

- **Corpus**: full EnterpriseRAG-Bench, normalized into a typed schema (`doc_id == dsid`).
- **Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` + **FAISS** (`IndexFlatIP`).
- **Baseline**: naive RAG (retrieve top-k → DeepSeek answer); no reliability logic yet.
- **Heavy compute on Modal** (L4 GPU) for the full-corpus index; local dev uses a small slice.

## Setup

```bash
conda activate llms          # Python 3.11 env with deps from requirements.txt
cp .env.example .env         # add DEEPSEEK_API_KEY (and HF_TOKEN for Modal)
```

## Run (local dev, $0)

```bash
python scripts/build_corpus.py --dev-docs 60 --dev-per-category 4
python scripts/index_corpus.py
python scripts/run_baseline.py --mock --questions dev --emit-bench
```

## Run (full corpus on Modal)

```bash
modal secret create temporalguard-secrets DEEPSEEK_API_KEY=... HF_TOKEN=...
modal volume create temporalguard-data
modal volume put temporalguard-data EnterpriseRAG-Bench/generated_data /bench/generated_data
modal run modal_app.py::index_corpus
```

## Layout

See `docs/01_ARCHITECTURE_AND_FOLDER_STRUCTURE.md`. Build phases in `docs/02..08`.
`CLAUDE.md` is the operating contract.

## Status

Phase 1 verified locally end-to-end. Full-corpus Modal index run is gated on explicit go-ahead.
