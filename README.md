# TemporalGuard RAG

A **pre-generation reliability layer** for RAG. Before answering, TemporalGuard decides
whether retrieved evidence is sufficient, missing, or contradictory — emitting one of:
`ANSWER` · `NOT_FOUND` · `LOW_CONFIDENCE_ABSTAIN` · `CONFLICT_DETECTED`.
(`STALE_INFO_RESOLVED` is reserved for future work.)

**What it shows:** `Baseline cheap RAG` vs `Baseline cheap RAG + TemporalGuard`. The goal
isn't to beat the benchmark's agentic retrieval at scale — it's to show a lightweight
reliability layer makes a *cheap* RAG **safer**: fewer unsupported answers, correct
abstention when the answer isn't in the corpus, and explicit flagging of contradictory sources.

Built on [EnterpriseRAG-Bench](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench)
(~512k enterprise docs), loaded directly from HuggingFace.

## Evaluation

- **Subset:** 70 questions — 30 answerable (basic/semantic) + 20 info-not-found + 20 conflicting.
- **Retrieval metrics:** reuse the benchmark's document-recall etc. (their functions, not re-implemented).
- **Reliability metrics (ours):** not-found accuracy, conflict-detection accuracy,
  unsupported-answer rate, false-abstention rate, safe-decision accuracy.
- **Dashboard:** baseline vs TemporalGuard across these metrics, with per-query evidence traces.

## Stack

- Corpus: EnterpriseRAG-Bench `documents` config from HuggingFace (`doc_id`/`source_type`/`title`/`content`)
- Retrieval: `sentence-transformers/all-MiniLM-L6-v2` + **FAISS** (`IndexFlatIP`)
- Chunking: `langchain-text-splitters` `RecursiveCharacterTextSplitter`
- LLM: **DeepSeek** (`deepseek-v4-flash`), OpenAI-compatible, cached by request hash
- Heavy compute: **Modal** (L4 GPU) — full-corpus embed/index, HF cache + FAISS index on a Volume

## Setup

```bash
conda activate llms          # Python 3.11 env; deps in requirements.txt
cp .env.example .env         # DEEPSEEK_API_KEY, HF_TOKEN
modal secret create temporalguard-secrets DEEPSEEK_API_KEY=... HF_TOKEN=...
```

## Run

```bash
# 1. Build the 70-question eval subset (downloads only the small questions config)
python scripts/build_corpus.py

# 2. Index the full corpus on Modal L4 (downloads docs from HF, embeds on GPU -> FAISS on Volume)
modal run modal_app.py::index_corpus

# 3. Baseline RAG over the subset (later: + TemporalGuard, eval, dashboard)
```

## Layout

See `docs/01_ARCHITECTURE_AND_FOLDER_STRUCTURE.md`. Phase docs in `docs/02..08`.
`CLAUDE.md` is the operating contract.
