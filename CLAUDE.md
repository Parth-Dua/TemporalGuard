# CLAUDE.md — TemporalGuard RAG Build Instructions

## Read First
Before coding, read these docs in order:

1. `docs/00_PRD.md` — product scope, success criteria, constraints.
2. `docs/01_ARCHITECTURE_AND_FOLDER_STRUCTURE.md` — target architecture, module boundaries, folder structure.
3. Current phase doc:
   - `docs/02_PHASE_1_DATASET_BASELINE_RAG.md`
   - `docs/03_PHASE_2_EVALUATION_HARNESS.md`
   - `docs/04_PHASE_3_EVIDENCE_SUPPORT_JUDGE.md`
   - `docs/05_PHASE_4_ABSTENTION_LAYER.md`
   - `docs/06_PHASE_5_CONFLICT_DETECTION.md`
   - `docs/07_PHASE_6_TEMPORAL_AUTHORITY_RESOLVER.md`
   - `docs/08_PHASE_7_DASHBOARD_ABLATIONS_POLISH.md`

Do not duplicate PRD content here. Treat this file as the operating contract for Claude Code.

## Mission
Build **TemporalGuard RAG**: an end-to-end resume project that adds a **reliability layer**
to a cheap baseline RAG. The system decides whether retrieved evidence is sufficient,
missing, or contradictory before answering — choosing to answer, abstain, or flag a conflict.

Main differentiator: **pre-generation evidence arbitration** (conflict detection + abstention),
not post-hoc RAG monitoring.

### Scope (pivot)
The project compares **Baseline cheap RAG** vs **Baseline cheap RAG + TemporalGuard reliability
layer** on a subset of EnterpriseRAG-Bench. We are NOT competing with the benchmark's agentic
retrieval at scale; we show that a lightweight reliability layer makes a *cheap* RAG safer
(fewer unsupported answers, correct abstention on not-found, conflict flagging).
Focus = **conflict + not-found + abstention**. Temporal/staleness is de-scoped for now
(the HuggingFace `documents` config carries no per-doc dates/status); the
`STALE_INFO_RESOLVED` label remains reserved but is not a Phase-1–7 deliverable.

## Hard Constraints
- One-week MVP.
- Total API spend cap: **$5**.
- **Corpus = the full EnterpriseRAG-Bench, loaded directly from HuggingFace**
  (`onyx-dot-app/EnterpriseRAG-Bench`, `documents` config: `doc_id`/`source_type`/`title`/
  `content`). No local files, no uploads — Modal pulls it from HF and caches on the Volume.
  We index the whole corpus so every question's gold docs are present by construction.
- **Eval subset = 70 questions**: 30 clear_answerable (basic/semantic) + 20 unanswerable
  (info_not_found) + 20 conflicting_info, selected deterministically (first-N per category).
- Streamlit dashboard for week one; React/Next.js is out of scope.
- Fine-tuning, NLI models, and rerankers are optional stretch work, not blockers.
- Cache all API calls.
- Prefer local embeddings and deterministic heuristics before paid LLM calls.
- **Heavy compute (full-corpus embedding/indexing) runs on Modal (L4 GPU)**; lightweight
  steps run locally. Never embed 512k docs on a laptop.
- Never build a generic chat-with-docs app; every feature must support reliability evaluation.

## Default Stack
- Python (run in the `llms` conda env, Python 3.11)
- Streamlit
- FastAPI-compatible backend modules, but no overbuilt API before pipeline works
- **FAISS** everywhere (single vector DB; `IndexFlatIP` over normalized embeddings)
- `sentence-transformers/all-MiniLM-L6-v2` local embeddings
- `langchain-text-splitters` (`RecursiveCharacterTextSplitter`) for chunking
- **Modal** for heavy compute: Volume (corpus + index + outputs), Secret (keys), L4 GPU
- **DeepSeek** (`deepseek-v4-flash` / `deepseek-v4-pro`, OpenAI-compatible) for baseline/judging/generation; cached by hash
- JSONL data files, YAML configs

## Differentiator vs. the benchmark
EnterpriseRAG-Bench's `answer_evaluation/` is a RAG-system grader (correctness /
completeness / document-recall) with a public leaderboard. It scores whether the **final
answer** is right — it does **not** measure **pre-generation reliability behavior**: does
the system *abstain* instead of bluffing, *flag which doc_ids conflict*, *pick current over
stale*. TemporalGuard adds (a) a system architected for that behavior, and (b) a
**reliability-evaluation track** (false-abstention rate, unsupported-answer rate,
conflict-detection accuracy, stale-source selection rate, aggregate Reliability Score)
layered on the same benchmark. Two eval tracks:
1. **Leaderboard track** — run THEIR scripts unmodified from the bench repo; we only read
   their results JSON to render in our dashboard (numbers stay leaderboard-comparable).
2. **Reliability track (our novelty)** — `sys.path` to the bench repo and import their metric
   functions (no re-implementation), then layer our reliability metrics beside them.

## Build Order
Follow the phase docs exactly:

1. Dataset + baseline RAG
2. Evaluation harness
3. Evidence support judge
4. Abstention / not-found layer
5. Conflict detection layer
6. Temporal + source authority resolver
7. Dashboard, ablations, polish, README/resume packaging

Do not skip the eval harness. The project is resume-worthy only if baseline vs TemporalGuard metrics are visible.

## Required Decision Labels
TemporalGuard must output one of:

```text
ANSWER
NOT_FOUND
LOW_CONFIDENCE_ABSTAIN
CONFLICT_DETECTED
STALE_INFO_RESOLVED
```

## Core Pipeline
Implement this flow:

```text
question
→ retrieve top-k chunks
→ judge evidence support per chunk
→ extract answer candidates
→ abstain if no sufficient support
→ detect candidate conflicts
→ resolve using recency + source authority + agreement
→ compose final answer with citations and trace
→ log metrics/cost/latency
```

## Engineering Rules
- Keep modules small and testable.
- Use typed dataclasses or Pydantic models for documents, chunks, judgments, decisions, and eval results.
- Every LLM call must return structured JSON and be cached by input hash.
- Save intermediate traces for every query: retrieved chunks, judgments, candidates, final decision, final answer.
- Prefer simple working heuristics over complex unfinished research implementations.
- Add tests for abstention, conflict detection, temporal resolution, and metrics.
- Keep prompts in `configs/prompts.yaml`, not scattered through the code.
- Keep source authority values in config (`configs/source_authority.yaml`), not hardcoded inside logic.
- Run everything in the `llms` conda env; on Modal, source is shipped via `add_local_dir("src")`.

## Cost Discipline
Use paid APIs only after retrieval has narrowed the problem.

Recommended defaults:
- Local embeddings for all indexing/retrieval.
- Cache all judge/generation calls.
- Run expensive judging only on top 5 chunks.
- Use small/cheap models for structured judging.
- Add a dry-run/mock mode for tests.

## Metrics That Must Exist
Two tracks (see Differentiator):

**Leaderboard track (reuse THEIR functions, do not re-implement):**
- correctness, completeness, document recall, invalid-extra-documents

**Reliability track (our novelty) — Baseline vs Baseline+TemporalGuard:**
- Not-found accuracy (correct NOT_FOUND on info_not_found questions)
- Conflict detection accuracy (correct CONFLICT_DETECTED on conflicting_info questions)
- Unsupported answer rate (answered when it shouldn't have)
- False abstention rate (abstained when it should have answered)
- Safe decision accuracy (aggregate: right decision label across all categories)
- Answerable accuracy (correct ANSWER on basic/semantic)
- Retrieval Recall@k, MRR (and reuse their document-recall)
- Latency, estimated cost

## Dashboard Must Show
Streamlit MVP should include:

1. Query playground
2. Retrieved evidence table
3. Evidence support labels
4. Conflict/staleness panel
5. Baseline vs TemporalGuard comparison
6. Eval metrics table/charts
7. Failure examples gallery

## Definition of Done
The project is complete when:

- A user can run the app locally from the README.
- Synthetic corpus + eval questions are generated or loaded.
- Baseline RAG answers all/most questions.
- TemporalGuard can answer, abstain, detect conflict, and resolve stale info.
- Dashboard shows baseline vs TemporalGuard metrics.
- At least 3 polished demo examples exist:
  1. normal answer,
  2. not found / abstention,
  3. stale conflict resolved.
- README contains architecture, setup, screenshots, metrics, limitations, and resume bullets.

## Do Not Do
- Do not spend week one on fine-tuning.
- Do not implement full multi-agent debate.
- Do not train custom rerankers.
- Do not embed the full 512k corpus locally — that pass belongs on Modal (L4 GPU).
- Do not re-implement the benchmark's leaderboard metrics — import their functions.
- Do not hide failures; the dashboard should make failures inspectable.
- Do not add broad product features unrelated to RAG reliability.

## Resume Target
Final project claim:

> Built **TemporalGuard RAG**, a conflict-aware enterprise RAG system that detects missing, contradictory, and stale evidence before generation, reducing unsupported answers versus baseline RAG using evidence support judging, abstention, temporal resolution, and a reliability eval dashboard.
