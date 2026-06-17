# TemporalGuard RAG — Product Requirements Document

## 1. Project Summary

**TemporalGuard RAG** is an end-to-end AI engineering resume project that builds a reliability layer on top of Retrieval-Augmented Generation (RAG). Instead of always answering, TemporalGuard decides whether retrieved evidence is sufficient, missing, contradictory, or stale before generating a final response.

The project is designed to be completed in about one week with a total API budget of approximately **$5**. The first version uses a small synthetic enterprise-style corpus, Streamlit dashboard, local embeddings where possible, and selective API calls for judgment/generation.

## 2. One-Line Differentiator

TemporalGuard performs **pre-generation evidence arbitration**: it detects missing, contradictory, and stale evidence before answering, instead of only monitoring RAG outputs after generation.

## 3. Target User

Primary audience:

- AI engineering recruiters and hiring managers
- Startup engineers evaluating production RAG reliability
- Developers building internal company-brain systems

Demo user:

- A knowledge worker asking questions over messy internal company docs such as Slack updates, runbooks, Jira tickets, old wiki pages, and product docs.

## 4. Problem

Normal RAG systems often fail silently when:

1. The answer is not in the documents.
2. Retrieved documents contradict each other.
3. Older documents rank higher than newer authoritative documents.
4. The model uses weakly related context and confidently hallucinates.

This is one of the main reasons companies do not trust RAG systems in production.

## 5. Product Goal

Build a working RAG app that can classify a query into one of these final decisions:

- `ANSWER`: enough clear evidence exists.
- `NOT_FOUND`: no retrieved evidence supports an answer.
- `CONFLICT_DETECTED`: multiple retrieved sources disagree.
- `STALE_INFO_RESOLVED`: old evidence conflicts with newer/more authoritative evidence; answer with warning.
- `LOW_CONFIDENCE_ABSTAIN`: evidence is partial or weak.

## 6. Non-Goals

For the one-week MVP, do **not** build:

- A full enterprise search product.
- A generic chat-with-PDF app.
- Full agentic workflows.
- Large-scale fine-tuning.
- Complex graph reasoning across the entire corpus.
- Production auth/multi-tenant deployment.
- Large paid API evaluation runs.

## 7. Constraints

- Timeline: approximately 7 days.
- Budget: maximum **$5 total API spend**.
- Frontend: Streamlit first; React/Next.js later if time permits.
- Dataset: mini synthetic enterprise corpus first; optional EnterpriseRAG-Bench subset later.
- LLM usage: use Claude/OpenAI APIs sparingly for structured judging and final answer generation.
- Fine-tuning: optional stretch, not part of MVP.
- Pretrained NLI/reranker models: optional stretch, not part of MVP.

## 8. MVP Scope

### Core MVP Features

1. Mini synthetic enterprise corpus generator or static corpus.
2. Baseline RAG pipeline.
3. Retrieval with local embeddings and vector store.
4. Evidence support judge.
5. Abstention layer.
6. Conflict detection layer.
7. Temporal/source-authority resolver.
8. Streamlit dashboard.
9. Evaluation harness comparing baseline RAG vs TemporalGuard.
10. Resume-ready metrics and examples.

### Demo Query Types

The mini corpus should include at least five query categories:

1. Clear answerable questions.
2. Unanswerable questions.
3. Conflicting information questions.
4. Stale information questions.
5. Partial evidence / low-confidence questions.

## 9. Dataset Strategy

### Primary Dataset — full EnterpriseRAG-Bench

The corpus is the **full EnterpriseRAG-Bench** (~512k real enterprise docs across
Slack, Gmail, Linear, Google Drive, Hubspot, Fireflies, GitHub, Jira, Confluence).
We index the **entire corpus** — no subset selection — so every question's gold docs
are present by construction (this removes the "orphaned question" problem of sampling).

Each bench doc is normalized into our schema, preserving the signals later phases need:

```json
{
  "doc_id": "dsid_ae068ee4...",            // == bench dsid (maps back for the leaderboard)
  "title": "...",
  "source_type": "github",                  // derived from the bench source dir
  "created_at": "2026-02-18",
  "updated_at": "2026-03-02",               // recency discriminator for stale/conflict
  "authority_score": 0.55,                  // default-by-source; refined later
  "status": "merged",
  "text": "...",
  "metadata": {"bench_source": "github", "bench_path": "github/..."}
}
```

Questions: we map the bench's labeled question types to our reliability categories —
`basic`/`semantic` → clear_answerable (ANSWER), `conflicting_info` → CONFLICT_DETECTED,
`info_not_found` → unanswerable (NOT_FOUND). ~340 of the 500 questions map cleanly.

### Optional demo seed

A small hand-authored synthetic corpus is kept as **opt-in seed** (`--include-seed`) for
deterministic demos / unit tests of later phases. It is not part of the default build path.

## 10. System Decisions

### LLM Provider Strategy

Use APIs only for: evidence judging, conflict detection, final answer generation.

- **DeepSeek `deepseek-v4-flash`** (OpenAI-compatible endpoint) is the default; `deepseek-v4-pro` for harder calls. Cheap and cached.
- Cache every API response to disk by hash of (model + messages + params).
- Mock mode (`--mock`) for $0 dry runs / tests.
- Estimated cost: even the full ~340-question pipeline + leaderboard judging is ≈ $1–2,
  well under the $5 cap. The real spend is **one-time Modal GPU time** for indexing.

### Retrieval Strategy

- local embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- vector store: **FAISS** (`IndexFlatIP` over L2-normalized vectors = cosine), single backend
- chunking: `RecursiveCharacterTextSplitter` (chunk_size 900, overlap 120)
- **full-corpus indexing runs on Modal (L4 GPU)** with a Volume holding corpus + index + outputs

Stretch: BM25 / hybrid retrieval, reranking, temporal reranking.

### Differentiator vs. the benchmark (novelty)

EnterpriseRAG-Bench ships a RAG-system grader (`answer_evaluation/`: correctness /
completeness / document-recall) + a public leaderboard. It scores whether the **final
answer** is correct — including on conflict and absent-info questions — but does **not**
measure **pre-generation reliability behavior**: abstaining instead of bluffing, flagging
*which* doc_ids conflict, picking current over stale. TemporalGuard contributes (a) a system
built for that behavior and (b) a **reliability-evaluation track** layered on the same
benchmark. Two tracks: (1) **leaderboard** — run their scripts unmodified and submit;
(2) **reliability** — import their metric functions (no re-implementation) and add ours.

## 11. Core Pipeline

```text
User Query
  ↓
Retrieve top-k chunks
  ↓
Evidence Support Judge
  ↓
Extract answer candidates
  ↓
Abstention decision
  ↓
Conflict detection
  ↓
Temporal + source authority resolver
  ↓
Final answer / not found / conflict warning
  ↓
Trace + metrics saved
```

## 12. Evaluation Metrics

### Retrieval Metrics

- Recall@k
- MRR
- gold document hit rate

### Reliability Metrics

- answerable accuracy
- unanswerable abstention recall
- false abstention rate
- unsupported answer rate
- conflict detection accuracy
- stale-source selection rate
- citation precision
- evidence support F1

### Engineering Metrics

- latency
- estimated token cost
- API call count
- JSON validity rate

### Aggregate Metric

Create a custom metric:

```text
Safe Decision Accuracy (aggregate reliability score) =
0.30 * answerable_accuracy            # correct ANSWER on basic/semantic
+ 0.30 * not_found_accuracy           # correct NOT_FOUND on info_not_found
+ 0.30 * conflict_detection_accuracy  # correct CONFLICT_DETECTED on conflicting_info
+ 0.10 * (1 - unsupported_answer_rate)
```
(Staleness/temporal resolution is reserved for future work — the HF `documents`
config carries no per-doc dates, so it is not part of the current metric set.)

## 13. Dashboard Requirements

Streamlit dashboard tabs:

1. **Query Playground**
  - user query input
  - baseline answer
  - TemporalGuard answer
  - final decision label
2. **Evidence Trace**
  - retrieved chunks
  - support labels
  - answer candidates
  - confidence scores
  - source metadata
3. **Conflict + Staleness Panel**
  - current answer
  - stale/conflicting answer
  - trusted sources
  - rejected sources
  - explanation
4. **Eval Dashboard**
  - baseline vs TemporalGuard metrics
  - reliability score
  - confusion matrix for decision types
5. **Failure Gallery**
  - examples where baseline hallucinated
  - examples where TemporalGuard abstained correctly
  - examples where stale conflict was resolved

## 14. Success Criteria

Minimum successful project:

- Runs locally end-to-end.
- Has a mini synthetic enterprise corpus.
- Has at least 25 labeled eval queries.
- Baseline RAG answers everything.
- TemporalGuard abstains on missing info.
- TemporalGuard flags at least some conflicts.
- Dashboard shows traces and metrics.
- README includes architecture, results, limitations, and resume bullets.

Strong resume version:

- 50+ eval queries.
- Baseline vs TemporalGuard comparison table.
- Clear reduction in unsupported/confident hallucinations.
- Temporal/source authority resolver works on stale examples.
- Dashboard has 3 polished demos.
- Optional EnterpriseRAG-Bench subset evaluation.

## 15. Resume Positioning Example (Not fixed) - Main takeway is that lots of metrics are required. 

Main bullet:

> Built TemporalGuard, a conflict-aware enterprise RAG system that detects unanswerable queries and stale contradictory documents before generation, reducing unsupported answers using evidence support judging, temporal reranking, and source authority scoring.

Metric-based bullet template:

> Improved RAG reliability score from [baseline] to [TemporalGuard] on a synthetic enterprise benchmark with answerable, unanswerable, conflicting, and stale-document queries.

Technical bullet:

> Implemented retrieval evaluation, evidence-support classification, abstention logic, conflict detection, temporal/source-authority resolution, and a Streamlit dashboard for trace-level RAG debugging.

