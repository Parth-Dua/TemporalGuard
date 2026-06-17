# Baseline Results — Cheap RAG (no reliability layer)

Run: 70-question eval subset (30 clear_answerable + 20 unanswerable + 20 conflicting_info)
against the full EnterpriseRAG-Bench corpus indexed on Modal.

- **Corpus index:** 3,769,662 chunks (≈512k HF docs, MiniLM 384-dim, FAISS IndexFlatIP)
- **Retrieval:** top_k = 5, `sentence-transformers/all-MiniLM-L6-v2`
- **Generator:** DeepSeek `deepseek-v4-flash` (OpenAI-compatible), naive prompt, always answers
- **Cost:** $0.045 total · **Median latency:** ~4.6 s/q · **Avg answer length:** ~166 chars

## Headline reliability metrics (baseline)

| Metric | Value | Meaning |
|---|---|---|
| Answerable accuracy | **33.3%** | correct ANSWER on the 30 answerable questions |
| Not-found accuracy | **95.0%** | correctly abstained on the 20 unanswerable |
| Conflict-detection accuracy | **0.0%** | correctly flagged CONFLICT on the 20 conflicting |
| Unsupported-answer rate | **25.0%** | answered when it shouldn't have *(lower better)* |
| False-abstention rate | **66.7%** | abstained on an answerable question *(lower better)* |
| Safe-decision accuracy | **41.4%** | overall correct decision label |

## The decisive finding: retrieval, not the LLM, is the bottleneck

**Recall@5 (gold doc retrieved) = 50%** — `clear_answerable` 15/30, `conflicting_info` 10/20.

Cheap MiniLM retrieval over 3.77M chunks surfaces the gold document only half the time.
When it misses, DeepSeek **honestly says "the provided context does not contain…"** rather
than hallucinating. So:

- The low **answerable accuracy (33%)** and high **false-abstention (67%)** are *retrieval
  failures*, not decision failures — the model abstained because the evidence wasn't retrieved.
- The high **not-found accuracy (95%)** is partly the model's honest hedging, not a deliberate
  reliability decision.

### Decision distribution (derived from answer text)

| Category | ANSWER | NOT_FOUND (hedged) |
|---|---|---|
| clear_answerable (30) | 10 | 20 |
| unanswerable (20) | 1 | 19 |
| conflicting_info (20) | 9 | 11 |

## What the baseline cannot do (the gap TemporalGuard fills)

1. **Conflict detection = 0%.** On all 20 conflicting questions the baseline either answered
   one side or hedged — it **never flags that sources disagree**. Example:
   > Q: *"What is the API endpoint to start a Unified Capacity Transition migration?"*
   > A: *"The provided context does not specify an API endpoint… None of the documents include such a…"*
   (No awareness that retrieved chunks contradict each other.)

2. **Abstention is accidental, not calibrated.** The baseline's "not found" behavior is the
   LLM spontaneously hedging, with no explicit decision, confidence, or trace. It can't tell
   "I retrieved nothing relevant" from "I retrieved conflicting things."

3. **Unsupported answers still happen (25%).** On 1 unanswerable + 9 conflicting questions it
   produced a substantive answer it shouldn't have.

## Examples

**Answered correctly (gold retrieved):**
> Q: *"What are the default size limits for file uploads and total request size…?"*
> A: *"The default size limits are 10 MiB per file and 50 MiB per request, as documented…"* ✓

**Correctly hedged on unanswerable:**
> Q: *"…which specific enterprise accounts were on the initial allowlist…?"*
> A: *"The provided context does not contain a list of specific enterprise accounts…"* ✓

## Open questions for brainstorming

1. **Recall ceiling.** 50% recall@5 caps answer quality. Options: bump top_k (15–20), better
   chunking + reindex, or a hybrid/reranker. Should we lift recall before layering reliability,
   or report it as a fixed baseline limitation and focus the contrast on conflict + abstention?
2. **Framing of the win.** Given DeepSeek already hedges honestly, TemporalGuard's clearest
   wins are (a) **conflict detection 0% → high**, and (b) turning messy hedging into **explicit,
   labeled, traced decisions** (NOT_FOUND vs CONFLICT vs LOW_CONFIDENCE). Is that the headline?
3. **Decision derivation.** Baseline decisions are regex-derived from answer text (it has no
   decision field). It correctly catches real refusals; do we keep it, or report baseline as
   literally always-ANSWER for a starker contrast?
4. **Metric weighting.** Should Safe-Decision-Accuracy down-weight retrieval-caused failures so
   it measures *decision* quality, not *retrieval* quality?

## Artifacts
- `cache/eval_runs/baseline_outputs.jsonl` — full per-question outputs (answer, retrieved chunks, cost, latency)
- `cache/eval_runs/baseline_bench_answers.jsonl` — bench answer_evaluation format (leaderboard track)
- `cache/eval_runs/reliability_report.json` — the metrics above
