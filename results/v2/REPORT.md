# Results — v2 (better retrieval + reranker + structured decisions)

_2026-06-18 · n=70 (30 clear_answerable + 20 unanswerable + 20 conflicting_info)_

v2 = baseline pipeline + three changes: **wide retrieve (50) → cross-encoder rerank → top-5**,
a **better evidence-grounded prompt**, and **structured JSON decisions** (explicit
ANSWER / NOT_FOUND / CONFLICT_DETECTED) instead of regex on free text.

## v2 vs Baseline

| Metric | Baseline | v2 | Δ |
|---|---|---|---|
| Recall@5 | 50.0% | **64.0%** | +14.0 |
| MRR | 0.382 | **0.536** | +0.15 |
| Answerable accuracy | 33.3% | **46.7%** | +13.4 |
| Not-found accuracy | 95.0% | **100.0%** | +5.0 |
| Conflict-detection accuracy | 0.0% | **25.0%** | +25.0 |
| Unsupported-answer rate (lower=better) | 25.0% | **17.5%** | −7.5 |
| False-abstention rate (lower=better) | 66.7% | **43.3%** | −23.4 |
| Safe-decision accuracy | 41.4% | **55.7%** | +14.3 |
| Cost | $0.045 | $0.080 | +$0.035 |
| Median latency | 4.6 s | 4.1 s | — |

**Every reliability metric improved.** Safe-decision accuracy 41% → 56%.

## What we infer

1. **The reranker is the main accuracy lever.** Recall@5 jumped 50%→64% and MRR 0.38→0.54
   purely from reranking 50 FAISS candidates down to 5. Better retrieval directly drove
   answerable accuracy up (33%→47%) and false-abstention down (67%→43%) — the baseline was
   abstaining mostly because it never retrieved the gold doc, and v2 fixed much of that.

2. **Structured output gives conflict detection for free.** With an explicit decision field,
   v2 flags CONFLICT_DETECTED on 5/20 conflicting questions (baseline: 0) — before any
   dedicated conflict layer, just from asking the LLM to decide.

3. **Not-found is solved (100%).** v2 never bluffs on truly-unanswerable questions.

## Where v2 still fails (the next levers)

- **Answerable misses (16/30), split evenly:**
  - **8 retrieval** — gold doc not in top-5 even after rerank → needs better recall (larger
    `retrieve_k`, better chunking + reindex, or hybrid BM25+dense).
  - **6 decision** — gold WAS retrieved but the model still said NOT_FOUND → prompt slightly
    too conservative, or the right passage of the gold doc wasn't the retrieved chunk.
  - **2 ERROR** — the structured call truncated even after the retry (very long contexts) →
    bump `max_tokens` / trim context.
- **Conflict detection (5/20):** the LLM only spontaneously flags a quarter of conflicts; on
  the rest it answers one side (7) or abstains (8). This is the clearest remaining gap and the
  target for a **dedicated conflict-detection layer** (compare candidate answers across the
  top-k chunks rather than relying on the LLM noticing on its own).

## Next steps (proposed)

1. **Cheap wins (no reindex):** bump `max_tokens` to kill the 2 ERRORs; nudge the prompt so a
   clearly-supported answer isn't over-abstained. Re-run on 70.
2. **Conflict layer (v3):** explicit cross-chunk disagreement detection — biggest remaining gap.
3. **Recall ceiling:** if answerable accuracy must climb further, try hybrid retrieval or a
   chunking change (needs one ~2.5h L4 reindex). Decide after the conflict layer.
4. **Validate on the full ~340** once a version is locked, for a credible headline number.

## Artifacts
`metrics.json` (combined metrics) · `metrics_rows.jsonl` (per-question decision + gold_retrieved)
· `outputs.jsonl` (raw, gitignored) · `bench_answers.jsonl` (leaderboard format, gitignored)
