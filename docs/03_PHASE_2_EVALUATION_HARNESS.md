# Phase 2 — Evaluation Harness

## Goal

Build the evaluation system before implementing advanced reliability logic. The project must prove improvement with numbers, not vibes.

## Time Budget

Target: Day 2.

## Deliverables

- Eval runner
- Metric functions
- Baseline result file
- TemporalGuard result file placeholder
- Comparison report
- JSON/CSV export for dashboard

## Eval Inputs

Use `data/synthetic/eval_questions.jsonl`.

Required fields:

```json
{
  "question_id": "q_deploy_timeout_current",
  "question": "What is the current deployment timeout?",
  "category": "stale_info",
  "expected_decision": "STALE_INFO_RESOLVED",
  "gold_answer": "60 seconds",
  "gold_doc_ids": ["runbook_deploy_2026_04"],
  "conflicting_doc_ids": ["wiki_deploy_2024_01"]
}
```

## Eval Outputs

Store result per question:

```json
{
  "question_id": "q_deploy_timeout_current",
  "pipeline": "baseline",
  "answer": "The deployment timeout is 30 seconds.",
  "decision": "ANSWER",
  "retrieved_doc_ids": ["wiki_deploy_2024_01", "runbook_deploy_2026_04"],
  "latency_ms": 1200,
  "estimated_cost_usd": 0.0008
}
```

## Metrics

### Retrieval Metrics

#### Recall@k

Whether any gold document appears in top-k retrieval results.

```text
Recall@k = questions_with_gold_doc_in_top_k / total_questions
```

#### MRR

Mean reciprocal rank of first gold document.

#### Gold Document Hit Rate

Simpler version of Recall@k for demo reporting.

### Decision Metrics

#### Answerable Accuracy

For questions where expected decision is `ANSWER`, did the system answer correctly?

#### Unanswerable Abstention Recall

For `NOT_FOUND` questions, did the system abstain?

#### False Abstention Rate

For answerable questions, how often did system incorrectly abstain?

#### Conflict Detection Accuracy

For conflict questions, did system return `CONFLICT_DETECTED` or `STALE_INFO_RESOLVED`?

#### Stale Resolution Accuracy

For stale questions, did system choose the current gold answer instead of stale answer?

#### Unsupported Answer Rate

How often did system answer when expected decision was `NOT_FOUND` or `LOW_CONFIDENCE_ABSTAIN`?

### Citation Metrics

#### Citation Precision

Of cited documents, how many are gold/current authoritative docs?

#### Stale Citation Rate

How often did system cite stale/conflicting docs as primary evidence?

### Engineering Metrics

- average latency
- p95 latency
- total API cost
- cost per query
- JSON validity rate

## Aggregate Reliability Score

Implement:

```python
reliability_score = (
    0.25 * answerable_accuracy
    + 0.25 * unanswerable_abstention_recall
    + 0.20 * conflict_detection_accuracy
    + 0.20 * stale_resolution_accuracy
    + 0.10 * citation_precision
)
```

This gives a single resume-friendly metric, while still showing detailed breakdowns.

## Baseline Expected Behavior

Baseline should likely have:

- decent answerable accuracy
- poor unanswerable abstention recall
- poor conflict detection accuracy
- poor stale resolution accuracy
- high unsupported answer rate

This contrast is useful.

## Implementation Tasks

1. Implement `eval/metrics.py`.
2. Implement `eval/run_eval.py`.
3. Implement `eval/baseline.py`.
4. Implement `eval/compare.py`.
5. Save per-question result traces.
6. Export summary JSON for Streamlit.

## Result Files

Recommended files:

```text
cache/eval_runs/
  baseline_run.jsonl
  temporalguard_run.jsonl
  comparison_summary.json
  failure_cases.jsonl
```

## Acceptance Criteria

- `python scripts/run_eval.py --pipeline baseline` runs end-to-end.
- Metrics are computed without manual work.
- Per-question traces are saved.
- Comparison report can compare baseline vs TemporalGuard once TemporalGuard exists.
- Dashboard can read the metric output.

## Common Pitfalls

- Do not rely only on LLM-as-judge for everything.
- Use gold labels for decision-type metrics.
- Keep semantic answer grading simple at first.
- Do not over-optimize metrics before the pipeline works.
