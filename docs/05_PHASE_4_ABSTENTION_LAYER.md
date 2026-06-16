# Phase 4 — Abstention / Not-Found Layer

## Goal

Make TemporalGuard refuse to answer when retrieved evidence does not support an answer.

This directly targets the most common production RAG failure: confident hallucination.

## Time Budget

Target: Day 4.

## Core Idea

After evidence judging, TemporalGuard should answer only if enough support exists.

Decision rule:

```python
if no high_confidence_supporting_evidence:
    return NOT_FOUND or LOW_CONFIDENCE_ABSTAIN
else:
    continue to conflict detection
```

## Decision Types

### NOT_FOUND

Use when retrieved chunks do not contain the answer.

Example response:

```text
I could not find this in the provided company documents. I should not answer without supporting evidence.
```

### LOW_CONFIDENCE_ABSTAIN

Use when some evidence is related but incomplete or weak.

Example response:

```text
I found related information, but not enough direct evidence to answer confidently.
```

## Abstention Policy V1

Simple threshold-based policy:

```python
supporting = [j for j in judgments if j.label == "SUPPORTS" and j.confidence >= 0.70]
partial = [j for j in judgments if j.label == "PARTIAL" and j.confidence >= 0.60]

if len(supporting) == 0 and len(partial) == 0:
    decision = "NOT_FOUND"
elif len(supporting) == 0 and len(partial) > 0:
    decision = "LOW_CONFIDENCE_ABSTAIN"
else:
    decision = "CONTINUE"
```

## Abstention Policy V2

Add retrieval score awareness:

```python
if max_retrieval_score < retrieval_threshold:
    decision = "NOT_FOUND"
elif no_supporting_evidence:
    decision = "NOT_FOUND"
else:
    decision = "CONTINUE"
```

## Abstention Policy V3 Stretch

Optional classifier or fine-tune:

Input:

```json
{
  "question": "...",
  "retrieved_chunks": [...],
  "support_labels": [...]
}
```

Output:

```json
{
  "answerability": "ANSWERABLE" | "UNANSWERABLE" | "PARTIAL",
  "confidence": 0.83
}
```

Not required for week-one MVP.

## Implementation Tasks

1. Create `reliability/abstention.py`.
2. Implement threshold policy.
3. Add config values to `configs/evaluation.yaml`.
4. Add abstention decisions to query trace.
5. Update response composer for not-found responses.
6. Add tests for not-found and low-confidence cases.

## Function Interface

```python
def decide_answerability(judgments: list[EvidenceJudgment], retrieval_scores: list[float]) -> AnswerabilityDecision:
    ...
```

## Eval Metrics

Primary:

- unanswerable abstention recall
- false abstention rate
- unsupported answer rate

Secondary:

- answerable accuracy preserved
- reliability score

## Dashboard Requirements

Show:

- decision label
- why system abstained
- top retrieved chunks
- support labels
- confidence scores

## Acceptance Criteria

- TemporalGuard returns `NOT_FOUND` on synthetic unanswerable questions.
- Baseline answers those same questions incorrectly or unsupported.
- Eval report shows unsupported answer rate decrease.
- Dashboard clearly shows evidence was insufficient.

## Common Pitfalls

- Do not abstain too aggressively on answerable questions.
- Keep thresholds configurable.
- Do not hide retrieved chunks; show why the system refused.
- Do not claim perfect hallucination prevention.
