# Phase 5 — Conflict Detection Layer

## Goal

Detect when retrieved documents contain different answers to the same question.

This is the second main reliability feature after abstention.

## Time Budget

Target: Day 5.

## Core Idea

After extracting answer candidates from supporting chunks, TemporalGuard checks whether those candidates are compatible.

Example:

```json
[
  {"answer_candidate": "30 seconds", "source": "old wiki"},
  {"answer_candidate": "60 seconds", "source": "new runbook"}
]
```

These conflict because they answer the same question with different values.

## MVP Conflict Types

Support these first:

1. Numeric/value conflict
   - 30 seconds vs 60 seconds
2. Policy conflict
   - remote allowed 5 days vs 2 days
3. Status conflict
   - feature enabled vs disabled
4. Deadline conflict
   - May 1 vs June 1

Avoid complex legal/scientific contradictions.

## Conflict Detection V1

Use answer candidate grouping.

Steps:

1. Keep judgments with label `SUPPORTS` or `CONTRADICTS`.
2. Extract `answer_candidate` from each.
3. Normalize candidates.
4. Compare candidates pairwise.
5. If incompatible candidates exist, mark conflict.

## Conflict Detection Prompt

```text
You are a contradiction detector for a RAG system.

Question:
{question}

Answer candidates:
{answer_candidates}

Decide whether the candidates are compatible or conflicting.
Return strict JSON:
{
  "conflict": true/false,
  "conflicting_candidates": [..],
  "compatible_groups": [..],
  "reason": "one short sentence"
}

Rules:
- Candidates conflict if they provide different answers to the same question.
- Old/new differences still count as conflict; resolution happens later.
- Do not resolve the conflict in this step.
```

## Conflict Output Schema

```json
{
  "conflict": true,
  "conflicting_candidates": [
    {
      "answer": "30 seconds",
      "chunk_id": "wiki_deploy_2024_chunk_1",
      "doc_id": "wiki_deploy_2024"
    },
    {
      "answer": "60 seconds",
      "chunk_id": "runbook_deploy_2026_chunk_1",
      "doc_id": "runbook_deploy_2026"
    }
  ],
  "reason": "The candidates give different deployment timeout values."
}
```

## Conflict Detection V2 Stretch

Atomic claim extraction:

```text
retrieve chunks → extract atomic claims → compare claims pairwise → build support/contradiction table
```

Do not implement first unless V1 works.

## Implementation Tasks

1. Create `reliability/conflict_detector.py`.
2. Implement answer candidate normalization.
3. Implement LLM-based candidate comparison.
4. Add cheap heuristic for exact numeric mismatch.
5. Save conflict results in trace.
6. Add tests for numeric, policy, and status conflicts.

## Function Interface

```python
def detect_conflicts(question: str, judgments: list[EvidenceJudgment]) -> ConflictResult:
    ...
```

## Heuristics to Add

- If multiple numeric candidates differ, conflict = true.
- If yes/no candidates differ, conflict = true.
- If date candidates differ, conflict = true.
- If all candidates are semantically equivalent, conflict = false.

## Eval Metrics

- conflict detection accuracy
- false conflict rate
- missed conflict rate
- conflict category breakdown

## Dashboard Requirements

Show:

- conflicting answer candidates
- source documents
- source dates
- support confidence
- one-sentence conflict reason

## Acceptance Criteria

- System flags obvious synthetic conflicts.
- System does not flag equivalent paraphrases as conflicts.
- Conflict results feed into temporal/source authority resolver.
- Eval report includes conflict detection accuracy.

## Common Pitfalls

- Do not resolve stale/current answer in this module.
- Do not mark partial evidence as full conflict unless it gives a competing answer.
- Keep top-k small to avoid too many pairwise comparisons.
