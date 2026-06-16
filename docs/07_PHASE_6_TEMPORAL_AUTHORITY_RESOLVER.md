# Phase 6 — Temporal + Source Authority Resolver

## Goal

When documents conflict, choose the most likely current answer using timestamp, source authority, support confidence, and agreement.

This turns conflict detection into useful product behavior.

## Time Budget

Target: Day 6.

## Core Idea

Not all evidence is equal.

A newer runbook should usually beat an old wiki.
An official policy should usually beat a Slack rumor.
Multiple agreeing sources should beat one isolated source.

## Evidence Score

MVP scoring formula:

```text
final_evidence_score =
  0.40 * support_confidence
+ 0.25 * recency_score
+ 0.25 * source_authority_score
+ 0.10 * agreement_score
```

Keep weights configurable.

## Source Authority Defaults

```yaml
official_docs: 1.00
runbook: 0.95
product_doc: 0.90
github: 0.80
jira: 0.80
email: 0.65
slack: 0.60
wiki: 0.50
old_wiki: 0.35
```

## Recency Score

Simple version:

```python
recency_score = normalized_date_between_0_and_1
```

Alternative:

```python
recency_score = exp(-age_days / decay_constant)
```

For synthetic corpus, normalized date is enough.

## Agreement Score

If multiple sources support the same candidate, increase score.

```python
agreement_score = min(num_sources_supporting_candidate / 3, 1.0)
```

## Resolver Output

```json
{
  "resolved": true,
  "decision": "STALE_INFO_RESOLVED",
  "current_answer": "60 seconds",
  "winning_candidate": {
    "answer": "60 seconds",
    "score": 0.89,
    "sources": ["runbook_deploy_2026_04", "slack_deploy_update_2026_03"]
  },
  "rejected_candidates": [
    {
      "answer": "30 seconds",
      "score": 0.48,
      "sources": ["wiki_deploy_2024_01"],
      "reason": "Older and lower authority source."
    }
  ],
  "explanation": "The newer runbook and Slack update agree on 60 seconds, while the older wiki appears stale."
}
```

## LLM Explanation Layer

Use heuristic scoring to choose the winner. Then optionally use a cheap LLM call to write a short explanation.

Prompt:

```text
Given the resolved evidence below, write a concise user-facing explanation.
Do not change the selected answer.
Mention the stale/conflicting source if relevant.

Resolved evidence:
{resolver_output}
```

## Implementation Tasks

1. Create `configs/source_authority.yaml`.
2. Create `reliability/temporal_resolver.py`.
3. Implement recency normalization.
4. Implement source authority loading.
5. Implement agreement grouping.
6. Implement final evidence score.
7. Add explanation generation.
8. Add tests for stale conflict examples.

## Function Interface

```python
def resolve_temporal_conflict(
    question: str,
    conflict_result: ConflictResult,
    judgments: list[EvidenceJudgment],
    chunks: list[Chunk]
) -> TemporalResolution:
    ...
```

## Final Decision Logic

```python
if no_supported_evidence:
    return NOT_FOUND
elif conflict_detected:
    resolution = resolve_temporal_conflict(...)
    if resolution.resolved:
        return STALE_INFO_RESOLVED
    else:
        return CONFLICT_DETECTED
else:
    return ANSWER
```

## Eval Metrics

- stale resolution accuracy
- stale source selection rate
- source authority accuracy
- current answer accuracy
- conflict explanation quality, manually inspected

## Dashboard Requirements

Show:

- winning answer candidate
- rejected stale candidates
- score breakdown
- source dates
- source authority scores
- explanation

## Acceptance Criteria

- For stale examples, system chooses newer/authoritative answer.
- For unresolved conflicts, system flags conflict instead of pretending certainty.
- Score breakdown is visible in dashboard.
- Eval report includes stale resolution accuracy.

## Common Pitfalls

- Do not rely only on recency; a new Slack rumor should not always beat official docs.
- Keep source authority configurable.
- Do not hide conflicts after resolution; show that a conflict existed.
- Avoid overclaiming correctness; say “based on available evidence.”
