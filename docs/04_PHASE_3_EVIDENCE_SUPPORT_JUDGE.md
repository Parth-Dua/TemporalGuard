# Phase 3 — Evidence Support Judge

## Goal

Implement the key reliability primitive: judge whether each retrieved chunk actually supports an answer to the user's question.

This is the first major difference between naive RAG and TemporalGuard.

## Time Budget

Target: Day 3.

## Core Idea

Normal RAG assumes retrieved context is useful. TemporalGuard checks:

> Does this chunk actually answer the question, contradict it, partially help, or say nothing useful?

## Labels

Use four labels:

```text
SUPPORTS
CONTRADICTS
PARTIAL
IRRELEVANT
```

Definitions:

- `SUPPORTS`: chunk directly contains evidence needed to answer.
- `CONTRADICTS`: chunk gives an answer that conflicts with another plausible answer or expected answer.
- `PARTIAL`: chunk is related but insufficient.
- `IRRELEVANT`: chunk does not help answer the question.

## Evidence Judgment Output

Use strict JSON:

```json
{
  "chunk_id": "runbook_deploy_2026_04_chunk_001",
  "label": "SUPPORTS",
  "answer_candidate": "60 seconds",
  "confidence": 0.91,
  "evidence_quote": "Current deployment timeout is 60 seconds.",
  "reasoning_summary": "The chunk directly states the current deployment timeout."
}
```

## API Cost Strategy

Because budget is $5:

1. Judge only top 5 chunks.
2. Cache every judgment.
3. Run on 25 eval queries first.
4. Use a cheap model for structured output.
5. Add a heuristic fallback for obvious cases.

## Suggested Prompt

```text
You are an evidence support classifier for a RAG system.

Question:
{question}

Chunk:
{chunk_text}

Decide whether the chunk SUPPORTS, CONTRADICTS, PARTIAL, or IRRELEVANT for answering the question.

Return strict JSON with:
- label: one of SUPPORTS, CONTRADICTS, PARTIAL, IRRELEVANT
- answer_candidate: short answer contained in the chunk, or null
- confidence: number from 0 to 1
- evidence_quote: exact short quote from the chunk, or null
- reasoning_summary: one short sentence

Rules:
- If the chunk does not directly answer the question, do not mark SUPPORTS.
- If it is only related background, mark PARTIAL or IRRELEVANT.
- Do not invent answer_candidate.
- evidence_quote must come from the chunk.
```

## Heuristic Fallback

For MVP, add simple fallback logic:

- If LLM call fails, mark `PARTIAL` with low confidence.
- If retrieved score is extremely low, mark `IRRELEVANT`.
- If chunk contains exact gold answer in eval mode, mark `SUPPORTS` for debugging only.

Do not use gold labels in production query mode.

## Implementation Tasks

1. Create `reliability/evidence_judge.py`.
2. Create structured output schema.
3. Add LLM cache.
4. Add batch judge function.
5. Save evidence judgments in query trace.
6. Add unit tests for parser and fallback behavior.

## Function Interface

```python
def judge_evidence(question: str, chunk: Chunk) -> EvidenceJudgment:
    ...


def judge_retrieved_chunks(question: str, chunks: list[Chunk]) -> list[EvidenceJudgment]:
    ...
```

## Acceptance Criteria

- Every retrieved chunk gets a support label.
- Output JSON is valid at least 95% of the time.
- Judgments are cached.
- Evidence quote is shown in dashboard.
- Support labels are available to abstention and conflict modules.

## Metrics to Track

- JSON validity rate
- average confidence by label
- support label distribution
- evidence support F1 on synthetic examples, if labels exist
- cost per judged query

## Common Pitfalls

- Do not ask the model to answer the full question here.
- Do not let the model use outside knowledge.
- Do not pass too many chunks at once initially.
- Do not expose long chain-of-thought; store short reasoning summary only.
