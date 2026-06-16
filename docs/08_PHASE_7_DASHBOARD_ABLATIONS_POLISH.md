# Phase 7 — Dashboard, Ablations, Polish, Resume Packaging

## Goal

Turn the working pipeline into a strong end-to-end resume project with a clean demo, metrics, and documentation.

## Time Budget

Target: Day 7.

## Deliverables

- Streamlit dashboard
- baseline vs TemporalGuard comparison
- demo examples
- README
- architecture diagram
- results table
- resume bullets
- optional ablation results

## Dashboard Pages

### Page 1: Query Playground

Inputs:
- question text
- top_k
- pipeline selector: baseline vs TemporalGuard

Outputs:
- baseline answer
- TemporalGuard answer
- decision label
- cited sources
- confidence
- cost estimate
- latency

### Page 2: Evidence Trace

Show:
- retrieved chunks
- retrieval score
- source type
- date
- authority score
- support label
- answer candidate
- evidence quote

### Page 3: Conflict + Staleness Panel

Show:
- conflict detected or not
- answer candidates
- winning candidate
- rejected stale candidates
- score breakdown
- explanation

### Page 4: Eval Dashboard

Show:
- baseline vs TemporalGuard metrics table
- reliability score
- category-wise performance
- unsupported answer rate
- conflict detection accuracy
- stale resolution accuracy

### Page 5: Failure Gallery

Show 5–10 curated examples:

1. baseline hallucinated on not-found query
2. baseline picked stale wiki answer
3. TemporalGuard resolved conflict
4. TemporalGuard correctly abstained
5. TemporalGuard failed and why

Including a failure case is good; it makes the project look honest.

## Ablations

Only do these if core app works.

### Ablation 1: No temporal/source authority scoring

Compare:

- conflict detection only
- conflict detection + temporal resolver

### Ablation 2: Top-k sensitivity

Run top_k = 3, 5, 8.

### Ablation 3: Judge threshold sensitivity

Run support threshold = 0.6, 0.7, 0.8.

### Ablation 4: API judge vs heuristic judge

Compare cost and performance.

## Optional Stretch Features

Not required for one week:

- pretrained NLI contradiction detector
- reranker model
- BM25 + dense hybrid retrieval
- fine-tuned answerability classifier
- EnterpriseRAG-Bench subset evaluation
- React/Next.js frontend
- GitHub Action for RAG regression tests

## README Structure

```markdown
# TemporalGuard RAG

## What it does
## Why normal RAG fails
## Key features
## Architecture
## Dataset
## How to run
## Evaluation
## Results
## Demo screenshots
## Limitations
## Future work
## Resume bullets
```

## Results Table Template

```markdown
| Metric | Baseline RAG | TemporalGuard |
|---|---:|---:|
| Answerable Accuracy |  |  |
| Unanswerable Abstention Recall |  |  |
| Unsupported Answer Rate |  |  |
| Conflict Detection Accuracy |  |  |
| Stale Resolution Accuracy |  |  |
| Citation Precision |  |  |
| Reliability Score |  |  |
```

## Resume Bullets

Use after replacing placeholders with real numbers:

> Built TemporalGuard, a conflict-aware enterprise RAG system that detects unanswerable queries and stale contradictory documents before generation using evidence support judging, abstention logic, temporal scoring, and source authority resolution.

> Improved RAG reliability score from [X] to [Y] on a synthetic enterprise benchmark covering answerable, unanswerable, conflicting, stale, and partial-evidence queries.

> Developed an end-to-end Streamlit/FastAPI evaluation dashboard showing retrieved evidence, support labels, conflict traces, stale-source resolution, latency, estimated cost, and baseline-vs-improved metrics.

## Demo Script

Use three queries:

### Demo 1: Clear Answer

Question:

```text
What is the current P0 incident escalation process?
```

Expected:

```text
ANSWER
```

### Demo 2: Not Found

Question:

```text
What is the company's quantum encryption key rotation policy?
```

Expected:

```text
NOT_FOUND
```

### Demo 3: Stale Conflict

Question:

```text
What is the current deployment timeout?
```

Expected:

```text
STALE_INFO_RESOLVED
```

Dashboard should show old wiki says 30 seconds, newer runbook says 60 seconds.

## Final Acceptance Criteria

- App runs locally with one command.
- Dashboard clearly shows baseline vs TemporalGuard.
- Metrics are computed automatically.
- At least 25 labeled eval questions.
- At least 3 polished demo cases.
- API cost remains under $5.
- README explains the problem, architecture, results, and limitations.

## Common Pitfalls

- Do not spend Day 7 adding new features before the demo is polished.
- Do not overstate results from synthetic data.
- Do not hide limitations.
- Do not claim enterprise production readiness; say production-style architecture.
