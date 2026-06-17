"""Reliability metrics — the TemporalGuard novelty track.

These measure *pre-generation reliability behavior* that the benchmark's answer
grader does not: does the system answer when it should, abstain (NOT_FOUND) when
the answer isn't in the corpus, and flag CONFLICT when sources disagree?

Each eval row needs:
  - category          : clear_answerable | unanswerable | conflicting_info
  - expected_decision : ANSWER | NOT_FOUND | CONFLICT_DETECTED
  - decision          : the system's predicted decision label (see derive_decision)

The naive baseline always ANSWERs (it has no reliability layer), so for it we
derive the decision from the answer text — detecting the rare case where the LLM
spontaneously refuses. TemporalGuard sets `decision` explicitly.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# Decisions that mean "I did not give a substantive answer".
ABSTAIN_DECISIONS = {"NOT_FOUND", "LOW_CONFIDENCE_ABSTAIN"}

# Phrases that indicate the model itself refused / couldn't find it, used to
# derive a decision for the baseline (which has no explicit decision field).
_REFUSAL_PATTERNS = [
    r"\bnot\s+(found|available|provided|present|in the (context|documents?))\b",
    r"\b(cannot|can't|could not|couldn't|unable to)\s+(find|answer|determine|locate)\b",
    r"\bno (information|documents?|context|evidence)\b.*\b(found|available|provided)\b",
    r"\bdoes(n't| not) (contain|mention|provide|include)\b",
    r"\bthere is no\b.*\b(information|mention|reference)\b",
    r"\binsufficient (information|context|evidence)\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def derive_baseline_decision(answer: str) -> str:
    """Map a baseline answer string -> decision. ANSWER unless it reads as a refusal."""
    if not answer or not answer.strip():
        return "NOT_FOUND"
    if _REFUSAL_RE.search(answer):
        return "NOT_FOUND"
    return "ANSWER"


def _is_correct_decision(expected: str, decision: str) -> bool:
    """A decision is 'correct' if it matches expected, treating the two abstain
    labels as interchangeable for unanswerable questions."""
    if expected == decision:
        return True
    if expected == "NOT_FOUND" and decision in ABSTAIN_DECISIONS:
        return True
    return False


def compute_reliability_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the reliability metric suite over scored rows.

    Each row must have `category`, `expected_decision`, and `decision`.
    Returns a dict of metrics in [0,1] plus per-category counts.
    """
    by_cat: Dict[str, List[Dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    def acc(cat: str) -> float | None:
        items = by_cat.get(cat, [])
        if not items:
            return None
        correct = sum(_is_correct_decision(r["expected_decision"], r["decision"]) for r in items)
        return correct / len(items)

    answerable = by_cat.get("clear_answerable", [])
    unanswerable = by_cat.get("unanswerable", [])
    conflicting = by_cat.get("conflicting_info", [])

    # Answerable accuracy: answered (ANSWER) on questions that should be answered.
    answerable_accuracy = acc("clear_answerable")
    # Not-found accuracy: correctly abstained on unanswerable questions.
    not_found_accuracy = acc("unanswerable")
    # Conflict-detection accuracy: correctly flagged CONFLICT on conflicting questions.
    conflict_detection_accuracy = acc("conflicting_info")

    # Unsupported answer rate: gave a substantive ANSWER when it should NOT have
    # (i.e. on unanswerable + conflicting questions). The core harm baseline causes.
    should_not_answer = unanswerable + conflicting
    unsupported = sum(r["decision"] == "ANSWER" for r in should_not_answer)
    unsupported_answer_rate = (unsupported / len(should_not_answer)) if should_not_answer else None

    # False abstention rate: abstained on a question that WAS answerable.
    false_abstentions = sum(r["decision"] in ABSTAIN_DECISIONS for r in answerable)
    false_abstention_rate = (false_abstentions / len(answerable)) if answerable else None

    # Safe decision accuracy: overall fraction with the correct decision label.
    total_correct = sum(_is_correct_decision(r["expected_decision"], r["decision"]) for r in rows)
    safe_decision_accuracy = total_correct / len(rows) if rows else None

    return {
        "n": len(rows),
        "answerable_accuracy": answerable_accuracy,
        "not_found_accuracy": not_found_accuracy,
        "conflict_detection_accuracy": conflict_detection_accuracy,
        "unsupported_answer_rate": unsupported_answer_rate,
        "false_abstention_rate": false_abstention_rate,
        "safe_decision_accuracy": safe_decision_accuracy,
        "counts": {cat: len(items) for cat, items in by_cat.items()},
    }
