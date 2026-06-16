"""Corpus layer.

Default corpus is the full EnterpriseRAG-Bench (see ``bench_import``). The
hand-authored reliability core (``engineering``/``policies``/``distractors``/
``questions``) is kept as **opt-in demo seed** for deterministic demos and tests
of later phases; it is not part of the default build path.
"""
from __future__ import annotations


def load_seed():
    """Return (CORE_DOCUMENTS, QUESTIONS) from the hand-authored demo seed.

    Imported lazily so the default bench path never pays for loading it.
    """
    from temporalguard.corpus import distractors, engineering, policies
    from temporalguard.corpus.questions import QUESTIONS

    core_documents = engineering.DOCUMENTS + policies.DOCUMENTS + distractors.DOCUMENTS
    return core_documents, QUESTIONS
