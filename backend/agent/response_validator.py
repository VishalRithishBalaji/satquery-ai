"""Checks whether a generated answer actually engaged with the query.

This is the "query-to-evidence grounding" gap from the technical note:
the system needs a way to notice when a model has fallen back to a
generic scene description instead of addressing the specific question,
so that case can be flagged (and retried) rather than silently returned.
"""

from __future__ import annotations

from .query_analysis import QueryIntent

GENERIC_PHRASES = (
    "this image shows",
    "the image depicts",
    "general overview of the scene",
    "overall, the image",
)


def grounding_score(answer: str, intent: QueryIntent) -> float:
    """Fraction of the query's attributes/key terms echoed in the answer,
    with a small penalty when the answer reads like a generic caption
    and barely touches the requested terms. Returns a value in [0, 1]."""
    if not answer or not answer.strip():
        return 0.0

    text = answer.lower()
    signals = list(dict.fromkeys([*intent.attributes, *intent.key_terms]))

    if not signals:
        return 0.6

    hits = 0
    for term in signals:
        needle = term.replace("_", " ")
        if needle in text or term.replace("_", "-") in text:
            hits += 1

    coverage = hits / len(signals)
    is_generic = any(phrase in text for phrase in GENERIC_PHRASES)
    penalty = 0.2 if is_generic and coverage < 0.34 else 0.0

    return max(0.0, min(1.0, coverage - penalty))


def is_well_grounded(answer: str, intent: QueryIntent, threshold: float = 0.25) -> bool:
    return grounding_score(answer, intent) >= threshold
