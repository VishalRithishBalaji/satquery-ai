"""Query-sensitivity guard.

Directly implements the technical note's recommendation: "If the response
does not change meaningfully for materially different queries, flag low
query sensitivity and escalate." This keeps a short, in-memory history of
recent (query signature, answer) pairs per image set. When a new,
materially different query produces an answer nearly identical to a prior
answer for that same imagery, it is flagged so the caller can retry with a
corrective prompt and/or discount confidence.

"Materially different" is judged from the query's extracted attributes and
key terms (see query_analysis.py) rather than raw sentence text: short
queries that share a lot of boilerplate phrasing ("Identify X." vs.
"Identify Y.") can look deceptively similar under plain character-level
diffing even when they ask about completely different things, which would
make the guard blind to the exact failure mode it exists to catch.

The cache is process-local and intentionally small (a prototype guard,
not a persistence layer) - it resets on backend restart.
"""

from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock

from .query_analysis import QueryIntent

_MAX_HISTORY = 5
ANSWER_SIMILARITY_THRESHOLD = 0.82
# Jaccard overlap of (attributes | key_terms) above which two queries are
# considered "the same ask" - a similar answer to those is expected, not a bug.
QUERY_OVERLAP_CEILING = 0.5

_cache: dict[str, list[tuple[frozenset, str, str]]] = {}
_lock = Lock()


def image_set_key(paths: list[str]) -> str:
    """Content-derived key so re-uploads of the same imagery under a new
    filename still land on the same sensitivity history."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        try:
            data = Path(path).read_bytes()
        except OSError:
            data = path.encode("utf-8", errors="ignore")
        digest.update(data[:1_000_000])
        digest.update(str(len(data)).encode())
    return digest.hexdigest()


def _signature(intent: QueryIntent) -> frozenset:
    return frozenset([*intent.attributes, *intent.key_terms])


def _query_overlap(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0  # neither query carried an extractable signal; treat as indistinguishable
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _answer_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def check_sensitivity(image_key: str, intent: QueryIntent, answer: str) -> dict:
    with _lock:
        history = list(_cache.get(image_key, []))

    signature = _signature(intent)
    best_answer_similarity = 0.0
    most_similar_answer: str | None = None

    for previous_signature, _previous_query, previous_answer in history:
        if _query_overlap(signature, previous_signature) >= QUERY_OVERLAP_CEILING:
            continue  # queries were already near-duplicates; a similar answer is expected, not a bug
        similarity = _answer_similarity(answer.lower(), previous_answer.lower())
        if similarity > best_answer_similarity:
            best_answer_similarity = similarity
            most_similar_answer = previous_answer

    return {
        "low_sensitivity": best_answer_similarity >= ANSWER_SIMILARITY_THRESHOLD,
        "answer_similarity": round(best_answer_similarity, 3),
        "most_similar_answer": most_similar_answer,
        "history_size": len(history),
    }


def remember(image_key: str, intent: QueryIntent, query: str, answer: str) -> None:
    with _lock:
        history = _cache.setdefault(image_key, [])
        history.append((_signature(intent), query, answer))
        del history[:-_MAX_HISTORY]


def reset() -> None:
    """Test helper - clears all cached history."""
    with _lock:
        _cache.clear()
