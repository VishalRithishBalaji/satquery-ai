"""Structured query understanding.

Turns a raw natural-language query into a task label (via the existing
router logic) plus a richer intent: which remote-sensing attributes were
asked about and what operation the user wants performed. Tools use this
to build task-specific prompts and to check whether their eventual answer
actually engaged with the question, instead of one generic prompt/answer
for every query on a given image pair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .planner import infer_task

ATTRIBUTE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "built_up": ("built-up", "built up", "urban", "buildings", "settlement", "construction", "infrastructure", "houses"),
    "water": ("water", "river", "lake", "flood", "wetland", "coast", "coastal", "reservoir", "pond"),
    "vegetation": ("vegetation", "forest", "tree", "green cover", "crop", "agriculture", "agricultural", "farmland", "cropland"),
    "roads": ("road", "highway", "transport network", "railway", "track"),
    "bare_soil": ("bare soil", "barren", "sand", "desert"),
    "cloud": ("cloud", "haze"),
    "ship": ("ship", "vessel", "boat"),
    "damage": ("damage", "disaster", "destroyed", "collapsed", "debris"),
    "change": ("change", "changed", "difference", "increased", "decreased", "growth", "shrunk"),
}

OPERATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "locate": ("where", "locate", "highlight", "show me", "bounding box", "point out", "draw a box"),
    "compare": ("compare", "change", "before", "after", "between", "difference", "increased", "decreased", "versus"),
    "quantify": ("how many", "count", "number of", "area of", "percentage", "fraction", "how much"),
    "confirm": ("is there", "are there", "does it", "any signs", "presence of", "can you confirm"),
    "describe": ("describe", "summarize", "what is visible", "land cover", "overview", "scene"),
}

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "of", "in", "on", "and", "or", "to", "for", "with", "this", "that",
    "these", "those", "image", "images", "please", "what", "how", "which", "does", "any", "there",
    "you", "your", "can", "using", "use", "provide", "give", "explain",
}


@dataclass
class QueryIntent:
    task: str
    operation: str
    attributes: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"task": self.task, "operation": self.operation, "attributes": list(self.attributes), "key_terms": list(self.key_terms)}


def _match_any(text: str, keyword_map: dict[str, tuple[str, ...]]) -> list[str]:
    return [label for label, keywords in keyword_map.items() if any(kw in text for kw in keywords)]


def infer_operation(query_lower: str) -> str:
    for operation, keywords in OPERATION_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return operation
    return "describe"


def extract_key_terms(query_lower: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[a-z0-9][a-z0-9\-]*", query_lower)
    seen: list[str] = []
    for word in words:
        if word in _STOP_WORDS or len(word) <= 2 or word in seen:
            continue
        seen.append(word)
        if len(seen) >= limit:
            break
    return seen


def analyze_query(query: str, image_count: int) -> QueryIntent:
    query_lower = (query or "").lower().strip()
    return QueryIntent(
        task=infer_task(query, image_count),
        operation=infer_operation(query_lower),
        attributes=_match_any(query_lower, ATTRIBUTE_KEYWORDS),
        key_terms=extract_key_terms(query_lower),
    )
