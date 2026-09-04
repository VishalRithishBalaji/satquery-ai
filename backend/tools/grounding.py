from __future__ import annotations

import re

from ..agent.registry import register_tool
from ..models.grounding_model import GroundingDINOModel
from ..preprocessing.raster import raster_to_rgb_image


STOP_WORDS = {
    "highlight",
    "locate",
    "find",
    "where",
    "the",
    "a",
    "an",
    "in",
    "this",
    "image",
    "region",
    "show",
    "identify",
}


def extract_target(query: str) -> str:
    q = query.lower()

    q = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        q,
    )

    words = [
        word
        for word in q.split()
        if word not in STOP_WORDS
    ]

    return " ".join(words[:5]).strip()


@register_tool("grounding")
def grounding_tool(state):

    image_paths = state.get(
        "image_paths",
        [],
    )

    if not image_paths:
        raise ValueError(
            "Grounding requires an image."
        )

    image_path = image_paths[0]

    target = extract_target(
        state["query"]
    )

    if not target:
        raise ValueError(
            "Could not determine grounding target."
        )

    image = raster_to_rgb_image(
        image_path
    )

    model = GroundingDINOModel.singleton(
        model_path=getattr(
            __import__(
                "backend.config.settings",
                fromlist=["settings"],
            ),
            "settings",
        ).grounding_model,
    )

    detections = model.ground(
        image,
        target,
    )

    if detections:
        avg_confidence = sum(
            d["confidence"]
            for d in detections
        ) / len(detections)

        answer = (
            f"Located {len(detections)} "
            f"region(s) related to '{target}'."
        )
    else:
        avg_confidence = 0.0

        answer = (
            f"No confident '{target}' region "
            "was found."
        )

    evidence = []

    for detection in detections:
        evidence.append(
            {
                "type": "bounding_box",
                "label": detection["label"],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "image": image_path,
            }
        )

    return {
        "answer": answer,
        "confidence": round(
            avg_confidence,
            4,
        ),
        "tool": "grounding",
        "model": model.model_path,
        "target": target,
        "detections": detections,
        "evidence": evidence,
    }