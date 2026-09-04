from __future__ import annotations

from collections import Counter

from ..agent.registry import register_tool
from ..config.settings import settings
from ..models.object_detection_model import ObjectDetectionModel


def _extract_target(query: str) -> str | None:
    q = query.lower().strip()

    targets = [
        "ship",
        "ships",
        "airplane",
        "plane",
        "aircraft",
        "vehicle",
        "vehicles",
        "building",
        "buildings",
        "harbor",
        "bridge",
        "storage tank",
        "helicopter",
        "airport",
    ]

    for target in targets:
        if target in q:
            return target

    return None


@register_tool("object_detection")
def object_detection_tool(state):
    image_paths = state.get("image_paths", [])

    if not image_paths:
        raise ValueError(
            "Object detection requires an image."
        )

    image_path = image_paths[0]

    target = _extract_target(
        state.get("query", "")
    )

    model = ObjectDetectionModel.singleton(
        model_path=getattr(
            settings,
            "object_detection_model",
            "yolo26n-obb.pt",
        ),
        confidence=getattr(
            settings,
            "object_detection_confidence",
            0.25,
        ),
        image_size=getattr(
            settings,
            "object_detection_image_size",
            1024,
        ),
    )

    detections = model.predict(image_path)

    if target:
        normalized_target = target.rstrip("s")

        detections = [
            detection
            for detection in detections
            if detection["label"].lower().rstrip("s")
            == normalized_target
        ]

    if detections:
        average_confidence = sum(
            d["confidence"]
            for d in detections
        ) / len(detections)
    else:
        average_confidence = 0.0

    counts = Counter(
        d["label"]
        for d in detections
    )

    evidence = []

    for detection in detections:
        evidence.append(
            {
                "type": "oriented_bounding_box",
                "label": detection["label"],
                "confidence": detection["confidence"],
                "polygon": detection["polygon"],
                "xywhr": detection["xywhr"],
                "image": image_path,
            }
        )

    if target and not detections:
        answer = (
            f"No confident {target} detections "
            "were found in the image."
        )
    else:
        summary = ", ".join(
            f"{label}: {count}"
            for label, count in sorted(
                counts.items()
            )
        )

        answer = (
            f"Detected {len(detections)} object(s). "
            f"{summary}."
            if summary
            else "No objects were detected."
        )

    return {
        "answer": answer,
        "confidence": round(
            average_confidence,
            4,
        ),
        "tool": "object_detection",
        "model": model.model_path,
        "target": target,
        "detections": detections,
        "evidence": evidence,
    }