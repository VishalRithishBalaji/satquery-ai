from uuid import uuid4

from ..agent.graph import run_agent
from ..preprocessing.validation import validate_image
from .. import tools  # noqa: F401


def _estimate_input_quality(metas: list[dict]) -> float:
    """Cheap, measurable proxy for input quality: images that failed to
    open as georeferenced rasters (no CRS) or that are unusually small are
    weaker evidence than a properly georeferenced, adequately sized image."""
    if not metas:
        return 1.0
    scores = []
    for meta in metas:
        score = 1.0
        if not meta.get("crs"):
            score -= 0.15
        width, height = meta.get("width"), meta.get("height")
        if not width or not height:
            score -= 0.2
        elif min(width, height) < 128:
            score -= 0.15
        scores.append(max(0.4, score))
    return sum(scores) / len(scores)


def analyze(query, image_paths):
    if not query.strip():
        raise ValueError("Query cannot be empty")

    if not image_paths:
        raise ValueError("At least one image is required")

    metas = [validate_image(p) for p in image_paths]

    # Do not require identical raster dimensions here.
    # Optical and SAR products commonly have different pixel dimensions.
    # Pair alignment/resampling is handled by the appropriate specialist tool.
    if len(image_paths) > 2:
        raise ValueError("A maximum of two images is supported")

    run_id = uuid4().hex

    state = {
        "run_id": run_id,
        "query": query.strip(),
        "image_paths": image_paths,
        "metadata": metas,
        "input_quality": _estimate_input_quality(metas),
        "trace": [
            {
                "stage": "validation",
                "status": "success",
                "image_count": len(image_paths),
            }
        ],
    }

    result = run_agent(state)

    result["summary"] = {
        "task": result.get("task"),
        "tools": result.get("selected_tools", []),
        "image_count": len(image_paths),
    }

    return result
