from uuid import uuid4

from ..agent.graph import run_agent
from ..preprocessing.validation import validate_image
from .. import tools  # noqa: F401


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
