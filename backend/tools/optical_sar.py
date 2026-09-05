from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from ..agent.query_analysis import QueryIntent, analyze_query
from ..agent.query_aware import run_query_aware_vlm
from ..agent.registry import register_tool
from ..agent.sensitivity import image_set_key
from ..config.settings import settings
from ..evidence.fusion import create_fusion_evidence
from ..models.geochat import GeoChatModel
from ..models.optical_sar_model import (
    OpticalSARFusionModel,
)
from ..preprocessing.alignment import (
    align_to_reference,
)


def raster_tensor(
    data: np.ndarray,
) -> torch.Tensor:

    data = data.astype(
        np.float32
    )

    bands = []

    for band in data:

        low, high = np.percentile(
            band,
            [2, 98],
        )

        if high <= low:
            high = low + 1.0

        band = np.clip(
            (band - low)
            / (high - low),
            0.0,
            1.0,
        )

        bands.append(band)

    array = np.stack(bands)

    return torch.from_numpy(
        array
    ).unsqueeze(0)


def load_optical(
    path: str,
) -> np.ndarray:

    with rasterio.open(path) as src:
        data = src.read()

    if data.shape[0] == 1:
        data = np.repeat(
            data,
            3,
            axis=0,
        )

    return data


_OPERATION_HINTS = {
    "locate": "State the approximate location within the scene (e.g. which quadrant or direction) for anything relevant to the query.",
    "compare": "Explicitly compare what the Optical and SAR evidence each show, and call out anything the disagreement map highlights.",
    "quantify": "Give an approximate extent or proportion of the scene that is supported by the evidence (e.g. small/moderate/large area).",
    "confirm": "Open with a direct yes/no-style determination before elaborating.",
    "describe": "Summarize the dominant land cover the imagery actually supports.",
}


def build_prompt(
    intent: QueryIntent,
    query: str,
    previous_answer: str | None = None,
    force: bool = False,
) -> str:

    if intent.attributes:
        focus_line = (
            "The user is specifically asking about: "
            + ", ".join(a.replace("_", " ") for a in intent.attributes)
            + ". Address each of these explicitly."
        )
    else:
        focus_line = (
            "Answer precisely what is asked below. Do not default to a "
            "generic scene description unless the user actually asked for one."
        )

    operation_hint = _OPERATION_HINTS.get(intent.operation, "")

    correction = ""
    if force and previous_answer:
        correction = f"""
IMPORTANT: An earlier answer to a DIFFERENT question about this same imagery was:
"{previous_answer}"
Your new answer must NOT repeat that text. It must specifically and only
address the question asked below, using the visual evidence.
"""
    elif force:
        correction = """
IMPORTANT: Your previous attempt did not clearly and specifically address
the question below. Do not give a generic scene description. Directly
answer the specific question now, using the visual evidence.
"""

    return f"""
You are SatQuery AI, a remote-sensing multimodal assistant.

You are given four visual inputs:

1. Optical satellite image
2. SAR satellite image
3. Optical-SAR agreement evidence map
4. Optical-SAR disagreement evidence map

The Optical image provides spectral and contextual information.
The SAR image provides complementary structural information.
The agreement map indicates regions where both modalities provide
consistent feature responses.
The disagreement map indicates regions where the modalities differ.

{focus_line}
{operation_hint}
{correction}
Lead with a direct answer to the user's question, then briefly explain the
most relevant visual evidence. Do not invent objects, locations, or changes
that are not supported by the imagery.

User query:
{query}
"""


@register_tool("optical_sar")
def optical_sar_tool(state):

    image_paths = state.get(
        "image_paths",
        [],
    )

    if len(image_paths) < 2:
        raise ValueError(
            "Optical-SAR analysis requires "
            "two images: Optical and SAR."
        )

    optical_path = image_paths[0]
    sar_path = image_paths[1]

    query = state.get(
        "query",
        "Analyze the optical and SAR images.",
    )

    intent = analyze_query(query, len(image_paths))

    # ---------------------------------------------------------
    # STEP 1 — Align SAR to Optical reference
    # ---------------------------------------------------------

    sar_data, _ = align_to_reference(
        sar_path,
        optical_path,
    )

    # ---------------------------------------------------------
    # STEP 2 — Load Optical
    # ---------------------------------------------------------

    optical_data = load_optical(
        optical_path
    )

    # ---------------------------------------------------------
    # STEP 3 — Convert to tensors
    # ---------------------------------------------------------

    optical_tensor = raster_tensor(
        optical_data
    )

    sar_tensor = raster_tensor(
        sar_data
    )

    # ---------------------------------------------------------
    # STEP 4 — Optical-SAR feature fusion
    # ---------------------------------------------------------

    fusion_model = (
        OpticalSARFusionModel.singleton()
    )

    fusion_result = fusion_model.fuse(
        optical_tensor,
        sar_tensor,
    )

    cross_modal_agreement = float(
        fusion_result["agreement_map"].mean().item()
    )

    # ---------------------------------------------------------
    # STEP 5 — Generate visual evidence
    # ---------------------------------------------------------

    evidence_dir = (
        Path(settings.output_path)
        / "optical_sar"
    )

    evidence_files = (
        create_fusion_evidence(
            fusion_result[
                "agreement_map"
            ],
            fusion_result[
                "disagreement_map"
            ],
            str(evidence_dir),
        )
    )

    # ---------------------------------------------------------
    # STEP 6 — Convert original imagery
    #         into RGB images for GeoQwen
    # ---------------------------------------------------------

    from ..preprocessing.raster import (
        raster_to_rgb_image,
    )

    optical_image = (
        raster_to_rgb_image(
            optical_path,
            settings.max_image_side,
        )
    )

    sar_image = (
        raster_to_rgb_image(
            sar_path,
            settings.max_image_side,
        )
    )

    agreement_image = raster_to_rgb_image(
        evidence_files["agreement"],
        settings.max_image_side,
    )

    disagreement_image = raster_to_rgb_image(
        evidence_files["disagreement"],
        settings.max_image_side,
    )

    # ---------------------------------------------------------
    # STEP 7 — Send task-specific, query-aware multimodal
    #         evidence to GeoQwen (with sensitivity/grounding
    #         checks and a corrective retry if the answer looks
    #         generic or unchanged from a prior, different query)
    # ---------------------------------------------------------

    vlm = GeoChatModel.singleton(
        settings.model_id,
        settings.base_model_id,
        settings.model_max_pixels,
    )

    generation = run_query_aware_vlm(
        model=vlm,
        images=[
            optical_image,
            sar_image,
            agreement_image,
            disagreement_image,
        ],
        query=query,
        intent=intent,
        image_key=image_set_key(image_paths),
        prompt_builder=build_prompt,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        base_confidence=0.55,
        cross_modal_agreement=cross_modal_agreement,
        input_quality=state.get("input_quality", 1.0),
    )

    evidence = [
        {
            "type": "optical_image",
            "path": optical_path,
        },
        {
            "type": "sar_image",
            "path": sar_path,
        },
        {
            "type": "fusion_agreement",
            "path": evidence_files[
                "agreement"
            ],
        },
        {
            "type": "fusion_disagreement",
            "path": evidence_files[
                "disagreement"
            ],
        },
    ]

    return {
        "answer": generation["answer"],
        "confidence": generation["confidence"],
        "tool": "optical_sar",
        "model": (
            f"{fusion_model.info()['architecture']} "
            "+ "
            f"{settings.model_id}"
        ),
        "query": query,
        "task_intent": {
            "operation": intent.operation,
            "attributes": intent.attributes,
        },
        "query_grounding_score": generation["query_grounding_score"],
        "model_certainty": generation["model_certainty"],
        "low_query_sensitivity": generation["low_query_sensitivity"],
        "retried_for_query_sensitivity": generation["retried_for_query_sensitivity"],
        "cross_modal_agreement": round(cross_modal_agreement, 4),
        "fusion": {
            "architecture": (
                "Dual Encoder + Feature Fusion"
            ),
            "optical_encoder": "ResNet18",
            "sar_encoder": "ResNet18",
            "reasoning_model": settings.model_id,
        },
        "evidence": evidence,
    }
