from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch

from ..agent.registry import register_tool
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
    # STEP 7 — Send multimodal evidence to GeoQwen
    # ---------------------------------------------------------

    vlm = GeoChatModel.singleton(
        settings.model_id,
        settings.base_model_id,
        settings.model_max_pixels,
    )

    prompt = f"""
You are SatQuery AI, a remote-sensing multimodal assistant.

You are given four visual inputs:

1. Optical satellite image
2. SAR satellite image
3. Optical-SAR agreement evidence map
4. Optical-SAR disagreement evidence map

Use all four visual inputs together.

The Optical image provides spectral and contextual information.
The SAR image provides complementary structural information.
The agreement map indicates regions where both modalities provide
consistent feature responses.
The disagreement map indicates regions where the modalities differ.

Answer the user's query using only evidence from these inputs.

Do not invent objects, locations, or changes that are not supported
by the imagery.

User query:
{query}

Provide a concise answer and explain the most relevant evidence.
"""

    answer = vlm.ask_multiple(
        [
            optical_image,
            sar_image,
            agreement_image,
            disagreement_image,
        ],
        prompt,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
    )

    # ---------------------------------------------------------
    # STEP 8 — Confidence estimation
    # ---------------------------------------------------------

    confidence = 0.70

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
        "answer": answer,
        "confidence": confidence,
        "tool": "optical_sar",
        "model": (
            f"{fusion_model.info()['architecture']} "
            "+ "
            f"{settings.model_id}"
        ),
        "query": query,
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