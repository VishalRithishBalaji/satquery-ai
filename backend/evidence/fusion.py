from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def tensor_to_heatmap(
    tensor: torch.Tensor,
    output_path: str,
) -> str:

    data = (
        tensor.detach()
        .float()
        .cpu()
        .numpy()
    )

    data = data - data.min()

    max_value = data.max()

    if max_value > 0:
        data = data / max_value

    data = (
        data * 255
    ).astype(np.uint8)

    image = Image.fromarray(
        data,
        mode="L",
    )

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)

    return output_path


def create_fusion_evidence(
    agreement_map: torch.Tensor,
    disagreement_map: torch.Tensor,
    output_dir: str,
) -> dict[str, str]:

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    agreement_path = (
        output_dir
        / "optical_sar_agreement.png"
    )

    disagreement_path = (
        output_dir
        / "optical_sar_disagreement.png"
    )

    return {
        "agreement": tensor_to_heatmap(
            agreement_map,
            str(agreement_path),
        ),
        "disagreement": tensor_to_heatmap(
            disagreement_map,
            str(disagreement_path),
        ),
    }