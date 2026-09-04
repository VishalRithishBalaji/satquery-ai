from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


def read_raster(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read()
        profile = src.profile.copy()

    return data, profile


def align_to_reference(
    source_path: str,
    reference_path: str,
    output_path: str | None = None,
) -> tuple[np.ndarray, dict]:

    with rasterio.open(reference_path) as ref:
        reference_crs = ref.crs
        reference_transform = ref.transform
        reference_width = ref.width
        reference_height = ref.height

    with rasterio.open(source_path) as src:
        source = src.read()

        destination = np.zeros(
            (
                src.count,
                reference_height,
                reference_width,
            ),
            dtype=source.dtype,
        )

        for band in range(src.count):
            reproject(
                source=source[band],
                destination=destination[band],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=reference_transform,
                dst_crs=reference_crs,
                resampling=Resampling.bilinear,
            )

        profile = src.profile.copy()
        profile.update(
            {
                "height": reference_height,
                "width": reference_width,
                "transform": reference_transform,
                "crs": reference_crs,
            }
        )

    if output_path:
        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            output_path,
            "w",
            **profile,
        ) as dst:
            dst.write(destination)

    return destination, profile


def normalize_array(
    data: np.ndarray,
) -> np.ndarray:

    data = data.astype(np.float32)

    normalized = []

    for band in data:
        low, high = np.percentile(band, [2, 98])

        if high <= low:
            high = low + 1.0

        band = np.clip(
            (band - low) / (high - low),
            0.0,
            1.0,
        )

        normalized.append(band)

    return np.stack(normalized)


def to_rgb(data: np.ndarray) -> np.ndarray:
    """
    Convert raster data to a 3-channel RGB-like float32 image.
    If 3+ bands are available, use the first 3.
    If 2 bands are available, duplicate the first band.
    If 1 band is available, repeat it across RGB.
    """
    data = normalize_array(data)

    if data.shape[0] >= 3:
        rgb = data[:3]
    elif data.shape[0] == 2:
        rgb = np.stack([data[0], data[1], data[0]])
    elif data.shape[0] == 1:
        rgb = np.repeat(data, 3, axis=0)
    else:
        raise ValueError("Raster contains no bands")

    return rgb.astype(np.float32)


def validate_alignment(
    image_a: np.ndarray,
    image_b: np.ndarray,
) -> bool:
    if image_a.ndim != 3 or image_b.ndim != 3:
        return False

    if image_a.shape[1:] != image_b.shape[1:]:
        return False

    return True


def align_pair_to_common_rgb(
    image_paths: list[str] | tuple[str, str],
) -> tuple[np.ndarray, np.ndarray]:

    if len(image_paths) != 2:
        raise ValueError("Exactly two images are required")

    first, first_profile = read_raster(image_paths[0])
    second, second_profile = read_raster(image_paths[1])

    first_rgb = to_rgb(first)

    if (
        first_profile.get("crs") is not None
        and second_profile.get("crs") is not None
        and (
            second.shape[1] != first.shape[1]
            or second.shape[2] != first.shape[2]
            or second_profile.get("transform") != first_profile.get("transform")
            or second_profile.get("crs") != first_profile.get("crs")
        )
    ):
        second_aligned, _ = align_to_reference(
            image_paths[1],
            image_paths[0],
        )
    else:
        second_aligned = second

    second_rgb = to_rgb(second_aligned)

    if not validate_alignment(first_rgb, second_rgb):
        raise ValueError(
            f"Images could not be aligned: "
            f"{first_rgb.shape} vs {second_rgb.shape}"
        )

    return first_rgb, second_rgb
