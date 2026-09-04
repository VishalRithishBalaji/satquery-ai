from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def draw_boxes(
    image_path: str,
    detections: list[dict],
    output_path: str,
) -> str:

    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    for detection in detections:

        bbox = detection.get(
            "bbox"
        )

        if not bbox:
            continue

        x1, y1, x2, y2 = bbox

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3,
        )

        label = (
            f"{detection.get('label', 'object')} "
            f"{detection.get('confidence', 0):.2f}"
        )

        draw.text(
            (x1, max(0, y1 - 15)),
            label,
            fill="red",
        )

    Path(
        output_path
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output_path
    )

    return output_path