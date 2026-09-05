from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import torch


class GroundingDINOModel:
    """Text-guided visual grounding using Grounding DINO."""

    _instance: "GroundingDINOModel | None" = None
    _lock = Lock()

    def __init__(
        self,
        model_path: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        self.model_path = model_path
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.model: Any = None
        self.processor: Any = None
        self.loaded = False
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.load_error: str | None = None

    @classmethod
    def singleton(
        cls,
        model_path: str = "IDEA-Research/grounding-dino-base",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> "GroundingDINOModel":

        with cls._lock:
            if (
                cls._instance is None
                or cls._instance.model_path != model_path
            ):
                cls._instance = cls(
                    model_path,
                    box_threshold,
                    text_threshold,
                )

        return cls._instance

    def load(self) -> "GroundingDINOModel":
        if self.loaded:
            return self

        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )

            self.processor = AutoProcessor.from_pretrained(
                self.model_path
            )

            self.model = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(
                    self.model_path,
                    torch_dtype=(
                        torch.float16
                        if self.device == "cuda"
                        else torch.float32
                    ),
                )
            )

            self.model.to(self.device)
            self.model.eval()

            self.loaded = True
            self.load_error = None

            return self

        except Exception as exc:
            self.loaded = False
            self.load_error = (
                f"{type(exc).__name__}: {exc}"
            )
            raise

    def unload(self) -> None:
        self.model = None
        self.processor = None
        self.loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ground(
        self,
        image,
        phrase: str,
    ) -> list[dict[str, Any]]:

        self.load()

        inputs = self.processor(
            images=image,
            text=[phrase],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[
                [image.height, image.width]
            ],
        )

        result = results[0]

        detections = []

        for box, score, label in zip(
            result["boxes"],
            result["scores"],
            result["labels"],
        ):
            detections.append(
                {
                    "label": str(label),
                    "confidence": float(score),
                    "bbox": [
                        round(float(x), 2)
                        for x in box.tolist()
                    ],
                }
            )

        return detections

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "model": self.model_path,
            "device": self.device,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "error": self.load_error,
        }
