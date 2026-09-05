from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

import torch


class ObjectDetectionModel:
    """
    Ultralytics YOLO26 OBB adapter.

    Uses the official yolo26n-obb.pt checkpoint for the prototype.
    The model is loaded lazily and can be unloaded to release GPU memory.
    """

    _instance: "ObjectDetectionModel | None" = None
    _lock = Lock()

    def __init__(
        self,
        model_path: str = "yolo26n-obb.pt",
        confidence: float = 0.25,
        image_size: int = 1024,
    ) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size

        self.model: Any = None
        self.loaded = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.load_error: str | None = None

    @classmethod
    def singleton(
        cls,
        model_path: str = "yolo26n-obb.pt",
        confidence: float = 0.25,
        image_size: int = 1024,
    ) -> "ObjectDetectionModel":
        with cls._lock:
            if (
                cls._instance is None
                or cls._instance.model_path != model_path
            ):
                cls._instance = cls(
                    model_path=model_path,
                    confidence=confidence,
                    image_size=image_size,
                )

        return cls._instance

    def load(self) -> "ObjectDetectionModel":
        if self.loaded:
            return self

        try:
            from ultralytics import YOLO

            # If a local path exists, use it.
            # Otherwise Ultralytics downloads the official checkpoint.
            local_path = Path(self.model_path)

            if local_path.exists():
                source = str(local_path)
            else:
                source = self.model_path

            self.model = YOLO(source)

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
        self.loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_loaded(self) -> None:
        if not self.loaded:
            self.load()

    def predict(
        self,
        image_path: str,
        confidence: float | None = None,
        image_size: int | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_loaded()

        conf = (
            self.confidence
            if confidence is None
            else confidence
        )

        imgsz = (
            self.image_size
            if image_size is None
            else image_size
        )

        results = self.model.predict(
            source=image_path,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []

        for result in results:
            if result.obb is None:
                continue

            names = result.names

            classes = (
                result.obb.cls.detach()
                .cpu()
                .tolist()
            )

            confidences = (
                result.obb.conf.detach()
                .cpu()
                .tolist()
            )

            polygons = (
                result.obb.xyxyxyxy.detach()
                .cpu()
                .tolist()
            )

            xywhr = (
                result.obb.xywhr.detach()
                .cpu()
                .tolist()
            )

            for class_id, score, polygon, rotated_box in zip(
                classes,
                confidences,
                polygons,
                xywhr,
            ):
                class_id = int(class_id)

                detections.append(
                    {
                        "class_id": class_id,
                        "label": names[class_id],
                        "confidence": float(score),
                        "polygon": polygon,
                        "xywhr": rotated_box,
                    }
                )

        return detections

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "model_path": self.model_path,
            "device": self.device,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "confidence": self.confidence,
            "image_size": self.image_size,
            "error": self.load_error,
        }
