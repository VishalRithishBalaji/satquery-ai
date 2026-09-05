from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RuntimeStatus:
    loaded: bool = False
    model_id: str | None = None
    device: str = "cpu"
    error: str | None = None


class ModelLoader:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.status_state = RuntimeStatus(device="cuda" if torch.cuda.is_available() else "cpu")

    def status(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "loaded": self.status_state.loaded,
            "model_id": self.status_state.model_id,
            "device": self.status_state.device,
            "error": self.status_state.error,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
