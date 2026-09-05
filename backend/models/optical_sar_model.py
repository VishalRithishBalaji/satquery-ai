from __future__ import annotations

from threading import Lock
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class OpticalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.float())


class SAREncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        original_conv = backbone.conv1

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            self.conv1.weight.copy_(
                original_conv.weight.mean(
                    dim=1,
                    keepdim=True,
                ).float()
            )

        self.features = nn.Sequential(
            self.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.float())


class OpticalSARFusionModel:
    """
    Lightweight Optical-SAR fusion prototype.

    Optical and SAR imagery are encoded independently, aligned in
    feature space, and fused into a joint representation.

    This is a prototype fusion architecture and is not an
    ISRO-specific trained Optical-SAR foundation model.
    """

    _instance: "OpticalSARFusionModel | None" = None
    _lock = Lock()

    def __init__(self, device: str | None = None) -> None:
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.optical_encoder = OpticalEncoder().to(self.device).float()
        self.sar_encoder = SAREncoder().to(self.device).float()

        self.optical_encoder.eval()
        self.sar_encoder.eval()

        self.loaded = True

    @classmethod
    def singleton(
        cls,
        device: str | None = None,
    ) -> "OpticalSARFusionModel":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(device=device)

        return cls._instance

    def _prepare_optical(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("Optical tensor must be NCHW.")

        image = image.float()

        if image.shape[1] == 3:
            return image

        if image.shape[1] == 1:
            return image.repeat(1, 3, 1, 1)

        return image[:, :3]

    def _prepare_sar(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError("SAR tensor must be NCHW.")

        image = image.float()

        if image.shape[1] == 1:
            return image

        return image[:, :1]

    @torch.inference_mode()
    def fuse(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
    ) -> dict[str, Any]:

        optical = (
            self._prepare_optical(optical)
            .to(self.device, dtype=torch.float32)
        )

        sar = (
            self._prepare_sar(sar)
            .to(self.device, dtype=torch.float32)
        )

        self.optical_encoder.float()
        self.sar_encoder.float()

        optical_features = self.optical_encoder(optical)
        sar_features = self.sar_encoder(sar)

        if optical_features.shape[-2:] != sar_features.shape[-2:]:
            sar_features = F.interpolate(
                sar_features,
                size=optical_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        optical_features = F.normalize(
            optical_features.float(),
            dim=1,
        )

        sar_features = F.normalize(
            sar_features.float(),
            dim=1,
        )

        fused_features = torch.cat(
            [
                optical_features,
                sar_features,
            ],
            dim=1,
        )

        disagreement_map = (
            optical_features.mean(dim=1)
            - sar_features.mean(dim=1)
        ).abs()

        disagreement_map = disagreement_map.squeeze(0)

        agreement_map = 1.0 / (
            1.0 + disagreement_map
        )

        return {
            "optical_features": optical_features,
            "sar_features": sar_features,
            "fused_features": fused_features,
            "agreement_map": agreement_map,
            "disagreement_map": disagreement_map,
        }

    def unload(self) -> None:
        self.optical_encoder = None
        self.sar_encoder = None
        self.loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def info(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "device": self.device,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "architecture": (
                "Dual Encoder + Optical-SAR "
                "Feature Fusion"
            ),
        }
