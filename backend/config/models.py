from .settings import settings

MODEL_CONFIG = {
    "vqa": {"name": "geoqwen-vl-2b-eurosat", "enabled": True},
    "caption": {"name": "geoqwen-vl-2b-eurosat", "enabled": True},
    "grounding": {"name": "geochat-grounding", "enabled": False},
    "object_detection": {"name": "object-detection", "enabled": False},
    "change_detection": {"name": "opencv-change-detection", "enabled": True},
    "change_vqa": {"name": "geoqwen-vl-2b-eurosat", "enabled": True},
    "optical_sar": {"name": "prototype-optical-sar-fusion", "enabled": True},
    "runtime": {
        "model_id": settings.model_id,
        "base_model_id": settings.base_model_id,
    },
}
