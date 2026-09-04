def mask_metadata(path: str, label: str = "region") -> dict:
    return {"type": "mask", "path": path, "label": label}
