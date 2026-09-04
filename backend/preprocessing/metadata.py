def classify_modality(metadata: dict) -> str:
    name = metadata.get("path", "").lower()
    if "sar" in name or "radar" in name:
        return "SAR"
    if metadata.get("count", 0) > 4:
        return "MULTISPECTRAL"
    return "OPTICAL"
