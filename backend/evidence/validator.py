def validate_evidence(result: dict) -> dict:
    return {
        "has_answer": bool(result.get("answer")),
        "has_evidence": bool(result.get("evidence")),
        "confidence": float(result.get("confidence", 0.0)),
    }
