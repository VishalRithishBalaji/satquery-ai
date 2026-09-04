def fuse_results(results: list[dict]) -> dict:
    usable = [r for r in results if r.get("answer")]
    if not usable:
        return {"answer": "No specialist tool produced a result.", "confidence": 0.0, "evidence": []}
    answer = "\n\n".join(r["answer"] for r in usable)
    confidences = [float(r.get("confidence", 0.5)) for r in usable]
    evidence = [e for r in usable for e in r.get("evidence", [])]
    return {"answer": answer, "confidence": sum(confidences) / len(confidences), "evidence": evidence}
