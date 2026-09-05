"""Evidence-based confidence.

Replaces the prototype's fixed per-tool confidence constants with a score
computed from whatever measurable signals a tool actually has: model
certainty (mean generation-token probability, where available), evidence
support (detector scores, change fraction, agreement-map strength),
cross-modal consistency (Optical-SAR agreement), how well the answer
engages the specific query, and input quality. Signals that a given tool
cannot supply are simply omitted and the remaining weights renormalize -
this is still a prototype estimate, not a calibrated probability, but it
is now derived from evidence rather than a hardcoded literal.
"""

from __future__ import annotations

from ..utils.helpers import clamp

_WEIGHTS = {
    "base": 0.15,
    "grounding_score": 0.30,
    "model_certainty": 0.20,
    "evidence_strength": 0.20,
    "cross_modal_agreement": 0.15,
}

_FLOOR = 0.05
_CEILING = 0.97


def confidence_from_result(result: dict) -> float:
    return clamp(float(result.get("confidence", 0.0)))


def compute_confidence(
    *,
    base: float = 0.5,
    grounding_score: float | None = None,
    model_certainty: float | None = None,
    evidence_strength: float | None = None,
    cross_modal_agreement: float | None = None,
    sensitivity_penalty: float = 0.0,
    input_quality: float = 1.0,
) -> float:
    signals = {"base": clamp(base)}

    if grounding_score is not None:
        signals["grounding_score"] = clamp(grounding_score)
    if model_certainty is not None:
        signals["model_certainty"] = clamp(model_certainty)
    if evidence_strength is not None:
        signals["evidence_strength"] = clamp(evidence_strength)
    if cross_modal_agreement is not None:
        signals["cross_modal_agreement"] = clamp(cross_modal_agreement)

    weight_sum = sum(_WEIGHTS[name] for name in signals)
    weighted = sum(_WEIGHTS[name] * value for name, value in signals.items()) / weight_sum

    weighted -= clamp(sensitivity_penalty)
    weighted *= clamp(input_quality, 0.0, 1.0)

    return round(clamp(weighted, _FLOOR, _CEILING), 4)


def change_evidence_strength(changed_fraction: float | None) -> float | None:
    """Very low or near-total change fraction is a weaker signal (likely
    noise, or seasonal/illumination/alignment effects dominating the whole
    scene); a moderate, localized fraction is the strongest signal that a
    real, bounded change was found."""
    if changed_fraction is None:
        return None
    if changed_fraction < 0.01:
        return 0.35
    if changed_fraction < 0.45:
        return 0.75
    return 0.4


def detection_evidence_strength(detections: list[dict]) -> float:
    if not detections:
        return 0.2
    scores = [float(d.get("confidence", 0.0)) for d in detections]
    return clamp(sum(scores) / len(scores))
