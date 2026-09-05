import pytest

from backend.evidence.confidence import (
    change_evidence_strength,
    compute_confidence,
    detection_evidence_strength,
)


def test_confidence_stays_in_bounds_even_with_out_of_range_inputs():
    value = compute_confidence(base=2.0, grounding_score=5.0, model_certainty=-3.0)
    assert 0.0 <= value <= 1.0


def test_higher_grounding_increases_confidence():
    low = compute_confidence(base=0.5, grounding_score=0.1)
    high = compute_confidence(base=0.5, grounding_score=0.9)
    assert high > low


def test_sensitivity_penalty_reduces_confidence():
    without_penalty = compute_confidence(base=0.5, grounding_score=0.8, sensitivity_penalty=0.0)
    with_penalty = compute_confidence(base=0.5, grounding_score=0.8, sensitivity_penalty=0.2)
    assert with_penalty < without_penalty


def test_missing_signals_still_produce_a_bounded_score():
    value = compute_confidence(base=0.5)
    assert 0.05 <= value <= 0.97


def test_change_evidence_strength_prefers_localized_change():
    assert change_evidence_strength(0.2) > change_evidence_strength(0.005)
    assert change_evidence_strength(0.2) > change_evidence_strength(0.9)
    assert change_evidence_strength(None) is None


def test_detection_evidence_strength_empty_list():
    assert detection_evidence_strength([]) < 0.5


def test_detection_evidence_strength_uses_average():
    detections = [{"confidence": 0.9}, {"confidence": 0.7}]
    assert detection_evidence_strength(detections) == pytest.approx(0.8)
