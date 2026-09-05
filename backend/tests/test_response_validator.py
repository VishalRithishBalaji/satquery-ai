from backend.agent.query_analysis import QueryIntent
from backend.agent.response_validator import grounding_score


def test_high_grounding_when_answer_addresses_attributes():
    intent = QueryIntent(
        task="optical_sar", operation="describe",
        attributes=["built_up", "water"], key_terms=["built-up", "water", "areas"],
    )
    answer = "Built-up areas are visible in the northeast, and a water body borders the southern edge."
    assert grounding_score(answer, intent) > 0.5


def test_low_grounding_for_generic_answer():
    intent = QueryIntent(
        task="optical_sar", operation="describe",
        attributes=["built_up", "water"], key_terms=["built-up", "water"],
    )
    answer = "This image shows a general overview of the scene with mixed terrain."
    assert grounding_score(answer, intent) < 0.3


def test_empty_answer_has_zero_grounding():
    intent = QueryIntent(task="vqa", operation="describe", attributes=[], key_terms=["ships"])
    assert grounding_score("", intent) == 0.0


def test_no_signals_returns_neutral_score():
    intent = QueryIntent(task="vqa", operation="describe", attributes=[], key_terms=[])
    assert grounding_score("Some answer.", intent) == 0.6
