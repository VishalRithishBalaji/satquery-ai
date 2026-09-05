from backend.agent import sensitivity
from backend.agent.query_analysis import QueryIntent

BUILT_UP_INTENT = QueryIntent(task="vqa", operation="describe", attributes=["built_up"], key_terms=["identify", "built-up", "areas"])
WATER_INTENT = QueryIntent(task="vqa", operation="describe", attributes=["water"], key_terms=["identify", "water", "bodies"])
REPHRASED_BUILT_UP_INTENT = QueryIntent(task="vqa", operation="describe", attributes=["built_up"], key_terms=["identify", "built-up", "areas"])

SETTLEMENT_ANSWER = "There is a small settlement in the northeast."
RIVER_ANSWER = "A river runs along the southern edge."


def test_first_query_on_new_image_is_never_flagged():
    sensitivity.reset()
    result = sensitivity.check_sensitivity("key-a", BUILT_UP_INTENT, SETTLEMENT_ANSWER)
    assert result["low_sensitivity"] is False


def test_materially_different_query_with_identical_answer_is_flagged():
    sensitivity.reset()
    key = "key-b"
    sensitivity.remember(key, BUILT_UP_INTENT, "Identify built-up areas.", SETTLEMENT_ANSWER)

    result = sensitivity.check_sensitivity(key, WATER_INTENT, SETTLEMENT_ANSWER)
    assert result["low_sensitivity"] is True
    assert result["most_similar_answer"] == SETTLEMENT_ANSWER


def test_query_asking_about_the_same_attribute_does_not_flag_a_similar_answer():
    sensitivity.reset()
    key = "key-c"
    sensitivity.remember(key, BUILT_UP_INTENT, "Identify built-up areas in this scene.", SETTLEMENT_ANSWER)

    result = sensitivity.check_sensitivity(key, REPHRASED_BUILT_UP_INTENT, SETTLEMENT_ANSWER)
    assert result["low_sensitivity"] is False


def test_different_query_with_different_answer_is_not_flagged():
    sensitivity.reset()
    key = "key-d"
    sensitivity.remember(key, BUILT_UP_INTENT, "Identify built-up areas.", SETTLEMENT_ANSWER)

    result = sensitivity.check_sensitivity(key, WATER_INTENT, RIVER_ANSWER)
    assert result["low_sensitivity"] is False


def test_history_is_isolated_per_image_key():
    sensitivity.reset()
    sensitivity.remember("key-e-1", BUILT_UP_INTENT, "Identify built-up areas.", SETTLEMENT_ANSWER)

    result = sensitivity.check_sensitivity("key-e-2", WATER_INTENT, SETTLEMENT_ANSWER)
    assert result["low_sensitivity"] is False
    assert result["history_size"] == 0
