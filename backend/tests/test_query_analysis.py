from backend.agent.query_analysis import analyze_query


def test_optical_sar_attributes_and_operation():
    intent = analyze_query(
        "Use the optical and SAR images together to identify built-up areas and water-covered regions.",
        2,
    )
    assert intent.task == "optical_sar"
    assert "built_up" in intent.attributes
    assert "water" in intent.attributes


def test_locate_operation():
    intent = analyze_query("Where is the largest building located?", 1)
    assert intent.operation == "locate"


def test_quantify_operation_and_attribute():
    intent = analyze_query("How many ships are visible in the harbor?", 1)
    assert intent.operation == "quantify"
    assert "ship" in intent.attributes


def test_compare_operation_for_change_query():
    intent = analyze_query("What changed between these images?", 2)
    assert intent.task == "change_vqa"
    assert intent.operation == "compare"


def test_default_describe_operation():
    intent = analyze_query("Tell me about this scene.", 1)
    assert intent.operation == "describe"


def test_different_queries_produce_different_attributes():
    a = analyze_query("Identify built-up areas in this image.", 1)
    b = analyze_query("Identify water bodies in this image.", 1)
    assert a.attributes != b.attributes
