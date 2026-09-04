from backend.agent.router import route


def test_vqa_route():
    task, tools = route("How many buildings are visible?", 1)
    assert task == "vqa"
    assert tools == ["vqa"]


def test_change_route():
    task, tools = route("What changed between these images?", 2)
    assert task == "change_vqa"
    assert tools == ["change_detection", "change_vqa"]


def test_optical_sar_route():
    task, tools = route("Use optical and SAR together.", 2)
    assert task == "optical_sar"
    assert tools == ["optical_sar"]
