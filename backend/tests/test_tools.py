from backend.agent.registry import list_tools


def test_tools_registered():
    expected = {"caption", "change_detection", "change_vqa", "geospatial", "grounding", "object_detection", "optical_sar", "vqa"}
    assert expected.issubset(set(list_tools()))
