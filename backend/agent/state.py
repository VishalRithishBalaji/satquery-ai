from typing import Any, TypedDict
class AgentState(TypedDict, total=False):
    run_id: str; query: str; image_paths: list[str]; metadata: list[dict[str, Any]]; task: str
    selected_tools: list[str]; tool_results: list[dict[str, Any]]; evidence: list[dict[str, Any]]
    answer: str; confidence: float; trace: list[dict[str, Any]]; escalation: bool
