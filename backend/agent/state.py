from typing import Any, TypedDict
class AgentState(TypedDict, total=False):
    run_id: str; query: str; image_paths: list[str]; metadata: list[dict[str, Any]]; task: str
    selected_tools: list[str]; tool_results: list[dict[str, Any]]; evidence: list[dict[str, Any]]
    answer: str; confidence: float; trace: list[dict[str, Any]]; escalation: bool
    input_quality: float
    # Surfaced onto the top-level response by finalize_node from whichever
    # tool result wins - LangGraph only tracks state channels for keys
    # declared here, so any field a tool returns must be listed below or it
    # is silently dropped between graph nodes.
    tool: str; model: str; fusion: dict[str, Any]; task_intent: dict[str, Any]
    query_grounding_score: float; model_certainty: float
    low_query_sensitivity: bool; retried_for_query_sensitivity: bool
    cross_modal_agreement: float; target: str | None; detections: list[dict[str, Any]]
