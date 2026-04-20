from __future__ import annotations

from typing import Any, Literal, TypedDict


AgentResponseStatus = Literal["success", "error"]


class AgentStepResponse(TypedDict):
    step_id: str
    skill_name: str
    status: AgentResponseStatus
    summary: str
    error: str | None


class AgentFinalResult(TypedDict):
    result_type: str
    title: str
    summary: str
    data: dict[str, Any]


class AgentResponse(TypedDict):
    goal_type: str
    status: AgentResponseStatus
    goal_summary: str
    step_count: int
    steps: list[AgentStepResponse]
    final_result: AgentFinalResult | None
    raw_results: list[dict[str, Any]]
    error: str | None
