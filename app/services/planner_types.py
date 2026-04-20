from __future__ import annotations

from typing import Any, Literal, TypedDict


PlanStatus = Literal["planned", "error"]
ExecutionStatus = Literal["success", "error"]


class PlannerStep(TypedDict):
    step_id: str
    skill_name: str
    payload: dict[str, Any]


class PlannerPlan(TypedDict):
    goal_type: str
    status: PlanStatus
    steps: list[PlannerStep]
    error: str | None


class PlannerExecutionResult(TypedDict):
    goal_type: str
    status: ExecutionStatus
    steps: list[PlannerStep]
    results: list[dict[str, Any]]
    error: str | None
