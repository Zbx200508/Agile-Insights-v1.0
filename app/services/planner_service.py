from __future__ import annotations

from typing import Any

from app.services.planner_types import PlannerExecutionResult, PlannerPlan, PlannerStep
from app.services.skill_registry import invoke_skill


SUPPORTED_GOAL_TYPES = {
    "summarize_document",
    "outline_document",
    "ask_document_question",
    "build_learning_map",
    "generate_learning_plan",
    "quick_learn_document",
    "system_learn_document",
}


def create_plan(payload: dict[str, Any]) -> PlannerPlan:
    normalized_payload = dict(payload or {})
    goal_type = str(normalized_payload.get("goal_type") or "").strip()

    try:
        if not goal_type:
            raise ValueError("goal_type is required")
        if goal_type not in SUPPORTED_GOAL_TYPES:
            raise ValueError(f"Unsupported goal_type: {goal_type}")

        parsed_name = _require_text_field(normalized_payload, "parsed_name")
        steps = _build_steps(goal_type, parsed_name, normalized_payload)
        return {
            "goal_type": goal_type,
            "status": "planned",
            "steps": steps,
            "error": None,
        }
    except Exception as exc:
        return {
            "goal_type": goal_type,
            "status": "error",
            "steps": [],
            "error": str(exc),
        }


def execute_plan(plan: dict[str, Any]) -> PlannerExecutionResult:
    goal_type = str(plan.get("goal_type") or "")
    steps = _normalize_steps(plan.get("steps", []))

    if plan.get("status") != "planned":
        return {
            "goal_type": goal_type,
            "status": "error",
            "steps": steps,
            "results": [],
            "error": str(plan.get("error") or "Plan is not executable"),
        }

    results: list[dict[str, Any]] = []

    for step in steps:
        step_id = step["step_id"]
        skill_name = step["skill_name"]
        step_payload = dict(step.get("payload") or {})

        try:
            skill_result = invoke_skill(skill_name, step_payload)
        except Exception as exc:
            skill_result = {
                "skill_name": skill_name,
                "status": "error",
                "input": step_payload,
                "output": {},
                "error": str(exc),
            }

        step_result = {
            "step_id": step_id,
            "skill_name": skill_name,
            "payload": step_payload,
            "status": skill_result.get("status", "error"),
            "result": skill_result,
            "error": skill_result.get("error"),
        }
        results.append(step_result)

        if skill_result.get("status") != "success":
            return {
                "goal_type": goal_type,
                "status": "error",
                "steps": steps,
                "results": results,
                "error": f"{step_id} failed: {skill_result.get('error')}",
            }

    return {
        "goal_type": goal_type,
        "status": "success",
        "steps": steps,
        "results": results,
        "error": None,
    }


def plan_and_execute(payload: dict[str, Any]) -> dict[str, Any]:
    plan = create_plan(payload)
    execution_result = execute_plan(plan)
    return {
        "plan": plan,
        "execution_result": execution_result,
    }


def _build_steps(
    goal_type: str,
    parsed_name: str,
    payload: dict[str, Any],
) -> list[PlannerStep]:
    if goal_type == "summarize_document":
        return [_make_step(1, "summary_skill", {"parsed_name": parsed_name})]

    if goal_type == "outline_document":
        return [_make_step(1, "outline_skill", {"parsed_name": parsed_name})]

    if goal_type == "ask_document_question":
        question = _require_text_field(payload, "question")
        step_payload = {"parsed_name": parsed_name, "question": question}
        _copy_optional_fields(payload, step_payload, ["top_k"])
        return [_make_step(1, "qa_rag_skill", step_payload)]

    if goal_type == "build_learning_map":
        step_payload = {"parsed_name": parsed_name}
        _copy_optional_fields(payload, step_payload, ["save"])
        return [_make_step(1, "learning_map_skill", step_payload)]

    if goal_type == "generate_learning_plan":
        mode = _require_text_field(payload, "mode")
        return [_make_step(1, "learning_plan_skill", _build_learning_plan_payload(parsed_name, mode, payload))]

    if goal_type == "quick_learn_document":
        return [
            _make_step(1, "learning_map_skill", {"parsed_name": parsed_name, "save": True}),
            _make_step(2, "learning_plan_skill", _build_learning_plan_payload(parsed_name, "quick", payload)),
        ]

    if goal_type == "system_learn_document":
        return [
            _make_step(1, "learning_map_skill", {"parsed_name": parsed_name, "save": True}),
            _make_step(2, "learning_plan_skill", _build_learning_plan_payload(parsed_name, "system", payload)),
        ]

    raise ValueError(f"Unsupported goal_type: {goal_type}")


def _build_learning_plan_payload(
    parsed_name: str,
    mode: str,
    source_payload: dict[str, Any],
) -> dict[str, Any]:
    step_payload = {
        "parsed_name": parsed_name,
        "mode": mode,
    }
    _copy_optional_fields(
        source_payload,
        step_payload,
        [
            "selected_chapters",
            "mastery_levels",
            "time_budget",
            "week_minutes",
            "target_minutes",
        ],
    )
    return step_payload


def _copy_optional_fields(
    source: dict[str, Any],
    target: dict[str, Any],
    field_names: list[str],
) -> None:
    for field_name in field_names:
        if field_name in source and source[field_name] is not None:
            target[field_name] = source[field_name]


def _make_step(index: int, skill_name: str, payload: dict[str, Any]) -> PlannerStep:
    return {
        "step_id": f"step_{index}",
        "skill_name": skill_name,
        "payload": payload,
    }


def _normalize_steps(raw_steps: Any) -> list[PlannerStep]:
    if not isinstance(raw_steps, list):
        return []

    steps: list[PlannerStep] = []
    for index, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            continue
        step_id = str(raw_step.get("step_id") or f"step_{index}")
        skill_name = str(raw_step.get("skill_name") or "").strip()
        step_payload = raw_step.get("payload") if isinstance(raw_step.get("payload"), dict) else {}
        if not skill_name:
            continue
        steps.append(
            {
                "step_id": step_id,
                "skill_name": skill_name,
                "payload": dict(step_payload),
            }
        )
    return steps


def _require_text_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()
