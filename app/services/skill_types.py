from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict


SkillStatus = Literal["success", "error"]


class SkillResult(TypedDict):
    skill_name: str
    status: SkillStatus
    input: dict[str, Any]
    output: dict[str, Any]
    error: str | None


class Skill(Protocol):
    skill_name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

    def execute(self, payload: dict[str, Any]) -> SkillResult:
        """Execute the skill with a normalized dict payload."""


def make_success_result(
    skill_name: str,
    payload: dict[str, Any],
    output: dict[str, Any],
) -> SkillResult:
    return {
        "skill_name": skill_name,
        "status": "success",
        "input": payload,
        "output": output,
        "error": None,
    }


def make_error_result(
    skill_name: str,
    payload: dict[str, Any],
    error: str,
) -> SkillResult:
    return {
        "skill_name": skill_name,
        "status": "error",
        "input": payload,
        "output": {},
        "error": error,
    }


def require_text_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()
