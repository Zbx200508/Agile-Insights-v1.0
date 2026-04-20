from __future__ import annotations

from typing import Any

from app.services.skill_types import Skill, SkillResult, make_error_result
from app.services.skills.learning_map_skill import learning_map_skill
from app.services.skills.learning_plan_skill import learning_plan_skill
from app.services.skills.outline_skill import outline_skill
from app.services.skills.qa_rag_skill import qa_rag_skill
from app.services.skills.summary_skill import summary_skill


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if not skill.skill_name:
            raise ValueError("skill_name is required")
        self._skills[skill.skill_name] = skill

    def get(self, skill_name: str) -> Skill | None:
        return self._skills.get(skill_name)

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "skill_name": skill.skill_name,
                "description": skill.description,
                "input_schema": skill.input_schema,
                "output_schema": skill.output_schema,
            }
            for skill in self._skills.values()
        ]

    def invoke(self, skill_name: str, payload: dict[str, Any]) -> SkillResult:
        normalized_payload = dict(payload or {})
        skill = self.get(skill_name)
        if skill is None:
            return make_error_result(
                skill_name=skill_name,
                payload=normalized_payload,
                error=f"Unknown skill: {skill_name}",
            )

        try:
            return skill.execute(normalized_payload)
        except Exception as exc:
            return make_error_result(
                skill_name=skill.skill_name,
                payload=normalized_payload,
                error=str(exc),
            )


default_registry = SkillRegistry()
default_registry.register(summary_skill)
default_registry.register(outline_skill)
default_registry.register(qa_rag_skill)
default_registry.register(learning_map_skill)
default_registry.register(learning_plan_skill)


def register_skill(skill: Skill) -> None:
    default_registry.register(skill)


def get_skill(skill_name: str) -> Skill | None:
    return default_registry.get(skill_name)


def list_skills() -> list[dict[str, Any]]:
    return default_registry.list_skills()


def invoke_skill(skill_name: str, payload: dict[str, Any]) -> SkillResult:
    return default_registry.invoke(skill_name, payload)
