from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.skill_types import (
    SkillResult,
    make_error_result,
    make_success_result,
    require_text_field,
)


BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"

SYSTEM_MODE = "system"
QUICK_MODE = "quick"


def _generate_system_plan(
    learning_map: dict[str, Any],
    week_minutes: dict[str, Any],
    confirmed_at: str,
) -> dict[str, Any]:
    from app.services.system_learning_plan_service import generate_system_learning_plan

    return generate_system_learning_plan(
        learning_map=learning_map,
        week_minutes=week_minutes,
        confirmed_at=confirmed_at,
    )


def _generate_quick_plan(
    learning_map: dict[str, Any],
    target_minutes: int,
    confirmed_at: str,
) -> dict[str, Any]:
    from app.services.quick_understanding_plan_service import generate_quick_understanding_plan

    return generate_quick_understanding_plan(
        learning_map=learning_map,
        target_minutes=target_minutes,
        confirmed_at=confirmed_at,
    )


def _normalize_mode(raw_mode: str) -> str:
    mode = (raw_mode or "").strip().lower()
    if mode in {"system", "system_learning"}:
        return SYSTEM_MODE
    if mode in {"quick", "quick_understanding"}:
        return QUICK_MODE
    raise ValueError("mode must be system or quick")


def _load_learning_map(parsed_name: str) -> tuple[dict[str, Any], Path]:
    if Path(parsed_name).name != parsed_name:
        raise ValueError("parsed_name must be a file name")

    learning_map_path = OUTPUTS_DIR / f"{Path(parsed_name).stem}_learning_map.json"
    if not learning_map_path.exists():
        raise FileNotFoundError(f"Learning map not found: {learning_map_path.name}")

    learning_map = json.loads(learning_map_path.read_text(encoding="utf-8"))
    if not isinstance(learning_map, dict):
        raise ValueError("Learning map must be a JSON object")

    return learning_map, learning_map_path


def _apply_selected_chapters(
    learning_map: dict[str, Any],
    selected_chapters: Any,
) -> list[str]:
    if not selected_chapters:
        return []
    if not isinstance(selected_chapters, list):
        raise ValueError("selected_chapters must be a list")

    selected_ids = {str(item).strip() for item in selected_chapters if str(item).strip()}
    chapters = learning_map.get("chapters", [])
    if not isinstance(chapters, list):
        return []

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_id = str(chapter.get("chapter_id") or "").strip()
        is_selected = chapter_id in selected_ids
        chapter["selected"] = is_selected

        topic_units = chapter.get("topic_units", [])
        if not isinstance(topic_units, list):
            continue
        for unit in topic_units:
            if isinstance(unit, dict):
                unit["selected"] = is_selected

    return sorted(selected_ids)


def _apply_mastery_levels(
    learning_map: dict[str, Any],
    mastery_levels: Any,
) -> dict[str, str]:
    if not mastery_levels:
        return {}
    if not isinstance(mastery_levels, dict):
        raise ValueError("mastery_levels must be an object")

    normalized_levels = {
        str(item_id).strip(): str(level).strip()
        for item_id, level in mastery_levels.items()
        if str(item_id).strip() and str(level).strip()
    }
    chapters = learning_map.get("chapters", [])
    if not isinstance(chapters, list):
        return normalized_levels

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue

        chapter_id = str(chapter.get("chapter_id") or "").strip()
        if chapter_id in normalized_levels:
            chapter["mastery_level"] = normalized_levels[chapter_id]

        topic_units = chapter.get("topic_units", [])
        if not isinstance(topic_units, list):
            continue
        for unit in topic_units:
            if not isinstance(unit, dict):
                continue
            unit_id = str(unit.get("unit_id") or "").strip()
            if unit_id in normalized_levels:
                unit["mastery_level"] = normalized_levels[unit_id]

    return normalized_levels


def _resolve_time_budget(payload: dict[str, Any], mode: str) -> Any:
    raw_budget = payload.get("time_budget")

    if mode == SYSTEM_MODE:
        if isinstance(raw_budget, dict):
            return raw_budget.get("week_minutes", raw_budget)
        return payload.get("week_minutes", {})

    if isinstance(raw_budget, dict):
        raw_target = raw_budget.get("target_minutes", payload.get("target_minutes", 30))
    else:
        raw_target = raw_budget if raw_budget is not None else payload.get("target_minutes", 30)

    try:
        return int(raw_target)
    except Exception:
        return 30


class LearningPlanSkill:
    skill_name = "learning_plan_skill"
    description = "Generate a learning plan from an existing learning map."
    input_schema = {
        "type": "object",
        "required": ["parsed_name", "mode"],
        "properties": {
            "parsed_name": {
                "type": "string",
                "description": "Parsed text file name for the current document.",
            },
            "mode": {
                "type": "string",
                "enum": ["system", "quick", "system_learning", "quick_understanding"],
            },
            "selected_chapters": {
                "type": "array",
                "description": "Optional chapter_id list to include.",
            },
            "mastery_levels": {
                "type": "object",
                "description": "Optional chapter_id/unit_id to mastery_level overrides.",
            },
            "time_budget": {
                "description": "For system mode, week_minutes object. For quick mode, target minutes.",
            },
        },
    }
    output_schema = {
        "type": "object",
        "required": ["plan", "mode", "summary"],
        "properties": {
            "plan": {"type": "object"},
            "mode": {"type": "string"},
            "summary": {"type": "object"},
            "applied_constraints": {"type": "object"},
        },
    }

    def execute(self, payload: dict) -> SkillResult:
        normalized_payload = dict(payload or {})
        try:
            parsed_name = require_text_field(normalized_payload, "parsed_name")
            mode = _normalize_mode(require_text_field(normalized_payload, "mode"))
            learning_map, learning_map_path = _load_learning_map(parsed_name)
            working_map = copy.deepcopy(learning_map)

            selected_chapters = _apply_selected_chapters(
                working_map,
                normalized_payload.get("selected_chapters"),
            )
            mastery_levels = _apply_mastery_levels(
                working_map,
                normalized_payload.get("mastery_levels"),
            )

            time_budget = _resolve_time_budget(normalized_payload, mode)
            confirmed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if mode == SYSTEM_MODE:
                plan = _generate_system_plan(
                    learning_map=working_map,
                    week_minutes=time_budget,
                    confirmed_at=confirmed_at,
                )
            else:
                plan = _generate_quick_plan(
                    learning_map=working_map,
                    target_minutes=time_budget,
                    confirmed_at=confirmed_at,
                )

            return make_success_result(
                skill_name=self.skill_name,
                payload=normalized_payload,
                output={
                    "plan": plan,
                    "mode": mode,
                    "summary": plan.get("plan_summary", {}),
                    "applied_constraints": {
                        "selected_chapters": selected_chapters,
                        "mastery_levels": mastery_levels,
                        "time_budget": time_budget,
                        "learning_map_path": str(learning_map_path),
                    },
                },
            )
        except Exception as exc:
            return make_error_result(
                skill_name=self.skill_name,
                payload=normalized_payload,
                error=str(exc),
            )


learning_plan_skill = LearningPlanSkill()
