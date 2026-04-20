from __future__ import annotations

from pathlib import Path

from app.services.skill_types import (
    SkillResult,
    make_error_result,
    make_success_result,
    require_text_field,
)


BASE_DIR = Path(__file__).resolve().parents[3]
PARSED_DIR = BASE_DIR / "data" / "parsed"


def _generate_outline(text: str) -> str:
    from app.services.llm_service import generate_outline

    return generate_outline(text)


def _load_parsed_text(parsed_name: str) -> str:
    if Path(parsed_name).name != parsed_name:
        raise ValueError("parsed_name must be a file name")

    parsed_path = PARSED_DIR / parsed_name
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed text not found: {parsed_name}")

    return parsed_path.read_text(encoding="utf-8")


class OutlineSkill:
    skill_name = "outline_skill"
    description = "Generate a structured outline for a parsed document."
    input_schema = {
        "type": "object",
        "required": ["parsed_name"],
        "properties": {
            "parsed_name": {
                "type": "string",
                "description": "Parsed text file name under data/parsed.",
            },
        },
    }
    output_schema = {
        "type": "object",
        "required": ["outline_text"],
        "properties": {
            "outline_text": {"type": "string"},
        },
    }

    def execute(self, payload: dict) -> SkillResult:
        normalized_payload = dict(payload or {})
        try:
            parsed_name = require_text_field(normalized_payload, "parsed_name")
            text = _load_parsed_text(parsed_name)
            outline_text = _generate_outline(text)
            return make_success_result(
                skill_name=self.skill_name,
                payload=normalized_payload,
                output={"outline_text": outline_text},
            )
        except Exception as exc:
            return make_error_result(
                skill_name=self.skill_name,
                payload=normalized_payload,
                error=str(exc),
            )


outline_skill = OutlineSkill()
