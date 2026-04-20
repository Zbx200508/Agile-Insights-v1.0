from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.skill_types import (
    SkillResult,
    make_error_result,
    make_success_result,
    require_text_field,
)


BASE_DIR = Path(__file__).resolve().parents[3]
PARSED_DIR = BASE_DIR / "data" / "parsed"
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"


def _generate_learning_map(parsed_name: str, text: str) -> dict[str, Any]:
    from app.services.learning_map_service import LearningMapInput, generate_learning_map

    document_id = Path(parsed_name).stem
    input_data = LearningMapInput(
        document_id=document_id,
        document_title=document_id,
        parsed_name=parsed_name,
        text=text,
    )
    return generate_learning_map(input_data)


def _load_parsed_text(parsed_name: str) -> str:
    if Path(parsed_name).name != parsed_name:
        raise ValueError("parsed_name must be a file name")

    parsed_path = PARSED_DIR / parsed_name
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed text not found: {parsed_name}")

    return parsed_path.read_text(encoding="utf-8")


def _get_learning_map_path(parsed_name: str) -> Path:
    return OUTPUTS_DIR / f"{Path(parsed_name).stem}_learning_map.json"


def _save_learning_map(parsed_name: str, learning_map: dict[str, Any]) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    learning_map_path = _get_learning_map_path(parsed_name)
    learning_map_path.write_text(
        json.dumps(learning_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return learning_map_path


class LearningMapSkill:
    skill_name = "learning_map_skill"
    description = "Generate a structured learning map for a parsed document."
    input_schema = {
        "type": "object",
        "required": ["parsed_name"],
        "properties": {
            "parsed_name": {
                "type": "string",
                "description": "Parsed text file name under data/parsed.",
            },
            "save": {
                "type": "boolean",
                "description": "Whether to save the map to data/outputs. Defaults to true.",
            },
        },
    }
    output_schema = {
        "type": "object",
        "required": ["learning_map", "chapter_count"],
        "properties": {
            "learning_map": {"type": "object"},
            "chapter_count": {"type": "integer"},
            "raw_map_path": {"type": "string"},
            "metadata": {"type": "object"},
        },
    }

    def execute(self, payload: dict) -> SkillResult:
        normalized_payload = dict(payload or {})
        try:
            parsed_name = require_text_field(normalized_payload, "parsed_name")
            should_save = bool(normalized_payload.get("save", True))

            text = _load_parsed_text(parsed_name)
            learning_map = _generate_learning_map(parsed_name, text)
            chapters = learning_map.get("chapters", [])
            chapter_count = len(chapters) if isinstance(chapters, list) else 0

            raw_map_path = ""
            if should_save:
                raw_map_path = str(_save_learning_map(parsed_name, learning_map))

            return make_success_result(
                skill_name=self.skill_name,
                payload=normalized_payload,
                output={
                    "learning_map": learning_map,
                    "chapter_count": chapter_count,
                    "raw_map_path": raw_map_path,
                    "metadata": {
                        "parsed_name": parsed_name,
                        "saved": should_save,
                    },
                },
            )
        except Exception as exc:
            return make_error_result(
                skill_name=self.skill_name,
                payload=normalized_payload,
                error=str(exc),
            )


learning_map_skill = LearningMapSkill()
