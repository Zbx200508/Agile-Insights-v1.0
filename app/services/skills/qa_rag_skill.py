from __future__ import annotations

from typing import Any

from app.services.skill_types import (
    SkillResult,
    make_error_result,
    make_success_result,
    require_text_field,
)


def _answer_question_with_rag(parsed_name: str, question: str, top_k: int = 5) -> dict[str, Any]:
    from app.services.rag_service import answer_question_with_rag

    return answer_question_with_rag(parsed_name, question, top_k=top_k)


class QaRagSkill:
    skill_name = "qa_rag_skill"
    description = "Answer a question against a parsed document through the existing RAG flow."
    input_schema = {
        "type": "object",
        "required": ["parsed_name", "question"],
        "properties": {
            "parsed_name": {
                "type": "string",
                "description": "Parsed text file name for the current document.",
            },
            "question": {
                "type": "string",
                "description": "User question to answer from the document.",
            },
            "top_k": {
                "type": "integer",
                "description": "Optional retrieval count. Defaults to 5.",
            },
        },
    }
    output_schema = {
        "type": "object",
        "required": ["answer", "citations"],
        "properties": {
            "answer": {"type": "string"},
            "citations": {"type": "array"},
        },
    }

    def execute(self, payload: dict) -> SkillResult:
        normalized_payload = dict(payload or {})
        try:
            parsed_name = require_text_field(normalized_payload, "parsed_name")
            question = require_text_field(normalized_payload, "question")
            top_k = int(normalized_payload.get("top_k") or 5)

            rag_result = _answer_question_with_rag(parsed_name, question, top_k=top_k)
            return make_success_result(
                skill_name=self.skill_name,
                payload=normalized_payload,
                output={
                    "answer": rag_result.get("answer", ""),
                    "citations": rag_result.get("citations", []),
                },
            )
        except Exception as exc:
            return make_error_result(
                skill_name=self.skill_name,
                payload=normalized_payload,
                error=str(exc),
            )


qa_rag_skill = QaRagSkill()
