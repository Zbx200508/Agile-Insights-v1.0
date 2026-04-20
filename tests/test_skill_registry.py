from __future__ import annotations

import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch

from app.services.skill_registry import invoke_skill, list_skills


summary_module = import_module("app.services.skills.summary_skill")
outline_module = import_module("app.services.skills.outline_skill")
qa_rag_module = import_module("app.services.skills.qa_rag_skill")
learning_map_module = import_module("app.services.skills.learning_map_skill")
learning_plan_module = import_module("app.services.skills.learning_plan_skill")


class SkillRegistryTest(unittest.TestCase):
    def test_builtin_skills_can_be_invoked_through_registry(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        parsed_name = Path(__file__).name

        with (
            patch.object(summary_module, "PARSED_DIR", tests_dir),
            patch.object(outline_module, "PARSED_DIR", tests_dir),
            patch.object(summary_module, "_generate_summary", return_value="mock summary"),
            patch.object(outline_module, "_generate_outline", return_value="mock outline"),
            patch.object(
                qa_rag_module,
                "_answer_question_with_rag",
                return_value={
                    "answer": "mock answer",
                    "citations": [{"chunk_id": "c_001"}],
                    "retrieved_chunks": [{"chunk_id": "c_001"}],
                },
            ),
        ):
            summary_result = invoke_skill("summary_skill", {"parsed_name": parsed_name})
            outline_result = invoke_skill("outline_skill", {"parsed_name": parsed_name})
            qa_result = invoke_skill(
                "qa_rag_skill",
                {"parsed_name": parsed_name, "question": "What is this about?"},
            )

        self.assertEqual(summary_result["status"], "success")
        self.assertEqual(summary_result["output"]["summary_text"], "mock summary")

        self.assertEqual(outline_result["status"], "success")
        self.assertEqual(outline_result["output"]["outline_text"], "mock outline")

        self.assertEqual(qa_result["status"], "success")
        self.assertEqual(qa_result["output"]["answer"], "mock answer")
        self.assertEqual(qa_result["output"]["citations"], [{"chunk_id": "c_001"}])

        registered_names = {item["skill_name"] for item in list_skills()}
        self.assertIn("summary_skill", registered_names)
        self.assertIn("outline_skill", registered_names)
        self.assertIn("qa_rag_skill", registered_names)
        self.assertIn("learning_map_skill", registered_names)
        self.assertIn("learning_plan_skill", registered_names)

    def test_learning_skills_can_be_invoked_through_registry(self) -> None:
        tests_dir = Path(__file__).resolve().parent
        parsed_name = Path(__file__).name
        mock_learning_map = {
            "document": {
                "document_id": "demo",
                "document_title": "Demo",
                "document_summary": "Demo summary",
                "estimated_total_minutes": 30,
                "chapter_count": 1,
                "source_parsed_name": parsed_name,
                "learning_status": "draft",
                "current_mode": "",
                "last_updated_at": "",
            },
            "chapters": [
                {
                    "chapter_id": "ch_1",
                    "order": 1,
                    "title": "Chapter 1",
                    "summary": "Chapter summary",
                    "estimated_minutes": 30,
                    "difficulty_level": "medium",
                    "selected": True,
                    "mastery_level": "unfamiliar",
                    "priority_level": "high",
                    "topic_unit_count": 0,
                    "source_scope": "Chapter 1",
                    "topic_units": [],
                }
            ],
        }
        mock_plan = {
            "mode": "system_learning",
            "confirmed_at": "2026-04-20 00:00:00",
            "plan_summary": {"title": "Mock plan", "highlights": []},
            "plan_detail": {"focuses": ["Chapter 1"], "days": []},
        }

        with (
            patch.object(learning_map_module, "PARSED_DIR", tests_dir),
            patch.object(learning_map_module, "_generate_learning_map", return_value=mock_learning_map),
            patch.object(
                learning_plan_module,
                "_load_learning_map",
                return_value=(mock_learning_map, tests_dir / "demo_learning_map.json"),
            ),
            patch.object(learning_plan_module, "_generate_system_plan", return_value=mock_plan),
        ):
            map_result = invoke_skill(
                "learning_map_skill",
                {"parsed_name": parsed_name, "save": False},
            )
            plan_result = invoke_skill(
                "learning_plan_skill",
                {
                    "parsed_name": parsed_name,
                    "mode": "system",
                    "time_budget": {"周一": 30},
                },
            )

        self.assertEqual(map_result["status"], "success")
        self.assertEqual(map_result["output"]["learning_map"], mock_learning_map)
        self.assertEqual(map_result["output"]["chapter_count"], 1)

        self.assertEqual(plan_result["status"], "success")
        self.assertEqual(plan_result["output"]["plan"], mock_plan)
        self.assertEqual(plan_result["output"]["mode"], "system")
        self.assertEqual(plan_result["output"]["summary"], mock_plan["plan_summary"])


if __name__ == "__main__":
    unittest.main()
