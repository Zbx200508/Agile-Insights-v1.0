from __future__ import annotations

import unittest
from unittest.mock import call, patch

from app.services.planner_service import create_plan, execute_plan, plan_and_execute


def _success(skill_name: str, payload: dict) -> dict:
    return {
        "skill_name": skill_name,
        "status": "success",
        "input": payload,
        "output": {"ok": True},
        "error": None,
    }


class PlannerServiceTest(unittest.TestCase):
    def test_summarize_document_plan_and_execute(self) -> None:
        plan = create_plan(
            {
                "goal_type": "summarize_document",
                "parsed_name": "demo.txt",
            }
        )

        self.assertEqual(plan["status"], "planned")
        self.assertEqual(
            plan["steps"],
            [
                {
                    "step_id": "step_1",
                    "skill_name": "summary_skill",
                    "payload": {"parsed_name": "demo.txt"},
                }
            ],
        )

        with patch(
            "app.services.planner_service.invoke_skill",
            side_effect=lambda skill_name, payload: _success(skill_name, payload),
        ) as invoke_mock:
            execution_result = execute_plan(plan)

        self.assertEqual(execution_result["status"], "success")
        self.assertEqual(len(execution_result["results"]), 1)
        invoke_mock.assert_called_once_with("summary_skill", {"parsed_name": "demo.txt"})

    def test_quick_learn_document_plan_and_execute(self) -> None:
        with patch(
            "app.services.planner_service.invoke_skill",
            side_effect=lambda skill_name, payload: _success(skill_name, payload),
        ) as invoke_mock:
            result = plan_and_execute(
                {
                    "goal_type": "quick_learn_document",
                    "parsed_name": "demo.txt",
                    "time_budget": {"target_minutes": 45},
                }
            )

        plan = result["plan"]
        execution_result = result["execution_result"]

        self.assertEqual(plan["status"], "planned")
        self.assertEqual(
            plan["steps"],
            [
                {
                    "step_id": "step_1",
                    "skill_name": "learning_map_skill",
                    "payload": {"parsed_name": "demo.txt", "save": True},
                },
                {
                    "step_id": "step_2",
                    "skill_name": "learning_plan_skill",
                    "payload": {
                        "parsed_name": "demo.txt",
                        "mode": "quick",
                        "time_budget": {"target_minutes": 45},
                    },
                },
            ],
        )
        self.assertEqual(execution_result["status"], "success")
        self.assertEqual(len(execution_result["results"]), 2)
        invoke_mock.assert_has_calls(
            [
                call("learning_map_skill", {"parsed_name": "demo.txt", "save": True}),
                call(
                    "learning_plan_skill",
                    {
                        "parsed_name": "demo.txt",
                        "mode": "quick",
                        "time_budget": {"target_minutes": 45},
                    },
                ),
            ]
        )

    def test_execute_plan_stops_on_step_error(self) -> None:
        plan = create_plan(
            {
                "goal_type": "system_learn_document",
                "parsed_name": "demo.txt",
            }
        )

        with patch(
            "app.services.planner_service.invoke_skill",
            return_value={
                "skill_name": "learning_map_skill",
                "status": "error",
                "input": {"parsed_name": "demo.txt", "save": True},
                "output": {},
                "error": "mock failure",
            },
        ) as invoke_mock:
            execution_result = execute_plan(plan)

        self.assertEqual(execution_result["status"], "error")
        self.assertEqual(len(execution_result["results"]), 1)
        self.assertEqual(execution_result["results"][0]["step_id"], "step_1")
        self.assertIn("step_1 failed", execution_result["error"])
        invoke_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
