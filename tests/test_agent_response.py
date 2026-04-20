from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.agent_response_service import build_agent_response, plan_execute_and_format
from app.services.planner_service import create_plan


def _skill_success(skill_name: str, payload: dict) -> dict:
    outputs = {
        "summary_skill": {
            "summary_text": "这是一段用于页面展示的摘要。",
        },
        "learning_map_skill": {
            "learning_map": {
                "document": {"document_title": "Demo Document"},
                "chapters": [{"chapter_id": "ch_1"}],
            },
            "chapter_count": 1,
            "raw_map_path": "data/outputs/demo_learning_map.json",
        },
        "learning_plan_skill": {
            "plan": {
                "mode": "quick_understanding",
                "plan_summary": {
                    "title": "最短理解路径",
                    "subtitle": "用 45 分钟抓住核心脉络。",
                    "highlights": ["推荐投入：45 分钟"],
                },
                "plan_detail": {"steps": []},
            },
            "mode": "quick",
            "summary": {
                "title": "最短理解路径",
                "subtitle": "用 45 分钟抓住核心脉络。",
                "highlights": ["推荐投入：45 分钟"],
            },
        },
    }
    return {
        "skill_name": skill_name,
        "status": "success",
        "input": payload,
        "output": outputs.get(skill_name, {}),
        "error": None,
    }


class AgentResponseServiceTest(unittest.TestCase):
    def test_summarize_document_agent_response(self) -> None:
        with patch(
            "app.services.planner_service.invoke_skill",
            side_effect=lambda skill_name, payload: _skill_success(skill_name, payload),
        ):
            response = plan_execute_and_format(
                {
                    "goal_type": "summarize_document",
                    "parsed_name": "demo.txt",
                }
            )

        self.assertEqual(response["goal_type"], "summarize_document")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["goal_summary"], "总结当前文档核心内容")
        self.assertEqual(response["step_count"], 1)
        self.assertEqual(response["steps"][0]["skill_name"], "summary_skill")
        self.assertEqual(response["steps"][0]["summary"], "已生成摘要")
        self.assertEqual(response["final_result"]["result_type"], "summary")
        self.assertEqual(response["final_result"]["data"]["summary_text"], "这是一段用于页面展示的摘要。")
        self.assertEqual(len(response["raw_results"]), 1)
        self.assertIsNone(response["error"])

    def test_quick_learn_document_agent_response(self) -> None:
        with patch(
            "app.services.planner_service.invoke_skill",
            side_effect=lambda skill_name, payload: _skill_success(skill_name, payload),
        ):
            response = plan_execute_and_format(
                {
                    "goal_type": "quick_learn_document",
                    "parsed_name": "demo.txt",
                    "time_budget": {"target_minutes": 45},
                }
            )

        self.assertEqual(response["goal_type"], "quick_learn_document")
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["goal_summary"], "快速生成当前文档的速通学习结果")
        self.assertEqual(response["step_count"], 2)
        self.assertEqual([step["skill_name"] for step in response["steps"]], ["learning_map_skill", "learning_plan_skill"])
        self.assertEqual(response["steps"][0]["summary"], "已生成学习地图")
        self.assertEqual(response["steps"][1]["summary"], "已生成学习方案")
        self.assertEqual(response["final_result"]["result_type"], "composite_result")
        self.assertEqual(response["final_result"]["title"], "最短理解路径")
        self.assertIn("learning_plan", response["final_result"]["data"])
        self.assertIn("learning_map", response["final_result"]["data"])
        self.assertEqual(response["final_result"]["data"]["learning_map"]["chapter_count"], 1)
        self.assertEqual(len(response["raw_results"]), 2)
        self.assertIsNone(response["error"])

    def test_error_response_keeps_raw_results_and_step_status(self) -> None:
        plan = create_plan(
            {
                "goal_type": "summarize_document",
                "parsed_name": "demo.txt",
            }
        )
        execution_result = {
            "goal_type": "summarize_document",
            "status": "error",
            "steps": plan["steps"],
            "results": [
                {
                    "step_id": "step_1",
                    "skill_name": "summary_skill",
                    "payload": {"parsed_name": "demo.txt"},
                    "status": "error",
                    "result": {
                        "skill_name": "summary_skill",
                        "status": "error",
                        "input": {"parsed_name": "demo.txt"},
                        "output": {},
                        "error": "mock failure",
                    },
                    "error": "mock failure",
                }
            ],
            "error": "step_1 failed: mock failure",
        }

        response = build_agent_response(plan, execution_result)

        self.assertEqual(response["status"], "error")
        self.assertEqual(response["steps"][0]["status"], "error")
        self.assertEqual(response["steps"][0]["summary"], "摘要生成失败")
        self.assertEqual(response["steps"][0]["error"], "mock failure")
        self.assertIsNone(response["final_result"])
        self.assertEqual(len(response["raw_results"]), 1)
        self.assertEqual(response["error"], "step_1 failed: mock failure")


if __name__ == "__main__":
    unittest.main()
