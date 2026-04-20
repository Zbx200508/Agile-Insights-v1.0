from __future__ import annotations

from typing import Any

from app.services.agent_response_types import AgentFinalResult, AgentResponse, AgentStepResponse
from app.services.planner_service import create_plan, execute_plan


GOAL_SUMMARIES = {
    "summarize_document": "总结当前文档核心内容",
    "outline_document": "梳理当前文档结构大纲",
    "ask_document_question": "围绕当前文档回答问题",
    "build_learning_map": "生成当前文档的学习地图",
    "generate_learning_plan": "生成当前文档的学习方案",
    "quick_learn_document": "快速生成当前文档的速通学习结果",
    "system_learn_document": "生成当前文档的系统学习结果",
}

STEP_SUCCESS_SUMMARIES = {
    "summary_skill": "已生成摘要",
    "outline_skill": "已生成大纲",
    "qa_rag_skill": "已完成原文问答",
    "learning_map_skill": "已生成学习地图",
    "learning_plan_skill": "已生成学习方案",
}

STEP_ERROR_SUMMARIES = {
    "summary_skill": "摘要生成失败",
    "outline_skill": "大纲生成失败",
    "qa_rag_skill": "原文问答失败",
    "learning_map_skill": "学习地图生成失败",
    "learning_plan_skill": "学习方案生成失败",
}

RESULT_TYPE_BY_GOAL = {
    "summarize_document": "summary",
    "outline_document": "outline",
    "ask_document_question": "qa_answer",
    "build_learning_map": "learning_map",
    "generate_learning_plan": "learning_plan",
    "quick_learn_document": "composite_result",
    "system_learn_document": "composite_result",
}


def build_agent_response(plan: dict[str, Any], execution_result: dict[str, Any]) -> AgentResponse:
    goal_type = _resolve_goal_type(plan, execution_result)
    raw_results = _normalize_raw_results(execution_result.get("results", []))
    status = "success" if execution_result.get("status") == "success" else "error"
    steps = _build_step_responses(plan, raw_results)
    error = None if status == "success" else str(execution_result.get("error") or plan.get("error") or "执行失败")

    final_result = None
    if status == "success":
        final_result = _build_final_result(goal_type, raw_results)

    return {
        "goal_type": goal_type,
        "status": status,
        "goal_summary": GOAL_SUMMARIES.get(goal_type, "执行当前目标"),
        "step_count": len(steps),
        "steps": steps,
        "final_result": final_result,
        "raw_results": raw_results,
        "error": error,
    }


def plan_execute_and_format(payload: dict[str, Any]) -> AgentResponse:
    plan = create_plan(payload)
    execution_result = execute_plan(plan)
    return build_agent_response(plan, execution_result)


def _resolve_goal_type(plan: dict[str, Any], execution_result: dict[str, Any]) -> str:
    return str(execution_result.get("goal_type") or plan.get("goal_type") or "")


def _normalize_raw_results(raw_results: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_results, list):
        return []
    return [item for item in raw_results if isinstance(item, dict)]


def _build_step_responses(
    plan: dict[str, Any],
    raw_results: list[dict[str, Any]],
) -> list[AgentStepResponse]:
    result_by_step_id = {
        str(item.get("step_id") or ""): item
        for item in raw_results
        if item.get("step_id")
    }
    plan_steps = plan.get("steps", [])
    if not isinstance(plan_steps, list):
        plan_steps = []

    step_responses: list[AgentStepResponse] = []

    for index, raw_step in enumerate(plan_steps, start=1):
        if not isinstance(raw_step, dict):
            continue

        step_id = str(raw_step.get("step_id") or f"step_{index}")
        skill_name = str(raw_step.get("skill_name") or "")
        result = result_by_step_id.get(step_id, {})
        status = "success" if result.get("status") == "success" else "error"
        error = result.get("error") or (None if result else "Step was not executed")

        step_responses.append(
            {
                "step_id": step_id,
                "skill_name": skill_name,
                "status": status,
                "summary": _build_step_summary(skill_name, status),
                "error": str(error) if error else None,
            }
        )

    if step_responses:
        return step_responses

    for index, result in enumerate(raw_results, start=1):
        step_id = str(result.get("step_id") or f"step_{index}")
        skill_name = str(result.get("skill_name") or "")
        status = "success" if result.get("status") == "success" else "error"
        error = result.get("error")

        step_responses.append(
            {
                "step_id": step_id,
                "skill_name": skill_name,
                "status": status,
                "summary": _build_step_summary(skill_name, status),
                "error": str(error) if error else None,
            }
        )

    return step_responses


def _build_step_summary(skill_name: str, status: str) -> str:
    if status == "success":
        return STEP_SUCCESS_SUMMARIES.get(skill_name, "步骤已完成")
    return STEP_ERROR_SUMMARIES.get(skill_name, "步骤执行失败")


def _build_final_result(goal_type: str, raw_results: list[dict[str, Any]]) -> AgentFinalResult:
    if goal_type == "summarize_document":
        output = _last_output(raw_results)
        summary_text = str(output.get("summary_text") or "")
        return {
            "result_type": "summary",
            "title": "文档摘要",
            "summary": _compact_text(summary_text),
            "data": output,
        }

    if goal_type == "outline_document":
        output = _last_output(raw_results)
        outline_text = str(output.get("outline_text") or "")
        return {
            "result_type": "outline",
            "title": "文档大纲",
            "summary": _compact_text(outline_text),
            "data": output,
        }

    if goal_type == "ask_document_question":
        output = _last_output(raw_results)
        answer = str(output.get("answer") or "")
        return {
            "result_type": "qa_answer",
            "title": "原文问答",
            "summary": _compact_text(answer),
            "data": output,
        }

    if goal_type == "build_learning_map":
        output = _last_output(raw_results)
        learning_map = output.get("learning_map") if isinstance(output.get("learning_map"), dict) else {}
        title = _learning_map_title(learning_map) or "学习地图"
        chapter_count = output.get("chapter_count", _chapter_count(learning_map))
        return {
            "result_type": "learning_map",
            "title": title,
            "summary": f"已生成学习地图，共 {chapter_count} 个章节",
            "data": output,
        }

    if goal_type == "generate_learning_plan":
        output = _last_output(raw_results)
        return _learning_plan_final_result(output, "learning_plan")

    if goal_type in {"quick_learn_document", "system_learn_document"}:
        map_output = _first_output_by_skill(raw_results, "learning_map_skill")
        plan_output = _first_output_by_skill(raw_results, "learning_plan_skill")
        plan_result = _learning_plan_final_result(plan_output, "composite_result")
        data = {
            "learning_plan": plan_output,
            "learning_map": map_output,
        }
        return {
            "result_type": "composite_result",
            "title": plan_result["title"],
            "summary": plan_result["summary"],
            "data": data,
        }

    output = _last_output(raw_results)
    return {
        "result_type": RESULT_TYPE_BY_GOAL.get(goal_type, "composite_result"),
        "title": "执行结果",
        "summary": "目标已执行完成",
        "data": output,
    }


def _learning_plan_final_result(output: dict[str, Any], result_type: str) -> AgentFinalResult:
    plan = output.get("plan") if isinstance(output.get("plan"), dict) else {}
    plan_summary = output.get("summary") if isinstance(output.get("summary"), dict) else {}
    if not plan_summary and isinstance(plan.get("plan_summary"), dict):
        plan_summary = plan["plan_summary"]

    title = str(plan_summary.get("title") or "学习方案")
    summary = str(plan_summary.get("subtitle") or "")
    if not summary:
        highlights = plan_summary.get("highlights", [])
        if isinstance(highlights, list):
            summary = "；".join(str(item) for item in highlights[:3] if str(item).strip())

    return {
        "result_type": result_type,
        "title": title,
        "summary": _compact_text(summary or title),
        "data": output,
    }


def _last_output(raw_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not raw_results:
        return {}
    result = raw_results[-1].get("result", {})
    output = result.get("output", {}) if isinstance(result, dict) else {}
    return output if isinstance(output, dict) else {}


def _first_output_by_skill(raw_results: list[dict[str, Any]], skill_name: str) -> dict[str, Any]:
    for item in raw_results:
        if item.get("skill_name") != skill_name:
            continue
        result = item.get("result", {})
        output = result.get("output", {}) if isinstance(result, dict) else {}
        return output if isinstance(output, dict) else {}
    return {}


def _learning_map_title(learning_map: dict[str, Any]) -> str:
    document = learning_map.get("document") if isinstance(learning_map, dict) else {}
    if not isinstance(document, dict):
        return ""
    return str(document.get("document_title") or "")


def _chapter_count(learning_map: dict[str, Any]) -> int:
    chapters = learning_map.get("chapters", []) if isinstance(learning_map, dict) else []
    return len(chapters) if isinstance(chapters, list) else 0


def _compact_text(text: str, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."
