from __future__ import annotations

from typing import Any
import logging

from app.services.llm_service import generate_quick_understanding_plan_raw
from app.services.system_learning_plan_service import extract_selected_learning_scope


logger = logging.getLogger(__name__)


class QuickUnderstandingPlanError(Exception):
    """速通理解方案生成失败。"""


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
        return number if number >= 0 else default
    except Exception:
        return default


def _fallback_quick_understanding_plan(
    learning_map: dict[str, Any],
    target_minutes: int,
    confirmed_at: str,
) -> dict[str, Any]:
    selected_items = extract_selected_learning_scope(learning_map)
    if not selected_items:
        selected_items = [
            {
                "title": "当前资料的核心概念",
                "summary": "建议先理解资料中的基础框架与关键概念。",
                "estimated_minutes": 10,
            },
            {
                "title": "当前资料的重要方法",
                "summary": "建议再看主要方法或分析框架。",
                "estimated_minutes": 10,
            },
            {
                "title": "当前资料的实际应用",
                "summary": "最后结合应用场景理解整体链路。",
                "estimated_minutes": 10,
            },
        ]

    chosen = []
    remaining = max(target_minutes, 15)

    for item in selected_items:
        if len(chosen) >= 4:
            break

        item_minutes = _safe_int(item.get("estimated_minutes", 10), 10)
        allocated = min(max(8, item_minutes), remaining) if remaining > 0 else max(8, item_minutes)
        chosen.append(
            {
                "title": item.get("title", "核心内容"),
                "minutes": allocated,
                "why": item.get("summary", "") or "这是当前学习范围中的核心内容，适合作为速通重点。",
            }
        )
        remaining -= allocated

    must_know = []
    for item in chosen:
        title = item["title"]
        if title not in must_know:
            must_know.append(title)

    return {
        "mode": "quick_understanding",
        "confirmed_at": confirmed_at,
        "plan_summary": {
            "title": "最短理解路径",
            "subtitle": "基于当前学习范围与目标投入时间生成的速通理解方案。",
            "highlights": [
                f"推荐投入：{target_minutes} 分钟",
                f"核心内容：{len(chosen)} 项",
            ],
        },
        "plan_detail": {
            "steps": chosen,
            "must_know": must_know[:5],
            "next_action": "若希望系统学习，建议保留当前高优先级内容进一步生成一周计划。",
        },
    }


def normalize_quick_understanding_plan(
    raw_data: dict[str, Any],
    confirmed_at: str,
) -> dict[str, Any]:
    if not isinstance(raw_data, dict):
        raise QuickUnderstandingPlanError("速通理解方案原始结果不是对象。")

    title = str(raw_data.get("title") or "最短理解路径").strip()
    subtitle = str(
        raw_data.get("subtitle")
        or "基于当前学习范围与目标投入时间生成的速通理解方案。"
    ).strip()

    raw_highlights = raw_data.get("highlights") or []
    highlights = [str(item).strip() for item in raw_highlights if str(item).strip()]
    if not highlights:
        highlights = ["推荐投入：30 分钟", "核心内容：3 项"]

    raw_steps = raw_data.get("steps") or []
    steps = []

    if isinstance(raw_steps, list):
        for item in raw_steps:
            if not isinstance(item, dict):
                continue

            step_title = str(item.get("title") or "").strip()
            step_minutes = _safe_int(item.get("minutes", 10), 10)
            step_why = str(item.get("why") or "").strip()

            if not step_title:
                continue

            steps.append(
                {
                    "title": step_title,
                    "minutes": step_minutes,
                    "why": step_why or "这是当前学习范围中的关键内容。",
                }
            )

    if not steps:
        raise QuickUnderstandingPlanError("速通理解方案缺少有效步骤。")

    raw_must_know = raw_data.get("must_know") or []
    must_know = [str(item).strip() for item in raw_must_know if str(item).strip()]
    if not must_know:
        must_know = [step["title"] for step in steps[:4]]

    next_action = str(
        raw_data.get("next_action")
        or "如果希望系统学习，建议保留当前高优先级内容进一步生成一周计划。"
    ).strip()

    return {
        "mode": "quick_understanding",
        "confirmed_at": confirmed_at,
        "plan_summary": {
            "title": title,
            "subtitle": subtitle,
            "highlights": highlights,
        },
        "plan_detail": {
            "steps": steps,
            "must_know": must_know,
            "next_action": next_action,
        },
    }


def generate_quick_understanding_plan(
    learning_map: dict[str, Any],
    target_minutes: int,
    confirmed_at: str,
) -> dict[str, Any]:
    selected_scope = extract_selected_learning_scope(learning_map)
    target_minutes = max(_safe_int(target_minutes, 30), 15)

    if not selected_scope:
        logger.warning("未提取到有效学习范围，速通理解方案走 fallback。")
        return _fallback_quick_understanding_plan(
            learning_map=learning_map,
            target_minutes=target_minutes,
            confirmed_at=confirmed_at,
        )

    try:
        raw_data = generate_quick_understanding_plan_raw(
            selected_scope=selected_scope,
            target_minutes=target_minutes,
        )
        return normalize_quick_understanding_plan(
            raw_data=raw_data,
            confirmed_at=confirmed_at,
        )
    except Exception as e:
        logger.warning("速通理解方案生成失败，走 fallback。原因：%s", e)
        return _fallback_quick_understanding_plan(
            learning_map=learning_map,
            target_minutes=target_minutes,
            confirmed_at=confirmed_at,
        )