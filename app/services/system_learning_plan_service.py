from __future__ import annotations

from math import ceil
from typing import Any
import logging

from app.services.llm_service import generate_system_learning_plan_raw


logger = logging.getLogger(__name__)


class SystemLearningPlanError(Exception):
    """系统学习方案生成失败。"""


DAY_ORDER = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = int(value)
        return number if number >= 0 else default
    except Exception:
        return default


def normalize_week_minutes(raw_week_minutes: dict[str, Any] | None) -> dict[str, int]:
    raw_week_minutes = raw_week_minutes or {}
    normalized = {}
    for day in DAY_ORDER:
        normalized[day] = _safe_int(raw_week_minutes.get(day, 0), 0)
    return normalized


def _priority_rank(priority: str) -> int:
    mapping = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }
    return mapping.get(priority, 2)


def _mastery_rank(mastery: str) -> int:
    """
    越不熟越优先。
    """
    mapping = {
        "unfamiliar": 3,
        "familiar": 2,
        "mastered": 1,
    }
    return mapping.get(mastery, 2)


def extract_selected_learning_scope(learning_map: dict[str, Any]) -> list[dict[str, Any]]:
    chapters = learning_map.get("chapters", [])
    if not isinstance(chapters, list):
        return []

    selected_items: list[dict[str, Any]] = []

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue

        chapter_selected = bool(chapter.get("selected", True))
        topic_units = chapter.get("topic_units", [])
        if not isinstance(topic_units, list):
            topic_units = []

        selected_units = [
            unit for unit in topic_units
            if isinstance(unit, dict) and bool(unit.get("selected", True))
        ]

        # 优先使用被选中的主题单元
        if selected_units:
            for unit in selected_units:
                selected_items.append(
                    {
                        "type": "topic_unit",
                        "chapter_id": chapter.get("chapter_id", ""),
                        "chapter_title": chapter.get("title", ""),
                        "title": unit.get("title", "未命名主题"),
                        "summary": unit.get("summary", ""),
                        "estimated_minutes": _safe_int(unit.get("estimated_minutes", 20), 20),
                        "priority_level": unit.get("priority_level", chapter.get("priority_level", "medium")),
                        "mastery_level": unit.get("mastery_level", chapter.get("mastery_level", "unfamiliar")),
                        "difficulty_level": unit.get("difficulty_level", chapter.get("difficulty_level", "medium")),
                    }
                )
            continue

        # 没有主题单元时，退回章节层
        if chapter_selected:
            selected_items.append(
                {
                    "type": "chapter",
                    "chapter_id": chapter.get("chapter_id", ""),
                    "chapter_title": chapter.get("title", "未命名章节"),
                    "title": chapter.get("title", "未命名章节"),
                    "summary": chapter.get("summary", ""),
                    "estimated_minutes": _safe_int(chapter.get("estimated_minutes", 30), 30),
                    "priority_level": chapter.get("priority_level", "medium"),
                    "mastery_level": chapter.get("mastery_level", "unfamiliar"),
                    "difficulty_level": chapter.get("difficulty_level", "medium"),
                }
            )

    selected_items.sort(
        key=lambda item: (
            -_priority_rank(item.get("priority_level", "medium")),
            -_mastery_rank(item.get("mastery_level", "familiar")),
            -_safe_int(item.get("estimated_minutes", 20), 20),
            item.get("title", ""),
        )
    )

    return selected_items


def _build_fallback_system_learning_plan(
    learning_map: dict[str, Any],
    week_minutes: dict[str, int],
    confirmed_at: str,
) -> dict[str, Any]:
    selected_items = extract_selected_learning_scope(learning_map)
    available_days = [(day, minutes) for day, minutes in week_minutes.items() if minutes > 0]

    if not available_days:
        available_days = [("周一", 30), ("周三", 30), ("周六", 60)]

    total_capacity = sum(minutes for _, minutes in available_days)
    total_estimated = sum(item["estimated_minutes"] for item in selected_items)

    focuses = [item["title"] for item in selected_items[:3]]
    if not focuses:
        focuses = ["当前资料的核心概念", "关键结构", "重点方法"]

    day_plans = []
    item_index = 0

    for day, capacity in available_days:
        remaining = capacity
        tasks = []

        while item_index < len(selected_items) and remaining >= 15:
            item = selected_items[item_index]
            task_minutes = min(item["estimated_minutes"], remaining)

            tasks.append(
                f"学习：{item['title']}（约 {task_minutes} 分钟）"
            )
            remaining -= task_minutes
            item_index += 1

            if len(tasks) >= 2:
                break

        if not tasks:
            tasks.append("轻复习：回顾前面已学重点（约 15 分钟）")

        day_plans.append(
            {
                "day": day,
                "minutes": capacity,
                "tasks": tasks,
            }
        )

    subtitle = "基于当前学习范围、掌握度与每周时间分配生成的系统学习计划。"
    highlights = [
        f"预计总投入：{total_estimated or total_capacity} 分钟",
        f"本周可投入：{total_capacity} 分钟",
        f"已纳入学习项：{len(selected_items)} 项",
    ]

    return {
        "mode": "system_learning",
        "confirmed_at": confirmed_at,
        "plan_summary": {
            "title": "本周学习计划",
            "subtitle": subtitle,
            "highlights": highlights,
        },
        "plan_detail": {
            "focuses": focuses,
            "days": day_plans,
            "review_note": "建议在本周后半段安排一次轻复习，巩固高优先级且尚未掌握的内容。",
        },
    }


def normalize_system_learning_plan(
    raw_data: dict[str, Any],
    confirmed_at: str,
) -> dict[str, Any]:
    if not isinstance(raw_data, dict):
        raise SystemLearningPlanError("系统学习方案原始结果不是对象。")

    title = str(raw_data.get("title") or "本周学习计划").strip()
    subtitle = str(
        raw_data.get("subtitle")
        or "基于当前学习范围、掌握度与每周时间分配生成的系统学习计划。"
    ).strip()

    raw_highlights = raw_data.get("highlights") or []
    highlights = [str(item).strip() for item in raw_highlights if str(item).strip()]
    if not highlights:
        highlights = ["已根据当前学习范围生成系统学习计划"]

    raw_focuses = raw_data.get("focuses") or []
    focuses = [str(item).strip() for item in raw_focuses if str(item).strip()]
    if not focuses:
        focuses = ["当前资料的核心内容", "优先级较高的学习项", "掌握度较低的重点内容"]

    raw_days = raw_data.get("days") or []
    days = []

    if isinstance(raw_days, list):
        for item in raw_days:
            if not isinstance(item, dict):
                continue

            day = str(item.get("day") or "").strip()
            minutes = _safe_int(item.get("minutes", 30), 30)
            raw_tasks = item.get("tasks") or []
            tasks = [str(task).strip() for task in raw_tasks if str(task).strip()]

            if not day or not tasks:
                continue

            days.append(
                {
                    "day": day,
                    "minutes": minutes,
                    "tasks": tasks,
                }
            )

    if not days:
        raise SystemLearningPlanError("系统学习方案缺少有效的日程安排。")

    review_note = str(
        raw_data.get("review_note")
        or "建议在本周后半段安排一次轻复习，巩固高优先级且尚未掌握的内容。"
    ).strip()

    return {
        "mode": "system_learning",
        "confirmed_at": confirmed_at,
        "plan_summary": {
            "title": title,
            "subtitle": subtitle,
            "highlights": highlights,
        },
        "plan_detail": {
            "focuses": focuses,
            "days": days,
            "review_note": review_note,
        },
    }


def generate_system_learning_plan(
    learning_map: dict[str, Any],
    week_minutes: dict[str, Any] | None,
    confirmed_at: str,
) -> dict[str, Any]:
    normalized_week_minutes = normalize_week_minutes(week_minutes)
    selected_scope = extract_selected_learning_scope(learning_map)

    if not selected_scope:
        logger.warning("未提取到有效学习范围，系统学习方案走 fallback。")
        return _build_fallback_system_learning_plan(
            learning_map=learning_map,
            week_minutes=normalized_week_minutes,
            confirmed_at=confirmed_at,
        )

    try:
        raw_data = generate_system_learning_plan_raw(
            selected_scope=selected_scope,
            week_minutes=normalized_week_minutes,
        )
        return normalize_system_learning_plan(
            raw_data=raw_data,
            confirmed_at=confirmed_at,
        )
    except Exception as e:
        logger.warning("系统学习方案生成失败，走 fallback。原因：%s", e)
        return _build_fallback_system_learning_plan(
            learning_map=learning_map,
            week_minutes=normalized_week_minutes,
            confirmed_at=confirmed_at,
        )