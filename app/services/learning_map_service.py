from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any
from app.services.llm_service import generate_learning_map_raw
import logging
import re


logger = logging.getLogger(__name__)


class LearningMapError(Exception):
    """学习地图生成或校验失败。"""


@dataclass
class LearningMapInput:
    document_id: str
    document_title: str
    parsed_name: str
    text: str


ALLOWED_DIFFICULTY = {"low", "medium", "high"}
ALLOWED_MASTERY = {"mastered", "familiar", "unfamiliar"}
ALLOWED_PRIORITY = {"low", "medium", "high"}


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = _clean_text(value, default=default).lower()
    return text if text in allowed else default


def _clamp_positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
        return number if number > 0 else default
    except Exception:
        return default


def _estimate_minutes_from_text(text: str) -> int:
    """
    非精确时间估算。
    这里先按中文阅读/理解的粗略速度估一个“可用”值，不追求精确。
    """
    char_count = len(text.strip())
    if char_count <= 0:
        return 15

    # 经验值：学习理解比纯阅读更慢，按每 180~220 字约 1 分钟粗估。
    minutes = ceil(char_count / 200)
    return max(15, minutes)


def _split_fallback_sections(text: str, max_sections: int = 4) -> list[str]:
    """
    当模型结果不可用时，按段落做一个最小可用切分。
    目标不是完美，而是至少能给前端一份可编辑的章节层学习地图。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return ["资料内容较短或结构不明显，已生成最小学习单元。"]

    chunk_size = max(1, ceil(len(paragraphs) / max_sections))
    sections = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = paragraphs[i:i + chunk_size]
        sections.append("\n\n".join(chunk))

    return sections[:max_sections]


def _guess_title_from_section(section_text: str, index: int) -> str:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    if not lines:
        return f"模块 {index}"

    first = lines[0]
    if len(first) <= 28:
        return first

    return f"模块 {index}"


def _summarize_section_brief(section_text: str, limit: int = 60) -> str:
    one_line = re.sub(r"\s+", " ", section_text).strip()
    if not one_line:
        return "该部分主要为资料内容片段。"
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit].rstrip() + "…"


def build_fallback_learning_map(input_data: LearningMapInput) -> dict[str, Any]:
    """
    最小兜底版本：
    - 没有可靠章节结构时，按文本顺序切成 3~4 个模块
    - topic_units 先为空，保证章节层可编辑
    """
    text = input_data.text.strip()
    sections = _split_fallback_sections(text, max_sections=4)

    chapters: list[dict[str, Any]] = []
    total_minutes = 0

    for idx, section in enumerate(sections, start=1):
        estimated_minutes = _estimate_minutes_from_text(section)
        total_minutes += estimated_minutes

        title = _guess_title_from_section(section, idx)
        summary = _summarize_section_brief(section)

        chapters.append(
            {
                "chapter_id": f"ch_{idx}",
                "order": idx,
                "title": title,
                "summary": summary,
                "estimated_minutes": estimated_minutes,
                "difficulty_level": "medium",
                "selected": True,
                "mastery_level": "unfamiliar",
                "priority_level": "medium",
                "topic_unit_count": 0,
                "source_scope": f"模块 {idx}",
                "topic_units": [],
            }
        )

    return {
        "document": {
            "document_id": input_data.document_id,
            "document_title": input_data.document_title,
            "document_summary": _summarize_section_brief(text, limit=100) or "该资料已生成最小学习地图。",
            "estimated_total_minutes": max(total_minutes, 15),
            "chapter_count": len(chapters),
            "source_parsed_name": input_data.parsed_name,
            "learning_status": "draft",
            "current_mode": "",
            "last_updated_at": "",
        },
        "chapters": chapters,
    }


def _normalize_topic_unit(raw_unit: dict[str, Any], chapter_order: int, unit_order: int) -> dict[str, Any]:
    title = _clean_text(raw_unit.get("title"), default=f"主题 {chapter_order}-{unit_order}")
    summary = _clean_text(raw_unit.get("summary"), default="该主题用于承接本章的一个核心学习单元。")

    return {
        "unit_id": _clean_text(raw_unit.get("unit_id"), default=f"u_{chapter_order}_{unit_order}"),
        "order": _clamp_positive_int(raw_unit.get("order"), unit_order),
        "title": title,
        "summary": summary,
        "estimated_minutes": _clamp_positive_int(raw_unit.get("estimated_minutes"), 20),
        "difficulty_level": _normalize_enum(raw_unit.get("difficulty_level"), ALLOWED_DIFFICULTY, "medium"),
        "selected": bool(raw_unit.get("selected", True)),
        "mastery_level": _normalize_enum(raw_unit.get("mastery_level"), ALLOWED_MASTERY, "unfamiliar"),
        "priority_level": _normalize_enum(raw_unit.get("priority_level"), ALLOWED_PRIORITY, "medium"),
        "source_scope": _clean_text(raw_unit.get("source_scope"), default=f"第{chapter_order}章-主题{unit_order}"),
    }


def _normalize_chapter(raw_chapter: dict[str, Any], order: int) -> dict[str, Any]:
    title = _clean_text(raw_chapter.get("title"), default=f"第 {order} 章")
    summary = _clean_text(raw_chapter.get("summary"), default="该章节为资料中的一个核心学习部分。")

    raw_units = raw_chapter.get("topic_units") or []
    if not isinstance(raw_units, list):
        raw_units = []

    topic_units = [
        _normalize_topic_unit(unit, chapter_order=order, unit_order=idx)
        for idx, unit in enumerate(raw_units, start=1)
        if isinstance(unit, dict)
    ]

    estimated_minutes = _clamp_positive_int(
        raw_chapter.get("estimated_minutes"),
        sum(unit["estimated_minutes"] for unit in topic_units) or 30,
    )

    return {
        "chapter_id": _clean_text(raw_chapter.get("chapter_id"), default=f"ch_{order}"),
        "order": _clamp_positive_int(raw_chapter.get("order"), order),
        "title": title,
        "summary": summary,
        "estimated_minutes": estimated_minutes,
        "difficulty_level": _normalize_enum(raw_chapter.get("difficulty_level"), ALLOWED_DIFFICULTY, "medium"),
        "selected": bool(raw_chapter.get("selected", True)),
        "mastery_level": _normalize_enum(raw_chapter.get("mastery_level"), ALLOWED_MASTERY, "unfamiliar"),
        "priority_level": _normalize_enum(raw_chapter.get("priority_level"), ALLOWED_PRIORITY, "medium"),
        "topic_unit_count": len(topic_units),
        "source_scope": _clean_text(raw_chapter.get("source_scope"), default=f"第{order}章"),
        "topic_units": topic_units,
    }


def normalize_learning_map(raw_data: dict[str, Any], input_data: LearningMapInput) -> dict[str, Any]:
    """
    把模型原始结果规范成前端可直接消费的统一结构。
    如果章节层完全不可用，抛出异常，交给上层走 fallback。
    """
    if not isinstance(raw_data, dict):
        raise LearningMapError("学习地图原始结果不是对象。")

    raw_document = raw_data.get("document") or {}
    raw_chapters = raw_data.get("chapters") or []

    if not isinstance(raw_document, dict):
        raw_document = {}
    if not isinstance(raw_chapters, list):
        raw_chapters = []

    chapters = [
        _normalize_chapter(chapter, idx)
        for idx, chapter in enumerate(raw_chapters, start=1)
        if isinstance(chapter, dict)
    ]

    if not chapters:
        raise LearningMapError("学习地图缺少可用的章节层结果。")

    estimated_total_minutes = sum(ch["estimated_minutes"] for ch in chapters)

    document = {
        "document_id": input_data.document_id,
        "document_title": _clean_text(raw_document.get("document_title"), default=input_data.document_title),
        "document_summary": _clean_text(
            raw_document.get("document_summary"),
            default="该资料已生成学习地图，可继续确认学习范围与优先级。",
        ),
        "estimated_total_minutes": _clamp_positive_int(
            raw_document.get("estimated_total_minutes"),
            max(estimated_total_minutes, 15),
        ),
        "chapter_count": len(chapters),
        "source_parsed_name": input_data.parsed_name,
        "learning_status": "draft",
        "current_mode": "",
        "last_updated_at": "",
    }

    return {
        "document": document,
        "chapters": chapters,
    }


def validate_learning_map(learning_map: dict[str, Any]) -> None:
    if not isinstance(learning_map, dict):
        raise LearningMapError("学习地图结果不是对象。")

    document = learning_map.get("document")
    chapters = learning_map.get("chapters")

    if not isinstance(document, dict):
        raise LearningMapError("学习地图缺少 document。")
    if not isinstance(chapters, list) or not chapters:
        raise LearningMapError("学习地图缺少 chapters。")

    required_document_keys = {
        "document_id",
        "document_title",
        "document_summary",
        "estimated_total_minutes",
        "chapter_count",
        "source_parsed_name",
        "learning_status",
        "current_mode",
        "last_updated_at",
    }
    missing_document_keys = required_document_keys - set(document.keys())
    if missing_document_keys:
        raise LearningMapError(f"document 缺少字段：{sorted(missing_document_keys)}")

    for chapter in chapters:
        required_chapter_keys = {
            "chapter_id",
            "order",
            "title",
            "summary",
            "estimated_minutes",
            "difficulty_level",
            "selected",
            "mastery_level",
            "priority_level",
            "topic_unit_count",
            "source_scope",
            "topic_units",
        }
        missing_chapter_keys = required_chapter_keys - set(chapter.keys())
        if missing_chapter_keys:
            raise LearningMapError(f"chapter 缺少字段：{sorted(missing_chapter_keys)}")


def build_learning_map_from_raw(
    
    input_data: LearningMapInput,
    raw_data: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    上层统一入口：
    - 有模型原始结果时：先 normalize + validate
    - 原始结果为空或不可用时：走 fallback
    """
    if raw_data is None:
        logger.warning("学习地图原始结果为空，走 fallback。")
        fallback = build_fallback_learning_map(input_data)
        validate_learning_map(fallback)
        return fallback

    try:
        normalized = normalize_learning_map(raw_data, input_data)
        validate_learning_map(normalized)
        return normalized
    except Exception as e:
        logger.warning("学习地图原始结果校验失败，走 fallback。原因：%s", e)
        fallback = build_fallback_learning_map(input_data)
        validate_learning_map(fallback)
        return fallback
    

def generate_learning_map(input_data: LearningMapInput) -> dict[str, Any]:
    """
    学习地图服务总入口：
    1. 调用大模型生成原始学习地图 JSON
    2. 做标准化、校验
    3. 失败时走 fallback
    """
    if not input_data.text or not input_data.text.strip():
        raise LearningMapError("解析文本为空，无法生成学习地图。")

    try:
        raw_data = generate_learning_map_raw(
            text=input_data.text,
            document_title=input_data.document_title,
        )
    except Exception as e:
        logger.warning("模型生成学习地图原始结果失败，走 fallback。原因：%s", e)
        raw_data = None

    return build_learning_map_from_raw(
        input_data=input_data,
        raw_data=raw_data,
    )