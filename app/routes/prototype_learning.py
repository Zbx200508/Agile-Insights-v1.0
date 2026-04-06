from pathlib import Path
import json

from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from urllib.parse import quote

from app.services.system_learning_plan_service import generate_system_learning_plan
from app.services.quick_understanding_plan_service import generate_quick_understanding_plan

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"


def _difficulty_label(value: str) -> str:
    mapping = {
        "low": "低",
        "medium": "中",
        "high": "高",
    }
    return mapping.get(value, "中")


def _mastery_label(value: str) -> str:
    mapping = {
        "mastered": "已掌握",
        "familiar": "了解一些",
        "unfamiliar": "几乎不会",
    }
    return mapping.get(value, "几乎不会")


def _priority_label(value: str) -> str:
    mapping = {
        "low": "低",
        "medium": "中",
        "high": "高",
    }
    return mapping.get(value, "中")


def _load_latest_learning_map() -> tuple[dict | None, str]:
    json_files = sorted(
        OUTPUTS_DIR.glob("*_learning_map.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not json_files:
        return None, ""

    latest_file = json_files[0]
    data = json.loads(latest_file.read_text(encoding="utf-8"))
    return data, latest_file.name


def _load_learning_map_by_parsed_name(parsed_name: str) -> tuple[dict | None, str]:
    if not parsed_name:
        return None, ""

    learning_map_path = OUTPUTS_DIR / f"{Path(parsed_name).stem}_learning_map.json"
    if not learning_map_path.exists():
        return None, ""

    try:
        data = json.loads(learning_map_path.read_text(encoding="utf-8"))
        return data, learning_map_path.name
    except Exception:
        return None, ""


def _build_mock_applied_plan(mode: str, confirmed_at: str) -> dict:
    if mode == "system_learning":
        return {
            "mode": mode,
            "confirmed_at": confirmed_at,
            "plan_summary": {
                "title": "本周学习计划",
                "subtitle": "优先安排高优先级且未掌握内容，后半周插入轻复习任务。",
                "highlights": [
                    "预计总投入：220 分钟",
                    "建议日均：30 分钟",
                ],
            },
            "plan_detail": {
                "focuses": [
                    "理解大模型基础概念",
                    "掌握 RAG 与 Agent 的区别",
                    "建立文档理解产品闭环认知",
                ],
                "days": [
                    {
                        "day": "Day1",
                        "minutes": 35,
                        "tasks": [
                            "新学：学习大模型是什么（20 分钟）",
                            "复习：回顾 AI 产品经理职责边界（15 分钟）",
                        ],
                    },
                    {
                        "day": "Day2",
                        "minutes": 30,
                        "tasks": [
                            "新学：学习 Prompt、RAG、Agent 的区别（30 分钟）",
                        ],
                    },
                ],
                "review_note": "Day3：复习大模型基础概念，防止遗忘，巩固整体框架。",
            },
        }

    return {
        "mode": mode,
        "confirmed_at": confirmed_at,
        "plan_summary": {
            "title": "最短理解路径",
            "subtitle": "在有限时间内抓住这份资料最核心的框架与关键概念。",
            "highlights": [
                "推荐投入：30 分钟",
                "核心内容：3 项",
            ],
        },
        "plan_detail": {
            "steps": [
                {
                    "title": "先看：大模型是什么",
                    "minutes": 10,
                    "why": "这是整份资料的概念底座。",
                },
                {
                    "title": "再看：Prompt、RAG、Agent 的区别",
                    "minutes": 12,
                    "why": "这是最容易混淆、但最关键的知识点。",
                },
                {
                    "title": "最后看：文档理解产品闭环",
                    "minutes": 8,
                    "why": "帮助你把前面的概念落到真实产品链路中。",
                },
            ],
            "must_know": [
                "大模型",
                "Prompt",
                "RAG",
                "Agent",
            ],
            "next_action": "如果希望系统学习，建议保留当前高优先级内容生成一周计划。",
        },
    }


def _apply_learning_plan(parsed_name: str, mode: str, week_minutes: dict | None = None) -> bool:
    if not parsed_name:
        return False

    if mode not in {"system_learning", "quick_understanding"}:
        return False

    learning_map_path = OUTPUTS_DIR / f"{Path(parsed_name).stem}_learning_map.json"
    if not learning_map_path.exists():
        return False

    try:
        data = json.loads(learning_map_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    document = data.get("document")
    if not isinstance(document, dict):
        return False

    confirmed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if document.get("learning_status", "draft") == "draft":
        document["learning_status"] = "ready"

    document["current_mode"] = mode
    document["last_updated_at"] = confirmed_at

    if mode == "system_learning":
        data["applied_plan"] = generate_system_learning_plan(
            learning_map=data,
            week_minutes=week_minutes or {},
            confirmed_at=confirmed_at,
        )
    else:
        try:
            target_minutes = int((week_minutes or {}).get("target_minutes", 30))
        except Exception:
            target_minutes = 30

        data["applied_plan"] = generate_quick_understanding_plan(
            learning_map=data,
            target_minutes=target_minutes,
            confirmed_at=confirmed_at,
        )

    learning_map_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True


def _build_summary_stats(learning_map: dict) -> dict:
    chapters = learning_map.get("chapters", [])

    chapter_count = len(chapters)
    selected_count = sum(1 for ch in chapters if ch.get("selected") is True)
    selected_minutes = sum(
        int(ch.get("estimated_minutes", 0))
        for ch in chapters
        if ch.get("selected") is True
    )
    high_priority_count = sum(
        1 for ch in chapters if ch.get("priority_level") == "high"
    )

    return {
        "chapter_count": chapter_count,
        "selected_count": selected_count,
        "selected_minutes": selected_minutes,
        "high_priority_count": high_priority_count,
    }


def _build_display_learning_map(learning_map: dict) -> dict:
    document = learning_map.get("document", {})
    chapters = learning_map.get("chapters", [])

    display_chapters = []
    for chapter in chapters:
        topic_units = chapter.get("topic_units", [])
        display_topic_units = []

        for unit in topic_units:
            display_topic_units.append(
                {
                    **unit,
                    "difficulty_label": _difficulty_label(unit.get("difficulty_level", "medium")),
                    "mastery_label": _mastery_label(unit.get("mastery_level", "unfamiliar")),
                    "priority_label": _priority_label(unit.get("priority_level", "medium")),
                }
            )

        display_chapters.append(
            {
                **chapter,
                "difficulty_label": _difficulty_label(chapter.get("difficulty_level", "medium")),
                "mastery_label": _mastery_label(chapter.get("mastery_level", "unfamiliar")),
                "priority_label": _priority_label(chapter.get("priority_level", "medium")),
                "topic_unit_count": len(topic_units),
                "topic_units": display_topic_units,
            }
        )

    return {
        "document": document,
        "chapters": display_chapters,
    }


@router.get("/prototype/learning", response_class=HTMLResponse)
async def prototype_learning(request: Request, parsed_name: str = ""):
    if parsed_name:
        learning_map_path = OUTPUTS_DIR / f"{Path(parsed_name).stem}_learning_map.json"
        if learning_map_path.exists():
            raw_learning_map = json.loads(learning_map_path.read_text(encoding="utf-8"))
            learning_map_filename = learning_map_path.name
        else:
            raw_learning_map, learning_map_filename = _load_latest_learning_map()
    else:
        raw_learning_map, learning_map_filename = _load_latest_learning_map()

    if raw_learning_map:
        learning_map = _build_display_learning_map(raw_learning_map)
        summary_stats = _build_summary_stats(raw_learning_map)
    else:
        learning_map = None
        summary_stats = {
            "chapter_count": 0,
            "selected_count": 0,
            "selected_minutes": 0,
            "high_priority_count": 0,
        }

    back_link = f"/document?parsed_name={quote(parsed_name)}" if parsed_name else "/"

    return templates.TemplateResponse(
        request,
        "learning_flow_prototype.html",
        {
            "request": request,
            "learning_map": learning_map,
            "summary_stats": summary_stats,
            "learning_map_filename": learning_map_filename,
            "parsed_name": parsed_name,
            "back_link": back_link,
        },
    )

@router.post("/prototype/apply-plan")
async def prototype_apply_plan(request: Request):
    try:
        payload = await request.json()
        parsed_name = str(payload.get("parsed_name", "")).strip()
        mode = str(payload.get("mode", "")).strip()
        week_minutes = payload.get("week_minutes", {})

        if not parsed_name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "parsed_name 不能为空"},
            )

        if mode not in {"system_learning", "quick_understanding"}:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "mode 非法"},
            )

        success = _apply_learning_plan(
            parsed_name=parsed_name,
            mode=mode,
            week_minutes=week_minutes,
        )

        return {"success": success}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"方案确认失败：{str(e)}"},
        )
    
@router.post("/prototype/preview-system-plan")
async def prototype_preview_system_plan(request: Request):
    try:
        payload = await request.json()
        parsed_name = str(payload.get("parsed_name", "")).strip()
        week_minutes = payload.get("week_minutes", {})

        if not parsed_name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "parsed_name 不能为空"},
            )

        learning_map, _ = _load_learning_map_by_parsed_name(parsed_name)
        if not learning_map:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "未找到对应 learning_map"},
            )

        preview_plan = generate_system_learning_plan(
            learning_map=learning_map,
            week_minutes=week_minutes or {},
            confirmed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        return {"success": True, "data": preview_plan}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"系统学习预览生成失败：{str(e)}"},
        )

@router.post("/prototype/preview-quick-plan")
async def prototype_preview_quick_plan(request: Request):
    try:
        payload = await request.json()
        parsed_name = str(payload.get("parsed_name", "")).strip()
        target_minutes = int(payload.get("target_minutes", 30))

        if not parsed_name:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "parsed_name 不能为空"},
            )

        learning_map, _ = _load_learning_map_by_parsed_name(parsed_name)
        if not learning_map:
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "未找到对应 learning_map"},
            )

        preview_plan = generate_quick_understanding_plan(
            learning_map=learning_map,
            target_minutes=target_minutes,
            confirmed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        return {"success": True, "data": preview_plan}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"速通理解预览生成失败：{str(e)}"},
        )