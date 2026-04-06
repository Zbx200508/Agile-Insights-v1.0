from pathlib import Path
from datetime import datetime
from urllib.parse import quote
import html
import json

from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import HTMLResponse

from app.routes.prototype_learning import router as prototype_learning_router
from app.services.pdf_parser import extract_text_from_pdf
from app.services.llm_service import generate_summary, generate_outline, answer_question
from app.services.retrieval import retrieve_relevant_chunks
from app.services.learning_map_service import LearningMapInput, generate_learning_map
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PARSED_DIR = DATA_DIR / "parsed"
OUTPUTS_DIR = DATA_DIR / "outputs"

for folder in [UPLOADS_DIR, PARSED_DIR, OUTPUTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="敏捷洞察1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 挂载学习流原型路由
app.include_router(prototype_learning_router)


def read_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def get_output_paths(parsed_name: str) -> tuple[Path, Path]:
    stem = Path(parsed_name).stem
    summary_path = OUTPUTS_DIR / f"{stem}_summary.txt"
    outline_path = OUTPUTS_DIR / f"{stem}_outline.txt"
    return summary_path, outline_path

def get_learning_map_path(parsed_name: str) -> Path:
    stem = Path(parsed_name).stem
    return OUTPUTS_DIR / f"{stem}_learning_map.json"

def format_datetime(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

def load_learning_map_if_exists(parsed_name: str) -> dict:
    learning_map_path = get_learning_map_path(parsed_name)
    if learning_map_path.exists():
        try:
            return json.loads(learning_map_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def get_status_label(status: str) -> str:
    mapping = {
        "draft": "待配置",
        "ready": "可开始",
        "in_progress": "学习中",
        "paused": "已暂停",
        "completed": "已完成",
    }
    return mapping.get(status, "待配置")


def get_mode_label(mode: str) -> str:
    mapping = {
        "system_learning": "系统学习",
        "quick_understanding": "速通理解",
        "": "",
    }
    return mapping.get(mode, "")

def list_uploaded_documents() -> list[dict]:
    documents = []

    learning_map_files = sorted(
        OUTPUTS_DIR.glob("*_learning_map.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    for learning_map_path in learning_map_files:
        try:
            learning_map = json.loads(learning_map_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(learning_map, dict):
            continue

        document_info = learning_map.get("document", {})
        if not isinstance(document_info, dict):
            continue

        parsed_name = document_info.get("source_parsed_name", "")
        if not parsed_name:
            continue

        title = document_info.get("document_title") or learning_map_path.stem.replace("_learning_map", "")
        learning_status = document_info.get("learning_status") or "draft"
        current_mode = document_info.get("current_mode") or ""
        estimated_total_minutes = document_info.get("estimated_total_minutes") or 0
        last_updated_at = format_datetime(learning_map_path.stat().st_mtime)

        documents.append(
            {
                "parsed_name": parsed_name,
                "document_title": title,
                "learning_status": learning_status,
                "learning_status_label": get_status_label(learning_status),
                "current_mode": current_mode,
                "current_mode_label": get_mode_label(current_mode),
                "estimated_total_minutes": estimated_total_minutes,
                "last_updated_at": last_updated_at,
            }
        )

    return documents

def build_document_workspace_context(parsed_name: str) -> dict:
    parsed_path = PARSED_DIR / parsed_name
    if not parsed_path.exists():
        raise FileNotFoundError(f"未找到解析文本：{parsed_name}")

    summary_path, outline_path = get_output_paths(parsed_name)
    summary = read_text_if_exists(summary_path)
    outline = read_text_if_exists(outline_path)

    learning_map = load_learning_map_if_exists(parsed_name)
    document_info = learning_map.get("document", {}) if isinstance(learning_map, dict) else {}

    learning_map_path = get_learning_map_path(parsed_name)
    if learning_map_path.exists():
        last_updated_at = format_datetime(learning_map_path.stat().st_mtime)
    else:
        last_updated_at = format_datetime(parsed_path.stat().st_mtime)

    title = document_info.get("document_title") or Path(parsed_name).stem
    learning_status = document_info.get("learning_status") or "draft"
    current_mode = document_info.get("current_mode") or ""
    estimated_total_minutes = document_info.get("estimated_total_minutes") or 0

    return {
            "parsed_name": parsed_name,
            "document_title": title,
            "learning_status": learning_status,
            "learning_status_label": get_status_label(learning_status),
            "current_mode": current_mode,
            "current_mode_label": get_mode_label(current_mode),
            "estimated_total_minutes": estimated_total_minutes,
            "last_updated_at": last_updated_at,
            "summary": summary,
            "outline": outline,
            "applied_plan": learning_map.get("applied_plan", {}) if isinstance(learning_map, dict) else {},
        }

def render_document_workspace(
    context: dict,
    message: str = "",
    error: str = "",
    question: str = "",
    answer: str = "",
    quotes: list[str] | None = None,
) -> str:
    quotes = quotes or []

    parsed_name = context["parsed_name"]
    document_title = context["document_title"]
    learning_status_label = context["learning_status_label"]
    current_mode_label = context["current_mode_label"]
    estimated_total_minutes = context["estimated_total_minutes"]
    last_updated_at = context["last_updated_at"]
    summary = context["summary"]
    outline = context["outline"]
    applied_plan = context.get("applied_plan", {})

    message_html = f'<div class="success">{html.escape(message)}</div>' if message else ""
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""

    mode_html = f'<span class="tag">{html.escape(current_mode_label)}</span>' if current_mode_label else ""
    time_html = f'<span class="tag">预计时长：约 {estimated_total_minutes} 分钟</span>' if estimated_total_minutes else ""

    summary_html = ""
    if summary:
        summary_html = f"""
        <div class="box">
            <h2>一页摘要</h2>
            <div class="content-box">{html.escape(summary)}</div>
        </div>
        """

    outline_html = ""
    if outline:
        outline_html = f"""
        <div class="box">
            <h2>三级逻辑大纲</h2>
            <div class="content-box">{html.escape(outline)}</div>
        </div>
        """

    quote_html = ""
    if quotes:
        quote_items = "".join(
            f'<div class="quote-box">{html.escape(q)}</div>' for q in quotes
        )
        quote_html = f"""
        <h3>引用片段</h3>
        {quote_items}
        """

    answer_html = ""
    if answer:
        answer_html = f"""
        <div class="qa-answer">
            <h3>回答结果</h3>
            <div class="content-box">{html.escape(answer)}</div>
            {quote_html}
        </div>
        """

    qa_box_html = f"""
    <div class="box">
        <h2>基于原文提问</h2>
        <p class="muted">当前问答仅基于这份文档对应的解析文本内容。</p>

        <form class="qa-form" action="/ask" method="post">
            <input type="hidden" name="parsed_name" value="{html.escape(parsed_name)}" />
            <textarea name="question" rows="4" placeholder="例如：这份材料的核心结论是什么？">{html.escape(question)}</textarea>
            <button type="submit">提交问题</button>
        </form>

        {answer_html}
    </div>
    """

    prototype_entry_html = ""
    if applied_plan:
        plan_mode = applied_plan.get("mode", "")
        plan_mode_label = get_mode_label(plan_mode)
        confirmed_at = applied_plan.get("confirmed_at", "")
        plan_summary = applied_plan.get("plan_summary", {})
        plan_detail = applied_plan.get("plan_detail", {})

        highlights = plan_summary.get("highlights", [])
        highlights_html = "".join(f'<span class="tag">{html.escape(item)}</span>' for item in highlights)

        plan_preview_html = ""

        if plan_mode == "system_learning":
            focuses = plan_detail.get("focuses", [])
            days = plan_detail.get("days", [])
            review_note = plan_detail.get("review_note", "")

            focuses_html = "".join(f"<li>{html.escape(item)}</li>" for item in focuses)
            days_html = ""
            for day in days[:2]:
                tasks_html = "".join(f"<li>{html.escape(task)}</li>" for task in day.get("tasks", []))
                days_html += f"""
                <div class="plan-subcard">
                    <strong>{html.escape(day.get("day", ""))}｜{day.get("minutes", 0)} 分钟</strong>
                    <ul>{tasks_html}</ul>
                </div>
                """

            review_html = f"<p class='muted'>{html.escape(review_note)}</p>" if review_note else ""

            plan_preview_html = f"""
            <div class="plan-subcard">
                <strong>本周重点</strong>
                <ul>{focuses_html}</ul>
            </div>
            {days_html}
            {review_html}
            """
        else:
            steps = plan_detail.get("steps", [])
            must_know = plan_detail.get("must_know", [])
            next_action = plan_detail.get("next_action", "")

            steps_html = ""
            for step in steps[:3]:
                steps_html += f"""
                <div class="plan-subcard">
                    <strong>{html.escape(step.get("title", ""))}（{step.get("minutes", 0)} 分钟）</strong>
                    <p class="muted">为什么先看：{html.escape(step.get("why", ""))}</p>
                </div>
                """

            must_know_html = "".join(f"<li>{html.escape(item)}</li>" for item in must_know)
            next_action_html = f"<p class='muted'>{html.escape(next_action)}</p>" if next_action else ""

            plan_preview_html = f"""
            {steps_html}
            <div class="plan-subcard">
                <strong>必懂概念</strong>
                <ul>{must_know_html}</ul>
            </div>
            {next_action_html}
            """

        prototype_entry_html = f"""
        <div class="box">
            <h2>当前采用方案</h2>
            <div class="tag">{html.escape(plan_mode_label)}</div>
            <p class="muted">确认时间：{html.escape(confirmed_at)}</p>
            <h3>{html.escape(plan_summary.get("title", ""))}</h3>
            <p class="muted">{html.escape(plan_summary.get("subtitle", ""))}</p>
            <div style="margin: 10px 0 16px 0;">{highlights_html}</div>
            {plan_preview_html}
            <a class="secondary-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">重新配置方案</a>
        </div>
        """
    else:
        prototype_entry_html = f"""
        <div class="box">
            <h2>学习流</h2>
            <p class="muted">你可以基于这份文档进入学习流，继续配置学习范围、模式与计划。</p>
            <a class="entry-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">配置学习方案</a>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <link rel="icon" href="/static/favicon.jpg" type="image/jpeg" />
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{html.escape(document_title)} - 敏捷洞察1.0</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 960px;
                margin: 0 auto;
                padding: 40px 20px;
                line-height: 1.6;
                background: #fafafa;
            }}
            .box {{
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 24px;
                margin-top: 24px;
                background: white;
            }}
            .tag {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: #f3f3f3;
                margin-right: 8px;
                margin-bottom: 8px;
                font-size: 14px;
            }}
            .muted {{
                color: #666;
            }}
            .entry-link {{
                display: inline-block;
                margin-top: 12px;
                padding: 10px 18px;
                border-radius: 8px;
                background: black;
                color: white;
                text-decoration: none;
            }}
            .secondary-link {{
                display: inline-block;
                margin-top: 12px;
                padding: 8px 14px;
                border-radius: 8px;
                background: #f3f4f6;
                color: #111827;
                text-decoration: none;
                font-size: 14px;
            }}
            .plan-subcard {{
                margin-top: 12px;
                padding: 14px;
                border: 1px solid #eee;
                border-radius: 10px;
                background: #fcfcfc;
            }}
            .plan-subcard ul {{
                margin: 10px 0 0 18px;
            }}
            .back-link {{
                display: inline-block;
                margin-bottom: 18px;
                color: #333;
                text-decoration: none;
                font-size: 14px;
            }}
            .qa-form button {{
                padding: 10px 18px;
                border: none;
                border-radius: 8px;
                background: black;
                color: white;
                cursor: pointer;
                margin-top: 12px;
            }}
            .qa-form textarea {{
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                box-sizing: border-box;
                resize: vertical;
            }}
            .success {{
                margin-top: 16px;
                padding: 12px 16px;
                background: #eef9f0;
                border: 1px solid #b7e4c7;
                border-radius: 8px;
                color: #1b5e20;
                white-space: pre-line;
            }}
            .error {{
                margin-top: 16px;
                padding: 12px 16px;
                background: #fff2f2;
                border: 1px solid #f5c2c2;
                border-radius: 8px;
                color: #a40000;
                white-space: pre-line;
            }}
            .content-box {{
                white-space: pre-line;
                background: #fcfcfc;
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 16px;
                margin-top: 12px;
            }}
            .quote-box {{
                white-space: pre-line;
                background: #f8f8f8;
                border-left: 4px solid #999;
                border-radius: 6px;
                padding: 12px 14px;
                margin-top: 12px;
            }}
            .qa-answer {{
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <a class="back-link" href="/">← 返回主页面</a>

        <h1>{html.escape(document_title)}</h1>
        <p class="muted">这是当前文档的工作台页面，你可以查看摘要、大纲、问答和学习流。</p>

        <div class="box">
            <h2>文档状态</h2>
            <div class="tag">{html.escape(learning_status_label)}</div>
            {mode_html}
            {time_html}
            <p class="muted">最近更新：{html.escape(last_updated_at)}</p>

            {message_html}
            {error_html}
        </div>

        {prototype_entry_html}
        {summary_html}
        {outline_html}
        {qa_box_html}
    </body>
    </html>
    """

def render_home(
    message: str = "",
    error: str = "",
    summary: str = "",
    outline: str = "",
    parsed_name: str = "",
    question: str = "",
    answer: str = "",
    quotes: list[str] | None = None,
    documents: list[dict] | None = None,
) -> str:
    quotes = quotes or []
    documents = documents or []

    message_html = f'<div class="success">{html.escape(message)}</div>' if message else ""
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""

    prototype_entry_html = ""
    if parsed_name:
        prototype_entry_html = f"""
        <div class="box">
            <h2>进入学习流</h2>
            <p class="muted">已生成学习地图，可以进入新的学习流原型继续操作。</p>
            <a class="entry-link" href="/prototype/learning">进入学习流原型</a>
        </div>
        """

    documents_html = ""
    if documents:
        items = []
        for doc in documents:
            mode_html = ""
            if doc["current_mode_label"]:
                mode_html = f'<span class="tag">{html.escape(doc["current_mode_label"])}</span>'

            time_html = ""
            if doc["estimated_total_minutes"]:
                time_html = f"<p class='muted'>预计时长：约 {doc['estimated_total_minutes']} 分钟</p>"

            items.append(f"""
            <div class="doc-card">
                <div class="doc-card-top">
                    <div>
                        <h3>{html.escape(doc["document_title"])}</h3>
                        <div class="doc-tags">
                            <span class="tag">{html.escape(doc["learning_status_label"])}</span>
                            {mode_html}
                        </div>
                    </div>
                    <a class="entry-link small-link" href="/document?parsed_name={quote(doc['parsed_name'])}">进入文档</a>
                </div>
                {time_html}
                <p class="muted">最近更新：{html.escape(doc["last_updated_at"])}</p>
            </div>
            """)

        documents_html = f"""
        <div class="box">
            <h2>已上传文档</h2>
            <p class="muted">你可以继续查看已上传资料的摘要、大纲、问答与学习流。</p>
            <div class="doc-list">
                {"".join(items)}
            </div>
        </div>
        """

    summary_html = ""
    if summary:
        summary_html = f"""
        <div class="box">
            <h2>一页摘要</h2>
            <div class="content-box">{html.escape(summary)}</div>
        </div>
        """

    outline_html = ""
    if outline:
        outline_html = f"""
        <div class="box">
            <h2>三级逻辑大纲</h2>
            <div class="content-box">{html.escape(outline)}</div>
        </div>
        """

    qa_box_html = ""
    if parsed_name:
        quote_html = ""
        if quotes:
            quote_items = "".join(
                f'<div class="quote-box">{html.escape(q)}</div>' for q in quotes
            )
            quote_html = f"""
            <h3>引用片段</h3>
            {quote_items}
            """

        answer_html = ""
        if answer:
            answer_html = f"""
            <div class="qa-answer">
                <h3>回答结果</h3>
                <div class="content-box">{html.escape(answer)}</div>
                {quote_html}
            </div>
            """

        qa_box_html = f"""
        <div class="box">
            <h2>基于原文提问</h2>
            <p class="muted">当前问答仅基于本次上传并解析的文件内容。</p>

            <form class="qa-form" action="/ask" method="post">
                <input type="hidden" name="parsed_name" value="{html.escape(parsed_name)}" />
                <textarea name="question" rows="4" placeholder="例如：这份材料的核心结论是什么？">{html.escape(question)}</textarea>
                <button type="submit">提交问题</button>
            </form>

            {answer_html}
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>敏捷洞察1.0</title>
        <link rel="icon" href="/static/favicon.jpg" type="image/jpeg" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 960px;
                margin: 0 auto;
                padding: 40px 20px;
                line-height: 1.6;
                background: #fafafa;
            }}
            .box {{
                border: 1px solid #ddd;
                border-radius: 12px;
                padding: 24px;
                margin-top: 24px;
                background: white;
            }}
            .tag {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: #f3f3f3;
                margin-right: 8px;
                margin-bottom: 8px;
                font-size: 14px;
            }}
            .muted {{
                color: #666;
            }}
            .upload-form input[type="file"] {{
                margin-bottom: 12px;
                display: block;
            }}
            .upload-form button,
            .qa-form button {{
                padding: 10px 18px;
                border: none;
                border-radius: 8px;
                background: black;
                color: white;
                cursor: pointer;
                margin-top: 12px;
            }}
            .qa-form textarea {{
                width: 100%;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                box-sizing: border-box;
                resize: vertical;
            }}
            .success {{
                margin-top: 16px;
                padding: 12px 16px;
                background: #eef9f0;
                border: 1px solid #b7e4c7;
                border-radius: 8px;
                color: #1b5e20;
                white-space: pre-line;
            }}
            .error {{
                margin-top: 16px;
                padding: 12px 16px;
                background: #fff2f2;
                border: 1px solid #f5c2c2;
                border-radius: 8px;
                color: #a40000;
                white-space: pre-line;
            }}
            .content-box {{
                white-space: pre-line;
                background: #fcfcfc;
                border: 1px solid #eee;
                border-radius: 8px;
                padding: 16px;
                margin-top: 12px;
            }}
            .quote-box {{
                white-space: pre-line;
                background: #f8f8f8;
                border-left: 4px solid #999;
                border-radius: 6px;
                padding: 12px 14px;
                margin-top: 12px;
            }}
            .qa-answer {{
                margin-top: 20px;
            }}
            .entry-link {{
                display: inline-block;
                margin-top: 12px;
                padding: 10px 18px;
                border-radius: 8px;
                background: black;
                color: white;
                text-decoration: none;
            }}
            .small-link {{
                margin-top: 0;
                padding: 8px 14px;
                font-size: 14px;
            }}
            .doc-list {{
                margin-top: 16px;
                display: flex;
                flex-direction: column;
                gap: 14px;
            }}
            .doc-card {{
                border: 1px solid #eee;
                border-radius: 12px;
                padding: 18px;
                background: #fcfcfc;
            }}
            .doc-card-top {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 16px;
            }}
            .doc-card h3 {{
                margin: 0 0 8px 0;
                font-size: 18px;
            }}
            .doc-tags {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-bottom: 8px;
            }}
                        .loading-overlay {{
                position: fixed;
                inset: 0;
                background: rgba(255, 255, 255, 0.88);
                display: none;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                backdrop-filter: blur(2px);
            }}
            .loading-overlay.active {{
                display: flex;
            }}
            .loading-card {{
                width: 320px;
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                padding: 28px 24px;
                text-align: center;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.08);
            }}
            .loading-card h3 {{
                margin: 18px 0 8px 0;
                font-size: 20px;
            }}
            .loading-card p {{
                margin: 0;
                color: #666;
                font-size: 14px;
            }}
            .book-loader {{
                position: relative;
                width: 88px;
                height: 64px;
                margin: 0 auto;
                perspective: 240px;
            }}
            .book-frame {{
                position: absolute;
                inset: 0;
                border: 4px solid #111;
                border-radius: 10px;
                background: #fff;
            }}
            .book-spine {{
                position: absolute;
                top: 8px;
                bottom: 8px;
                left: 50%;
                width: 4px;
                transform: translateX(-50%);
                background: #111;
                border-radius: 2px;
                z-index: 3;
            }}
            .book-page-left,
            .book-page-right,
            .book-page-flip {{
                position: absolute;
                top: 8px;
                bottom: 8px;
                background: #fff;
                box-shadow: inset 0 0 0 2px #111;
                border-radius: 4px;
            }}
            .book-page-left {{
                left: 8px;
                width: 28px;
                z-index: 1;
                animation: leftPageBreath 1.6s ease-in-out infinite;
            }}
            .book-page-right {{
                right: 8px;
                width: 28px;
                z-index: 1;
                animation: rightPageBreath 1.6s ease-in-out infinite;
            }}
            .book-page-flip {{
                left: 50%;
                width: 30px;
                transform-origin: left center;
                transform: translateX(-2px) rotateY(0deg);
                z-index: 2;
                animation: flipPage 1.4s ease-in-out infinite;
            }}
            @keyframes flipPage {{
                0% {{
                    transform: translateX(-2px) rotateY(0deg);
                    opacity: 1;
                }}
                50% {{
                    transform: translateX(-2px) rotateY(-170deg);
                    opacity: 0.92;
                }}
                100% {{
                    transform: translateX(-2px) rotateY(0deg);
                    opacity: 1;
                }}
            }}
            @keyframes leftPageBreath {{
                0%, 100% {{
                    transform: scaleY(1);
                }}
                50% {{
                    transform: scaleY(0.96);
                }}
            }}
            @keyframes rightPageBreath {{
                0%, 100% {{
                    transform: scaleY(1);
                }}
                50% {{
                    transform: scaleY(0.97);
                }}
            }}
            .upload-form button:disabled {{
                opacity: 0.7;
                cursor: not-allowed;
            }}
        </style>
    </head>
    <body>
        <h1>敏捷洞察1.0</h1>
        <p class="muted">面向培训资料与行业白皮书的结构化总结助手</p>

        <div class="box">
            <h2>当前版本目标</h2>
            <div class="tag">PDF 上传</div>
            <div class="tag">一页摘要</div>
            <div class="tag">三级逻辑大纲</div>
            <div class="tag">基于原文问答</div>
            <div class="tag">引用片段返回</div>
        </div>

        <div class="box">
            <h2>上传 PDF</h2>
            <p class="muted">当前阶段验证：上传 → 解析 → 摘要 → 大纲 → 问答。</p>

            <form id="uploadForm" class="upload-form" action="/upload" method="post" enctype="multipart/form-data">
                <input id="uploadFileInput" type="file" name="file" accept=".pdf" required />
                <button id="uploadSubmitBtn" type="submit">上传 PDF</button>
            </form>

            {message_html}
            {error_html}
        </div>

        {documents_html}
        {prototype_entry_html}
        {summary_html}
        {outline_html}
        {qa_box_html}

        <div class="box">
            <h2>当前开发进度</h2>
            <p>FastAPI 骨架、上传、PDF 文本解析、一页摘要、三级逻辑大纲、基于原文问答链路已接入。</p>
            <p>下一步将继续优化引用质量和上线部署。</p>
        </div>
                <div id="uploadLoadingOverlay" class="loading-overlay">
            <div class="loading-card">
                <div class="book-loader">
                    <div class="book-frame"></div>
                    <div class="book-page-left"></div>
                    <div class="book-page-right"></div>
                    <div class="book-page-flip"></div>
                    <div class="book-spine"></div>
                </div>
                <h3>正在处理资料，请稍候…</h3>
                <p id="loadingSubtext">上传文件中</p>
            </div>
        </div>

        <script>
            const uploadForm = document.getElementById("uploadForm");
            const uploadFileInput = document.getElementById("uploadFileInput");
            const uploadSubmitBtn = document.getElementById("uploadSubmitBtn");
            const uploadLoadingOverlay = document.getElementById("uploadLoadingOverlay");
            const loadingSubtext = document.getElementById("loadingSubtext");

            if (uploadForm && uploadFileInput && uploadSubmitBtn && uploadLoadingOverlay && loadingSubtext) {{
                uploadForm.addEventListener("submit", function (event) {{
                    if (!uploadFileInput.files || uploadFileInput.files.length === 0) {{
                        return;
                    }}

                    const messages = [
                        "上传文件中",
                        "解析文档中",
                        "生成学习地图中",
                        "整理摘要与大纲中"
                    ];

                    let index = 0;
                    loadingSubtext.textContent = messages[index];

                    uploadLoadingOverlay.classList.add("active");
                    uploadSubmitBtn.disabled = true;
                    uploadSubmitBtn.textContent = "处理中...";

                    window.__uploadLoadingTimer = setInterval(function () {{
                        index = (index + 1) % messages.length;
                        loadingSubtext.textContent = messages[index];
                    }}, 1200);
                }});
            }}
        </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return render_home(documents=list_uploaded_documents())

@app.get("/document", response_class=HTMLResponse)
def document_workspace(
    parsed_name: str = Query(...),
    message: str = Query(default="")
):
    try:
        context = build_document_workspace_context(parsed_name)
        return render_document_workspace(
            context,
            message=message,
        )
    except FileNotFoundError:
        return render_home(
            error="未找到对应文档，请从主页面重新进入。",
            documents=list_uploaded_documents(),
        )

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "敏捷洞察1.0",
        "message": "服务运行正常"
    }

@app.post("/upload", response_class=HTMLResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename:
        return render_home(error="未检测到文件，请重新选择 PDF。")

    if not file.filename.lower().endswith(".pdf"):
        return render_home(error="当前仅支持上传 PDF 文件。")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(file.filename).name
    save_name = f"{timestamp}_{safe_name}"
    save_path = UPLOADS_DIR / save_name

    try:
        file_bytes = await file.read()
        save_path.write_bytes(file_bytes)

        parse_result = extract_text_from_pdf(save_path)

        parsed_name = save_path.stem + ".txt"
        parsed_path = PARSED_DIR / parsed_name
        parsed_path.write_text(parse_result["text"], encoding="utf-8")

        summary = generate_summary(parse_result["text"])
        outline = generate_outline(parse_result["text"])

        learning_map_input = LearningMapInput(
            document_id=save_path.stem,
            document_title=Path(file.filename).stem,
            parsed_name=parsed_name,
            text=parse_result["text"],
        )
        learning_map = generate_learning_map(learning_map_input)

        summary_name = save_path.stem + "_summary.txt"
        summary_path = OUTPUTS_DIR / summary_name
        summary_path.write_text(summary, encoding="utf-8")

        outline_name = save_path.stem + "_outline.txt"
        outline_path = OUTPUTS_DIR / outline_name
        outline_path.write_text(outline, encoding="utf-8")

        learning_map_path = get_learning_map_path(parsed_name)
        learning_map_path.write_text(
            json.dumps(learning_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        learning_map_name = learning_map_path.name

        size_kb = round(len(file_bytes) / 1024, 2)

        message = (
            f"上传成功：{save_name}\n"
            f"文件大小：约 {size_kb} KB\n"
            f"总页数：{parse_result['page_count']} 页\n"
            f"有文本页数：{parse_result['non_empty_pages']} 页\n"
            f"解析字符数：{parse_result['char_count']} 字\n"
            f"解析文本已保存：{parsed_name}\n"
            f"摘要已保存：{summary_name}\n"
            f"大纲已保存：{outline_name}\n"
            f"学习地图已保存：{learning_map_name}"
        )
        context = build_document_workspace_context(parsed_name)
        return render_document_workspace(
            context,
            message=message,
        )

    except Exception as e:
        return render_home(error=f"上传、解析、摘要或大纲生成失败：{str(e)}",
                           documents=list_uploaded_documents(),
        )


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(
    parsed_name: str = Form(...),
    question: str = Form(...),
):
    try:
        parsed_path = PARSED_DIR / parsed_name
        if not parsed_path.exists():
            return render_home(error="未找到对应的解析文本，请重新上传 PDF。")

        text = parsed_path.read_text(encoding="utf-8")
        summary_path, outline_path = get_output_paths(parsed_name)
        summary = read_text_if_exists(summary_path)
        outline = read_text_if_exists(outline_path)

        relevant_chunks = retrieve_relevant_chunks(question, text, top_k=3)
        answer = answer_question(question, relevant_chunks)

        context = build_document_workspace_context(parsed_name)
        return render_document_workspace(
            context,
            question=question,
            answer=answer,
            quotes=relevant_chunks[:2],
            message=f"当前问答基于：{parsed_name}",
        )

    except Exception as e:
        try:
            context = build_document_workspace_context(parsed_name)
            return render_document_workspace(
                context,
                question=question,
                error=f"问答失败：{str(e)}",
            )
        except Exception:
            return render_home(
                error=f"问答失败：{str(e)}",
                documents=list_uploaded_documents(),
            )