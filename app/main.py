from pathlib import Path
from datetime import datetime
from urllib.parse import quote
import html
import json

from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import HTMLResponse

from app.routes.prototype_learning import router as prototype_learning_router
from app.services.pdf_parser import extract_text_from_pdf
from app.services.chunking_service import build_chunks_from_parsed_text, save_chunks
from app.services.vector_store_service import build_and_save_document_index
from app.services.llm_service import generate_summary, generate_outline
from app.services.learning_map_service import LearningMapInput, generate_learning_map
from app.services.rag_service import answer_question_with_rag
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

def get_chunks_path(parsed_name: str) -> Path:
    stem = Path(parsed_name).stem
    return OUTPUTS_DIR / f"{stem}_chunks.json"

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
    citations: list[dict] | None = None,
) -> str:
    quotes = quotes or []
    citations = citations or []

    parsed_name = context["parsed_name"]
    document_title = context["document_title"]
    learning_status_label = context["learning_status_label"]
    current_mode_label = context["current_mode_label"]
    estimated_total_minutes = context["estimated_total_minutes"]
    last_updated_at = context["last_updated_at"]
    summary = context["summary"]
    outline = context["outline"]
    applied_plan = context.get("applied_plan", {})

    technical_details_html = ""
    if message:
        technical_details_html = f"""
        <details class="technical-details">
            <summary>查看技术细节</summary>
            <div>{html.escape(message)}</div>
        </details>
        """
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""

    mode_display = current_mode_label or "待配置"
    time_display = f"约 {estimated_total_minutes} 分钟" if estimated_total_minutes else "待评估"
    map_ready = get_learning_map_path(parsed_name).exists()

    status_items_html = f"""
    <div class="status-item">
        <span>当前状态</span>
        <strong>{html.escape(learning_status_label)}</strong>
    </div>
    <div class="status-item">
        <span>当前模式</span>
        <strong>{html.escape(mode_display)}</strong>
    </div>
    <div class="status-item">
        <span>预计学习时长</span>
        <strong>{html.escape(time_display)}</strong>
    </div>
    <div class="status-item">
        <span>最近更新</span>
        <strong>{html.escape(last_updated_at)}</strong>
    </div>
    """

    ready_count = sum([bool(summary), bool(outline), bool(map_ready)])
    readiness_hint = (
        "摘要 / 大纲 / 学习地图已就绪，可为当前方案服务。"
        if ready_count == 3
        else "辅助内容正在准备，已生成内容会作为当前方案的支撑视图。"
    )
    content_readiness_html = f"""
    <div class="content-readiness">
        <div class="readiness-copy">
            <span class="readiness-label">辅助内容状态</span>
            <span>{readiness_hint}</span>
        </div>
        <div class="overview-checks compact">
            <span class="check-pill">摘要：{'已生成' if summary else '未生成'}</span>
            <span class="check-pill">大纲：{'已生成' if outline else '未生成'}</span>
            <span class="check-pill">学习地图：{'已生成' if map_ready else '未生成'}</span>
        </div>
        {technical_details_html}
        {error_html}
    </div>
    """

    quote_html = ""
    if citations:
        quote_items = "".join(
            f"""
            <div class="quote-box">
                <strong>{html.escape(str(item.get("chunk_id", "")))}</strong>
                <div>页码：{html.escape(str(item.get("page_range") or "unknown"))}</div>
                <div>章节：{html.escape(str(item.get("chapter_title") or ""))}</div>
                <div>范围：{html.escape(str(item.get("source_scope") or ""))}</div>
                <div>{html.escape(str(item.get("quote") or ""))}</div>
            </div>
            """
            for item in citations
        )
        quote_html = f"""
        <div class="quote-section">
            <h3>引用依据</h3>
            {quote_items}
        </div>
        """
    elif quotes:
        quote_items = "".join(
            f'<div class="quote-box">{html.escape(q)}</div>' for q in quotes
        )
        quote_html = f"""
        <div class="quote-section">
            <h3>引用片段</h3>
            {quote_items}
        </div>
        """

    answer_html = ""
    if answer:
        answer_html = f"""
        <div class="qa-answer result-card">
            <div class="result-heading">
                <p class="eyebrow">回答结果</p>
                <h3>基于原文的回答</h3>
            </div>
            <div class="content-box">{html.escape(answer)}</div>
            {quote_html}
        </div>
        """
    else:
        answer_html = """
        <div class="qa-empty-note">
            <p>围绕当前学习方案提问，优先澄清核心概念与执行疑问。回答会依据当前文档的解析文本生成。</p>
        </div>
        """

    qa_panel_body_html = f"""
        <div class="qa-workspace">
            <div class="qa-input-card">
                <p class="muted">围绕当前学习方案提问，优先澄清核心概念与执行疑问。</p>

                <form class="qa-form" action="/ask" method="post">
                    <input type="hidden" name="parsed_name" value="{html.escape(parsed_name)}" />
                    <textarea name="question" rows="4" placeholder="例如：当前方案里我应该先澄清哪个概念？">{html.escape(question)}</textarea>
                    <div class="qa-actions">
                        <button type="submit">提交问题</button>
                        <span>提交后会保留当前文档上下文。</span>
                    </div>
                </form>
            </div>

            {answer_html}
        </div>
    """

    plan_overview_html = ""
    primary_action_html = ""
    plan_detail_panel_html = ""
    if applied_plan:
        plan_mode = applied_plan.get("mode", "")
        plan_mode_label = get_mode_label(plan_mode)
        confirmed_at = applied_plan.get("confirmed_at", "")
        plan_summary = applied_plan.get("plan_summary", {}) or {}
        plan_detail = applied_plan.get("plan_detail", {}) or {}

        plan_title = plan_summary.get("title", "") or "当前采用方案"
        plan_goal = plan_summary.get("subtitle", "") or "围绕当前文档继续推进学习计划。"
        highlights = plan_summary.get("highlights", [])
        highlights_html = "".join(f'<span class="tag">{html.escape(item)}</span>' for item in highlights)
        plan_detail_highlights_html = "".join(f'<span class="check-pill">{html.escape(item)}</span>' for item in highlights)
        plan_highlights_block_html = f'<div class="plan-highlights">{highlights_html}</div>' if highlights_html else ""
        plan_detail_tags_html = f'<div class="overview-checks">{plan_detail_highlights_html}</div>' if plan_detail_highlights_html else ""

        recommended_action = plan_detail.get("next_action", "") or "先查看摘要，再围绕核心概念发起提问。"
        if plan_mode == "system_learning":
            first_day = ""
            plan_days = plan_detail.get("days", [])
            if plan_days:
                first_day = plan_days[0].get("day", "")
            recommended_action = f"进入本周学习计划，先完成{first_day or '周一'}内容。"
        if not recommended_action.startswith("下一步"):
            recommended_action = f"下一步：{recommended_action}"

        plan_time_html = f"<span>{html.escape(time_display)}</span>" if estimated_total_minutes else "<span>时间预算待评估</span>"
        confirmed_html = f"<span>确认时间：{html.escape(confirmed_at)}</span>" if confirmed_at else ""
        primary_action_html = f"""
        <a class="entry-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">继续学习</a>
        """
        plan_actions_html = f"""
        <div class="plan-actions">
            <a class="entry-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">继续学习</a>
            <a class="secondary-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">重新配置方案</a>
        </div>
        """
        plan_overview_html = f"""
        <section class="plan-overview">
            <div class="plan-overview-main">
                <p class="eyebrow">当前任务中心</p>
                <h2>{html.escape(plan_title)}</h2>
                <div class="plan-meta-line">
                    <span>{html.escape(plan_mode_label or mode_display)}</span>
                    {plan_time_html}
                    {confirmed_html}
                </div>
                {plan_highlights_block_html}
                <div class="plan-goal">
                    <span>核心目标</span>
                    <strong>{html.escape(plan_goal)}</strong>
                </div>
            </div>
            <div class="plan-overview-action">
                <span class="readiness-label">推荐动作</span>
                <p class="next-step-copy">{html.escape(recommended_action)}</p>
                <a class="secondary-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">重新配置方案</a>
            </div>
        </section>
        """

        if plan_mode == "system_learning":
            focuses = plan_detail.get("focuses", [])
            days = plan_detail.get("days", [])
            review_note = plan_detail.get("review_note", "")

            focuses_html = "".join(f"<li>{html.escape(item)}</li>" for item in focuses)
            days_html = ""
            for day in days:
                tasks_html = "".join(f"<li>{html.escape(task)}</li>" for task in day.get("tasks", []))
                days_html += f"""
                <article class="detail-card">
                    <h3>{html.escape(day.get("day", ""))}</h3>
                    <p class="muted">约 {day.get("minutes", 0)} 分钟</p>
                    <ul>{tasks_html}</ul>
                </article>
                """

            review_html = f"<p class='muted'>{html.escape(review_note)}</p>" if review_note else ""
            plan_detail_panel_html = f"""
            <div class="tab-section">
                <p class="eyebrow">当前方案 / 执行面板</p>
                <h2>{html.escape(plan_title)}</h2>
                <div class="plan-facts">
                    <div><span>模式</span><strong>{html.escape(plan_mode_label or mode_display)}</strong></div>
                    <div><span>总时长</span><strong>{html.escape(time_display)}</strong></div>
                    <div><span>确认时间</span><strong>{html.escape(confirmed_at or "未记录")}</strong></div>
                    <div><span>方案目标</span><strong>{html.escape(plan_goal)}</strong></div>
                </div>
                {plan_detail_tags_html}
            </div>
            <div class="detail-grid">
                <article class="detail-card">
                    <h3>本周重点</h3>
                    <ul>{focuses_html}</ul>
                </article>
                {days_html}
            </div>
            {review_html}
            {plan_actions_html}
            """
        else:
            steps = plan_detail.get("steps", [])
            must_know = plan_detail.get("must_know", [])
            next_action = plan_detail.get("next_action", "")

            steps_html = ""
            for step in steps:
                steps_html += f"""
                <article class="detail-card">
                    <h3>{html.escape(step.get("title", ""))}</h3>
                    <p class="muted">约 {step.get("minutes", 0)} 分钟</p>
                    <p>{html.escape(step.get("why", ""))}</p>
                </article>
                """

            must_know_html = "".join(f"<li>{html.escape(item)}</li>" for item in must_know)
            next_action_html = f"<p class='muted'>{html.escape(next_action)}</p>" if next_action else ""
            plan_detail_panel_html = f"""
            <div class="tab-section">
                <p class="eyebrow">当前方案 / 执行面板</p>
                <h2>{html.escape(plan_title)}</h2>
                <div class="plan-facts">
                    <div><span>模式</span><strong>{html.escape(plan_mode_label or mode_display)}</strong></div>
                    <div><span>总时长</span><strong>{html.escape(time_display)}</strong></div>
                    <div><span>确认时间</span><strong>{html.escape(confirmed_at or "未记录")}</strong></div>
                    <div><span>方案目标</span><strong>{html.escape(plan_goal)}</strong></div>
                </div>
                {plan_detail_tags_html}
            </div>
            <div class="detail-grid">
                <article class="detail-card">
                    <h3>阶段重点</h3>
                    <ul>{must_know_html}</ul>
                </article>
                {steps_html}
            </div>
            {next_action_html}
            {plan_actions_html}
            """
    else:
        primary_action_html = f"""
        <a class="entry-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">配置学习方案</a>
        """
        plan_overview_html = f"""
        <section class="plan-overview">
            <div class="plan-overview-main">
                <p class="eyebrow">当前任务中心</p>
                <h2>尚未配置学习方案</h2>
                <div class="plan-meta-line">
                    <span>{html.escape(mode_display)}</span>
                    <span>{html.escape(time_display)}</span>
                </div>
                <div class="plan-goal">
                    <span>核心目标</span>
                    <strong>先确认采用方案，让摘要、大纲、问答和助手入口围绕方案服务。</strong>
                </div>
            </div>
            <div class="plan-overview-action">
                <span class="readiness-label">推荐动作</span>
                <p class="next-step-copy">下一步：配置学习方案，确认模式与时间预算。</p>
                <a class="secondary-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">配置学习方案</a>
            </div>
        </section>
        """
        plan_detail_panel_html = f"""
        <div class="empty-state">
            <p class="eyebrow">当前方案 / 执行面板</p>
            <h2>尚未配置学习方案</h2>
            <p class="muted">配置后，这里会展示更完整的学习模式、时间预算和计划详情。</p>
            <div class="plan-facts">
                <div><span>模式</span><strong>{html.escape(mode_display)}</strong></div>
                <div><span>总时长</span><strong>{html.escape(time_display)}</strong></div>
                <div><span>状态</span><strong>{html.escape(learning_status_label)}</strong></div>
                <div><span>方案目标</span><strong>建立可执行的学习路径</strong></div>
            </div>
            <a class="entry-link" href="/prototype/learning?parsed_name={quote(parsed_name)}">配置学习方案</a>
        </div>
        """

    active_tab = "qa" if question or answer else "plan"

    def tab_button_class(tab_name: str) -> str:
        return "tab-button active" if active_tab == tab_name else "tab-button"

    def tab_panel_class(tab_name: str) -> str:
        return "tab-panel active" if active_tab == tab_name else "tab-panel"

    summary_panel_body_html = (
        f'<div class="reading-box">{html.escape(summary)}</div>'
        if summary
        else '<div class="empty-state"><h2>摘要暂未生成</h2><p class="muted">上传并解析文档后，这里会展示一页摘要。</p></div>'
    )
    outline_panel_body_html = (
        f'<div class="reading-box">{html.escape(outline)}</div>'
        if outline
        else '<div class="empty-state"><h2>大纲暂未生成</h2><p class="muted">上传并解析文档后，这里会展示三级逻辑大纲。</p></div>'
    )

    workspace_tabs_html = f"""
    <section class="workspace-tabs" aria-label="文档工作台内容区">
        <div class="tab-nav" role="tablist" aria-label="内容切换">
            <button class="{tab_button_class('plan')}" type="button" role="tab" aria-selected="{'true' if active_tab == 'plan' else 'false'}" aria-controls="panel-plan" data-tab-target="plan">当前方案</button>
            <button class="{tab_button_class('summary')}" type="button" role="tab" aria-selected="{'true' if active_tab == 'summary' else 'false'}" aria-controls="panel-summary" data-tab-target="summary">摘要</button>
            <button class="{tab_button_class('outline')}" type="button" role="tab" aria-selected="{'true' if active_tab == 'outline' else 'false'}" aria-controls="panel-outline" data-tab-target="outline">大纲</button>
            <button class="{tab_button_class('qa')}" type="button" role="tab" aria-selected="{'true' if active_tab == 'qa' else 'false'}" aria-controls="panel-qa" data-tab-target="qa">原文问答</button>
            <button class="{tab_button_class('agent')}" type="button" role="tab" aria-selected="{'true' if active_tab == 'agent' else 'false'}" aria-controls="panel-agent" data-tab-target="agent">Agent 助手</button>
        </div>
        <div class="tab-content">
            <section id="panel-plan" class="{tab_panel_class('plan')}" role="tabpanel" data-tab-panel="plan">
                {plan_detail_panel_html}
            </section>
            <section id="panel-summary" class="{tab_panel_class('summary')}" role="tabpanel" data-tab-panel="summary">
                <div class="tab-section">
                    <p class="eyebrow">摘要</p>
                    <h2>一页摘要</h2>
                    <p class="muted">用于建立当前方案所需的整体认知，需要补齐背景时再进入这里。</p>
                </div>
                {summary_panel_body_html}
            </section>
            <section id="panel-outline" class="{tab_panel_class('outline')}" role="tabpanel" data-tab-panel="outline">
                <div class="tab-section">
                    <p class="eyebrow">大纲</p>
                    <h2>三级逻辑大纲</h2>
                    <p class="muted">用于定位文档结构，并找到与当前方案对应的章节和内容脉络。</p>
                </div>
                {outline_panel_body_html}
            </section>
            <section id="panel-qa" class="{tab_panel_class('qa')}" role="tabpanel" data-tab-panel="qa">
                <div class="tab-section">
                    <p class="eyebrow">原文问答</p>
                    <h2>基于原文提问</h2>
                    <p class="muted">围绕当前学习方案提问，优先澄清核心概念与执行疑问。</p>
                </div>
                {qa_panel_body_html}
            </section>
            <section id="panel-agent" class="{tab_panel_class('agent')}" role="tabpanel" data-tab-panel="agent">
                <div class="empty-state agent-preview">
                    <p class="eyebrow">Agent 助手</p>
                    <h2>方案推进助手建设中</h2>
                    <p class="muted">后续将围绕当前方案拆解任务、澄清阻塞点，并调用相关能力输出可执行结果。</p>
                    <div class="capability-list">
                        <div><strong>目标输入</strong><span>围绕当前方案输入要完成的任务、疑问或交付物。</span></div>
                        <div><strong>方案感知</strong><span>基于当前方案、摘要、大纲和解析文本组织任务上下文。</span></div>
                        <div><strong>能力编排</strong><span>按推进需要调用问答、总结、计划等能力，输出可继续编辑的结果。</span></div>
                    </div>
                </div>
            </section>
        </div>
    </section>
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
                max-width: 1120px;
                margin: 0 auto;
                padding: 40px 20px 56px;
                line-height: 1.6;
                background: #111315;
                color: #F3F4F6;
            }}
            .box {{
                border: 1px solid #2A2F36;
                border-radius: 8px;
                padding: 24px;
                margin-top: 24px;
                background: #171A1F;
            }}
            h1, h2, h3, p {{
                margin-top: 0;
            }}
            h1 {{
                margin-bottom: 6px;
                font-size: 32px;
                line-height: 1.2;
            }}
            h2 {{
                font-size: 20px;
                margin-bottom: 10px;
            }}
            .tag {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: #20252C;
                color: #D7DCE4;
                margin-right: 8px;
                margin-bottom: 8px;
                font-size: 14px;
                border: 1px solid #343A43;
            }}
            .muted {{
                color: #A7AFBC;
            }}
            .entry-link {{
                display: inline-block;
                padding: 11px 18px;
                border-radius: 8px;
                background: #F3F4F6;
                color: #111315;
                text-decoration: none;
                font-weight: 700;
            }}
            .entry-link:hover {{
                background: #E5E7EB;
            }}
            .secondary-link {{
                display: inline-block;
                margin-top: 12px;
                padding: 8px 14px;
                border-radius: 8px;
                background: transparent;
                color: #E5E7EB;
                text-decoration: none;
                font-size: 14px;
                border: 1px solid #343A43;
            }}
            .secondary-link:hover {{
                background: #20252C;
            }}
            .back-link {{
                display: inline-block;
                color: #A7AFBC;
                text-decoration: none;
                font-size: 14px;
            }}
            .workspace-hero {{
                border: 1px solid #2A2F36;
                border-radius: 8px;
                padding: 24px;
                background: #171A1F;
                box-shadow: 0 18px 36px rgba(0, 0, 0, 0.18);
            }}
            .hero-top {{
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 24px;
            }}
            .hero-title-group {{
                min-width: 0;
            }}
            .hero-title-group .muted {{
                margin-bottom: 0;
            }}
            .hero-actions {{
                flex: 0 0 auto;
                padding-top: 2px;
            }}
            .status-strip {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin-top: 22px;
            }}
            .status-item {{
                min-height: 74px;
                padding: 14px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #1D2127;
            }}
            .status-item span {{
                display: block;
                color: #7C8491;
                font-size: 13px;
                margin-bottom: 6px;
            }}
            .status-item strong {{
                display: block;
                color: #F3F4F6;
                font-size: 16px;
                line-height: 1.35;
            }}
            .plan-overview {{
                display: grid;
                grid-template-columns: minmax(0, 1.6fr) minmax(240px, 0.8fr);
                gap: 18px;
                margin-top: 18px;
                padding: 20px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #1D2127;
            }}
            .plan-overview-main p,
            .plan-overview-action p {{
                color: #A7AFBC;
                margin-bottom: 0;
            }}
            .plan-overview h2 {{
                color: #F3F4F6;
                margin-bottom: 8px;
            }}
            .plan-meta-line {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 14px;
                color: #C7D2E1;
                font-size: 14px;
            }}
            .plan-meta-line span {{
                padding: 4px 9px;
                border: 1px solid #343A43;
                border-radius: 999px;
                background: #20252C;
            }}
            .plan-highlights {{
                margin-top: 14px;
            }}
            .plan-goal {{
                margin-top: 14px;
                padding: 12px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #171A1F;
            }}
            .plan-goal span {{
                display: block;
                color: #7C8491;
                font-size: 12px;
                margin-bottom: 4px;
            }}
            .plan-goal strong {{
                display: block;
                color: #F3F4F6;
                font-size: 15px;
                line-height: 1.5;
            }}
            .plan-overview-action {{
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                justify-content: space-between;
                gap: 12px;
                padding: 16px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #171A1F;
            }}
            .next-step-copy {{
                color: #F3F4F6 !important;
                font-weight: 700;
                line-height: 1.5;
            }}
            .content-readiness {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 10px 14px;
                margin-top: 12px;
                padding: 10px 12px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: rgba(17, 19, 21, 0.68);
            }}
            .readiness-copy {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 8px;
                color: #A7AFBC;
                font-size: 14px;
            }}
            .readiness-label {{
                color: #F3F4F6;
                font-size: 13px;
                font-weight: 700;
            }}
            .eyebrow {{
                margin-bottom: 6px;
                color: #94A3B8;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 0;
            }}
            .overview-checks {{
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 14px;
            }}
            .overview-checks.compact {{
                gap: 6px;
                margin-top: 0;
            }}
            .check-pill {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid #343A43;
                background: #20252C;
                color: #D7DCE4;
                font-size: 14px;
            }}
            .overview-checks.compact .check-pill {{
                padding: 4px 8px;
                font-size: 13px;
            }}
            .content-readiness .technical-details,
            .content-readiness .error {{
                flex-basis: 100%;
                margin-top: 4px;
            }}
            .technical-details {{
                margin-top: 14px;
                color: #A7AFBC;
                font-size: 14px;
            }}
            .technical-details summary {{
                cursor: pointer;
                color: #C7D2E1;
            }}
            .technical-details div {{
                margin-top: 10px;
                white-space: pre-line;
                color: #7C8491;
            }}
            .qa-form button {{
                padding: 10px 18px;
                border: none;
                border-radius: 8px;
                background: #F3F4F6;
                color: #111315;
                cursor: pointer;
                margin-top: 12px;
                font-weight: 700;
            }}
            .qa-form button:hover {{
                background: #E5E7EB;
            }}
            .qa-form textarea {{
                width: 100%;
                border: 1px solid #343A43;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                box-sizing: border-box;
                resize: vertical;
                color: #F3F4F6;
                background: #111315;
            }}
            .error {{
                margin-top: 16px;
                padding: 12px 16px;
                background: #26191B;
                border: 1px solid #5D3238;
                border-radius: 8px;
                color: #FCA5A5;
                white-space: pre-line;
            }}
            .content-box {{
                white-space: pre-line;
                background: #111315;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                padding: 16px;
                margin-top: 12px;
                color: #E5E7EB;
            }}
            .quote-box {{
                white-space: pre-line;
                background: #111315;
                border-left: 4px solid #4B5563;
                border-radius: 6px;
                padding: 12px 14px;
                margin-top: 12px;
            }}
            .quote-section {{
                margin-top: 18px;
                padding-top: 16px;
                border-top: 1px solid #2A2F36;
            }}
            .quote-section h3 {{
                font-size: 15px;
                margin-bottom: 8px;
            }}
            .qa-answer {{
                margin-top: 0;
            }}
            .workspace-tabs {{
                margin-top: 24px;
            }}
            .tab-nav {{
                display: flex;
                align-items: center;
                gap: 4px;
                padding: 0 2px;
                border-bottom: 1px solid #2A2F36;
                overflow-x: auto;
            }}
            .tab-button {{
                flex: 0 0 auto;
                position: relative;
                padding: 12px 14px;
                border: none;
                border-radius: 8px 8px 0 0;
                background: transparent;
                color: #7C8491;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
            }}
            .tab-button:hover {{
                color: #E5E7EB;
                background: rgba(255, 255, 255, 0.03);
            }}
            .tab-button.active {{
                color: #F3F4F6;
                background: #171A1F;
                box-shadow: inset 0 -2px 0 #F3F4F6;
                font-weight: 700;
            }}
            .tab-content {{
                border: 1px solid #2A2F36;
                border-top: none;
                border-radius: 0 0 8px 8px;
                background: #171A1F;
                min-height: 420px;
            }}
            .tab-panel {{
                display: none;
                padding: 28px;
            }}
            .tab-panel.active {{
                display: block;
            }}
            .tab-section {{
                max-width: 820px;
                margin-bottom: 20px;
            }}
            .tab-section h2 {{
                margin-bottom: 6px;
            }}
            .reading-box {{
                max-width: 860px;
                white-space: pre-line;
                color: #E5E7EB;
                background: #111315;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                padding: 24px;
                line-height: 1.8;
            }}
            .detail-grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 14px;
                margin: 18px 0;
            }}
            .detail-card {{
                padding: 16px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #1D2127;
            }}
            .detail-card h3 {{
                margin-bottom: 8px;
                font-size: 16px;
            }}
            .detail-card p {{
                margin-bottom: 8px;
                color: #A7AFBC;
            }}
            .detail-card ul {{
                margin: 10px 0 0 18px;
                padding: 0;
                color: #E5E7EB;
            }}
            .plan-facts {{
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 10px;
                margin-top: 18px;
            }}
            .plan-facts div {{
                padding: 12px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #111315;
            }}
            .plan-facts span {{
                display: block;
                color: #7C8491;
                font-size: 12px;
                margin-bottom: 4px;
            }}
            .plan-facts strong {{
                display: block;
                color: #F3F4F6;
                font-size: 14px;
                line-height: 1.35;
            }}
            .plan-actions {{
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 10px;
                margin-top: 18px;
            }}
            .plan-actions .secondary-link {{
                margin-top: 0;
            }}
            .qa-workspace {{
                display: grid;
                grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
                gap: 18px;
                align-items: start;
            }}
            .qa-input-card,
            .result-card {{
                padding: 18px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #1D2127;
            }}
            .qa-input-card .muted {{
                margin-bottom: 14px;
            }}
            .qa-actions {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-top: 12px;
            }}
            .qa-actions span {{
                color: #7C8491;
                font-size: 13px;
            }}
            .qa-empty-note {{
                padding: 18px;
                border: 1px dashed #343A43;
                border-radius: 8px;
                background: #111315;
                color: #A7AFBC;
            }}
            .qa-empty-note p {{
                margin-bottom: 0;
            }}
            .result-heading h3 {{
                font-size: 17px;
                margin-bottom: 12px;
            }}
            .empty-state {{
                max-width: 760px;
                padding: 28px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #111315;
            }}
            .empty-state h2 {{
                margin-bottom: 8px;
            }}
            .capability-list {{
                display: grid;
                gap: 10px;
                margin-top: 18px;
            }}
            .capability-list div {{
                padding: 14px;
                border: 1px solid #2A2F36;
                border-radius: 8px;
                background: #171A1F;
            }}
            .capability-list strong,
            .capability-list span {{
                display: block;
            }}
            .capability-list strong {{
                margin-bottom: 4px;
                color: #F3F4F6;
                font-size: 14px;
            }}
            .capability-list span {{
                color: #A7AFBC;
                font-size: 14px;
            }}
            @media (max-width: 820px) {{
                .hero-top,
                .plan-overview {{
                    grid-template-columns: 1fr;
                    display: grid;
                }}
                .hero-actions {{
                    width: 100%;
                }}
                .entry-link {{
                    box-sizing: border-box;
                    text-align: center;
                    width: 100%;
                }}
                .status-strip {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
                .detail-grid {{
                    grid-template-columns: 1fr;
                }}
                .qa-workspace,
                .plan-facts {{
                    grid-template-columns: 1fr;
                }}
            }}
            @media (max-width: 560px) {{
                body {{
                    padding: 24px 14px 40px;
                }}
                h1 {{
                    font-size: 26px;
                }}
                .workspace-hero {{
                    padding: 18px;
                }}
                .status-strip {{
                    grid-template-columns: 1fr;
                }}
                .tab-panel {{
                    padding: 18px;
                }}
                .tab-button {{
                    padding: 10px 11px;
                }}
                .empty-state {{
                    padding: 20px;
                }}
                .qa-actions {{
                    align-items: flex-start;
                    flex-direction: column;
                }}
            }}
        </style>
    </head>
    <body>
        <section class="workspace-hero">
            <div class="hero-top">
                <div class="hero-title-group">
                    <a class="back-link" href="/">← 返回主页面</a>
                    <h1>{html.escape(document_title)}</h1>
                    <p class="muted">先确认当前方案，再用摘要、大纲、原文问答和助手入口补齐理解。</p>
                </div>
                <div class="hero-actions">
                    {primary_action_html}
                </div>
            </div>
            <div class="status-strip">
                {status_items_html}
            </div>
            {plan_overview_html}
            {content_readiness_html}
        </section>

        {workspace_tabs_html}
        <script>
            document.querySelectorAll("[data-tab-target]").forEach((button) => {{
                button.addEventListener("click", () => {{
                    const target = button.dataset.tabTarget;
                    document.querySelectorAll("[data-tab-target]").forEach((item) => {{
                        const isActive = item.dataset.tabTarget === target;
                        item.classList.toggle("active", isActive);
                        item.setAttribute("aria-selected", isActive ? "true" : "false");
                    }});
                    document.querySelectorAll("[data-tab-panel]").forEach((panel) => {{
                        panel.classList.toggle("active", panel.dataset.tabPanel === target);
                    }});
                }});
            }});
        </script>
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

        chunks_payload = build_chunks_from_parsed_text(parse_result["text"], parsed_name)
        chunks_path = get_chunks_path(parsed_name)
        save_chunks(chunks_payload, chunks_path)
        chunks_name = chunks_path.name
        vector_index_message = ""
        try:
            vector_index_result = build_and_save_document_index(chunks_payload, OUTPUTS_DIR)
            vector_index_message = (
                f"Embedding 模式：{vector_index_result['embedding_mode']} / "
                f"{vector_index_result['embedding_provider']} / "
                f"{vector_index_result['embedding_model']}\n"
                f"Embedding 产物已保存：{vector_index_result['embeddings_path'].name}\n"
                f"向量索引已保存：{vector_index_result['index_path'].name}"
            )
        except Exception as index_error:
            vector_index_message = f"Embedding / 向量索引生成失败，不影响主流程：{index_error}"

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
            f"Chunk 产物已保存：{chunks_name}\n"
            f"{vector_index_message}\n"
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

        rag_result = answer_question_with_rag(parsed_name, question, top_k=5)

        context = build_document_workspace_context(parsed_name)
        return render_document_workspace(
            context,
            question=question,
            answer=rag_result.get("answer", ""),
            citations=rag_result.get("citations", []),
            message=f"当前问答基于 RAG 检索：{parsed_name}",
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
