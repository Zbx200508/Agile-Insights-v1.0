from pathlib import Path
from datetime import datetime
import html

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse

from app.services.pdf_parser import extract_text_from_pdf
from app.services.llm_service import generate_summary, generate_outline, answer_question
from app.services.retrieval import retrieve_relevant_chunks


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PARSED_DIR = DATA_DIR / "parsed"
OUTPUTS_DIR = DATA_DIR / "outputs"

for folder in [UPLOADS_DIR, PARSED_DIR, OUTPUTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="敏捷洞察1.0")


def read_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def get_output_paths(parsed_name: str) -> tuple[Path, Path]:
    stem = Path(parsed_name).stem
    summary_path = OUTPUTS_DIR / f"{stem}_summary.txt"
    outline_path = OUTPUTS_DIR / f"{stem}_outline.txt"
    return summary_path, outline_path


def render_home(
    message: str = "",
    error: str = "",
    summary: str = "",
    outline: str = "",
    parsed_name: str = "",
    question: str = "",
    answer: str = "",
    quotes: list[str] | None = None,
) -> str:
    quotes = quotes or []

    message_html = f'<div class="success">{html.escape(message)}</div>' if message else ""
    error_html = f'<div class="error">{html.escape(error)}</div>' if error else ""

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

            <form class="upload-form" action="/upload" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".pdf" required />
                <button type="submit">上传 PDF</button>
            </form>

            {message_html}
            {error_html}
        </div>

        {summary_html}
        {outline_html}
        {qa_box_html}

        <div class="box">
            <h2>当前开发进度</h2>
            <p>FastAPI 骨架、上传、PDF 文本解析、一页摘要、三级逻辑大纲、基于原文问答链路已接入。</p>
            <p>下一步将继续优化引用质量和上线部署。</p>
        </div>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
def home():
    return render_home()


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

        summary_name = save_path.stem + "_summary.txt"
        summary_path = OUTPUTS_DIR / summary_name
        summary_path.write_text(summary, encoding="utf-8")

        outline_name = save_path.stem + "_outline.txt"
        outline_path = OUTPUTS_DIR / outline_name
        outline_path.write_text(outline, encoding="utf-8")

        size_kb = round(len(file_bytes) / 1024, 2)

        message = (
            f"上传成功：{save_name}\n"
            f"文件大小：约 {size_kb} KB\n"
            f"总页数：{parse_result['page_count']} 页\n"
            f"有文本页数：{parse_result['non_empty_pages']} 页\n"
            f"解析字符数：{parse_result['char_count']} 字\n"
            f"解析文本已保存：{parsed_name}\n"
            f"摘要已保存：{summary_name}\n"
            f"大纲已保存：{outline_name}"
        )
        return render_home(
            message=message,
            summary=summary,
            outline=outline,
            parsed_name=parsed_name,
        )

    except Exception as e:
        return render_home(error=f"上传、解析、摘要或大纲生成失败：{str(e)}")


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

        return render_home(
            summary=summary,
            outline=outline,
            parsed_name=parsed_name,
            question=question,
            answer=answer,
            quotes=relevant_chunks[:2],
            message=f"当前问答基于：{parsed_name}",
        )

    except Exception as e:
        return render_home(error=f"问答失败：{str(e)}")