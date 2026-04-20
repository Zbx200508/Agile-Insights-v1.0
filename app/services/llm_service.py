import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("API_KEY")
api_base = os.getenv("API_BASE")
model_name = os.getenv("MODEL_NAME")

if not api_key or not api_base or not model_name:
    raise ValueError("请检查 .env，API_KEY / API_BASE / MODEL_NAME 不能为空")

client = OpenAI(
    api_key=api_key,
    base_url=api_base,
)


def generate_summary(text: str) -> str:
    if not text or not text.strip():
        return "未提取到有效文本，无法生成摘要。"

    trimmed_text = text[:12000]

    prompt = f"""
请基于以下材料内容，生成一份简洁的一页摘要。

要求：
1. 只基于给定内容总结，不补充外部信息。
2. 重点提炼：主题、核心结论、关键信息、适用场景。
3. 输出为中文。
4. 控制在 4-6 条要点内，适合快速阅读。
5. 不要输出“根据材料”“本文主要讲了”这类空话。

材料内容：
{trimmed_text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个擅长提炼培训资料和行业白皮书内容的总结助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def generate_outline(text: str) -> str:
    if not text or not text.strip():
        return "未提取到有效文本，无法生成逻辑大纲。"

    trimmed_text = text[:16000]

    prompt = f"""
请基于以下材料内容，输出一份清晰的三级逻辑大纲。

要求：
1. 只基于给定内容，不补充外部信息。
2. 输出中文。
3. 使用严格三级结构：
   一、一级标题
   （一）二级标题
   1. 三级要点
4. 如果材料本身层级不完整，也要尽量整理成清晰结构。
5. 不要写前言、后记、总结说明，只输出大纲本身。
6. 大纲要能帮助用户快速理解材料的逻辑结构，而不是简单重复原文句子。

材料内容：
{trimmed_text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个擅长梳理培训资料和行业白皮书结构的逻辑大纲助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def answer_question(question: str, context_chunks: list[str]) -> str:
    if not question or not question.strip():
        return "问题为空，无法回答。"

    if not context_chunks:
        return "当前材料中没有检索到可用内容，暂时无法回答。"

    context_text = "\n\n".join(
        [f"[片段{i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )[:16000]

    prompt = f"""
请只基于给定材料片段回答问题，不得补充外部信息。

要求：
1. 只依据材料片段回答。
2. 如果材料中没有明确答案，要直接说明“材料中未明确提及”或“无法根据当前材料确定”。
3. 先给直接答案，再用 2-4 条要点补充说明。
4. 输出中文，表达简洁，不说空话。
5. 不要编造数字、结论或背景。

问题：
{question}

材料片段：
{context_text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个严格依据原文内容回答问题的资料问答助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()


def generate_rag_answer(question: str, retrieved_chunks: list[dict]) -> str:
    if not question or not question.strip():
        return "问题为空，无法回答。"

    if not retrieved_chunks:
        return "当前没有检索到可用原文片段，无法基于当前文档回答。"

    context_text = "\n\n".join(
        [
            (
                f"[{chunk.get('chunk_id', f'chunk_{i + 1}')}]\n"
                f"页码：{chunk.get('page_range') or 'unknown'}\n"
                f"章节：{chunk.get('chapter_title') or ''}\n"
                f"范围：{chunk.get('source_scope') or ''}\n"
                f"原文片段：\n{chunk.get('chunk_text', '')}"
            )
            for i, chunk in enumerate(retrieved_chunks)
        ]
    )[:18000]

    prompt = f"""
请只基于给定的检索片段回答用户问题，不要补充外部知识。

回答要求：
1. 只能使用检索片段中明确支持的信息。
2. 如果片段不足以确认答案，要直接说明“当前检索片段不足以确认”。
3. 不要把推断、常识或外部背景写成原文事实。
4. 尽量用简洁中文回答，先给直接答案，再用 2-4 条要点说明。
5. 如果使用了某个片段的信息，可在句末用对应 chunk_id 标注，例如 [c_029]。

用户问题：
{question}

检索片段：
{context_text}
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是严格基于检索片段回答问题的 RAG 问答助手。不得使用外部知识，不得编造引用。"
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content.strip()

def generate_learning_map_raw(text: str, document_title: str = "") -> dict:
    """
    调用大模型生成“学习地图原始 JSON”。
    这里只负责让模型尽量吐出结构化结果，不负责最终校验。
    最终校验、标准化与兜底交给 learning_map_service.py。
    """
    if not text or not text.strip():
        raise ValueError("解析文本为空，无法生成学习地图。")

    trimmed_text = text[:20000]
    title_part = f"文档标题：{document_title}\n" if document_title else ""

    prompt = f"""
你是一个“学习资料结构化拆解助手”。

你的任务是：把一份学习资料拆成“学习地图 JSON”，供后续产品生成学习计划和速通路径使用。

请严格遵守以下要求：

1. 输出必须是合法 JSON。
2. 不要输出任何解释、前后缀、说明文字、markdown 代码块。
3. 顶层必须包含两个字段：
   - document
   - chapters
4. document 至少包含：
   - document_title
   - document_summary
5. chapters 必须是数组，每个 chapter 至少包含：
   - chapter_id
   - order
   - title
   - summary
   - estimated_minutes
   - difficulty_level
   - selected
   - mastery_level
   - priority_level
   - source_scope
   - topic_units
6. difficulty_level 只能是：
   - low
   - medium
   - high
7. mastery_level 只能是：
   - mastered
   - familiar
   - unfamiliar
8. priority_level 只能是：
   - low
   - medium
   - high
9. selected 默认尽量为 true，除非明显是附录、补充说明、纯案例延伸部分。
10. topic_units 也必须是数组；如果某章不适合继续拆分，可以给空数组。
11. 不要拆得过细。章节建议 4~10 个，单章 topic_units 建议 0~5 个。
12. summary 必须简洁明确，适合直接给用户看。
13. estimated_minutes 只给大致可用的整数分钟数，不要追求精确。
14. document_title 可以参考传入标题，也可以根据正文修正得更自然。
15. 章节标题尽量贴近资料原始结构；如果原结构不清晰，也可以生成适合学习的模块标题。

输出 JSON 结构参考如下：

{{
  "document": {{
    "document_title": "示例标题",
    "document_summary": "一句话概括整份资料的内容。"
  }},
  "chapters": [
    {{
      "chapter_id": "ch_1",
      "order": 1,
      "title": "第一章标题",
      "summary": "一句话概括这一章讲什么。",
      "estimated_minutes": 40,
      "difficulty_level": "medium",
      "selected": true,
      "mastery_level": "unfamiliar",
      "priority_level": "medium",
      "source_scope": "第1章",
      "topic_units": [
        {{
          "unit_id": "u_1_1",
          "order": 1,
          "title": "主题标题",
          "summary": "一句话概括该主题。",
          "estimated_minutes": 20,
          "difficulty_level": "medium",
          "selected": true,
          "mastery_level": "unfamiliar",
          "priority_level": "medium",
          "source_scope": "第1章-主题1"
        }}
      ]
    }}
  ]
}}

现在请基于下面资料生成 JSON。

{title_part}
资料正文：
{trimmed_text}
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一个擅长把学习资料拆成可执行学习地图的结构化助手。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content.strip()

    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"学习地图原始 JSON 解析失败：{e}\\n模型原始输出前1000字符：{cleaned[:1000]}"
        )
    
def generate_system_learning_plan_raw(
    selected_scope: list[dict],
    week_minutes: dict[str, int],
) -> dict:
    """
    基于已选学习范围和一周可投入时间，生成系统学习计划原始 JSON。
    最终规范化与兜底交给 system_learning_plan_service.py。
    """
    if not selected_scope:
        raise ValueError("selected_scope 为空，无法生成系统学习计划。")

    scope_json = json.dumps(selected_scope[:30], ensure_ascii=False, indent=2)
    week_json = json.dumps(week_minutes, ensure_ascii=False, indent=2)

    prompt = f"""
你是一个“学习计划编排助手”。

你的任务是：
基于用户当前选中的学习范围、掌握程度、优先级，以及未来 7 天可投入时间，
生成一份“系统学习计划”的 JSON。

请严格遵守以下要求：

1. 输出必须是合法 JSON，不要输出解释、前后缀、markdown 代码块。
2. 顶层字段固定为：
   - title
   - subtitle
   - highlights
   - focuses
   - days
   - review_note
3. highlights 必须是字符串数组，适合展示在页面顶部，建议 2~4 条。
4. focuses 必须是字符串数组，表示本周重点，建议 2~4 条。
5. days 必须是数组，每一项至少包含：
   - day
   - minutes
   - tasks
6. tasks 必须是字符串数组，每条任务都要明确“学什么”，不要写空话。
7. 计划必须尽量符合每一天的可投入时间，不要把超量任务塞进低时长日期。
8. 优先安排：
   - priority_level 高的内容
   - mastery_level 为 unfamiliar 的内容
9. 对已经 mastered 的内容，可以少安排或只做轻复习。
10. 整体风格要像真实学习计划，而不是泛泛总结。
11. review_note 用一句话说明本周如何复习。

输出格式示例：

{{
  "title": "本周学习计划",
  "subtitle": "基于当前学习范围、掌握度与每周时间分配生成的系统学习计划。",
  "highlights": [
    "预计总投入：220 分钟",
    "建议日均：30 分钟"
  ],
  "focuses": [
    "理解大模型基础概念",
    "掌握 RAG 与 Agent 的区别"
  ],
  "days": [
    {{
      "day": "周一",
      "minutes": 30,
      "tasks": [
        "学习：大模型是什么（20 分钟）",
        "复习：AI 产品经理职责边界（10 分钟）"
      ]
    }}
  ],
  "review_note": "建议在周后半段安排一次轻复习，巩固高优先级且尚未掌握的内容。"
}}

当前选中的学习范围如下：
{scope_json}

未来 7 天可投入时间如下：
{week_json}
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一个擅长根据学习范围和时间预算编排系统学习计划的助手。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content.strip()
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"系统学习方案 JSON 解析失败：{e}\\n模型原始输出前1000字符：{cleaned[:1000]}"
        )
    
def generate_quick_understanding_plan_raw(
    selected_scope: list[dict],
    target_minutes: int,
) -> dict:
    """
    基于已选学习范围和目标投入时间，生成速通理解方案原始 JSON。
    最终规范化与兜底交给 quick_understanding_plan_service.py。
    """
    if not selected_scope:
        raise ValueError("selected_scope 为空，无法生成速通理解方案。")

    scope_json = json.dumps(selected_scope[:30], ensure_ascii=False, indent=2)

    prompt = f"""
你是一个“速通理解路径规划助手”。

你的任务是：
基于用户当前选中的学习范围，以及用户计划投入的总时长，
生成一份“最短理解路径”的 JSON，让用户能在最短时间内抓住核心内容。

请严格遵守以下要求：

1. 输出必须是合法 JSON，不要输出解释、前后缀、markdown 代码块。
2. 顶层字段固定为：
   - title
   - subtitle
   - highlights
   - steps
   - must_know
   - next_action
3. highlights 必须是字符串数组，适合展示在页面顶部，建议 2~3 条。
4. steps 必须是数组，每一项至少包含：
   - title
   - minutes
   - why
5. steps 建议 2~4 步，不要太多。
6. 每一步必须明确写“先看什么 / 再看什么 / 最后看什么”这种强引导式内容。
7. must_know 必须是字符串数组，列出最值得记住的核心概念，建议 3~5 个。
8. next_action 用一句话告诉用户下一步怎么学。
9. 速通理解优先安排：
   - priority_level 高的内容
   - mastery_level 为 unfamiliar 的内容
   - 能帮助用户快速建立整体框架的内容
10. 总时长尽量贴近用户的 target_minutes。

输出格式示例：

{{
  "title": "最短理解路径",
  "subtitle": "基于当前学习范围与目标投入时间生成的速通理解方案。",
  "highlights": [
    "推荐投入：30 分钟",
    "核心内容：3 项"
  ],
  "steps": [
    {{
      "title": "先看：大模型是什么",
      "minutes": 10,
      "why": "这是整份资料的概念底座。"
    }},
    {{
      "title": "再看：Prompt、RAG、Agent 的区别",
      "minutes": 12,
      "why": "这是最容易混淆、但最关键的知识点。"
    }}
  ],
  "must_know": [
    "大模型",
    "Prompt",
    "RAG",
    "Agent"
  ],
  "next_action": "如果希望系统学习，建议保留当前高优先级内容进一步生成一周计划。"
}}

当前选中的学习范围如下：
{scope_json}

用户计划投入总时长：
{target_minutes} 分钟
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "你是一个擅长把复杂学习内容压缩成最短理解路径的助手。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content.strip()
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):].strip()

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"速通理解方案 JSON 解析失败：{e}\\n模型原始输出前1000字符：{cleaned[:1000]}"
        )
