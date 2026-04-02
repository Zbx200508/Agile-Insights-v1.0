import os
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