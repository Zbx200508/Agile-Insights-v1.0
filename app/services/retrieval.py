import re
from typing import List


def split_text_into_chunks(text: str, max_chars: int = 800) -> List[str]:
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current += ("\n" if current else "") + para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def _extract_keywords(question: str) -> List[str]:
    raw_tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", question)
    tokens = [t for t in raw_tokens if len(t.strip()) >= 2]
    return tokens


def _score_chunk(question: str, chunk: str) -> float:
    tokens = _extract_keywords(question)
    score = 0.0

    for token in tokens:
        count = chunk.count(token)
        if count > 0:
            score += count * max(len(token), 2)

    # 补一个字符级弱匹配，提升中文问答的召回
    question_chars = {c for c in question if "\u4e00" <= c <= "\u9fff" or c.isalnum()}
    chunk_chars = set(chunk)
    score += len(question_chars & chunk_chars) * 0.2

    return score


def retrieve_relevant_chunks(question: str, text: str, top_k: int = 3) -> List[str]:
    chunks = split_text_into_chunks(text)
    if not chunks:
        return []

    scored = []
    for chunk in chunks:
        score = _score_chunk(question, chunk)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 如果完全没有命中，就退化返回前 1-2 个块，避免空上下文
    if scored[0][0] <= 0:
        return [c for _, c in scored[:2]]

    return [c for _, c in scored[:top_k]]