from __future__ import annotations

from typing import Any

from app.services.llm_service import generate_rag_answer
from app.services.vector_store_service import get_vector_index_path, search_document_chunks


class RagServiceError(Exception):
    """Raised when the single-document RAG flow cannot complete."""


def answer_question_with_rag(parsed_name: str, question: str, top_k: int = 5) -> dict[str, Any]:
    if not parsed_name or not parsed_name.strip():
        raise RagServiceError("缺少当前文档名称，无法执行当前文档内检索。")
    if not question or not question.strip():
        raise RagServiceError("问题为空，无法执行问答。")

    index_path = get_vector_index_path(parsed_name)
    if not index_path.exists():
        raise RagServiceError(
            f"当前文档索引不存在，无法执行 RAG 问答：{index_path.name}。请重新上传文档或先生成索引。"
        )

    try:
        retrieved_chunks = search_document_chunks(parsed_name, question, top_k=top_k)
    except Exception as exc:
        raise RagServiceError(f"当前文档检索失败：{exc}") from exc

    if not retrieved_chunks:
        return {
            "answer": "当前文档索引没有检索到可用片段，无法基于原文回答。",
            "citations": [],
            "retrieved_chunks": [],
        }

    try:
        answer = generate_rag_answer(question, retrieved_chunks)
    except Exception as exc:
        raise RagServiceError(f"基于检索片段生成答案失败：{exc}") from exc

    return {
        "answer": answer,
        "citations": _build_citations(retrieved_chunks),
        "retrieved_chunks": retrieved_chunks,
    }


def _build_citations(retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for chunk in retrieved_chunks:
        citations.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "page_range": chunk.get("page_range"),
                "chapter_title": chunk.get("chapter_title", ""),
                "source_scope": chunk.get("source_scope", ""),
                "quote": _make_quote(chunk.get("chunk_text", "")),
            }
        )
    return citations


def _make_quote(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."
