from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.services.rag_service import answer_question_with_rag  # noqa: E402
from app.services.vector_store_service import search_document_chunks  # noqa: E402


DEFAULT_EVAL_SET = (
    ROOT_DIR
    / "data"
    / "evals"
    / "20260406_224716_俞军产品方法论_eval_set.json"
)


def load_eval_set(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {}, payload
    if isinstance(payload, dict):
        evals = payload.get("evals", [])
        if not isinstance(evals, list):
            raise ValueError("eval_set JSON must contain an evals list.")
        return payload.get("document", {}), evals
    raise ValueError("eval_set JSON must be either a list or an object.")


def preview_eval_set(
    eval_set_path: Path,
    parsed_name: str | None,
    top_k: int,
    with_answer: bool,
) -> None:
    document, evals = load_eval_set(eval_set_path)
    target_parsed_name = parsed_name or document.get("parsed_name")
    if not target_parsed_name:
        raise ValueError("parsed_name is required. Pass --parsed-name or include document.parsed_name.")

    print(
        json.dumps(
            {
                "eval_set": str(eval_set_path),
                "parsed_name": target_parsed_name,
                "question_count": len(evals),
                "top_k": top_k,
                "with_answer": with_answer,
            },
            ensure_ascii=False,
        )
    )

    for item in evals:
        eval_id = item.get("eval_id", "")
        question = item.get("question", "")

        preview: dict[str, Any] = {
            "eval_id": eval_id,
            "question": question,
            "question_type": item.get("question_type", ""),
            "should_allow_insufficient": item.get("should_allow_insufficient", False),
            "expected_support": item.get("expected_support", []),
            "retrieval": [],
        }

        try:
            retrieved = search_document_chunks(target_parsed_name, question, top_k=top_k)
            preview["retrieval"] = [
                {
                    "rank": rank,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": chunk.get("score"),
                    "page_range": chunk.get("page_range"),
                    "chapter_title": chunk.get("chapter_title", ""),
                    "source_scope": chunk.get("source_scope", ""),
                    "text_preview": _truncate(chunk.get("chunk_text", ""), 180),
                }
                for rank, chunk in enumerate(retrieved, start=1)
            ]
        except Exception as exc:
            preview["retrieval_error"] = str(exc)

        if with_answer:
            try:
                rag_result = answer_question_with_rag(target_parsed_name, question, top_k=top_k)
                preview["answer_preview"] = _truncate(rag_result.get("answer", ""), 500)
                preview["citations"] = rag_result.get("citations", [])
            except Exception as exc:
                preview["answer_error"] = str(exc)

        print(json.dumps(preview, ensure_ascii=False))


def _truncate(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview single-document RAG eval retrieval hits and optional answer output."
    )
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=DEFAULT_EVAL_SET,
        help="Path to eval_set JSON.",
    )
    parser.add_argument(
        "--parsed-name",
        default=None,
        help="Parsed document name. Defaults to document.parsed_name in eval_set.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks per question.")
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="Also call the RAG answer service. This may call the configured chat model.",
    )
    args = parser.parse_args()

    preview_eval_set(
        eval_set_path=args.eval_set,
        parsed_name=args.parsed_name,
        top_k=args.top_k,
        with_answer=args.with_answer,
    )


if __name__ == "__main__":
    main()
