from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.embedding_service import create_embedding, create_embeddings_with_metadata


BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"


class VectorStoreError(Exception):
    """Raised when the local single-document vector index cannot be used."""


def get_chunks_path(parsed_name: str, output_dir: Path = OUTPUTS_DIR) -> Path:
    return output_dir / f"{Path(parsed_name).stem}_chunks.json"


def get_embeddings_path(parsed_name: str, output_dir: Path = OUTPUTS_DIR) -> Path:
    return output_dir / f"{Path(parsed_name).stem}_embeddings.json"


def get_vector_index_path(parsed_name: str, output_dir: Path = OUTPUTS_DIR) -> Path:
    return output_dir / f"{Path(parsed_name).stem}_vector_index.json"


def build_and_save_document_index(
    chunks_payload: dict[str, Any],
    output_dir: Path = OUTPUTS_DIR,
) -> dict[str, Any]:
    parsed_name = _require_text(chunks_payload.get("parsed_name"), "parsed_name")
    chunks = _load_chunk_items(chunks_payload)
    chunk_texts = [chunk["chunk_text"] for chunk in chunks]

    embeddings, embedding_run_info = create_embeddings_with_metadata(chunk_texts)
    _validate_embeddings(embeddings, expected_count=len(chunks))

    embeddings_payload = _build_embeddings_payload(chunks_payload, chunks, embeddings, embedding_run_info)
    index_payload = _build_index_payload(chunks_payload, chunks, embeddings, embeddings_payload)

    embeddings_path = get_embeddings_path(parsed_name, output_dir=output_dir)
    index_path = get_vector_index_path(parsed_name, output_dir=output_dir)

    _write_json(embeddings_payload, embeddings_path)
    _write_json(index_payload, index_path)

    return {
        "parsed_name": parsed_name,
        "embedding_mode": embeddings_payload["embedding_mode"],
        "embedding_model": embeddings_payload["embedding_model"],
        "embedding_provider": embeddings_payload["embedding_provider"],
        "embedding_dim": embeddings_payload["embedding_dim"],
        "fallback_reason": embeddings_payload["fallback_reason"],
        "chunk_count": len(chunks),
        "embeddings_path": embeddings_path,
        "index_path": index_path,
    }


def build_and_save_document_index_from_chunks_file(
    chunks_path: Path,
    output_dir: Path = OUTPUTS_DIR,
) -> dict[str, Any]:
    if not chunks_path.exists():
        raise VectorStoreError(f"Chunks file not found: {chunks_path}")

    chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    return build_and_save_document_index(chunks_payload, output_dir=output_dir)


def search_document_chunks(parsed_name: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    if not query or not query.strip():
        return []
    if top_k <= 0:
        return []

    index_path = get_vector_index_path(parsed_name)
    if not index_path.exists():
        chunks_path = get_chunks_path(parsed_name)
        if not chunks_path.exists():
            raise VectorStoreError(f"Vector index and chunks file are both missing for: {parsed_name}")
        build_and_save_document_index_from_chunks_file(chunks_path)

    index_payload = _read_json(index_path)
    records = index_payload.get("records")
    if not isinstance(records, list) or not records:
        return []

    model = _require_text(index_payload.get("embedding_model"), "embedding_model")
    query_vector = _normalize_vector(create_embedding(query, model=model))
    if not query_vector:
        return []

    scored_records = []
    for record in records:
        vector = record.get("vector")
        if not isinstance(vector, list) or len(vector) != len(query_vector):
            continue
        score = _dot(query_vector, vector)
        scored_records.append((score, record))

    scored_records.sort(key=lambda item: item[0], reverse=True)
    limit = top_k

    results: list[dict[str, Any]] = []
    for score, record in scored_records[:limit]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        results.append(
            {
                "chunk_id": record.get("chunk_id", ""),
                "score": round(float(score), 6),
                "chunk_text": record.get("chunk_text", ""),
                "page_range": metadata.get("page_range"),
                "chapter_title": metadata.get("chapter_title", ""),
                "source_scope": metadata.get("source_scope", ""),
            }
        )

    return results


def _load_chunk_items(chunks_payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_chunks = chunks_payload.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise VectorStoreError("Chunks payload must contain a non-empty chunks list.")

    chunks = []
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        chunk_text = str(raw_chunk.get("chunk_text", "")).strip()
        chunk_id = str(raw_chunk.get("chunk_id", "")).strip()
        if not chunk_text or not chunk_id:
            continue
        chunks.append(raw_chunk)

    if not chunks:
        raise VectorStoreError("No valid chunks found for embedding generation.")
    return chunks


def _build_embeddings_payload(
    chunks_payload: dict[str, Any],
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    embedding_run_info: dict[str, str | None],
) -> dict[str, Any]:
    parsed_name = _require_text(chunks_payload.get("parsed_name"), "parsed_name")
    document_id = _require_text(chunks_payload.get("document_id"), "document_id")
    embedding_dim = len(embeddings[0]) if embeddings else 0

    return {
        "document_id": document_id,
        "parsed_name": parsed_name,
        "embedding_mode": embedding_run_info.get("embedding_mode", ""),
        "embedding_model": embedding_run_info.get("embedding_model", ""),
        "embedding_provider": embedding_run_info.get("embedding_provider", ""),
        "embedding_dim": embedding_dim,
        "fallback_reason": embedding_run_info.get("fallback_reason"),
        "chunk_count": len(chunks),
        "created_at": _now_iso(),
        "embeddings": [
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "chunk_order": chunk.get("chunk_order"),
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ],
    }


def _build_index_payload(
    chunks_payload: dict[str, Any],
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    embeddings_payload: dict[str, Any],
) -> dict[str, Any]:
    parsed_name = _require_text(chunks_payload.get("parsed_name"), "parsed_name")
    document_id = _require_text(chunks_payload.get("document_id"), "document_id")

    records = []
    for chunk, embedding in zip(chunks, embeddings):
        records.append(
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "chunk_order": chunk.get("chunk_order"),
                "chunk_text": chunk.get("chunk_text", ""),
                "vector": _normalize_vector(embedding),
                "metadata": {
                    "document_id": chunk.get("document_id", document_id),
                    "parsed_name": chunk.get("parsed_name", parsed_name),
                    "page_range": chunk.get("page_range"),
                    "chapter_id": chunk.get("chapter_id", ""),
                    "chapter_title": chunk.get("chapter_title", ""),
                    "source_scope": chunk.get("source_scope", ""),
                    "chunk_type": chunk.get("chunk_type", ""),
                    "summary_hint": chunk.get("summary_hint", ""),
                },
            }
        )

    return {
        "document_id": document_id,
        "parsed_name": parsed_name,
        "index_type": "local_json_cosine_v1",
        "embedding_mode": embeddings_payload["embedding_mode"],
        "embedding_model": embeddings_payload["embedding_model"],
        "embedding_provider": embeddings_payload["embedding_provider"],
        "embedding_dim": embeddings_payload["embedding_dim"],
        "fallback_reason": embeddings_payload["fallback_reason"],
        "chunk_count": len(records),
        "created_at": _now_iso(),
        "embedding_source": get_embeddings_path(parsed_name).name,
        "records": records,
    }


def _validate_embeddings(embeddings: list[list[float]], expected_count: int) -> None:
    if len(embeddings) != expected_count:
        raise VectorStoreError(
            f"Embedding count mismatch: expected {expected_count}, got {len(embeddings)}."
        )
    if not embeddings:
        raise VectorStoreError("Embedding result is empty.")

    first_dim = len(embeddings[0])
    if first_dim <= 0:
        raise VectorStoreError("Embedding dimension is empty.")

    for index, embedding in enumerate(embeddings, start=1):
        if len(embedding) != first_dim:
            raise VectorStoreError(
                f"Embedding dimension mismatch at item {index}: expected {first_dim}, got {len(embedding)}."
            )


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0:
        return []
    return [float(value) / norm for value in vector]


def _dot(first: list[float], second: list[float]) -> float:
    return sum(a * b for a, b in zip(first, second))


def _require_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise VectorStoreError(f"Missing required field: {field_name}")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VectorStoreError(f"Failed to read JSON file: {path}") from exc

    if not isinstance(data, dict):
        raise VectorStoreError(f"JSON file must contain an object: {path}")
    return data


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
