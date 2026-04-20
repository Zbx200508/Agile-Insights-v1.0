from __future__ import annotations

import os
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from dotenv import load_dotenv

try:
    from volcenginesdkarkruntime import Ark
except ImportError:  # pragma: no cover - handled by local fallback at runtime
    Ark = None


load_dotenv()


DEFAULT_EMBEDDING_MODEL = "doubao-embedding-vision-251215"
LOCAL_FALLBACK_EMBEDDING_MODEL = "local-hash-embedding-v1"
LOCAL_FALLBACK_DIMENSIONS = 384
VOLCENGINE_EMBEDDING_MODELS = [
    "doubao-embedding-vision-251215",
    "doubao-embedding-large-text-250515",
    "doubao-embedding-large-text-240915",
    "doubao-embedding-text-240715",
    "doubao-embedding-text-240515",
]
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_INPUT_CHARS = 3000


class EmbeddingServiceError(Exception):
    """Raised when embedding generation cannot be completed."""


@dataclass(frozen=True)
class EmbeddingRunInfo:
    embedding_mode: str
    embedding_model: str
    embedding_provider: str
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "embedding_mode": self.embedding_mode,
            "embedding_model": self.embedding_model,
            "embedding_provider": self.embedding_provider,
            "fallback_reason": self.fallback_reason,
        }


def get_embedding_model_name() -> str:
    return get_configured_embedding_model_name() or DEFAULT_EMBEDDING_MODEL


def get_configured_embedding_model_name() -> str | None:
    return os.getenv("EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL_NAME")


def get_embedding_model_candidates(model: str | None = None) -> list[str]:
    if model:
        return [model]

    configured_candidates = _split_env_list(os.getenv("EMBEDDING_MODEL_CANDIDATES"))
    if configured_candidates:
        return configured_candidates

    configured_model = get_configured_embedding_model_name()
    if configured_model:
        return [configured_model]

    return VOLCENGINE_EMBEDDING_MODELS


def get_embedding_client() -> Any:
    if Ark is None:
        raise EmbeddingServiceError(
            "volcenginesdkarkruntime is required for remote embedding generation."
        )

    api_key = os.getenv("ARK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise EmbeddingServiceError("ARK_API_KEY or API_KEY is required for embedding generation.")

    return Ark(api_key=api_key)


def get_embedding_provider(model: str | None = None) -> str:
    if model == LOCAL_FALLBACK_EMBEDDING_MODEL:
        return "local"

    return "volcengine_ark"


def create_embedding(text: str, model: str | None = None) -> list[float]:
    embeddings = create_embeddings([text], model=model)
    return embeddings[0] if embeddings else []


def create_embeddings(
    texts: Iterable[str],
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    embeddings, _ = create_embeddings_with_model(texts, model=model, batch_size=batch_size)
    return embeddings


def create_embeddings_with_model(
    texts: Iterable[str],
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[list[float]], str]:
    embeddings, run_info = create_embeddings_with_metadata(
        texts,
        model=model,
        batch_size=batch_size,
    )
    return embeddings, str(run_info["embedding_model"])


def create_embeddings_with_metadata(
    texts: Iterable[str],
    model: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[list[list[float]], dict[str, str | None]]:
    clean_texts = [_prepare_embedding_input(text) for text in texts]
    if not clean_texts:
        run_info = EmbeddingRunInfo(
            embedding_mode="remote",
            embedding_model=model or get_embedding_model_name(),
            embedding_provider=get_embedding_provider(model),
            fallback_reason=None,
        )
        return [], run_info.to_dict()

    configured_model = get_configured_embedding_model_name()
    if model == LOCAL_FALLBACK_EMBEDDING_MODEL or (
        model is None and configured_model == LOCAL_FALLBACK_EMBEDDING_MODEL
    ):
        return _create_local_hash_embeddings(clean_texts), _local_run_info(
            "explicit local fallback model configured"
        ).to_dict()

    try:
        client = get_embedding_client()
    except Exception as exc:
        if _local_fallback_enabled(model):
            return _create_local_hash_embeddings(clean_texts), _local_run_info(
                f"remote client unavailable: {_truncate_error(exc)}"
            ).to_dict()
        raise

    last_error: Exception | None = None
    tried_models: list[str] = []
    failed_attempts: list[str] = []
    provider = get_embedding_provider(model)

    for model_name in get_embedding_model_candidates(model):
        tried_models.append(model_name)
        embeddings: list[list[float]] = []
        try:
            for prepared_text in clean_texts:
                embeddings.append(_create_remote_embedding(client, model_name, prepared_text))

            run_info = EmbeddingRunInfo(
                embedding_mode="remote",
                embedding_model=model_name,
                embedding_provider=provider,
                fallback_reason=None,
            )
            return embeddings, run_info.to_dict()
        except Exception as exc:
            last_error = exc
            failed_attempts.append(
                f"{model_name}: {type(exc).__name__}: {_truncate_error(exc, limit=240)}"
            )

    if _local_fallback_enabled(model):
        reason = (
            "remote embedding failed after trying "
            f"[{', '.join(tried_models)}]. "
            f"attempt_errors=[{_truncate_text('; '.join(failed_attempts), limit=1200)}]"
        )
        return _create_local_hash_embeddings(clean_texts), _local_run_info(reason).to_dict()

    tried = ", ".join(tried_models)
    raise EmbeddingServiceError(f"Embedding request failed after trying [{tried}]: {last_error}") from last_error


def probe_remote_embedding_models(
    models: Iterable[str] | None = None,
    sample_text: str = "embedding probe",
) -> list[dict[str, object]]:
    client = get_embedding_client()
    candidates = list(models) if models is not None else get_embedding_model_candidates()

    results: list[dict[str, object]] = []
    for model_name in candidates:
        try:
            embedding = _create_remote_embedding(
                client,
                model_name,
                _prepare_embedding_input(sample_text),
            )
            results.append(
                {
                    "embedding_model": model_name,
                    "embedding_provider": get_embedding_provider(model_name),
                    "ok": True,
                    "embedding_dim": len(embedding),
                    "error_type": None,
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "embedding_model": model_name,
                    "embedding_provider": get_embedding_provider(model_name),
                    "ok": False,
                    "embedding_dim": None,
                    "error_type": type(exc).__name__,
                    "error": _truncate_error(exc),
                }
            )
    return results


def _create_remote_embedding(client: Any, model_name: str, prepared_text: str) -> list[float]:
    response = client.multimodal_embeddings.create(
        model=model_name,
        input=[{"type": "text", "text": prepared_text}],
    )
    embedding = _extract_multimodal_embedding(response)
    if not embedding:
        raise EmbeddingServiceError("Remote embedding response did not contain an embedding.")
    return embedding


def _extract_multimodal_embedding(response: Any) -> list[float]:
    data = getattr(response, "data", None)

    if isinstance(data, list):
        if not data:
            return []
        first_item = data[0]
        embedding = getattr(first_item, "embedding", None)
        if embedding is None and isinstance(first_item, dict):
            embedding = first_item.get("embedding")
        return _coerce_embedding(embedding)

    if data is not None:
        embedding = getattr(data, "embedding", None)
        if embedding is None and isinstance(data, dict):
            embedding = data.get("embedding")
        return _coerce_embedding(embedding)

    embedding = getattr(response, "embedding", None)
    if embedding is None and isinstance(response, dict):
        embedding = response.get("embedding")
    return _coerce_embedding(embedding)


def _coerce_embedding(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        return [float(item) for item in value]
    except TypeError:
        return []


def _prepare_embedding_input(text: str) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return "empty chunk"
    return clean[:DEFAULT_MAX_INPUT_CHARS]


def _local_fallback_enabled(model: str | None) -> bool:
    if model:
        return False
    value = os.getenv("ENABLE_LOCAL_EMBEDDING_FALLBACK", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _local_run_info(fallback_reason: str) -> EmbeddingRunInfo:
    return EmbeddingRunInfo(
        embedding_mode="local_fallback",
        embedding_model=LOCAL_FALLBACK_EMBEDDING_MODEL,
        embedding_provider="local",
        fallback_reason=fallback_reason,
    )


def _truncate_error(error: Exception | None, limit: int = 500) -> str:
    if error is None:
        return "unknown error"
    text = " ".join(str(error).split())
    return _truncate_text(text, limit=limit)


def _truncate_text(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _create_local_hash_embeddings(texts: list[str]) -> list[list[float]]:
    return [_create_local_hash_embedding(text) for text in texts]


def _create_local_hash_embedding(text: str, dimensions: int = LOCAL_FALLBACK_DIMENSIONS) -> list[float]:
    vector = [0.0] * dimensions
    for feature in _iter_local_features(text):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        index = value % dimensions
        sign = 1.0 if (value >> 8) % 2 == 0 else -1.0
        vector[index] += sign
    return vector


def _iter_local_features(text: str) -> list[str]:
    clean = " ".join((text or "").lower().split())
    if not clean:
        return ["empty"]

    features: list[str] = []
    cjk_chars = [char for char in clean if "\u4e00" <= char <= "\u9fff"]
    features.extend(cjk_chars)
    features.extend(_ngrams(cjk_chars, 2))
    features.extend(_ngrams(cjk_chars, 3))

    words = re.findall(r"[a-z0-9]{2,}", clean)
    features.extend(words)

    compact = re.sub(r"\s+", "", clean)
    features.extend(_ngrams(list(compact), 4))
    return features or ["empty"]


def _ngrams(items: list[str], size: int) -> list[str]:
    if len(items) < size:
        return []
    return ["".join(items[index : index + size]) for index in range(0, len(items) - size + 1)]
