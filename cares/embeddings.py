from __future__ import annotations

import os
from typing import Any

import numpy as np

from .log import get_logger

log = get_logger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-large-en-v1.5"


def _disable_implicit_token() -> None:
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    try:
        import huggingface_hub.constants as constants

        constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
    except ImportError:  # pragma: no cover
        pass


def _is_auth_error(exc: BaseException) -> bool:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}"
        if "401" in text or "Unauthorized" in text or "token has expired" in text:
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def load_encoder(model_name: str = DEFAULT_EMBED_MODEL) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("This stage requires `pip install 'cares[analysis]'`") from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        if not _is_auth_error(exc):
            raise
        log.warning(
            "Hugging Face refused '%s' with a 401: the local token is most "
            "likely expired. The repository is public, retrying without "
            "authentication. For a lasting fix: `huggingface-cli logout`, or "
            "HF_HUB_DISABLE_IMPLICIT_TOKEN=1.",
            model_name)
        _disable_implicit_token()
        return SentenceTransformer(model_name)


def encode(
    model: Any,
    texts: list[str],
    *,
    batch_size: int = 32,
    instruction: str = "",
    show_progress: bool = False,
) -> np.ndarray:
    if instruction:
        texts = [instruction + t for t in texts]
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )


def cosine(model: Any, a: str, b: str) -> float | None:
    if model is None or not a or not b:
        return None
    embs = model.encode([a, b], convert_to_numpy=True, show_progress_bar=False)
    va = np.asarray(embs[0], dtype="float64")
    vb = np.asarray(embs[1], dtype="float64")
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return None
    return float(np.dot(va, vb) / (na * nb))
