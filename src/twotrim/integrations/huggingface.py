"""HuggingFace model management — lazy loading and caching."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_models: dict[str, Any] = {}


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2", device: str = "auto") -> Any:
    """Get or load a sentence-transformers embedding model."""
    key = f"emb:{model_name}"
    if key in _models:
        return _models[key]

    try:
        from sentence_transformers import SentenceTransformer
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = SentenceTransformer(model_name, device=device)
        _models[key] = model
        logger.info("Loaded embedding model: %s on %s", model_name, device)
        return model
    except Exception as e:
        logger.warning("Failed to load embedding model %s: %s", model_name, e)
        return None


def get_summarization_pipeline(model_name: str = "facebook/bart-large-cnn", device: str = "auto") -> Any:
    """Get or load a summarization pipeline."""
    key = f"sum:{model_name}"
    if key in _models:
        return _models[key]

    try:
        import torch
        from transformers import pipeline

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe = pipeline(
            "summarization",
            model=model_name,
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        _models[key] = pipe
        logger.info("Loaded summarization model: %s on %s", model_name, device)
        return pipe
    except Exception as e:
        logger.warning("Failed to load summarization model %s: %s", model_name, e)
        return None


def get_device() -> str:
    """Detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def clear_models() -> None:
    """Unload all cached models."""
    _models.clear()
    logger.info("All HuggingFace models unloaded")
