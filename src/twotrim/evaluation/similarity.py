"""Semantic similarity scoring for evaluation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SimilarityScorer:
    """Compute semantic similarity between original and compressed text."""

    def __init__(self, model: Any = None) -> None:
        self._model = model
        self._available: bool | None = None

    def _ensure_model(self) -> bool:
        if self._available is not None:
            return self._available
        if self._model is not None:
            self._available = True
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._available = True
        except Exception as e:
            logger.warning("Similarity scorer model unavailable: %s", e)
            self._available = False
        return self._available

    def score(self, original: str, compressed: str) -> float:
        """Compute semantic similarity [0, 1] between two texts."""
        if original == compressed:
            return 1.0

        if not original or not compressed:
            return 0.0

        if self._ensure_model() and self._model is not None:
            return self._embedding_similarity(original, compressed)
        return self._lexical_similarity(original, compressed)

    def _embedding_similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity via embeddings."""
        try:
            embeddings = self._model.encode([text_a, text_b], convert_to_numpy=True, show_progress_bar=False)
            emb_a = embeddings[0] / (np.linalg.norm(embeddings[0]) + 1e-8)
            emb_b = embeddings[1] / (np.linalg.norm(embeddings[1]) + 1e-8)
            return float(np.dot(emb_a, emb_b))
        except Exception as e:
            logger.warning("Embedding similarity failed: %s", e)
            return self._lexical_similarity(text_a, text_b)

    def _lexical_similarity(self, text_a: str, text_b: str) -> float:
        """Fallback: Jaccard word overlap similarity."""
        import re
        words_a = set(re.findall(r"\w+", text_a.lower()))
        words_b = set(re.findall(r"\w+", text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0
