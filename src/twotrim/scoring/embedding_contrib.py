"""Embedding contribution scoring.

Measures how much each segment contributes to the overall
semantic meaning of the text based on embedding distances.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def score_by_embedding_contribution(
    segments: list[str],
    model: Any = None,
) -> list[float]:
    """Score each segment by its contribution to overall meaning.

    Computes the embedding of the full text and each leave-one-out
    variant, scoring segments by how much the embedding changes
    when they are removed.

    Args:
        segments: Text segments to score.
        model: A sentence-transformers model. If None, falls back
               to uniform scores.

    Returns:
        List of scores [0, 1], one per segment.
    """
    if len(segments) <= 1:
        return [1.0] * len(segments)

    if model is None:
        return _fallback_scoring(segments)

    try:
        return _embedding_scoring(segments, model)
    except Exception as e:
        logger.warning("Embedding contribution scoring failed: %s", e)
        return _fallback_scoring(segments)


def _embedding_scoring(segments: list[str], model: Any) -> list[float]:
    """Score using actual embeddings."""
    full_text = " ".join(segments)
    texts = [full_text] + [
        " ".join(segments[:i] + segments[i + 1:])
        for i in range(len(segments))
    ]

    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    full_emb = embeddings[0]
    full_norm = full_emb / (np.linalg.norm(full_emb) + 1e-8)

    scores: list[float] = []
    for i in range(len(segments)):
        leave_out_emb = embeddings[i + 1]
        leave_out_norm = leave_out_emb / (np.linalg.norm(leave_out_emb) + 1e-8)
        # cosine distance = how much the embedding changes
        similarity = float(np.dot(full_norm, leave_out_norm))
        contribution = 1.0 - similarity  # higher = more important
        scores.append(contribution)

    # Normalize to [0, 1]
    if scores:
        max_s = max(scores) if max(scores) > 0 else 1
        scores = [s / max_s for s in scores]

    return scores


def _fallback_scoring(segments: list[str]) -> list[float]:
    """Fallback: score by length and position heuristics."""
    n = len(segments)
    scores: list[float] = []
    total_len = sum(len(s.split()) for s in segments)

    for i, seg in enumerate(segments):
        seg_len = len(seg.split())
        # Length contribution
        len_score = seg_len / max(total_len, 1)

        # Position boost (first and last are usually important)
        if i == 0:
            pos_score = 1.0
        elif i == n - 1:
            pos_score = 0.8
        else:
            pos_score = 0.5

        scores.append(len_score * 0.4 + pos_score * 0.6)

    # Normalize
    if scores:
        max_s = max(scores) if max(scores) > 0 else 1
        scores = [s / max_s for s in scores]

    return scores
