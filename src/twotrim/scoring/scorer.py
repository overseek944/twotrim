"""Unified token/segment importance scorer.

Combines frequency, embedding, positional, and entity scores
into a single importance score per text segment.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from twotrim.config import get_config
from twotrim.scoring.embedding_contrib import score_by_embedding_contribution
from twotrim.scoring.frequency import compute_redundancy, score_by_frequency
from twotrim.scoring.heuristics import entity_scores, positional_scores
from twotrim.types import ScoringResult, SegmentScore

logger = logging.getLogger(__name__)


class ImportanceScorer:
    """Compute importance scores for text segments."""

    def __init__(self, embedding_model: Any = None) -> None:
        self._model = embedding_model
        self._model_loaded = embedding_model is not None

    async def score(self, text: str) -> ScoringResult:
        """Score all segments in the text."""
        cfg = get_config().scoring

        segments = self._segment(text)
        if not segments:
            return ScoringResult(segments=[], mean_score=0, min_score=0, max_score=0)

        seg_texts = [s[2] for s in segments]

        # Component scores
        freq_scores = self._get_frequency_scores(seg_texts)
        pos_scores = positional_scores(seg_texts)
        ent_scores = entity_scores(seg_texts)
        emb_scores = score_by_embedding_contribution(seg_texts, self._model)
        redundancy = compute_redundancy(seg_texts)

        weights = cfg.weights
        scored: list[SegmentScore] = []

        for i, (start, end, text_seg) in enumerate(segments):
            freq = freq_scores[i] if i < len(freq_scores) else 0.5
            pos = pos_scores[i] if i < len(pos_scores) else 0.5
            ent = ent_scores[i] if i < len(ent_scores) else 0.0
            emb = emb_scores[i] if i < len(emb_scores) else 0.5
            red = redundancy.get(i, 0.0)

            # Weighted combination
            combined = (
                freq * weights.frequency +
                emb * weights.embedding +
                pos * weights.position +
                ent * weights.entity
            )

            # Penalize redundant segments
            combined *= (1 - red * 0.5)

            scored.append(SegmentScore(
                text=text_seg,
                start=start,
                end=end,
                score=round(combined, 4),
                components={
                    "frequency": round(freq, 4),
                    "embedding": round(emb, 4),
                    "position": round(pos, 4),
                    "entity": round(ent, 4),
                    "redundancy": round(red, 4),
                },
            ))

        all_scores = [s.score for s in scored]
        return ScoringResult(
            segments=scored,
            mean_score=round(sum(all_scores) / len(all_scores), 4),
            min_score=round(min(all_scores), 4),
            max_score=round(max(all_scores), 4),
        )

    async def filter_by_importance(
        self, text: str, min_score: float | None = None
    ) -> str:
        """Remove segments below the importance threshold."""
        if min_score is None:
            min_score = get_config().scoring.min_importance

        result = await self.score(text)
        kept = [s.text for s in result.segments if s.score >= min_score]

        if not kept:
            # Always keep at least the top segment
            top = max(result.segments, key=lambda s: s.score)
            kept = [top.text]

        return "\n\n".join(kept)

    def _segment(self, text: str) -> list[tuple[int, int, str]]:
        """Split text into scorable segments with positions."""
        segments: list[tuple[int, int, str]] = []
        pos = 0

        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue

            start = text.find(para, pos)
            if start == -1:
                start = pos
            end = start + len(para)
            pos = end

            # Split long paragraphs into sentences
            if len(para.split()) > 60:
                sent_pos = start
                for sentence in re.split(r"(?<=[.!?])\s+", para):
                    sentence = sentence.strip()
                    if sentence:
                        s_start = text.find(sentence, sent_pos)
                        if s_start == -1:
                            s_start = sent_pos
                        s_end = s_start + len(sentence)
                        segments.append((s_start, s_end, sentence))
                        sent_pos = s_end
            else:
                segments.append((start, end, para))

        return segments

    def _get_frequency_scores(self, segments: list[str]) -> list[float]:
        """Get frequency scores per segment."""
        full_text = " ".join(segments)
        word_scores = score_by_frequency(full_text, segments)

        seg_scores: list[float] = []
        for seg in segments:
            words = re.findall(r"\b\w+\b", seg.lower())
            if not words:
                seg_scores.append(0.0)
                continue
            avg = sum(word_scores.get(w, 0) for w in words) / len(words)
            seg_scores.append(avg)

        return seg_scores
