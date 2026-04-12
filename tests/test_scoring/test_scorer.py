"""Tests for token importance scoring."""

import pytest
from twotrim.scoring.scorer import ImportanceScorer
from twotrim.scoring.frequency import score_by_frequency, compute_redundancy
from twotrim.scoring.heuristics import positional_scores, entity_scores


class TestFrequencyScoring:

    def test_basic_scoring(self):
        scores = score_by_frequency("The cat sat on the mat. The dog sat on the mat.")
        assert "the" in scores
        assert "cat" in scores
        assert all(0 <= v <= 1 for v in scores.values())

    def test_empty_text(self):
        scores = score_by_frequency("")
        assert scores == {}

    def test_redundancy_detection(self):
        sentences = [
            "Machine learning is great.",
            "Machine learning is wonderful.",
            "The weather is nice today.",
        ]
        redundancy = compute_redundancy(sentences)
        # First two should have higher redundancy
        assert redundancy[0] > redundancy[2] or redundancy[1] > redundancy[2]


class TestHeuristics:

    def test_positional_first_last_important(self):
        sentences = ["First.", "Middle.", "Middle.", "Middle.", "Last."]
        scores = positional_scores(sentences)
        assert scores[0] >= scores[2]
        assert scores[-1] >= scores[2]

    def test_entity_scores(self):
        sentences = [
            "John Smith visited New York on January 15th.",
            "the quick brown fox jumped.",
        ]
        scores = entity_scores(sentences)
        assert scores[0] > scores[1]

    def test_single_sentence(self):
        scores = positional_scores(["Only one."])
        assert scores == [1.0]


class TestImportanceScorer:

    @pytest.mark.asyncio
    async def test_basic_scoring(self, sample_text):
        scorer = ImportanceScorer()
        result = await scorer.score(sample_text)
        assert len(result.segments) > 0
        assert result.mean_score > 0
        assert result.min_score <= result.max_score

    @pytest.mark.asyncio
    async def test_filter_by_importance(self, sample_text):
        scorer = ImportanceScorer()
        filtered = await scorer.filter_by_importance(sample_text, min_score=0.5)
        assert len(filtered) > 0
        assert len(filtered) <= len(sample_text)

    @pytest.mark.asyncio
    async def test_empty_text(self):
        scorer = ImportanceScorer()
        result = await scorer.score("")
        assert len(result.segments) == 0
