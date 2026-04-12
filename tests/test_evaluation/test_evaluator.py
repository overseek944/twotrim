"""Tests for evaluation system."""

import pytest
from twotrim.evaluation.evaluator import Evaluator
from twotrim.evaluation.similarity import SimilarityScorer
from twotrim.evaluation.guardrails import Guardrails
from twotrim.types import CompressionResult, EvalResult


class TestSimilarityScorer:

    def test_identical_texts(self):
        scorer = SimilarityScorer()
        score = scorer.score("Hello world", "Hello world")
        assert score == 1.0

    def test_empty_text(self):
        scorer = SimilarityScorer()
        score = scorer.score("", "Hello")
        assert score == 0.0

    def test_similar_texts(self):
        scorer = SimilarityScorer()
        score = scorer.score(
            "Machine learning is a field of AI",
            "ML is an area of artificial intelligence"
        )
        assert 0 < score <= 1.0

    def test_different_texts(self):
        scorer = SimilarityScorer()
        score = scorer.score(
            "Python is a programming language",
            "The weather is sunny today"
        )
        # Should be relatively low
        assert score < 0.8


class TestGuardrails:

    def test_pass(self):
        guardrails = Guardrails(similarity_threshold=0.9)
        result = EvalResult(
            request_id="test",
            similarity_score=0.95,
            passed=True,
            threshold=0.9,
            original_tokens=100,
            compressed_tokens=60,
            compression_ratio=0.4,
        )
        assert guardrails.check(result) is True
        assert guardrails.violation_count == 0

    def test_violation(self):
        guardrails = Guardrails(similarity_threshold=0.9)
        result = EvalResult(
            request_id="test",
            similarity_score=0.7,
            passed=False,
            threshold=0.9,
            original_tokens=100,
            compressed_tokens=30,
            compression_ratio=0.7,
        )
        assert guardrails.check(result) is False
        assert guardrails.violation_count == 1

    def test_rollback_detection(self):
        guardrails = Guardrails(similarity_threshold=0.9, max_degradation=0.05)
        for i in range(5):
            guardrails.check(EvalResult(
                request_id=f"test_{i}",
                similarity_score=0.5,
                passed=False,
                threshold=0.9,
                original_tokens=100,
                compressed_tokens=30,
                compression_ratio=0.7,
            ))
        assert guardrails.should_rollback(window=5)


class TestEvaluator:

    @pytest.mark.asyncio
    async def test_evaluate_identical(self):
        evaluator = Evaluator()
        result = CompressionResult(
            original_text="Hello world test prompt",
            compressed_text="Hello world test prompt",
            original_tokens=10,
            compressed_tokens=10,
            overall_ratio=0.0,
        )
        # No compression = skip
        eval_result = await evaluator.evaluate(result)
        assert eval_result is None

    @pytest.mark.asyncio
    async def test_evaluate_compressed(self):
        evaluator = Evaluator()
        result = CompressionResult(
            original_text="Machine learning is a field of artificial intelligence that enables systems to learn from data.",
            compressed_text="ML is an AI field enabling data-driven learning.",
            original_tokens=20,
            compressed_tokens=10,
            overall_ratio=0.5,
        )
        eval_result = await evaluator.evaluate(result)
        assert eval_result is not None
        assert 0 <= eval_result.similarity_score <= 1.0

    def test_should_evaluate_sampling(self):
        evaluator = Evaluator()
        # With 10% sample rate, should sometimes return True
        results = [evaluator.should_evaluate() for _ in range(100)]
        assert any(results)  # At least one should be True
        assert not all(results)  # Not all should be True
