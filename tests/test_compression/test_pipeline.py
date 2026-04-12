"""Tests for compression pipeline."""

import pytest
from twotrim.compression.pipeline import CompressionPipeline
from twotrim.types import CompressionMode, PolicyDecision, StrategyName


@pytest.fixture
def pipeline():
    return CompressionPipeline()


class TestCompressionPipeline:

    @pytest.mark.asyncio
    async def test_lossless_compression(self, pipeline, sample_text):
        decision = PolicyDecision(
            mode=CompressionMode.LOSSLESS,
            strategies=[StrategyName.RULE_BASED, StrategyName.CANONICALIZE],
            target_reduction=0.10,
            max_degradation=0.01,
        )
        result = await pipeline.compress(sample_text, decision)
        assert result.compressed_tokens <= result.original_tokens
        assert result.overall_ratio >= 0
        assert len(result.strategies_applied) > 0

    @pytest.mark.asyncio
    async def test_empty_strategies(self, pipeline, sample_text):
        decision = PolicyDecision(
            mode=CompressionMode.LOSSLESS,
            strategies=[],
            target_reduction=0.0,
            max_degradation=0.0,
        )
        result = await pipeline.compress(sample_text, decision)
        assert result.compressed_text == sample_text
        assert result.overall_ratio == 0.0

    @pytest.mark.asyncio
    async def test_short_text(self, pipeline, short_text):
        decision = PolicyDecision(
            mode=CompressionMode.BALANCED,
            strategies=[StrategyName.RULE_BASED],
            target_reduction=0.40,
            max_degradation=0.05,
        )
        result = await pipeline.compress(short_text, decision)
        # Short text shouldn't change much
        assert result.original_tokens > 0

    @pytest.mark.asyncio
    async def test_compression_time_tracked(self, pipeline, sample_text):
        decision = PolicyDecision(
            mode=CompressionMode.LOSSLESS,
            strategies=[StrategyName.RULE_BASED],
            target_reduction=0.10,
            max_degradation=0.01,
        )
        result = await pipeline.compress(sample_text, decision)
        assert result.compression_time_ms > 0

    @pytest.mark.asyncio
    async def test_strategy_results_populated(self, pipeline, sample_text):
        decision = PolicyDecision(
            mode=CompressionMode.LOSSLESS,
            strategies=[StrategyName.RULE_BASED, StrategyName.CANONICALIZE],
            target_reduction=0.10,
            max_degradation=0.01,
        )
        result = await pipeline.compress(sample_text, decision)
        for sr in result.strategies_applied:
            assert sr.strategy in (StrategyName.RULE_BASED, StrategyName.CANONICALIZE)
            assert sr.original_tokens > 0
