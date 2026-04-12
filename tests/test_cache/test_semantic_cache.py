"""Tests for semantic and prompt caches."""

import pytest
from twotrim.cache.prompt_cache import PromptCache
from twotrim.types import CompressionResult


class TestPromptCache:

    @pytest.fixture
    def cache(self, tmp_path):
        return PromptCache(db_path=str(tmp_path / "test_cache.db"))

    @pytest.mark.asyncio
    async def test_store_and_lookup(self, cache):
        result = CompressionResult(
            original_text="original test text",
            compressed_text="compressed",
            original_tokens=10,
            compressed_tokens=5,
            overall_ratio=0.5,
        )
        await cache.store("original test text", result, "balanced")
        lookup = await cache.lookup("original test text", "balanced")
        assert lookup is not None
        assert lookup.compressed_text == "compressed"
        assert lookup.overall_ratio == 0.5

    @pytest.mark.asyncio
    async def test_miss(self, cache):
        lookup = await cache.lookup("nonexistent text", "balanced")
        assert lookup is None

    @pytest.mark.asyncio
    async def test_different_modes(self, cache):
        result = CompressionResult(
            original_text="test",
            compressed_text="t",
            original_tokens=5,
            compressed_tokens=2,
            overall_ratio=0.6,
        )
        await cache.store("test", result, "balanced")
        # Same text, different mode = miss
        lookup = await cache.lookup("test", "aggressive")
        assert lookup is None

    @pytest.mark.asyncio
    async def test_stats(self, cache):
        result = CompressionResult(
            original_text="test",
            compressed_text="t",
            original_tokens=5,
            compressed_tokens=2,
            overall_ratio=0.6,
        )
        await cache.store("test", result, "balanced")
        await cache.lookup("test", "balanced")

        s = await cache.stats()
        assert s["entries"] == 1
        assert s["total_hits"] >= 1

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        result = CompressionResult(
            original_text="test",
            compressed_text="t",
            original_tokens=5,
            compressed_tokens=2,
            overall_ratio=0.6,
        )
        await cache.store("test", result, "balanced")
        await cache.clear()
        lookup = await cache.lookup("test", "balanced")
        assert lookup is None
