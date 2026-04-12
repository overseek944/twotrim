"""Tests for rule-based compression."""

import pytest
from twotrim.compression.rule_based import RuleBasedCompressor


@pytest.fixture
def compressor():
    return RuleBasedCompressor()


class TestRuleBasedCompressor:

    @pytest.mark.asyncio
    async def test_whitespace_collapse(self, compressor):
        text = "Hello    world.   This   has   extra    spaces."
        result = await compressor.compress(text)
        assert "    " not in result.compressed_text
        assert result.compression_ratio >= 0

    @pytest.mark.asyncio
    async def test_filler_removal(self, compressor):
        text = "Basically, machine learning is essentially a method of training algorithms."
        result = await compressor.compress(text)
        assert "basically" not in result.compressed_text.lower()
        assert "essentially" not in result.compressed_text.lower()
        assert result.compression_ratio > 0

    @pytest.mark.asyncio
    async def test_duplicate_sentence_removal(self, compressor):
        text = "The sky is blue. The grass is green. The sky is blue. The sun is bright."
        result = await compressor.compress(text)
        # Should appear only once
        assert result.compressed_text.lower().count("the sky is blue") == 1
        assert result.compression_ratio > 0

    @pytest.mark.asyncio
    async def test_unicode_normalization(self, compressor):
        text = "Here\u2019s a \u201csmart quote\u201d test with an em\u2014dash."
        result = await compressor.compress(text)
        assert "\u2019" not in result.compressed_text
        assert "\u201c" not in result.compressed_text

    @pytest.mark.asyncio
    async def test_code_block_preservation(self, compressor):
        text = "Here is code:\n\n```python\ndef foo():\n    x = 1\n    x = 1\n```\n\nEnd."
        result = await compressor.compress(text)
        assert "```python" in result.compressed_text
        assert "def foo():" in result.compressed_text

    @pytest.mark.asyncio
    async def test_empty_text(self, compressor):
        result = await compressor.compress("")
        assert result.compressed_text == ""

    @pytest.mark.asyncio
    async def test_short_text_minimal_change(self, compressor):
        text = "Hello world."
        result = await compressor.compress(text)
        assert result.compressed_text == "Hello world."

    @pytest.mark.asyncio
    async def test_preamble_removal(self, compressor):
        text = "Sure! Here's the answer: The capital of France is Paris."
        result = await compressor.compress(text)
        assert result.compression_ratio > 0

    @pytest.mark.asyncio
    async def test_multiple_blank_lines(self, compressor):
        text = "Line 1.\n\n\n\n\nLine 2.\n\n\n\nLine 3."
        result = await compressor.compress(text)
        assert "\n\n\n" not in result.compressed_text

    @pytest.mark.asyncio
    async def test_estimate_reduction(self, compressor):
        text = "Basically, this is essentially a very simple test."
        est = await compressor.estimate_reduction(text)
        assert 0 <= est <= 1.0
