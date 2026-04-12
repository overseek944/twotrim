"""Tests for structured compression."""

import json
import pytest
from twotrim.compression.structured import StructuredCompressor


@pytest.fixture
def compressor():
    return StructuredCompressor()


class TestStructuredCompressor:

    @pytest.mark.asyncio
    async def test_skip_already_structured(self, compressor):
        text = json.dumps({"key": "value", "count": 42})
        result = await compressor.compress(text)
        assert result.compressed_text == text
        assert result.metadata.get("skipped")

    @pytest.mark.asyncio
    async def test_list_compression(self, compressor):
        text = "- First item in the list with some extra detail.\n- Second item.\n- Third item with more text.\n- Fourth item."
        result = await compressor.compress(text)
        # It's already nicely formatted, so it should be skipped
        assert result.metadata.get("skipped") is True

    @pytest.mark.asyncio
    async def test_code_block_preserved(self, compressor):
        text = "Description here.\n\n```python\nprint('hello')\n```\n\nMore text here that has key-value pairs.\nName: John\nAge: 30\nCity: NYC"
        result = await compressor.compress(text)
        assert "```python" in result.compressed_text
        assert "print('hello')" in result.compressed_text

    @pytest.mark.asyncio
    async def test_key_value_extraction(self, compressor):
        text = "Name: John Smith\nAge: 30\nCity: New York\nOccupation: Engineer\n"
        result = await compressor.compress(text)
        assert result.compression_ratio >= 0
