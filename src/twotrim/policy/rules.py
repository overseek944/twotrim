"""Rule-based policy selection.

Determines compression mode based on model, request type,
token count, and other signals.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from twotrim.types import CompressionMode


def classify_request_type(messages: list[dict[str, Any]] | None = None, prompt: str | None = None) -> str:
    """Classify the request type from message content."""
    text = ""
    if messages:
        for m in messages:
            content = m.get("content", "")
            if content:
                text += " " + content
    if prompt:
        text += " " + prompt

    text = text.lower()

    # Coding indicators
    coding_signals = ["code", "function", "class", "debug", "error", "bug", "implement",
                      "refactor", "```", "def ", "import ", "return "]
    if sum(1 for s in coding_signals if s in text) >= 2:
        return "coding"

    # Reasoning indicators
    reasoning_signals = ["step by step", "reason", "think", "analyze", "prove",
                        "derive", "calculate", "math", "logic"]
    if sum(1 for s in reasoning_signals if s in text) >= 2:
        return "reasoning"

    # Summarization indicators
    if any(s in text for s in ["summarize", "summary", "tldr", "brief overview", "condense"]):
        return "summarization"

    # Translation
    if any(s in text for s in ["translate", "translation"]):
        return "translation"

    # Data extraction
    if any(s in text for s in ["extract", "parse", "convert to json", "get the data"]):
        return "data_extraction"

    # Creative
    if any(s in text for s in ["write a story", "creative", "poem", "fiction", "imagine"]):
        return "creative"

    return "chat"


def select_mode_for_model(
    model: str,
    per_model_config: dict[str, dict[str, Any]],
) -> CompressionMode | None:
    """Select compression mode based on model name rules."""
    for pattern, config in per_model_config.items():
        if fnmatch.fnmatch(model, pattern):
            mode_str = config.get("mode", "balanced")
            try:
                return CompressionMode(mode_str)
            except ValueError:
                return CompressionMode.BALANCED
    return None


def should_skip_compression(
    model: str,
    token_count: int,
    request_type: str,
) -> bool:
    """Determine if compression should be skipped entirely."""
    # Very short prompts — overhead not worth it
    if token_count < 50:
        return True

    # Embedding requests — don't compress
    if "embed" in model.lower():
        return True

    return False


def estimate_token_count(messages: list[dict[str, Any]] | None = None, prompt: str | None = None) -> int:
    """Quick estimate of total token count."""
    text = ""
    if messages:
        for m in messages:
            content = m.get("content", "")
            if content:
                text += content + " "
    if prompt:
        text += prompt

    # Rough: 1 token ≈ 4 chars for English
    return max(1, len(text) // 4)
