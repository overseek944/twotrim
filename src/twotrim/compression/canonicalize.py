"""Prompt canonicalization — map verbose prompts to minimal equivalents.

Maintains a library of canonical prompt templates and uses fuzzy
matching to detect and replace verbose prompt patterns.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from twotrim.types import StrategyName, StrategyResult

logger = logging.getLogger(__name__)

# Built-in canonical templates: (verbose_pattern, canonical_form)
BUILTIN_TEMPLATES: list[tuple[str, str]] = [
    (
        "You are a helpful assistant. Please answer the following question "
        "to the best of your ability, providing detailed and accurate information.",
        "Answer accurately and in detail.",
    ),
    (
        "Please provide a comprehensive and detailed explanation of",
        "Explain:",
    ),
    (
        "Can you help me understand",
        "Explain:",
    ),
    (
        "I would like you to summarize the following text. "
        "Please provide a concise summary that captures the main points.",
        "Summarize concisely:",
    ),
    (
        "Please translate the following text from English to",
        "Translate to",
    ),
    (
        "You are an expert programmer. Please review the following code "
        "and provide suggestions for improvement.",
        "Review this code and suggest improvements:",
    ),
    (
        "Please analyze the following data and provide insights",
        "Analyze and provide insights:",
    ),
    (
        "I need you to write a",
        "Write a",
    ),
    (
        "Could you please help me with",
        "Help with:",
    ),
    (
        "I want you to act as",
        "Act as",
    ),
]


class PromptCanonicalizer:
    """Map verbose prompts to minimal canonical equivalents."""

    def __init__(
        self,
        template_dir: str | None = None,
        similarity_threshold: float = 0.70,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.name = StrategyName.CANONICALIZE
        self._templates: list[tuple[str, str]] = list(BUILTIN_TEMPLATES)
        if template_dir:
            self._load_templates(template_dir)

    def _load_templates(self, template_dir: str) -> None:
        """Load custom templates from a directory."""
        from pathlib import Path
        import json

        p = Path(template_dir)
        if not p.exists():
            logger.warning("Template directory not found: %s", template_dir)
            return

        for f in p.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                for entry in data:
                    if "verbose" in entry and "canonical" in entry:
                        self._templates.append((entry["verbose"], entry["canonical"]))
            except Exception as e:
                logger.warning("Failed to load template file %s: %s", f, e)

    async def compress(
        self,
        text: str,
        token_counter: object | None = None,
        query: str | None = None,
        target_reduction: float = 0.0,
    ) -> StrategyResult:
        """Apply canonicalization to prompt text."""
        original_tokens = _count_tokens(text, token_counter)
        compressed = self._canonicalize(text)
        compressed_tokens = _count_tokens(compressed, token_counter)

        return StrategyResult(
            strategy=self.name,
            original_text=text,
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1 - (compressed_tokens / max(original_tokens, 1)),
            metadata={"templates_matched": compressed != text},
        )

    async def estimate_reduction(self, text: str) -> float:
        return 0.10

    def _canonicalize(self, text: str) -> str:
        """Apply template matching and common pattern reduction."""
        result = text

        # Apply template matching
        for verbose, canonical in self._templates:
            result = self._fuzzy_replace(result, verbose, canonical)

        # Apply general reduction patterns
        result = self._reduce_instruction_verbosity(result)
        result = self._compress_system_prompts(result)

        return result

    def _fuzzy_replace(self, text: str, pattern: str, replacement: str) -> str:
        """Replace text matching pattern with replacement using fuzzy matching."""
        pattern_lower = pattern.lower()

        # Try exact substring match first
        idx = text.lower().find(pattern_lower)
        if idx != -1:
            return text[:idx] + replacement + text[idx + len(pattern):]

        # Try fuzzy match on text windows
        pattern_words = pattern_lower.split()
        text_words = text.split()

        if len(pattern_words) > len(text_words):
            return text

        best_score = 0.0
        best_start = -1
        best_end = -1
        window_size = len(pattern_words)

        for i in range(len(text_words) - window_size + 1):
            window = " ".join(text_words[i:i + window_size]).lower()
            score = SequenceMatcher(None, pattern_lower, window).ratio()
            if score > best_score:
                best_score = score
                best_start = i
                best_end = i + window_size

        if best_score >= self.similarity_threshold and best_start >= 0:
            before = " ".join(text_words[:best_start])
            after = " ".join(text_words[best_end:])
            parts = [p for p in [before, replacement, after] if p]
            return " ".join(parts)

        return text

    def _reduce_instruction_verbosity(self, text: str) -> str:
        """Reduce verbose instruction patterns."""
        reductions = [
            (r"Please (?:make sure to |ensure that you )?provide (?:a |an )?", "Give "),
            (r"It is (?:very )?important (?:that|to) ", ""),
            (r"Make sure (?:that )?you ", ""),
            (r"I would (?:really )?(?:like|appreciate) (?:it )?if you (?:could|would) ", ""),
            (r"Please note that ", "Note: "),
            (r"Keep in mind that ", "Note: "),
            (r"(?:Remember|Don't forget) (?:that |to )", ""),
        ]

        for pattern, repl in reductions:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        return text

    def _compress_system_prompts(self, text: str) -> str:
        """Compress common system prompt verbosity."""
        reductions = [
            (
                r"You are an? (?:very )?(?:helpful|knowledgeable|experienced|expert) "
                r"(?:AI )?assistant(?:\.|,)?\s*",
                "",
            ),
            (
                r"Your (?:primary |main )?(?:goal|task|objective|job|role) is to\s+",
                "",
            ),
            (
                r"When (?:answering|responding)(?: to questions)?,?\s*"
                r"(?:please )?(?:make sure to |ensure that you )?\s*",
                "",
            ),
        ]

        for pattern, repl in reductions:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        return text.strip()

    def add_template(self, verbose: str, canonical: str) -> None:
        """Add a custom template at runtime."""
        self._templates.append((verbose, canonical))


def _count_tokens(text: str, counter: object | None = None) -> int:
    if counter is not None and hasattr(counter, "count"):
        return counter.count(text)
    return max(1, int(len(text.split()) / 0.75))
