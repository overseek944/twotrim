"""Rule-based compression — zero-cost, lossless text reduction.

Applies deterministic transformations: whitespace normalization,
deduplication, filler removal, unicode normalization.
"""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha256

import regex

from twotrim.types import CompressionResult, StrategyName, StrategyResult

# ---------------------------------------------------------------------------
# Filler words and phrases that can be safely removed
# ---------------------------------------------------------------------------

FILLER_WORDS: set[str] = {
    "basically", "essentially", "actually", "literally", "obviously",
    "clearly", "simply", "just", "really", "very", "quite",
    "pretty much", "kind of", "sort of", "in fact",
    "as a matter of fact", "it is worth noting that",
    "it should be noted that", "it is important to note that",
    "needless to say", "it goes without saying",
    "as you can see", "as we can see", "as mentioned earlier",
    "as previously mentioned", "as stated above", "as noted above",
    "in other words", "that is to say", "to put it simply",
    "to put it another way", "what this means is",
}

FILLER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b,?\s*",
               re.IGNORECASE),
]

# Redundant preamble patterns
PREAMBLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(Sure[!,.]?\s*(Here'?s?|I'?d be happy to|Let me)\s.*?[:.]\s*)", re.IGNORECASE),
    re.compile(r"^(Of course[!,.]?\s*.*?[:.]\s*)", re.IGNORECASE),
    re.compile(r"^(Absolutely[!,.]?\s*.*?[:.]\s*)", re.IGNORECASE),
    re.compile(r"^(Great question[!,.]?\s*)", re.IGNORECASE),
]


class RuleBasedCompressor:
    """Deterministic, zero-cost compression using text transformation rules."""

    def __init__(
        self,
        remove_filler_words: bool = True,
        collapse_whitespace: bool = True,
        deduplicate_sentences: bool = True,
        normalize_unicode: bool = True,
    ) -> None:
        self.remove_filler_words = remove_filler_words
        self.collapse_whitespace = collapse_whitespace
        self.deduplicate_sentences = deduplicate_sentences
        self.normalize_unicode = normalize_unicode
        self.name = StrategyName.RULE_BASED

    async def compress(
        self,
        text: str,
        token_counter: object | None = None,
        query: str | None = None,
        target_reduction: float = 0.0,
    ) -> StrategyResult:
        """Apply all enabled rule-based transformations."""
        original = text
        original_tokens = _count_tokens(original, token_counter)

        if self.normalize_unicode:
            text = self._normalize_unicode(text)

        if self.collapse_whitespace:
            text = self._collapse_whitespace(text)

        if self.deduplicate_sentences:
            text = self._deduplicate_sentences(text)

        if self.remove_filler_words:
            text = self._remove_fillers(text)

        # Clean up any artifacts
        text = self._final_cleanup(text)

        compressed_tokens = _count_tokens(text, token_counter)

        return StrategyResult(
            strategy=self.name,
            original_text=original,
            compressed_text=text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1 - (compressed_tokens / max(original_tokens, 1)),
        )

    async def estimate_reduction(self, text: str) -> float:
        """Estimate compression ratio without actually compressing."""
        filler_count = sum(len(p.findall(text)) for p in FILLER_PATTERNS)
        whitespace_excess = len(text) - len(" ".join(text.split()))
        total_reduction = (filler_count * 8 + whitespace_excess) / max(len(text), 1)
        return min(total_reduction, 0.20)  # cap estimate at 20%

    # -----------------------------------------------------------------------
    # Transformation methods
    # -----------------------------------------------------------------------

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode to NFC form, replace fancy quotes etc."""
        text = unicodedata.normalize("NFC", text)
        # Replace smart quotes with standard quotes
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        # Replace em/en dashes with hyphens
        text = text.replace("\u2014", " - ").replace("\u2013", "-")
        # Replace ellipsis character
        text = text.replace("\u2026", "...")
        # Replace non-breaking spaces
        text = text.replace("\u00a0", " ")
        return text

    def _collapse_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into singles."""
        # Preserve code blocks
        parts = _split_preserving_code(text)
        result = []
        for part, is_code in parts:
            if is_code:
                result.append(part)
            else:
                # Collapse multiple blank lines to one
                part = re.sub(r"\n{3,}", "\n\n", part)
                # Collapse multiple spaces to one
                part = re.sub(r"[ \t]+", " ", part)
                # Remove trailing whitespace per line
                part = re.sub(r" +\n", "\n", part)
                result.append(part)
        return "".join(result)

    def _deduplicate_sentences(self, text: str) -> str:
        """Remove duplicate or near-duplicate sentences."""
        parts = _split_preserving_code(text)
        result = []
        for part, is_code in parts:
            if is_code:
                result.append(part)
            else:
                result.append(self._dedup_text_block(part))
        return "".join(result)

    def _dedup_text_block(self, text: str) -> str:
        """Deduplicate sentences within a text block."""
        # Split into sentences
        sentences = regex.split(r"(?<=[.!?])\s+", text)
        if len(sentences) <= 1:
            return text

        seen_hashes: set[str] = set()
        unique_sentences: list[str] = []

        for sentence in sentences:
            # Normalize for comparison: lowercase, strip punctuation
            normalized = re.sub(r"[^\w\s]", "", sentence.lower()).strip()
            if not normalized:
                unique_sentences.append(sentence)
                continue

            h = sha256(normalized.encode()).hexdigest()[:16]
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_sentences.append(sentence)

        return " ".join(unique_sentences)

    def _remove_fillers(self, text: str) -> str:
        """Remove filler words and redundant phrases."""
        parts = _split_preserving_code(text)
        result = []
        for part, is_code in parts:
            if is_code:
                result.append(part)
            else:
                for pattern in FILLER_PATTERNS:
                    part = pattern.sub("", part)
                for pattern in PREAMBLE_PATTERNS:
                    part = pattern.sub("", part, count=1)
                result.append(part)
        return "".join(result)

    def _final_cleanup(self, text: str) -> str:
        """Final pass to clean up transformation artifacts."""
        # Remove double spaces
        text = re.sub(r"  +", " ", text)
        # Remove space before punctuation
        text = re.sub(r" ([.,;:!?])", r"\1", text)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        # Remove leading/trailing whitespace overall
        return text.strip()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _split_preserving_code(text: str) -> list[tuple[str, bool]]:
    """Split text into (content, is_code) segments, preserving code blocks."""
    parts: list[tuple[str, bool]] = []
    # Match fenced code blocks (``` ... ```)
    pattern = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
    last_end = 0
    for match in pattern.finditer(text):
        if match.start() > last_end:
            parts.append((text[last_end:match.start()], False))
        parts.append((match.group(0), True))
        last_end = match.end()
    if last_end < len(text):
        parts.append((text[last_end:], False))
    if not parts:
        parts.append((text, False))
    return parts


def _count_tokens(text: str, counter: object | None = None) -> int:
    """Count tokens using provided counter or fall back to word-split estimate."""
    if counter is not None and hasattr(counter, "count"):
        return counter.count(text)  # type: ignore[union-attr]
    # Rough estimate: ~0.75 words per token for English
    return max(1, int(len(text.split()) / 0.75))
