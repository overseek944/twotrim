"""Gradient-free saliency heuristics.

Positional importance, named entity preservation, and structural cues.
"""

from __future__ import annotations

import re


def positional_scores(sentences: list[str]) -> list[float]:
    """Score sentences by position — first/last are most important."""
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    scores: list[float] = []
    for i in range(n):
        if i == 0:
            score = 1.0
        elif i == n - 1:
            score = 0.85
        elif i < 3:
            score = 0.7 - (i * 0.1)
        elif i >= n - 3:
            score = 0.6 + ((n - i) * 0.05)
        else:
            score = 0.3 + (0.2 * (1 - abs(i - n / 2) / (n / 2)))

        scores.append(score)

    return scores


def entity_scores(sentences: list[str]) -> list[float]:
    """Score sentences by presence of named entities and key terms.

    Uses regex-based NER approximation (no model required).
    """
    scores: list[float] = []

    for sentence in sentences:
        score = 0.0
        words = sentence.split()
        n_words = max(len(words), 1)

        # Capitalized words (potential proper nouns) — not at sentence start
        caps = [w for w in words[1:] if w[0].isupper() and len(w) > 1] if len(words) > 1 else []
        score += len(caps) * 0.15

        # Numbers and measurements
        numbers = re.findall(r"\b\d+(?:\.\d+)?(?:%|km|mb|gb|tb|ms|s|m|kg|lb)?\b", sentence, re.I)
        score += len(numbers) * 0.1

        # Dates
        dates = re.findall(
            r"\b(?:\d{4}[-/]\d{2}[-/]\d{2}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2})\b",
            sentence, re.I
        )
        score += len(dates) * 0.2

        # URLs and emails
        urls = re.findall(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", sentence)
        score += len(urls) * 0.15

        # Technical terms (camelCase, snake_case, ACRONYMS)
        technical = re.findall(r"\b(?:[a-z]+[A-Z]\w+|[a-z]+_[a-z]+|[A-Z]{2,})\b", sentence)
        score += len(technical) * 0.1

        # Quoted strings
        quotes = re.findall(r'"[^"]{2,}"', sentence)
        score += len(quotes) * 0.1

        # Normalize by sentence length
        score = min(1.0, score / max(1, n_words / 10))
        scores.append(score)

    return scores


def structural_scores(text: str) -> dict[str, float]:
    """Score structural elements of text."""
    scores: dict[str, float] = {}

    # Headers get high importance
    for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
        level = len(match.group(1))
        scores[match.group(2).strip()] = 1.0 - (level - 1) * 0.1

    # List items get moderate importance
    for match in re.finditer(r"^[\-\*•]\s+(.+)$", text, re.MULTILINE):
        scores[match.group(1).strip()] = 0.6

    # Code blocks get high importance
    for match in re.finditer(r"```[\s\S]*?```", text):
        scores[match.group(0)[:50]] = 0.9

    return scores
