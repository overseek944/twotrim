"""Frequency-based importance scoring using TF-IDF principles."""

from __future__ import annotations

import math
import re
from collections import Counter


def score_by_frequency(text: str, corpus: list[str] | None = None) -> dict[str, float]:
    """Score words by TF-IDF-like importance.

    Args:
        text: The text to score.
        corpus: Optional corpus of documents for IDF calculation.
                If None, uses paragraphs within the text.

    Returns:
        Mapping of word -> importance score [0, 1].
    """
    words = _tokenize(text)
    if not words:
        return {}

    # Term frequency
    tf = Counter(words)
    max_tf = max(tf.values())
    tf_norm = {w: c / max_tf for w, c in tf.items()}

    # IDF from corpus or self-paragraphs
    if corpus is None:
        corpus = [p for p in text.split("\n\n") if p.strip()]
    if len(corpus) < 2:
        corpus = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    n_docs = len(corpus) if corpus else 1
    doc_freq: Counter[str] = Counter()
    for doc in corpus:
        doc_words = set(_tokenize(doc))
        for w in doc_words:
            doc_freq[w] += 1

    scores: dict[str, float] = {}
    for word in set(words):
        tf_val = tf_norm.get(word, 0)
        df = doc_freq.get(word, 1)
        idf = math.log(1 + n_docs / df)
        scores[word] = tf_val * idf

    # Normalize to [0, 1]
    if scores:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {w: s / max_score for w, s in scores.items()}

    return scores


def compute_redundancy(sentences: list[str]) -> dict[int, float]:
    """Compute redundancy score for each sentence.

    Returns mapping of sentence_index -> redundancy [0, 1] where
    1 means the sentence is completely redundant.
    """
    if len(sentences) < 2:
        return {0: 0.0} if sentences else {}

    word_sets = [set(_tokenize(s)) for s in sentences]
    redundancy: dict[int, float] = {}

    for i, words_i in enumerate(word_sets):
        if not words_i:
            redundancy[i] = 1.0
            continue

        max_overlap = 0.0
        for j, words_j in enumerate(word_sets):
            if i == j:
                continue
            overlap = len(words_i & words_j) / len(words_i)
            max_overlap = max(max_overlap, overlap)

        redundancy[i] = max_overlap

    return redundancy


def _tokenize(text: str) -> list[str]:
    """Simple word tokenization."""
    return [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 1]
