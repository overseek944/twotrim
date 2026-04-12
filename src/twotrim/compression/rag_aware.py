"""RAG-aware compression — optimize retrieved context chunks.

Detects RAG patterns in prompts (multiple retrieved documents/chunks),
scores relevance, drops low-value chunks, and summarizes verbose ones.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from twotrim.types import StrategyName, StrategyResult

logger = logging.getLogger(__name__)

# Patterns that indicate retrieved context in prompts
RAG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:Context|Document|Source|Reference|Passage|Chunk)\s*(?:\d+|#\d+)\s*[:.]", re.I),
    re.compile(r"---+\s*(?:Document|Source|Context)", re.I),
    re.compile(r"\[(?:Document|Source|Context)\s*\d+\]", re.I),
    re.compile(r"<(?:context|document|source)(?:\s+\w+=[\"'][^\"']*[\"'])*>", re.I),
    re.compile(r"Retrieved (?:documents?|passages?|contexts?):", re.I),
    re.compile(r"The following (?:documents?|passages?|information) (?:were|was) retrieved:", re.I),
]

# Delimiter patterns between chunks
CHUNK_DELIMITERS: list[re.Pattern[str]] = [
    re.compile(r"\n---+\n"),
    re.compile(r"\n={3,}\n"),
    re.compile(r"\n\[(?:Document|Source|Context)\s*\d+\]\n", re.I),
    re.compile(r"\n(?:Document|Source|Context|Chunk)\s*(?:\d+|#\d+)\s*[:.]", re.I),
    re.compile(r"</(?:context|document|source)>\s*<(?:context|document|source)", re.I),
]


class RagAwareCompressor:
    """Compress RAG-retrieved context by relevance scoring and pruning."""

    def __init__(
        self,
        min_relevance_score: float = 0.3,
        max_chunks: int = 10,
        embedding_model: Any = None,
    ) -> None:
        self.min_relevance_score = min_relevance_score
        self.max_chunks = max_chunks
        self._embedding_model = embedding_model
        self.name = StrategyName.RAG_AWARE

    async def compress(
        self,
        text: str,
        token_counter: object | None = None,
        query: str | None = None,
        target_reduction: float = 0.0,
    ) -> StrategyResult:
        """Detect RAG context, score chunks, prune low-relevance ones."""
        original_tokens = _count_tokens(text, token_counter)

        # Detect if this contains RAG patterns
        if not self._is_rag_prompt(text):
            return StrategyResult(
                strategy=self.name,
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=0.0,
                metadata={"skipped": True, "reason": "not_rag_prompt"},
            )

        # Extract the query and context chunks
        query, chunks, preamble, postamble = self._extract_chunks(text)

        if len(chunks) <= 1:
            return StrategyResult(
                strategy=self.name,
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=0.0,
                metadata={"skipped": True, "reason": "single_chunk"},
            )

        # Score and filter chunks
        scored_chunks = await self._score_chunks(query, chunks)
        filtered = [
            (chunk, score) for chunk, score in scored_chunks
            if score >= self.min_relevance_score
        ]

        # Keep at most max_chunks
        filtered = filtered[: self.max_chunks]

        if not filtered:
            # Keep at least the top chunk
            filtered = [scored_chunks[0]]

        # Reassemble
        compressed_text = self._reassemble(preamble, filtered, postamble)
        compressed_tokens = _count_tokens(compressed_text, token_counter)

        return StrategyResult(
            strategy=self.name,
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1 - (compressed_tokens / max(original_tokens, 1)),
            metadata={
                "original_chunks": len(chunks),
                "retained_chunks": len(filtered),
                "dropped_chunks": len(chunks) - len(filtered),
                "scores": [round(s, 3) for _, s in scored_chunks],
            },
        )

    async def estimate_reduction(self, text: str) -> float:
        if not self._is_rag_prompt(text):
            return 0.0
        return 0.30  # typical RAG reduction

    def _is_rag_prompt(self, text: str) -> bool:
        """Check if text contains RAG-style retrieved context."""
        return any(p.search(text) for p in RAG_PATTERNS)

    def _extract_chunks(
        self, text: str
    ) -> tuple[str, list[str], str, str]:
        """Extract query, context chunks, preamble, and postamble."""
        # Try to find the query/question portion
        query = ""
        query_patterns = [
            re.compile(r"(?:Question|Query|User|Prompt):\s*(.*?)(?:\n\n|$)", re.I | re.S),
            re.compile(r"(?:Based on|Using|Given) .*?(?:answer|respond to|address):\s*(.*?)$", re.I | re.S),
        ]
        for p in query_patterns:
            m = p.search(text)
            if m:
                query = m.group(1).strip()
                break

        # If no query found, use last paragraph as query
        if not query:
            paragraphs = text.split("\n\n")
            if len(paragraphs) > 1:
                query = paragraphs[-1].strip()

        # Split into chunks using delimiters
        chunks: list[str] = []
        preamble = ""
        postamble = ""

        # Find the best delimiter
        for delimiter in CHUNK_DELIMITERS:
            parts = delimiter.split(text)
            if len(parts) > 1:
                preamble = parts[0].strip()
                chunks = [p.strip() for p in parts[1:] if p.strip()]
                # Check if last chunk looks like a question/instruction
                if chunks and len(chunks[-1].split()) < 50 and "?" in chunks[-1]:
                    postamble = chunks.pop()
                    query = postamble if not query else query
                break

        if not chunks:
            # Fall back to paragraph splitting
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if len(paragraphs) > 2:
                preamble = paragraphs[0]
                chunks = paragraphs[1:-1]
                postamble = paragraphs[-1]
                query = postamble if not query else query
            else:
                chunks = paragraphs

        return query, chunks, preamble, postamble

    async def _score_chunks(
        self, query: str, chunks: list[str]
    ) -> list[tuple[str, float]]:
        """Score chunk relevance — use embeddings if available, else heuristic."""
        if self._embedding_model is not None:
            return await self._embedding_score(query, chunks)
        return self._heuristic_score(query, chunks)

    async def _embedding_score(
        self, query: str, chunks: list[str]
    ) -> list[tuple[str, float]]:
        """Score chunks by embedding similarity to query."""
        import asyncio
        import numpy as np

        def _compute() -> list[tuple[str, float]]:
            all_texts = [query] + chunks
            embeddings = self._embedding_model.encode(all_texts, show_progress_bar=False)
            query_emb = embeddings[0]
            chunk_embs = embeddings[1:]

            # Cosine similarity
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            scores: list[tuple[str, float]] = []
            for i, chunk in enumerate(chunks):
                chunk_norm = chunk_embs[i] / (np.linalg.norm(chunk_embs[i]) + 1e-8)
                sim = float(np.dot(query_norm, chunk_norm))
                scores.append((chunk, sim))

            scores.sort(key=lambda x: x[1], reverse=True)
            return scores

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _compute)

    def _heuristic_score(
        self, query: str, chunks: list[str]
    ) -> list[tuple[str, float]]:
        """Score chunks by keyword overlap with query."""
        query_words = set(re.findall(r"\w+", query.lower()))

        scored: list[tuple[str, float]] = []
        for chunk in chunks:
            chunk_words = set(re.findall(r"\w+", chunk.lower()))
            if not chunk_words:
                scored.append((chunk, 0.0))
                continue

            # Jaccard-like overlap
            overlap = len(query_words & chunk_words)
            score = overlap / max(len(query_words), 1)

            # Length penalty — very short chunks are less useful
            if len(chunk.split()) < 10:
                score *= 0.5

            scored.append((chunk, min(score, 1.0)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _reassemble(
        self,
        preamble: str,
        chunks: list[tuple[str, float]],
        postamble: str,
    ) -> str:
        """Reassemble compressed prompt from scored chunks."""
        parts: list[str] = []
        if preamble:
            parts.append(preamble)

        for i, (chunk, score) in enumerate(chunks):
            parts.append(f"[Source {i + 1}] (relevance: {score:.2f})\n{chunk}")

        if postamble:
            parts.append(postamble)

        return "\n\n".join(parts)


def _count_tokens(text: str, counter: object | None = None) -> int:
    if counter is not None and hasattr(counter, "count"):
        return counter.count(text)  # type: ignore[union-attr]
    return max(1, int(len(text.split()) / 0.75))
