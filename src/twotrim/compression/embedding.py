"""Embedding-based reduction — cluster and deduplicate via semantic similarity.

Uses sentence-transformers and FAISS to identify and remove redundant
text segments based on embedding similarity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from twotrim.types import StrategyName, StrategyResult

logger = logging.getLogger(__name__)


class EmbeddingCompressor:
    """Reduce redundancy by clustering semantically similar text segments."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        clustering_method: str = "agglomerative",
        min_cluster_size: int = 2,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size
        self.device = device
        self.name = StrategyName.EMBEDDING
        self._model: Any = None
        self._available: bool | None = None

    async def _ensure_model(self) -> bool:
        """Lazy-load the embedding model."""
        if self._available is not None:
            return self._available
        try:
            from sentence_transformers import SentenceTransformer

            device = self.device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = SentenceTransformer(self.model_name, device=device)
            self._available = True
            logger.info("Loaded embedding model: %s on %s", self.model_name, device)
        except Exception as e:
            logger.warning("Embedding model unavailable: %s", e)
            self._available = False
        return self._available

    async def compress(
        self,
        text: str,
        token_counter: object | None = None,
        query: str | None = None,
        target_reduction: float = 0.0,
    ) -> StrategyResult:
        """Cluster similar segments and keep representatives."""
        original_tokens = _count_tokens(text, token_counter)

        segments = self._segment_text(text)

        # Need at least 3 segments to do meaningful clustering
        if len(segments) < 3:
            return StrategyResult(
                strategy=self.name,
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=0.0,
                metadata={"skipped": True, "reason": "too_few_segments"},
            )

        available = await self._ensure_model()

        if available and self._model is not None:
            compressed = await self._embedding_reduce(segments)
        else:
            compressed = self._hash_reduce(segments)

        compressed_text = "\n\n".join(compressed)
        compressed_tokens = _count_tokens(compressed_text, token_counter)

        return StrategyResult(
            strategy=self.name,
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=1 - (compressed_tokens / max(original_tokens, 1)),
            metadata={
                "original_segments": len(segments),
                "retained_segments": len(compressed),
                "method": "embedding" if available else "hash",
            },
        )

    async def estimate_reduction(self, text: str) -> float:
        """Estimate potential reduction from deduplication."""
        segments = self._segment_text(text)
        if len(segments) < 3:
            return 0.0

        # Use hash-based estimate
        from hashlib import sha256
        hashes = set()
        duplicates = 0
        for seg in segments:
            normalized = " ".join(seg.lower().split())
            h = sha256(normalized.encode()).hexdigest()[:16]
            if h in hashes:
                duplicates += 1
            hashes.add(h)

        return duplicates / len(segments)

    async def _embedding_reduce(self, segments: list[str]) -> list[str]:
        """Use embedding clustering to remove redundant segments."""
        import asyncio

        def _compute() -> list[str]:
            # Encode all segments
            embeddings = self._model.encode(segments, show_progress_bar=False, convert_to_numpy=True)
            embeddings = np.array(embeddings)

            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            embeddings_normalized = embeddings / norms

            if self.clustering_method == "agglomerative":
                return self._agglomerative_reduce(segments, embeddings_normalized)
            else:
                return self._simple_threshold_reduce(segments, embeddings_normalized)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _compute)

    def _agglomerative_reduce(
        self, segments: list[str], embeddings: np.ndarray
    ) -> list[str]:
        """Use agglomerative clustering to group similar segments."""
        from sklearn.cluster import AgglomerativeClustering

        # Distance threshold = 1 - similarity_threshold (cosine distance)
        distance_threshold = 1 - self.similarity_threshold

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        # For each cluster, keep the segment closest to centroid
        representatives: list[str] = []
        cluster_order: list[tuple[int, str]] = []

        for label in sorted(set(labels)):
            indices = [i for i, l in enumerate(labels) if l == label]

            if len(indices) == 1:
                cluster_order.append((indices[0], segments[indices[0]]))
                continue

            # Compute centroid
            cluster_embeddings = embeddings[indices]
            centroid = cluster_embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Find closest to centroid
            similarities = cluster_embeddings @ centroid
            best_local = int(np.argmax(similarities))
            best_global = indices[best_local]
            cluster_order.append((best_global, segments[best_global]))

        # Preserve original order
        cluster_order.sort(key=lambda x: x[0])
        return [s for _, s in cluster_order]

    def _simple_threshold_reduce(
        self, segments: list[str], embeddings: np.ndarray
    ) -> list[str]:
        """Simple greedy deduplication: skip segments too similar to a kept one."""
        kept: list[int] = [0]

        for i in range(1, len(segments)):
            similarities = embeddings[i] @ embeddings[kept].T
            if similarities.max() < self.similarity_threshold:
                kept.append(i)

        return [segments[i] for i in kept]

    def _hash_reduce(self, segments: list[str]) -> list[str]:
        """Fallback: hash-based deduplication when embeddings unavailable."""
        from hashlib import sha256

        seen: set[str] = set()
        result: list[str] = []

        for seg in segments:
            # Normalize: lowercase, collapse whitespace, strip punctuation
            import re
            normalized = re.sub(r"[^\w\s]", "", seg.lower())
            normalized = " ".join(normalized.split())
            h = sha256(normalized.encode()).hexdigest()[:16]
            if h not in seen:
                seen.add(h)
                result.append(seg)

        return result

    def _segment_text(self, text: str) -> list[str]:
        """Split text into meaningful segments for comparison."""
        import re

        # Split on double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)
        segments: list[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is very long, split further into sentences
            if len(para.split()) > 100:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                # Group sentences into chunks of ~50 words
                chunk: list[str] = []
                chunk_len = 0
                for sentence in sentences:
                    s_len = len(sentence.split())
                    if chunk_len + s_len > 50 and chunk:
                        segments.append(" ".join(chunk))
                        chunk = [sentence]
                        chunk_len = s_len
                    else:
                        chunk.append(sentence)
                        chunk_len += s_len
                if chunk:
                    segments.append(" ".join(chunk))
            else:
                segments.append(para)

        return segments


def _count_tokens(text: str, counter: object | None = None) -> int:
    if counter is not None and hasattr(counter, "count"):
        return counter.count(text)  # type: ignore[union-attr]
    return max(1, int(len(text.split()) / 0.75))
