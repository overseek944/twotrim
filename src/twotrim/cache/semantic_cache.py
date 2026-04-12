"""Semantic cache — FAISS-backed response cache using embedding similarity.

Caches LLM responses indexed by query embeddings. Returns cached
responses for semantically similar queries without re-calling the LLM.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from twotrim.types import CacheEntry, CacheHit

logger = logging.getLogger(__name__)


class SemanticCache:
    """FAISS-backed semantic response cache."""

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 10000,
        ttl_seconds: int = 3600,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir: str | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._embedding_model_name = embedding_model
        self._persist_dir = persist_dir
        self._model: Any = None
        self._index: Any = None
        self._entries: list[CacheEntry] = []
        self._lock = threading.Lock()
        self._dimension: int = 0
        self._initialized = False

    def _ensure_init(self) -> bool:
        """Lazy-initialize FAISS index and embedding model."""
        if self._initialized:
            return self._model is not None

        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            self._model = SentenceTransformer(self._embedding_model_name)
            # Get embedding dimension
            test_emb = self._model.encode(["test"], convert_to_numpy=True)
            self._dimension = test_emb.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)  # Inner product (cosine on normalized)
            self._initialized = True

            if self._persist_dir:
                self._load_from_disk()

            logger.info("Semantic cache initialized (dim=%d, threshold=%.2f)",
                       self._dimension, self.similarity_threshold)
            return True
        except Exception as e:
            logger.warning("Semantic cache unavailable: %s", e)
            self._initialized = True
            return False

    async def lookup(self, query: str) -> CacheHit:
        """Look up a query in the semantic cache."""
        if not self._ensure_init() or self._index is None:
            return CacheHit(hit=False, source="semantic")

        embedding = self._encode(query)
        if embedding is None:
            return CacheHit(hit=False, source="semantic")

        with self._lock:
            if self._index.ntotal == 0:
                return CacheHit(hit=False, source="semantic")

            # Search
            scores, indices = self._index.search(
                embedding.reshape(1, -1).astype(np.float32), min(5, self._index.ntotal)
            )

            now = time.time()
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._entries):
                    continue

                entry = self._entries[idx]

                # Check TTL
                if now - entry.created_at > entry.ttl_seconds:
                    continue

                if float(score) >= self.similarity_threshold:
                    entry.hit_count += 1
                    entry.last_accessed = now
                    return CacheHit(
                        hit=True,
                        entry=entry,
                        similarity=float(score),
                        source="semantic",
                    )

        return CacheHit(hit=False, source="semantic")

    async def store(self, query: str, response: dict[str, Any]) -> None:
        """Store a query-response pair in the cache."""
        if not self._ensure_init() or self._index is None:
            return

        embedding = self._encode(query)
        if embedding is None:
            return

        entry = CacheEntry(
            key=query[:200],
            query_text=query,
            response=response,
            ttl_seconds=self.ttl_seconds,
        )

        with self._lock:
            # Evict if at capacity
            if len(self._entries) >= self.max_entries:
                self._evict()

            self._index.add(embedding.reshape(1, -1).astype(np.float32))
            self._entries.append(entry)

    async def invalidate(self, query: str) -> None:
        """Invalidate a cached entry."""
        # FAISS doesn't support deletion easily, so we mark as expired
        if not self._entries:
            return
        embedding = self._encode(query)
        if embedding is None:
            return

        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return
            scores, indices = self._index.search(
                embedding.reshape(1, -1).astype(np.float32), 1
            )
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self._entries) and float(score) >= 0.98:
                    self._entries[idx].created_at = 0  # Force expire

    async def clear(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            if self._index is not None:
                import faiss
                self._index = faiss.IndexFlatIP(self._dimension)
            self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    def _encode(self, text: str) -> np.ndarray | None:
        """Encode text to normalized embedding."""
        if self._model is None:
            return None
        try:
            emb = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)
            # L2 normalize for cosine similarity via inner product
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            norm = np.where(norm == 0, 1, norm)
            return (emb / norm)[0]
        except Exception as e:
            logger.warning("Encoding failed: %s", e)
            return None

    def _evict(self) -> None:
        """Evict least recently used entries to make space."""
        import faiss

        # Sort by last_accessed, remove oldest 20%
        n_remove = max(1, len(self._entries) // 5)
        indexed = sorted(enumerate(self._entries), key=lambda x: x[1].last_accessed)
        remove_indices = set(i for i, _ in indexed[:n_remove])

        # Rebuild index without evicted entries
        kept_entries: list[CacheEntry] = []
        kept_embeddings: list[np.ndarray] = []

        for i, entry in enumerate(self._entries):
            if i not in remove_indices:
                kept_entries.append(entry)
                emb = self._encode(entry.query_text)
                if emb is not None:
                    kept_embeddings.append(emb)

        self._entries = kept_entries
        self._index = faiss.IndexFlatIP(self._dimension)
        if kept_embeddings:
            self._index.add(np.stack(kept_embeddings).astype(np.float32))

        logger.debug("Evicted %d entries, %d remaining", n_remove, len(self._entries))

    def _load_from_disk(self) -> None:
        """Load persisted cache from disk."""
        if not self._persist_dir:
            return
        cache_file = Path(self._persist_dir) / "semantic_cache.json"
        if not cache_file.exists():
            return
        try:
            data = json.loads(cache_file.read_text())
            for item in data:
                entry = CacheEntry(**item)
                if time.time() - entry.created_at < entry.ttl_seconds:
                    emb = self._encode(entry.query_text)
                    if emb is not None and self._index is not None:
                        self._index.add(emb.reshape(1, -1).astype(np.float32))
                        self._entries.append(entry)
            logger.info("Loaded %d entries from cache", len(self._entries))
        except Exception as e:
            logger.warning("Failed to load cache: %s", e)

    async def persist(self) -> None:
        """Persist cache to disk."""
        if not self._persist_dir:
            return
        p = Path(self._persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        cache_file = p / "semantic_cache.json"
        data = [e.model_dump() for e in self._entries]
        cache_file.write_text(json.dumps(data))
