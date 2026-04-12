"""KV cache eviction strategies.

Implements LRU, attention-score-based, and hybrid eviction
for managing bounded KV cache memory.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CacheSlot:
    """A single slot in the KV cache."""
    token_id: int
    position: int
    attention_score: float = 0.0
    access_count: int = 0
    key_tensor: Any = None
    value_tensor: Any = None


class EvictionStrategy(ABC):
    """Base class for KV cache eviction strategies."""

    @abstractmethod
    def select_for_eviction(
        self, cache: dict[int, CacheSlot], n_evict: int
    ) -> list[int]:
        """Select positions to evict from the cache."""
        ...


class LRUEviction(EvictionStrategy):
    """Least Recently Used eviction."""

    def __init__(self) -> None:
        self._access_order: OrderedDict[int, float] = OrderedDict()

    def record_access(self, position: int, timestamp: float) -> None:
        self._access_order[position] = timestamp
        self._access_order.move_to_end(position)

    def select_for_eviction(
        self, cache: dict[int, CacheSlot], n_evict: int
    ) -> list[int]:
        # Evict least recently used
        positions = list(self._access_order.keys())
        evict = positions[:n_evict]
        for pos in evict:
            self._access_order.pop(pos, None)
        return evict


class AttentionScoreEviction(EvictionStrategy):
    """Evict tokens with lowest cumulative attention scores."""

    def select_for_eviction(
        self, cache: dict[int, CacheSlot], n_evict: int
    ) -> list[int]:
        scored = sorted(cache.items(), key=lambda x: x[1].attention_score)
        return [pos for pos, _ in scored[:n_evict]]


class HybridEviction(EvictionStrategy):
    """Combine attention scores with recency for eviction decisions."""

    def __init__(self, attention_weight: float = 0.6, recency_weight: float = 0.4) -> None:
        self.attention_weight = attention_weight
        self.recency_weight = recency_weight

    def select_for_eviction(
        self, cache: dict[int, CacheSlot], n_evict: int
    ) -> list[int]:
        if not cache:
            return []

        max_pos = max(slot.position for slot in cache.values()) or 1
        max_attn = max(slot.attention_score for slot in cache.values()) or 1.0

        scored: list[tuple[int, float]] = []
        for pos, slot in cache.items():
            recency = slot.position / max_pos  # newer = higher
            attn = slot.attention_score / max_attn

            # Lower combined score = more likely to evict
            combined = attn * self.attention_weight + recency * self.recency_weight
            scored.append((pos, combined))

        scored.sort(key=lambda x: x[1])
        return [pos for pos, _ in scored[:n_evict]]


class SlidingWindowEviction(EvictionStrategy):
    """Keep only tokens within a sliding window plus important anchors."""

    def __init__(self, window_size: int = 2048, anchor_count: int = 64) -> None:
        self.window_size = window_size
        self.anchor_count = anchor_count

    def select_for_eviction(
        self, cache: dict[int, CacheSlot], n_evict: int
    ) -> list[int]:
        if len(cache) <= self.window_size:
            return []

        positions = sorted(cache.keys())
        window_start = max(0, len(positions) - self.window_size)

        # Always keep the first anchor_count tokens (system prompt, instructions)
        anchors = set(positions[:self.anchor_count])
        window = set(positions[window_start:])

        # Everything not in window or anchors is evictable
        evictable = [p for p in positions if p not in anchors and p not in window]
        return evictable[:n_evict]


def create_eviction_strategy(name: str, **kwargs: Any) -> EvictionStrategy:
    """Factory for eviction strategies."""
    strategies = {
        "lru": LRUEviction,
        "attention_score": AttentionScoreEviction,
        "hybrid": HybridEviction,
        "sliding_window": SlidingWindowEviction,
    }

    cls = strategies.get(name)
    if cls is None:
        logger.warning("Unknown eviction strategy '%s', defaulting to LRU", name)
        return LRUEviction()

    return cls(**kwargs)
