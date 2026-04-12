"""Plugin base classes — interfaces for custom extensions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from twotrim.types import StrategyName, StrategyResult


class CompressionPlugin(ABC):
    """Base class for custom compression strategy plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this plugin."""
        ...

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    async def compress(self, text: str, token_counter: Any = None) -> StrategyResult:
        """Compress the given text."""
        ...

    async def estimate_reduction(self, text: str) -> float:
        """Estimate compression ratio without compressing."""
        return 0.0

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the plugin with custom settings."""
        pass


class EvaluationPlugin(ABC):
    """Base class for custom evaluation strategy plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def evaluate(
        self, original: str, compressed: str, response: dict[str, Any] | None = None
    ) -> float:
        """Return a quality score [0, 1]."""
        ...


class CachePlugin(ABC):
    """Base class for custom cache backend plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def lookup(self, key: str) -> Any | None:
        ...

    @abstractmethod
    async def store(self, key: str, value: Any, ttl: int = 3600) -> None:
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        ...
