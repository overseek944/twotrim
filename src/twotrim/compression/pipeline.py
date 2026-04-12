"""Compression pipeline orchestrator.

Runs multiple compression strategies in priority order,
tracks cumulative compression, and short-circuits when targets are met.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, runtime_checkable

from twotrim.config import get_config
from twotrim.types import (
    CompressionMode,
    CompressionResult,
    PolicyDecision,
    StrategyName,
    StrategyResult,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class CompressionStrategy(Protocol):
    """Protocol for compression strategies."""
    name: StrategyName
    async def compress(
        self,
        text: str,
        token_counter: object | None = None,
        query: str | None = None,
        target_reduction: float = 0.0,
    ) -> StrategyResult: ...
    async def estimate_reduction(self, text: str) -> float: ...


class TokenCounter:
    """Token counting using tiktoken with fallback."""

    def __init__(self, model: str = "gpt-4") -> None:
        self._encoder: Any = None
        self._model = model
        self._init()

    def _init(self) -> None:
        try:
            import tiktoken
            try:
                self._encoder = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.debug("tiktoken not available, using word-based estimate")

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self._encoder:
            return len(self._encoder.encode(text))
        return max(1, int(len(text.split()) / 0.75))


class CompressionPipeline:
    """Orchestrates multiple compression strategies in sequence."""

    def __init__(self) -> None:
        self._strategies: dict[StrategyName, CompressionStrategy] = {}
        self._token_counter = TokenCounter()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-initialize all strategies from config."""
        if self._initialized:
            return

        cfg = get_config().compression.strategies

        from twotrim.compression.rule_based import RuleBasedCompressor
        from twotrim.compression.semantic import SemanticCompressor
        from twotrim.compression.embedding import EmbeddingCompressor
        from twotrim.compression.rag_aware import RagAwareCompressor
        from twotrim.compression.structured import StructuredCompressor
        from twotrim.compression.canonicalize import PromptCanonicalizer

        models_cfg = get_config().models

        if cfg.rule_based.enabled:
            self._strategies[StrategyName.RULE_BASED] = RuleBasedCompressor(
                remove_filler_words=cfg.rule_based.remove_filler_words,
                collapse_whitespace=cfg.rule_based.collapse_whitespace,
                deduplicate_sentences=cfg.rule_based.deduplicate_sentences,
                normalize_unicode=cfg.rule_based.normalize_unicode,
            )

        if cfg.embedding.enabled:
            self._strategies[StrategyName.EMBEDDING] = EmbeddingCompressor(
                model_name=cfg.embedding.model or models_cfg.embedding,
                similarity_threshold=cfg.embedding.similarity_threshold,
                clustering_method=cfg.embedding.clustering_method,
                min_cluster_size=cfg.embedding.min_cluster_size,
                device=models_cfg.device,
            )

        if cfg.semantic.enabled:
            self._strategies[StrategyName.SEMANTIC] = SemanticCompressor(
                model_name=cfg.semantic.model or models_cfg.summarization,
                max_summary_ratio=cfg.semantic.max_summary_ratio,
                min_summary_ratio=cfg.semantic.min_summary_ratio,
                min_input_length=cfg.semantic.min_input_length,
                batch_size=cfg.semantic.batch_size,
                prefer_extractive=cfg.semantic.prefer_extractive,
                device=models_cfg.device,
            )

        if cfg.rag_aware.enabled:
            self._strategies[StrategyName.RAG_AWARE] = RagAwareCompressor(
                min_relevance_score=cfg.rag_aware.min_relevance_score,
                max_chunks=cfg.rag_aware.max_chunks,
            )

        if cfg.structured.enabled:
            self._strategies[StrategyName.STRUCTURED] = StructuredCompressor(
                prefer_json=cfg.structured.prefer_json,
                preserve_code_blocks=cfg.structured.preserve_code_blocks,
            )

        if cfg.canonicalize.enabled:
            self._strategies[StrategyName.CANONICALIZE] = PromptCanonicalizer(
                template_dir=cfg.canonicalize.template_dir,
            )

        self._initialized = True

    async def compress(
        self,
        text: str,
        decision: PolicyDecision,
    ) -> CompressionResult:
        """Run the compression pipeline based on policy decision."""
        self._ensure_initialized()
        start = time.perf_counter()

        original_tokens = self._token_counter.count(text)
        current_text = text
        strategy_results: list[StrategyResult] = []
        target_tokens = int(original_tokens * (1 - decision.target_reduction))

        # Sort requested strategies by their configured priority
        ordered = self._order_strategies(decision.strategies)
        
        trigger_threshold = get_config().compression.semantic_trigger_threshold

        for strategy_name in ordered:
            # Skip costly semantic strategy if prompt is short
            if strategy_name == StrategyName.SEMANTIC and original_tokens < trigger_threshold:
                logger.debug("Skipping semantic strategy: tokens (%d) < threshold (%d)",
                             original_tokens, trigger_threshold)
                continue
            strategy = self._strategies.get(strategy_name)
            if strategy is None:
                continue

            current_tokens = self._token_counter.count(current_text)
            if current_tokens <= target_tokens:
                logger.debug("Target reached after %d strategies, skipping %s",
                             len(strategy_results), strategy_name)
                break

            try:
                result = await strategy.compress(
                    current_text,
                    self._token_counter,
                    query=decision.query,
                    target_reduction=decision.target_reduction
                )
                strategy_results.append(result)

                if result.compression_ratio > 0.01:
                    current_text = result.compressed_text
                    logger.debug(
                        "Strategy %s: %d -> %d tokens (%.1f%% reduction)",
                        strategy_name.value,
                        result.original_tokens,
                        result.compressed_tokens,
                        result.compression_ratio * 100,
                    )
            except Exception:
                logger.exception("Strategy %s failed", strategy_name.value)

        final_tokens = self._token_counter.count(current_text)
        elapsed = (time.perf_counter() - start) * 1000

        return CompressionResult(
            original_text=text,
            compressed_text=current_text,
            original_tokens=original_tokens,
            compressed_tokens=final_tokens,
            overall_ratio=1 - (final_tokens / max(original_tokens, 1)),
            strategies_applied=strategy_results,
            compression_time_ms=elapsed,
        )

    def _order_strategies(self, names: list[StrategyName]) -> list[StrategyName]:
        """Order strategies by configured priority."""
        cfg = get_config().compression.strategies
        priorities = {
            StrategyName.RULE_BASED: cfg.rule_based.priority,
            StrategyName.SEMANTIC: cfg.semantic.priority,
            StrategyName.EMBEDDING: cfg.embedding.priority,
            StrategyName.RAG_AWARE: cfg.rag_aware.priority,
            StrategyName.STRUCTURED: cfg.structured.priority,
            StrategyName.CANONICALIZE: cfg.canonicalize.priority,
        }
        return sorted(names, key=lambda n: priorities.get(n, 99))

    def register_strategy(self, strategy: CompressionStrategy) -> None:
        """Register a custom compression strategy (plugin support)."""
        self._strategies[strategy.name] = strategy

    def get_token_counter(self) -> TokenCounter:
        return self._token_counter


# Module-level singleton
_pipeline: CompressionPipeline | None = None


def get_pipeline() -> CompressionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CompressionPipeline()
    return _pipeline
