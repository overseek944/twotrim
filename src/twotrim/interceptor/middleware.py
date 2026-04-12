"""Request/response middleware for the interceptor proxy."""

from __future__ import annotations

import logging
import time
from typing import Any

from twotrim.cache.semantic_cache import SemanticCache
from twotrim.cache.prompt_cache import PromptCache
from twotrim.compression.pipeline import CompressionPipeline, get_pipeline
from twotrim.config import get_config
from twotrim.evaluation.evaluator import Evaluator, get_evaluator
from twotrim.integrations.openai_compat import (
    build_compressed_body,
    extract_text_from_messages,
    parse_openai_request,
    rebuild_messages_from_compressed,
)
from twotrim.observability.metrics import MetricsCollector, get_metrics_collector
from twotrim.policy.engine import PolicyEngine, get_policy_engine
from twotrim.response.compressor import ResponseCompressor
from twotrim.types import ChatMessage, CompressionMode, CompressionResult, RequestType

logger = logging.getLogger(__name__)


class CompressionMiddleware:
    """Core middleware that orchestrates the compression fabric."""

    def __init__(self) -> None:
        self._pipeline: CompressionPipeline | None = None
        self._policy: PolicyEngine | None = None
        self._evaluator: Evaluator | None = None
        self._metrics: MetricsCollector | None = None
        self._semantic_cache: SemanticCache | None = None
        self._prompt_cache: PromptCache | None = None
        self._response_compressor: ResponseCompressor | None = None
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return

        cfg = get_config()

        self._pipeline = get_pipeline()
        self._policy = get_policy_engine()
        self._evaluator = get_evaluator()
        self._metrics = get_metrics_collector()

        if cfg.cache.semantic.enabled:
            self._semantic_cache = SemanticCache(
                similarity_threshold=cfg.cache.semantic.similarity_threshold,
                max_entries=cfg.cache.semantic.max_entries,
                ttl_seconds=cfg.cache.semantic.ttl_seconds,
                embedding_model=cfg.cache.semantic.embedding_model,
            )

        if cfg.cache.prompt.enabled:
            self._prompt_cache = PromptCache(
                backend=cfg.cache.prompt.backend,
                db_path=cfg.cache.prompt.db_path,
                max_entries=cfg.cache.prompt.max_entries,
            )

        if cfg.response.compress:
            self._response_compressor = ResponseCompressor(
                remove_verbosity=cfg.response.remove_verbosity,
                enforce_structured=cfg.response.enforce_structured,
                max_output_tokens=cfg.response.max_output_tokens,
            )

        self._initialized = True

    async def process_request(
        self,
        body: dict[str, Any],
        override_mode: CompressionMode | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Process an incoming request through the compression fabric.

        Returns: (compressed_body, metadata)
        """
        self._ensure_init()
        start = time.perf_counter()

        request_type, messages, prompt, model = parse_openai_request(body)
        meta: dict[str, Any] = {"request_id": "", "cache_hit": False}

        # Extract text for compression
        if messages:
            text = extract_text_from_messages(messages)
        elif prompt:
            text = prompt
        else:
            return body, {"skipped": True, "reason": "no_content"}

        # Check semantic cache
        if self._semantic_cache:
            cache_hit = await self._semantic_cache.lookup(text)
            if cache_hit.hit and cache_hit.entry:
                meta["cache_hit"] = True
                meta["cache_similarity"] = cache_hit.similarity
                meta["cached_response"] = cache_hit.entry.response
                return body, meta

        # Extract query if possible (last user message)
        query = None
        if messages:
            for msg in reversed(messages):
                if msg.role == "user" and msg.content:
                    query = msg.content
                    break

        # Get policy decision
        assert self._policy is not None
        messages_dicts = [m.model_dump(exclude_none=True) for m in messages] if messages else None
        decision = self._policy.decide(
            model=model,
            messages=messages_dicts,
            prompt=prompt,
            override_mode=override_mode,
        )
        decision.query = query

        if not decision.strategies:
            elapsed = (time.perf_counter() - start) * 1000
            meta["skipped"] = True
            meta["reason"] = decision.metadata.get("reason", "no_strategies")
            meta["compression_time_ms"] = elapsed
            return body, meta

        # Check prompt cache
        if self._prompt_cache:
            cached_result = await self._prompt_cache.lookup(text, decision.mode.value)
            if cached_result:
                compressed_body = self._rebuild_body(
                    body, request_type, messages, cached_result.compressed_text
                )
                elapsed = (time.perf_counter() - start) * 1000
                meta["prompt_cache_hit"] = True
                meta["compression_result"] = cached_result.model_dump()
                meta["compression_time_ms"] = elapsed
                self._record_metrics(
                    cached_result, model, request_type, elapsed, elapsed,
                    cache_hit=True,
                )
                return compressed_body, meta

        # Run compression pipeline
        assert self._pipeline is not None
        result = await self._pipeline.compress(text, decision)
        meta["request_id"] = result.request_id

        # Store in prompt cache
        if self._prompt_cache and result.overall_ratio > 0.01:
            await self._prompt_cache.store(text, result, decision.mode.value)

        # Evaluate quality (sampled)
        assert self._evaluator is not None
        if self._evaluator.should_evaluate():
            eval_result = await self._evaluator.evaluate(result)
            if eval_result:
                meta["eval"] = eval_result.model_dump()
                result.quality_score = eval_result.similarity_score

        # Rebuild request body
        compressed_body = self._rebuild_body(
            body, request_type, messages, result.compressed_text
        )

        elapsed = (time.perf_counter() - start) * 1000
        meta["compression_result"] = {
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "ratio": round(result.overall_ratio, 4),
            "strategies": [s.strategy.value for s in result.strategies_applied],
            "compression_time_ms": round(result.compression_time_ms, 2),
        }
        meta["compression_time_ms"] = elapsed

        self._record_metrics(
            result, model, request_type, result.compression_time_ms,
            elapsed, cache_hit=False,
        )

        return compressed_body, meta

    async def process_response(
        self,
        response: dict[str, Any],
        request_text: str | None = None,
    ) -> dict[str, Any]:
        """Process an LLM response (post-compression, caching)."""
        self._ensure_init()

        # Cache the response
        if self._semantic_cache and request_text:
            await self._semantic_cache.store(request_text, response)

        # Compress response
        if self._response_compressor:
            response = self._response_compressor.compress(response)

        return response

    def _rebuild_body(
        self,
        original_body: dict[str, Any],
        request_type: RequestType,
        messages: list[ChatMessage] | None,
        compressed_text: str,
    ) -> dict[str, Any]:
        """Rebuild the request body with compressed content."""
        if messages and request_type == RequestType.CHAT_COMPLETION:
            compressed_messages = rebuild_messages_from_compressed(messages, compressed_text)
            return build_compressed_body(
                original_body, compressed_messages=compressed_messages
            )
        else:
            return build_compressed_body(
                original_body, compressed_prompt=compressed_text
            )

    def _record_metrics(
        self,
        result: CompressionResult,
        model: str,
        request_type: RequestType,
        compression_time_ms: float,
        total_time_ms: float,
        cache_hit: bool,
    ) -> None:
        """Record request metrics."""
        if self._metrics is None:
            return

        metrics = self._metrics.create_request_metrics(
            request_id=result.request_id,
            request_type=request_type.value,
            model=model,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_time_ms=compression_time_ms,
            total_time_ms=total_time_ms,
            cache_hit=cache_hit,
            strategies=[s.strategy.value for s in result.strategies_applied],
            quality_score=result.quality_score,
        )
        self._metrics.record(metrics)


_middleware: CompressionMiddleware | None = None


def get_middleware() -> CompressionMiddleware:
    global _middleware
    if _middleware is None:
        _middleware = CompressionMiddleware()
    return _middleware
