"""Core data types used throughout the TwoTrim system."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CompressionMode(str, Enum):
    """Compression aggressiveness level."""
    LOSSLESS = "lossless"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class StrategyName(str, Enum):
    """Available compression strategies."""
    RULE_BASED = "rule_based"
    SEMANTIC = "semantic"
    EMBEDDING = "embedding"
    RAG_AWARE = "rag_aware"
    STRUCTURED = "structured"
    CANONICALIZE = "canonicalize"


class RequestType(str, Enum):
    """Type of LLM request."""
    CHAT_COMPLETION = "chat_completion"
    COMPLETION = "completion"
    EMBEDDING = "embedding"


class CacheBackend(str, Enum):
    """Cache storage backend."""
    SQLITE = "sqlite"
    REDIS = "redis"


# ---------------------------------------------------------------------------
# Message types (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single chat message."""
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


# ---------------------------------------------------------------------------
# Compression types
# ---------------------------------------------------------------------------

class StrategyResult(BaseModel):
    """Result from a single compression strategy."""
    strategy: StrategyName
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompressionResult(BaseModel):
    """Full result of the compression pipeline."""
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    overall_ratio: float
    strategies_applied: list[StrategyResult] = Field(default_factory=list)
    quality_score: float | None = None
    compression_time_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompressedMessage(BaseModel):
    """A chat message after compression."""
    original: ChatMessage
    compressed: ChatMessage
    result: CompressionResult | None = None


# ---------------------------------------------------------------------------
# Scoring types
# ---------------------------------------------------------------------------

class SegmentScore(BaseModel):
    """Importance score for a text segment."""
    text: str
    start: int
    end: int
    score: float
    components: dict[str, float] = Field(default_factory=dict)


class ScoringResult(BaseModel):
    """Result of token/segment importance scoring."""
    segments: list[SegmentScore]
    mean_score: float
    min_score: float
    max_score: float


# ---------------------------------------------------------------------------
# Cache types
# ---------------------------------------------------------------------------

class CacheEntry(BaseModel):
    """An entry in the semantic cache."""
    key: str
    query_embedding: list[float] | None = None
    query_text: str
    response: dict[str, Any]
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    hit_count: int = 0
    ttl_seconds: int = 3600


class CacheHit(BaseModel):
    """A cache lookup result."""
    hit: bool
    entry: CacheEntry | None = None
    similarity: float = 0.0
    source: str = ""  # "semantic" | "prompt" | "exact"


# ---------------------------------------------------------------------------
# Policy types
# ---------------------------------------------------------------------------

class PolicyDecision(BaseModel):
    """What the policy engine decided for a request."""
    mode: CompressionMode
    strategies: list[StrategyName]
    target_reduction: float
    max_degradation: float
    skip_cache: bool = False
    query: str | None = None  # To support question-aware summarization
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation types
# ---------------------------------------------------------------------------

class EvalResult(BaseModel):
    """Result of quality evaluation."""
    request_id: str
    similarity_score: float
    passed: bool
    threshold: float
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy_scores: dict[str, float] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Metrics types
# ---------------------------------------------------------------------------

class RequestMetrics(BaseModel):
    """Metrics for a single request."""
    request_id: str
    request_type: RequestType
    model: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    tokens_saved: int
    estimated_cost_saved_usd: float
    compression_time_ms: float
    total_time_ms: float
    cache_hit: bool
    strategies_applied: list[str]
    quality_score: float | None = None
    timestamp: float = Field(default_factory=time.time)


class AggregateMetrics(BaseModel):
    """Aggregate metrics over a time window."""
    total_requests: int = 0
    total_tokens_original: int = 0
    total_tokens_compressed: int = 0
    total_tokens_saved: int = 0
    total_cost_saved_usd: float = 0.0
    avg_compression_ratio: float = 0.0
    avg_quality_score: float = 0.0
    cache_hit_rate: float = 0.0
    avg_compression_time_ms: float = 0.0
    strategy_usage: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Proxy / interceptor types
# ---------------------------------------------------------------------------

class UpstreamTarget(BaseModel):
    """An upstream LLM endpoint."""
    base_url: str
    api_key: str | None = None
    timeout_seconds: int = 120
    extra_headers: dict[str, str] = Field(default_factory=dict)


class ProxyRequest(BaseModel):
    """Internal representation of an intercepted request."""
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    request_type: RequestType
    model: str
    messages: list[ChatMessage] | None = None
    prompt: str | None = None
    upstream: UpstreamTarget | None = None
    compression_mode: CompressionMode | None = None
    raw_body: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Plugin types
# ---------------------------------------------------------------------------

class PluginInfo(BaseModel):
    """Metadata about a loaded plugin."""
    name: str
    version: str
    description: str
    plugin_type: str  # "compression" | "evaluation" | "cache"
    enabled: bool = True


# ---------------------------------------------------------------------------
# Model pricing (for cost estimation)
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    # price per 1K tokens (input, output)
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
    # Open models — typically self-hosted, cost ~= compute only
    "llama-*": {"input": 0.0002, "output": 0.0002},
    "mistral-*": {"input": 0.0002, "output": 0.0002},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Estimate cost in USD for a given model and token count."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        # Try wildcard match
        for pattern, p in MODEL_PRICING.items():
            if pattern.endswith("*") and model.startswith(pattern[:-1]):
                pricing = p
                break
    if not pricing:
        pricing = {"input": 0.001, "output": 0.002}  # fallback

    return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
