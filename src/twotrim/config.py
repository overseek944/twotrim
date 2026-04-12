"""Configuration management for TwoTrim.

Loads from YAML file, with environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout_seconds: int = 120
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class UpstreamRoutes(BaseModel):
    default_base_url: str = "https://api.openai.com/v1"
    routes: dict[str, str] = Field(default_factory=dict)


class RuleBasedConfig(BaseModel):
    enabled: bool = True
    priority: int = 1
    remove_filler_words: bool = True
    collapse_whitespace: bool = True
    deduplicate_sentences: bool = True
    normalize_unicode: bool = True


class SemanticConfig(BaseModel):
    enabled: bool = True
    priority: int = 3
    model: str = "facebook/bart-large-cnn"
    max_summary_ratio: float = 0.5
    min_summary_ratio: float = 0.3
    min_input_length: int = 200
    batch_size: int = 4
    prefer_extractive: bool = False  # Faster, non-abstractive compression


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    priority: int = 2
    model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.85
    clustering_method: str = "agglomerative"
    min_cluster_size: int = 2


class RagAwareConfig(BaseModel):
    enabled: bool = True
    priority: int = 4
    min_relevance_score: float = 0.3
    max_chunks: int = 10


class StructuredConfig(BaseModel):
    enabled: bool = True
    priority: int = 5
    prefer_json: bool = True
    preserve_code_blocks: bool = True


class CanonicalizeConfig(BaseModel):
    enabled: bool = True
    priority: int = 6
    template_dir: str | None = None


class StrategiesConfig(BaseModel):
    rule_based: RuleBasedConfig = Field(default_factory=RuleBasedConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rag_aware: RagAwareConfig = Field(default_factory=RagAwareConfig)
    structured: StructuredConfig = Field(default_factory=StructuredConfig)
    canonicalize: CanonicalizeConfig = Field(default_factory=CanonicalizeConfig)


class CompressionConfig(BaseModel):
    mode: str = "balanced"
    max_input_tokens: int = 128000
    target_reduction: float = 0.40
    semantic_trigger_threshold: int = 4000  # Only use semantic for long context
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)


class ScoringWeights(BaseModel):
    frequency: float = 0.25
    embedding: float = 0.35
    position: float = 0.15
    entity: float = 0.25


class ScoringConfig(BaseModel):
    enabled: bool = True
    min_importance: float = 0.3
    weights: ScoringWeights = Field(default_factory=ScoringWeights)


class SemanticCacheConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.92
    max_entries: int = 10000
    ttl_seconds: int = 3600
    embedding_model: str = "all-MiniLM-L6-v2"


class PromptCacheConfig(BaseModel):
    enabled: bool = True
    backend: str = "sqlite"
    db_path: str = ".twotrim/prompt_cache.db"
    max_entries: int = 50000


class MemoryCacheConfig(BaseModel):
    enabled: bool = True
    max_sessions: int = 1000
    compress_after_turns: int = 5


class ContextStoreConfig(BaseModel):
    enabled: bool = True
    backend: str = "sqlite"
    db_path: str = ".twotrim/context_store.db"


class CacheConfig(BaseModel):
    semantic: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    prompt: PromptCacheConfig = Field(default_factory=PromptCacheConfig)
    memory: MemoryCacheConfig = Field(default_factory=MemoryCacheConfig)
    store: ContextStoreConfig = Field(default_factory=ContextStoreConfig)


class PolicyConfig(BaseModel):
    default_mode: str = "balanced"
    auto_adjust: bool = True
    max_degradation: float = 0.05
    per_model: dict[str, dict[str, Any]] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    enabled: bool = True
    sample_rate: float = 0.1
    similarity_threshold: float = 0.90
    store_results: bool = True
    results_path: str = ".twotrim/eval_results.jsonl"


class KVQuantizationConfig(BaseModel):
    enabled: bool = False
    bits: int = 8


class KVEvictionConfig(BaseModel):
    strategy: str = "lru"
    max_tokens: int = 4096


class KVTokenPruningConfig(BaseModel):
    enabled: bool = False
    min_attention: float = 0.01


class KVTokenMergingConfig(BaseModel):
    enabled: bool = False
    similarity_threshold: float = 0.95


class KVCacheConfig(BaseModel):
    enabled: bool = False
    quantization: KVQuantizationConfig = Field(default_factory=KVQuantizationConfig)
    eviction: KVEvictionConfig = Field(default_factory=KVEvictionConfig)
    token_pruning: KVTokenPruningConfig = Field(default_factory=KVTokenPruningConfig)
    token_merging: KVTokenMergingConfig = Field(default_factory=KVTokenMergingConfig)


class ResponseConfig(BaseModel):
    compress: bool = True
    remove_verbosity: bool = True
    enforce_structured: bool = False
    max_output_tokens: int | None = None


class PrometheusConfig(BaseModel):
    enabled: bool = True
    port: int | None = None


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str | None = None


class RequestLoggingConfig(BaseModel):
    enabled: bool = True
    log_prompts: bool = False


class ObservabilityConfig(BaseModel):
    prometheus: PrometheusConfig = Field(default_factory=PrometheusConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    request_logging: RequestLoggingConfig = Field(default_factory=RequestLoggingConfig)


class PluginsConfig(BaseModel):
    enabled: bool = True
    dirs: list[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    embedding: str = "all-MiniLM-L6-v2"
    summarization: str = "facebook/bart-large-cnn"
    device: str = "auto"
    cache_dir: str | None = None


class TwoTrimConfig(BaseModel):
    """Root configuration for the entire TwoTrim system."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    upstream: UpstreamRoutes = Field(default_factory=UpstreamRoutes)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    kv_cache: KVCacheConfig = Field(default_factory=KVCacheConfig)
    response: ResponseConfig = Field(default_factory=ResponseConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_config: TwoTrimConfig | None = None


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply TWOTRIM_* environment variables as overrides.

    Format: TWOTRIM_SECTION__KEY=value  (double underscore separator)
    """
    prefix = "TWOTRIM_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix):].lower().split("__")
        d = data
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        # Auto-convert types
        final_key = parts[-1]
        if value.lower() in ("true", "false"):
            d[final_key] = value.lower() == "true"
        elif value.isdigit():
            d[final_key] = int(value)
        else:
            try:
                d[final_key] = float(value)
            except ValueError:
                d[final_key] = value
    return data


def load_config(path: str | Path | None = None) -> TwoTrimConfig:
    """Load configuration from YAML file with env var overrides."""
    global _config

    data: dict[str, Any] = {}

    if path is not None:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f) or {}
    else:
        # Search default locations
        for candidate in ["config.yaml", "twotrim.yaml", ".twotrim/config.yaml"]:
            if Path(candidate).exists():
                with open(candidate) as f:
                    data = yaml.safe_load(f) or {}
                break

    data = _apply_env_overrides(data)
    _config = TwoTrimConfig(**data)
    return _config


def get_config() -> TwoTrimConfig:
    """Get the currently loaded config, or load defaults."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset config to force reload."""
    global _config
    _config = None
