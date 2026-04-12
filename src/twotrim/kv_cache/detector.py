"""Auto-detect model capabilities for KV cache optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelCapabilities:
    """Detected capabilities for a model."""
    name: str
    supports_kv_cache: bool = False
    supports_quantization: bool = False
    supports_token_pruning: bool = False
    supports_token_merging: bool = False
    supports_flash_attention: bool = False
    max_context_length: int = 4096
    is_local: bool = False
    runtime: str = "unknown"  # vllm | tgi | ollama | hf | api
    metadata: dict[str, str] = field(default_factory=dict)


# Known model families and their capabilities
MODEL_CAPABILITIES: dict[str, dict[str, bool | int]] = {
    "llama": {
        "supports_kv_cache": True,
        "supports_quantization": True,
        "supports_token_pruning": True,
        "supports_flash_attention": True,
        "max_context_length": 128000,
    },
    "mistral": {
        "supports_kv_cache": True,
        "supports_quantization": True,
        "supports_token_pruning": True,
        "supports_flash_attention": True,
        "max_context_length": 32768,
    },
    "phi": {
        "supports_kv_cache": True,
        "supports_quantization": True,
        "supports_flash_attention": True,
        "max_context_length": 128000,
    },
    "qwen": {
        "supports_kv_cache": True,
        "supports_quantization": True,
        "supports_flash_attention": True,
        "max_context_length": 32768,
    },
    "gemma": {
        "supports_kv_cache": True,
        "supports_quantization": True,
        "supports_flash_attention": True,
        "max_context_length": 8192,
    },
}


def detect_capabilities(model_name: str, base_url: str = "") -> ModelCapabilities:
    """Detect model capabilities based on model name and endpoint."""
    model_lower = model_name.lower()

    caps = ModelCapabilities(name=model_name)

    # Detect runtime from URL
    if "localhost" in base_url or "127.0.0.1" in base_url:
        caps.is_local = True
        if ":8000" in base_url or "vllm" in base_url.lower():
            caps.runtime = "vllm"
        elif ":8080" in base_url or "tgi" in base_url.lower():
            caps.runtime = "tgi"
        elif ":11434" in base_url or "ollama" in base_url.lower():
            caps.runtime = "ollama"
        else:
            caps.runtime = "local"

    # Cloud API models — no KV cache access
    if any(api in base_url for api in ["openai.com", "anthropic.com", "googleapis.com"]):
        caps.runtime = "api"
        caps.is_local = False
        # API models have large contexts but no cache access
        if "gpt-4" in model_lower:
            caps.max_context_length = 128000
        elif "claude" in model_lower:
            caps.max_context_length = 200000
        return caps

    # Match against known model families
    for family, family_caps in MODEL_CAPABILITIES.items():
        if family in model_lower:
            for key, value in family_caps.items():
                setattr(caps, key, value)
            break

    # vLLM-specific capabilities
    if caps.runtime == "vllm":
        caps.supports_kv_cache = True
        caps.supports_quantization = True
        caps.supports_flash_attention = True

    logger.debug("Detected capabilities for %s: %s", model_name, caps)
    return caps


def check_vllm_available() -> bool:
    """Check if vLLM is available in the environment."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def check_flash_attention_available() -> bool:
    """Check if FlashAttention is available."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False
