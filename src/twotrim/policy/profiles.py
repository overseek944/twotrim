"""Compression profiles — presets for different use cases."""

from __future__ import annotations

from twotrim.types import CompressionMode, PolicyDecision, StrategyName

# ---------------------------------------------------------------------------
# Strategy sets per mode
# ---------------------------------------------------------------------------

LOSSLESS_STRATEGIES: list[StrategyName] = [
    StrategyName.RULE_BASED,
    StrategyName.CANONICALIZE,
]

BALANCED_STRATEGIES: list[StrategyName] = [
    StrategyName.RULE_BASED,
    StrategyName.EMBEDDING,
    StrategyName.CANONICALIZE,
    StrategyName.RAG_AWARE,
    StrategyName.STRUCTURED,
]

AGGRESSIVE_STRATEGIES: list[StrategyName] = [
    StrategyName.RULE_BASED,
    StrategyName.EMBEDDING,
    StrategyName.SEMANTIC,
    StrategyName.RAG_AWARE,
    StrategyName.STRUCTURED,
    StrategyName.CANONICALIZE,
]

# ---------------------------------------------------------------------------
# Target reductions per mode
# ---------------------------------------------------------------------------

TARGET_REDUCTIONS: dict[CompressionMode, float] = {
    CompressionMode.LOSSLESS: 0.10,
    CompressionMode.BALANCED: 0.40,
    CompressionMode.AGGRESSIVE: 0.65,
}

MAX_DEGRADATIONS: dict[CompressionMode, float] = {
    CompressionMode.LOSSLESS: 0.01,
    CompressionMode.BALANCED: 0.05,
    CompressionMode.AGGRESSIVE: 0.15,
}


def get_profile(mode: CompressionMode) -> PolicyDecision:
    """Get the default policy decision for a compression mode."""
    strategies_map = {
        CompressionMode.LOSSLESS: LOSSLESS_STRATEGIES,
        CompressionMode.BALANCED: BALANCED_STRATEGIES,
        CompressionMode.AGGRESSIVE: AGGRESSIVE_STRATEGIES,
    }

    return PolicyDecision(
        mode=mode,
        strategies=strategies_map[mode],
        target_reduction=TARGET_REDUCTIONS[mode],
        max_degradation=MAX_DEGRADATIONS[mode],
    )


# ---------------------------------------------------------------------------
# Request type hints for policy adaptation
# ---------------------------------------------------------------------------

REQUEST_TYPE_PROFILES: dict[str, CompressionMode] = {
    "chat": CompressionMode.BALANCED,
    "reasoning": CompressionMode.LOSSLESS,
    "coding": CompressionMode.LOSSLESS,
    "creative": CompressionMode.BALANCED,
    "summarization": CompressionMode.AGGRESSIVE,
    "translation": CompressionMode.LOSSLESS,
    "analysis": CompressionMode.BALANCED,
    "data_extraction": CompressionMode.AGGRESSIVE,
    "qa": CompressionMode.BALANCED,
}
