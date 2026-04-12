"""Adaptive policy engine.

Decides compression strategy per-request based on model type,
request classification, cost sensitivity, and feedback from evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

from twotrim.config import get_config
from twotrim.policy.profiles import get_profile
from twotrim.policy.rules import (
    classify_request_type,
    estimate_token_count,
    select_mode_for_model,
    should_skip_compression,
)
from twotrim.types import CompressionMode, PolicyDecision, StrategyName

logger = logging.getLogger(__name__)


class PolicyEngine:
    """Adaptive compression policy engine."""

    def __init__(self) -> None:
        self._degradation_history: list[float] = []
        self._auto_downgrade_active = False

    def decide(
        self,
        model: str,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        override_mode: CompressionMode | None = None,
    ) -> PolicyDecision:
        """Make a policy decision for the given request."""
        cfg = get_config().policy

        # Estimate token count
        token_count = estimate_token_count(messages, prompt)

        # Classify request type
        request_type = classify_request_type(messages, prompt)

        # Check if compression should be skipped
        if should_skip_compression(model, token_count, request_type):
            return PolicyDecision(
                mode=CompressionMode.LOSSLESS,
                strategies=[],
                target_reduction=0.0,
                max_degradation=0.0,
                skip_cache=True,
                metadata={"reason": "skipped", "token_count": token_count},
            )

        # Determine mode
        if override_mode is not None:
            mode = override_mode
        else:
            # Check model-specific config
            model_mode = select_mode_for_model(model, cfg.per_model)
            if model_mode:
                mode = model_mode
            else:
                mode = CompressionMode(cfg.default_mode)

        # Auto-downgrade if evaluation has detected quality issues
        if self._auto_downgrade_active and mode == CompressionMode.AGGRESSIVE:
            mode = CompressionMode.BALANCED
            logger.info("Auto-downgraded from AGGRESSIVE to BALANCED due to quality feedback")

        # Get base profile
        decision = get_profile(mode)

        # Adjust based on request type
        decision = self._adjust_for_request_type(decision, request_type)

        # Adjust based on token count
        decision = self._adjust_for_token_count(decision, token_count)

        # Override max_degradation from config
        if cfg.max_degradation:
            decision.max_degradation = cfg.max_degradation

        decision.metadata = {
            "model": model,
            "request_type": request_type,
            "token_count": token_count,
            "auto_downgrade": self._auto_downgrade_active,
        }

        logger.debug(
            "Policy: mode=%s, strategies=%s, target=%.0f%%, request_type=%s",
            mode.value,
            [s.value for s in decision.strategies],
            decision.target_reduction * 100,
            request_type,
        )

        return decision

    def report_quality(self, similarity_score: float, threshold: float) -> None:
        """Report quality evaluation result for adaptive adjustment."""
        cfg = get_config().policy

        self._degradation_history.append(similarity_score)

        # Keep last 50 results
        if len(self._degradation_history) > 50:
            self._degradation_history = self._degradation_history[-50:]

        # Check if we should auto-downgrade
        if cfg.auto_adjust and len(self._degradation_history) >= 10:
            recent = self._degradation_history[-10:]
            failures = sum(1 for s in recent if s < threshold)
            if failures >= 3:
                self._auto_downgrade_active = True
                logger.warning(
                    "Auto-downgrade activated: %d/%d recent evaluations below threshold",
                    failures, len(recent),
                )
            elif failures == 0 and self._auto_downgrade_active:
                self._auto_downgrade_active = False
                logger.info("Auto-downgrade deactivated: quality restored")

    def _adjust_for_request_type(
        self, decision: PolicyDecision, request_type: str
    ) -> PolicyDecision:
        """Adjust strategy based on request type."""
        if request_type in ("coding", "reasoning", "translation"):
            # Conservative: remove semantic compression
            decision.strategies = [
                s for s in decision.strategies
                if s != StrategyName.SEMANTIC
            ]
            decision.target_reduction = min(decision.target_reduction, 0.25)
            decision.max_degradation = min(decision.max_degradation, 0.03)

        elif request_type in ("summarization", "data_extraction"):
            # Can be aggressive
            if StrategyName.SEMANTIC not in decision.strategies:
                decision.strategies.append(StrategyName.SEMANTIC)
            decision.target_reduction = max(decision.target_reduction, 0.50)

        return decision

    def _adjust_for_token_count(
        self, decision: PolicyDecision, token_count: int
    ) -> PolicyDecision:
        """Adjust strategy based on input size."""
        if token_count > 10000:
            # Large inputs benefit more from embedding + semantic
            if StrategyName.EMBEDDING not in decision.strategies:
                decision.strategies.append(StrategyName.EMBEDDING)
            decision.target_reduction = min(decision.target_reduction + 0.10, 0.80)

        elif token_count < 200:
            # Small inputs: only rule-based
            decision.strategies = [
                s for s in decision.strategies
                if s in (StrategyName.RULE_BASED, StrategyName.CANONICALIZE)
            ]
            decision.target_reduction = min(decision.target_reduction, 0.15)

        return decision


# Module singleton
_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    global _engine
    if _engine is None:
        _engine = PolicyEngine()
    return _engine
