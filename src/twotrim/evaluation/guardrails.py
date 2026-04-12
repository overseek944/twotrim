"""Guardrails — degradation thresholds and automatic rollback."""

from __future__ import annotations

import logging
from twotrim.types import EvalResult

logger = logging.getLogger(__name__)


class Guardrails:
    """Enforce quality thresholds for compression."""

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        max_degradation: float = 0.05,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_degradation = max_degradation
        self._violations: list[EvalResult] = []

    def check(self, result: EvalResult) -> bool:
        """Check if an evaluation result passes guardrails.

        Returns True if passed, False if violated.
        """
        passed = result.similarity_score >= self.similarity_threshold

        if not passed:
            self._violations.append(result)
            logger.warning(
                "Guardrail violation: similarity=%.3f < threshold=%.3f (request=%s)",
                result.similarity_score, self.similarity_threshold, result.request_id,
            )

        return passed

    def should_rollback(self, window: int = 20) -> bool:
        """Check if recent violations warrant rolling back compression."""
        if len(self._violations) < 3:
            return False

        recent = self._violations[-window:]
        violation_rate = len(recent) / window
        return violation_rate > self.max_degradation * 2

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    def reset(self) -> None:
        self._violations.clear()
