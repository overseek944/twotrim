"""Quality evaluation system.

Runs shadow evaluations on sampled requests to monitor
compression quality and feed back into the policy engine.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from twotrim.config import get_config
from twotrim.evaluation.guardrails import Guardrails
from twotrim.evaluation.similarity import SimilarityScorer
from twotrim.policy.engine import get_policy_engine
from twotrim.types import CompressionResult, EvalResult

logger = logging.getLogger(__name__)


class Evaluator:
    """Shadow evaluation system for compression quality."""

    def __init__(self) -> None:
        self._scorer = SimilarityScorer()
        self._guardrails: Guardrails | None = None
        self._results: list[EvalResult] = []

    def _ensure_init(self) -> None:
        if self._guardrails is not None:
            return
        cfg = get_config().evaluation
        self._guardrails = Guardrails(
            similarity_threshold=cfg.similarity_threshold,
        )

    def should_evaluate(self) -> bool:
        """Determine if this request should be sampled for evaluation."""
        cfg = get_config().evaluation
        if not cfg.enabled:
            return False
        return random.random() < cfg.sample_rate

    async def evaluate(self, result: CompressionResult) -> EvalResult | None:
        """Evaluate compression quality for a single result."""
        self._ensure_init()
        cfg = get_config().evaluation

        if result.overall_ratio < 0.01:
            return None  # No meaningful compression occurred

        # Compute similarity
        similarity = self._scorer.score(result.original_text, result.compressed_text)

        eval_result = EvalResult(
            request_id=result.request_id,
            similarity_score=similarity,
            passed=similarity >= cfg.similarity_threshold,
            threshold=cfg.similarity_threshold,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            compression_ratio=result.overall_ratio,
        )

        # Check guardrails
        if self._guardrails:
            self._guardrails.check(eval_result)

        # Store result
        self._results.append(eval_result)

        # Feed back into policy engine
        policy = get_policy_engine()
        policy.report_quality(similarity, cfg.similarity_threshold)

        # Persist to file
        if cfg.store_results:
            await self._store_result(eval_result, cfg.results_path)

        logger.info(
            "Eval: similarity=%.3f, passed=%s, ratio=%.1f%% (request=%s)",
            similarity, eval_result.passed, result.overall_ratio * 100, result.request_id,
        )

        return eval_result

    async def _store_result(self, result: EvalResult, path: str) -> None:
        """Append evaluation result to JSONL file."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a") as f:
                f.write(result.model_dump_json() + "\n")
        except Exception as e:
            logger.warning("Failed to store eval result: %s", e)

    def get_stats(self) -> dict[str, Any]:
        """Get evaluation statistics."""
        if not self._results:
            return {"total_evaluations": 0}

        scores = [r.similarity_score for r in self._results]
        passed = [r for r in self._results if r.passed]

        return {
            "total_evaluations": len(self._results),
            "pass_rate": len(passed) / len(self._results),
            "avg_similarity": sum(scores) / len(scores),
            "min_similarity": min(scores),
            "max_similarity": max(scores),
            "violation_count": self._guardrails.violation_count if self._guardrails else 0,
        }


# Module singleton
_evaluator: Evaluator | None = None


def get_evaluator() -> Evaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = Evaluator()
    return _evaluator
