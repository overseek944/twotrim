"""Tests for policy engine."""

import pytest
from twotrim.policy.engine import PolicyEngine
from twotrim.types import CompressionMode, StrategyName


@pytest.fixture
def engine():
    return PolicyEngine()


class TestPolicyEngine:

    def test_default_balanced(self, engine):
        content = "Hello, this is a longer test message to ensure the token count is above the skipping threshold for the policy engine to function normally without skipping." * 10
        decision = engine.decide(model="gpt-4", messages=[{"role": "user", "content": content}])
        assert decision.mode == CompressionMode.BALANCED

    def test_override_mode(self, engine):
        content = "Hello world test " * 20
        decision = engine.decide(
            model="gpt-4",
            messages=[{"role": "user", "content": content}],
            override_mode=CompressionMode.AGGRESSIVE,
        )
        assert decision.mode == CompressionMode.AGGRESSIVE

    def test_skip_short_prompts(self, engine):
        decision = engine.decide(model="gpt-4", prompt="Hi")
        assert len(decision.strategies) == 0
        assert decision.metadata.get("reason") == "skipped"

    def test_coding_request_conservative(self, engine):
        content = "Debug this Python code:\n```\ndef foo(): return bar\n```\nI'm getting an error. " * 10
        decision = engine.decide(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": content
            }],
        )
        # Should not include semantic compression for coding
        assert StrategyName.SEMANTIC not in decision.strategies

    def test_summarization_aggressive(self, engine):
        content = "Please summarize the following long text about machine learning and AI. " * 30
        decision = engine.decide(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": content
            }],
        )
        assert decision.target_reduction >= 0.40

    def test_quality_feedback_downgrade(self, engine):
        # Report multiple quality failures
        for _ in range(10):
            engine.report_quality(0.7, 0.9)

        content = "Test message for quality check " * 15
        decision = engine.decide(
            model="gpt-4",
            messages=[{"role": "user", "content": content}],
            override_mode=CompressionMode.AGGRESSIVE,
        )
        # Should auto-downgrade from aggressive
        assert decision.mode == CompressionMode.BALANCED

    def test_quality_feedback_restore(self, engine):
        # First cause downgrade
        for _ in range(10):
            engine.report_quality(0.7, 0.9)
        # Then restore
        for _ in range(10):
            engine.report_quality(0.95, 0.9)

        assert not engine._auto_downgrade_active
