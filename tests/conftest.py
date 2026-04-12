"""Shared test fixtures."""

from __future__ import annotations

import os
import pytest

from twotrim.config import load_config, reset_config, TwoTrimConfig


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config between tests."""
    reset_config()
    # Set minimal defaults for testing
    os.environ.setdefault("TWOTRIM_COMPRESSION__MODE", "balanced")
    yield
    reset_config()


@pytest.fixture
def config() -> TwoTrimConfig:
    """Load default config."""
    return load_config()


@pytest.fixture
def sample_text() -> str:
    """A realistic sample text for compression testing."""
    return (
        "Machine learning is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Basically, machine learning algorithms "
        "use historical data as input to predict new output values. It is important to "
        "note that recommendation engines are a common use case for machine learning. "
        "Machine learning is essentially a method of training algorithms such that they "
        "can learn to make decisions and predictions based on data. In other words, "
        "machine learning enables a system to learn from data rather than through "
        "explicit programming.\n\n"
        "There are several types of machine learning algorithms. Supervised learning "
        "algorithms are trained using labeled examples. Unsupervised learning algorithms "
        "are used against data that has no historical labels. Semi-supervised learning "
        "falls between supervised and unsupervised learning.\n\n"
        "Deep learning is actually a subset of machine learning. Deep learning uses "
        "neural networks with many layers. It is worth noting that deep learning has "
        "driven many recent advances in AI, including computer vision and natural language "
        "processing. As a matter of fact, deep learning models are behind many of the "
        "AI applications we use today."
    )


@pytest.fixture
def short_text() -> str:
    """Short text that should be minimally compressed."""
    return "What is the capital of France?"


@pytest.fixture
def rag_text() -> str:
    """A RAG-style prompt with retrieved documents."""
    return (
        "Based on the following retrieved documents, answer the question.\n\n"
        "Document 1: Python is a high-level programming language known for its "
        "readability and versatility. It was created by Guido van Rossum and first "
        "released in 1991.\n\n"
        "Document 2: Python supports multiple programming paradigms, including "
        "procedural, object-oriented, and functional programming. It has a large "
        "standard library.\n\n"
        "Document 3: The weather in Paris today is sunny with temperatures around "
        "22 degrees Celsius. Perfect weather for outdoor activities.\n\n"
        "Question: What programming paradigms does Python support?"
    )


@pytest.fixture
def chat_messages() -> list[dict[str, str]]:
    """Sample chat messages."""
    return [
        {
            "role": "system",
            "content": (
                "You are a very helpful and knowledgeable AI assistant. "
                "Please provide detailed and accurate information when answering questions. "
                "Make sure to be thorough in your explanations."
            ),
        },
        {
            "role": "user",
            "content": "What is machine learning? Please explain in detail.",
        },
    ]
