"""TwoTrim SDK client — drop-in replacement for OpenAI client.

Can work in two modes:
1. Proxy mode: point at a running TwoTrim proxy server
2. Inline mode: run compression locally before calling upstream LLM

Usage:
    from twotrim.sdk.client import TwoTrimClient

    client = TwoTrimClient(
        upstream_base_url="https://api.openai.com/v1",
        api_key="sk-...",
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from __future__ import annotations

import os
import json
import asyncio
import logging
import time
from typing import Any, Iterator, Optional

import httpx

from twotrim.compression.pipeline import get_pipeline
from twotrim.policy.profiles import get_profile
from twotrim.types import CompressionMode, CompressionResult

logger = logging.getLogger(__name__)


class TwoTrimResponse(dict):
    """Wrapper for API responses to add metadata attribute."""
    def __init__(self, data: dict[str, Any], metadata: Optional[CompressionResult] = None):
        super().__init__(data)
        self.twotrim_metadata = {
            "original_tokens": metadata.original_tokens,
            "compressed_tokens": metadata.compressed_tokens,
            "compression_ratio": metadata.overall_ratio,
            "compression_time_ms": metadata.compression_time_ms,
            "strategies": [s.strategy.value for s in metadata.strategies_applied]
        } if metadata else {}

    @property
    def choices(self):
        # Mocking OpenAI response object structure for basic attribute access
        from dataclasses import dataclass
        @dataclass
        class Message:
            content: str
        @dataclass
        class Choice:
            message: Message
        
        choices_data = self.get("choices", [])
        return [Choice(Message(c.get("message", {}).get("content", ""))) for n, c in enumerate(choices_data)]


class _ChatCompletions:
    """Chat completions API (OpenAI-compatible)."""

    def __init__(self, client: TwoTrimClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> TwoTrimResponse:
        """Create a chat completion (synchronous)."""
        metadata = None
        if self._client._mode == "inline":
            metadata, kwargs = self._client._compress_sync(kwargs)
            
        resp_data = self._client._request("POST", "/chat/completions", json=kwargs)
        return TwoTrimResponse(resp_data, metadata)

    async def acreate(self, **kwargs: Any) -> TwoTrimResponse:
        """Create a chat completion (async)."""
        metadata = None
        if self._client._mode == "inline":
            metadata, kwargs = await self._client._compress_async(kwargs)
            
        resp_data = await self._client._arequest("POST", "/chat/completions", json=kwargs)
        return TwoTrimResponse(resp_data, metadata)


class _Completions:
    """Legacy completions API."""

    def __init__(self, client: TwoTrimClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._client._request("POST", "/completions", json=kwargs)

    async def acreate(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client._arequest("POST", "/completions", json=kwargs)


class _Embeddings:
    """Embeddings API."""

    def __init__(self, client: TwoTrimClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        return self._client._request("POST", "/embeddings", json=kwargs)

    async def acreate(self, **kwargs: Any) -> dict[str, Any]:
        return await self._client._arequest("POST", "/embeddings", json=kwargs)


class _Chat:
    """Chat namespace."""

    def __init__(self, client: TwoTrimClient) -> None:
        self.completions = _ChatCompletions(client)


class TwoTrimClient:
    """Drop-in replacement for OpenAI client that routes through TwoTrim.

    Args:
        proxy_url: URL of a running TwoTrim proxy. If set, all requests
                   go through the proxy. Mutually exclusive with upstream_base_url
                   in inline mode.
        upstream_base_url: Direct upstream LLM URL (for inline mode).
        api_key: API key for the upstream LLM.
        compression_mode: Default compression mode (lossless/balanced/aggressive).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        proxy_url: str | None = None,
        upstream_base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        compression_mode: str = "balanced",
        timeout: int = 120,
    ) -> None:
        if proxy_url:
            self._base_url = proxy_url.rstrip("/")
            self._mode = "proxy"
        else:
            self._base_url = upstream_base_url.rstrip("/")
            self._mode = "inline"

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._compression_mode = compression_mode
        self._timeout = timeout

        # API namespaces
        self.chat = _Chat(self)
        self.completions = _Completions(self)
        self.embeddings = _Embeddings(self)

    def _compress_sync(self, kwargs: dict[str, Any]) -> tuple[CompressionResult, dict[str, Any]]:
        """Run compression synchronously (wraps async)."""
        return asyncio.run(self._compress_async(kwargs))

    async def _compress_async(self, kwargs: dict[str, Any]) -> tuple[CompressionResult, dict[str, Any]]:
        """Run compression on the latest user message."""
        messages = kwargs.get("messages", [])
        if not messages:
            return None, kwargs

        # Get latest user message
        last_user_idx = -1
        for idx, msg in enumerate(reversed(messages)):
            if msg.get("role") == "user":
                last_user_idx = len(messages) - 1 - idx
                break
        
        if last_user_idx == -1:
            return None, kwargs

        text = messages[last_user_idx].get("content", "")
        if not text:
            return None, kwargs

        # Extract potential query from the end of the text (e.g. "Question: ...")
        # For Qasper, the prompt follows "... \n\n Question: [Question]"
        query_text = None
        context = text
        if "Question:" in text:
            parts = text.rsplit("Question:", 1)
            context = parts[0].strip()
            query_text = parts[1].strip()
        elif "?" in text:
            # Fallback: if last sentence has a question mark, keep it intact
            import re
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if len(sentences) > 1 and "?" in sentences[-1]:
                query_text = sentences[-1].strip()
                context = " ".join(sentences[:-1]).strip()

        # Determine mode
        extra = kwargs.get("extra_body") or {}
        mode_val = extra.get("compression_mode") or self._compression_mode
        mode = CompressionMode(mode_val)
        
        # Strip extension parameters before sending to upstream LLM
        if "extra_body" in kwargs:
            del kwargs["extra_body"]
        
        pipeline = get_pipeline()
        decision = get_profile(mode)
        # Pass the extracted query to guide summarization
        decision.query = query_text if query_text else context

        # Compress only the context part, keeping the query bit intact
        result = await pipeline.compress(context, decision)
        
        # Update messages: Re-append the original question to the compressed context
        new_messages = list(messages)
        if query_text:
            new_messages[last_user_idx]["content"] = f"{result.compressed_text}\n\nQuestion: {query_text}"
        else:
            new_messages[last_user_idx]["content"] = result.compressed_text
        kwargs["messages"] = new_messages
        
        return result, kwargs

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Synchronous HTTP request."""
        url = f"{self._base_url}{path}"
        headers = self._build_headers()

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.request(method, url, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp.json()

    async def _arequest(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Async HTTP request."""
        url = f"{self._base_url}{path}"
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.request(method, url, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp.json()

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._mode == "proxy":
            headers["X-TwoTrim-Mode"] = self._compression_mode
        return headers

    def get_stats(self) -> dict[str, Any]:
        """Get compression statistics from the proxy."""
        if self._mode != "proxy":
            return {"error": "Stats only available in proxy mode"}
        return self._request("GET", "/stats")

    def get_recent_metrics(self, n: int = 50) -> list[dict[str, Any]]:
        """Get recent request metrics."""
        if self._mode != "proxy":
            return [{"error": "Metrics only available in proxy mode"}]
        return self._request("GET", f"/stats/recent?n={n}")

    def health(self) -> dict[str, Any]:
        """Check proxy health."""
        return self._request("GET", "/health")
