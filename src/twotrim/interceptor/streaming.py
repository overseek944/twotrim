"""SSE streaming handler for proxied LLM responses."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


async def stream_response(
    upstream_response: Any,
    request_id: str,
    twotrim_meta: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Stream SSE events from upstream response, adding TwoTrim metadata.

    Args:
        upstream_response: httpx streaming response
        request_id: Request ID for tracking
        twotrim_meta: Compression metadata to inject
    """
    full_content: list[str] = []
    chunk_count = 0

    try:
        async for line in upstream_response.aiter_lines():
            if not line:
                yield "\n"
                continue

            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    # Inject final metadata event before DONE
                    if twotrim_meta:
                        meta_event = {
                            "id": request_id,
                            "object": "twotrim.metadata",
                            **twotrim_meta,
                        }
                        yield f"data: {json.dumps(meta_event)}\n\n"
                    yield "data: [DONE]\n\n"
                    break

                try:
                    chunk = json.loads(data)
                    # Track content for metrics
                    choices = chunk.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_content.append(content)

                    chunk_count += 1
                    yield f"data: {json.dumps(chunk)}\n\n"
                except json.JSONDecodeError:
                    yield f"data: {data}\n\n"
            else:
                yield f"{line}\n"

    except Exception as e:
        logger.error("Streaming error for request %s: %s", request_id, e)
        error_chunk = {
            "error": {"message": str(e), "type": "stream_error"},
            "id": request_id,
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    logger.debug(
        "Stream complete: request=%s, chunks=%d, chars=%d",
        request_id, chunk_count, sum(len(c) for c in full_content),
    )


def is_streaming_request(body: dict[str, Any]) -> bool:
    """Check if the request asks for streaming."""
    return body.get("stream", False) is True
