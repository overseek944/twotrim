"""OpenAI API compatibility layer.

Handles request/response format conversion to ensure full
compatibility with the OpenAI API specification.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from twotrim.types import ChatMessage, RequestType


def parse_openai_request(body: dict[str, Any]) -> tuple[RequestType, list[ChatMessage] | None, str | None, str]:
    """Parse an incoming OpenAI-format request.

    Returns: (request_type, messages, prompt, model)
    """
    model = body.get("model", "unknown")

    # Chat completion
    if "messages" in body:
        messages = [ChatMessage(**m) for m in body["messages"]]
        return RequestType.CHAT_COMPLETION, messages, None, model

    # Legacy completion
    if "prompt" in body:
        prompt = body["prompt"]
        if isinstance(prompt, list):
            prompt = "\n".join(str(p) for p in prompt)
        return RequestType.COMPLETION, None, prompt, model

    # Embedding
    if "input" in body:
        inp = body["input"]
        if isinstance(inp, list):
            inp = " ".join(str(i) for i in inp)
        return RequestType.EMBEDDING, None, inp, model

    return RequestType.CHAT_COMPLETION, None, None, model


def build_compressed_body(
    original_body: dict[str, Any],
    compressed_messages: list[ChatMessage] | None = None,
    compressed_prompt: str | None = None,
) -> dict[str, Any]:
    """Rebuild the request body with compressed content."""
    body = original_body.copy()

    if compressed_messages is not None:
        body["messages"] = [m.model_dump(exclude_none=True) for m in compressed_messages]
    elif compressed_prompt is not None:
        if "prompt" in body:
            body["prompt"] = compressed_prompt
        elif "input" in body:
            body["input"] = compressed_prompt

    return body


def extract_text_from_messages(messages: list[ChatMessage]) -> str:
    """Extract all text content from messages for compression."""
    parts: list[str] = []
    for msg in messages:
        if msg.content:
            parts.append(f"[{msg.role}] {msg.content}")
    return "\n\n".join(parts)


def rebuild_messages_from_compressed(
    original_messages: list[ChatMessage],
    compressed_text: str,
) -> list[ChatMessage]:
    """Rebuild message list from compressed text.

    Strategy: compress user messages and long system messages,
    preserve assistant messages and tool calls.
    """
    result: list[ChatMessage] = []

    # Find which messages had compressible content
    compressible_indices: list[int] = []
    for i, msg in enumerate(original_messages):
        if msg.role in ("user", "system") and msg.content and len(msg.content) > 50:
            compressible_indices.append(i)

    if not compressible_indices:
        return original_messages

    # Simple approach: replace the largest compressible message content
    # with the compressed version, keep others as-is
    if len(compressible_indices) == 1:
        idx = compressible_indices[0]
        for i, msg in enumerate(original_messages):
            if i == idx:
                # Extract the compressed content for this role
                role_prefix = f"[{msg.role}] "
                if compressed_text.startswith(role_prefix):
                    compressed_content = compressed_text[len(role_prefix):]
                else:
                    compressed_content = compressed_text
                result.append(ChatMessage(
                    role=msg.role,
                    content=compressed_content,
                    name=msg.name,
                ))
            else:
                result.append(msg)
    else:
        # Multiple compressible messages: split compressed text by role markers
        compressed_parts = _split_by_role_markers(compressed_text)
        part_idx = 0

        for i, msg in enumerate(original_messages):
            if i in compressible_indices and part_idx < len(compressed_parts):
                _role, content = compressed_parts[part_idx]
                result.append(ChatMessage(
                    role=msg.role,
                    content=content,
                    name=msg.name,
                ))
                part_idx += 1
            else:
                result.append(msg)

    return result


def _split_by_role_markers(text: str) -> list[tuple[str, str]]:
    """Split compressed text by [role] markers."""
    import re
    parts: list[tuple[str, str]] = []
    pattern = re.compile(r"\[(\w+)\]\s*")

    matches = list(pattern.finditer(text))
    for i, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        parts.append((role, content))

    if not parts:
        parts.append(("user", text))

    return parts


def wrap_openai_response(
    response: dict[str, Any],
    twotrim_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ensure response conforms to OpenAI format and add TwoTrim metadata."""
    # Ensure required fields
    response.setdefault("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    response.setdefault("object", "chat.completion")
    response.setdefault("created", int(time.time()))

    if twotrim_meta:
        response["_twotrim"] = twotrim_meta

    return response
