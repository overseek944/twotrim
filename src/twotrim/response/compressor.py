"""Response compression — post-process LLM outputs.

Removes verbosity, enforces structured output where applicable,
and compresses responses for caching/reuse.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class ResponseCompressor:
    """Post-process and compress LLM responses."""

    def __init__(
        self,
        remove_verbosity: bool = True,
        enforce_structured: bool = False,
        max_output_tokens: int | None = None,
    ) -> None:
        self.remove_verbosity = remove_verbosity
        self.enforce_structured = enforce_structured
        self.max_output_tokens = max_output_tokens

    def compress(self, response: dict[str, Any]) -> dict[str, Any]:
        """Compress a response dict (OpenAI format)."""
        choices = response.get("choices", [])
        if not choices:
            return response

        compressed_response = response.copy()
        compressed_choices = []

        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")

            if content and isinstance(content, str):
                compressed_content = self._compress_content(content)
                compressed_message = {**message, "content": compressed_content}
                compressed_choices.append({**choice, "message": compressed_message})
            else:
                compressed_choices.append(choice)

        compressed_response["choices"] = compressed_choices

        # Add compression metadata
        original_tokens = response.get("usage", {}).get("completion_tokens", 0)
        if original_tokens and compressed_choices:
            first_content = compressed_choices[0].get("message", {}).get("content", "")
            compressed_tokens = max(1, int(len(first_content.split()) / 0.75))
            compressed_response.setdefault("_twotrim", {})["response_compression"] = {
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "ratio": round(1 - compressed_tokens / max(original_tokens, 1), 3),
            }

        return compressed_response

    def _compress_content(self, content: str) -> str:
        """Apply compression to response content."""
        result = content

        if self.remove_verbosity:
            result = self._remove_verbosity(result)

        if self.enforce_structured:
            result = self._to_structured(result)

        if self.max_output_tokens:
            result = self._truncate(result, self.max_output_tokens)

        return result

    def _remove_verbosity(self, text: str) -> str:
        """Remove verbose preambles and filler from responses."""
        # Remove common verbose openings
        verbose_openings = [
            r"^(?:Sure[!,.]?\s*(?:Here(?:'s| is| are)?\s*)?)",
            r"^(?:Of course[!,.]?\s*(?:Here(?:'s| is| are)?\s*)?)",
            r"^(?:Absolutely[!,.]?\s*)",
            r"^(?:Great question[!,.]?\s*)",
            r"^(?:That's a (?:great|good|interesting) (?:question|point)[!,.]?\s*)",
            r"^(?:I'd be happy to help[!,.]?\s*)",
            r"^(?:Let me (?:help you with that|explain)[!,.]?\s*)",
        ]

        for pattern in verbose_openings:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Remove verbose closings
        verbose_closings = [
            r"\s*(?:I hope (?:this|that) helps[!.]?\s*(?:Let me know if.*)?$)",
            r"\s*(?:Feel free to (?:ask|reach out|let me know).*$)",
            r"\s*(?:Is there anything else.*$)",
            r"\s*(?:Don't hesitate to.*$)",
            r"\s*(?:Happy to help further[!.]?\s*$)",
        ]

        for pattern in verbose_closings:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Collapse repeated transitions
        text = re.sub(r"(?:Additionally|Furthermore|Moreover),\s*(?:it(?:'s| is) (?:also )?(?:worth|important) (?:noting|mentioning) that\s*)", "", text, flags=re.IGNORECASE)

        return text.strip()

    def _to_structured(self, text: str) -> str:
        """Attempt to convert response to structured format."""
        # Check if response is already JSON
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                return json.dumps(parsed, separators=(",", ":"))
            except json.JSONDecodeError:
                pass

        # Try to extract key-value pairs
        kv_pattern = re.compile(r"^\s*[-•*]?\s*\*?\*?(.+?)\*?\*?\s*[:–—]\s*(.+)$", re.MULTILINE)
        matches = kv_pattern.findall(text)
        if len(matches) >= 3:
            result = {k.strip(): v.strip() for k, v in matches}
            return json.dumps(result, indent=2)

        return text

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit."""
        words = text.split()
        # ~0.75 tokens per word
        max_words = int(max_tokens * 0.75)
        if len(words) <= max_words:
            return text

        truncated = " ".join(words[:max_words])
        # Try to end at a sentence boundary
        last_period = truncated.rfind(".")
        if last_period > len(truncated) * 0.7:
            truncated = truncated[:last_period + 1]

        return truncated

    def create_summary(self, content: str, max_length: int = 200) -> str:
        """Create a short summary of response content for caching."""
        sentences = re.split(r"(?<=[.!?])\s+", content)
        summary_parts: list[str] = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > max_length and summary_parts:
                break
            summary_parts.append(sentence)
            current_length += len(sentence)

        return " ".join(summary_parts)
