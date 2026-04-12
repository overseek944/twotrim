"""Cross-request memory — maintain compressed context across sessions.

Keeps a rolling compressed history per session, auto-compressing
older turns to save tokens on subsequent requests.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from typing import Any

from twotrim.types import ChatMessage

logger = logging.getLogger(__name__)


class SessionMemory:
    """Compressed context for a single session."""

    def __init__(self, session_id: str, compress_after_turns: int = 5) -> None:
        self.session_id = session_id
        self.compress_after_turns = compress_after_turns
        self.messages: list[ChatMessage] = []
        self.compressed_summary: str = ""
        self.turn_count: int = 0
        self.last_accessed: float = time.time()

    def add_messages(self, messages: list[ChatMessage]) -> None:
        """Add new messages and trigger compression if needed."""
        self.messages.extend(messages)
        self.turn_count += len([m for m in messages if m.role == "user"])
        self.last_accessed = time.time()

        if self.turn_count > self.compress_after_turns:
            self._compress_history()

    def get_compressed_context(self) -> list[ChatMessage]:
        """Get messages with older context compressed into a summary."""
        result: list[ChatMessage] = []

        if self.compressed_summary:
            result.append(ChatMessage(
                role="system",
                content=f"[Previous conversation summary]\n{self.compressed_summary}",
            ))

        result.extend(self.messages)
        return result

    def _compress_history(self) -> None:
        """Compress older messages into a summary, keep recent ones."""
        if len(self.messages) <= 4:
            return

        # Keep the last 4 messages, compress everything before
        to_compress = self.messages[:-4]
        self.messages = self.messages[-4:]

        # Build summary from compressed messages
        summary_parts: list[str] = []
        if self.compressed_summary:
            summary_parts.append(self.compressed_summary)

        for msg in to_compress:
            if msg.content:
                role = msg.role.upper()
                # Extract key points (first sentence of each)
                content = msg.content
                sentences = re.split(r"(?<=[.!?])\s+", content)
                if len(sentences) > 2:
                    content = " ".join(sentences[:2]) + "..."
                elif len(content) > 200:
                    content = content[:197] + "..."
                summary_parts.append(f"{role}: {content}")

        self.compressed_summary = "\n".join(summary_parts)

        # Limit summary length
        if len(self.compressed_summary) > 2000:
            lines = self.compressed_summary.split("\n")
            self.compressed_summary = "\n".join(lines[-20:])

        self.turn_count = 0


class MemoryManager:
    """Manage cross-request memory across multiple sessions."""

    def __init__(
        self,
        max_sessions: int = 1000,
        compress_after_turns: int = 5,
    ) -> None:
        self.max_sessions = max_sessions
        self.compress_after_turns = compress_after_turns
        self._sessions: OrderedDict[str, SessionMemory] = OrderedDict()

    def get_session(self, session_id: str) -> SessionMemory:
        """Get or create a session memory."""
        if session_id not in self._sessions:
            if len(self._sessions) >= self.max_sessions:
                self._sessions.popitem(last=False)  # Remove oldest

            self._sessions[session_id] = SessionMemory(
                session_id=session_id,
                compress_after_turns=self.compress_after_turns,
            )
        else:
            self._sessions.move_to_end(session_id)

        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def stats(self) -> dict[str, Any]:
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
        }
