"""Deduplicated context store.

Store large text corpora once and reference via IDs instead
of resending full content on every request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ContextStore:
    """Store and retrieve deduplicated context by content hash."""

    def __init__(
        self,
        backend: str = "sqlite",
        db_path: str = ".twotrim/context_store.db",
    ) -> None:
        self.backend = backend
        self.db_path = db_path
        self._initialized = False

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        if self.backend == "sqlite":
            import aiosqlite
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS context_store (
                        content_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        created_at REAL,
                        access_count INTEGER DEFAULT 0,
                        token_count INTEGER DEFAULT 0
                    )
                """)
                await db.commit()
        self._initialized = True

    async def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store content and return a compact reference ID."""
        await self._ensure_init()
        content_id = self._content_hash(content)

        if self.backend == "sqlite":
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                existing = await (await db.execute(
                    "SELECT content_id FROM context_store WHERE content_id = ?",
                    (content_id,)
                )).fetchone()

                if not existing:
                    token_count = max(1, int(len(content.split()) / 0.75))
                    await db.execute(
                        """INSERT INTO context_store
                           (content_id, content, metadata, created_at, token_count)
                           VALUES (?, ?, ?, ?, ?)""",
                        (content_id, content,
                         json.dumps(metadata) if metadata else None,
                         time.time(), token_count)
                    )
                else:
                    await db.execute(
                        "UPDATE context_store SET access_count = access_count + 1 WHERE content_id = ?",
                        (content_id,)
                    )
                await db.commit()

        return content_id

    async def retrieve(self, content_id: str) -> str | None:
        """Retrieve stored content by ID."""
        await self._ensure_init()

        if self.backend == "sqlite":
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                row = await (await db.execute(
                    "SELECT content FROM context_store WHERE content_id = ?",
                    (content_id,)
                )).fetchone()
                if row:
                    await db.execute(
                        "UPDATE context_store SET access_count = access_count + 1 WHERE content_id = ?",
                        (content_id,)
                    )
                    await db.commit()
                    return row[0]
        return None

    async def replace_with_refs(self, text: str, min_length: int = 200) -> tuple[str, list[str]]:
        """Replace large text blocks with references.

        Returns the modified text and list of stored content IDs.
        """
        import re

        refs: list[str] = []
        parts = re.split(r"\n\s*\n", text)
        result_parts: list[str] = []

        for part in parts:
            if len(part) >= min_length:
                content_id = await self.store(part)
                refs.append(content_id)
                # Replace with compact reference
                preview = part[:50].replace("\n", " ") + "..."
                result_parts.append(f"[context_ref:{content_id}] {preview}")
            else:
                result_parts.append(part)

        return "\n\n".join(result_parts), refs

    async def resolve_refs(self, text: str) -> str:
        """Resolve context references back to full content."""
        import re

        pattern = re.compile(r"\[context_ref:([a-f0-9]+)\][^\n]*")

        async def _replace(match: re.Match[str]) -> str:
            content_id = match.group(1)
            content = await self.retrieve(content_id)
            return content if content else match.group(0)

        # Since re.sub doesn't support async, collect and replace
        result = text
        for match in pattern.finditer(text):
            content_id = match.group(1)
            content = await self.retrieve(content_id)
            if content:
                result = result.replace(match.group(0), content)

        return result

    async def stats(self) -> dict[str, Any]:
        await self._ensure_init()
        if self.backend == "sqlite":
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                count = await (await db.execute("SELECT COUNT(*), SUM(token_count), SUM(access_count) FROM context_store")).fetchone()
                return {
                    "stored_items": count[0] if count else 0,
                    "total_tokens": count[1] if count and count[1] else 0,
                    "total_accesses": count[2] if count and count[2] else 0,
                }
        return {}

    def _content_hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
