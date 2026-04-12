"""Prompt cache — store compressed prompt forms to avoid recomputation.

Uses SQLite by default with optional Redis backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from twotrim.types import CompressionResult

logger = logging.getLogger(__name__)


class PromptCache:
    """Cache for compressed prompt forms keyed by content hash."""

    def __init__(
        self,
        backend: str = "sqlite",
        db_path: str = ".twotrim/prompt_cache.db",
        max_entries: int = 50000,
    ) -> None:
        self.backend = backend
        self.db_path = db_path
        self.max_entries = max_entries
        self._initialized = False

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        if self.backend == "sqlite":
            await self._init_sqlite()
        self._initialized = True

    async def _init_sqlite(self) -> None:
        import aiosqlite
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS prompt_cache (
                    hash TEXT PRIMARY KEY,
                    original_text TEXT,
                    compressed_text TEXT,
                    compression_result TEXT,
                    mode TEXT,
                    created_at REAL,
                    last_accessed REAL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_accessed ON prompt_cache(last_accessed)"
            )
            await db.commit()

    async def lookup(self, text: str, mode: str = "balanced") -> CompressionResult | None:
        """Look up a cached compression result."""
        await self._ensure_init()
        h = self._hash(text, mode)

        if self.backend == "sqlite":
            return await self._sqlite_lookup(h)
        return None

    async def store(self, text: str, result: CompressionResult, mode: str = "balanced") -> None:
        """Store a compression result."""
        await self._ensure_init()
        h = self._hash(text, mode)

        if self.backend == "sqlite":
            await self._sqlite_store(h, text, result, mode)

    async def _sqlite_lookup(self, h: str) -> CompressionResult | None:
        import aiosqlite
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    "SELECT * FROM prompt_cache WHERE hash = ?", (h,)
                )
                row = await cursor.fetchone()
                if row is None:
                    return None

                # Update access time
                await db.execute(
                    "UPDATE prompt_cache SET last_accessed = ?, hit_count = hit_count + 1 WHERE hash = ?",
                    (time.time(), h)
                )
                await db.commit()

                data = json.loads(row["compression_result"])
                return CompressionResult(**data)
        except Exception as e:
            logger.warning("Prompt cache lookup failed: %s", e)
            return None

    async def _sqlite_store(
        self, h: str, text: str, result: CompressionResult, mode: str
    ) -> None:
        import aiosqlite
        try:
            async with aiosqlite.connect(self.db_path) as db:
                now = time.time()
                await db.execute(
                    """INSERT OR REPLACE INTO prompt_cache
                       (hash, original_text, compressed_text, compression_result, mode, created_at, last_accessed, hit_count)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                    (h, text[:5000], result.compressed_text[:5000],
                     result.model_dump_json(), mode, now, now)
                )

                # Evict if over capacity
                count_cursor = await db.execute("SELECT COUNT(*) FROM prompt_cache")
                count_row = await count_cursor.fetchone()
                if count_row and count_row[0] > self.max_entries:
                    n_remove = count_row[0] - self.max_entries + (self.max_entries // 10)
                    await db.execute(
                        "DELETE FROM prompt_cache WHERE hash IN "
                        "(SELECT hash FROM prompt_cache ORDER BY last_accessed ASC LIMIT ?)",
                        (n_remove,)
                    )

                await db.commit()
        except Exception as e:
            logger.warning("Prompt cache store failed: %s", e)

    async def clear(self) -> None:
        """Clear the prompt cache."""
        await self._ensure_init()
        if self.backend == "sqlite":
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("DELETE FROM prompt_cache")
                await db.commit()

    async def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        await self._ensure_init()
        if self.backend == "sqlite":
            import aiosqlite
            async with aiosqlite.connect(self.db_path) as db:
                count = await (await db.execute("SELECT COUNT(*) FROM prompt_cache")).fetchone()
                hits = await (await db.execute("SELECT SUM(hit_count) FROM prompt_cache")).fetchone()
                return {
                    "entries": count[0] if count else 0,
                    "total_hits": hits[0] if hits and hits[0] else 0,
                }
        return {}

    def _hash(self, text: str, mode: str) -> str:
        return hashlib.sha256(f"{mode}:{text}".encode()).hexdigest()[:32]
