"""Request routing — resolve upstream targets for each request."""

from __future__ import annotations

import fnmatch
import logging
import os
from typing import Any

from twotrim.config import get_config
from twotrim.types import UpstreamTarget

logger = logging.getLogger(__name__)


class Router:
    """Route requests to appropriate upstream LLM endpoints."""

    def __init__(self) -> None:
        self._routes: dict[str, str] = {}
        self._default_base_url: str = ""
        self._initialized = False

    def _ensure_init(self) -> None:
        if self._initialized:
            return
        cfg = get_config().upstream
        self._default_base_url = cfg.default_base_url
        self._routes = cfg.routes
        self._initialized = True

    def resolve(self, model: str, api_key: str | None = None) -> UpstreamTarget:
        """Resolve the upstream target for a given model."""
        self._ensure_init()

        # Check model-specific routes
        for pattern, base_url in self._routes.items():
            if fnmatch.fnmatch(model, pattern):
                return UpstreamTarget(
                    base_url=base_url,
                    api_key=api_key or self._get_api_key(base_url),
                )

        # Default route
        return UpstreamTarget(
            base_url=self._default_base_url,
            api_key=api_key or self._get_api_key(self._default_base_url),
        )

    def _get_api_key(self, base_url: str) -> str | None:
        """Resolve API key from environment based on provider."""
        if "openai.com" in base_url:
            return os.environ.get("OPENAI_API_KEY")
        if "anthropic.com" in base_url:
            return os.environ.get("ANTHROPIC_API_KEY")
        if "googleapis.com" in base_url:
            return os.environ.get("GOOGLE_API_KEY")
        # Local endpoints typically don't need keys
        if any(h in base_url for h in ["localhost", "127.0.0.1", "0.0.0.0"]):
            return None
        return os.environ.get("LLM_API_KEY")

    def add_route(self, pattern: str, base_url: str) -> None:
        """Add a routing rule at runtime."""
        self._routes[pattern] = base_url

    def list_routes(self) -> dict[str, str]:
        """List all configured routes."""
        self._ensure_init()
        return {"default": self._default_base_url, **self._routes}


_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router
