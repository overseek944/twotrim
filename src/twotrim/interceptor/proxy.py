"""FastAPI proxy server — OpenAI-compatible LLM interceptor.

This is the main entry point for the TwoTrim proxy. All
LLM requests pass through here for compression, caching,
evaluation, and forwarding to upstream providers.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import version as _get_version
from typing import Any

_PKG_VERSION = _get_version("twotrim")

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from twotrim.config import get_config, load_config
from twotrim.interceptor.middleware import CompressionMiddleware, get_middleware
from twotrim.interceptor.router import Router, get_router
from twotrim.interceptor.streaming import is_streaming_request, stream_response
from twotrim.observability.logger import setup_logging
from twotrim.observability.metrics import get_metrics_collector
from twotrim.types import CompressionMode

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    cfg = get_config()
    setup_logging(
        level=cfg.observability.logging.level,
        fmt=cfg.observability.logging.format,
        log_file=cfg.observability.logging.file,
    )
    logger.info("TwoTrim proxy starting on %s:%d", cfg.server.host, cfg.server.port)
    logger.info("Default upstream: %s", cfg.upstream.default_base_url)
    logger.info("Compression mode: %s", cfg.compression.mode)

    yield

    logger.info("TwoTrim proxy shutting down")


def create_app(config_path: str | None = None) -> FastAPI:
    """Create the FastAPI application."""
    if config_path:
        load_config(config_path)

    cfg = get_config()

    app = FastAPI(
        title="TwoTrim",
        description="Universal Token Compression Fabric for LLM Applications",
        version=_PKG_VERSION,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -----------------------------------------------------------------------
    # Health + Info
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": _PKG_VERSION}

    @app.get("/v1/models")
    async def list_models():
        """Proxy models endpoint to upstream."""
        router = get_router()
        target = router.resolve("default")
        try:
            async with httpx.AsyncClient() as client:
                headers = {}
                if target.api_key:
                    headers["Authorization"] = f"Bearer {target.api_key}"
                resp = await client.get(
                    f"{target.base_url}/models",
                    headers=headers,
                    timeout=30,
                )
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception as e:
            return JSONResponse(
                content={"error": str(e)},
                status_code=502,
            )

    # -----------------------------------------------------------------------
    # Chat Completions
    # -----------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        x_twotrim_mode: str | None = Header(None, alias="X-TwoTrim-Mode"),
    ):
        """OpenAI-compatible chat completions endpoint."""
        return await _handle_proxy_request(
            request, "chat/completions", x_twotrim_mode
        )

    # -----------------------------------------------------------------------
    # Legacy Completions
    # -----------------------------------------------------------------------

    @app.post("/v1/completions")
    async def completions(
        request: Request,
        x_twotrim_mode: str | None = Header(None, alias="X-TwoTrim-Mode"),
    ):
        return await _handle_proxy_request(
            request, "completions", x_twotrim_mode
        )

    # -----------------------------------------------------------------------
    # Embeddings (pass-through, no compression)
    # -----------------------------------------------------------------------

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        """Pass-through embeddings — no compression applied."""
        body = await request.json()
        model = body.get("model", "unknown")
        router = get_router()
        target = router.resolve(model, _extract_api_key(request))

        try:
            async with httpx.AsyncClient() as client:
                headers = _build_upstream_headers(target.api_key)
                resp = await client.post(
                    f"{target.base_url}/embeddings",
                    json=body,
                    headers=headers,
                    timeout=target.timeout_seconds,
                )
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception as e:
            return _error_response(str(e), 502)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        try:
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            from starlette.responses import Response
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )
        except ImportError:
            return JSONResponse(
                content=get_metrics_collector().get_aggregate().model_dump(),
            )

    @app.get("/stats")
    async def stats():
        """Get aggregate statistics."""
        collector = get_metrics_collector()
        return JSONResponse(content=collector.get_aggregate().model_dump())

    @app.get("/stats/recent")
    async def recent_stats(n: int = 50):
        """Get recent request metrics."""
        collector = get_metrics_collector()
        recent = collector.get_recent(n)
        return JSONResponse(content=[m.model_dump() for m in recent])

    return app


# ---------------------------------------------------------------------------
# Core proxy handler
# ---------------------------------------------------------------------------

async def _handle_proxy_request(
    request: Request,
    endpoint: str,
    mode_header: str | None,
) -> JSONResponse | StreamingResponse:
    """Core request handling: compress → forward → return."""
    request_id = uuid.uuid4().hex[:16]
    start = time.perf_counter()

    try:
        body = await request.json()
    except Exception:
        return _error_response("Invalid JSON body", 400)

    model = body.get("model", "unknown")
    streaming = is_streaming_request(body)

    # Parse compression mode override
    override_mode: CompressionMode | None = None
    if mode_header:
        try:
            override_mode = CompressionMode(mode_header.lower())
        except ValueError:
            pass

    # Run through compression middleware
    middleware = get_middleware()
    compressed_body, meta = await middleware.process_request(body, override_mode)

    # Check for cached response
    if meta.get("cache_hit") and meta.get("cached_response"):
        logger.info(
            "[%s] Cache hit (similarity=%.3f) for model=%s",
            request_id, meta.get("cache_similarity", 0), model,
        )
        response = meta["cached_response"]
        response["_twotrim"] = {
            "request_id": request_id,
            "cache_hit": True,
            "cache_similarity": meta.get("cache_similarity"),
        }
        return JSONResponse(content=response)

    # Forward to upstream
    router = get_router()
    api_key = _extract_api_key(request)
    target = router.resolve(model, api_key)
    headers = _build_upstream_headers(target.api_key)

    try:
        if streaming:
            return await _handle_streaming(
                target.base_url, endpoint, compressed_body,
                headers, target.timeout_seconds, request_id, meta,
            )
        else:
            return await _handle_non_streaming(
                target.base_url, endpoint, compressed_body,
                headers, target.timeout_seconds, request_id, meta,
                middleware, start,
            )
    except httpx.TimeoutException:
        return _error_response("Upstream timeout", 504)
    except httpx.ConnectError:
        return _error_response(f"Cannot connect to upstream: {target.base_url}", 502)
    except Exception as e:
        logger.exception("Proxy error for request %s", request_id)
        return _error_response(str(e), 502)


async def _handle_streaming(
    base_url: str,
    endpoint: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    request_id: str,
    meta: dict[str, Any],
) -> StreamingResponse:
    """Handle streaming proxy request."""
    client = httpx.AsyncClient(timeout=timeout)

    async def _stream():
        try:
            async with client.stream(
                "POST",
                f"{base_url}/{endpoint}",
                json=body,
                headers=headers,
            ) as resp:
                async for chunk in stream_response(resp, request_id, meta.get("compression_result")):
                    yield chunk
        finally:
            await client.aclose()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-TwoTrim-Request-Id": request_id,
        },
    )


async def _handle_non_streaming(
    base_url: str,
    endpoint: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    request_id: str,
    meta: dict[str, Any],
    middleware: CompressionMiddleware,
    start_time: float,
) -> JSONResponse:
    """Handle non-streaming proxy request."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{base_url}/{endpoint}",
            json=body,
            headers=headers,
            timeout=timeout,
        )

    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    response = resp.json()

    # Post-process response
    response = await middleware.process_response(response)

    # Add TwoTrim metadata
    elapsed = (time.perf_counter() - start_time) * 1000
    response["_twotrim"] = {
        "request_id": request_id,
        "cache_hit": meta.get("cache_hit", False),
        "total_time_ms": round(elapsed, 2),
        **(meta.get("compression_result", {})),
    }

    logger.info(
        "[%s] model=%s ratio=%.1f%% time=%.0fms strategies=%s",
        request_id,
        body.get("model", "?"),
        meta.get("compression_result", {}).get("ratio", 0) * 100,
        elapsed,
        meta.get("compression_result", {}).get("strategies", []),
    )

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_api_key(request: Request) -> str | None:
    """Extract API key from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def _build_upstream_headers(api_key: str | None) -> dict[str, str]:
    """Build headers for upstream request."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _error_response(message: str, status: int) -> JSONResponse:
    """Create an error response."""
    return JSONResponse(
        content={"error": {"message": message, "type": "proxy_error"}},
        status_code=status,
    )
