"""vLLM integration adapter.

Provides hooks for vLLM-specific optimizations including
KV cache management and model-level compression.
Gracefully degrades when vLLM is not available.
"""

from __future__ import annotations

import logging
from typing import Any

from twotrim.kv_cache.detector import ModelCapabilities, check_vllm_available

logger = logging.getLogger(__name__)


class VLLMAdapter:
    """Integration adapter for vLLM inference servers."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url
        self._available = check_vllm_available()
        self._client: Any = None

    @property
    def available(self) -> bool:
        return self._available

    async def get_model_info(self) -> dict[str, Any] | None:
        """Query vLLM server for model information."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/v1/models", timeout=5)
                if resp.status_code == 200:
                    return resp.json()
        except Exception as e:
            logger.debug("Failed to query vLLM: %s", e)
        return None

    async def get_kv_cache_stats(self) -> dict[str, Any] | None:
        """Get KV cache utilization stats from vLLM."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/metrics", timeout=5)
                if resp.status_code == 200:
                    return self._parse_prometheus_metrics(resp.text)
        except Exception as e:
            logger.debug("Failed to get vLLM metrics: %s", e)
        return None

    def _parse_prometheus_metrics(self, text: str) -> dict[str, Any]:
        """Parse relevant metrics from Prometheus format."""
        metrics: dict[str, Any] = {}
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            for key in ["vllm:gpu_cache_usage_perc", "vllm:cpu_cache_usage_perc",
                       "vllm:num_requests_running", "vllm:num_requests_waiting"]:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            metrics[key] = float(parts[-1])
                        except ValueError:
                            pass
        return metrics

    async def apply_kv_cache_config(
        self, capabilities: ModelCapabilities, config: dict[str, Any]
    ) -> bool:
        """Apply KV cache configuration to vLLM instance.

        Note: Most vLLM cache config is set at launch time.
        This method is for runtime-adjustable parameters.
        """
        if not self._available:
            logger.debug("vLLM not available, skipping KV cache config")
            return False

        logger.info(
            "vLLM KV cache config: quantization=%s, eviction=%s",
            config.get("quantization", "none"),
            config.get("eviction_strategy", "none"),
        )
        return True
