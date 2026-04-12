"""Plugin registry — discover, load, and manage plugins."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from twotrim.plugins.base import CachePlugin, CompressionPlugin, EvaluationPlugin
from twotrim.types import PluginInfo

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for TwoTrim plugins."""

    def __init__(self) -> None:
        self._compression: dict[str, CompressionPlugin] = {}
        self._evaluation: dict[str, EvaluationPlugin] = {}
        self._cache: dict[str, CachePlugin] = {}

    def register_compression(self, plugin: CompressionPlugin) -> None:
        self._compression[plugin.name] = plugin
        logger.info("Registered compression plugin: %s", plugin.name)

    def register_evaluation(self, plugin: EvaluationPlugin) -> None:
        self._evaluation[plugin.name] = plugin
        logger.info("Registered evaluation plugin: %s", plugin.name)

    def register_cache(self, plugin: CachePlugin) -> None:
        self._cache[plugin.name] = plugin
        logger.info("Registered cache plugin: %s", plugin.name)

    def get_compression(self, name: str) -> CompressionPlugin | None:
        return self._compression.get(name)

    def get_evaluation(self, name: str) -> EvaluationPlugin | None:
        return self._evaluation.get(name)

    def get_cache(self, name: str) -> CachePlugin | None:
        return self._cache.get(name)

    def list_plugins(self) -> list[PluginInfo]:
        plugins: list[PluginInfo] = []
        for name, p in self._compression.items():
            plugins.append(PluginInfo(
                name=name, version=p.version, description=p.description,
                plugin_type="compression",
            ))
        for name, p in self._evaluation.items():
            plugins.append(PluginInfo(
                name=name, version=getattr(p, "version", "0.1.0"),
                description=getattr(p, "description", ""),
                plugin_type="evaluation",
            ))
        for name, p in self._cache.items():
            plugins.append(PluginInfo(
                name=name, version=getattr(p, "version", "0.1.0"),
                description=getattr(p, "description", ""),
                plugin_type="cache",
            ))
        return plugins

    def load_from_directory(self, directory: str) -> None:
        """Load plugins from a directory containing Python files."""
        p = Path(directory)
        if not p.exists():
            logger.warning("Plugin directory not found: %s", directory)
            return

        for py_file in p.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"twotrim_plugin_{py_file.stem}", py_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Auto-register plugins
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type):
                            if issubclass(attr, CompressionPlugin) and attr is not CompressionPlugin:
                                self.register_compression(attr())
                            elif issubclass(attr, EvaluationPlugin) and attr is not EvaluationPlugin:
                                self.register_evaluation(attr())
                            elif issubclass(attr, CachePlugin) and attr is not CachePlugin:
                                self.register_cache(attr())

                    logger.info("Loaded plugin module: %s", py_file.name)
            except Exception as e:
                logger.error("Failed to load plugin %s: %s", py_file, e)

    def load_from_entrypoints(self, group: str = "twotrim.plugins") -> None:
        """Load plugins from setuptools entry points."""
        try:
            from importlib.metadata import entry_points
            eps = entry_points()
            plugin_eps = eps.get(group, []) if isinstance(eps, dict) else eps.select(group=group)

            for ep in plugin_eps:
                try:
                    plugin_class = ep.load()
                    plugin = plugin_class()
                    if isinstance(plugin, CompressionPlugin):
                        self.register_compression(plugin)
                    elif isinstance(plugin, EvaluationPlugin):
                        self.register_evaluation(plugin)
                    elif isinstance(plugin, CachePlugin):
                        self.register_cache(plugin)
                except Exception as e:
                    logger.error("Failed to load entry point %s: %s", ep.name, e)
        except Exception as e:
            logger.debug("Entry point loading not available: %s", e)


# Module singleton
_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry
