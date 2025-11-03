"""
Core Orchestration Module

This module provides the main orchestration capabilities for the Semantica framework.

Exports:
    - Semantica: Main framework class for orchestration
    - Config: Configuration data class
    - ConfigManager: Configuration management system
    - LifecycleManager: Lifecycle management
    - PluginRegistry: Plugin management system
"""

from .orchestrator import Semantica
from .config_manager import Config, ConfigManager
from .lifecycle import LifecycleManager, SystemState, HealthStatus
from .plugin_registry import PluginRegistry, PluginInfo, LoadedPlugin

__all__ = [
    # Main orchestrator
    "Semantica",
    # Configuration
    "Config",
    "ConfigManager",
    # Lifecycle
    "LifecycleManager",
    "SystemState",
    "HealthStatus",
    # Plugins
    "PluginRegistry",
    "PluginInfo",
    "LoadedPlugin",
]