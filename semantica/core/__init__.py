"""
Core Orchestration Module

This module provides comprehensive orchestration capabilities for the Semantica framework,
enabling framework initialization, knowledge base construction, pipeline execution, configuration
management, lifecycle management, and plugin system integration.

Key Features:
    - Framework initialization and lifecycle management
    - Knowledge base construction from various data sources
    - Pipeline execution and resource management
    - Configuration loading, validation, and management
    - Plugin discovery, loading, and lifecycle management
    - System health monitoring and status tracking
    - Method registry for extensible orchestration methods

Algorithms Used:
    - Configuration Management: YAML/JSON parsing, environment variable resolution, validation
    - Lifecycle Management: Priority-based hook execution, state machine transitions
    - Plugin Management: Dynamic module loading, dependency resolution, version management
    - Resource Management: Dynamic allocation, cleanup, graceful shutdown
    - Health Monitoring: Component health checks, status aggregation, error tracking

Main Components:
    - Semantica: Main framework class for orchestration and knowledge base building
    - Config: Configuration data class with validation
    - ConfigManager: Configuration loading, validation, and management
    - LifecycleManager: System lifecycle management with hooks and health monitoring
    - PluginRegistry: Dynamic plugin discovery, loading, and management
    - MethodRegistry: Registry for custom orchestration methods
    - Orchestration Methods: Reusable functions for common orchestration tasks

Example Usage:
    >>> from semantica.core import Semantica, build
    >>> # Using main class
    >>> framework = Semantica()
    >>> framework.initialize()
    >>> result = framework.build_knowledge_base(sources=["doc1.pdf"])
    >>> 
    >>> # Using convenience function
    >>> from semantica.core import build
    >>> result = build(sources=["doc1.pdf", "doc2.docx"], embeddings=True, graph=True)
    >>> 
    >>> # Using methods directly
    >>> from semantica.core.methods import build_knowledge_base
    >>> result = build_knowledge_base(sources=["doc.pdf"], method="default")

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .orchestrator import Semantica
from .config_manager import Config, ConfigManager
from .lifecycle import LifecycleManager, SystemState, HealthStatus
from .plugin_registry import PluginRegistry, PluginInfo, LoadedPlugin
from .registry import MethodRegistry, method_registry
from .methods import (
    build_knowledge_base,
    run_pipeline,
    initialize_framework,
    get_status,
    get_orchestration_method,
    list_available_methods
)

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
    
    # Registry
    "MethodRegistry",
    "method_registry",
    
    # Methods
    "build_knowledge_base",
    "run_pipeline",
    "initialize_framework",
    "get_status",
    "get_orchestration_method",
    "list_available_methods",
    
    # Convenience
    "build",
]


def build(
    sources: Union[str, List[Union[str, Path]]],
    extract_entities: bool = True,
    extract_relations: bool = True,
    embeddings: bool = True,
    graph: bool = True,
    **options
) -> Dict[str, Any]:
    """
    Build knowledge base from sources (module-level convenience function).
    
    This is a user-friendly wrapper that performs comprehensive knowledge base
    construction including entity extraction, relation extraction, embeddings,
    and knowledge graph building.
    
    Args:
        sources: Input source or list of sources (files, URLs, streams)
        extract_entities: Whether to extract named entities (default: True)
        extract_relations: Whether to extract relationships (default: True)
        embeddings: Whether to generate embeddings (default: True)
        graph: Whether to build knowledge graph (default: True)
        **options: Additional processing options
        
    Returns:
        Dictionary containing:
            - knowledge_graph: Knowledge graph data
            - embeddings: Embedding vectors
            - results: Processing results
            - statistics: Processing statistics
            - metadata: Processing metadata
            
    Examples:
        >>> import semantica
        >>> result = semantica.core.build(
        ...     sources=["doc1.pdf", "doc2.docx"],
        ...     extract_entities=True,
        ...     extract_relations=True,
        ...     embeddings=True,
        ...     graph=True
        ... )
        >>> print(f"Processed {result['statistics']['sources_processed']} sources")
    """
    # Normalize sources to list
    if isinstance(sources, str):
        sources = [sources]
    
    # Build pipeline configuration from options
    pipeline_config = options.get("pipeline", {})
    if extract_entities or extract_relations:
        pipeline_config.setdefault("extract", {})
        if extract_entities:
            pipeline_config["extract"]["entities"] = True
        if extract_relations:
            pipeline_config["extract"]["relations"] = True
    
    # Build knowledge base
    return build_knowledge_base(
        sources=sources,
        method=options.get("method", "default"),
        embeddings=embeddings,
        graph=graph,
        pipeline=pipeline_config,
        **{k: v for k, v in options.items() if k not in ["pipeline", "method"]}
    )