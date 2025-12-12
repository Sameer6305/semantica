"""
Semantica - Semantic Layer & Knowledge Engineering Framework

A comprehensive Python framework for transforming unstructured data into 
semantic layers, knowledge graphs, and embeddings.

Main exports:
    - Semantica: Main framework class
    - PipelineBuilder: Pipeline construction DSL
    - Config: Configuration management
"""

__version__ = "0.0.5"
__author__ = "Semantica Contributors"
__license__ = "MIT"

# Import submodules for dot notation access
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Core imports
from .core import Config, ConfigManager, LifecycleManager, PluginRegistry, Semantica


# Pipeline imports
from .pipeline import (
    ExecutionEngine,
    FailureHandler,
    ParallelismManager,
    PipelineBuilder,
    PipelineValidator,
    ResourceScheduler,
)

# Visualization
from .visualization import (
    AnalyticsVisualizer,
    EmbeddingVisualizer,
    KGVisualizer,
    OntologyVisualizer,
    QualityVisualizer,
    SemanticNetworkVisualizer,
    TemporalVisualizer,
)


# Module proxy class for submodule access
class _ModuleProxy:
    """Proxy class to enable dot notation access to submodules."""

    def __init__(self, module_name: str):
        self._module_name = module_name
        self._module = None

    def _get_module(self):
        """Lazy load the module."""
        if self._module is None:
            self._module = importlib.import_module(f"semantica.{self._module_name}")
        return self._module

    def __getattr__(self, name: str):
        """Delegate attribute access to the actual module."""
        return getattr(self._get_module(), name)

    def __dir__(self):
        """Return directory of the actual module."""
        return dir(self._get_module())


# Create module proxies for submodule access
class _SemanticaModules:
    """Container for submodule proxies."""

    def __init__(self):
        self._kg = None
        self._ingest = None
        self._embeddings = None
        self._semantic_extract = None
        self._visualization = None
        self._kg_qa = None
        self._pipeline = None
        self._parse = None
        self._normalize = None
        self._export = None
        self._vector_store = None
        self._triplet_store = None
        self._graph_store = None
        self._ontology = None
        self._evals = None

    @property
    def kg(self):
        """Access knowledge graph module."""
        if self._kg is None:
            self._kg = _ModuleProxy("kg")
        return self._kg

    @property
    def ingest(self):
        """Access ingestion module."""
        if self._ingest is None:
            self._ingest = _ModuleProxy("ingest")
        return self._ingest

    @property
    def embeddings(self):
        """Access embeddings module."""
        if self._embeddings is None:
            self._embeddings = _ModuleProxy("embeddings")
        return self._embeddings

    @property
    def semantic_extract(self):
        """Access semantic extraction module."""
        if self._semantic_extract is None:
            self._semantic_extract = _ModuleProxy("semantic_extract")
        return self._semantic_extract

    @property
    def visualization(self):
        """Access visualization module."""
        if self._visualization is None:
            self._visualization = _ModuleProxy("visualization")
        return self._visualization

    @property
    def pipeline(self):
        """Access pipeline module."""
        if self._pipeline is None:
            self._pipeline = _ModuleProxy("pipeline")
        return self._pipeline

    @property
    def parse(self):
        """Access parsing module."""
        if self._parse is None:
            self._parse = _ModuleProxy("parse")
        return self._parse

    @property
    def normalize(self):
        """Access normalization module."""
        if self._normalize is None:
            self._normalize = _ModuleProxy("normalize")
        return self._normalize

    @property
    def export(self):
        """Access export module."""
        if self._export is None:
            self._export = _ModuleProxy("export")
        return self._export

    @property
    def vector_store(self):
        """Access vector store module."""
        if self._vector_store is None:
            self._vector_store = _ModuleProxy("vector_store")
        return self._vector_store

    @property
    def triplet_store(self):
        """Access triplet store module."""
        if self._triplet_store is None:
            self._triplet_store = _ModuleProxy("triplet_store")
        return self._triplet_store

    @property
    def graph_store(self):
        """Access graph store module."""
        if self._graph_store is None:
            self._graph_store = _ModuleProxy("graph_store")
        return self._graph_store

    @property
    def ontology(self):
        """Access ontology module."""
        if self._ontology is None:
            self._ontology = _ModuleProxy("ontology")
        return self._ontology

    @property
    def evals(self):
        """Access evaluation module."""
        if self._evals is None:
            self._evals = _ModuleProxy("evals")
        return self._evals


# Create singleton instance for module access
_modules = _SemanticaModules()

# Singleton Semantica instance for module-level build()
_semantica_instance: Optional[Semantica] = None


def _get_semantica_instance(
    config: Optional[Union[Config, Dict[str, Any]]] = None, **kwargs
) -> Semantica:
    """Get or create singleton Semantica instance."""
    global _semantica_instance
    if _semantica_instance is None:
        _semantica_instance = Semantica(config=config, **kwargs)
    return _semantica_instance


def build(
    sources: Union[List[Union[str, Path]], str, Path],
    config: Optional[Union[Config, Dict[str, Any]]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build knowledge base from data sources (module-level convenience function).

    This is a user-friendly wrapper around Semantica.build_knowledge_base()
    that handles initialization automatically.

    Args:
        sources: Data source(s) - can be a single path/URL or list of paths/URLs
        config: Optional configuration object or dict
        **kwargs: Additional processing options:
            - embeddings: Whether to generate embeddings (default: True)
            - graph: Whether to build knowledge graph (default: True)
            - normalize: Whether to normalize data (default: True)
            - pipeline: Custom pipeline configuration
            - fail_fast: Whether to fail on first error (default: False)

    Returns:
        Dictionary containing:
            - knowledge_graph: Knowledge graph data
            - embeddings: Embedding vectors
            - metadata: Processing metadata
            - statistics: Processing statistics
            - results: Processing results

    Examples:
        >>> from semantica import build
        >>> result = build(["doc1.pdf", "doc2.docx"], embeddings=True, graph=True)
        >>> print(result["statistics"])
    """
    # Normalize sources to list
    if isinstance(sources, (str, Path)):
        sources = [sources]

    # Get or create Semantica instance
    semantica = _get_semantica_instance(config=config, **kwargs)

    # Build knowledge base (auto-initializes if needed)
    return semantica.build_knowledge_base(sources, **kwargs)


__all__ = [
    # Core
    "Semantica",
    "Config",
    "ConfigManager",
    "LifecycleManager",
    "PluginRegistry",
    # Pipeline
    "PipelineBuilder",
    "ExecutionEngine",
    "FailureHandler",
    "ParallelismManager",
    "ResourceScheduler",
    "PipelineValidator",
    # Visualization
    "KGVisualizer",
    "OntologyVisualizer",
    "EmbeddingVisualizer",
    "SemanticNetworkVisualizer",
    "QualityVisualizer",
    "AnalyticsVisualizer",
    "TemporalVisualizer",
]


# Make submodules accessible via dot notation
def __getattr__(name: str):
    """Enable dot notation access to submodules."""
    if name in [
        "kg",
        "ingest",
        "embeddings",
        "semantic_extract",
        "visualization",
        "pipeline",
        "parse",
        "normalize",
        "export",
        "vector_store",
        "triplet_store",
        "graph_store",
        "ontology",
        "evals",
    ]:
        return getattr(_modules, name)
    raise AttributeError(f"module 'semantica' has no attribute '{name}'")
