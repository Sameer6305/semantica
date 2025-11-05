"""
Semantica - Semantic Layer & Knowledge Engineering Framework

A comprehensive Python framework for transforming unstructured data into 
semantic layers, knowledge graphs, and embeddings.

Main exports:
    - Semantica: Main framework class
    - PipelineBuilder: Pipeline construction DSL
    - Config: Configuration management
"""

__version__ = "0.1.0"
__author__ = "Semantica Contributors"
__license__ = "MIT"

# Core imports
from .core import Semantica, Config, ConfigManager, LifecycleManager, PluginRegistry

# Pipeline imports
from .pipeline import (
    PipelineBuilder,
    ExecutionEngine,
    FailureHandler,
    ParallelismManager,
    ResourceScheduler,
    PipelineValidator,
)

# KG Quality Assurance
from .kg_qa import (
    KGQualityAssessor,
    ConsistencyChecker,
    CompletenessValidator,
    QualityMetrics,
    CompletenessMetrics,
    ConsistencyMetrics,
    ValidationEngine,
    RuleValidator,
    ConstraintValidator,
    QualityReporter,
    IssueTracker,
    ImprovementSuggestions,
    AutomatedFixer,
    AutoMerger,
    AutoResolver,
)

# Visualization
from .visualization import (
    KGVisualizer,
    OntologyVisualizer,
    EmbeddingVisualizer,
    SemanticNetworkVisualizer,
    QualityVisualizer,
    AnalyticsVisualizer,
    TemporalVisualizer,
)

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
    # KG Quality Assurance
    "KGQualityAssessor",
    "ConsistencyChecker",
    "CompletenessValidator",
    "QualityMetrics",
    "CompletenessMetrics",
    "ConsistencyMetrics",
    "ValidationEngine",
    "RuleValidator",
    "ConstraintValidator",
    "QualityReporter",
    "IssueTracker",
    "ImprovementSuggestions",
    "AutomatedFixer",
    "AutoMerger",
    "AutoResolver",
    # Visualization
    "KGVisualizer",
    "OntologyVisualizer",
    "EmbeddingVisualizer",
    "SemanticNetworkVisualizer",
    "QualityVisualizer",
    "AnalyticsVisualizer",
    "TemporalVisualizer",
]

