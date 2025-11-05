"""
Advanced Deduplication Module

This module provides semantic entity deduplication and merging
to keep knowledge graphs clean and maintain single source of truth.
"""

from .entity_merger import EntityMerger, MergeOperation
from .similarity_calculator import SimilarityCalculator, SimilarityResult
from .duplicate_detector import DuplicateDetector, DuplicateCandidate, DuplicateGroup
from .merge_strategy import MergeStrategyManager, MergeStrategy, MergeResult, PropertyMergeRule
from .cluster_builder import ClusterBuilder, Cluster, ClusterResult

__all__ = [
    "EntityMerger",
    "MergeOperation",
    "SimilarityCalculator",
    "SimilarityResult",
    "DuplicateDetector",
    "DuplicateCandidate",
    "DuplicateGroup",
    "MergeStrategyManager",
    "MergeStrategy",
    "MergeResult",
    "PropertyMergeRule",
    "ClusterBuilder",
    "Cluster",
    "ClusterResult",
]
