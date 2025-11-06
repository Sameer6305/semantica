"""
Advanced Deduplication Module

This module provides comprehensive semantic entity deduplication and merging
capabilities for the Semantica framework, helping keep knowledge graphs clean
and maintain a single source of truth.

Key Features:
    - Duplicate detection using similarity metrics
    - Entity merging with configurable strategies
    - Similarity calculation using multiple factors
    - Cluster-based batch deduplication
    - Provenance preservation during merges

Main Components:
    - DuplicateDetector: Detects duplicate entities and relationships
    - EntityMerger: Merges duplicate entities using strategies
    - SimilarityCalculator: Calculates multi-factor similarity
    - MergeStrategyManager: Manages merge strategies and conflict resolution
    - ClusterBuilder: Builds clusters for batch deduplication

Example Usage:
    >>> from semantica.deduplication import DuplicateDetector, EntityMerger
    >>> detector = DuplicateDetector(similarity_threshold=0.8)
    >>> duplicates = detector.detect_duplicates(entities)
    >>> merger = EntityMerger()
    >>> merged = merger.merge_duplicates(entities)

Author: Semantica Contributors
License: MIT
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
