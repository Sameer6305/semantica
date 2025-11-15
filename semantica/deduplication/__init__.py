"""
Advanced Deduplication Module

This module provides comprehensive semantic entity deduplication and merging
capabilities for the Semantica framework, helping keep knowledge graphs clean
and maintain a single source of truth. It supports multiple similarity calculation
methods, duplicate detection algorithms, entity merging strategies, and clustering
approaches for efficient batch processing.

Key Features:
    - Multiple similarity calculation methods (exact, Levenshtein, Jaro-Winkler, cosine, embedding)
    - Duplicate detection using similarity metrics and confidence scoring
    - Entity merging with configurable strategies (keep_first, keep_most_complete, etc.)
    - Cluster-based batch deduplication for large datasets
    - Provenance preservation during merges
    - Method registry for extensible deduplication methods
    - Reusable method functions for common deduplication tasks

Algorithms Used:

Similarity Calculation:
    - Levenshtein Distance: Dynamic programming algorithm for edit distance calculation
    - Jaro Similarity: Character-based similarity with match window algorithm
    - Jaro-Winkler Similarity: Jaro with prefix bonus (up to 4 characters)
    - Cosine Similarity: Vector dot product divided by magnitudes for embeddings
    - Jaccard Similarity: Intersection over union for relationship sets
    - Property Matching: Weighted comparison of property values
    - Multi-factor Aggregation: Weighted sum of similarity components

Duplicate Detection:
    - Pairwise Comparison: O(n²) all-pairs similarity calculation
    - Batch Processing: Vectorized similarity calculations for efficiency
    - Union-Find Algorithm: Disjoint set union for duplicate group formation
    - Confidence Scoring: Multi-factor confidence calculation (similarity + name match + property matches)
    - Incremental Processing: O(n×m) efficient new vs existing entity comparison

Clustering:
    - Union-Find (Disjoint Set Union): Connected component detection for graph-based clustering
    - Hierarchical Clustering: Agglomerative bottom-up clustering for large datasets
    - Similarity Graph: Graph construction from similarity scores
    - Cluster Quality Metrics: Cohesion and separation measures

Entity Merging:
    - Strategy Pattern: Multiple merge strategies (keep_first, keep_last, keep_most_complete, etc.)
    - Conflict Resolution: Voting, credibility-weighted, temporal, confidence-based resolution
    - Property Merging: Rule-based property combination with custom rules
    - Relationship Preservation: Union of relationship sets during merges
    - Provenance Tracking: Metadata preservation during merges

Main Components:
    - DuplicateDetector: Detects duplicate entities and relationships using similarity metrics
    - EntityMerger: Merges duplicate entities using configurable strategies
    - SimilarityCalculator: Calculates multi-factor similarity between entities
    - MergeStrategyManager: Manages merge strategies and conflict resolution
    - ClusterBuilder: Builds clusters for batch deduplication
    - MethodRegistry: Registry for custom deduplication methods
    - Deduplication Methods: Reusable functions for common deduplication tasks

Example Usage:
    >>> from semantica.deduplication import DuplicateDetector, EntityMerger, deduplicate
    >>> # Using main classes
    >>> detector = DuplicateDetector(similarity_threshold=0.8)
    >>> duplicates = detector.detect_duplicates(entities)
    >>> merger = EntityMerger()
    >>> merged = merger.merge_duplicates(entities)
    >>> 
    >>> # Using convenience function
    >>> from semantica.deduplication import deduplicate
    >>> result = deduplicate(entities, similarity_threshold=0.8, merge_strategy="keep_most_complete")
    >>> 
    >>> # Using methods directly
    >>> from semantica.deduplication.methods import detect_duplicates, calculate_similarity
    >>> duplicates = detect_duplicates(entities, method="pairwise", similarity_threshold=0.8)
    >>> similarity = calculate_similarity(entity1, entity2, method="levenshtein")

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union

from .entity_merger import EntityMerger, MergeOperation
from .similarity_calculator import SimilarityCalculator, SimilarityResult
from .duplicate_detector import DuplicateDetector, DuplicateCandidate, DuplicateGroup
from .merge_strategy import MergeStrategyManager, MergeStrategy, MergeResult, PropertyMergeRule
from .cluster_builder import ClusterBuilder, Cluster, ClusterResult
from .registry import MethodRegistry, method_registry
from .methods import (
    detect_duplicates,
    merge_entities,
    calculate_similarity,
    build_clusters,
    get_deduplication_method,
    list_available_methods
)
from .config import DeduplicationConfig, dedup_config

__all__ = [
    # Main classes
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
    
    # Registry
    "MethodRegistry",
    "method_registry",
    
    # Methods
    "detect_duplicates",
    "merge_entities",
    "calculate_similarity",
    "build_clusters",
    "get_deduplication_method",
    "list_available_methods",
    
    # Config
    "DeduplicationConfig",
    "dedup_config",
    
    # Convenience
    "deduplicate",
]


def deduplicate(
    entities: List[Dict[str, Any]],
    similarity_threshold: float = 0.7,
    confidence_threshold: float = 0.6,
    merge_strategy: str = "keep_most_complete",
    preserve_provenance: bool = True,
    **options
) -> Dict[str, Any]:
    """
    Deduplicate entities (module-level convenience function).
    
    This is a user-friendly wrapper that performs comprehensive deduplication
    including duplicate detection and entity merging.
    
    Args:
        entities: List of entity dictionaries to deduplicate
        similarity_threshold: Minimum similarity score to consider duplicates (default: 0.7)
        confidence_threshold: Minimum confidence score for duplicate candidates (default: 0.6)
        merge_strategy: Merge strategy to use (default: "keep_most_complete")
            - "keep_first": Preserve first entity, merge others
            - "keep_last": Preserve last entity, merge others
            - "keep_most_complete": Preserve entity with most properties/relationships
            - "keep_highest_confidence": Preserve entity with highest confidence
            - "merge_all": Combine all properties and relationships
        preserve_provenance: Whether to preserve provenance information (default: True)
        **options: Additional deduplication options
        
    Returns:
        Dictionary containing:
            - merged_entities: List of merged entities
            - duplicate_groups: List of duplicate groups found
            - merge_operations: List of merge operations performed
            - statistics: Deduplication statistics
            
    Examples:
        >>> import semantica
        >>> entities = [
        ...     {"id": "1", "name": "Apple Inc.", "type": "Company"},
        ...     {"id": "2", "name": "Apple", "type": "Company"},
        ...     {"id": "3", "name": "Microsoft", "type": "Company"}
        ... ]
        >>> result = semantica.deduplication.deduplicate(
        ...     entities,
        ...     similarity_threshold=0.8,
        ...     merge_strategy="keep_most_complete"
        ... )
        >>> print(f"Merged {len(result['merged_entities'])} entities")
    """
    # Detect duplicates
    detector = DuplicateDetector(
        similarity_threshold=similarity_threshold,
        confidence_threshold=confidence_threshold,
        **options.get("detection_config", {})
    )
    
    duplicate_groups = detector.detect_duplicate_groups(entities, **options)
    
    # Merge duplicates
    merger = EntityMerger(
        preserve_provenance=preserve_provenance,
        **options.get("merger_config", {})
    )
    
    # Map strategy string to enum
    strategy_map = {
        "keep_first": MergeStrategy.KEEP_FIRST,
        "keep_last": MergeStrategy.KEEP_LAST,
        "keep_most_complete": MergeStrategy.KEEP_MOST_COMPLETE,
        "keep_highest_confidence": MergeStrategy.KEEP_HIGHEST_CONFIDENCE,
        "merge_all": MergeStrategy.MERGE_ALL,
    }
    
    strategy = strategy_map.get(merge_strategy, MergeStrategy.KEEP_MOST_COMPLETE)
    
    merge_operations = merger.merge_duplicates(
        entities,
        strategy=strategy,
        **options
    )
    
    # Collect merged entities
    merged_entities = [op.merged_entity for op in merge_operations]
    
    # Collect non-duplicate entities
    duplicate_entity_ids = set()
    for group in duplicate_groups:
        for entity in group.entities:
            entity_id = entity.get("id") or id(entity)
            duplicate_entity_ids.add(entity_id)
    
    non_duplicate_entities = [
        e for e in entities
        if (e.get("id") or id(e)) not in duplicate_entity_ids
    ]
    
    # Combine merged and non-duplicate entities
    all_merged = merged_entities + non_duplicate_entities
    
    # Calculate statistics
    statistics = {
        "total_entities": len(entities),
        "duplicate_groups": len(duplicate_groups),
        "merged_entities": len(merged_entities),
        "non_duplicate_entities": len(non_duplicate_entities),
        "final_entities": len(all_merged),
        "reduction": len(entities) - len(all_merged),
    }
    
    return {
        "merged_entities": all_merged,
        "duplicate_groups": duplicate_groups,
        "merge_operations": merge_operations,
        "statistics": statistics,
    }
