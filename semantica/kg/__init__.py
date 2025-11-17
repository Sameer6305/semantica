"""
Knowledge Graph Management Module

This module provides comprehensive knowledge graph construction and management capabilities
for the Semantica framework, including temporal knowledge graph support for time-aware
knowledge representation, graph analytics, entity resolution, conflict detection, and
provenance tracking.

Algorithms Used:

Knowledge Graph Construction:
    - Graph Building: Entity-relationship graph construction from multiple sources
    - Entity Resolution: Fuzzy string matching, exact matching, semantic similarity matching for duplicate detection
    - Conflict Detection: Value conflict detection (same entity with different property values), relationship conflict detection
    - Conflict Resolution: Highest confidence strategy, source-based resolution
    - Temporal Graph Support: Time-aware edge creation with valid_from/valid_until timestamps
    - Temporal Granularity: Time normalization (second, minute, hour, day, week, month, year)
    - Entity Merging: Property aggregation, metadata merging, provenance tracking
    - Incremental Building: Batch processing for large graphs

Graph Analysis:
    - Degree Centrality: Normalized degree calculation (degree / (n-1)) for connectivity measure
    - Betweenness Centrality: Shortest path counting (BFS-based), path normalization by (n-1)*(n-2)/2
    - Closeness Centrality: Average distance calculation, (n-1) / sum of distances normalization
    - Eigenvector Centrality: Power iteration method, adjacency matrix eigenvalue computation, convergence tolerance
    - Community Detection: Louvain algorithm (greedy modularity optimization), Leiden algorithm (with refinement step)
    - Overlapping Communities: K-clique community detection, dense subgraph detection
    - Modularity Calculation: Q = (1/2m) * Σ(A_ij - k_i*k_j/2m) * δ(c_i, c_j)
    - Graph Connectivity: DFS-based connected component detection, component size analysis
    - Bridge Detection: Edge removal and connectivity change detection
    - Path Finding: BFS shortest path algorithm, all-pairs shortest paths computation
    - Graph Density: E / (n*(n-1)/2) calculation for undirected graphs
    - Structure Classification: Density-based classification (sparse, moderate, dense, disconnected)

Entity Resolution:
    - Duplicate Detection: Similarity-based grouping using threshold matching
    - Fuzzy Matching: String similarity algorithms (Levenshtein, Jaro-Winkler)
    - Semantic Matching: Embedding-based similarity for semantic entity matching
    - Entity Merging: Property conflict resolution, metadata aggregation
    - ID Normalization: Canonical ID assignment for merged entities

Conflict Detection:
    - Value Conflict Detection: Property value comparison, unique value set extraction
    - Relationship Conflict Detection: Relationship property comparison, conflict identification
    - Source Tracking: Multi-source conflict tracking, provenance-based conflict resolution
    - Conflict Categorization: Value conflicts vs relationship conflicts

Graph Validation:
    - Entity Validation: Required field checking (ID, type), unique ID verification
    - Relationship Validation: Source/target reference validation, required field checking
    - Consistency Checking: Type consistency verification, circular relationship detection (DFS-based cycle detection)
    - Orphaned Entity Detection: Relationship-based entity connectivity checking
    - Validation Reporting: Error and warning categorization

Temporal Operations:
    - Time-Point Queries: Temporal filtering using valid_from/valid_until comparison
    - Time-Range Queries: Interval overlap detection, union/intersection aggregation
    - Temporal Pattern Detection: Sequence detection, cycle detection, trend analysis
    - Graph Evolution Analysis: Time-series relationship counting, diversity metrics, stability measures
    - Temporal Path Finding: BFS with temporal validity constraints
    - Version Management: Snapshot creation, version comparison, timestamp-based versioning

Deduplication:
    - Duplicate Group Detection: Similarity-based clustering, threshold-based grouping
    - Entity Merging: Property aggregation strategies, metadata merging
    - Provenance Tracking: Source tracking for merged entities, lineage maintenance

Provenance Tracking:
    - Source Tracking: Multi-source entity tracking, timestamp recording
    - Lineage Retrieval: Complete provenance history reconstruction
    - Metadata Aggregation: Source metadata merging, temporal metadata tracking
    - Temporal Tracking: First seen, last updated timestamp management

Seed Management:
    - Data Normalization: Format conversion (list, dict, single entity), ID generation
    - Source Tracking: Source identifier assignment, metadata attachment
    - File Loading: JSON parsing, file-based seed data loading

Key Features:
    - Knowledge graph construction from multiple sources
    - Temporal knowledge graph support with time-aware edges
    - Entity resolution and deduplication
    - Conflict detection and resolution
    - Comprehensive graph analytics (centrality, communities, connectivity)
    - Graph validation and consistency checking
    - Temporal queries and pattern detection
    - Provenance tracking and lineage management
    - Method registry for extensibility
    - Configuration management with environment variables and config files

Main Classes:
    - GraphBuilder: Knowledge graph construction with temporal support
    - EntityResolver: Entity resolution and deduplication
    - GraphAnalyzer: Graph analytics with temporal evolution analysis
    - TemporalGraphQuery: Time-aware graph querying
    - TemporalPatternDetector: Temporal pattern detection
    - TemporalVersionManager: Temporal versioning and snapshots
    - ConflictDetector: Conflict detection and resolution
    - ProvenanceTracker: Provenance tracking and management
    - CentralityCalculator: Centrality measures calculation
    - CommunityDetector: Community detection
    - ConnectivityAnalyzer: Connectivity analysis
    - Deduplicator: Graph deduplication
    - GraphValidator: Graph validation
    - SeedManager: Seed data management
    - MethodRegistry: Registry for custom KG methods
    - KGConfig: Configuration manager for KG module

Convenience Functions:
    - build: Build knowledge graph from sources
    - build_kg: Knowledge graph building wrapper
    - analyze_graph: Graph analysis wrapper
    - resolve_entities: Entity resolution wrapper
    - validate_graph: Graph validation wrapper
    - detect_conflicts: Conflict detection wrapper
    - calculate_centrality: Centrality calculation wrapper
    - detect_communities: Community detection wrapper
    - analyze_connectivity: Connectivity analysis wrapper
    - deduplicate_graph: Deduplication wrapper
    - query_temporal: Temporal query wrapper
    - get_kg_method: Get KG method by name
    - list_available_methods: List registered methods

Example Usage:
    >>> from semantica.kg import build, build_kg, analyze_graph, calculate_centrality
    >>> # Using convenience function
    >>> kg = build(sources=[{"entities": [...], "relationships": [...]}])
    >>> # Using method functions
    >>> kg = build_kg(sources, method="default")
    >>> analysis = analyze_graph(kg, method="default")
    >>> centrality = calculate_centrality(kg, method="degree")
    >>> # Using classes directly
    >>> from semantica.kg import GraphBuilder
    >>> builder = GraphBuilder(merge_entities=True, resolve_conflicts=True)
    >>> graph = builder.build(sources)

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union

from .graph_builder import GraphBuilder
from .entity_resolver import EntityResolver
from .graph_analyzer import GraphAnalyzer
from .temporal_query import TemporalGraphQuery, TemporalPatternDetector, TemporalVersionManager
from .conflict_detector import ConflictDetector
from .provenance_tracker import ProvenanceTracker
from .centrality_calculator import CentralityCalculator
from .community_detector import CommunityDetector
from .connectivity_analyzer import ConnectivityAnalyzer
from .deduplicator import Deduplicator
from .graph_validator import GraphValidator
from .seed_manager import SeedManager
from .registry import MethodRegistry, method_registry
from .methods import (
    build_kg,
    analyze_graph,
    resolve_entities,
    validate_graph,
    detect_conflicts,
    calculate_centrality,
    detect_communities,
    analyze_connectivity,
    deduplicate_graph,
    query_temporal,
    get_kg_method,
    list_available_methods,
)
from .config import KGConfig, kg_config

__all__ = [
    # Core Classes
    "GraphBuilder",
    "EntityResolver",
    "GraphAnalyzer",
    "TemporalGraphQuery",
    "TemporalPatternDetector",
    "TemporalVersionManager",
    "ConflictDetector",
    "ProvenanceTracker",
    "CentralityCalculator",
    "CommunityDetector",
    "ConnectivityAnalyzer",
    "Deduplicator",
    "GraphValidator",
    "SeedManager",
    # Registry and Methods
    "MethodRegistry",
    "method_registry",
    "build_kg",
    "analyze_graph",
    "resolve_entities",
    "validate_graph",
    "detect_conflicts",
    "calculate_centrality",
    "detect_communities",
    "analyze_connectivity",
    "deduplicate_graph",
    "query_temporal",
    "get_kg_method",
    "list_available_methods",
    # Configuration
    "KGConfig",
    "kg_config",
    # Convenience Functions
    "build",
]


def build(
    sources: Union[List[Any], Any],
    merge_entities: bool = True,
    entity_resolution_strategy: str = "fuzzy",
    resolve_conflicts: bool = True,
    enable_temporal: bool = False,
    temporal_granularity: str = "day",
    track_history: bool = False,
    version_snapshots: bool = False,
    entity_resolver: Optional[EntityResolver] = None,
    **options
) -> Dict[str, Any]:
    """
    Build knowledge graph from sources (module-level convenience function).
    
    This is a user-friendly wrapper around GraphBuilder.build() that creates
    a GraphBuilder instance and builds the knowledge graph.
    
    Args:
        sources: List of sources (documents, entities, relationships, or dicts with entities/relationships)
        merge_entities: Whether to merge duplicate entities (default: True)
        entity_resolution_strategy: Strategy for entity resolution - "fuzzy", "exact", "semantic" (default: "fuzzy")
        resolve_conflicts: Whether to resolve conflicts (default: True)
        enable_temporal: Enable temporal knowledge graph features (default: False)
        temporal_granularity: Time granularity - "second", "minute", "hour", "day", etc. (default: "day")
        track_history: Track historical changes (default: False)
        version_snapshots: Create version snapshots (default: False)
        entity_resolver: Optional EntityResolver instance (default: None, creates one if needed)
        **options: Additional build options
        
    Returns:
        Dictionary containing:
            - entities: List of entities
            - relationships: List of relationships
            - metadata: Graph metadata including counts and timestamps
            
    Examples:
        >>> import semantica
        >>> result = semantica.kg.build(
        ...     sources=[{"entities": [...], "relationships": [...]}],
        ...     merge_entities=True,
        ...     resolve_conflicts=True
        ... )
        >>> print(f"Built graph with {result['metadata']['num_entities']} entities")
    """
    # Normalize sources to list
    if not isinstance(sources, list):
        sources = [sources]
    
    # Create GraphBuilder instance
    graph_builder = GraphBuilder(
        merge_entities=merge_entities,
        entity_resolution_strategy=entity_resolution_strategy,
        resolve_conflicts=resolve_conflicts,
        enable_temporal=enable_temporal,
        temporal_granularity=temporal_granularity,
        track_history=track_history,
        version_snapshots=version_snapshots,
        **options
    )
    
    # Build knowledge graph
    graph = graph_builder.build(sources, entity_resolver=entity_resolver, **options)
    
    return graph
