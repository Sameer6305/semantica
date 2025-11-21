"""
Conflict Detection and Resolution Module

This module provides comprehensive conflict detection and resolution capabilities
for the Semantica framework, identifying conflicts from multiple sources and
providing investigation guides for discrepancies. It enables source tracking,
conflict analysis, and automated resolution strategies.

Algorithms Used:

Conflict Detection:
    - Value Comparison: Property value comparison across sources with equality checking
    - Type Mismatch Detection: Entity type comparison and mismatch identification
    - Relationship Consistency: Relationship property comparison and inconsistency detection
    - Temporal Analysis: Time-based conflict detection using timestamp comparison
    - Logical Consistency: Logical rule validation and inconsistency detection
    - Severity Calculation: Multi-factor severity scoring (property importance, value difference, source count)
    - Confidence Scoring: Confidence calculation based on source credibility and value diversity

Conflict Resolution:
    - Voting Algorithm: Majority value selection using Counter-based frequency counting
    - Credibility Weighted: Weighted average calculation using source credibility scores
    - Temporal Selection: Timestamp-based selection (newest/oldest value)
    - Confidence Selection: Maximum confidence value selection
    - Manual Flagging: Conflict flagging for human review workflow

Conflict Analysis:
    - Pattern Identification: Frequency-based pattern detection using Counter and defaultdict
    - Type Classification: Conflict type categorization and grouping
    - Severity Analysis: Severity-based grouping and statistical analysis
    - Source Analysis: Source-based conflict aggregation and analysis
    - Trend Analysis: Temporal trend identification using time-series analysis
    - Statistical Analysis: Conflict statistics calculation (mean, median, distribution)

Source Tracking:
    - Property Source Tracking: Dictionary-based property-to-source mapping
    - Entity Source Tracking: Entity-to-source relationship tracking
    - Relationship Source Tracking: Relationship-to-source mapping
    - Credibility Scoring: Source credibility calculation based on historical accuracy
    - Traceability Chain: Graph-based traceability chain generation

Investigation Guide:
    - Guide Generation: Template-based investigation guide generation
    - Checklist Generation: Step-by-step checklist creation
    - Context Extraction: Conflict context and metadata extraction
    - Step Generation: Investigation step generation based on conflict type

Key Features:
    - Multi-source conflict detection (value, type, relationship, temporal, logical)
    - Source tracking and provenance management
    - Conflict analysis and pattern identification
    - Multiple resolution strategies (voting, credibility-weighted, recency, confidence)
    - Investigation guide generation
    - Source credibility scoring
    - Conflict reporting and statistics
    - Method registry for custom conflict methods
    - Configuration management with environment variables and config files

Main Classes:
    - ConflictDetector: Detects conflicts from multiple sources
    - ConflictResolver: Resolves conflicts using various strategies
    - ConflictAnalyzer: Analyzes conflict patterns and trends
    - SourceTracker: Tracks source information and provenance
    - InvestigationGuideGenerator: Generates investigation guides
    - MethodRegistry: Registry for custom conflict methods
    - ConflictsConfig: Configuration manager for conflicts module

Convenience Functions:
    - detect_and_resolve: Detect and resolve conflicts in one call

Example Usage:
    >>> from semantica.conflicts import detect_and_resolve, ConflictDetector, ConflictResolver
    >>> # Using convenience function
    >>> results = detect_and_resolve(entities, property_name="name", resolution_strategy="voting")
    >>> # Using classes directly
    >>> detector = ConflictDetector()
    >>> conflicts = detector.detect_value_conflicts(entities, "name")
    >>> resolver = ConflictResolver()
    >>> results = resolver.resolve_conflicts(conflicts, strategy="voting")

Author: Semantica Contributors
License: MIT
"""

from .conflict_detector import ConflictDetector, Conflict, ConflictType
from .source_tracker import SourceTracker, SourceReference, PropertySource
from .conflict_resolver import ConflictResolver, ResolutionResult, ResolutionStrategy
from .investigation_guide import (
    InvestigationGuideGenerator,
    InvestigationGuide,
    InvestigationStep,
)
from .conflict_analyzer import ConflictAnalyzer, ConflictPattern
from .registry import MethodRegistry, method_registry
from .methods import (
    detect_conflicts,
    resolve_conflicts,
    analyze_conflicts,
    track_sources,
    generate_investigation_guide,
    get_conflict_method,
    list_available_methods,
)
from .config import ConflictsConfig, conflicts_config


def detect_and_resolve(
    entities,
    property_name: str = None,
    entity_type: str = None,
    detection_method: str = "value",
    resolution_strategy: str = "voting",
    **kwargs
):
    """
    Detect and resolve conflicts in entities (convenience function).
    
    This is a user-friendly wrapper that detects and resolves conflicts in one call.
    
    Args:
        entities: List of entity dictionaries to check for conflicts
        property_name: Property name to check (required for value conflicts)
        entity_type: Optional entity type filter
        detection_method: Detection method (default: "value")
        resolution_strategy: Resolution strategy (default: "voting")
        **kwargs: Additional options passed to detector and resolver
        
    Returns:
        Tuple of (conflicts, resolution_results)
        
    Examples:
        >>> from semantica.conflicts import detect_and_resolve
        >>> entities = [
        ...     {"id": "1", "name": "Apple Inc.", "source": "doc1"},
        ...     {"id": "1", "name": "Apple", "source": "doc2"}
        ... ]
        >>> conflicts, results = detect_and_resolve(
        ...     entities,
        ...     property_name="name",
        ...     resolution_strategy="voting"
        ... )
    """
    conflicts = detect_conflicts(
        entities,
        method=detection_method,
        property_name=property_name,
        entity_type=entity_type,
        **kwargs
    )
    
    if not conflicts:
        return conflicts, []
    
    results = resolve_conflicts(
        conflicts,
        method=resolution_strategy,
        **kwargs
    )
    
    return conflicts, results


__all__ = [
    # Core Classes
    "ConflictDetector",
    "Conflict",
    "ConflictType",
    "SourceTracker",
    "SourceReference",
    "PropertySource",
    "ConflictResolver",
    "ResolutionResult",
    "ResolutionStrategy",
    "InvestigationGuideGenerator",
    "InvestigationGuide",
    "InvestigationStep",
    "ConflictAnalyzer",
    "ConflictPattern",
    # Registry and Methods
    "MethodRegistry",
    "method_registry",
    "detect_conflicts",
    "resolve_conflicts",
    "analyze_conflicts",
    "track_sources",
    "generate_investigation_guide",
    "get_conflict_method",
    "list_available_methods",
    # Configuration
    "ConflictsConfig",
    "conflicts_config",
    # Convenience Functions
    "detect_and_resolve",
]
