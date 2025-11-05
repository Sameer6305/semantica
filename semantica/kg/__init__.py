"""
Knowledge Graph Management Module

This module provides comprehensive knowledge graph construction and management capabilities,
including temporal knowledge graph support for time-aware knowledge representation.

Exports:
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
"""

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

__all__ = [
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
]
