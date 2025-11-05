"""
Conflict Detection Module

This module identifies conflicts from multiple sources and
provides investigation guides for discrepancies.
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

__all__ = [
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
]
