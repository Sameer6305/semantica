"""
Knowledge Graph Quality Assurance Module

Comprehensive quality assurance for production-ready Knowledge Graphs.

Key Features:
    - Quality metrics calculation
    - Consistency checking
    - Completeness validation
    - Automated fixes
    - Quality reporting

Main Classes:
    - KGQualityAssessor: Overall quality assessment
    - ConsistencyChecker: Consistency validation
    - CompletenessValidator: Completeness validation
"""

from .kg_quality_assessor import (
    KGQualityAssessor,
    ConsistencyChecker,
    CompletenessValidator,
)
from .quality_metrics import QualityMetrics, CompletenessMetrics, ConsistencyMetrics
from .validation_engine import ValidationEngine, RuleValidator, ConstraintValidator
from .reporting import QualityReporter, IssueTracker, ImprovementSuggestions, QualityReport
from .automated_fixes import AutomatedFixer, AutoMerger, AutoResolver

__all__ = [
    # Main classes
    "KGQualityAssessor",
    "ConsistencyChecker",
    "CompletenessValidator",
    # Quality metrics
    "QualityMetrics",
    "CompletenessMetrics",
    "ConsistencyMetrics",
    # Validation
    "ValidationEngine",
    "RuleValidator",
    "ConstraintValidator",
    # Reporting
    "QualityReporter",
    "IssueTracker",
    "ImprovementSuggestions",
    "QualityReport",
    # Automated fixes
    "AutomatedFixer",
    "AutoMerger",
    "AutoResolver",
]

