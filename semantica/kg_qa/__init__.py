"""
Knowledge Graph Quality Assurance Module

This module provides comprehensive quality assurance capabilities for the
Semantica framework, enabling production-ready knowledge graph quality
assessment, validation, and automated fixes.

Algorithms Used:

Quality Metrics Calculation:
    - Weighted Averaging: Overall quality score aggregation using weighted average formula: overall = (0.6 * completeness) + (0.4 * consistency)
    - Entity Quality Scoring: Required field presence checking (ID/URI, type), binary scoring (0.5 per field), average calculation across entities: sum(scores) / len(scores)
    - Relationship Quality Scoring: Required field presence checking (source/subject, target/object, type/predicate), weighted scoring (0.33 per field), average calculation across relationships
    - Score Normalization: Min-max normalization with clamping to 0.0-1.0 range: min(1.0, max(0.0, score))
    - Consistency Score Calculation: Logical inconsistency detection (placeholder for reasoner-based checking)

Completeness Metrics:
    - Entity Completeness Calculation: Schema-based required property validation, ratio calculation present_props / required_props, average completeness across entities
    - Relationship Completeness Calculation: Required field validation (source, target, type), completeness ratio (has_source + has_target + has_type) / 3.0, average across relationships
    - Property Completeness Calculation: Schema-based property validation per entity type, completeness ratio calculation, average across entity types
    - Schema Constraint Matching: Entity type to constraint mapping, required property extraction from schema constraints

Consistency Metrics:
    - Logical Consistency Checking: Contradiction detection, conflicting relationship identification, inconsistent property value detection (placeholder for reasoner integration)
    - Temporal Consistency Checking: Temporal contradiction detection, invalid time range validation, conflicting temporal relationship identification
    - Hierarchical Consistency Checking: Circular inheritance detection (DFS-based cycle detection), invalid parent-child relationship validation, hierarchical structure validation

Validation Engine:
    - Rule-Based Validation: Custom rule function execution, rule result parsing (error/warning extraction from dict), exception handling and error collection
    - Constraint-Based Validation: Entity constraint validation (required properties), relationship constraint validation (domain and range), constraint matching algorithms
    - Domain and Range Validation: Relationship type to domain/range mapping, entity type compatibility checking
    - Validation Result Aggregation: Error and warning collection, validity determination (valid = len(errors) == 0)

Quality Reporting:
    - Issue Identification: Threshold-based issue detection (overall < 0.7, completeness < 0.8), issue type classification (quality, completeness, consistency), severity assignment (low, medium, high)
    - Recommendation Generation: Issue-based recommendation generation, score-based recommendation generation, actionable suggestion creation
    - Report Serialization: JSON serialization (ISO timestamp formatting, nested structure), YAML serialization (with PyYAML fallback), HTML report generation (planned)
    - Issue Tracking: Dictionary-based issue storage (ID as key), severity-based filtering, issue resolution tracking

Automated Fixes:
    - Duplicate Detection: Entity duplicate identification (using deduplication module), relationship duplicate identification (same source, target, type matching)
    - Duplicate Merging: Property aggregation strategies, relationship reference updating, entity consolidation
    - Conflict Resolution: Conflicting property value detection, resolution strategy selection (highest confidence, most recent, source-based), conflict merging
    - Missing Property Completion: Schema-based required property identification, default value assignment, value inference from context (planned)
    - Inconsistency Resolution: Logical inconsistency detection, resolution strategy application, graph update

Quality Assessment Coordination:
    - Metric Aggregation: Multi-metric collection (overall, completeness, consistency), score combination, report generation coordination
    - Component Integration: Quality metrics integration, validation engine integration, reporting integration, automated fixing integration

Key Features:
    - Quality metrics calculation (overall, completeness, consistency)
    - Consistency checking (logical, temporal, hierarchical)
    - Completeness validation (entity, relationship, property)
    - Automated fixes (duplicates, inconsistencies, missing properties)
    - Quality reporting with issue tracking
    - Validation engine with rules and constraints
    - Method registry for extensibility
    - Configuration management with environment variables and config files

Main Classes:
    - KGQualityAssessor: Overall quality assessment coordinator
    - ConsistencyChecker: Consistency validation engine
    - CompletenessValidator: Completeness validation engine
    - QualityMetrics: Quality metrics calculator
    - CompletenessMetrics: Completeness metrics calculator
    - ConsistencyMetrics: Consistency metrics calculator
    - ValidationEngine: Rule and constraint validation
    - RuleValidator: Rule-based validation
    - ConstraintValidator: Constraint-based validation
    - QualityReporter: Quality report generation
    - IssueTracker: Issue tracking and management
    - ImprovementSuggestions: Improvement suggestions generator
    - AutomatedFixer: Automated issue fixing
    - AutoMerger: Automatic merging of duplicates and conflicts
    - AutoResolver: Automatic conflict and inconsistency resolution
    - MethodRegistry: Registry for custom QA methods
    - KGQAConfig: Configuration manager for KG QA module

Convenience Functions:
    - assess_quality: Quality assessment wrapper
    - generate_quality_report: Quality report generation wrapper
    - identify_quality_issues: Quality issue identification wrapper
    - check_consistency: Consistency checking wrapper
    - validate_completeness: Completeness validation wrapper
    - calculate_quality_metrics: Quality metrics calculation wrapper
    - validate_graph: Graph validation wrapper
    - export_report: Report export wrapper
    - fix_issues: Automated fixing wrapper
    - get_qa_method: Get QA method by name
    - list_available_methods: List registered methods

Example Usage:
    >>> from semantica.kg_qa import assess_quality, generate_quality_report, KGQualityAssessor
    >>> # Using convenience functions
    >>> score = assess_quality(knowledge_graph, method="default")
    >>> report = generate_quality_report(knowledge_graph, schema, method="default")
    >>> # Using classes directly
    >>> from semantica.kg_qa import KGQualityAssessor
    >>> assessor = KGQualityAssessor()
    >>> score = assessor.assess_overall_quality(knowledge_graph)
    >>> report = assessor.generate_quality_report(knowledge_graph, schema)

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union

from .kg_quality_assessor import (
    KGQualityAssessor,
    ConsistencyChecker,
    CompletenessValidator,
)
from .quality_metrics import QualityMetrics, CompletenessMetrics, ConsistencyMetrics
from .validation_engine import ValidationEngine, RuleValidator, ConstraintValidator
from .reporting import QualityReporter, IssueTracker, ImprovementSuggestions, QualityReport
from .automated_fixes import AutomatedFixer, AutoMerger, AutoResolver, FixResult
from .registry import MethodRegistry, method_registry
from .methods import (
    assess_quality,
    generate_quality_report,
    identify_quality_issues,
    check_consistency,
    validate_completeness,
    calculate_quality_metrics,
    validate_graph,
    export_report,
    fix_issues,
    get_qa_method,
    list_available_methods,
)
from .config import KGQAConfig, kg_qa_config

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
    "FixResult",
    # Registry and Methods
    "MethodRegistry",
    "method_registry",
    "assess_quality",
    "generate_quality_report",
    "identify_quality_issues",
    "check_consistency",
    "validate_completeness",
    "calculate_quality_metrics",
    "validate_graph",
    "export_report",
    "fix_issues",
    "get_qa_method",
    "list_available_methods",
    # Configuration
    "KGQAConfig",
    "kg_qa_config",
]

