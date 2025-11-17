"""
Knowledge Graph Quality Assurance Methods Module

This module provides all KG QA methods as simple, reusable functions for
quality assessment, validation, reporting, and automated fixing. It supports
multiple approaches and integrates with the method registry for extensibility.

Supported Methods:

Quality Assessment:
    - "default": Default quality assessment using KGQualityAssessor
    - "comprehensive": Comprehensive assessment with all metrics
    - "quick": Quick assessment with basic metrics

Quality Reporting:
    - "default": Default report generation
    - "detailed": Detailed report with all issues
    - "summary": Summary report only

Consistency Checking:
    - "logical": Logical consistency checking
    - "temporal": Temporal consistency checking
    - "hierarchical": Hierarchical consistency checking
    - "all": All consistency checks

Completeness Validation:
    - "entity": Entity completeness validation
    - "relationship": Relationship completeness validation
    - "property": Property completeness validation
    - "all": All completeness checks

Quality Metrics:
    - "overall": Overall quality score
    - "entity": Entity quality score
    - "relationship": Relationship quality score
    - "completeness": Completeness metrics
    - "consistency": Consistency metrics

Validation:
    - "default": Default validation with stored rules
    - "custom": Custom rule validation
    - "constraints": Constraint-based validation

Automated Fixes:
    - "duplicates": Fix duplicate entities
    - "inconsistencies": Fix inconsistencies
    - "missing_properties": Fix missing properties
    - "all": Apply all fixes

Algorithms Used:

Quality Metrics Calculation:
    - Weighted Averaging: Overall quality score aggregation using weighted average formula: overall = (0.6 * completeness) + (0.4 * consistency)
    - Entity Quality Scoring: Required field presence checking (ID/URI, type), binary scoring (0.5 per field), average calculation across entities
    - Relationship Quality Scoring: Required field presence checking (source/subject, target/object, type/predicate), weighted scoring (0.33 per field), average calculation across relationships
    - Score Normalization: Min-max normalization with clamping to 0.0-1.0 range

Completeness Metrics:
    - Entity Completeness Calculation: Schema-based required property validation, ratio calculation present_props / required_props, average completeness across entities
    - Relationship Completeness Calculation: Required field validation (source, target, type), completeness ratio calculation, average across relationships
    - Property Completeness Calculation: Schema-based property validation per entity type, completeness ratio calculation, average across entity types

Consistency Metrics:
    - Logical Consistency Checking: Contradiction detection, conflicting relationship identification, inconsistent property value detection
    - Temporal Consistency Checking: Temporal contradiction detection, invalid time range validation, conflicting temporal relationship identification
    - Hierarchical Consistency Checking: Circular inheritance detection (DFS-based cycle detection), invalid parent-child relationship validation

Validation Engine:
    - Rule-Based Validation: Custom rule function execution, rule result parsing (error/warning extraction), exception handling and error collection
    - Constraint-Based Validation: Entity constraint validation (required properties), relationship constraint validation (domain and range), constraint matching algorithms

Quality Reporting:
    - Issue Identification: Threshold-based issue detection (overall < 0.7, completeness < 0.8), issue type classification, severity assignment
    - Recommendation Generation: Issue-based recommendation generation, score-based recommendation generation, actionable suggestion creation
    - Report Serialization: JSON serialization (ISO timestamp formatting), YAML serialization (with PyYAML fallback), HTML report generation

Automated Fixes:
    - Duplicate Detection: Entity duplicate identification (using deduplication module), relationship duplicate identification
    - Duplicate Merging: Property aggregation strategies, relationship reference updating, entity consolidation
    - Conflict Resolution: Conflicting property value detection, resolution strategy selection, conflict merging
    - Missing Property Completion: Schema-based required property identification, default value assignment, value inference

Key Features:
    - Multiple QA operation methods
    - Quality assessment with method dispatch
    - Method dispatchers with registry support
    - Custom method registration capability
    - Consistent interface across all methods

Main Functions:
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
    >>> from semantica.kg_qa.methods import assess_quality, generate_quality_report
    >>> score = assess_quality(knowledge_graph, method="default")
    >>> report = generate_quality_report(knowledge_graph, schema, method="default")
"""

from typing import Any, Dict, List, Optional, Callable, Union

from ..utils.logging import get_logger
from ..utils.exceptions import ProcessingError, ConfigurationError
from .kg_quality_assessor import KGQualityAssessor, ConsistencyChecker, CompletenessValidator
from .quality_metrics import QualityMetrics, CompletenessMetrics, ConsistencyMetrics
from .validation_engine import ValidationEngine
from .reporting import QualityReporter, QualityReport
from .automated_fixes import AutomatedFixer, FixResult
from .registry import method_registry
from .config import kg_qa_config

logger = get_logger("kg_qa_methods")


def assess_quality(
    knowledge_graph: Any,
    method: str = "default",
    **kwargs
) -> float:
    """
    Assess overall quality of knowledge graph (convenience function).
    
    This is a user-friendly wrapper that assesses knowledge graph quality
    using the specified method.
    
    Args:
        knowledge_graph: Knowledge graph instance (object with entities
                       and relationships, or dict with "entities" and
                       "relationships" keys)
        method: Assessment method (default: "default")
            - "default": Use KGQualityAssessor with default settings
            - "comprehensive": Comprehensive assessment with all metrics
            - "quick": Quick assessment with basic metrics
        **kwargs: Additional options passed to KGQualityAssessor
        
    Returns:
        float: Overall quality score between 0.0 and 1.0 (higher is better)
        
    Examples:
        >>> from semantica.kg_qa.methods import assess_quality
        >>> score = assess_quality(knowledge_graph, method="default")
        >>> quick_score = assess_quality(knowledge_graph, method="quick")
    """
    custom_method = method_registry.get("assess", method)
    if custom_method:
        try:
            return custom_method(knowledge_graph, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("assess")
        config.update(kwargs)
        
        assessor = KGQualityAssessor(**config)
        return assessor.assess_overall_quality(knowledge_graph)
        
    except Exception as e:
        logger.error(f"Failed to assess quality: {e}")
        raise


def generate_quality_report(
    knowledge_graph: Any,
    schema: Optional[Dict[str, Any]] = None,
    method: str = "default",
    **kwargs
) -> QualityReport:
    """
    Generate comprehensive quality report (convenience function).
    
    This is a user-friendly wrapper that generates a quality report using
    the specified method.
    
    Args:
        knowledge_graph: Knowledge graph instance
        schema: Optional schema definition for validation
        method: Report generation method (default: "default")
            - "default": Use KGQualityAssessor with default settings
            - "detailed": Detailed report with all issues
            - "summary": Summary report only
        **kwargs: Additional options passed to KGQualityAssessor
        
    Returns:
        QualityReport: Comprehensive quality report containing:
            - timestamp: Report generation timestamp
            - overall_score: Overall quality score
            - completeness_score: Completeness score
            - consistency_score: Consistency score
            - issues: List of identified quality issues
            - recommendations: List of improvement recommendations
            - metadata: Additional report metadata
            
    Examples:
        >>> from semantica.kg_qa.methods import generate_quality_report
        >>> report = generate_quality_report(knowledge_graph, schema, method="default")
    """
    custom_method = method_registry.get("report", method)
    if custom_method:
        try:
            return custom_method(knowledge_graph, schema, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("report")
        config.update(kwargs)
        
        assessor = KGQualityAssessor(**config)
        return assessor.generate_quality_report(knowledge_graph, schema)
        
    except Exception as e:
        logger.error(f"Failed to generate quality report: {e}")
        raise


def identify_quality_issues(
    knowledge_graph: Any,
    schema: Optional[Dict[str, Any]] = None,
    method: str = "default",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Identify quality issues in knowledge graph (convenience function).
    
    This is a user-friendly wrapper that identifies quality issues using
    the specified method.
    
    Args:
        knowledge_graph: Knowledge graph instance
        schema: Optional schema for validation
        method: Issue identification method (default: "default")
        **kwargs: Additional options passed to KGQualityAssessor
        
    Returns:
        list: List of quality issue dictionaries, each containing:
            - id: Issue identifier
            - type: Issue type (e.g., "completeness", "consistency")
            - severity: Issue severity ("low", "medium", "high")
            - description: Issue description
            - entity_id: Related entity ID (if applicable)
            - relationship_id: Related relationship ID (if applicable)
            
    Examples:
        >>> from semantica.kg_qa.methods import identify_quality_issues
        >>> issues = identify_quality_issues(knowledge_graph, schema, method="default")
    """
    custom_method = method_registry.get("assess", method)
    if custom_method:
        try:
            return custom_method(knowledge_graph, schema, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("assess")
        config.update(kwargs)
        
        assessor = KGQualityAssessor(**config)
        return assessor.identify_quality_issues(knowledge_graph, schema)
        
    except Exception as e:
        logger.error(f"Failed to identify quality issues: {e}")
        raise


def check_consistency(
    knowledge_graph: Any,
    consistency_type: str = "logical",
    method: str = "default",
    **kwargs
) -> Union[bool, Dict[str, bool]]:
    """
    Check consistency of knowledge graph (convenience function).
    
    This is a user-friendly wrapper that checks consistency using
    the specified method.
    
    Args:
        knowledge_graph: Knowledge graph instance
        consistency_type: Type of consistency to check (default: "logical")
            - "logical": Logical consistency checking
            - "temporal": Temporal consistency checking
            - "hierarchical": Hierarchical consistency checking
            - "all": All consistency checks (returns dict)
        method: Consistency checking method (default: "default")
        **kwargs: Additional options passed to ConsistencyChecker
        
    Returns:
        bool or dict: Consistency check result(s)
            - If consistency_type is "all", returns dict with keys:
              "logical", "temporal", "hierarchical"
            - Otherwise returns bool (True if consistent)
            
    Examples:
        >>> from semantica.kg_qa.methods import check_consistency
        >>> is_consistent = check_consistency(knowledge_graph, consistency_type="logical")
        >>> all_checks = check_consistency(knowledge_graph, consistency_type="all")
    """
    custom_method = method_registry.get("consistency", method)
    if custom_method:
        try:
            return custom_method(knowledge_graph, consistency_type, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("consistency")
        config.update(kwargs)
        
        checker = ConsistencyChecker(**config)
        
        if consistency_type == "all":
            return {
                "logical": checker.check_logical_consistency(knowledge_graph),
                "temporal": checker.check_temporal_consistency(knowledge_graph),
                "hierarchical": checker.check_hierarchical_consistency(knowledge_graph)
            }
        elif consistency_type == "logical":
            return checker.check_logical_consistency(knowledge_graph)
        elif consistency_type == "temporal":
            return checker.check_temporal_consistency(knowledge_graph)
        elif consistency_type == "hierarchical":
            return checker.check_hierarchical_consistency(knowledge_graph)
        else:
            raise ValueError(f"Unknown consistency type: {consistency_type}")
        
    except Exception as e:
        logger.error(f"Failed to check consistency: {e}")
        raise


def validate_completeness(
    entities: Optional[List[Dict[str, Any]]] = None,
    relationships: Optional[List[Dict[str, Any]]] = None,
    properties: Optional[Dict[str, Any]] = None,
    schema: Dict[str, Any] = None,
    completeness_type: str = "entity",
    method: str = "default",
    **kwargs
) -> Union[bool, Dict[str, bool]]:
    """
    Validate completeness of knowledge graph (convenience function).
    
    This is a user-friendly wrapper that validates completeness using
    the specified method.
    
    Args:
        entities: Optional list of entity dictionaries
        relationships: Optional list of relationship dictionaries
        properties: Optional properties dictionary
        schema: Schema definition containing constraints
        completeness_type: Type of completeness to validate (default: "entity")
            - "entity": Entity completeness validation
            - "relationship": Relationship completeness validation
            - "property": Property completeness validation
            - "all": All completeness checks (returns dict)
        method: Completeness validation method (default: "default")
        **kwargs: Additional options passed to CompletenessValidator
        
    Returns:
        bool or dict: Completeness validation result(s)
            - If completeness_type is "all", returns dict with keys:
              "entity", "relationship", "property"
            - Otherwise returns bool (True if complete)
            
    Examples:
        >>> from semantica.kg_qa.methods import validate_completeness
        >>> is_complete = validate_completeness(entities, schema, completeness_type="entity")
        >>> all_checks = validate_completeness(entities, relationships, properties, schema, completeness_type="all")
    """
    custom_method = method_registry.get("completeness", method)
    if custom_method:
        try:
            return custom_method(entities, relationships, properties, schema, completeness_type, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("completeness")
        config.update(kwargs)
        
        validator = CompletenessValidator(**config)
        
        if completeness_type == "all":
            results = {}
            if entities and schema:
                results["entity"] = validator.validate_entity_completeness(entities, schema)
            if relationships and schema:
                results["relationship"] = validator.validate_relationship_completeness(relationships, schema)
            if properties and schema:
                results["property"] = validator.validate_property_completeness(properties, schema)
            return results
        elif completeness_type == "entity":
            if not entities or not schema:
                raise ValueError("entities and schema are required for entity completeness validation")
            return validator.validate_entity_completeness(entities, schema)
        elif completeness_type == "relationship":
            if not relationships or not schema:
                raise ValueError("relationships and schema are required for relationship completeness validation")
            return validator.validate_relationship_completeness(relationships, schema)
        elif completeness_type == "property":
            if not properties or not schema:
                raise ValueError("properties and schema are required for property completeness validation")
            return validator.validate_property_completeness(properties, schema)
        else:
            raise ValueError(f"Unknown completeness type: {completeness_type}")
        
    except Exception as e:
        logger.error(f"Failed to validate completeness: {e}")
        raise


def calculate_quality_metrics(
    knowledge_graph: Any,
    metrics_type: str = "overall",
    method: str = "default",
    **kwargs
) -> Union[float, Dict[str, float]]:
    """
    Calculate quality metrics for knowledge graph (convenience function).
    
    This is a user-friendly wrapper that calculates quality metrics using
    the specified method.
    
    Args:
        knowledge_graph: Knowledge graph instance
        metrics_type: Type of metrics to calculate (default: "overall")
            - "overall": Overall quality score
            - "entity": Entity quality score
            - "relationship": Relationship quality score
            - "completeness": Completeness metrics
            - "consistency": Consistency metrics
            - "all": All metrics (returns dict)
        method: Metrics calculation method (default: "default")
        **kwargs: Additional options passed to QualityMetrics
        
    Returns:
        float or dict: Quality metric(s)
            - If metrics_type is "all", returns dict with all metrics
            - Otherwise returns float score
            
    Examples:
        >>> from semantica.kg_qa.methods import calculate_quality_metrics
        >>> score = calculate_quality_metrics(knowledge_graph, metrics_type="overall")
        >>> all_metrics = calculate_quality_metrics(knowledge_graph, metrics_type="all")
    """
    custom_method = method_registry.get("metrics", method)
    if custom_method:
        try:
            return custom_method(knowledge_graph, metrics_type, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("metrics")
        config.update(kwargs)
        
        quality_metrics = QualityMetrics(**config)
        entities = getattr(knowledge_graph, "entities", knowledge_graph.get("entities", []) if isinstance(knowledge_graph, dict) else [])
        relationships = getattr(knowledge_graph, "relationships", knowledge_graph.get("relationships", []) if isinstance(knowledge_graph, dict) else [])
        
        if metrics_type == "all":
            return {
                "overall": quality_metrics.calculate_overall_score(knowledge_graph),
                "entity": quality_metrics.calculate_entity_quality(entities) if entities else 0.0,
                "relationship": quality_metrics.calculate_relationship_quality(relationships) if relationships else 0.0
            }
        elif metrics_type == "overall":
            return quality_metrics.calculate_overall_score(knowledge_graph)
        elif metrics_type == "entity":
            if not entities:
                raise ValueError("Knowledge graph must have entities for entity quality calculation")
            return quality_metrics.calculate_entity_quality(entities)
        elif metrics_type == "relationship":
            if not relationships:
                raise ValueError("Knowledge graph must have relationships for relationship quality calculation")
            return quality_metrics.calculate_relationship_quality(relationships)
        else:
            raise ValueError(f"Unknown metrics type: {metrics_type}")
        
    except Exception as e:
        logger.error(f"Failed to calculate quality metrics: {e}")
        raise


def validate_graph(
    knowledge_graph: Any,
    rules: Optional[List[Callable]] = None,
    method: str = "default",
    **kwargs
) -> Any:
    """
    Validate knowledge graph (convenience function).
    
    This is a user-friendly wrapper that validates a knowledge graph using
    the specified method.
    
    Args:
        knowledge_graph: Knowledge graph instance to validate
        rules: Optional list of validation rule functions
        method: Validation method (default: "default")
            - "default": Default validation with stored rules
            - "custom": Custom rule validation
            - "constraints": Constraint-based validation
        **kwargs: Additional options passed to ValidationEngine
        
    Returns:
        ValidationResult: Validation result containing:
            - valid: True if no errors, False otherwise
            - errors: List of error messages
            - warnings: List of warning messages
            - metadata: Additional validation metadata
            
    Examples:
        >>> from semantica.kg_qa.methods import validate_graph
        >>> result = validate_graph(knowledge_graph, method="default")
    """
    custom_method = method_registry.get("validate", method)
    if custom_method:
        try:
            return custom_method(knowledge_graph, rules, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("validate")
        config.update(kwargs)
        
        engine = ValidationEngine(**config)
        return engine.validate(knowledge_graph, rules)
        
    except Exception as e:
        logger.error(f"Failed to validate graph: {e}")
        raise


def export_report(
    report: QualityReport,
    format: str = "json",
    method: str = "default",
    **kwargs
) -> str:
    """
    Export quality report to specified format (convenience function).
    
    This is a user-friendly wrapper that exports a quality report using
    the specified method.
    
    Args:
        report: Quality report to export
        format: Export format (default: "json")
            - "json": JSON format
            - "yaml": YAML format
            - "html": HTML format (planned)
        method: Export method (default: "default")
        **kwargs: Additional options passed to QualityReporter
        
    Returns:
        str: Exported report as string in the specified format
        
    Examples:
        >>> from semantica.kg_qa.methods import export_report
        >>> json_report = export_report(report, format="json")
        >>> yaml_report = export_report(report, format="yaml")
    """
    custom_method = method_registry.get("report", method)
    if custom_method:
        try:
            return custom_method(report, format, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("report")
        config.update(kwargs)
        
        reporter = QualityReporter(**config)
        return reporter.export_report(report, format=format)
        
    except Exception as e:
        logger.error(f"Failed to export report: {e}")
        raise


def fix_issues(
    knowledge_graph: Any,
    fix_type: str = "duplicates",
    schema: Optional[Dict[str, Any]] = None,
    method: str = "default",
    **kwargs
) -> FixResult:
    """
    Fix quality issues in knowledge graph (convenience function).
    
    This is a user-friendly wrapper that fixes quality issues using
    the specified method.
    
    Args:
        knowledge_graph: Knowledge graph instance
        fix_type: Type of fix to apply (default: "duplicates")
            - "duplicates": Fix duplicate entities
            - "inconsistencies": Fix inconsistencies
            - "missing_properties": Fix missing properties
            - "all": Apply all fixes
        schema: Optional schema definition (required for missing_properties)
        method: Fixing method (default: "default")
        **kwargs: Additional options passed to AutomatedFixer
        
    Returns:
        FixResult: Fix result containing:
            - success: Whether fixing was successful
            - fixed_count: Number of issues fixed
            - errors: List of error messages
            - metadata: Additional fix metadata
            
    Examples:
        >>> from semantica.kg_qa.methods import fix_issues
        >>> result = fix_issues(knowledge_graph, fix_type="duplicates")
        >>> result = fix_issues(knowledge_graph, fix_type="missing_properties", schema=schema)
    """
    custom_method = method_registry.get("fix", method)
    if custom_method:
        try:
            return custom_method(knowledge_graph, fix_type, schema, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = kg_qa_config.get_method_config("fix")
        config.update(kwargs)
        
        fixer = AutomatedFixer(**config)
        
        if fix_type == "all":
            # Apply all fixes sequentially
            results = []
            results.append(fixer.fix_duplicates(knowledge_graph))
            results.append(fixer.fix_inconsistencies(knowledge_graph))
            if schema:
                results.append(fixer.fix_missing_properties(knowledge_graph, schema))
            
            total_fixed = sum(r.fixed_count for r in results)
            all_errors = []
            for r in results:
                all_errors.extend(r.errors)
            
            return FixResult(
                success=all(r.success for r in results),
                fixed_count=total_fixed,
                errors=all_errors,
                metadata={"fixes_applied": [fix_type for r in results if r.success]}
            )
        elif fix_type == "duplicates":
            return fixer.fix_duplicates(knowledge_graph)
        elif fix_type == "inconsistencies":
            return fixer.fix_inconsistencies(knowledge_graph)
        elif fix_type == "missing_properties":
            if not schema:
                raise ValueError("schema is required for missing_properties fix")
            return fixer.fix_missing_properties(knowledge_graph, schema)
        else:
            raise ValueError(f"Unknown fix type: {fix_type}")
        
    except Exception as e:
        logger.error(f"Failed to fix issues: {e}")
        raise


def get_qa_method(task: str, name: str) -> Optional[Callable]:
    """Get QA method by task and name."""
    return method_registry.get(task, name)


def list_available_methods(task: Optional[str] = None) -> Dict[str, List[str]]:
    """List all registered QA methods."""
    return method_registry.list_all(task)


# Register default methods
method_registry.register("assess", "default", assess_quality)
method_registry.register("report", "default", generate_quality_report)
method_registry.register("consistency", "default", check_consistency)
method_registry.register("completeness", "default", validate_completeness)
method_registry.register("metrics", "default", calculate_quality_metrics)
method_registry.register("validate", "default", validate_graph)
method_registry.register("fix", "default", fix_issues)

