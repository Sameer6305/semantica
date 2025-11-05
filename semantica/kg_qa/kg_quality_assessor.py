"""
KG Quality Assessor Module

Main quality assessment class that coordinates all quality assurance components.
"""

from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from .quality_metrics import QualityMetrics, CompletenessMetrics, ConsistencyMetrics
from .validation_engine import ValidationEngine
from .reporting import QualityReporter, QualityReport


class KGQualityAssessor:
    """
    Knowledge Graph Quality Assessor.
    
    Main class for assessing Knowledge Graph quality.
    Coordinates all quality assurance components.
    """
    
    def __init__(self, **kwargs):
        """Initialize KG quality assessor."""
        self.logger = get_logger("kg_quality_assessor")
        self.config = kwargs
        
        # Initialize components
        self.quality_metrics = QualityMetrics(**kwargs)
        self.completeness_metrics = CompletenessMetrics(**kwargs)
        self.consistency_metrics = ConsistencyMetrics(**kwargs)
        self.validation_engine = ValidationEngine(**kwargs)
        self.quality_reporter = QualityReporter(**kwargs)
    
    def assess_overall_quality(
        self,
        knowledge_graph: Any
    ) -> float:
        """
        Assess overall quality of knowledge graph.
        
        Args:
            knowledge_graph: Knowledge graph instance
            
        Returns:
            Overall quality score (0.0 to 1.0)
        """
        self.logger.info("Assessing overall quality")
        
        # Calculate metrics
        overall_score = self.quality_metrics.calculate_overall_score(knowledge_graph)
        
        return overall_score
    
    def generate_quality_report(
        self,
        knowledge_graph: Any,
        schema: Optional[Dict[str, Any]] = None
    ) -> QualityReport:
        """
        Generate comprehensive quality report.
        
        Args:
            knowledge_graph: Knowledge graph instance
            schema: Optional schema for validation
            
        Returns:
            Quality report
        """
        self.logger.info("Generating quality report")
        
        # Calculate metrics
        overall_score = self.quality_metrics.calculate_overall_score(knowledge_graph)
        
        # Get entities and relationships (simplified - in practice would query graph)
        entities = getattr(knowledge_graph, "entities", [])
        relationships = getattr(knowledge_graph, "relationships", [])
        
        completeness_score = 0.0
        if schema and entities:
            completeness_score = self.completeness_metrics.calculate_entity_completeness(
                entities,
                schema
            )
        
        consistency_score = self.consistency_metrics.calculate_logical_consistency(
            knowledge_graph
        )
        
        quality_metrics = {
            "overall": overall_score,
            "completeness": completeness_score,
            "consistency": consistency_score
        }
        
        # Generate report
        report = self.quality_reporter.generate_report(
            knowledge_graph,
            quality_metrics
        )
        
        return report
    
    def identify_quality_issues(
        self,
        knowledge_graph: Any,
        schema: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify quality issues in knowledge graph.
        
        Args:
            knowledge_graph: Knowledge graph instance
            schema: Optional schema for validation
            
        Returns:
            List of quality issues
        """
        self.logger.info("Identifying quality issues")
        
        # Generate report to get issues
        report = self.generate_quality_report(knowledge_graph, schema)
        
        # Convert issues to dictionaries
        issues = [
            {
                "id": issue.id,
                "type": issue.type,
                "severity": issue.severity,
                "description": issue.description,
                "entity_id": issue.entity_id,
                "relationship_id": issue.relationship_id
            }
            for issue in report.issues
        ]
        
        return issues


class ConsistencyChecker:
    """
    Consistency checker.
    
    Checks consistency of Knowledge Graph.
    """
    
    def __init__(self, **kwargs):
        """Initialize consistency checker."""
        self.logger = get_logger("consistency_checker")
        self.consistency_metrics = ConsistencyMetrics(**kwargs)
        self.config = kwargs
    
    def check_logical_consistency(
        self,
        knowledge_graph: Any
    ) -> bool:
        """
        Check logical consistency.
        
        Args:
            knowledge_graph: Knowledge graph instance
            
        Returns:
            True if consistent, False otherwise
        """
        score = self.consistency_metrics.calculate_logical_consistency(knowledge_graph)
        return score >= 0.8
    
    def check_temporal_consistency(
        self,
        knowledge_graph: Any
    ) -> bool:
        """
        Check temporal consistency.
        
        Args:
            knowledge_graph: Knowledge graph instance
            
        Returns:
            True if consistent, False otherwise
        """
        score = self.consistency_metrics.calculate_temporal_consistency(knowledge_graph)
        return score >= 0.8
    
    def check_hierarchical_consistency(
        self,
        knowledge_graph: Any
    ) -> bool:
        """
        Check hierarchical consistency.
        
        Args:
            knowledge_graph: Knowledge graph instance
            
        Returns:
            True if consistent, False otherwise
        """
        score = self.consistency_metrics.calculate_hierarchical_consistency(knowledge_graph)
        return score >= 0.8


class CompletenessValidator:
    """
    Completeness validator.
    
    Validates completeness of Knowledge Graph.
    """
    
    def __init__(self, **kwargs):
        """Initialize completeness validator."""
        self.logger = get_logger("completeness_validator")
        self.completeness_metrics = CompletenessMetrics(**kwargs)
        self.config = kwargs
    
    def validate_entity_completeness(
        self,
        entities: List[Dict[str, Any]],
        schema: Dict[str, Any]
    ) -> bool:
        """
        Validate entity completeness.
        
        Args:
            entities: List of entities
            schema: Schema definition
            
        Returns:
            True if complete, False otherwise
        """
        score = self.completeness_metrics.calculate_entity_completeness(entities, schema)
        return score >= 0.8
    
    def validate_relationship_completeness(
        self,
        relationships: List[Dict[str, Any]],
        schema: Dict[str, Any]
    ) -> bool:
        """
        Validate relationship completeness.
        
        Args:
            relationships: List of relationships
            schema: Schema definition
            
        Returns:
            True if complete, False otherwise
        """
        score = self.completeness_metrics.calculate_relationship_completeness(
            relationships,
            schema
        )
        return score >= 0.8
    
    def validate_property_completeness(
        self,
        properties: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> bool:
        """
        Validate property completeness.
        
        Args:
            properties: Properties dictionary
            schema: Schema definition
            
        Returns:
            True if complete, False otherwise
        """
        score = self.completeness_metrics.calculate_property_completeness(properties, schema)
        return score >= 0.8

