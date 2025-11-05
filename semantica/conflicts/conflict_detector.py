"""
Conflict Detector for Semantica framework.

Detects conflicts from multiple sources and tracks source
disagreements for compliance and investigation.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .source_tracker import SourceTracker, SourceReference
from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


class ConflictType(str, Enum):
    """Conflict type enumeration."""
    VALUE_CONFLICT = "value_conflict"
    TYPE_CONFLICT = "type_conflict"
    RELATIONSHIP_CONFLICT = "relationship_conflict"
    TEMPORAL_CONFLICT = "temporal_conflict"
    LOGICAL_CONFLICT = "logical_conflict"


@dataclass
class Conflict:
    """Conflict information."""
    conflict_id: str
    conflict_type: ConflictType
    entity_id: Optional[str] = None
    property_name: Optional[str] = None
    relationship_id: Optional[str] = None
    conflicting_values: List[Any] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    severity: str = "medium"  # low, medium, high, critical
    recommended_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictDetector:
    """
    Conflict detector for multi-source conflict identification.
    
    • Detects property value conflicts
    • Identifies relationship conflicts
    • Tracks source disagreements
    • Generates conflict reports
    • Provides investigation guides
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize conflict detector.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - track_provenance: Track source provenance (default: True)
                - conflict_fields: Fields to monitor for conflicts
                - confidence_threshold: Minimum confidence for conflicts (default: 0.7)
                - auto_resolve: Auto-resolve simple conflicts (default: False)
        """
        self.logger = get_logger("conflict_detector")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.source_tracker = SourceTracker()
        self.track_provenance = self.config.get("track_provenance", True)
        self.conflict_fields = self.config.get("conflict_fields", {})
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.auto_resolve = self.config.get("auto_resolve", False)
        
        self.detected_conflicts: Dict[str, Conflict] = {}
    
    def detect_value_conflicts(
        self,
        entities: List[Dict[str, Any]],
        property_name: str,
        entity_type: Optional[str] = None
    ) -> List[Conflict]:
        """
        Detect property value conflicts.
        
        Args:
            entities: List of entity dictionaries
            property_name: Property name to check
            entity_type: Optional entity type filter
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group entities by ID (same entity from different sources)
        entity_groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            if not entity_id:
                continue
            
            if entity_type and entity.get("type") != entity_type:
                continue
            
            if entity_id not in entity_groups:
                entity_groups[entity_id] = []
            entity_groups[entity_id].append(entity)
        
        # Check each entity group for conflicts
        for entity_id, entity_list in entity_groups.items():
            if len(entity_list) < 2:
                continue  # Need at least 2 sources to have conflict
            
            values = []
            sources = []
            
            for entity in entity_list:
                if property_name in entity:
                    value = entity[property_name]
                    values.append(value)
                    
                    # Track source if available
                    if self.track_provenance:
                        source_ref = SourceReference(
                            document=entity.get("source", "unknown"),
                            page=entity.get("page"),
                            section=entity.get("section"),
                            confidence=entity.get("confidence", 1.0),
                            metadata=entity.get("metadata", {})
                        )
                        self.source_tracker.track_property_source(
                            entity_id,
                            property_name,
                            value,
                            source_ref
                        )
                        sources.append({
                            "document": source_ref.document,
                            "page": source_ref.page,
                            "confidence": source_ref.confidence
                        })
            
            # Check for value conflicts
            unique_values = list(set(str(v) for v in values if v is not None))
            
            if len(unique_values) > 1:
                conflict = Conflict(
                    conflict_id=f"{entity_id}_{property_name}_conflict",
                    conflict_type=ConflictType.VALUE_CONFLICT,
                    entity_id=entity_id,
                    property_name=property_name,
                    conflicting_values=values,
                    sources=sources,
                    confidence=self._calculate_conflict_confidence(values, sources),
                    severity=self._calculate_severity(property_name, values),
                    recommended_action=self._recommend_action(property_name, values)
                )
                conflicts.append(conflict)
                self.detected_conflicts[conflict.conflict_id] = conflict
                
                self.logger.warning(
                    f"Value conflict detected: {entity_id}.{property_name} "
                    f"has conflicting values: {unique_values}"
                )
        
        return conflicts
    
    def detect_property_conflicts(
        self,
        entities: List[Dict[str, Any]],
        property_name: str
    ) -> List[Conflict]:
        """
        Detect conflicts in a specific property across entities.
        
        Args:
            entities: List of entity dictionaries
            property_name: Property name to check
            
        Returns:
            List of detected conflicts
        """
        return self.detect_value_conflicts(entities, property_name)
    
    def detect_relationship_conflicts(
        self,
        relationships: List[Dict[str, Any]]
    ) -> List[Conflict]:
        """
        Detect conflicts in relationships.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        
        # Group relationships by ID
        rel_groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for rel in relationships:
            rel_id = rel.get("id") or f"{rel.get('source_id')}_{rel.get('target_id')}_{rel.get('type')}"
            
            if rel_id not in rel_groups:
                rel_groups[rel_id] = []
            rel_groups[rel_id].append(rel)
        
        # Check for conflicts
        for rel_id, rel_list in rel_groups.items():
            if len(rel_list) < 2:
                continue
            
            # Check for conflicting properties
            for prop_name in ["type", "properties", "confidence"]:
                values = [r.get(prop_name) for r in rel_list if prop_name in r]
                unique_values = list(set(str(v) for v in values if v is not None))
                
                if len(unique_values) > 1:
                    conflict = Conflict(
                        conflict_id=f"{rel_id}_{prop_name}_conflict",
                        conflict_type=ConflictType.RELATIONSHIP_CONFLICT,
                        relationship_id=rel_id,
                        property_name=prop_name,
                        conflicting_values=values,
                        confidence=0.8,
                        severity="medium",
                        recommended_action="Review relationship definition"
                    )
                    conflicts.append(conflict)
                    self.detected_conflicts[conflict.conflict_id] = conflict
        
        return conflicts
    
    def detect_entity_conflicts(
        self,
        entities: List[Dict[str, Any]],
        entity_type: Optional[str] = None
    ) -> List[Conflict]:
        """
        Detect conflicts across all monitored properties for entities.
        
        Args:
            entities: List of entity dictionaries
            entity_type: Optional entity type filter
            
        Returns:
            List of detected conflicts
        """
        all_conflicts = []
        
        # Get fields to monitor
        if entity_type and entity_type in self.conflict_fields:
            fields_to_check = self.conflict_fields[entity_type]
        else:
            # Check all common properties
            fields_to_check = set()
            for entity in entities:
                fields_to_check.update(entity.keys())
            fields_to_check = list(fields_to_check - {"id", "entity_id", "type", "source", "metadata"})
        
        # Check each field
        for field in fields_to_check:
            conflicts = self.detect_value_conflicts(entities, field, entity_type)
            all_conflicts.extend(conflicts)
        
        return all_conflicts
    
    def _calculate_conflict_confidence(
        self,
        values: List[Any],
        sources: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for conflict."""
        if not sources:
            return 0.5
        
        # Average confidence of sources
        avg_confidence = sum(s.get("confidence", 0.5) for s in sources) / len(sources)
        
        # Higher confidence if values are very different
        value_diversity = len(set(str(v) for v in values)) / len(values) if values else 0
        
        return min(1.0, avg_confidence * (1 + value_diversity))
    
    def _calculate_severity(
        self,
        property_name: str,
        values: List[Any]
    ) -> str:
        """Calculate conflict severity."""
        # Critical fields
        critical_fields = ["id", "name", "type", "founded_year", "revenue"]
        if property_name.lower() in critical_fields:
            return "critical"
        
        # High severity for numeric conflicts with large differences
        try:
            numeric_values = [float(v) for v in values if v is not None]
            if numeric_values:
                value_range = max(numeric_values) - min(numeric_values)
                if value_range > 1000:  # Large difference
                    return "high"
        except (ValueError, TypeError):
            pass
        
        return "medium"
    
    def _recommend_action(
        self,
        property_name: str,
        values: List[Any]
    ) -> str:
        """Recommend action for conflict."""
        if len(set(values)) == 2:
            return "Compare source documents and use most recent or authoritative source"
        else:
            return "Multiple conflicting values detected. Manual review recommended."
    
    def get_conflict_report(self) -> Dict[str, Any]:
        """
        Generate conflict report.
        
        Returns:
            Conflict report dictionary
        """
        report = {
            "total_conflicts": len(self.detected_conflicts),
            "by_type": {},
            "by_severity": {},
            "conflicts": []
        }
        
        for conflict in self.detected_conflicts.values():
            # Count by type
            conflict_type = conflict.conflict_type.value
            report["by_type"][conflict_type] = report["by_type"].get(conflict_type, 0) + 1
            
            # Count by severity
            report["by_severity"][conflict.severity] = report["by_severity"].get(conflict.severity, 0) + 1
            
            # Add conflict details
            conflict_data = {
                "conflict_id": conflict.conflict_id,
                "type": conflict_type,
                "entity_id": conflict.entity_id,
                "property_name": conflict.property_name,
                "severity": conflict.severity,
                "conflicting_values": conflict.conflicting_values,
                "sources": conflict.sources,
                "recommended_action": conflict.recommended_action
            }
            report["conflicts"].append(conflict_data)
        
        return report
    
    def clear_conflicts(self) -> None:
        """Clear all detected conflicts."""
        self.detected_conflicts.clear()
        self.logger.info("Cleared all detected conflicts")
