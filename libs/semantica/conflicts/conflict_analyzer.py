"""
Conflict Analyzer for Semantica framework.

Analyzes patterns in conflicts, identifies conflict types, and
provides insights into conflict sources and trends.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime

from .conflict_detector import Conflict, ConflictType
from ..utils.logging import get_logger


@dataclass
class ConflictPattern:
    """Conflict pattern analysis."""
    pattern_type: str
    frequency: int
    affected_entities: List[str] = field(default_factory=list)
    affected_properties: List[str] = field(default_factory=list)
    common_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictAnalyzer:
    """
    Conflict analyzer for pattern identification and insights.
    
    • Analyzes conflict patterns and trends
    • Classifies conflict types
    • Identifies high-conflict areas
    • Generates conflict statistics
    • Provides conflict insights and recommendations
    • Supports conflict prevention strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize conflict analyzer.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("conflict_analyzer")
        self.config = config or {}
        self.config.update(kwargs)
    
    def analyze_conflicts(
        self,
        conflicts: List[Conflict]
    ) -> Dict[str, Any]:
        """
        Analyze conflicts and generate insights.
        
        Args:
            conflicts: List of conflicts to analyze
            
        Returns:
            Analysis results dictionary
        """
        analysis = {
            "total_conflicts": len(conflicts),
            "by_type": self._analyze_by_type(conflicts),
            "by_severity": self._analyze_by_severity(conflicts),
            "by_entity": self._analyze_by_entity(conflicts),
            "by_property": self._analyze_by_property(conflicts),
            "patterns": self._identify_patterns(conflicts),
            "recommendations": self._generate_recommendations(conflicts),
            "statistics": self._calculate_statistics(conflicts)
        }
        
        return analysis
    
    def _analyze_by_type(self, conflicts: List[Conflict]) -> Dict[str, Any]:
        """Analyze conflicts by type."""
        by_type = defaultdict(int)
        type_details = defaultdict(list)
        
        for conflict in conflicts:
            conflict_type = conflict.conflict_type.value
            by_type[conflict_type] += 1
            type_details[conflict_type].append({
                "conflict_id": conflict.conflict_id,
                "entity_id": conflict.entity_id,
                "severity": conflict.severity
            })
        
        return {
            "counts": dict(by_type),
            "details": dict(type_details)
        }
    
    def _analyze_by_severity(self, conflicts: List[Conflict]) -> Dict[str, Any]:
        """Analyze conflicts by severity."""
        by_severity = defaultdict(int)
        severity_details = defaultdict(list)
        
        for conflict in conflicts:
            severity = conflict.severity
            by_severity[severity] += 1
            severity_details[severity].append({
                "conflict_id": conflict.conflict_id,
                "type": conflict.conflict_type.value,
                "entity_id": conflict.entity_id,
                "property_name": conflict.property_name
            })
        
        return {
            "counts": dict(by_severity),
            "details": dict(severity_details)
        }
    
    def _analyze_by_entity(self, conflicts: List[Conflict]) -> Dict[str, Any]:
        """Analyze conflicts by entity."""
        by_entity = defaultdict(int)
        entity_conflicts = defaultdict(list)
        
        for conflict in conflicts:
            if conflict.entity_id:
                by_entity[conflict.entity_id] += 1
                entity_conflicts[conflict.entity_id].append({
                    "conflict_id": conflict.conflict_id,
                    "property_name": conflict.property_name,
                    "type": conflict.conflict_type.value,
                    "severity": conflict.severity
                })
        
        # Get top entities with most conflicts
        top_entities = sorted(
            by_entity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "counts": dict(by_entity),
            "top_entities": [{"entity_id": eid, "conflict_count": count} 
                            for eid, count in top_entities],
            "details": dict(entity_conflicts)
        }
    
    def _analyze_by_property(self, conflicts: List[Conflict]) -> Dict[str, Any]:
        """Analyze conflicts by property."""
        by_property = defaultdict(int)
        property_conflicts = defaultdict(list)
        
        for conflict in conflicts:
            if conflict.property_name:
                by_property[conflict.property_name] += 1
                property_conflicts[conflict.property_name].append({
                    "conflict_id": conflict.conflict_id,
                    "entity_id": conflict.entity_id,
                    "type": conflict.conflict_type.value,
                    "severity": conflict.severity
                })
        
        # Get top properties with most conflicts
        top_properties = sorted(
            by_property.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "counts": dict(by_property),
            "top_properties": [{"property_name": prop, "conflict_count": count}
                              for prop, count in top_properties],
            "details": dict(property_conflicts)
        }
    
    def _identify_patterns(self, conflicts: List[Conflict]) -> List[ConflictPattern]:
        """Identify conflict patterns."""
        patterns = []
        
        # Pattern 1: Same property conflicts across multiple entities
        property_entity_map = defaultdict(set)
        for conflict in conflicts:
            if conflict.property_name and conflict.entity_id:
                property_entity_map[conflict.property_name].add(conflict.entity_id)
        
        for prop, entities in property_entity_map.items():
            if len(entities) > 3:  # Multiple entities affected
                patterns.append(ConflictPattern(
                    pattern_type="widespread_property_conflict",
                    frequency=len(entities),
                    affected_properties=[prop],
                    affected_entities=list(entities),
                    metadata={"description": f"Property '{prop}' has conflicts across {len(entities)} entities"}
                ))
        
        # Pattern 2: Same source appears in multiple conflicts
        source_conflict_map = defaultdict(int)
        for conflict in conflicts:
            for source in conflict.sources:
                source_conflict_map[source.get("document", "unknown")] += 1
        
        problematic_sources = [
            (source, count) for source, count in source_conflict_map.items()
            if count > 5
        ]
        
        for source, count in problematic_sources:
            patterns.append(ConflictPattern(
                pattern_type="problematic_source",
                frequency=count,
                common_sources=[source],
                metadata={"description": f"Source '{source}' appears in {count} conflicts"}
            ))
        
        # Pattern 3: High-severity conflicts cluster
        critical_conflicts = [c for c in conflicts if c.severity == "critical"]
        if len(critical_conflicts) > 5:
            patterns.append(ConflictPattern(
                pattern_type="critical_conflict_cluster",
                frequency=len(critical_conflicts),
                metadata={"description": f"{len(critical_conflicts)} critical conflicts detected"}
            ))
        
        return patterns
    
    def _generate_recommendations(self, conflicts: List[Conflict]) -> List[str]:
        """Generate recommendations based on conflict analysis."""
        recommendations = []
        
        if not conflicts:
            return ["No conflicts detected. Continue monitoring."]
        
        # Analyze patterns
        by_property = self._analyze_by_property(conflicts)
        by_entity = self._analyze_by_entity(conflicts)
        patterns = self._identify_patterns(conflicts)
        
        # Recommendation 1: High-conflict properties
        top_properties = by_property.get("top_properties", [])
        if top_properties:
            top_prop = top_properties[0]
            recommendations.append(
                f"Property '{top_prop['property_name']}' has {top_prop['conflict_count']} conflicts. "
                f"Consider implementing stricter validation or source verification."
            )
        
        # Recommendation 2: Problematic sources
        problematic_source_patterns = [p for p in patterns if p.pattern_type == "problematic_source"]
        if problematic_source_patterns:
            for pattern in problematic_source_patterns:
                recommendations.append(
                    f"Source '{pattern.common_sources[0]}' appears in multiple conflicts. "
                    f"Review source quality and credibility."
                )
        
        # Recommendation 3: High-conflict entities
        top_entities = by_entity.get("top_entities", [])
        if top_entities:
            top_entity = top_entities[0]
            recommendations.append(
                f"Entity '{top_entity['entity_id']}' has {top_entity['conflict_count']} conflicts. "
                f"Review entity data sources and merge strategy."
            )
        
        # Recommendation 4: Critical conflicts
        critical_count = len([c for c in conflicts if c.severity == "critical"])
        if critical_count > 0:
            recommendations.append(
                f"{critical_count} critical conflicts detected. Immediate review required."
            )
        
        if not recommendations:
            recommendations.append("Monitor conflicts and review patterns regularly.")
        
        return recommendations
    
    def _calculate_statistics(self, conflicts: List[Conflict]) -> Dict[str, Any]:
        """Calculate conflict statistics."""
        if not conflicts:
            return {
                "total_conflicts": 0,
                "average_confidence": 0.0,
                "conflict_rate": 0.0
            }
        
        total_conflicts = len(conflicts)
        avg_confidence = sum(c.confidence for c in conflicts) / total_conflicts
        
        # Count unique entities and properties
        unique_entities = len(set(c.entity_id for c in conflicts if c.entity_id))
        unique_properties = len(set(c.property_name for c in conflicts if c.property_name))
        
        # Count by source
        source_counts = Counter()
        for conflict in conflicts:
            for source in conflict.sources:
                source_counts[source.get("document", "unknown")] += 1
        
        return {
            "total_conflicts": total_conflicts,
            "average_confidence": avg_confidence,
            "unique_entities_affected": unique_entities,
            "unique_properties_affected": unique_properties,
            "top_sources": dict(source_counts.most_common(5)),
            "conflict_rate": total_conflicts / max(unique_entities, 1)  # Conflicts per entity
        }
    
    def generate_insights_report(self, conflicts: List[Conflict]) -> Dict[str, Any]:
        """
        Generate comprehensive insights report.
        
        Args:
            conflicts: List of conflicts
            
        Returns:
            Insights report dictionary
        """
        analysis = self.analyze_conflicts(conflicts)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_conflicts": analysis["total_conflicts"],
                "critical_count": analysis["by_severity"]["counts"].get("critical", 0),
                "high_count": analysis["by_severity"]["counts"].get("high", 0),
                "medium_count": analysis["by_severity"]["counts"].get("medium", 0),
                "low_count": analysis["by_severity"]["counts"].get("low", 0)
            },
            "analysis": analysis,
            "insights": {
                "most_conflict_prone_properties": [
                    p["property_name"] for p in analysis["by_property"]["top_properties"][:5]
                ],
                "most_conflict_prone_entities": [
                    e["entity_id"] for e in analysis["by_entity"]["top_entities"][:5]
                ],
                "patterns_detected": len(analysis["patterns"]),
                "recommendations": analysis["recommendations"]
            }
        }
        
        return report
