"""
Conflict detector for Semantica framework.

This module provides conflict identification and resolution
for knowledge graph inconsistencies.
"""

from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from ..conflicts.conflict_detector import ConflictDetector as BaseConflictDetector, Conflict
from ..conflicts.conflict_resolver import ConflictResolver


class ConflictDetector:
    """
    Conflict detector.
    
    Identifies conflicts and inconsistencies in knowledge graphs.
    """
    
    def __init__(self, **config):
        """Initialize conflict detector."""
        self.logger = get_logger("conflict_detector")
        self.config = config
        
        # Initialize conflict detection components
        self.base_detector = BaseConflictDetector(**config.get("detection", {}))
        self.resolver = ConflictResolver(**config.get("resolution", {}))
    
    def detect_conflicts(self, knowledge_graph: Any) -> List[Dict[str, Any]]:
        """
        Detect conflicts in knowledge graph.
        
        Args:
            knowledge_graph: Knowledge graph instance
            
        Returns:
            List of detected conflicts
        """
        self.logger.info("Detecting conflicts in knowledge graph")
        
        # Extract entities and relationships from graph
        entities = []
        relationships = []
        
        if hasattr(knowledge_graph, "entities"):
            entities = knowledge_graph.entities
        elif hasattr(knowledge_graph, "get_entities"):
            entities = knowledge_graph.get_entities()
        elif isinstance(knowledge_graph, dict):
            entities = knowledge_graph.get("entities", [])
            relationships = knowledge_graph.get("relationships", [])
        
        if hasattr(knowledge_graph, "relationships"):
            relationships = knowledge_graph.relationships
        elif hasattr(knowledge_graph, "get_relationships"):
            relationships = knowledge_graph.get_relationships()
        
        conflicts = []
        
        # Detect value conflicts
        entity_properties = {}
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            if not entity_id:
                continue
            
            for prop_name, prop_value in entity.items():
                if prop_name in ["id", "entity_id", "type", "source"]:
                    continue
                
                if entity_id not in entity_properties:
                    entity_properties[entity_id] = {}
                
                if prop_name not in entity_properties[entity_id]:
                    entity_properties[entity_id][prop_name] = []
                
                entity_properties[entity_id][prop_name].append({
                    "value": prop_value,
                    "entity": entity
                })
        
        # Check for conflicts
        for entity_id, properties in entity_properties.items():
            for prop_name, values in properties.items():
                unique_values = {str(v["value"]) for v in values if v["value"] is not None}
                if len(unique_values) > 1:
                    conflicts.append({
                        "entity_id": entity_id,
                        "property": prop_name,
                        "conflicting_values": list(unique_values),
                        "type": "value_conflict",
                        "sources": [v["entity"].get("source", "unknown") for v in values]
                    })
        
        # Detect relationship conflicts
        relationship_map = {}
        for rel in relationships:
            source = rel.get("source") or rel.get("subject")
            target = rel.get("target") or rel.get("object")
            rel_type = rel.get("type") or rel.get("predicate")
            
            key = f"{source}::{rel_type}::{target}"
            if key not in relationship_map:
                relationship_map[key] = []
            relationship_map[key].append(rel)
        
        # Check for relationship conflicts
        for key, rels in relationship_map.items():
            if len(rels) > 1:
                # Check for conflicting properties
                properties = {}
                for rel in rels:
                    for prop_name, prop_value in rel.items():
                        if prop_name in ["source", "target", "subject", "object", "type", "predicate"]:
                            continue
                        if prop_name not in properties:
                            properties[prop_name] = []
                        properties[prop_name].append(prop_value)
                
                for prop_name, values in properties.items():
                    unique_values = {str(v) for v in values if v is not None}
                    if len(unique_values) > 1:
                        conflicts.append({
                            "relationship": key,
                            "property": prop_name,
                            "conflicting_values": list(unique_values),
                            "type": "relationship_conflict",
                            "sources": [rel.get("source", "unknown") for rel in rels]
                        })
        
        self.logger.info(f"Detected {len(conflicts)} conflicts")
        return conflicts
    
    def resolve_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve detected conflicts.
        
        Args:
            conflicts: List of conflicts
            
        Returns:
            Resolution results
        """
        self.logger.info(f"Resolving {len(conflicts)} conflicts")
        
        resolved = []
        unresolved = []
        
        for conflict in conflicts:
            try:
                # Convert to Conflict object for resolver
                conflict_obj = Conflict(
                    conflict_id=conflict.get("entity_id") or conflict.get("relationship", "unknown"),
                    conflict_type=conflict.get("type", "value_conflict"),
                    entity_id=conflict.get("entity_id"),
                    property_name=conflict.get("property"),
                    conflicting_values=conflict.get("conflicting_values", []),
                    sources=[{"source": s} for s in conflict.get("sources", [])]
                )
                
                # Resolve conflict
                resolution = self.resolver.resolve_conflict(
                    conflict_obj,
                    strategy="highest_confidence"
                )
                
                if resolution.success:
                    resolved.append({
                        "conflict": conflict,
                        "resolution": resolution.resolved_value,
                        "strategy": resolution.strategy
                    })
                else:
                    unresolved.append(conflict)
                    
            except Exception as e:
                self.logger.error(f"Error resolving conflict: {e}")
                unresolved.append(conflict)
        
        return {
            "resolved": resolved,
            "unresolved": unresolved,
            "total": len(conflicts),
            "resolved_count": len(resolved),
            "unresolved_count": len(unresolved)
        }
