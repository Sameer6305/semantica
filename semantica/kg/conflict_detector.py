"""
Conflict Detection Module

This module provides comprehensive conflict identification and resolution
capabilities for the Semantica framework, enabling detection of inconsistencies
in knowledge graphs.

Key Features:
    - Value conflict detection (conflicting property values for same entity)
    - Relationship conflict detection (conflicting relationship properties)
    - Conflict resolution with multiple strategies
    - Source tracking for conflicts
    - Conflict categorization and prioritization

Main Classes:
    - ConflictDetector: Main conflict detection and resolution engine

Example Usage:
    >>> from semantica.kg import ConflictDetector
    >>> detector = ConflictDetector()
    >>> conflicts = detector.detect_conflicts(knowledge_graph)
    >>> resolution = detector.resolve_conflicts(conflicts)

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from ..conflicts.conflict_detector import ConflictDetector as BaseConflictDetector, Conflict
from ..conflicts.conflict_resolver import ConflictResolver


class ConflictDetector:
    """
    Conflict detection and resolution engine.
    
    This class identifies conflicts and inconsistencies in knowledge graphs,
    including value conflicts (same entity with different property values)
    and relationship conflicts. Provides conflict resolution capabilities.
    
    Features:
        - Entity property value conflict detection
        - Relationship property conflict detection
        - Conflict resolution with configurable strategies
        - Source tracking for conflict origins
    
    Example Usage:
        >>> detector = ConflictDetector()
        >>> conflicts = detector.detect_conflicts(knowledge_graph)
        >>> resolution = detector.resolve_conflicts(conflicts)
    """
    
    def __init__(self, **config):
        """
        Initialize conflict detector.
        
        Sets up the detector with base conflict detection and resolution
        components from the conflicts module.
        
        Args:
            **config: Configuration options:
                - detection: Configuration for conflict detection (optional)
                - resolution: Configuration for conflict resolution (optional)
        """
        self.logger = get_logger("conflict_detector")
        self.config = config
        
        # Initialize progress tracker
        self.progress_tracker = get_progress_tracker()
        
        # Initialize conflict detection components
        self.base_detector = BaseConflictDetector(**config.get("detection", {}))
        self.resolver = ConflictResolver(**config.get("resolution", {}))
        
        self.logger.debug("Conflict detector initialized")
    
    def detect_conflicts(self, knowledge_graph: Any) -> List[Dict[str, Any]]:
        """
        Detect conflicts in knowledge graph.
        
        This method identifies conflicts in the knowledge graph, including:
        - Value conflicts: Same entity with different values for the same property
        - Relationship conflicts: Same relationship with conflicting properties
        
        Args:
            knowledge_graph: Knowledge graph instance (object with entities/relationships
                           attributes, or dict with "entities" and "relationships" keys)
            
        Returns:
            list: List of conflict dictionaries, each containing:
                - entity_id or relationship: Identifier of conflicted element
                - property: Property name with conflict
                - conflicting_values: List of conflicting values
                - type: Conflict type ("value_conflict" or "relationship_conflict")
                - sources: List of source identifiers for conflicting values
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
            
            self.progress_tracker.update_tracking(tracking_id, message="Detecting value conflicts...")
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
            
            self.progress_tracker.update_tracking(tracking_id, message="Detecting relationship conflicts...")
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
            self.progress_tracker.stop_tracking(tracking_id, status="completed",
                                               message=f"Detected {len(conflicts)} conflicts")
            return conflicts
            
        except Exception as e:
            self.progress_tracker.stop_tracking(tracking_id, status="failed", message=str(e))
            raise
    
    def resolve_conflicts(
        self,
        conflicts: List[Dict[str, Any]],
        strategy: str = "highest_confidence"
    ) -> Dict[str, Any]:
        """
        Resolve detected conflicts.
        
        This method attempts to resolve conflicts using the configured conflict
        resolver with the specified strategy. Converts conflict dictionaries
        to Conflict objects for resolution.
        
        Args:
            conflicts: List of conflict dictionaries from detect_conflicts()
            strategy: Resolution strategy to use (default: "highest_confidence")
            
        Returns:
            dict: Resolution results containing:
                - resolved: List of successfully resolved conflicts with resolutions
                - unresolved: List of conflicts that could not be resolved
                - total: Total number of conflicts
                - resolved_count: Number of resolved conflicts
                - unresolved_count: Number of unresolved conflicts
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
                    strategy=strategy
                )
                
                if resolution.resolved:
                    resolved.append({
                        "conflict": conflict,
                        "resolution": resolution.resolved_value,
                        "strategy": resolution.resolution_strategy
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
