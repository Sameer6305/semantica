"""
Graph validator for Semantica framework.

This module provides consistency validation and quality checking
for knowledge graphs.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..utils.logging import get_logger


@dataclass
class ValidationResult:
    """Validation result."""
    
    valid: bool
    errors: List[str]
    warnings: List[str]


class GraphValidator:
    """
    Graph validator.
    
    Validates consistency and quality of knowledge graphs.
    """
    
    def __init__(self, **config):
        """Initialize graph validator."""
        self.logger = get_logger("graph_validator")
        self.config = config
    
    def validate(self, knowledge_graph: Any) -> ValidationResult:
        """
        Validate knowledge graph.
        
        Args:
            knowledge_graph: Knowledge graph instance
            
        Returns:
            Validation result
        """
        self.logger.info("Validating knowledge graph")
        
        errors = []
        warnings = []
        
        # Extract entities and relationships
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
        
        # Validate entities
        entity_ids = set()
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            if not entity_id:
                errors.append("Entity missing required 'id' field")
                continue
            
            if entity_id in entity_ids:
                errors.append(f"Duplicate entity ID: {entity_id}")
            else:
                entity_ids.add(entity_id)
            
            if not entity.get("type"):
                warnings.append(f"Entity {entity_id} missing 'type' field")
        
        # Validate relationships
        for rel in relationships:
            source = rel.get("source") or rel.get("subject")
            target = rel.get("target") or rel.get("object")
            rel_type = rel.get("type") or rel.get("predicate")
            
            if not source:
                errors.append("Relationship missing 'source' field")
            elif source not in entity_ids:
                warnings.append(f"Relationship references unknown source entity: {source}")
            
            if not target:
                errors.append("Relationship missing 'target' field")
            elif target not in entity_ids:
                warnings.append(f"Relationship references unknown target entity: {target}")
            
            if not rel_type:
                errors.append("Relationship missing 'type' field")
        
        # Check for orphaned entities (entities with no relationships)
        entity_has_relationships = set()
        for rel in relationships:
            source = rel.get("source") or rel.get("subject")
            target = rel.get("target") or rel.get("object")
            if source:
                entity_has_relationships.add(source)
            if target:
                entity_has_relationships.add(target)
        
        orphaned = entity_ids - entity_has_relationships
        if orphaned:
            warnings.append(f"Found {len(orphaned)} orphaned entities (no relationships)")
        
        valid = len(errors) == 0
        
        self.logger.info(f"Validation complete: {len(errors)} errors, {len(warnings)} warnings")
        
        return ValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings
        )
    
    def check_consistency(self, knowledge_graph: Any) -> bool:
        """
        Check graph consistency.
        
        Args:
            knowledge_graph: Knowledge graph instance
            
        Returns:
            True if consistent, False otherwise
        """
        self.logger.info("Checking graph consistency")
        
        # Use validation to check consistency
        validation_result = self.validate(knowledge_graph)
        
        # Check for logical inconsistencies
        # Extract entities and relationships
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
        
        # Check for type consistency
        entity_types = {}
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            entity_type = entity.get("type")
            if entity_id and entity_type:
                if entity_id in entity_types and entity_types[entity_id] != entity_type:
                    self.logger.warning(f"Inconsistent type for entity {entity_id}")
                    return False
                entity_types[entity_id] = entity_type
        
        # Check for circular relationships
        # Build adjacency list
        adjacency = {}
        for rel in relationships:
            source = rel.get("source") or rel.get("subject")
            target = rel.get("target") or rel.get("object")
            if source and target:
                if source not in adjacency:
                    adjacency[source] = []
                adjacency[source].append(target)
        
        # Simple cycle detection (DFS)
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        visited = set()
        for node in adjacency:
            if node not in visited:
                if has_cycle(node, visited, set()):
                    self.logger.warning("Found circular relationship")
                    return False
        
        return validation_result.valid
