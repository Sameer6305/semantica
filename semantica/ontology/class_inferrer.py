"""
Class inference for Semantica framework.

This module provides automatic class discovery and inference
from extracted entities and data patterns.
"""

from typing import Any, Dict, List, Optional, Set
from collections import Counter, defaultdict

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from .naming_conventions import NamingConventions


class ClassInferrer:
    """
    Class inference engine for ontology generation.
    
    • Automatic class discovery from entities
    • Pattern-based class inference
    • Hierarchical class structure building
    • Class validation and consistency checking
    • Multi-domain class support
    • Performance optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize class inferrer.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - min_occurrences: Minimum occurrences for class inference (default: 2)
                - similarity_threshold: Similarity threshold for class merging (default: 0.8)
        """
        self.logger = get_logger("class_inferrer")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.naming_conventions = NamingConventions(**self.config)
        self.min_occurrences = self.config.get("min_occurrences", 2)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
    
    def infer_classes(
        self,
        entities: List[Dict[str, Any]],
        **options
    ) -> List[Dict[str, Any]]:
        """
        Infer classes from entities.
        
        Args:
            entities: List of entity dictionaries
            **options: Additional options
        
        Returns:
            List of inferred class definitions
        """
        # Group entities by type
        entity_types = defaultdict(list)
        for entity in entities:
            entity_type = entity.get("type") or entity.get("entity_type", "Entity")
            entity_types[entity_type].append(entity)
        
        # Infer classes from entity types
        classes = []
        for entity_type, type_entities in entity_types.items():
            if len(type_entities) >= self.min_occurrences:
                class_def = self._create_class_from_entities(entity_type, type_entities, **options)
                classes.append(class_def)
        
        # Build hierarchy
        if options.get("build_hierarchy", True):
            classes = self.build_class_hierarchy(classes, **options)
        
        return classes
    
    def build_class_hierarchy(
        self,
        classes: List[Dict[str, Any]],
        **options
    ) -> List[Dict[str, Any]]:
        """
        Build class hierarchy from classes.
        
        Args:
            classes: List of class definitions
            **options: Additional options
        
        Returns:
            Classes with hierarchy information
        """
        # Create class map
        class_map = {cls["name"]: cls for cls in classes}
        
        # Infer parent-child relationships
        for cls in classes:
            if "parent" not in cls:
                # Try to find parent based on naming patterns
                parent = self._find_parent_class(cls["name"], class_map)
                if parent:
                    cls["subClassOf"] = parent
                    cls["parent"] = parent
        
        return classes
    
    def _create_class_from_entities(
        self,
        entity_type: str,
        entities: List[Dict[str, Any]],
        **options
    ) -> Dict[str, Any]:
        """Create class definition from entities."""
        # Normalize class name
        class_name = self.naming_conventions.normalize_class_name(entity_type)
        
        # Extract common properties
        properties = self._extract_common_properties(entities)
        
        # Create class definition
        class_def = {
            "name": class_name,
            "uri": options.get("namespace_manager", None).generate_class_iri(class_name) if options.get("namespace_manager") else None,
            "label": class_name,
            "comment": f"Class representing {class_name.lower()} entities",
            "properties": properties,
            "entity_count": len(entities),
            "metadata": {
                "inferred_from": entity_type,
                "inferred_count": len(entities)
            }
        }
        
        return class_def
    
    def _extract_common_properties(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Extract common properties from entities."""
        # Count property occurrences
        property_counts = Counter()
        
        for entity in entities:
            # Count properties in entity
            for key in entity.keys():
                if key not in ["id", "type", "entity_type", "text", "label", "confidence"]:
                    property_counts[key] += 1
        
        # Return properties that appear in at least 50% of entities
        threshold = len(entities) * 0.5
        common_properties = [
            prop for prop, count in property_counts.items()
            if count >= threshold
        ]
        
        return common_properties
    
    def _find_parent_class(
        self,
        class_name: str,
        class_map: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """Find parent class based on naming patterns."""
        # Simple heuristic: look for more general class names
        words = class_name.split()
        
        # Try to find parent by removing words
        for i in range(len(words) - 1, 0, -1):
            parent_candidate = ''.join(words[:i])
            if parent_candidate in class_map:
                return parent_candidate
        
        # Check for common parent classes
        common_parents = ["Entity", "Thing", "Resource"]
        for parent in common_parents:
            if parent in class_map:
                return parent
        
        return None
    
    def validate_classes(
        self,
        classes: List[Dict[str, Any]],
        **criteria
    ) -> Dict[str, Any]:
        """
        Validate inferred classes.
        
        Args:
            classes: List of class definitions
            **criteria: Validation criteria
        
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check for duplicate class names
        class_names = [cls["name"] for cls in classes]
        duplicates = [name for name, count in Counter(class_names).items() if count > 1]
        
        if duplicates:
            errors.append(f"Duplicate class names found: {duplicates}")
        
        # Validate naming conventions
        for cls in classes:
            is_valid, suggestion = self.naming_conventions.validate_class_name(cls["name"])
            if not is_valid:
                warnings.append(f"Class '{cls['name']}' doesn't follow conventions. Suggested: {suggestion}")
        
        # Check for circular hierarchies
        hierarchy_errors = self._check_circular_hierarchy(classes)
        errors.extend(hierarchy_errors)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _check_circular_hierarchy(self, classes: List[Dict[str, Any]]) -> List[str]:
        """Check for circular inheritance."""
        errors = []
        
        # Build parent map
        parent_map = {}
        for cls in classes:
            if "subClassOf" in cls or "parent" in cls:
                parent = cls.get("subClassOf") or cls.get("parent")
                if parent:
                    parent_map[cls["name"]] = parent
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            if node in parent_map:
                parent = parent_map[node]
                if parent in rec_stack:
                    return True
                if parent not in visited and has_cycle(parent):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for cls in classes:
            if cls["name"] not in visited:
                if has_cycle(cls["name"]):
                    errors.append(f"Circular hierarchy detected involving class: {cls['name']}")
        
        return errors
