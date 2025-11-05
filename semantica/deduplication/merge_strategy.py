"""
Merge Strategy Manager

Manages different strategies for merging duplicate entities
including property merging rules and relationship preservation.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


class MergeStrategy(Enum):
    """Merge strategy types."""
    
    KEEP_FIRST = "keep_first"
    KEEP_LAST = "keep_last"
    KEEP_MOST_COMPLETE = "keep_most_complete"
    KEEP_HIGHEST_CONFIDENCE = "keep_highest_confidence"
    MERGE_ALL = "merge_all"
    CUSTOM = "custom"


@dataclass
class PropertyMergeRule:
    """Rule for merging properties."""
    
    property_name: str
    strategy: MergeStrategy
    conflict_resolution: Optional[Callable] = None
    priority: int = 0


@dataclass
class MergeResult:
    """Result of merge operation."""
    
    merged_entity: Dict[str, Any]
    merged_entities: List[Dict[str, Any]]
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MergeStrategyManager:
    """
    Merge strategy management engine.
    
    • Manages property merging strategies
    • Preserves relationships during merge
    • Resolves merge conflicts
    • Makes confidence-based merge decisions
    • Validates merge quality
    • Supports custom merge strategies
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize merge strategy manager."""
        self.logger = get_logger("merge_strategy_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.default_strategy = MergeStrategy(config.get("default_strategy", "keep_most_complete"))
        self.property_rules: Dict[str, PropertyMergeRule] = {}
        self.custom_strategies: Dict[str, Callable] = {}
    
    def add_property_rule(
        self,
        property_name: str,
        strategy: MergeStrategy,
        conflict_resolution: Optional[Callable] = None,
        priority: int = 0
    ) -> None:
        """
        Add property merge rule.
        
        Args:
            property_name: Property name
            strategy: Merge strategy
            conflict_resolution: Custom conflict resolution function
            priority: Rule priority
        """
        rule = PropertyMergeRule(
            property_name=property_name,
            strategy=strategy,
            conflict_resolution=conflict_resolution,
            priority=priority
        )
        
        self.property_rules[property_name] = rule
    
    def merge_entities(
        self,
        entities: List[Dict[str, Any]],
        strategy: Optional[MergeStrategy] = None,
        **options
    ) -> MergeResult:
        """
        Merge entities using specified strategy.
        
        Args:
            entities: List of entities to merge
            strategy: Merge strategy (uses default if None)
            **options: Merge options
            
        Returns:
            MergeResult with merged entity
        """
        if not entities:
            raise ValidationError("No entities to merge")
        
        if len(entities) == 1:
            return MergeResult(
                merged_entity=entities[0],
                merged_entities=entities
            )
        
        strategy = strategy or self.default_strategy
        
        # Select base entity
        base_entity = self._select_base_entity(entities, strategy)
        
        # Merge properties
        merged_properties, property_conflicts = self._merge_properties(
            entities,
            base_entity,
            strategy
        )
        
        # Merge relationships
        merged_relationships = self._merge_relationships(entities, base_entity)
        
        # Build merged entity
        merged_entity = {
            "id": base_entity.get("id"),
            "name": base_entity.get("name"),
            "type": base_entity.get("type"),
            "properties": merged_properties,
            "relationships": merged_relationships,
            "metadata": self._merge_metadata(entities, base_entity),
            "merged_from": [e.get("id") for e in entities if e.get("id")],
            "merge_strategy": strategy.value
        }
        
        return MergeResult(
            merged_entity=merged_entity,
            merged_entities=entities,
            conflicts=property_conflicts,
            metadata={"strategy": strategy.value}
        )
    
    def _select_base_entity(
        self,
        entities: List[Dict[str, Any]],
        strategy: MergeStrategy
    ) -> Dict[str, Any]:
        """Select base entity for merge."""
        if strategy == MergeStrategy.KEEP_FIRST:
            return entities[0]
        elif strategy == MergeStrategy.KEEP_LAST:
            return entities[-1]
        elif strategy == MergeStrategy.KEEP_MOST_COMPLETE:
            return max(
                entities,
                key=lambda e: len(e.get("properties", {})) + len(e.get("relationships", []))
            )
        elif strategy == MergeStrategy.KEEP_HIGHEST_CONFIDENCE:
            return max(
                entities,
                key=lambda e: e.get("confidence", 0.0)
            )
        else:
            return entities[0]
    
    def _merge_properties(
        self,
        entities: List[Dict[str, Any]],
        base_entity: Dict[str, Any],
        strategy: MergeStrategy
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Merge properties from all entities."""
        merged_properties = base_entity.get("properties", {}).copy()
        conflicts = []
        
        for entity in entities:
            if entity == base_entity:
                continue
            
            entity_props = entity.get("properties", {})
            
            for prop_name, prop_value in entity_props.items():
                if prop_name not in merged_properties:
                    # New property, add it
                    merged_properties[prop_name] = prop_value
                elif merged_properties[prop_name] != prop_value:
                    # Conflict - resolve using rule or strategy
                    conflict = self._resolve_property_conflict(
                        prop_name,
                        merged_properties[prop_name],
                        prop_value,
                        strategy
                    )
                    
                    if conflict.get("resolved"):
                        merged_properties[prop_name] = conflict["value"]
                    else:
                        conflicts.append({
                            "property": prop_name,
                            "values": [merged_properties[prop_name], prop_value],
                            "resolution": conflict.get("resolution")
                        })
        
        return merged_properties, conflicts
    
    def _resolve_property_conflict(
        self,
        property_name: str,
        value1: Any,
        value2: Any,
        default_strategy: MergeStrategy
    ) -> Dict[str, Any]:
        """Resolve property conflict."""
        # Check for property-specific rule
        rule = self.property_rules.get(property_name)
        
        if rule and rule.conflict_resolution:
            try:
                resolved_value = rule.conflict_resolution(value1, value2)
                return {
                    "resolved": True,
                    "value": resolved_value,
                    "resolution": "custom_rule"
                }
            except Exception as e:
                self.logger.warning(f"Custom conflict resolution failed: {e}")
        
        strategy = rule.strategy if rule else default_strategy
        
        if strategy == MergeStrategy.KEEP_FIRST:
            return {"resolved": True, "value": value1, "resolution": "keep_first"}
        elif strategy == MergeStrategy.KEEP_LAST:
            return {"resolved": True, "value": value2, "resolution": "keep_last"}
        elif strategy == MergeStrategy.MERGE_ALL:
            # Merge into list if not already
            if isinstance(value1, list):
                if value2 not in value1:
                    value1.append(value2)
                return {"resolved": True, "value": value1, "resolution": "merge_all"}
            else:
                return {"resolved": True, "value": [value1, value2], "resolution": "merge_all"}
        else:
            # Default: keep first
            return {"resolved": True, "value": value1, "resolution": "default"}
    
    def _merge_relationships(
        self,
        entities: List[Dict[str, Any]],
        base_entity: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Merge relationships from all entities."""
        all_relationships = []
        seen_relationships = set()
        
        for entity in entities:
            relationships = entity.get("relationships", [])
            
            for rel in relationships:
                # Create unique key for relationship
                rel_key = (
                    rel.get("subject"),
                    rel.get("predicate"),
                    rel.get("object")
                )
                
                if rel_key not in seen_relationships:
                    all_relationships.append(rel)
                    seen_relationships.add(rel_key)
        
        return all_relationships
    
    def _merge_metadata(
        self,
        entities: List[Dict[str, Any]],
        base_entity: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge metadata from all entities."""
        merged_metadata = base_entity.get("metadata", {}).copy()
        
        for entity in entities:
            if entity == base_entity:
                continue
            
            entity_metadata = entity.get("metadata", {})
            for key, value in entity_metadata.items():
                if key not in merged_metadata:
                    merged_metadata[key] = value
                elif isinstance(merged_metadata[key], list) and isinstance(value, list):
                    merged_metadata[key].extend(value)
                elif isinstance(merged_metadata[key], dict) and isinstance(value, dict):
                    merged_metadata[key].update(value)
        
        return merged_metadata
    
    def validate_merge(self, merge_result: MergeResult) -> Dict[str, Any]:
        """Validate merge result quality."""
        merged_entity = merge_result.merged_entity
        issues = []
        
        # Check completeness
        if not merged_entity.get("name"):
            issues.append("Missing name")
        
        if not merged_entity.get("type"):
            issues.append("Missing type")
        
        # Check for conflicts
        if merge_result.conflicts:
            issues.append(f"{len(merge_result.conflicts)} unresolved conflicts")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "quality_score": 1.0 - (len(issues) * 0.1)
        }
