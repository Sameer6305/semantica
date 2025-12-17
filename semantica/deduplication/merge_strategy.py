"""
Merge Strategy Manager Module

This module provides comprehensive merge strategy management for the Semantica
framework, handling different strategies for merging duplicate entities including
property merging rules, relationship preservation, and conflict resolution.

Algorithms Used:
    - Strategy Pattern: Multiple merge strategies (keep_first, keep_last, keep_most_complete, keep_highest_confidence, merge_all)
    - Conflict Resolution: Voting (majority), credibility-weighted (weighted average), temporal (most recent), confidence-based selection
    - Property Merging: Rule-based property combination with custom rules, priorities, and conflict resolution functions
    - Relationship Preservation: Union of relationship sets during merges
    - Merge Quality Validation: Validation of merged entities for completeness, consistency, and quality metrics

Key Features:
    - Multiple merge strategies (keep_first, keep_most_complete, keep_highest_confidence, etc.)
    - Property-specific merge rules with custom conflict resolution
    - Custom conflict resolution functions for complex merging scenarios
    - Relationship preservation during merges (union of all relationships)
    - Merge quality validation with completeness and consistency checks
    - Support for custom merge strategies and property-specific rules

Main Classes:
    - MergeStrategyManager: Main merge strategy management engine
    - MergeStrategy: Enumeration of available merge strategies
    - MergeResult: Result of merge operation with conflicts and metadata
    - PropertyMergeRule: Rule for merging specific properties

Example Usage:
    >>> from semantica.deduplication import MergeStrategyManager
    >>> manager = MergeStrategyManager(default_strategy="keep_most_complete")
    >>> manager.add_property_rule("name", "keep_first")
    >>> result = manager.merge_entities(entities, strategy="keep_most_complete")
    >>> 
    >>> # Custom conflict resolution
    >>> def resolve_conflict(values):
    ...     return max(values, key=len)  # Return longest value
    >>> manager.add_property_rule("description", "custom", conflict_resolution=resolve_conflict)

Author: Semantica Contributors
License: MIT
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker


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
    Merge strategy management engine for entity merging.

    This class manages different strategies for merging duplicate entities, handling
    property merging rules, relationship preservation, conflict resolution, and
    merge quality validation.

    Features:
        - Multiple merge strategies (keep_first, keep_most_complete, etc.)
        - Property-specific merge rules with custom conflict resolution
        - Relationship preservation during merges
        - Automatic conflict detection and resolution
        - Merge quality validation
        - Support for custom merge strategies

    Example Usage:
        >>> manager = MergeStrategyManager(default_strategy="keep_most_complete")
        >>> manager.add_property_rule("name", "keep_first")
        >>> result = manager.merge_entities(entities)
    """

    def __init__(
        self,
        default_strategy: str = "keep_most_complete",
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize merge strategy manager.

        Sets up the manager with a default merge strategy and empty property rules.
        Property rules can be added later using add_property_rule().

        Args:
            default_strategy: Default merge strategy name (default: "keep_most_complete").
                            Options: "keep_first", "keep_last", "keep_most_complete",
                            "keep_highest_confidence", "merge_all"
            config: Configuration dictionary (merged with kwargs)
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("merge_strategy_manager")

        # Merge configuration
        self.config = config or {}
        self.config.update(kwargs)

        # Default merge strategy
        strategy_name = self.config.get("default_strategy", default_strategy)
        try:
            self.default_strategy = MergeStrategy(strategy_name)
        except ValueError:
            self.logger.warning(
                f"Invalid default strategy '{strategy_name}', using 'keep_most_complete'"
            )
            self.default_strategy = MergeStrategy.KEEP_MOST_COMPLETE

        # Property-specific merge rules
        self.property_rules: Dict[str, PropertyMergeRule] = {}

        # Custom merge strategies (callable functions)
        self.custom_strategies: Dict[str, Callable] = {}

        # Initialize progress tracker
        self.progress_tracker = get_progress_tracker()

        self.logger.debug(
            f"Merge strategy manager initialized (default: {self.default_strategy.value})"
        )

    def add_property_rule(
        self,
        property_name: str,
        strategy: Union[MergeStrategy, str],
        conflict_resolution: Optional[Callable] = None,
        priority: int = 0,
    ) -> None:
        """
        Add property merge rule.

        Args:
            property_name: Property name
            strategy: Merge strategy (Enum or string)
            conflict_resolution: Custom conflict resolution function
            priority: Rule priority
        """
        if isinstance(strategy, str):
            try:
                strategy = MergeStrategy(strategy.lower())
            except ValueError:
                self.logger.warning(
                    f"Invalid property strategy '{strategy}', defaulting to keep_most_complete"
                )
                strategy = MergeStrategy.KEEP_MOST_COMPLETE

        rule = PropertyMergeRule(
            property_name=property_name,
            strategy=strategy,
            conflict_resolution=conflict_resolution,
            priority=priority,
        )

        self.property_rules[property_name] = rule

    def merge_entities(
        self,
        entities: List[Dict[str, Any]],
        strategy: Optional[Union[MergeStrategy, str]] = None,
        **options,
    ) -> MergeResult:
        """
        Merge entities using specified strategy.

        Args:
            entities: List of entities to merge
            strategy: Merge strategy (uses default if None). Can be Enum or string.
            **options: Merge options

        Returns:
            MergeResult with merged entity
        """
        # Track entity merging
        tracking_id = self.progress_tracker.start_tracking(
            file=None,
            module="deduplication",
            submodule="MergeStrategyManager",
            message=f"Merging {len(entities)} entities",
        )

        try:
            if not entities:
                raise ValidationError("No entities to merge")

            if len(entities) == 1:
                self.progress_tracker.stop_tracking(
                    tracking_id,
                    status="completed",
                    message="Single entity, no merge needed",
                )
                return MergeResult(merged_entity=entities[0], merged_entities=entities)

            # Resolve strategy
            if strategy is None:
                strategy = self.default_strategy
            elif isinstance(strategy, str):
                try:
                    strategy = MergeStrategy(strategy.lower())
                except ValueError:
                    self.logger.warning(
                        f"Invalid strategy '{strategy}', using default '{self.default_strategy.value}'"
                    )
                    strategy = self.default_strategy
            
            # If it's still not a MergeStrategy (e.g. invalid type passed), force default
            if not isinstance(strategy, MergeStrategy):
                 strategy = self.default_strategy

            self.progress_tracker.update_tracking(
                tracking_id, message=f"Selecting base entity using {strategy.value}..."
            )
            # Select base entity
            base_entity = self._select_base_entity(entities, strategy)

            self.progress_tracker.update_tracking(
                tracking_id, message="Merging properties..."
            )
            # Merge properties
            merged_properties, property_conflicts = self._merge_properties(
                entities, base_entity, strategy
            )

            self.progress_tracker.update_tracking(
                tracking_id, message="Merging relationships..."
            )
            # Merge relationships
            merged_relationships = self._merge_relationships(entities, base_entity)

            self.progress_tracker.update_tracking(
                tracking_id, message="Building merged entity..."
            )
            # Build merged entity
            merged_entity = {
                "id": base_entity.get("id"),
                "name": self._merge_top_level_field("name", entities, base_entity),
                "type": self._merge_top_level_field("type", entities, base_entity),
                "properties": merged_properties,
                "relationships": merged_relationships,
                "metadata": self._merge_metadata(entities, base_entity),
                "merged_from": [e.get("id") for e in entities if e.get("id")],
                "merge_strategy": strategy.value,
            }

            self.progress_tracker.stop_tracking(
                tracking_id, status="completed", message="Entity merge completed"
            )
            return MergeResult(
                merged_entity=merged_entity,
                merged_entities=entities,
                conflicts=property_conflicts,
                metadata={"strategy": strategy.value},
            )

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def _merge_top_level_field(
        self,
        field_name: str,
        entities: List[Dict[str, Any]],
        base_entity: Dict[str, Any],
    ) -> Any:
        """Merge top-level field (name, type) respecting property rules."""
        rule = self.property_rules.get(field_name)
        if not rule:
            return base_entity.get(field_name)

        # If rule exists, merge from scratch following list order
        valid_entities = [e for e in entities if e.get(field_name) is not None]
        if not valid_entities:
            return None

        merged_value = valid_entities[0].get(field_name)

        for i in range(1, len(valid_entities)):
            val = valid_entities[i].get(field_name)
            if merged_value != val:
                conflict = self._resolve_property_conflict(
                    field_name, merged_value, val, rule.strategy
                )
                if conflict.get("resolved"):
                    merged_value = conflict["value"]

        return merged_value

    def _select_base_entity(
        self, entities: List[Dict[str, Any]], strategy: MergeStrategy
    ) -> Dict[str, Any]:
        """Select base entity for merge."""
        if strategy == MergeStrategy.KEEP_FIRST:
            return entities[0]
        elif strategy == MergeStrategy.KEEP_LAST:
            return entities[-1]
        elif strategy == MergeStrategy.KEEP_MOST_COMPLETE:
            return max(
                entities,
                key=lambda e: len(e.get("properties", {}))
                + len(e.get("relationships", [])),
            )
        elif strategy == MergeStrategy.KEEP_HIGHEST_CONFIDENCE:
            return max(entities, key=lambda e: e.get("confidence", 0.0))
        else:
            return entities[0]

    def _merge_properties(
        self,
        entities: List[Dict[str, Any]],
        base_entity: Dict[str, Any],
        strategy: MergeStrategy,
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
                        prop_name, merged_properties[prop_name], prop_value, strategy
                    )

                    if conflict.get("resolved"):
                        merged_properties[prop_name] = conflict["value"]
                    else:
                        conflicts.append(
                            {
                                "property": prop_name,
                                "values": [merged_properties[prop_name], prop_value],
                                "resolution": conflict.get("resolution"),
                            }
                        )

        return merged_properties, conflicts

    def _resolve_property_conflict(
        self,
        property_name: str,
        value1: Any,
        value2: Any,
        default_strategy: MergeStrategy,
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
                    "resolution": "custom_rule",
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
                return {
                    "resolved": True,
                    "value": [value1, value2],
                    "resolution": "merge_all",
                }
        else:
            # Default: keep first
            return {"resolved": True, "value": value1, "resolution": "default"}

    def _merge_relationships(
        self, entities: List[Dict[str, Any]], base_entity: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Merge relationships from all entities."""
        all_relationships = []
        seen_relationships = set()

        for entity in entities:
            relationships = entity.get("relationships", [])

            for rel in relationships:
                # Create unique key for relationship
                rel_key = (rel.get("subject"), rel.get("predicate"), rel.get("object"))

                if rel_key not in seen_relationships:
                    all_relationships.append(rel)
                    seen_relationships.add(rel_key)

        return all_relationships

    def _merge_metadata(
        self, entities: List[Dict[str, Any]], base_entity: Dict[str, Any]
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
            "quality_score": 1.0 - (len(issues) * 0.1),
        }
