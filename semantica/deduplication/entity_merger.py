"""
Entity Merger for Semantica framework.

Performs semantic deduplication to merge semantically similar
entities and maintain graph cleanliness.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from .duplicate_detector import DuplicateDetector, DuplicateGroup
from .merge_strategy import MergeStrategyManager, MergeStrategy, MergeResult


@dataclass
class MergeOperation:
    """Entity merge operation representation."""
    
    source_entities: List[Dict[str, Any]]
    merged_entity: Dict[str, Any]
    merge_result: MergeResult
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = None


class EntityMerger:
    """
    Entity merging engine.
    
    • Calculates semantic similarity
    • Detects and groups duplicates
    • Merges entities with conflict resolution
    • Merges properties and relationships
    • Preserves provenance during merge
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize entity merger."""
        self.logger = get_logger("entity_merger")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.duplicate_detector = DuplicateDetector(**config.get("detector", {}))
        self.merge_strategy_manager = MergeStrategyManager(**config.get("strategy", {}))
        
        self.merge_history: List[MergeOperation] = []
        self.preserve_provenance = config.get("preserve_provenance", True)
    
    def merge_duplicates(
        self,
        entities: List[Dict[str, Any]],
        strategy: Optional[MergeStrategy] = None,
        **options
    ) -> List[MergeOperation]:
        """
        Merge duplicate entities.
        
        Args:
            entities: List of entities to process
            strategy: Merge strategy
            **options: Merge options
            
        Returns:
            List of merge operations
        """
        # Detect duplicate groups
        duplicate_groups = self.duplicate_detector.detect_duplicate_groups(
            entities,
            **options
        )
        
        merge_operations = []
        
        for group in duplicate_groups:
            if len(group.entities) < 2:
                continue
            
            # Merge group
            merge_result = self.merge_strategy_manager.merge_entities(
                group.entities,
                strategy=strategy,
                **options
            )
            
            # Preserve provenance
            if self.preserve_provenance:
                merge_result.merged_entity = self._add_provenance(
                    merge_result.merged_entity,
                    group.entities
                )
            
            operation = MergeOperation(
                source_entities=group.entities,
                merged_entity=merge_result.merged_entity,
                merge_result=merge_result,
                metadata={
                    "group_confidence": group.confidence,
                    "similarity_scores": group.similarity_scores
                }
            )
            
            merge_operations.append(operation)
            self.merge_history.append(operation)
        
        return merge_operations
    
    def merge_entity_group(
        self,
        entities: List[Dict[str, Any]],
        strategy: Optional[MergeStrategy] = None,
        **options
    ) -> MergeOperation:
        """
        Merge a specific group of entities.
        
        Args:
            entities: Entities to merge
            strategy: Merge strategy
            **options: Merge options
            
        Returns:
            MergeOperation result
        """
        if len(entities) < 2:
            raise ValidationError("Need at least 2 entities to merge")
        
        merge_result = self.merge_strategy_manager.merge_entities(
            entities,
            strategy=strategy,
            **options
        )
        
        # Preserve provenance
        if self.preserve_provenance:
            merge_result.merged_entity = self._add_provenance(
                merge_result.merged_entity,
                entities
            )
        
        operation = MergeOperation(
            source_entities=entities,
            merged_entity=merge_result.merged_entity,
            merge_result=merge_result
        )
        
        self.merge_history.append(operation)
        
        return operation
    
    def incremental_merge(
        self,
        new_entities: List[Dict[str, Any]],
        existing_entities: List[Dict[str, Any]],
        **options
    ) -> List[MergeOperation]:
        """
        Incremental merge of new entities with existing ones.
        
        Args:
            new_entities: New entities to merge
            existing_entities: Existing entities
            **options: Merge options
            
        Returns:
            List of merge operations
        """
        # Detect duplicates between new and existing
        candidates = self.duplicate_detector.incremental_detect(
            new_entities,
            existing_entities,
            **options
        )
        
        merge_operations = []
        processed_new = set()
        processed_existing = set()
        
        for candidate in candidates:
            new_entity_id = candidate.entity1.get("id") or id(candidate.entity1)
            existing_entity_id = candidate.entity2.get("id") or id(candidate.entity2)
            
            if new_entity_id in processed_new or existing_entity_id in processed_existing:
                continue
            
            # Merge the pair
            operation = self.merge_entity_group(
                [candidate.entity1, candidate.entity2],
                **options
            )
            
            merge_operations.append(operation)
            processed_new.add(new_entity_id)
            processed_existing.add(existing_entity_id)
        
        return merge_operations
    
    def _add_provenance(
        self,
        merged_entity: Dict[str, Any],
        source_entities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add provenance information to merged entity."""
        if "provenance" not in merged_entity.get("metadata", {}):
            merged_entity.setdefault("metadata", {})["provenance"] = {}
        
        provenance = merged_entity["metadata"]["provenance"]
        provenance["merged_from"] = [
            {
                "id": e.get("id"),
                "name": e.get("name"),
                "source": e.get("metadata", {}).get("source")
            }
            for e in source_entities
        ]
        provenance["merge_count"] = len(source_entities)
        
        return merged_entity
    
    def get_merge_history(self) -> List[MergeOperation]:
        """Get merge operation history."""
        return self.merge_history
    
    def validate_merge_quality(self, merge_operation: MergeOperation) -> Dict[str, Any]:
        """Validate quality of merge operation."""
        return self.merge_strategy_manager.validate_merge(merge_operation.merge_result)
