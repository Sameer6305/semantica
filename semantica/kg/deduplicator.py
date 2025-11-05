"""
Deduplicator for Semantica framework.

This module provides duplicate detection and merging
for knowledge graph entities and relationships.
"""

from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from ..deduplication.duplicate_detector import DuplicateDetector, DuplicateGroup
from ..deduplication.entity_merger import EntityMerger


class Deduplicator:
    """
    Deduplicator.
    
    Detects and merges duplicate entities and relationships in knowledge graphs.
    """
    
    def __init__(self, **config):
        """Initialize deduplicator."""
        self.logger = get_logger("deduplicator")
        self.config = config
        
        # Initialize deduplication components
        self.duplicate_detector = DuplicateDetector(**config.get("detection", {}))
        self.entity_merger = EntityMerger(**config.get("merger", {}))
    
    def find_duplicates(self, entities: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Find duplicate entities.
        
        Args:
            entities: List of entities
            
        Returns:
            List of duplicate groups
        """
        self.logger.info(f"Finding duplicates in {len(entities)} entities")
        
        # Detect duplicate groups
        duplicate_groups = self.duplicate_detector.detect_duplicate_groups(
            entities,
            **self.config
        )
        
        # Convert to list of lists
        result = []
        for group in duplicate_groups:
            if len(group.entities) >= 2:
                result.append(group.entities)
        
        self.logger.info(f"Found {len(result)} duplicate groups")
        return result
    
    def merge_duplicates(
        self,
        duplicate_groups: List[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Merge duplicate entities.
        
        Args:
            duplicate_groups: Groups of duplicate entities
            
        Returns:
            Merged entities
        """
        self.logger.info(f"Merging {len(duplicate_groups)} duplicate groups")
        
        merged_entities = []
        processed_ids = set()
        
        for group in duplicate_groups:
            if len(group) < 2:
                continue
            
            # Merge the group
            merge_operations = self.entity_merger.merge_duplicates(
                group,
                **self.config
            )
            
            for operation in merge_operations:
                merged_entity = operation.merged_entity
                merged_entities.append(merged_entity)
                
                # Mark source entities as processed
                for source_entity in operation.source_entities:
                    entity_id = source_entity.get("id") or source_entity.get("entity_id")
                    if entity_id:
                        processed_ids.add(entity_id)
        
        self.logger.info(f"Merged to {len(merged_entities)} entities")
        return merged_entities
