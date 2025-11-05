"""
Entity resolver for Semantica framework.

This module provides entity disambiguation and resolution
for knowledge graph construction.
"""

from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from ..deduplication.duplicate_detector import DuplicateDetector
from ..deduplication.entity_merger import EntityMerger


class EntityResolver:
    """
    Entity resolver.
    
    Provides entity disambiguation and resolution for knowledge graph construction.
    """
    
    def __init__(self, **config):
        """Initialize entity resolver."""
        self.logger = get_logger("entity_resolver")
        self.config = config
        
        # Initialize deduplication components
        self.duplicate_detector = DuplicateDetector(**config.get("deduplication", {}))
        self.entity_merger = EntityMerger(**config.get("merger", {}))
        
        self.resolution_strategy = config.get("strategy", "fuzzy")
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
    
    def resolve_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve and disambiguate entities.
        
        Args:
            entities: List of entities to resolve
            
        Returns:
            Resolved entities
        """
        self.logger.info(f"Resolving {len(entities)} entities")
        
        if not entities:
            return []
        
        # Step 1: Detect duplicates
        duplicate_groups = self.duplicate_detector.detect_duplicate_groups(
            entities,
            threshold=self.similarity_threshold
        )
        
        # Step 2: Merge duplicates
        merged_entities = []
        processed_ids = set()
        
        for group in duplicate_groups:
            if len(group.entities) < 2:
                continue
            
            # Merge the group
            merge_operations = self.entity_merger.merge_duplicates(
                group.entities,
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
        
        # Step 3: Add non-duplicate entities
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            if entity_id and entity_id not in processed_ids:
                merged_entities.append(entity)
        
        self.logger.info(f"Resolved to {len(merged_entities)} unique entities")
        return merged_entities
    
    def merge_duplicates(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge duplicate entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Merged entities
        """
        self.logger.info(f"Merging duplicates from {len(entities)} entities")
        
        merge_operations = self.entity_merger.merge_duplicates(
            entities,
            **self.config
        )
        
        merged_entities = [op.merged_entity for op in merge_operations]
        
        # Add non-duplicate entities
        processed_ids = set()
        for op in merge_operations:
            for source_entity in op.source_entities:
                entity_id = source_entity.get("id") or source_entity.get("entity_id")
                if entity_id:
                    processed_ids.add(entity_id)
        
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            if entity_id and entity_id not in processed_ids:
                merged_entities.append(entity)
        
        self.logger.info(f"Merged to {len(merged_entities)} entities")
        return merged_entities
