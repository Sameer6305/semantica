"""
Provenance tracker for Semantica framework.

This module provides source tracking and lineage
for knowledge graph entities and relationships.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from ..utils.logging import get_logger


class ProvenanceTracker:
    """
    Provenance tracker.
    
    Tracks source and lineage for knowledge graph entities and relationships.
    """
    
    def __init__(self, **config):
        """Initialize provenance tracker."""
        self.logger = get_logger("provenance_tracker")
        self.config = config
        self.provenance_data: Dict[str, Any] = {}
    
    def track_entity(
        self,
        entity_id: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track entity provenance.
        
        Args:
            entity_id: Entity identifier
            source: Source identifier
            metadata: Optional metadata
        """
        if entity_id not in self.provenance_data:
            self.provenance_data[entity_id] = {
                "sources": [],
                "first_seen": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "metadata": {}
            }
        
        # Add source tracking
        source_entry = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.provenance_data[entity_id]["sources"].append(source_entry)
        self.provenance_data[entity_id]["last_updated"] = datetime.now().isoformat()
        
        # Merge metadata
        if metadata:
            self.provenance_data[entity_id]["metadata"].update(metadata)
        
        self.logger.debug(f"Tracked provenance for entity {entity_id} from source {source}")
    
    def track_relationship(
        self,
        relationship_id: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track relationship provenance.
        
        Args:
            relationship_id: Relationship identifier
            source: Source identifier
            metadata: Optional metadata
        """
        if relationship_id not in self.provenance_data:
            self.provenance_data[relationship_id] = {
                "sources": [],
                "first_seen": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "metadata": {}
            }
        
        source_entry = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.provenance_data[relationship_id]["sources"].append(source_entry)
        self.provenance_data[relationship_id]["last_updated"] = datetime.now().isoformat()
        
        if metadata:
            self.provenance_data[relationship_id]["metadata"].update(metadata)
    
    def get_all_sources(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get all sources for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            List of source entries
        """
        if entity_id not in self.provenance_data:
            return []
        
        return self.provenance_data[entity_id].get("sources", [])
    
    def get_lineage(self, entity_id: str) -> Dict[str, Any]:
        """
        Get complete lineage for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Lineage information
        """
        if entity_id not in self.provenance_data:
            return {}
        
        return self.provenance_data[entity_id].copy()
    
    def get_provenance(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get provenance for entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Provenance information or None
        """
        return self.provenance_data.get(entity_id)
