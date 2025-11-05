"""
Provenance tracker for Semantica framework.

This module provides source tracking for document chunks
to maintain data lineage and traceability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .semantic_chunker import Chunk


@dataclass
class ProvenanceInfo:
    """Provenance information representation."""
    
    chunk_id: str
    source_document: str
    source_path: Optional[str] = None
    start_index: int = 0
    end_index: int = 0
    parent_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    timestamp: Optional[str] = None


class ProvenanceTracker:
    """Provenance tracker for chunk source tracking."""
    
    def __init__(self, **config):
        """
        Initialize provenance tracker.
        
        Args:
            **config: Configuration options:
                - store_metadata: Store chunk metadata (default: True)
                - track_versions: Track version history (default: False)
        """
        self.logger = get_logger("provenance_tracker")
        self.config = config
        
        self.store_metadata = config.get("store_metadata", True)
        self.track_versions = config.get("track_versions", False)
        
        # In-memory store (could be replaced with database)
        self._provenance_store: Dict[str, ProvenanceInfo] = {}
        self._chunk_registry: Dict[str, str] = {}  # chunk_id -> provenance_id
    
    def track_chunk(
        self,
        chunk: Chunk,
        source_document: str,
        source_path: Optional[str] = None,
        parent_chunk_id: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Track provenance for a chunk.
        
        Args:
            chunk: Chunk to track
            source_document: Source document identifier
            source_path: Path to source document
            parent_chunk_id: ID of parent chunk (if chunk was split)
            **metadata: Additional metadata
            
        Returns:
            str: Provenance ID
        """
        chunk_id = str(uuid4())
        provenance_id = str(uuid4())
        
        provenance_info = ProvenanceInfo(
            chunk_id=chunk_id,
            source_document=source_document,
            source_path=source_path,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            parent_chunk_id=parent_chunk_id,
            metadata={
                **chunk.metadata,
                **metadata,
                "chunk_size": len(chunk.text)
            } if self.store_metadata else {},
            version="1.0"
        )
        
        self._provenance_store[provenance_id] = provenance_info
        self._chunk_registry[chunk_id] = provenance_id
        
        return provenance_id
    
    def track_chunks(
        self,
        chunks: List[Chunk],
        source_document: str,
        source_path: Optional[str] = None,
        **metadata
    ) -> List[str]:
        """
        Track provenance for multiple chunks.
        
        Args:
            chunks: List of chunks to track
            source_document: Source document identifier
            source_path: Path to source document
            **metadata: Additional metadata
            
        Returns:
            list: List of provenance IDs
        """
        provenance_ids = []
        parent_chunk_id = None
        
        for chunk in chunks:
            provenance_id = self.track_chunk(
                chunk,
                source_document,
                source_path,
                parent_chunk_id,
                **metadata
            )
            provenance_ids.append(provenance_id)
            parent_chunk_id = provenance_id
        
        return provenance_ids
    
    def get_provenance(self, chunk_id: str) -> Optional[ProvenanceInfo]:
        """
        Get provenance information for a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            ProvenanceInfo: Provenance information or None
        """
        provenance_id = self._chunk_registry.get(chunk_id)
        if provenance_id:
            return self._provenance_store.get(provenance_id)
        return None
    
    def get_source_chunks(self, source_document: str) -> List[ProvenanceInfo]:
        """
        Get all chunks from a source document.
        
        Args:
            source_document: Source document identifier
            
        Returns:
            list: List of provenance information
        """
        return [
            prov for prov in self._provenance_store.values()
            if prov.source_document == source_document
        ]
    
    def get_chunk_lineage(self, chunk_id: str) -> List[ProvenanceInfo]:
        """
        Get lineage (parent chain) for a chunk.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            list: Lineage chain (oldest to newest)
        """
        lineage = []
        current_chunk_id = chunk_id
        
        while current_chunk_id:
            provenance = self.get_provenance(current_chunk_id)
            if not provenance:
                break
            
            lineage.insert(0, provenance)
            current_chunk_id = provenance.parent_chunk_id
        
        return lineage
    
    def link_chunks(self, chunk_id1: str, chunk_id2: str, relationship: str = "related") -> bool:
        """
        Link two chunks with a relationship.
        
        Args:
            chunk_id1: First chunk ID
            chunk_id2: Second chunk ID
            relationship: Relationship type
            
        Returns:
            bool: True if linked successfully
        """
        prov1 = self.get_provenance(chunk_id1)
        prov2 = self.get_provenance(chunk_id2)
        
        if not prov1 or not prov2:
            return False
        
        # Add relationship to metadata
        if "relationships" not in prov1.metadata:
            prov1.metadata["relationships"] = []
        prov1.metadata["relationships"].append({
            "chunk_id": chunk_id2,
            "relationship": relationship
        })
        
        return True
    
    def export_provenance(self, source_document: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Export provenance information.
        
        Args:
            source_document: Filter by source document (optional)
            
        Returns:
            list: List of provenance dictionaries
        """
        if source_document:
            provenance_list = self.get_source_chunks(source_document)
        else:
            provenance_list = list(self._provenance_store.values())
        
        return [self._provenance_to_dict(prov) for prov in provenance_list]
    
    def _provenance_to_dict(self, prov: ProvenanceInfo) -> Dict[str, Any]:
        """Convert ProvenanceInfo to dictionary."""
        return {
            "chunk_id": prov.chunk_id,
            "source_document": prov.source_document,
            "source_path": prov.source_path,
            "start_index": prov.start_index,
            "end_index": prov.end_index,
            "parent_chunk_id": prov.parent_chunk_id,
            "metadata": prov.metadata,
            "version": prov.version,
            "timestamp": prov.timestamp
        }
    
    def clear(self, source_document: Optional[str] = None) -> int:
        """
        Clear provenance data.
        
        Args:
            source_document: Clear only for specific document (optional)
            
        Returns:
            int: Number of entries cleared
        """
        if source_document:
            # Remove only entries for this document
            to_remove = [
                prov_id for prov_id, prov in self._provenance_store.items()
                if prov.source_document == source_document
            ]
            for prov_id in to_remove:
                prov = self._provenance_store.pop(prov_id, None)
                if prov:
                    self._chunk_registry.pop(prov.chunk_id, None)
            return len(to_remove)
        else:
            # Clear all
            count = len(self._provenance_store)
            self._provenance_store.clear()
            self._chunk_registry.clear()
            return count
