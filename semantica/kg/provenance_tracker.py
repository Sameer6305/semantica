"""
Provenance Tracking Module

This module provides comprehensive source tracking and lineage capabilities
for the Semantica framework, enabling tracking of data origins and evolution
for knowledge graph entities and relationships.

Key Features:
    - Entity provenance tracking (source, timestamp, metadata)
    - Relationship provenance tracking
    - Lineage retrieval (complete provenance history)
    - Source aggregation (multiple sources per entity)
    - Temporal tracking (first seen, last updated)

Main Classes:
    - ProvenanceTracker: Main provenance tracking engine

Example Usage:
    >>> from semantica.kg import ProvenanceTracker
    >>> tracker = ProvenanceTracker()
    >>> tracker.track_entity("entity_1", source="source_1", metadata={"confidence": 0.9})
    >>> lineage = tracker.get_lineage("entity_1")
    >>> sources = tracker.get_all_sources("entity_1")

Author: Semantica Contributors
License: MIT
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker


class ProvenanceTracker:
    """
    Provenance tracking engine.

    This class provides provenance tracking capabilities, maintaining source
    and lineage information for knowledge graph entities and relationships.
    Tracks multiple sources per entity and maintains temporal information.

    Features:
        - Entity and relationship provenance tracking
        - Multiple source support per entity
        - Temporal tracking (first seen, last updated)
        - Metadata storage and aggregation
        - Lineage retrieval

    Example Usage:
        >>> tracker = ProvenanceTracker()
        >>> tracker.track_entity("entity_1", source="source_1", metadata={"confidence": 0.9})
        >>> lineage = tracker.get_lineage("entity_1")
        >>> sources = tracker.get_all_sources("entity_1")
    """

    def __init__(self, **config):
        """
        Initialize provenance tracker.

        Sets up the tracker with configuration and initializes provenance
        data storage.

        Args:
            **config: Configuration options (currently unused)
        """
        self.logger = get_logger("provenance_tracker")
        self.config = config
        self.provenance_data: Dict[str, Any] = {}

        # Initialize progress tracker
        self.progress_tracker = get_progress_tracker()
        # Ensure progress tracker is enabled
        if not self.progress_tracker.enabled:
            self.progress_tracker.enabled = True

        self.logger.debug("Provenance tracker initialized")

    def track_entity(
        self, entity_id: str, source: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track entity provenance.

        This method records provenance information for an entity, including
        source, timestamp, and optional metadata. Supports multiple sources
        per entity and maintains first seen and last updated timestamps.

        Args:
            entity_id: Entity identifier
            source: Source identifier (e.g., "file_1", "api_endpoint_2")
            metadata: Optional metadata dictionary (e.g., confidence scores,
                     extraction methods, etc.)
        """
        if entity_id not in self.provenance_data:
            self.provenance_data[entity_id] = {
                "sources": [],
                "first_seen": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "metadata": {},
            }

        # Add source tracking
        source_entry = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.provenance_data[entity_id]["sources"].append(source_entry)
        self.provenance_data[entity_id]["last_updated"] = datetime.now().isoformat()

        # Merge metadata
        if metadata:
            self.provenance_data[entity_id]["metadata"].update(metadata)

        self.logger.debug(
            f"Tracked provenance for entity {entity_id} from source {source}"
        )

    def track_relationship(
        self,
        relationship_id: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Track relationship provenance.

        This method records provenance information for a relationship, including
        source, timestamp, and optional metadata. Similar to track_entity()
        but for relationships.

        Args:
            relationship_id: Relationship identifier
            source: Source identifier
            metadata: Optional metadata dictionary
        """
        if relationship_id not in self.provenance_data:
            self.provenance_data[relationship_id] = {
                "sources": [],
                "first_seen": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "metadata": {},
            }

        source_entry = {
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        self.provenance_data[relationship_id]["sources"].append(source_entry)
        self.provenance_data[relationship_id][
            "last_updated"
        ] = datetime.now().isoformat()

        if metadata:
            self.provenance_data[relationship_id]["metadata"].update(metadata)

    def get_all_sources(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Get all sources for an entity.

        This method retrieves all source entries for a given entity, including
        source identifiers, timestamps, and metadata.

        Args:
            entity_id: Entity identifier

        Returns:
            list: List of source entry dictionaries, each containing:
                - source: Source identifier
                - timestamp: ISO format timestamp
                - metadata: Source metadata dictionary
        """
        if entity_id not in self.provenance_data:
            return []

        return self.provenance_data[entity_id].get("sources", [])

    def get_lineage(self, entity_id: str) -> Dict[str, Any]:
        """
        Get complete lineage for an entity.

        This method retrieves complete lineage information for an entity,
        including all sources, temporal information, and aggregated metadata.

        Args:
            entity_id: Entity identifier

        Returns:
            dict: Complete lineage information containing:
                - sources: List of all source entries
                - first_seen: ISO timestamp of first source
                - last_updated: ISO timestamp of most recent source
                - metadata: Aggregated metadata dictionary
        """
        if entity_id not in self.provenance_data:
            return {}

        return self.provenance_data[entity_id].copy()

    def get_provenance(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get provenance for entity.

        This method is an alias for get_lineage(), returning the complete
        provenance information for an entity, or None if not tracked.

        Args:
            entity_id: Entity identifier

        Returns:
            dict: Complete provenance information (same as get_lineage()),
                  or None if entity is not tracked
        """
        return self.provenance_data.get(entity_id)

    def track_entities_batch(
        self,
        entities: List[Dict[str, Any]],
        source: str,
        pipeline_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Track provenance for multiple entities in batch.

        Args:
            entities: List of entity dictionaries, each containing at least 'id' key
            source: Source identifier
            pipeline_id: Optional pipeline ID for progress tracking
            metadata: Optional metadata dictionary to apply to all entities

        Returns:
            int: Number of entities tracked
        """
        if not entities:
            return 0

        tracking_id = self.progress_tracker.start_tracking(
            module="kg",
            submodule="ProvenanceTracker",
            message=f"Tracking provenance for {len(entities)} entities",
            pipeline_id=pipeline_id,
        )

        try:
            tracked_count = 0
            for i, entity in enumerate(entities):
                entity_id = entity.get("id") or entity.get("entity_id")
                if not entity_id:
                    self.logger.warning(f"Skipping entity without ID: {entity}")
                    continue

                # Merge entity-specific metadata with batch metadata
                entity_metadata = {**(metadata or {}), **(entity.get("metadata", {}))}
                self.track_entity(entity_id, source, entity_metadata)

                tracked_count += 1

                # Update progress
                self.progress_tracker.update_progress(
                    tracking_id,
                    processed=i + 1,
                    total=len(entities),
                    message=f"Tracking entity {i+1}/{len(entities)}...",
                )

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Tracked {tracked_count} entities",
            )
            return tracked_count

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def track_relationships_batch(
        self,
        relationships: List[Dict[str, Any]],
        source: str,
        pipeline_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Track provenance for multiple relationships in batch.

        Args:
            relationships: List of relationship dictionaries, each containing at least 'id' key
            source: Source identifier
            pipeline_id: Optional pipeline ID for progress tracking
            metadata: Optional metadata dictionary to apply to all relationships

        Returns:
            int: Number of relationships tracked
        """
        if not relationships:
            return 0

        tracking_id = self.progress_tracker.start_tracking(
            module="kg",
            submodule="ProvenanceTracker",
            message=f"Tracking provenance for {len(relationships)} relationships",
            pipeline_id=pipeline_id,
        )

        try:
            tracked_count = 0
            for i, relationship in enumerate(relationships):
                relationship_id = relationship.get("id") or relationship.get("relationship_id")
                if not relationship_id:
                    self.logger.warning(f"Skipping relationship without ID: {relationship}")
                    continue

                # Merge relationship-specific metadata with batch metadata
                rel_metadata = {**(metadata or {}), **(relationship.get("metadata", {}))}
                self.track_relationship(relationship_id, source, rel_metadata)

                tracked_count += 1

                # Update progress
                self.progress_tracker.update_progress(
                    tracking_id,
                    processed=i + 1,
                    total=len(relationships),
                    message=f"Tracking relationship {i+1}/{len(relationships)}...",
                )

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Tracked {tracked_count} relationships",
            )
            return tracked_count

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise