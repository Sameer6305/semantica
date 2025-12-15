"""
Knowledge Graph Builder Module

This module provides comprehensive knowledge graph construction capabilities
from extracted entities and relationships, with full support for temporal
knowledge graphs and advanced graph operations.

Key Features:
    - Build knowledge graphs from entities and relationships
    - Temporal knowledge graph support with time-aware edges
    - Entity resolution and deduplication
    - Conflict detection and resolution
    - Temporal snapshots and versioning
    - Neo4j integration for graph storage

Example Usage:
    >>> from semantica.kg import GraphBuilder
    >>> builder = GraphBuilder(merge_entities=True, resolve_conflicts=True)
    >>> graph = builder.build(sources=[{"entities": [...], "relationships": [...]}])

Author: Semantica Contributors
License: MIT
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class GraphBuilder:
    """
    Knowledge graph builder with temporal support.

    • Constructs knowledge graphs from entities and relationships
    • Supports temporal knowledge graphs with time-aware edges
    • Manages node and edge creation with temporal annotations
    • Handles graph structure optimization
    • Supports incremental graph building
    • Enables temporal versioning and snapshots

    Attributes:
        • merge_entities: Whether to merge duplicate entities
        • entity_resolution_strategy: Strategy for entity resolution
        • resolve_conflicts: Whether to resolve conflicts
        • enable_temporal: Enable temporal knowledge graph features
        • temporal_granularity: Time granularity (second, minute, hour, day, etc.)
        • track_history: Track historical changes
        • version_snapshots: Create version snapshots

    Methods:
        • build(): Build knowledge graph from sources
        • add_temporal_edge(): Add edge with temporal validity
        • create_temporal_snapshot(): Create temporal snapshot
        • query_temporal(): Query graph at specific time point
    """

    def __init__(
        self,
        merge_entities=True,
        entity_resolution_strategy="fuzzy",
        resolve_conflicts=True,
        enable_temporal=False,
        temporal_granularity="day",
        track_history=False,
        version_snapshots=False,
        **kwargs,
    ):
        """
        Initialize graph builder.

        Args:
            merge_entities: Whether to merge duplicate entities
            entity_resolution_strategy: Strategy for entity resolution ("fuzzy", "exact", "ml-based")
            resolve_conflicts: Whether to resolve conflicts
            enable_temporal: Enable temporal knowledge graph features
            temporal_granularity: Time granularity ("second", "minute", "hour", "day", "week", "month", "year")
            track_history: Track historical changes to entities/relationships
            version_snapshots: Create version snapshots at intervals
            **kwargs: Additional configuration options
        """
        self.merge_entities = merge_entities
        self.entity_resolution_strategy = entity_resolution_strategy
        self.resolve_conflicts = resolve_conflicts
        self.enable_temporal = enable_temporal
        self.temporal_granularity = temporal_granularity
        self.track_history = track_history
        self.version_snapshots = version_snapshots

        # Initialize logging
        from ..utils.logging import get_logger
        from ..utils.progress_tracker import get_progress_tracker

        self.logger = get_logger("graph_builder")
        self.progress_tracker = get_progress_tracker()

        # Initialize entity resolver if entity merging is enabled
        # This helps deduplicate and merge similar entities
        if self.merge_entities:
            from .entity_resolver import EntityResolver

            entity_resolution_config = kwargs.get("entity_resolution", {})
            self.entity_resolver = EntityResolver(
                strategy=self.entity_resolution_strategy, **entity_resolution_config
            )
            self.logger.debug(
                f"Entity resolver initialized with strategy: {self.entity_resolution_strategy}"
            )
        else:
            self.entity_resolver = None
            self.logger.debug("Entity merging disabled, skipping entity resolver")

        # Initialize conflict detector if conflict resolution is enabled
        # This helps detect and resolve conflicting information in the graph
        if self.resolve_conflicts:
            from ..conflicts.conflict_detector import ConflictDetector

            conflict_detection_config = kwargs.get("conflict_detection", {})
            self.conflict_detector = ConflictDetector(**conflict_detection_config)
            self.logger.debug("Conflict detector initialized")
        else:
            self.conflict_detector = None
            self.logger.debug("Conflict resolution disabled")

    def _process_item(self, item: Any, all_entities: List[Any], all_relationships: List[Any]):
        """Helper to process a single item and add to entities or relationships list."""
        if hasattr(item, "text") and hasattr(item, "label"):
            # It's likely an Entity object
            # Convert to dict format expected by graph builder
            entity_dict = {
                "id": getattr(item, "id", item.text), # Use text as ID if no ID
                "name": item.text,
                "type": item.label,
                "confidence": getattr(item, "confidence", 1.0),
                "metadata": getattr(item, "metadata", {})
            }
            all_entities.append(entity_dict)
        elif hasattr(item, "subject") and hasattr(item, "predicate") and hasattr(item, "object"):
            # It's likely a Relation object
            # Convert to dict format
            # Subject and Object in Relation might be Entity objects or strings
            subj = item.subject
            obj = item.object
            
            subj_id = getattr(subj, "text", subj) if not isinstance(subj, str) else subj
            obj_id = getattr(obj, "text", obj) if not isinstance(obj, str) else obj
            
            rel_dict = {
                "source": subj_id,
                "target": obj_id,
                "type": item.predicate,
                "confidence": getattr(item, "confidence", 1.0),
                "metadata": getattr(item, "metadata", {})
            }
            all_relationships.append(rel_dict)
        elif isinstance(item, dict):
            processed = False
            if "entities" in item:
                all_entities.extend(item["entities"])
                processed = True
            if "relationships" in item:
                all_relationships.extend(item["relationships"])
                processed = True
            
            if not processed:
                if "source" in item and "target" in item:
                    all_relationships.append(item)
                elif "id" in item or "entity_id" in item or "name" in item:
                    all_entities.append(item)
        else:
            # Unknown type, try to treat as entity if it has string representation
            pass

    def build(
        self,
        sources: Union[List[Any], Any],
        second_arg: Optional[Any] = None,
        **options,
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from sources.

        Args:
            sources: Entities or sources list
            second_arg: Optional relationships list or entity_resolver (for backward compatibility)
            **options: Additional build options

        Returns:
            Dictionary containing entities and relationships
        """
        # Handle arguments
        entity_resolver = None
        explicit_relationships = None

        # Check if second_arg is entity_resolver or relationships
        if second_arg is not None:
            if hasattr(second_arg, "resolve"): # Duck typing for EntityResolver
                entity_resolver = second_arg
            elif isinstance(second_arg, list):
                explicit_relationships = second_arg
        
        # Also check options for named arguments
        if "entity_resolver" in options:
            entity_resolver = options.pop("entity_resolver")
        if "relationships" in options:
            explicit_relationships = options.pop("relationships")

        # Normalize sources to list
        if not isinstance(sources, list):
            sources = [sources]

        # Track graph building
        tracking_id = self.progress_tracker.start_tracking(
            module="kg",
            submodule="GraphBuilder",
            message=f"Knowledge graph from {len(sources)} source(s)",
        )

        try:
            self.logger.info(f"Building knowledge graph from {len(sources)} source(s)")

            # Use provided resolver or default instance resolver
            resolver_to_use = entity_resolver or self.entity_resolver

            # Extract entities and relationships from all sources
            all_entities = []
            all_relationships = []

            # Process sources (which might be entities)
            for source in sources:
                if isinstance(source, list):
                     # List of items (could be entities, relations, or mixed)
                    for item in source:
                        self._process_item(item, all_entities, all_relationships)
                else:
                    self._process_item(source, all_entities, all_relationships)

            # Process explicit relationships if provided
            if explicit_relationships:
                for rel_item in explicit_relationships:
                    if isinstance(rel_item, list):
                        for item in rel_item:
                             self._process_item(item, all_entities, all_relationships)
                    else:
                        self._process_item(rel_item, all_entities, all_relationships)

            self.logger.debug(
                f"Extracted {len(all_entities)} entities and "
                f"{len(all_relationships)} relationships from sources"
            )

            # Resolve entities (deduplicate and merge) if resolver is available
            resolved_entities = all_entities
            if resolver_to_use and all_entities:
                self.logger.info(
                    f"Resolving {len(all_entities)} entities using {self.entity_resolution_strategy} strategy"
                )
                resolved_entities = resolver_to_use.resolve_entities(all_entities)
                self.logger.info(
                    f"Entity resolution complete: {len(all_entities)} -> {len(resolved_entities)} unique entities"
                )

            # Build graph structure
            graph = {
                "entities": resolved_entities,
                "relationships": all_relationships,
                "metadata": {
                    "num_entities": len(resolved_entities),
                    "num_relationships": len(all_relationships),
                    "temporal_enabled": self.enable_temporal,
                    "timestamp": self._get_timestamp(),
                    "entity_resolution_applied": resolver_to_use is not None,
                },
            }

            # Detect and resolve conflicts if conflict detector is available
            if self.conflict_detector:
                self.logger.debug("Detecting conflicts in graph")
                # Pass only entities to detect_conflicts as it expects List[Dict]
                detected_conflicts = self.conflict_detector.detect_conflicts(graph["entities"])

                if detected_conflicts:
                    conflict_count = len(detected_conflicts)
                    self.logger.warning(
                        f"Detected {conflict_count} conflict(s) in graph"
                    )

                    # Attempt to resolve conflicts
                    resolution_result = self.conflict_detector.resolve_conflicts(
                        detected_conflicts
                    )
                    resolved_count = resolution_result.get("resolved_count", 0)

                    if resolved_count > 0:
                        self.logger.info(
                            f"Successfully resolved {resolved_count} out of {conflict_count} conflict(s)"
                        )
                    else:
                        self.logger.warning("No conflicts were automatically resolved")

            # Log final graph statistics
            self.logger.info(
                f"Knowledge graph built successfully: "
                f"{len(resolved_entities)} entities, {len(all_relationships)} relationships"
            )

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Built graph with {len(resolved_entities)} entities, {len(all_relationships)} relationships",
            )
            return graph

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def add_temporal_edge(
        self,
        graph,
        source,
        target,
        relationship,
        valid_from=None,
        valid_until=None,
        temporal_metadata=None,
        **kwargs,
    ):
        """
        Add edge with temporal validity information.

        Args:
            graph: Knowledge graph to add edge to
            source: Source entity/node
            target: Target entity/node
            relationship: Relationship type
            valid_from: Start time for relationship validity (datetime, timestamp, or ISO string)
            valid_until: End time for relationship validity (None for ongoing)
            temporal_metadata: Additional temporal metadata (timezone, precision, etc.)
            **kwargs: Additional edge properties

        Returns:
            Edge object with temporal annotations
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="kg",
            submodule="GraphBuilder",
            message=f"Adding temporal edge: {source} -{relationship}-> {target}",
        )

        try:
            self.logger.info(
                f"Adding temporal edge: {source} -{relationship}-> {target}"
            )

            # Parse temporal information
            valid_from = self._parse_time(valid_from) or self._get_timestamp()
            valid_until = self._parse_time(valid_until) if valid_until else None

            # Create edge with temporal information
            edge = {
                "source": source,
                "target": target,
                "type": relationship,
                "valid_from": valid_from,
                "valid_until": valid_until,
                "temporal_metadata": temporal_metadata or {},
                **kwargs,
            }

            # Add to graph
            if "relationships" not in graph:
                graph["relationships"] = []
            graph["relationships"].append(edge)

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Added temporal edge: {source} -{relationship}-> {target}",
            )
            return edge

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def create_temporal_snapshot(
        self, graph, timestamp=None, snapshot_name=None, **options
    ):
        """
        Create temporal snapshot of graph at specific time point.

        Args:
            graph: Knowledge graph to snapshot
            timestamp: Time point for snapshot (None for current time)
            snapshot_name: Optional name for snapshot
            **options: Additional snapshot options

        Returns:
            Temporal snapshot object
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="kg",
            submodule="GraphBuilder",
            message=f"Creating temporal snapshot: {snapshot_name or 'unnamed'}",
        )

        try:
            self.logger.info(
                f"Creating temporal snapshot: {snapshot_name or 'unnamed'}"
            )

            snapshot_time = self._parse_time(timestamp) or self._get_timestamp()

            self.progress_tracker.update_tracking(
                tracking_id, message="Filtering entities and relationships..."
            )

            # Filter entities and relationships valid at snapshot time
            entities = []
            relationships = []

            # Get all entities
            if "entities" in graph:
                entities = graph["entities"].copy()

            # Filter relationships valid at snapshot time
            if "relationships" in graph:
                for rel in graph["relationships"]:
                    valid_from = self._parse_time(rel.get("valid_from"))
                    valid_until = self._parse_time(rel.get("valid_until"))

                    # Check if relationship is valid at snapshot time
                    if (
                        valid_from
                        and self._compare_times(snapshot_time, valid_from) < 0
                    ):
                        continue
                    if (
                        valid_until
                        and self._compare_times(snapshot_time, valid_until) > 0
                    ):
                        continue

                    relationships.append(rel)

            snapshot = {
                "name": snapshot_name or f"snapshot_{snapshot_time}",
                "timestamp": snapshot_time,
                "entities": entities,
                "relationships": relationships,
                "metadata": {
                    "num_entities": len(entities),
                    "num_relationships": len(relationships),
                    "snapshot_time": snapshot_time,
                },
            }

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Created snapshot with {len(entities)} entities, {len(relationships)} relationships",
            )
            return snapshot

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def query_temporal(
        self,
        graph,
        query,
        at_time=None,
        time_range=None,
        temporal_window=None,
        **options,
    ):
        """
        Query graph at specific time point or time range.

        Args:
            graph: Knowledge graph to query
            query: Query (Cypher, SPARQL, or natural language)
            at_time: Query at specific time point
            time_range: Query within time range (start, end)
            temporal_window: Temporal window size
            **options: Additional query options

        Returns:
            Query results with temporal context
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="kg",
            submodule="GraphBuilder",
            message=f"Executing temporal query: {query[:50]}...",
        )

        try:
            self.logger.info(f"Executing temporal query: {query[:50]}...")

            self.progress_tracker.update_tracking(
                tracking_id, message="Creating temporal snapshot for query..."
            )

            # Create snapshot for query time
            if at_time:
                snapshot = self.create_temporal_snapshot(graph, timestamp=at_time)
            elif time_range:
                start_time, end_time = time_range
                # Query at end time
                snapshot = self.create_temporal_snapshot(graph, timestamp=end_time)
            else:
                # Use current graph
                snapshot = graph

            self.progress_tracker.update_tracking(
                tracking_id, message="Executing query..."
            )

            # Basic query execution (simplified)
            # In a real implementation, this would use a proper query engine
            results = {
                "query": query,
                "timestamp": at_time or (time_range[1] if time_range else None),
                "entities": snapshot.get("entities", []),
                "relationships": snapshot.get("relationships", []),
                "metadata": snapshot.get("metadata", {}),
            }

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Query executed: {len(results.get('entities', []))} entities, {len(results.get('relationships', []))} relationships",
            )
            return results

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def load_from_neo4j(
        self,
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
        enable_temporal=False,
        temporal_property="valid_time",
        **kwargs,
    ):
        """
        Load graph from Neo4j database.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            enable_temporal: Enable temporal features for loaded graph
            temporal_property: Property name for temporal data
            **kwargs: Additional connection options

        Returns:
            Knowledge graph loaded from Neo4j
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="kg",
            submodule="GraphBuilder",
            message=f"Loading graph from Neo4j: {uri}",
        )

        try:
            self.logger.info(f"Loading graph from Neo4j: {uri}")

            from neo4j import GraphDatabase

            self.progress_tracker.update_tracking(
                tracking_id, message="Connecting to Neo4j..."
            )
            driver = GraphDatabase.driver(uri, auth=(username, password))

            with driver.session(database=database) as session:
                # Load nodes
                self.progress_tracker.update_tracking(
                    tracking_id, message="Loading nodes from Neo4j..."
                )
                nodes_result = session.run("MATCH (n) RETURN n")
                entities = []
                for record in nodes_result:
                    node = record["n"]
                    entity = {
                        "id": str(node.id),
                        "type": list(node.labels)[0] if node.labels else "Entity",
                        "properties": dict(node),
                    }
                    entities.append(entity)

                # Load relationships
                self.progress_tracker.update_tracking(
                    tracking_id, message="Loading relationships from Neo4j..."
                )
                rels_result = session.run("MATCH (a)-[r]->(b) RETURN a, r, b")
                relationships = []
                for record in rels_result:
                    source = record["a"]
                    rel = record["r"]
                    target = record["b"]

                    relationship = {
                        "source": str(source.id),
                        "target": str(target.id),
                        "type": rel.type,
                        "properties": dict(rel),
                    }

                    # Add temporal information if enabled
                    if enable_temporal and temporal_property in rel:
                        relationship["valid_from"] = rel[temporal_property]

                    relationships.append(relationship)

            driver.close()

            graph = {
                "entities": entities,
                "relationships": relationships,
                "metadata": {
                    "source": "neo4j",
                    "uri": uri,
                    "database": database,
                    "temporal_enabled": enable_temporal,
                },
            }

            self.logger.info(
                f"Loaded {len(entities)} entities and {len(relationships)} relationships from Neo4j"
            )
            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Loaded {len(entities)} entities and {len(relationships)} relationships from Neo4j",
            )
            return graph

        except ImportError:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message="neo4j library not available"
            )
            raise ImportError(
                "neo4j library not available. Install with: pip install neo4j"
            )
        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            self.logger.error(f"Error loading from Neo4j: {e}")
            raise

    def _get_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.

        Returns:
            ISO format timestamp string (e.g., "2024-01-15T10:30:00")
        """
        return datetime.now().isoformat()

    def _parse_time(
        self, time_value: Optional[Union[str, datetime, Any]]
    ) -> Optional[str]:
        """
        Parse time value to ISO format string.

        This method handles various time input formats and converts them
        to a standardized ISO format string for temporal operations.

        Args:
            time_value: Time value in various formats:
                - None: Returns None
                - str: ISO format string (returned as-is)
                - datetime: Converted to ISO string
                - Other: Converted to string

        Returns:
            ISO format timestamp string or None
        """
        if time_value is None:
            return None

        # Already a string - assume it's in correct format
        if isinstance(time_value, str):
            return time_value

        # datetime object - convert to ISO string
        if isinstance(time_value, datetime):
            return time_value.isoformat()

        # Other types - convert to string
        return str(time_value)

    def _compare_times(self, time1: Optional[str], time2: Optional[str]) -> int:
        """
        Compare two time strings.

        This method performs lexicographic comparison of ISO format time strings.
        Returns -1 if time1 < time2, 0 if equal, 1 if time1 > time2.

        Args:
            time1: First time string (ISO format)
            time2: Second time string (ISO format)

        Returns:
            Comparison result: -1, 0, or 1
        """
        # Handle None values
        if time1 is None or time2 is None:
            return 0

        # Simple lexicographic comparison works for ISO format strings
        # ISO format: YYYY-MM-DDTHH:MM:SS ensures correct string comparison
        if time1 < time2:
            return -1
        elif time1 > time2:
            return 1
        else:
            return 0
