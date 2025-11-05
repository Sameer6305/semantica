"""
Knowledge graph builder for Semantica framework.

This module provides knowledge graph construction capabilities
from extracted entities and relationships, with support for
temporal knowledge graphs.
"""


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
        **kwargs
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
        
        # Initialize graph building components
        from ..utils.logging import get_logger
        self.logger = get_logger("graph_builder")
        config = kwargs
        
        # Initialize entity resolver if needed
        if self.merge_entities:
            from .entity_resolver import EntityResolver
            self.entity_resolver = EntityResolver(
                strategy=self.entity_resolution_strategy,
                **kwargs.get("entity_resolution", {})
            )
        else:
            self.entity_resolver = None
        
        # Initialize conflict detector if needed
        if self.resolve_conflicts:
            from .conflict_detector import ConflictDetector
            self.conflict_detector = ConflictDetector(**kwargs.get("conflict_detection", {}))
        else:
            self.conflict_detector = None
    
    def build(self, sources, entity_resolver=None, **options):
        """
        Build knowledge graph from sources.
        
        Args:
            sources: List of sources (documents, entities, relationships)
            entity_resolver: Optional entity resolver
            **options: Additional build options
            
        Returns:
            Knowledge graph object
        """
        self.logger.info(f"Building knowledge graph from {len(sources)} sources")
        
        # Use provided resolver or default
        resolver = entity_resolver or self.entity_resolver
        
        # Extract entities and relationships
        entities = []
        relationships = []
        
        for source in sources:
            if isinstance(source, dict):
                # Extract entities and relationships from source
                if "entities" in source:
                    entities.extend(source["entities"])
                elif "id" in source or "entity_id" in source:
                    entities.append(source)
                
                if "relationships" in source:
                    relationships.extend(source["relationships"])
                elif "source" in source and "target" in source:
                    relationships.append(source)
            elif isinstance(source, list):
                # Assume list of entities or relationships
                for item in source:
                    if isinstance(item, dict):
                        if "source" in item and "target" in item:
                            relationships.append(item)
                        else:
                            entities.append(item)
        
        # Resolve entities if needed
        if resolver and entities:
            self.logger.info(f"Resolving {len(entities)} entities")
            entities = resolver.resolve_entities(entities)
            self.logger.info(f"Resolved to {len(entities)} unique entities")
        
        # Build graph structure
        graph = {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "num_entities": len(entities),
                "num_relationships": len(relationships),
                "temporal_enabled": self.enable_temporal,
                "timestamp": self._get_timestamp()
            }
        }
        
        # Detect and resolve conflicts if needed
        if self.conflict_detector:
            conflicts = self.conflict_detector.detect_conflicts(graph)
            if conflicts:
                self.logger.warning(f"Detected {len(conflicts)} conflicts")
                resolution = self.conflict_detector.resolve_conflicts(conflicts)
                self.logger.info(f"Resolved {resolution.get('resolved_count', 0)} conflicts")
        
        self.logger.info(f"Built graph with {len(entities)} entities and {len(relationships)} relationships")
        return graph
    
    def add_temporal_edge(
        self,
        graph,
        source,
        target,
        relationship,
        valid_from=None,
        valid_until=None,
        temporal_metadata=None,
        **kwargs
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
        self.logger.info(f"Adding temporal edge: {source} -{relationship}-> {target}")
        
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
            **kwargs
        }
        
        # Add to graph
        if "relationships" not in graph:
            graph["relationships"] = []
        graph["relationships"].append(edge)
        
        return edge
    
    def create_temporal_snapshot(self, graph, timestamp=None, snapshot_name=None, **options):
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
        self.logger.info(f"Creating temporal snapshot: {snapshot_name or 'unnamed'}")
        
        snapshot_time = self._parse_time(timestamp) or self._get_timestamp()
        
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
                if valid_from and self._compare_times(snapshot_time, valid_from) < 0:
                    continue
                if valid_until and self._compare_times(snapshot_time, valid_until) > 0:
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
                "snapshot_time": snapshot_time
            }
        }
        
        return snapshot
    
    def query_temporal(
        self,
        graph,
        query,
        at_time=None,
        time_range=None,
        temporal_window=None,
        **options
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
        self.logger.info(f"Executing temporal query: {query[:50]}...")
        
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
        
        # Basic query execution (simplified)
        # In a real implementation, this would use a proper query engine
        results = {
            "query": query,
            "timestamp": at_time or (time_range[1] if time_range else None),
            "entities": snapshot.get("entities", []),
            "relationships": snapshot.get("relationships", []),
            "metadata": snapshot.get("metadata", {})
        }
        
        return results
    
    def load_from_neo4j(
        self,
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        database="neo4j",
        enable_temporal=False,
        temporal_property="valid_time",
        **kwargs
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
        self.logger.info(f"Loading graph from Neo4j: {uri}")
        
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session(database=database) as session:
                # Load nodes
                nodes_result = session.run("MATCH (n) RETURN n")
                entities = []
                for record in nodes_result:
                    node = record["n"]
                    entity = {
                        "id": str(node.id),
                        "type": list(node.labels)[0] if node.labels else "Entity",
                        "properties": dict(node)
                    }
                    entities.append(entity)
                
                # Load relationships
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
                        "properties": dict(rel)
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
                    "temporal_enabled": enable_temporal
                }
            }
            
            self.logger.info(f"Loaded {len(entities)} entities and {len(relationships)} relationships from Neo4j")
            return graph
            
        except ImportError:
            raise ImportError("neo4j library not available. Install with: pip install neo4j")
        except Exception as e:
            self.logger.error(f"Error loading from Neo4j: {e}")
            raise
    
    def _get_timestamp(self):
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _parse_time(self, time_value):
        """Parse time value to ISO string."""
        from datetime import datetime
        
        if time_value is None:
            return None
        
        if isinstance(time_value, str):
            return time_value
        
        if isinstance(time_value, datetime):
            return time_value.isoformat()
        
        return str(time_value)
    
    def _compare_times(self, time1, time2):
        """Compare two time strings."""
        if time1 is None or time2 is None:
            return 0
        
        # Simple string comparison for ISO format
        return (time1 > time2) - (time1 < time2)
