"""
Graph Store Core Module

This module provides the core graph store interface and management classes,
providing a unified interface across multiple graph database backends
(Neo4j, KuzuDB, FalkorDB).

Key Features:
    - Unified graph store interface
    - Multi-backend support
    - Node and relationship management
    - Query execution engine
    - Graph analytics and algorithms
    - Transaction support
    - Configuration management

Main Classes:
    - GraphStore: Main graph store interface
    - GraphManager: Graph store management and operations
    - NodeManager: Node CRUD operations
    - RelationshipManager: Relationship CRUD operations
    - QueryEngine: Query execution and optimization
    - GraphAnalytics: Graph algorithms and analytics

Example Usage:
    >>> from semantica.graph_store import GraphStore
    >>> store = GraphStore(backend="neo4j", uri="bolt://localhost:7687")
    >>> store.create_node(labels=["Person"], properties={"name": "Alice"})
    >>> results = store.execute_query("MATCH (n) RETURN n LIMIT 10")

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .config import graph_store_config


class NodeManager:
    """Manager for node CRUD operations."""

    def __init__(self, adapter: Any):
        """
        Initialize node manager.

        Args:
            adapter: Graph database adapter instance
        """
        self.adapter = adapter
        self.logger = get_logger("node_manager")

    def create(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        **options,
    ) -> Dict[str, Any]:
        """
        Create a node.

        Args:
            labels: Node labels
            properties: Node properties
            **options: Additional options

        Returns:
            Created node information
        """
        return self.adapter.create_node(labels, properties, **options)

    def create_batch(
        self,
        nodes: List[Dict[str, Any]],
        **options,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple nodes in batch.

        Args:
            nodes: List of node dictionaries with 'labels' and 'properties'
            **options: Additional options

        Returns:
            List of created node information
        """
        return self.adapter.create_nodes(nodes, **options)

    def get(
        self,
        node_id: Union[int, str] = None,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **options,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get node(s).

        Args:
            node_id: Specific node ID to get
            labels: Filter by labels
            properties: Filter by properties
            limit: Maximum number of nodes
            **options: Additional options

        Returns:
            Node or list of nodes
        """
        if node_id is not None:
            return self.adapter.get_node(node_id, **options)
        return self.adapter.get_nodes(labels, properties, limit, **options)

    def update(
        self,
        node_id: Union[int, str],
        properties: Dict[str, Any],
        merge: bool = True,
        **options,
    ) -> Dict[str, Any]:
        """
        Update a node.

        Args:
            node_id: Node ID
            properties: Properties to update
            merge: If True, merge; if False, replace
            **options: Additional options

        Returns:
            Updated node information
        """
        return self.adapter.update_node(node_id, properties, merge, **options)

    def delete(
        self,
        node_id: Union[int, str],
        detach: bool = True,
        **options,
    ) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node ID
            detach: If True, delete relationships as well
            **options: Additional options

        Returns:
            True if deleted
        """
        return self.adapter.delete_node(node_id, detach, **options)


class RelationshipManager:
    """Manager for relationship CRUD operations."""

    def __init__(self, adapter: Any):
        """
        Initialize relationship manager.

        Args:
            adapter: Graph database adapter instance
        """
        self.adapter = adapter
        self.logger = get_logger("relationship_manager")

    def create(
        self,
        start_node_id: Union[int, str],
        end_node_id: Union[int, str],
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
        **options,
    ) -> Dict[str, Any]:
        """
        Create a relationship.

        Args:
            start_node_id: Start node ID
            end_node_id: End node ID
            rel_type: Relationship type
            properties: Relationship properties
            **options: Additional options

        Returns:
            Created relationship information
        """
        return self.adapter.create_relationship(
            start_node_id, end_node_id, rel_type, properties, **options
        )

    def get(
        self,
        node_id: Optional[Union[int, str]] = None,
        rel_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
        **options,
    ) -> List[Dict[str, Any]]:
        """
        Get relationships.

        Args:
            node_id: Filter by node ID
            rel_type: Filter by relationship type
            direction: Direction ("in", "out", "both")
            limit: Maximum number of relationships
            **options: Additional options

        Returns:
            List of relationships
        """
        return self.adapter.get_relationships(node_id, rel_type, direction, limit, **options)

    def delete(
        self,
        rel_id: Union[int, str],
        **options,
    ) -> bool:
        """
        Delete a relationship.

        Args:
            rel_id: Relationship ID
            **options: Additional options

        Returns:
            True if deleted
        """
        return self.adapter.delete_relationship(rel_id, **options)


class QueryEngine:
    """Engine for query execution and optimization."""

    def __init__(self, adapter: Any):
        """
        Initialize query engine.

        Args:
            adapter: Graph database adapter instance
        """
        self.adapter = adapter
        self.logger = get_logger("query_engine")
        self._cache: Dict[str, Any] = {}
        self._cache_enabled = True

    def execute(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
        **options,
    ) -> Dict[str, Any]:
        """
        Execute a Cypher/OpenCypher query.

        Args:
            query: Query string
            parameters: Query parameters
            use_cache: Whether to use query caching
            **options: Additional options

        Returns:
            Query results
        """
        # Check cache
        if use_cache and self._cache_enabled:
            cache_key = self._generate_cache_key(query, parameters)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Execute query
        result = self.adapter.execute_query(query, parameters, **options)

        # Cache result
        if use_cache and self._cache_enabled:
            self._cache[cache_key] = result

        return result

    def _generate_cache_key(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]],
    ) -> str:
        """Generate cache key for query."""
        import hashlib
        key_str = f"{query}:{str(parameters)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear query cache."""
        self._cache.clear()

    def enable_cache(self) -> None:
        """Enable query caching."""
        self._cache_enabled = True

    def disable_cache(self) -> None:
        """Disable query caching."""
        self._cache_enabled = False


class GraphAnalytics:
    """Graph analytics and algorithms."""

    def __init__(self, adapter: Any):
        """
        Initialize graph analytics.

        Args:
            adapter: Graph database adapter instance
        """
        self.adapter = adapter
        self.logger = get_logger("graph_analytics")

    def shortest_path(
        self,
        start_node_id: Union[int, str],
        end_node_id: Union[int, str],
        rel_type: Optional[str] = None,
        max_depth: int = 10,
        **options,
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two nodes.

        Args:
            start_node_id: Starting node ID
            end_node_id: Ending node ID
            rel_type: Filter by relationship type
            max_depth: Maximum path length
            **options: Additional options

        Returns:
            Path information or None
        """
        return self.adapter.shortest_path(start_node_id, end_node_id, rel_type, max_depth, **options)

    def get_neighbors(
        self,
        node_id: Union[int, str],
        rel_type: Optional[str] = None,
        direction: str = "both",
        depth: int = 1,
        **options,
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes.

        Args:
            node_id: Starting node ID
            rel_type: Filter by relationship type
            direction: Direction ("in", "out", "both")
            depth: Traversal depth
            **options: Additional options

        Returns:
            List of neighboring nodes
        """
        return self.adapter.get_neighbors(node_id, rel_type, direction, depth, **options)

    def degree_centrality(
        self,
        labels: Optional[List[str]] = None,
        rel_type: Optional[str] = None,
        direction: str = "both",
        **options,
    ) -> List[Dict[str, Any]]:
        """
        Calculate degree centrality for nodes.

        Args:
            labels: Filter by node labels
            rel_type: Filter by relationship type
            direction: Direction ("in", "out", "both")
            **options: Additional options

        Returns:
            List of nodes with centrality scores
        """
        # Build query based on direction
        if labels:
            label_str = ":".join(labels)
            match = f"MATCH (n:{label_str})"
        else:
            match = "MATCH (n)"

        type_filter = f":{rel_type}" if rel_type else ""

        if direction == "out":
            query = f"""
                {match}
                OPTIONAL MATCH (n)-[r{type_filter}]->()
                WITH n, count(r) as degree
                RETURN id(n) as id, n, degree
                ORDER BY degree DESC
            """
        elif direction == "in":
            query = f"""
                {match}
                OPTIONAL MATCH (n)<-[r{type_filter}]-()
                WITH n, count(r) as degree
                RETURN id(n) as id, n, degree
                ORDER BY degree DESC
            """
        else:
            query = f"""
                {match}
                OPTIONAL MATCH (n)-[r{type_filter}]-()
                WITH n, count(r) as degree
                RETURN id(n) as id, n, degree
                ORDER BY degree DESC
            """

        result = self.adapter.execute_query(query)
        return result.get("records", [])

    def connected_components(
        self,
        labels: Optional[List[str]] = None,
        **options,
    ) -> Dict[str, Any]:
        """
        Find connected components in the graph.

        Note: Full connected components requires GDS library for Neo4j
        or custom implementation. This provides a basic approximation.

        Args:
            labels: Filter by node labels
            **options: Additional options

        Returns:
            Component information
        """
        self.logger.warning(
            "Full connected components algorithm requires graph data science extensions. "
            "Returning basic component approximation."
        )

        # Get all nodes and their connections
        if labels:
            label_str = ":".join(labels)
            query = f"MATCH (n:{label_str})-[r]-(m) RETURN DISTINCT id(n) as node_id, id(m) as connected_id"
        else:
            query = "MATCH (n)-[r]-(m) RETURN DISTINCT id(n) as node_id, id(m) as connected_id"

        result = self.adapter.execute_query(query)

        return {
            "message": "Connected components approximation",
            "connections": result.get("records", []),
        }


class GraphManager:
    """Manager for graph store operations."""

    def __init__(self, adapter: Any):
        """
        Initialize graph manager.

        Args:
            adapter: Graph database adapter instance
        """
        self.adapter = adapter
        self.logger = get_logger("graph_manager")
        self.nodes = NodeManager(adapter)
        self.relationships = RelationshipManager(adapter)
        self.query_engine = QueryEngine(adapter)
        self.analytics = GraphAnalytics(adapter)

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return self.adapter.get_stats()

    def create_index(
        self,
        label: str,
        property_name: str,
        index_type: str = "btree",
        **options,
    ) -> bool:
        """
        Create an index.

        Args:
            label: Node label
            property_name: Property to index
            index_type: Index type
            **options: Additional options

        Returns:
            True if created
        """
        return self.adapter.create_index(label, property_name, index_type, **options)


class GraphStore:
    """
    Main graph store interface.

    Provides a unified interface for working with property graph databases,
    supporting Neo4j, KuzuDB, and FalkorDB backends.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        **config,
    ):
        """
        Initialize graph store.

        Args:
            backend: Backend type ("neo4j", "kuzu", "falkordb")
            **config: Backend-specific configuration
        """
        self.logger = get_logger("graph_store")
        self.progress_tracker = get_progress_tracker()

        # Determine backend
        self.backend = backend or config.get("backend") or graph_store_config.get("default_backend", "neo4j")
        self.config = config

        # Initialize adapter
        self._adapter = None
        self._manager = None
        self._initialize_adapter()

    def _initialize_adapter(self) -> None:
        """Initialize the appropriate adapter based on backend."""
        if self.backend == "neo4j":
            from .neo4j_adapter import Neo4jAdapter
            neo4j_config = graph_store_config.get_neo4j_config()
            neo4j_config.update(self.config)
            self._adapter = Neo4jAdapter(**neo4j_config)

        elif self.backend == "kuzu":
            from .kuzu_adapter import KuzuAdapter
            kuzu_config = graph_store_config.get_kuzu_config()
            kuzu_config.update(self.config)
            self._adapter = KuzuAdapter(**kuzu_config)

        elif self.backend == "falkordb":
            from .falkordb_adapter import FalkorDBAdapter
            falkordb_config = graph_store_config.get_falkordb_config()
            falkordb_config.update(self.config)
            self._adapter = FalkorDBAdapter(**falkordb_config)

        else:
            raise ValidationError(f"Unknown backend: {self.backend}")

        self._manager = GraphManager(self._adapter)

    def connect(self, **options) -> bool:
        """
        Connect to the graph database.

        Args:
            **options: Connection options

        Returns:
            True if connected
        """
        return self._adapter.connect(**options)

    def close(self) -> None:
        """Close connection to the graph database."""
        if self._adapter:
            self._adapter.close()

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    # Node operations
    def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        **options,
    ) -> Dict[str, Any]:
        """Create a node."""
        return self._manager.nodes.create(labels, properties, **options)

    def create_nodes(
        self,
        nodes: List[Dict[str, Any]],
        **options,
    ) -> List[Dict[str, Any]]:
        """Create multiple nodes."""
        return self._manager.nodes.create_batch(nodes, **options)

    def get_node(
        self,
        node_id: Union[int, str],
        **options,
    ) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        return self._manager.nodes.get(node_id=node_id, **options)

    def get_nodes(
        self,
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **options,
    ) -> List[Dict[str, Any]]:
        """Get nodes matching criteria."""
        return self._manager.nodes.get(labels=labels, properties=properties, limit=limit, **options)

    def update_node(
        self,
        node_id: Union[int, str],
        properties: Dict[str, Any],
        merge: bool = True,
        **options,
    ) -> Dict[str, Any]:
        """Update a node."""
        return self._manager.nodes.update(node_id, properties, merge, **options)

    def delete_node(
        self,
        node_id: Union[int, str],
        detach: bool = True,
        **options,
    ) -> bool:
        """Delete a node."""
        return self._manager.nodes.delete(node_id, detach, **options)

    # Relationship operations
    def create_relationship(
        self,
        start_node_id: Union[int, str],
        end_node_id: Union[int, str],
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
        **options,
    ) -> Dict[str, Any]:
        """Create a relationship."""
        return self._manager.relationships.create(
            start_node_id, end_node_id, rel_type, properties, **options
        )

    def get_relationships(
        self,
        node_id: Optional[Union[int, str]] = None,
        rel_type: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
        **options,
    ) -> List[Dict[str, Any]]:
        """Get relationships."""
        return self._manager.relationships.get(node_id, rel_type, direction, limit, **options)

    def delete_relationship(
        self,
        rel_id: Union[int, str],
        **options,
    ) -> bool:
        """Delete a relationship."""
        return self._manager.relationships.delete(rel_id, **options)

    # Query operations
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options,
    ) -> Dict[str, Any]:
        """Execute a Cypher query."""
        return self._manager.query_engine.execute(query, parameters, **options)

    # Analytics operations
    def shortest_path(
        self,
        start_node_id: Union[int, str],
        end_node_id: Union[int, str],
        rel_type: Optional[str] = None,
        max_depth: int = 10,
        **options,
    ) -> Optional[Dict[str, Any]]:
        """Find shortest path between nodes."""
        return self._manager.analytics.shortest_path(
            start_node_id, end_node_id, rel_type, max_depth, **options
        )

    def get_neighbors(
        self,
        node_id: Union[int, str],
        rel_type: Optional[str] = None,
        direction: str = "both",
        depth: int = 1,
        **options,
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes."""
        return self._manager.analytics.get_neighbors(
            node_id, rel_type, direction, depth, **options
        )

    # Management operations
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return self._manager.get_stats()

    def create_index(
        self,
        label: str,
        property_name: str,
        index_type: str = "btree",
        **options,
    ) -> bool:
        """Create an index."""
        return self._manager.create_index(label, property_name, index_type, **options)

    @property
    def nodes(self) -> NodeManager:
        """Get node manager."""
        return self._manager.nodes

    @property
    def relationships(self) -> RelationshipManager:
        """Get relationship manager."""
        return self._manager.relationships

    @property
    def query_engine(self) -> QueryEngine:
        """Get query engine."""
        return self._manager.query_engine

    @property
    def analytics(self) -> GraphAnalytics:
        """Get analytics engine."""
        return self._manager.analytics

