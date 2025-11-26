"""
KuzuDB Adapter Module

This module provides KuzuDB embedded graph database integration for property graph
storage and Cypher querying in the Semantica framework, supporting high-performance
analytical queries with in-memory and persistent storage.

Key Features:
    - Embedded graph database (no server required)
    - Full Cypher query language support
    - High-performance analytical queries
    - In-memory and persistent storage modes
    - Node table and relationship table management
    - Schema-based property graph model
    - COPY FROM for bulk data loading
    - Optional dependency handling

Main Classes:
    - KuzuAdapter: Main KuzuDB adapter for graph operations
    - KuzuDatabase: Database wrapper
    - KuzuConnection: Connection wrapper
    - KuzuQuery: Query execution wrapper

Example Usage:
    >>> from semantica.graph_store import KuzuAdapter
    >>> adapter = KuzuAdapter(database_path="./kuzu_db")
    >>> adapter.connect()
    >>> adapter.create_node_table("Person", {"name": "STRING", "age": "INT64"})
    >>> node_id = adapter.create_node("Person", {"name": "Alice", "age": 30})
    >>> results = adapter.execute_query("MATCH (p:Person) RETURN p.name")
    >>> adapter.close()

Author: Semantica Contributors
License: MIT
"""

import os
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker

# Optional KuzuDB import
try:
    import kuzu

    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False
    kuzu = None


class KuzuDatabase:
    """KuzuDB database wrapper."""

    def __init__(self, database: Any):
        """Initialize KuzuDB database wrapper."""
        self.database = database
        self.logger = get_logger("kuzu_database")

    def get_connection(self) -> "KuzuConnection":
        """
        Get a connection to the database.

        Returns:
            KuzuConnection instance
        """
        if not KUZU_AVAILABLE:
            raise ProcessingError("KuzuDB not available")

        try:
            conn = kuzu.Connection(self.database)
            return KuzuConnection(conn)
        except Exception as e:
            raise ProcessingError(f"Failed to create connection: {str(e)}")


class KuzuConnection:
    """KuzuDB connection wrapper."""

    def __init__(self, connection: Any):
        """Initialize KuzuDB connection wrapper."""
        self.connection = connection
        self.logger = get_logger("kuzu_connection")

    def execute(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> "KuzuQuery":
        """
        Execute a Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            KuzuQuery result wrapper
        """
        if not KUZU_AVAILABLE:
            raise ProcessingError("KuzuDB not available")

        try:
            if parameters:
                result = self.connection.execute(query, parameters)
            else:
                result = self.connection.execute(query)
            return KuzuQuery(result)
        except Exception as e:
            raise ProcessingError(f"Query execution failed: {str(e)}")

    def set_max_threads(self, num_threads: int) -> None:
        """Set maximum number of threads for query execution."""
        if self.connection and hasattr(self.connection, "set_max_threads_for_exec"):
            self.connection.set_max_threads_for_exec(num_threads)


class KuzuQuery:
    """KuzuDB query result wrapper."""

    def __init__(self, result: Any):
        """Initialize KuzuDB query result wrapper."""
        self.result = result
        self.logger = get_logger("kuzu_query")

    def has_next(self) -> bool:
        """Check if there are more results."""
        if self.result:
            return self.result.has_next()
        return False

    def get_next(self) -> List[Any]:
        """Get next result row."""
        if self.result:
            return self.result.get_next()
        return []

    def get_all(self) -> List[List[Any]]:
        """Get all results as a list of rows."""
        results = []
        while self.has_next():
            results.append(self.get_next())
        return results

    def get_column_names(self) -> List[str]:
        """Get column names from result."""
        if self.result and hasattr(self.result, "get_column_names"):
            return self.result.get_column_names()
        return []

    def get_column_types(self) -> List[str]:
        """Get column types from result."""
        if self.result and hasattr(self.result, "get_column_data_types"):
            return [str(t) for t in self.result.get_column_data_types()]
        return []


class KuzuAdapter:
    """
    KuzuDB adapter for embedded property graph storage and Cypher querying.

    • Embedded database (no server required)
    • Schema-based node and relationship tables
    • High-performance analytical queries
    • In-memory and persistent storage
    • Bulk data loading with COPY FROM
    • Performance optimization
    • Error handling and recovery
    """

    def __init__(
        self,
        database_path: Optional[str] = None,
        buffer_pool_size: Optional[int] = None,
        max_num_threads: int = 0,
        **config,
    ):
        """
        Initialize KuzuDB adapter.

        Args:
            database_path: Path to database directory
            buffer_pool_size: Buffer pool size in bytes
            max_num_threads: Maximum number of threads (0 = auto)
            **config: Additional configuration options
        """
        self.logger = get_logger("kuzu_adapter")
        self.config = config
        self.progress_tracker = get_progress_tracker()

        self.database_path = database_path or config.get("database_path", "./kuzu_db")
        self.buffer_pool_size = buffer_pool_size or config.get("buffer_pool_size", 268435456)
        self.max_num_threads = max_num_threads or config.get("max_num_threads", 0)

        self._database: Optional[KuzuDatabase] = None
        self._connection: Optional[KuzuConnection] = None

        # Track created tables for schema management
        self._node_tables: Dict[str, Dict[str, str]] = {}
        self._rel_tables: Dict[str, Dict[str, Any]] = {}

        # Check KuzuDB availability
        if not KUZU_AVAILABLE:
            self.logger.warning(
                "KuzuDB not available. Install with: pip install kuzu"
            )

    def connect(self, database_path: Optional[str] = None, **options) -> bool:
        """
        Connect to (or create) KuzuDB database.

        Args:
            database_path: Path to database directory
            **options: Connection options

        Returns:
            True if connected successfully
        """
        if not KUZU_AVAILABLE:
            raise ProcessingError(
                "KuzuDB is not available. Install it with: pip install kuzu"
            )

        database_path = database_path or self.database_path

        try:
            # Create directory if it doesn't exist
            os.makedirs(database_path, exist_ok=True)

            # Create database
            db = kuzu.Database(database_path, buffer_pool_size=self.buffer_pool_size)
            self._database = KuzuDatabase(db)

            # Create connection
            self._connection = self._database.get_connection()

            if self.max_num_threads > 0:
                self._connection.set_max_threads(self.max_num_threads)

            self.logger.info(f"Connected to KuzuDB at {database_path}")
            return True

        except Exception as e:
            raise ProcessingError(f"Failed to connect to KuzuDB: {str(e)}")

    def close(self) -> None:
        """Close connection to KuzuDB."""
        self._connection = None
        self._database = None
        self.logger.info("Disconnected from KuzuDB")

    def _ensure_connection(self) -> KuzuConnection:
        """Ensure connection is established."""
        if self._connection is None:
            self.connect()
        return self._connection

    def create_node_table(
        self,
        table_name: str,
        properties: Dict[str, str],
        primary_key: str = "id",
        **options,
    ) -> bool:
        """
        Create a node table with schema.

        Args:
            table_name: Name of the node table
            properties: Property schema {property_name: type}
                Types: STRING, INT64, INT32, DOUBLE, FLOAT, BOOLEAN, DATE, TIMESTAMP
            primary_key: Primary key property name
            **options: Additional options

        Returns:
            True if created successfully
        """
        try:
            conn = self._ensure_connection()

            # Build property list with primary key
            prop_list = []
            for prop_name, prop_type in properties.items():
                if prop_name == primary_key:
                    prop_list.insert(0, f"{prop_name} {prop_type} PRIMARY KEY")
                else:
                    prop_list.append(f"{prop_name} {prop_type}")

            # Ensure primary key is in properties
            if primary_key not in properties:
                prop_list.insert(0, f"{primary_key} SERIAL PRIMARY KEY")

            schema_def = ", ".join(prop_list)
            query = f"CREATE NODE TABLE IF NOT EXISTS {table_name}({schema_def})"

            conn.execute(query)
            self._node_tables[table_name] = properties
            self.logger.info(f"Created node table: {table_name}")
            return True

        except Exception as e:
            raise ProcessingError(f"Failed to create node table: {str(e)}")

    def create_rel_table(
        self,
        table_name: str,
        from_table: str,
        to_table: str,
        properties: Optional[Dict[str, str]] = None,
        **options,
    ) -> bool:
        """
        Create a relationship table.

        Args:
            table_name: Name of the relationship table
            from_table: Source node table name
            to_table: Target node table name
            properties: Property schema {property_name: type}
            **options: Additional options

        Returns:
            True if created successfully
        """
        try:
            conn = self._ensure_connection()

            # Build property list
            if properties:
                prop_list = [f"{name} {ptype}" for name, ptype in properties.items()]
                schema_def = ", " + ", ".join(prop_list)
            else:
                schema_def = ""

            query = f"CREATE REL TABLE IF NOT EXISTS {table_name}(FROM {from_table} TO {to_table}{schema_def})"

            conn.execute(query)
            self._rel_tables[table_name] = {
                "from": from_table,
                "to": to_table,
                "properties": properties or {},
            }
            self.logger.info(f"Created relationship table: {table_name}")
            return True

        except Exception as e:
            raise ProcessingError(f"Failed to create relationship table: {str(e)}")

    def create_node(
        self,
        table_name: str,
        properties: Dict[str, Any],
        **options,
    ) -> Dict[str, Any]:
        """
        Create a node in a table.

        Args:
            table_name: Node table name
            properties: Node properties
            **options: Additional options

        Returns:
            Created node information
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="graph_store",
            submodule="KuzuAdapter",
            message=f"Creating node in table {table_name}",
        )

        try:
            conn = self._ensure_connection()

            # Build property assignment
            prop_names = list(properties.keys())
            prop_values = []
            for value in properties.values():
                if isinstance(value, str):
                    prop_values.append(f"'{value}'")
                elif value is None:
                    prop_values.append("NULL")
                else:
                    prop_values.append(str(value))

            names_str = ", ".join(prop_names)
            values_str = ", ".join(prop_values)

            query = f"CREATE (n:{table_name} {{{names_str}: [{values_str}]}}) RETURN n"
            # Alternative simpler syntax
            query = f"CREATE (:{table_name} {{{', '.join(f'{k}: {repr(v) if isinstance(v, str) else v}' for k, v in properties.items())}}})"

            conn.execute(query)

            # Get the created node (KuzuDB uses SERIAL for auto-incrementing IDs)
            result = conn.execute(f"MATCH (n:{table_name}) WHERE n.{list(properties.keys())[0]} = {repr(list(properties.values())[0]) if isinstance(list(properties.values())[0], str) else list(properties.values())[0]} RETURN n")

            node_data = {
                "table": table_name,
                "properties": properties,
            }

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Created node in {table_name}",
            )
            return node_data

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise ProcessingError(f"Failed to create node: {str(e)}")

    def create_nodes(
        self,
        table_name: str,
        nodes: List[Dict[str, Any]],
        **options,
    ) -> List[Dict[str, Any]]:
        """
        Create multiple nodes in batch.

        Args:
            table_name: Node table name
            nodes: List of node property dictionaries
            **options: Additional options

        Returns:
            List of created node information
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="graph_store",
            submodule="KuzuAdapter",
            message=f"Creating {len(nodes)} nodes in table {table_name}",
        )

        try:
            conn = self._ensure_connection()
            created_nodes = []

            for node_props in nodes:
                props_str = ", ".join(
                    f"{k}: {repr(v) if isinstance(v, str) else v}"
                    for k, v in node_props.items()
                )
                query = f"CREATE (:{table_name} {{{props_str}}})"
                conn.execute(query)
                created_nodes.append({
                    "table": table_name,
                    "properties": node_props,
                })

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Created {len(created_nodes)} nodes",
            )
            return created_nodes

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise ProcessingError(f"Failed to create nodes: {str(e)}")

    def get_nodes(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **options,
    ) -> List[Dict[str, Any]]:
        """
        Get nodes from a table.

        Args:
            table_name: Node table name
            filters: Property filters
            limit: Maximum number of nodes
            **options: Additional options

        Returns:
            List of nodes
        """
        try:
            conn = self._ensure_connection()

            query = f"MATCH (n:{table_name})"

            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(f"n.{key} = '{value}'")
                    else:
                        conditions.append(f"n.{key} = {value}")
                query += " WHERE " + " AND ".join(conditions)

            query += f" RETURN n LIMIT {limit}"

            result = conn.execute(query)
            nodes = []

            while result.has_next():
                row = result.get_next()
                if row and len(row) > 0:
                    node = row[0]
                    if isinstance(node, dict):
                        nodes.append({
                            "table": table_name,
                            "properties": node,
                        })
                    else:
                        # Handle node object
                        nodes.append({
                            "table": table_name,
                            "properties": dict(node) if hasattr(node, "__iter__") else {"_raw": str(node)},
                        })

            return nodes

        except Exception as e:
            raise ProcessingError(f"Failed to get nodes: {str(e)}")

    def update_node(
        self,
        table_name: str,
        filters: Dict[str, Any],
        properties: Dict[str, Any],
        **options,
    ) -> bool:
        """
        Update node properties.

        Args:
            table_name: Node table name
            filters: Filters to identify node(s)
            properties: Properties to update
            **options: Additional options

        Returns:
            True if updated successfully
        """
        try:
            conn = self._ensure_connection()

            # Build WHERE clause
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f"n.{key} = '{value}'")
                else:
                    conditions.append(f"n.{key} = {value}")

            # Build SET clause
            updates = []
            for key, value in properties.items():
                if isinstance(value, str):
                    updates.append(f"n.{key} = '{value}'")
                else:
                    updates.append(f"n.{key} = {value}")

            query = f"MATCH (n:{table_name}) WHERE {' AND '.join(conditions)} SET {', '.join(updates)}"
            conn.execute(query)
            return True

        except Exception as e:
            raise ProcessingError(f"Failed to update node: {str(e)}")

    def delete_node(
        self,
        table_name: str,
        filters: Dict[str, Any],
        **options,
    ) -> bool:
        """
        Delete node(s).

        Args:
            table_name: Node table name
            filters: Filters to identify node(s)
            **options: Additional options

        Returns:
            True if deleted successfully
        """
        try:
            conn = self._ensure_connection()

            # Build WHERE clause
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f"n.{key} = '{value}'")
                else:
                    conditions.append(f"n.{key} = {value}")

            query = f"MATCH (n:{table_name}) WHERE {' AND '.join(conditions)} DETACH DELETE n"
            conn.execute(query)
            return True

        except Exception as e:
            raise ProcessingError(f"Failed to delete node: {str(e)}")

    def create_relationship(
        self,
        rel_table: str,
        from_table: str,
        from_filters: Dict[str, Any],
        to_table: str,
        to_filters: Dict[str, Any],
        properties: Optional[Dict[str, Any]] = None,
        **options,
    ) -> Dict[str, Any]:
        """
        Create a relationship between nodes.

        Args:
            rel_table: Relationship table name
            from_table: Source node table name
            from_filters: Filters to identify source node
            to_table: Target node table name
            to_filters: Filters to identify target node
            properties: Relationship properties
            **options: Additional options

        Returns:
            Created relationship information
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="graph_store",
            submodule="KuzuAdapter",
            message=f"Creating relationship [{rel_table}]",
        )

        try:
            conn = self._ensure_connection()

            # Build WHERE clauses
            from_conditions = []
            for key, value in from_filters.items():
                if isinstance(value, str):
                    from_conditions.append(f"a.{key} = '{value}'")
                else:
                    from_conditions.append(f"a.{key} = {value}")

            to_conditions = []
            for key, value in to_filters.items():
                if isinstance(value, str):
                    to_conditions.append(f"b.{key} = '{value}'")
                else:
                    to_conditions.append(f"b.{key} = {value}")

            # Build property string
            if properties:
                props_str = "{" + ", ".join(
                    f"{k}: {repr(v) if isinstance(v, str) else v}"
                    for k, v in properties.items()
                ) + "}"
            else:
                props_str = ""

            query = f"""
                MATCH (a:{from_table}), (b:{to_table})
                WHERE {' AND '.join(from_conditions)} AND {' AND '.join(to_conditions)}
                CREATE (a)-[:{rel_table} {props_str}]->(b)
            """

            conn.execute(query)

            rel_data = {
                "type": rel_table,
                "from_table": from_table,
                "to_table": to_table,
                "properties": properties or {},
            }

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Created relationship [{rel_table}]",
            )
            return rel_data

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise ProcessingError(f"Failed to create relationship: {str(e)}")

    def get_relationships(
        self,
        rel_table: Optional[str] = None,
        from_table: Optional[str] = None,
        to_table: Optional[str] = None,
        limit: int = 100,
        **options,
    ) -> List[Dict[str, Any]]:
        """
        Get relationships.

        Args:
            rel_table: Relationship table name
            from_table: Source node table filter
            to_table: Target node table filter
            limit: Maximum number of relationships
            **options: Additional options

        Returns:
            List of relationships
        """
        try:
            conn = self._ensure_connection()

            from_pattern = f":{from_table}" if from_table else ""
            to_pattern = f":{to_table}" if to_table else ""
            rel_pattern = f":{rel_table}" if rel_table else ""

            query = f"MATCH (a{from_pattern})-[r{rel_pattern}]->(b{to_pattern}) RETURN a, r, b LIMIT {limit}"

            result = conn.execute(query)
            relationships = []

            while result.has_next():
                row = result.get_next()
                if row and len(row) >= 3:
                    relationships.append({
                        "from_node": row[0],
                        "relationship": row[1],
                        "to_node": row[2],
                    })

            return relationships

        except Exception as e:
            raise ProcessingError(f"Failed to get relationships: {str(e)}")

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **options,
    ) -> Dict[str, Any]:
        """
        Execute a Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters
            **options: Additional options

        Returns:
            Query results
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="graph_store",
            submodule="KuzuAdapter",
            message="Executing Cypher query",
        )

        try:
            conn = self._ensure_connection()
            result = conn.execute(query, parameters)

            records = result.get_all()
            column_names = result.get_column_names()
            column_types = result.get_column_types()

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Query returned {len(records)} records",
            )

            return {
                "success": True,
                "records": records,
                "column_names": column_names,
                "column_types": column_types,
                "metadata": {"query": query},
            }

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise ProcessingError(f"Query execution failed: {str(e)}")

    def shortest_path(
        self,
        from_table: str,
        from_filters: Dict[str, Any],
        to_table: str,
        to_filters: Dict[str, Any],
        rel_table: Optional[str] = None,
        max_depth: int = 10,
        **options,
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two nodes.

        Args:
            from_table: Source node table
            from_filters: Filters to identify source node
            to_table: Target node table
            to_filters: Filters to identify target node
            rel_table: Relationship table filter
            max_depth: Maximum path length
            **options: Additional options

        Returns:
            Shortest path information or None
        """
        try:
            conn = self._ensure_connection()

            # Build WHERE clauses
            from_conditions = []
            for key, value in from_filters.items():
                if isinstance(value, str):
                    from_conditions.append(f"a.{key} = '{value}'")
                else:
                    from_conditions.append(f"a.{key} = {value}")

            to_conditions = []
            for key, value in to_filters.items():
                if isinstance(value, str):
                    to_conditions.append(f"b.{key} = '{value}'")
                else:
                    to_conditions.append(f"b.{key} = {value}")

            rel_pattern = f":{rel_table}" if rel_table else ""

            query = f"""
                MATCH (a:{from_table}), (b:{to_table}),
                path = SHORTEST 1 GROUPS (a)-[r{rel_pattern}*..{max_depth}]-(b)
                WHERE {' AND '.join(from_conditions)} AND {' AND '.join(to_conditions)}
                RETURN path, length(path) as length
            """

            result = conn.execute(query)

            if result.has_next():
                row = result.get_next()
                return {
                    "path": row[0],
                    "length": row[1] if len(row) > 1 else 0,
                }

            return None

        except Exception as e:
            # Kuzu might not support all path queries, try simpler query
            self.logger.warning(f"Shortest path query failed: {str(e)}")
            return None

    def bulk_load_nodes(
        self,
        table_name: str,
        file_path: str,
        **options,
    ) -> Dict[str, Any]:
        """
        Bulk load nodes from CSV file.

        Args:
            table_name: Node table name
            file_path: Path to CSV file
            **options: Additional options (header, delimiter, etc.)

        Returns:
            Load result information
        """
        try:
            conn = self._ensure_connection()

            header = options.get("header", True)
            delimiter = options.get("delimiter", ",")

            query = f"COPY {table_name} FROM '{file_path}' (HEADER={str(header).lower()}, DELIM='{delimiter}')"
            conn.execute(query)

            return {
                "success": True,
                "table": table_name,
                "file": file_path,
            }

        except Exception as e:
            raise ProcessingError(f"Bulk load failed: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            conn = self._ensure_connection()
            stats = {
                "node_tables": list(self._node_tables.keys()),
                "rel_tables": list(self._rel_tables.keys()),
                "database_path": self.database_path,
            }

            # Get node counts per table
            for table_name in self._node_tables.keys():
                try:
                    result = conn.execute(f"MATCH (n:{table_name}) RETURN count(n) as count")
                    if result.has_next():
                        row = result.get_next()
                        stats[f"{table_name}_count"] = row[0] if row else 0
                except Exception:
                    pass

            return stats

        except Exception as e:
            self.logger.warning(f"Failed to get stats: {str(e)}")
            return {"status": "error", "message": str(e)}

