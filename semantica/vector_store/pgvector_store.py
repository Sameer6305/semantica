"""
PgVector Store Module

This module provides PostgreSQL pgvector extension integration for vector storage and
similarity search in the Semantica framework, supporting various distance metrics,
index types (IVFFlat, HNSW), and efficient vector operations with JSONB metadata.

Key Features:
    - Distance metrics (Cosine, L2/Euclidean, Inner Product)
    - Index types (IVFFlat, HNSW)
    - Connection pooling with psycopg3/psycopg2
    - JSONB metadata storage and filtering
    - Batch vector operations
    - Idempotent index creation
    - Proper error handling and clean exceptions

Main Classes:
    - PgVectorStore: Main PgVector store for vector operations

Example Usage:
    >>> from semantica.vector_store import PgVectorStore
    >>> store = PgVectorStore(
    ...     connection_string="postgresql://user:pass@localhost/db",
    ...     table_name="vectors",
    ...     dimension=768,
    ...     distance_metric="cosine"
    ... )
    >>> store.add(vectors, metadata, ids)
    >>> results = store.search(query_vector, top_k=10)
    >>> store.create_index(index_type="hnsw", params={"m": 16, "ef_construction": 64})

Author: Semantica Contributors
License: MIT
"""

import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger

# Optional psycopg imports - prefer psycopg3, fallback to psycopg2
try:
    import psycopg
    from psycopg_pool import ConnectionPool

    PSYCOPG3_AVAILABLE = True
    PSYCOPG2_AVAILABLE = False
except (ImportError, OSError):
    PSYCOPG3_AVAILABLE = False
    try:
        import psycopg2
        from psycopg2 import pool

        PSYCOPG2_AVAILABLE = True
    except (ImportError, OSError):
        PSYCOPG2_AVAILABLE = False
        psycopg2 = None

# Optional pgvector import
try:
    import pgvector

    PGVECTOR_AVAILABLE = True
except (ImportError, OSError):
    PGVECTOR_AVAILABLE = False
    pgvector = None


class PgVectorStore:
    """
    PostgreSQL pgvector store for vector storage and similarity search.

    - Vector storage with pgvector extension
    - Similarity search with multiple distance metrics
    - JSONB metadata storage and filtering
    - Index creation (IVFFlat, HNSW)
    - Connection pooling
    - Batch operations support
    """

    SUPPORTED_METRICS = {"cosine", "l2", "inner_product"}
    SUPPORTED_INDEX_TYPES = {"ivfflat", "hnsw"}

    def __init__(
        self,
        connection_string: str,
        table_name: str,
        dimension: int,
        distance_metric: str = "cosine",
        **kwargs,
    ):
        """
        Initialize PgVectorStore.

        Args:
            connection_string: PostgreSQL connection string
            table_name: Name of the table to store vectors
            dimension: Vector dimension
            distance_metric: Distance metric (cosine, l2, inner_product)
            **kwargs: Additional options (pool_size, etc.)

        Raises:
            ValidationError: If distance_metric is not supported
            ProcessingError: If pgvector extension is not available
        """
        self.logger = get_logger("pgvector_store")

        # Validate dependencies
        if not PSYCOPG3_AVAILABLE and not PSYCOPG2_AVAILABLE:
            raise ProcessingError(
                "Neither psycopg3 nor psycopg2 is available. "
                "Install with: pip install psycopg[binary] or psycopg2-binary"
            )

        if not PGVECTOR_AVAILABLE:
            raise ProcessingError(
                "pgvector Python client is not available. "
                "Install with: pip install pgvector"
            )

        # Validate parameters
        if distance_metric.lower() not in self.SUPPORTED_METRICS:
            raise ValidationError(
                f"Unsupported distance metric: {distance_metric}. "
                f"Supported: {', '.join(self.SUPPORTED_METRICS)}"
            )

        self.connection_string = connection_string
        self.table_name = table_name
        self.dimension = dimension
        self.distance_metric = distance_metric.lower()
        self.config = kwargs

        # Connection pool settings
        self.pool_size = kwargs.get("pool_size", 10)
        self.max_overflow = kwargs.get("max_overflow", 10)

        # Initialize connection pool
        self._pool = None
        self._init_pool()

        # Verify pgvector extension is installed (do NOT auto-create)
        self._verify_pgvector_extension()

        # Ensure table exists
        self._ensure_table_exists()

        self.logger.info(
            f"Initialized PgVectorStore: table={table_name}, "
            f"dimension={dimension}, metric={distance_metric}"
        )

    def _init_pool(self):
        """Initialize connection pool."""
        try:
            if PSYCOPG3_AVAILABLE:
                self._pool = ConnectionPool(
                    self.connection_string,
                    min_size=1,
                    max_size=self.pool_size,
                )
            else:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=self.pool_size,
                    dsn=self.connection_string,
                )
            self.logger.debug("Connection pool initialized")
        except Exception as e:
            raise ProcessingError(f"Failed to initialize connection pool: {str(e)}")

    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool."""
        conn = None
        try:
            if PSYCOPG3_AVAILABLE:
                conn = self._pool.getconn()
            else:
                conn = self._pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise ProcessingError(f"Database connection error: {str(e)}")
        finally:
            if conn:
                if PSYCOPG3_AVAILABLE:
                    self._pool.putconn(conn)
                else:
                    self._pool.putconn(conn)

    def _verify_pgvector_extension(self):
        """Verify pgvector extension is installed. Fail clearly if not."""
        with self._get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                result = cur.fetchone()
                cur.close()

                if not result:
                    raise ProcessingError(
                        "pgvector extension is not installed in PostgreSQL. "
                        "Please install it first: https://github.com/pgvector/pgvector#installation"
                    )
                self.logger.debug("pgvector extension verified")
            except Exception as e:
                if "pgvector extension is not installed" in str(e):
                    raise
                raise ProcessingError(f"Failed to verify pgvector extension: {str(e)}")

    def _ensure_table_exists(self):
        """Ensure the vector table exists."""
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                vector VECTOR({self.dimension}),
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """

        with self._get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute(create_table_sql)
                conn.commit()
                cur.close()
                self.logger.debug(f"Table {self.table_name} ensured")
            except Exception as e:
                conn.rollback()
                raise ProcessingError(f"Failed to create table: {str(e)}")

    def add(
        self,
        vectors: Union[List[np.ndarray], np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors to the store.

        Args:
            vectors: List of vectors or numpy array
            metadata: List of metadata dictionaries (one per vector)
            ids: Optional list of IDs (auto-generated if not provided)

        Returns:
            List of vector IDs

        Raises:
            ValidationError: If input dimensions don't match
            ProcessingError: If database operation fails
        """
        # Convert to list if numpy array
        if isinstance(vectors, np.ndarray):
            vectors = [vectors[i] for i in range(len(vectors))]

        num_vectors = len(vectors)

        # Validate dimensions
        for i, vec in enumerate(vectors):
            if len(vec) != self.dimension:
                raise ValidationError(
                    f"Vector at index {i} has dimension {len(vec)}, "
                    f"expected {self.dimension}"
                )

        # Generate IDs if not provided
        if ids is None:
            # Get current count for sequential IDs
            with self._get_connection() as conn:
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                count = cur.fetchone()[0]
                cur.close()
            ids = [f"vec_{count + i}" for i in range(num_vectors)]

        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in range(num_vectors)]
        elif len(metadata) != num_vectors:
            raise ValidationError(
                f"Metadata length ({len(metadata)}) must match vectors length ({num_vectors})"
            )

        # Batch insert
        insert_sql = f"""
            INSERT INTO {self.table_name} (id, vector, metadata)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                vector = EXCLUDED.vector,
                metadata = EXCLUDED.metadata
        """

        with self._get_connection() as conn:
            try:
                cur = conn.cursor()
                for vec_id, vec, meta in zip(ids, vectors, metadata):
                    vec_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
                    cur.execute(insert_sql, (vec_id, vec_list, json.dumps(meta)))
                conn.commit()
                cur.close()
                self.logger.info(f"Added {num_vectors} vectors")
                return ids
            except Exception as e:
                conn.rollback()
                raise ProcessingError(f"Failed to add vectors: {str(e)}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            top_k: Number of results to return
            filter: Optional metadata filter (dict of key-value pairs)

        Returns:
            List of results with id, score, and metadata

        Raises:
            ValidationError: If query vector dimension doesn't match
            ProcessingError: If database operation fails
        """
        if len(query_vector) != self.dimension:
            raise ValidationError(
                f"Query vector has dimension {len(query_vector)}, expected {self.dimension}"
            )

        # Determine operator based on distance metric
        operator_map = {
            "cosine": "<=>",  # cosine distance
            "l2": "<->",  # L2 distance
            "inner_product": "<#>",  # negative inner product (for ordering)
        }
        op = operator_map[self.distance_metric]

        # Build query
        query_list = query_vector.tolist() if isinstance(query_vector, np.ndarray) else list(query_vector)

        if filter:
            # Build metadata filter condition
            filter_conditions = []
            filter_values = []
            for key, value in filter.items():
                filter_conditions.append(f"metadata->>{key} = %s")
                filter_values.append(str(value))

            where_clause = " AND ".join(filter_conditions)
            search_sql = f"""
                SELECT id, vector {op} %s::vector AS score, metadata
                FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY vector {op} %s::vector
                LIMIT %s
            """
            params = [query_list] + filter_values + [query_list, top_k]
        else:
            search_sql = f"""
                SELECT id, vector {op} %s::vector AS score, metadata
                FROM {self.table_name}
                ORDER BY vector {op} %s::vector
                LIMIT %s
            """
            params = [query_list, query_list, top_k]

        with self._get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute(search_sql, params)
                rows = cur.fetchall()
                cur.close()

                results = []
                for row in rows:
                    vec_id, score, meta = row
                    # Convert distance to similarity score
                    if self.distance_metric in ("cosine", "l2"):
                        # Lower distance = higher similarity
                        similarity = 1.0 / (1.0 + float(score))
                    else:  # inner_product
                        # Negative because we used <#> operator
                        similarity = -float(score)

                    results.append({
                        "id": vec_id,
                        "score": similarity,
                        "metadata": meta if isinstance(meta, dict) else json.loads(meta),
                    })

                return results
            except Exception as e:
                raise ProcessingError(f"Failed to search vectors: {str(e)}")

    def delete(self, ids: List[str]) -> bool:
        """
        Delete vectors by ID.

        Args:
            ids: List of vector IDs to delete

        Returns:
            True if successful

        Raises:
            ProcessingError: If database operation fails
        """
        if not ids:
            return True

        delete_sql = f"DELETE FROM {self.table_name} WHERE id = ANY(%s)"

        with self._get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute(delete_sql, (ids,))
                conn.commit()
                deleted_count = cur.rowcount
                cur.close()
                self.logger.info(f"Deleted {deleted_count} vectors")
                return True
            except Exception as e:
                conn.rollback()
                raise ProcessingError(f"Failed to delete vectors: {str(e)}")

    def update(
        self,
        ids: List[str],
        vectors: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> bool:
        """
        Update existing vectors.

        Args:
            ids: List of vector IDs to update
            vectors: Optional new vectors
            metadata: Optional new metadata

        Returns:
            True if successful

        Raises:
            ValidationError: If input dimensions don't match
            ProcessingError: If database operation fails
        """
        if not ids:
            return True

        if vectors is None and metadata is None:
            raise ValidationError("Either vectors or metadata must be provided for update")

        if vectors is not None:
            if isinstance(vectors, np.ndarray):
                vectors = [vectors[i] for i in range(len(vectors))]
            if len(vectors) != len(ids):
                raise ValidationError("Vectors length must match IDs length")

        if metadata is not None and len(metadata) != len(ids):
            raise ValidationError("Metadata length must match IDs length")

        with self._get_connection() as conn:
            try:
                cur = conn.cursor()

                for i, vec_id in enumerate(ids):
                    updates = []
                    params = []

                    if vectors is not None:
                        vec = vectors[i]
                        if len(vec) != self.dimension:
                            raise ValidationError(
                                f"Vector at index {i} has dimension {len(vec)}, expected {self.dimension}"
                            )
                        vec_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
                        updates.append("vector = %s")
                        params.append(vec_list)

                    if metadata is not None:
                        updates.append("metadata = %s")
                        params.append(json.dumps(metadata[i]))

                    update_sql = f"""
                        UPDATE {self.table_name}
                        SET {', '.join(updates)}
                        WHERE id = %s
                    """
                    params.append(vec_id)
                    cur.execute(update_sql, params)

                conn.commit()
                cur.close()
                self.logger.info(f"Updated {len(ids)} vectors")
                return True
            except Exception as e:
                conn.rollback()
                raise ProcessingError(f"Failed to update vectors: {str(e)}")

    def get(self, ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get vectors by ID.

        Args:
            ids: List of vector IDs

        Returns:
            List of dictionaries with id, vector, and metadata

        Raises:
            ProcessingError: If database operation fails
        """
        if not ids:
            return []

        get_sql = f"""
            SELECT id, vector, metadata
            FROM {self.table_name}
            WHERE id = ANY(%s)
        """

        with self._get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute(get_sql, (ids,))
                rows = cur.fetchall()
                cur.close()

                results = []
                for row in rows:
                    vec_id, vec, meta = row
                    results.append({
                        "id": vec_id,
                        "vector": np.array(vec) if vec else None,
                        "metadata": meta if isinstance(meta, dict) else json.loads(meta) if meta else {},
                    })

                return results
            except Exception as e:
                raise ProcessingError(f"Failed to get vectors: {str(e)}")

    def create_index(
        self,
        index_type: str = "hnsw",
        params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create an index on the vector column. Idempotent - safe to call multiple times.

        Args:
            index_type: Index type (ivfflat, hnsw)
            params: Index-specific parameters
                For IVFFlat: {"lists": 100}
                For HNSW: {"m": 16, "ef_construction": 64}

        Returns:
            True if successful

        Raises:
            ValidationError: If index_type is not supported
            ProcessingError: If database operation fails
        """
        index_type = index_type.lower()
        if index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValidationError(
                f"Unsupported index type: {index_type}. "
                f"Supported: {', '.join(self.SUPPORTED_INDEX_TYPES)}"
            )

        params = params or {}
        index_name = f"{self.table_name}_vector_{index_type}_idx"

        # Determine distance function
        distance_map = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }
        ops = distance_map[self.distance_metric]

        with self._get_connection() as conn:
            try:
                cur = conn.cursor()

                # Check if index already exists
                cur.execute("""
                    SELECT indexname FROM pg_indexes
                    WHERE indexname = %s AND tablename = %s
                """, (index_name, self.table_name))

                if cur.fetchone():
                    self.logger.info(f"Index {index_name} already exists")
                    cur.close()
                    return True

                # Build index creation SQL
                if index_type == "ivfflat":
                    lists = params.get("lists", 100)
                    create_sql = f"""
                        CREATE INDEX {index_name}
                        ON {self.table_name}
                        USING ivfflat (vector {ops})
                        WITH (lists = {lists})
                    """
                elif index_type == "hnsw":
                    m = params.get("m", 16)
                    ef_construction = params.get("ef_construction", 64)
                    create_sql = f"""
                        CREATE INDEX {index_name}
                        ON {self.table_name}
                        USING hnsw (vector {ops})
                        WITH (m = {m}, ef_construction = {ef_construction})
                    """

                cur.execute(create_sql)
                conn.commit()
                cur.close()
                self.logger.info(f"Created {index_type} index: {index_name}")
                return True
            except Exception as e:
                conn.rollback()
                # If index already exists (race condition), consider it success
                if "already exists" in str(e).lower():
                    self.logger.info(f"Index {index_name} already exists (race condition)")
                    return True
                raise ProcessingError(f"Failed to create index: {str(e)}")

    def drop_index(self, index_type: str = "hnsw") -> bool:
        """
        Drop a vector index.

        Args:
            index_type: Index type (ivfflat, hnsw)

        Returns:
            True if successful
        """
        index_name = f"{self.table_name}_vector_{index_type}_idx"

        with self._get_connection() as conn:
            try:
                cur = conn.cursor()
                cur.execute(f"DROP INDEX IF EXISTS {index_name}")
                conn.commit()
                cur.close()
                self.logger.info(f"Dropped index: {index_name}")
                return True
            except Exception as e:
                conn.rollback()
                raise ProcessingError(f"Failed to drop index: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get store statistics.

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            try:
                cur = conn.cursor()

                # Get row count
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
                count = cur.fetchone()[0]

                # Get index info
                cur.execute("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = %s
                """, (self.table_name,))
                indexes = [{"name": row[0], "definition": row[1]} for row in cur.fetchall()]

                cur.close()

                return {
                    "table_name": self.table_name,
                    "dimension": self.dimension,
                    "distance_metric": self.distance_metric,
                    "vector_count": count,
                    "indexes": indexes,
                    "psycopg_version": "3" if PSYCOPG3_AVAILABLE else "2",
                }
            except Exception as e:
                raise ProcessingError(f"Failed to get stats: {str(e)}")

    def close(self):
        """Close the connection pool."""
        if self._pool:
            if PSYCOPG3_AVAILABLE:
                self._pool.close()
            else:
                self._pool.closeall()
            self.logger.info("Connection pool closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
