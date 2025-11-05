"""
Milvus adapter for Semantica framework.

This module provides Milvus integration for vector storage
and similarity search.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger

# Optional Milvus import
try:
    from pymilvus import (
        connections, Collection, FieldSchema, CollectionSchema, DataType,
        utility, MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    connections = None
    Collection = None
    FieldSchema = None
    CollectionSchema = None
    DataType = None
    utility = None
    MilvusException = None


class MilvusClient:
    """Milvus client wrapper."""
    
    def __init__(self, alias: str = "default"):
        """Initialize Milvus client wrapper."""
        self.alias = alias
        self.logger = get_logger("milvus_client")
    
    def connect(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        **options
    ) -> bool:
        """Connect to Milvus server."""
        if not MILVUS_AVAILABLE:
            raise ProcessingError("Milvus not available")
        
        try:
            connections.connect(
                alias=self.alias,
                host=host,
                port=port,
                user=user,
                password=password,
                **options
            )
            return True
        except Exception as e:
            raise ProcessingError(f"Failed to connect to Milvus: {str(e)}")
    
    def disconnect(self):
        """Disconnect from Milvus server."""
        if not MILVUS_AVAILABLE:
            return
        
        try:
            connections.disconnect(self.alias)
        except Exception as e:
            self.logger.warning(f"Failed to disconnect: {str(e)}")


class MilvusCollection:
    """Milvus collection wrapper."""
    
    def __init__(self, collection: Any, collection_name: str):
        """Initialize Milvus collection wrapper."""
        self.collection = collection
        self.collection_name = collection_name
        self.logger = get_logger("milvus_collection")
    
    def insert(
        self,
        data: List[List[Any]],
        **options
    ) -> Any:
        """Insert data into collection."""
        if not MILVUS_AVAILABLE:
            raise ProcessingError("Milvus not available")
        
        try:
            insert_result = self.collection.insert(data, **options)
            return insert_result
        except Exception as e:
            raise ProcessingError(f"Failed to insert data: {str(e)}")
    
    def search(
        self,
        vectors: List[np.ndarray],
        anns_field: str,
        param: Dict[str, Any],
        limit: int = 10,
        expr: Optional[str] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """Search vectors in collection."""
        if not MILVUS_AVAILABLE:
            raise ProcessingError("Milvus not available")
        
        try:
            search_results = self.collection.search(
                data=[v.tolist() for v in vectors],
                anns_field=anns_field,
                param=param,
                limit=limit,
                expr=expr,
                **options
            )
            
            results = []
            for hits in search_results:
                batch_results = []
                for hit in hits:
                    batch_results.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": 1.0 - hit.distance if hit.distance <= 1.0 else 1.0 / (1.0 + hit.distance)
                    })
                results.append(batch_results)
            
            return results[0] if len(results) == 1 else results
        except Exception as e:
            raise ProcessingError(f"Failed to search: {str(e)}")
    
    def load(self, **options) -> bool:
        """Load collection into memory."""
        if not MILVUS_AVAILABLE:
            raise ProcessingError("Milvus not available")
        
        try:
            self.collection.load(**options)
            return True
        except Exception as e:
            raise ProcessingError(f"Failed to load collection: {str(e)}")
    
    def release(self) -> bool:
        """Release collection from memory."""
        if not MILVUS_AVAILABLE:
            raise ProcessingError("Milvus not available")
        
        try:
            self.collection.release()
            return True
        except Exception as e:
            raise ProcessingError(f"Failed to release collection: {str(e)}")


class MilvusSearch:
    """Milvus search operations."""
    
    def __init__(self, collection: MilvusCollection):
        """Initialize Milvus search."""
        self.collection = collection
        self.logger = get_logger("milvus_search")
    
    def similarity_search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        metric_type: str = "L2",
        expr: Optional[str] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search.
        
        Args:
            query_vector: Query vector
            limit: Number of results
            metric_type: Distance metric ("L2", "IP", "COSINE")
            expr: Filter expression
            **options: Additional options
            
        Returns:
            List of search results
        """
        search_params = {
            "metric_type": metric_type,
            "params": options.get("params", {"nprobe": 10})
        }
        
        return self.collection.search(
            vectors=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=limit,
            expr=expr,
            **options
        )


class MilvusAdapter:
    """
    Milvus adapter for vector storage and similarity search.
    
    • Milvus connection and authentication
    • Collection and partition management
    • Vector storage and retrieval
    • Similarity search and filtering
    • Performance optimization
    • Error handling and recovery
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: Optional[str] = None,
        password: Optional[str] = None,
        **config
    ):
        """Initialize Milvus adapter."""
        self.logger = get_logger("milvus_adapter")
        self.config = config
        self.host = host or config.get("host", "localhost")
        self.port = port or config.get("port", 19530)
        self.user = user or config.get("user")
        self.password = password or config.get("password")
        
        self.client: Optional[MilvusClient] = None
        self.collection: Optional[MilvusCollection] = None
        self.search_engine: Optional[MilvusSearch] = None
        
        # Check Milvus availability
        if not MILVUS_AVAILABLE:
            self.logger.warning(
                "Milvus not available. Install with: pip install pymilvus"
            )
    
    def connect(self, **options) -> bool:
        """
        Connect to Milvus service.
        
        Args:
            **options: Connection options
            
        Returns:
            True if connected successfully
        """
        if not MILVUS_AVAILABLE:
            raise ProcessingError(
                "Milvus is not available. Install it with: pip install pymilvus"
            )
        
        try:
            self.client = MilvusClient()
            self.client.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                **options
            )
            
            self.logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
        except Exception as e:
            raise ProcessingError(f"Failed to connect to Milvus: {str(e)}")
    
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric_type: str = "L2",
        **options
    ) -> MilvusCollection:
        """
        Create Milvus collection.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            metric_type: Distance metric ("L2", "IP", "COSINE")
            **options: Additional options
            
        Returns:
            MilvusCollection instance
        """
        if self.client is None:
            self.connect()
        
        if not MILVUS_AVAILABLE:
            raise ProcessingError("Milvus not available")
        
        try:
            # Check if collection exists
            if utility.has_collection(collection_name):
                self.logger.info(f"Collection {collection_name} already exists")
                return self.get_collection(collection_name)
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"Vector collection for {collection_name}"
            )
            
            # Create collection
            collection = Collection(
                name=collection_name,
                schema=schema
            )
            
            # Create index
            index_params = {
                "metric_type": metric_type,
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="vector", index_params=index_params)
            
            self.collection = MilvusCollection(collection, collection_name)
            self.search_engine = MilvusSearch(self.collection)
            
            self.logger.info(f"Created Milvus collection: {collection_name}")
            return self.collection
        
        except Exception as e:
            raise ProcessingError(f"Failed to create collection: {str(e)}")
    
    def get_collection(self, collection_name: str) -> MilvusCollection:
        """
        Get existing collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MilvusCollection instance
        """
        if self.client is None:
            self.connect()
        
        if not MILVUS_AVAILABLE:
            raise ProcessingError("Milvus not available")
        
        try:
            if not utility.has_collection(collection_name):
                raise ProcessingError(f"Collection {collection_name} does not exist")
            
            collection = Collection(collection_name)
            self.collection = MilvusCollection(collection, collection_name)
            self.search_engine = MilvusSearch(self.collection)
            return self.collection
        except Exception as e:
            raise ProcessingError(f"Failed to get collection: {str(e)}")
    
    def insert_vectors(
        self,
        vectors: List[Union[np.ndarray, List[float]]],
        **options
    ) -> Any:
        """
        Insert vectors into collection.
        
        Args:
            vectors: List of vectors
            **options: Additional options
            
        Returns:
            Insert result
        """
        if self.collection is None:
            raise ProcessingError("Collection not initialized. Call create_collection() or get_collection() first.")
        
        if not MILVUS_AVAILABLE:
            raise ProcessingError("Milvus not available")
        
        try:
            # Convert vectors to list format
            vector_data = []
            for vector in vectors:
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                vector_data.append(vector)
            
            data = [vector_data]
            return self.collection.insert(data, **options)
        
        except Exception as e:
            raise ProcessingError(f"Failed to insert vectors: {str(e)}")
    
    def search_vectors(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        metric_type: str = "L2",
        expr: Optional[str] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Search vectors in collection.
        
        Args:
            query_vector: Query vector
            limit: Number of results
            metric_type: Distance metric
            expr: Filter expression
            **options: Additional options
            
        Returns:
            List of search results
        """
        if self.search_engine is None:
            raise ProcessingError("Collection not initialized. Call create_collection() or get_collection() first.")
        
        # Load collection if not loaded
        if not self.collection.collection.has_index():
            self.collection.load()
        
        return self.search_engine.similarity_search(
            query_vector, limit, metric_type, expr, **options
        )
    
    def get_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.collection is None and collection_name:
            self.get_collection(collection_name)
        
        if self.collection is None:
            raise ProcessingError("Collection not initialized. Call create_collection() or get_collection() first.")
        
        try:
            stats = self.collection.collection.num_entities
            return {
                "entity_count": stats,
                "collection_name": self.collection.collection_name
            }
        except Exception as e:
            self.logger.warning(f"Failed to get stats: {str(e)}")
            return {"status": "unknown"}
