"""
Qdrant adapter for Semantica framework.

This module provides Qdrant integration for vector storage
and similarity search.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger

# Optional Qdrant import
try:
    from qdrant_client import QdrantClient as QdrantClientLib
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, FieldCondition,
        MatchValue, CollectionStatus
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClientLib = None
    Distance = None
    VectorParams = None
    PointStruct = None
    Filter = None
    FieldCondition = None
    MatchValue = None
    CollectionStatus = None


class QdrantClient:
    """Qdrant client wrapper."""
    
    def __init__(self, client: Any):
        """Initialize Qdrant client wrapper."""
        self.client = client
        self.logger = get_logger("qdrant_client")
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        **options
    ) -> bool:
        """Create a collection in Qdrant."""
        if not QDRANT_AVAILABLE:
            raise ProcessingError("Qdrant not available")
        
        try:
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclidean": Distance.EUCLID,
                "Dot": Distance.DOT
            }
            
            distance_enum = distance_map.get(distance, Distance.COSINE)
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_enum
                ),
                **options
            )
            return True
        except Exception as e:
            raise ProcessingError(f"Failed to create collection: {str(e)}")
    
    def get_collection(self, collection_name: str) -> Any:
        """Get collection info."""
        if not QDRANT_AVAILABLE:
            raise ProcessingError("Qdrant not available")
        
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            raise ProcessingError(f"Failed to get collection: {str(e)}")


class QdrantCollection:
    """Qdrant collection wrapper."""
    
    def __init__(self, client: Any, collection_name: str):
        """Initialize Qdrant collection wrapper."""
        self.client = client
        self.collection_name = collection_name
        self.logger = get_logger("qdrant_collection")
    
    def upsert_points(
        self,
        points: List[PointStruct],
        **options
    ) -> Dict[str, Any]:
        """Upsert points to collection."""
        if not QDRANT_AVAILABLE:
            raise ProcessingError("Qdrant not available")
        
        try:
            response = self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                **options
            )
            return {"status": response.status}
        except Exception as e:
            raise ProcessingError(f"Failed to upsert points: {str(e)}")
    
    def search_points(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """Search for similar points."""
        if not QDRANT_AVAILABLE:
            raise ProcessingError("Qdrant not available")
        
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
                **options
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload or {}
                })
            
            return results
        except Exception as e:
            raise ProcessingError(f"Failed to search points: {str(e)}")
    
    def delete_points(self, point_ids: List[Union[str, int]], **options) -> Dict[str, Any]:
        """Delete points from collection."""
        if not QDRANT_AVAILABLE:
            raise ProcessingError("Qdrant not available")
        
        try:
            response = self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
                **options
            )
            return {"status": response.status}
        except Exception as e:
            raise ProcessingError(f"Failed to delete points: {str(e)}")


class QdrantSearch:
    """Qdrant search operations."""
    
    def __init__(self, collection: QdrantCollection):
        """Initialize Qdrant search."""
        self.collection = collection
        self.logger = get_logger("qdrant_search")
    
    def similarity_search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search.
        
        Args:
            query_vector: Query vector
            limit: Number of results
            filter: Metadata filter
            **options: Additional options
            
        Returns:
            List of search results
        """
        query_filter = None
        if filter and QDRANT_AVAILABLE:
            # Build Qdrant filter from dict
            conditions = []
            for key, value in filter.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                query_filter = Filter(must=conditions)
        
        return self.collection.search_points(query_vector, limit, query_filter, **options)


class QdrantAdapter:
    """
    Qdrant adapter for vector storage and similarity search.
    
    • Qdrant connection and authentication
    • Collection and point management
    • Vector storage and retrieval
    • Similarity search and filtering
    • Performance optimization
    • Error handling and recovery
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        **config
    ):
        """Initialize Qdrant adapter."""
        self.logger = get_logger("qdrant_adapter")
        self.config = config
        self.url = url or config.get("url", "http://localhost:6333")
        self.api_key = api_key or config.get("api_key")
        
        self.client: Optional[Any] = None
        self.collection: Optional[QdrantCollection] = None
        self.search_engine: Optional[QdrantSearch] = None
        
        # Check Qdrant availability
        if not QDRANT_AVAILABLE:
            self.logger.warning(
                "Qdrant not available. Install with: pip install qdrant-client"
            )
    
    def connect(self, url: Optional[str] = None, api_key: Optional[str] = None, **options) -> bool:
        """
        Connect to Qdrant service.
        
        Args:
            url: Qdrant URL
            api_key: API key for authentication
            **options: Connection options
            
        Returns:
            True if connected successfully
        """
        if not QDRANT_AVAILABLE:
            raise ProcessingError(
                "Qdrant is not available. Install it with: pip install qdrant-client"
            )
        
        url = url or self.url
        api_key = api_key or self.api_key
        
        try:
            if api_key:
                self.client = QdrantClientLib(url=url, api_key=api_key, **options)
            else:
                self.client = QdrantClientLib(url=url, **options)
            
            self.logger.info(f"Connected to Qdrant at {url}")
            return True
        except Exception as e:
            raise ProcessingError(f"Failed to connect to Qdrant: {str(e)}")
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "Cosine",
        **options
    ) -> QdrantCollection:
        """
        Create Qdrant collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Vector dimension
            distance: Distance metric ("Cosine", "Euclidean", "Dot")
            **options: Additional options
            
        Returns:
            QdrantCollection instance
        """
        if self.client is None:
            self.connect()
        
        if not QDRANT_AVAILABLE:
            raise ProcessingError("Qdrant not available")
        
        try:
            client_wrapper = QdrantClient(self.client)
            client_wrapper.create_collection(collection_name, vector_size, distance, **options)
            
            self.collection = QdrantCollection(self.client, collection_name)
            self.search_engine = QdrantSearch(self.collection)
            
            self.logger.info(f"Created Qdrant collection: {collection_name}")
            return self.collection
        
        except Exception as e:
            raise ProcessingError(f"Failed to create collection: {str(e)}")
    
    def get_collection(self, collection_name: str) -> QdrantCollection:
        """
        Get existing collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            QdrantCollection instance
        """
        if self.client is None:
            self.connect()
        
        if not QDRANT_AVAILABLE:
            raise ProcessingError("Qdrant not available")
        
        try:
            client_wrapper = QdrantClient(self.client)
            client_wrapper.get_collection(collection_name)
            
            self.collection = QdrantCollection(self.client, collection_name)
            self.search_engine = QdrantSearch(self.collection)
            return self.collection
        except Exception as e:
            raise ProcessingError(f"Failed to get collection: {str(e)}")
    
    def insert_vectors(
        self,
        vectors: List[Union[np.ndarray, List[float]]],
        ids: List[Union[str, int]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        **options
    ) -> Dict[str, Any]:
        """
        Insert vectors into collection.
        
        Args:
            vectors: List of vectors
            ids: Point IDs
            payloads: Optional metadata payloads
            **options: Additional options
            
        Returns:
            Insert response
        """
        if self.collection is None:
            raise ProcessingError("Collection not initialized. Call create_collection() or get_collection() first.")
        
        if not QDRANT_AVAILABLE:
            raise ProcessingError("Qdrant not available")
        
        try:
            points = []
            for i, (vector, point_id) in enumerate(zip(vectors, ids)):
                if isinstance(vector, np.ndarray):
                    vector = vector.tolist()
                
                payload = payloads[i] if payloads and i < len(payloads) else {}
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            
            return self.collection.upsert_points(points, **options)
        
        except Exception as e:
            raise ProcessingError(f"Failed to insert vectors: {str(e)}")
    
    def search_vectors(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Search vectors in collection.
        
        Args:
            query_vector: Query vector
            limit: Number of results
            filter: Metadata filter
            **options: Additional options
            
        Returns:
            List of search results
        """
        if self.search_engine is None:
            raise ProcessingError("Collection not initialized. Call create_collection() or get_collection() first.")
        
        return self.search_engine.similarity_search(query_vector, limit, filter, **options)
    
    def delete_vectors(
        self,
        point_ids: List[Union[str, int]],
        **options
    ) -> Dict[str, Any]:
        """
        Delete vectors from collection.
        
        Args:
            point_ids: Point IDs to delete
            **options: Additional options
            
        Returns:
            Delete response
        """
        if self.collection is None:
            raise ProcessingError("Collection not initialized. Call create_collection() or get_collection() first.")
        
        return self.collection.delete_points(point_ids, **options)
    
    def get_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get collection statistics."""
        if self.collection is None and collection_name:
            self.get_collection(collection_name)
        
        if self.collection is None:
            raise ProcessingError("Collection not initialized. Call create_collection() or get_collection() first.")
        
        try:
            collection_info = self.client.get_collection(self.collection.collection_name)
            return {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": str(collection_info.status) if hasattr(collection_info, 'status') else "unknown"
            }
        except Exception as e:
            self.logger.warning(f"Failed to get stats: {str(e)}")
            return {"status": "unknown"}
