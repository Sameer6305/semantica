"""
Pinecone adapter for Semantica framework.

This module provides Pinecone integration for vector storage
and similarity search.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger

# Optional Pinecone import
try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec, PodSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None
    Pinecone = None
    ServerlessSpec = None
    PodSpec = None


class PineconeIndex:
    """Pinecone index wrapper."""
    
    def __init__(self, index: Any, index_name: str):
        """Initialize Pinecone index wrapper."""
        self.index = index
        self.index_name = index_name
        self.logger = get_logger("pinecone_index")
    
    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        namespace: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """Upsert vectors to index."""
        if not PINECONE_AVAILABLE:
            raise ProcessingError("Pinecone not available")
        
        try:
            response = self.index.upsert(
                vectors=vectors,
                namespace=namespace,
                **options
            )
            return response
        except Exception as e:
            raise ProcessingError(f"Failed to upsert vectors: {str(e)}")
    
    def query_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **options
    ) -> Dict[str, Any]:
        """Query similar vectors."""
        if not PINECONE_AVAILABLE:
            raise ProcessingError("Pinecone not available")
        
        try:
            response = self.index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                namespace=namespace,
                filter=filter,
                include_metadata=True,
                **options
            )
            return response
        except Exception as e:
            raise ProcessingError(f"Failed to query vectors: {str(e)}")
    
    def delete_vectors(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """Delete vectors from index."""
        if not PINECONE_AVAILABLE:
            raise ProcessingError("Pinecone not available")
        
        try:
            response = self.index.delete(ids=ids, namespace=namespace, **options)
            return response
        except Exception as e:
            raise ProcessingError(f"Failed to delete vectors: {str(e)}")
    
    def fetch_vectors(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """Fetch vectors by IDs."""
        if not PINECONE_AVAILABLE:
            raise ProcessingError("Pinecone not available")
        
        try:
            response = self.index.fetch(ids=ids, namespace=namespace, **options)
            return response
        except Exception as e:
            raise ProcessingError(f"Failed to fetch vectors: {str(e)}")
    
    def describe_index_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get index statistics."""
        if not PINECONE_AVAILABLE:
            raise ProcessingError("Pinecone not available")
        
        try:
            stats = self.index.describe_index_stats(namespace=namespace)
            return stats
        except Exception as e:
            raise ProcessingError(f"Failed to get index stats: {str(e)}")


class PineconeQuery:
    """Pinecone query builder."""
    
    def __init__(self, index: PineconeIndex):
        """Initialize Pinecone query builder."""
        self.index = index
        self.logger = get_logger("pinecone_query")
    
    def build_query(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **options
    ) -> Dict[str, Any]:
        """Build query parameters."""
        return {
            "vector": query_vector.tolist(),
            "top_k": top_k,
            "namespace": namespace,
            "filter": filter,
            **options
        }
    
    def execute(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute query and format results."""
        response = self.index.query_vectors(**query_params)
        
        results = []
        for match in response.get("matches", []):
            results.append({
                "id": match.get("id"),
                "score": match.get("score", 0.0),
                "metadata": match.get("metadata", {})
            })
        
        return results


class PineconeMetadata:
    """Pinecone metadata handler."""
    
    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize metadata."""
        # Pinecone metadata restrictions
        validated = {}
        
        for key, value in metadata.items():
            # Convert to allowed types
            if isinstance(value, (str, int, float, bool, list)):
                validated[key] = value
            elif isinstance(value, dict):
                # Nested dicts not directly supported
                validated[key] = str(value)
            else:
                validated[key] = str(value)
        
        return validated


class PineconeAdapter:
    """
    Pinecone adapter for vector storage and similarity search.
    
    • Pinecone connection and authentication
    • Vector storage and retrieval
    • Similarity search and filtering
    • Namespace and index management
    • Performance optimization
    • Error handling and recovery
    """
    
    def __init__(self, api_key: Optional[str] = None, environment: Optional[str] = None, **config):
        """Initialize Pinecone adapter."""
        self.logger = get_logger("pinecone_adapter")
        self.config = config
        self.api_key = api_key or config.get("api_key")
        self.environment = environment or config.get("environment")
        
        self.client: Optional[Any] = None
        self.index: Optional[PineconeIndex] = None
        self.query_builder: Optional[PineconeQuery] = None
        
        # Check Pinecone availability
        if not PINECONE_AVAILABLE:
            self.logger.warning(
                "Pinecone not available. Install with: pip install pinecone-client"
            )
    
    def connect(self, api_key: Optional[str] = None, **options) -> bool:
        """
        Connect to Pinecone service.
        
        Args:
            api_key: Pinecone API key
            **options: Connection options
            
        Returns:
            True if connected successfully
        """
        if not PINECONE_AVAILABLE:
            raise ProcessingError(
                "Pinecone is not available. Install it with: pip install pinecone-client"
            )
        
        api_key = api_key or self.api_key
        if not api_key:
            raise ValidationError("Pinecone API key is required")
        
        try:
            self.client = Pinecone(api_key=api_key)
            self.logger.info("Connected to Pinecone")
            return True
        except Exception as e:
            raise ProcessingError(f"Failed to connect to Pinecone: {str(e)}")
    
    def create_index(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        spec: Optional[Dict[str, Any]] = None,
        **options
    ) -> PineconeIndex:
        """
        Create new vector index.
        
        Args:
            index_name: Name of the index
            dimension: Vector dimension
            metric: Distance metric ("cosine", "euclidean", "dotproduct")
            spec: Index specification (serverless or pod)
            **options: Additional options
            
        Returns:
            PineconeIndex instance
        """
        if self.client is None:
            self.connect()
        
        if not PINECONE_AVAILABLE:
            raise ProcessingError("Pinecone not available")
        
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.client.list_indexes()]
            if index_name in existing_indexes:
                self.logger.info(f"Index {index_name} already exists")
                return self.get_index(index_name)
            
            # Create index specification
            if spec is None:
                spec = ServerlessSpec(cloud="aws", region="us-east-1")
            
            # Create index
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=spec,
                **options
            )
            
            self.logger.info(f"Created Pinecone index: {index_name}")
            return self.get_index(index_name)
        
        except Exception as e:
            raise ProcessingError(f"Failed to create index: {str(e)}")
    
    def get_index(self, index_name: str) -> PineconeIndex:
        """
        Get existing index.
        
        Args:
            index_name: Name of the index
            
        Returns:
            PineconeIndex instance
        """
        if self.client is None:
            self.connect()
        
        if not PINECONE_AVAILABLE:
            raise ProcessingError("Pinecone not available")
        
        try:
            index = self.client.Index(index_name)
            self.index = PineconeIndex(index, index_name)
            self.query_builder = PineconeQuery(self.index)
            return self.index
        except Exception as e:
            raise ProcessingError(f"Failed to get index: {str(e)}")
    
    def upsert_vectors(
        self,
        vectors: List[Union[np.ndarray, List[float]]],
        ids: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """
        Insert or update vectors.
        
        Args:
            vectors: List of vectors
            ids: Vector IDs
            metadata: Vector metadata
            namespace: Namespace name
            **options: Additional options
            
        Returns:
            Upsert response
        """
        if self.index is None:
            raise ProcessingError("Index not initialized. Call create_index() or get_index() first.")
        
        # Format vectors
        formatted_vectors = []
        for i, vector in enumerate(vectors):
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            
            vector_data = {"id": ids[i], "values": vector}
            
            if metadata and i < len(metadata):
                vector_data["metadata"] = PineconeMetadata.validate_metadata(metadata[i])
            
            formatted_vectors.append(vector_data)
        
        return self.index.upsert_vectors(formatted_vectors, namespace, **options)
    
    def query_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors.
        
        Args:
            query_vector: Query vector
            top_k: Number of results
            namespace: Namespace name
            filter: Metadata filter
            **options: Additional options
            
        Returns:
            List of search results
        """
        if self.query_builder is None:
            raise ProcessingError("Index not initialized. Call create_index() or get_index() first.")
        
        query_params = self.query_builder.build_query(
            query_vector, top_k, namespace, filter, **options
        )
        
        return self.query_builder.execute(query_params)
    
    def delete_vectors(
        self,
        ids: List[str],
        namespace: Optional[str] = None,
        **options
    ) -> Dict[str, Any]:
        """
        Delete vectors from index.
        
        Args:
            ids: Vector IDs to delete
            namespace: Namespace name
            **options: Additional options
            
        Returns:
            Delete response
        """
        if self.index is None:
            raise ProcessingError("Index not initialized. Call create_index() or get_index() first.")
        
        return self.index.delete_vectors(ids, namespace, **options)
    
    def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            raise ProcessingError("Index not initialized. Call create_index() or get_index() first.")
        
        stats = self.index.describe_index_stats(namespace)
        return {
            "total_vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "index_fullness": stats.get("index_fullness", 0.0),
            "namespaces": stats.get("namespaces", {})
        }
