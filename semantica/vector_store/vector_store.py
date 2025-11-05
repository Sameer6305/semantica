"""
Vector Store Module

Handles vector storage, indexing, and retrieval operations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


class VectorStore:
    """
    Vector store interface and management.
    
    • Stores and manages vector embeddings
    • Provides similarity search capabilities
    • Handles vector indexing and retrieval
    • Manages vector metadata and provenance
    • Supports multiple vector store backends
    • Provides vector store operations
    """
    
    def __init__(self, backend="faiss", config=None, **kwargs):
        """Initialize vector store."""
        self.logger = get_logger("vector_store")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.backend = backend
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.dimension = config.get("dimension", 768)
        
        # Initialize backend-specific indexer
        self.indexer = VectorIndexer(backend=backend, dimension=self.dimension, **config)
        self.retriever = VectorRetriever(backend=backend, **config)
    
    def store_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: Optional[List[Dict[str, Any]]] = None,
        **options
    ) -> List[str]:
        """
        Store vectors in vector store.
        
        Args:
            vectors: List of vector arrays
            metadata: List of metadata dictionaries
            **options: Storage options
            
        Returns:
            List of vector IDs
        """
        vector_ids = []
        metadata = metadata or [{}] * len(vectors)
        
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            vector_id = f"vec_{len(self.vectors) + i}"
            self.vectors[vector_id] = vector
            self.metadata[vector_id] = meta
            vector_ids.append(vector_id)
        
        # Update index
        self.indexer.create_index(list(self.vectors.values()), vector_ids)
        
        return vector_ids
    
    def search_vectors(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            **options: Search options
            
        Returns:
            List of search results with scores
        """
        if not self.vectors:
            return []
        
        # Use retriever for similarity search
        results = self.retriever.search_similar(
            query_vector,
            list(self.vectors.values()),
            list(self.vectors.keys()),
            k=k,
            **options
        )
        
        return results
    
    def update_vectors(
        self,
        vector_ids: List[str],
        new_vectors: List[np.ndarray],
        **options
    ) -> bool:
        """Update existing vectors."""
        for vec_id, new_vec in zip(vector_ids, new_vectors):
            if vec_id in self.vectors:
                self.vectors[vec_id] = new_vec
        
        # Rebuild index
        self.indexer.create_index(list(self.vectors.values()), list(self.vectors.keys()))
        
        return True
    
    def delete_vectors(self, vector_ids: List[str], **options) -> bool:
        """Delete vectors from store."""
        for vec_id in vector_ids:
            self.vectors.pop(vec_id, None)
            self.metadata.pop(vec_id, None)
        
        # Rebuild index
        if self.vectors:
            self.indexer.create_index(list(self.vectors.values()), list(self.vectors.keys()))
        
        return True
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get vector by ID."""
        return self.vectors.get(vector_id)
    
    def get_metadata(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for vector."""
        return self.metadata.get(vector_id)


class VectorIndexer:
    """Vector indexing engine."""
    
    def __init__(self, backend: str = "faiss", dimension: int = 768, **config):
        """Initialize vector indexer."""
        self.logger = get_logger("vector_indexer")
        self.config = config
        self.backend = backend
        self.dimension = dimension
        self.index = None
    
    def create_index(self, vectors: List[np.ndarray], ids: Optional[List[str]] = None, **options) -> Any:
        """
        Create vector index.
        
        Args:
            vectors: List of vectors
            ids: Vector IDs
            **options: Indexing options
            
        Returns:
            Index object
        """
        if not vectors:
            return None
        
        # Convert to numpy array
        if isinstance(vectors[0], list):
            vectors = np.array(vectors)
        else:
            vectors = np.vstack(vectors)
        
        # Simple in-memory index (would use FAISS, etc. in production)
        self.index = {
            "vectors": vectors,
            "ids": ids or list(range(len(vectors)))
        }
        
        return self.index
    
    def update_index(self, index: Any, new_vectors: List[np.ndarray], **options) -> Any:
        """Update existing index."""
        # Simplified - rebuild index
        return self.create_index(
            list(index["vectors"]) + new_vectors,
            index["ids"] + [f"new_{i}" for i in range(len(new_vectors))]
        )
    
    def optimize_index(self, index: Any, **options) -> Any:
        """Optimize index for better performance."""
        # Simplified - return as-is
        return index


class VectorRetriever:
    """Vector retrieval engine."""
    
    def __init__(self, backend: str = "faiss", **config):
        """Initialize vector retriever."""
        self.logger = get_logger("vector_retriever")
        self.config = config
        self.backend = backend
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        vectors: List[np.ndarray],
        ids: List[str],
        k: int = 10,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            vectors: List of vectors to search
            ids: Vector IDs
            k: Number of results
            
        Returns:
            List of results with scores
        """
        if not vectors:
            return []
        
        # Convert to numpy
        if isinstance(vectors[0], list):
            vectors = np.array(vectors)
        else:
            vectors = np.vstack(vectors)
        
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
        
        # Calculate cosine similarity
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(vectors, axis=1)
        
        similarities = np.dot(vectors, query_vector) / (vector_norms * query_norm)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "id": ids[idx],
                "vector": vectors[idx],
                "score": float(similarities[idx])
            })
        
        return results
    
    def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        **options
    ) -> List[Dict[str, Any]]:
        """Search vectors by metadata."""
        results = []
        
        for vec, meta in zip(vectors, metadata):
            match = True
            for key, value in metadata_filters.items():
                if key not in meta or meta[key] != value:
                    match = False
                    break
            
            if match:
                results.append({"vector": vec, "metadata": meta})
        
        return results
    
    def search_hybrid(
        self,
        query_vector: np.ndarray,
        metadata_filters: Dict[str, Any],
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        **options
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search."""
        # Filter by metadata first
        filtered = self.search_by_metadata(metadata_filters, vectors, metadata)
        
        if not filtered:
            return []
        
        # Then search by similarity
        filtered_vectors = [r["vector"] for r in filtered]
        filtered_ids = [i for i in range(len(filtered_vectors))]
        
        return self.search_similar(query_vector, filtered_vectors, filtered_ids, **options)


class VectorManager:
    """Vector store management engine."""
    
    def __init__(self, **config):
        """Initialize vector manager."""
        self.logger = get_logger("vector_manager")
        self.config = config
    
    def manage_store(self, store: VectorStore, **operations: Dict[str, Any]) -> Dict[str, Any]:
        """Manage vector store operations."""
        results = {}
        
        for op_name, op_config in operations.items():
            if op_name == "optimize":
                results["optimize"] = self.maintain_store(store)
            elif op_name == "statistics":
                results["statistics"] = self.collect_statistics(store)
        
        return results
    
    def maintain_store(self, store: VectorStore, **options: Dict[str, Any]) -> Dict[str, Any]:
        """Maintain vector store health."""
        # Check integrity
        vector_count = len(store.vectors)
        metadata_count = len(store.metadata)
        
        return {
            "healthy": vector_count == metadata_count,
            "vector_count": vector_count,
            "metadata_count": metadata_count
        }
    
    def collect_statistics(self, store: VectorStore) -> Dict[str, Any]:
        """Collect vector store statistics."""
        return {
            "total_vectors": len(store.vectors),
            "dimension": store.dimension,
            "backend": store.backend
        }
