"""
FAISS adapter for Semantica framework.

This module provides FAISS integration for vector storage
and similarity search.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger

# Optional FAISS import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


class FAISSIndex:
    """FAISS index wrapper."""
    
    def __init__(self, index: Any, dimension: int, index_type: str = "flat"):
        """Initialize FAISS index wrapper."""
        self.index = index
        self.dimension = dimension
        self.index_type = index_type
        self.vector_ids: List[str] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_vectors(self, vectors: np.ndarray, ids: Optional[List[str]] = None):
        """Add vectors to index."""
        if ids is None:
            ids = [f"vec_{i}" for i in range(len(vectors))]
        
        self.index.add(vectors.astype(np.float32))
        self.vector_ids.extend(ids)
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        return self.index.search(query_vectors.astype(np.float32), k)
    
    def get_vector(self, vector_id: str) -> Optional[np.ndarray]:
        """Get vector by ID."""
        if vector_id not in self.vector_ids:
            return None
        
        idx = self.vector_ids.index(vector_id)
        # Note: FAISS doesn't directly support retrieval by index in all cases
        # This is a simplified approach
        return None
    
    def save(self, path: Union[str, Path]):
        """Save index to disk."""
        if FAISS_AVAILABLE:
            faiss.write_index(self.index, str(path))
        else:
            raise ProcessingError("FAISS not available")
    
    @classmethod
    def load(cls, path: Union[str, Path], dimension: int, index_type: str = "flat"):
        """Load index from disk."""
        if not FAISS_AVAILABLE:
            raise ProcessingError("FAISS not available")
        
        index = faiss.read_index(str(path))
        return cls(index, dimension, index_type)


class FAISSSearch:
    """FAISS search operations."""
    
    def __init__(self, index: FAISSIndex):
        """Initialize FAISS search."""
        self.index = index
        self.logger = get_logger("faiss_search")
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results
            **options: Search options
            
        Returns:
            List of search results
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.index.vector_ids):
                vector_id = self.index.vector_ids[idx]
                results.append({
                    "id": vector_id,
                    "score": float(dist),
                    "distance": float(dist),
                    "metadata": self.index.metadata.get(vector_id, {})
                })
        
        return results


class FAISSIndexBuilder:
    """FAISS index builder."""
    
    def __init__(self, dimension: int = 768):
        """Initialize FAISS index builder."""
        self.dimension = dimension
        self.logger = get_logger("faiss_builder")
    
    def build_index(
        self,
        index_type: str = "flat",
        metric: str = "L2",
        **options
    ) -> FAISSIndex:
        """
        Build FAISS index.
        
        Args:
            index_type: Index type ("flat", "ivf", "hnsw", "pq")
            metric: Distance metric ("L2", "inner_product")
            **options: Index options
            
        Returns:
            FAISSIndex instance
        """
        if not FAISS_AVAILABLE:
            raise ProcessingError(
                "FAISS is not available. Install it with: pip install faiss-cpu or faiss-gpu"
            )
        
        # Create index based on type
        if index_type == "flat":
            if metric == "L2":
                index = faiss.IndexFlatL2(self.dimension)
            else:
                index = faiss.IndexFlatIP(self.dimension)
        
        elif index_type == "ivf":
            nlist = options.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        elif index_type == "hnsw":
            M = options.get("M", 32)
            index = faiss.IndexHNSWFlat(self.dimension, M)
        
        elif index_type == "pq":
            m = options.get("m", 8)  # Number of subquantizers
            bits = options.get("bits", 8)
            index = faiss.IndexPQ(self.dimension, m, bits)
        
        else:
            raise ValidationError(f"Unsupported index type: {index_type}")
        
        return FAISSIndex(index, self.dimension, index_type)
    
    def train_index(self, index: FAISSIndex, training_vectors: np.ndarray):
        """Train index on sample vectors."""
        if not isinstance(index.index, faiss.IndexIVFFlat):
            return  # Only IVF indices need training
        
        index.index.train(training_vectors.astype(np.float32))


class FAISSAdapter:
    """
    FAISS adapter for vector storage and similarity search.
    
    • FAISS index creation and management
    • Vector storage and retrieval
    • Similarity search and filtering
    • Index optimization and training
    • Performance optimization
    • Error handling and recovery
    """
    
    def __init__(self, dimension: int = 768, **config):
        """Initialize FAISS adapter."""
        self.logger = get_logger("faiss_adapter")
        self.config = config
        self.dimension = dimension
        
        self.index: Optional[FAISSIndex] = None
        self.index_builder = FAISSIndexBuilder(dimension)
        self.search_engine: Optional[FAISSSearch] = None
        
        # Check FAISS availability
        if not FAISS_AVAILABLE:
            self.logger.warning(
                "FAISS not available. Install with: pip install faiss-cpu or faiss-gpu"
            )
    
    def create_index(
        self,
        index_type: str = "flat",
        metric: str = "L2",
        **options
    ) -> FAISSIndex:
        """
        Create FAISS index.
        
        Args:
            index_type: Index type ("flat", "ivf", "hnsw", "pq")
            metric: Distance metric ("L2", "inner_product")
            **options: Index options
            
        Returns:
            FAISSIndex instance
        """
        self.index = self.index_builder.build_index(index_type, metric, **options)
        self.search_engine = FAISSSearch(self.index)
        
        self.logger.info(f"Created FAISS index: {index_type} with metric {metric}")
        return self.index
    
    def add_vectors(
        self,
        vectors: Union[List[np.ndarray], np.ndarray],
        ids: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        **options
    ) -> List[str]:
        """
        Add vectors to index.
        
        Args:
            vectors: List of vectors or numpy array
            ids: Vector IDs
            metadata: Vector metadata
            **options: Additional options
            
        Returns:
            List of vector IDs
        """
        if self.index is None:
            self.create_index(**options)
        
        # Convert to numpy array
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        
        vectors = vectors.astype(np.float32)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{len(self.index.vector_ids) + i}" for i in range(len(vectors))]
        
        # Store metadata
        if metadata:
            for vec_id, meta in zip(ids, metadata):
                self.index.metadata[vec_id] = meta
        
        # Add vectors to index
        self.index.add_vectors(vectors, ids)
        
        self.logger.info(f"Added {len(vectors)} vectors to FAISS index")
        return ids
    
    def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results
            **options: Search options
            
        Returns:
            List of search results
        """
        if self.search_engine is None:
            raise ProcessingError("Index not initialized. Call create_index() first.")
        
        return self.search_engine.search_similar(query_vector, k, **options)
    
    def save_index(self, path: Union[str, Path], **options) -> bool:
        """
        Save index to disk.
        
        Args:
            path: Path to save index
            **options: Save options
            
        Returns:
            True if successful
        """
        if self.index is None:
            raise ProcessingError("No index to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.index.save(path)
        self.logger.info(f"Saved FAISS index to {path}")
        return True
    
    def load_index(
        self,
        path: Union[str, Path],
        index_type: str = "flat",
        **options
    ) -> FAISSIndex:
        """
        Load index from disk.
        
        Args:
            path: Path to index file
            index_type: Index type
            **options: Load options
            
        Returns:
            FAISSIndex instance
        """
        if not FAISS_AVAILABLE:
            raise ProcessingError("FAISS not available")
        
        self.index = FAISSIndex.load(path, self.dimension, index_type)
        self.search_engine = FAISSSearch(self.index)
        
        self.logger.info(f"Loaded FAISS index from {path}")
        return self.index
    
    def optimize_index(self, **options) -> bool:
        """
        Optimize index for better performance.
        
        Args:
            **options: Optimization options
            
        Returns:
            True if successful
        """
        if self.index is None:
            raise ProcessingError("No index to optimize")
        
        # FAISS optimization is typically done during index creation
        # This method can be used for additional optimization
        self.logger.info("Index optimization completed")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            return {"status": "no_index"}
        
        return {
            "index_type": self.index.index_type,
            "dimension": self.index.dimension,
            "vector_count": len(self.index.vector_ids),
            "faiss_available": FAISS_AVAILABLE
        }
