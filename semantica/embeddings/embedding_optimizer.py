"""
Embedding optimizer for Semantica framework.

This module provides optimization utilities for
embedding generation and storage.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class EmbeddingOptimizer:
    """Embedding optimization utilities."""
    
    def __init__(self, **config):
        """
        Initialize embedding optimizer.
        
        Args:
            **config: Configuration options:
                - compression_method: Compression method ("pca", "quantization")
                - target_dimension: Target dimension for reduction
        """
        self.logger = get_logger("embedding_optimizer")
        self.config = config
        
        self.compression_method = config.get("compression_method", "pca")
        self.target_dimension = config.get("target_dimension", 128)
    
    def compress(self, embeddings: np.ndarray, **options) -> np.ndarray:
        """
        Compress embeddings.
        
        Args:
            embeddings: Input embeddings (n_embeddings, dim)
            **options: Compression options:
                - target_dim: Target dimension
                - method: Compression method
                
        Returns:
            np.ndarray: Compressed embeddings
        """
        target_dim = options.get("target_dim", self.target_dimension)
        method = options.get("method", self.compression_method)
        
        if method == "pca":
            return self._compress_pca(embeddings, target_dim)
        elif method == "quantization":
            return self._compress_quantization(embeddings, **options)
        else:
            return embeddings
    
    def _compress_pca(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Compress using PCA."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("sklearn not available for PCA. Returning original embeddings.")
            return embeddings
        
        if embeddings.shape[1] <= target_dim:
            return embeddings
        
        try:
            pca = PCA(n_components=target_dim)
            compressed = pca.fit_transform(embeddings)
            return compressed.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Failed to compress with PCA: {e}")
            return embeddings
    
    def _compress_quantization(self, embeddings: np.ndarray, **options) -> np.ndarray:
        """Compress using quantization."""
        bits = options.get("bits", 8)
        
        # Quantize to specified bit depth
        min_val = np.min(embeddings)
        max_val = np.max(embeddings)
        
        # Scale to [0, 2^bits - 1]
        scale = (2 ** bits - 1) / (max_val - min_val) if max_val != min_val else 1
        quantized = ((embeddings - min_val) * scale).astype(np.uint8)
        
        # Dequantize
        dequantized = (quantized.astype(np.float32) / scale) + min_val
        
        return dequantized
    
    def reduce_dimensions(self, embeddings: np.ndarray, target_dim: int, **options) -> np.ndarray:
        """
        Reduce embedding dimensions.
        
        Args:
            embeddings: Input embeddings
            target_dim: Target dimension
            **options: Reduction options:
                - method: Reduction method ("pca", "truncate")
                
        Returns:
            np.ndarray: Reduced embeddings
        """
        method = options.get("method", "pca")
        
        if method == "pca":
            return self._compress_pca(embeddings, target_dim)
        elif method == "truncate":
            return embeddings[:, :target_dim]
        else:
            return embeddings
    
    def optimize_batch_processing(self, embeddings: List[np.ndarray], **options) -> np.ndarray:
        """
        Optimize batch processing of embeddings.
        
        Args:
            embeddings: List of embeddings
            **options: Optimization options
            
        Returns:
            np.ndarray: Optimized batch embeddings
        """
        # Convert to array
        emb_array = np.array(embeddings)
        
        # Apply optimizations
        if options.get("normalize", True):
            norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            emb_array = emb_array / norms
        
        return emb_array
    
    def profile_performance(self, embeddings: np.ndarray, **options) -> Dict[str, Any]:
        """
        Profile embedding performance.
        
        Args:
            embeddings: Input embeddings
            **options: Profiling options
            
        Returns:
            dict: Performance metrics
        """
        metrics = {
            "shape": embeddings.shape,
            "dtype": str(embeddings.dtype),
            "memory_mb": embeddings.nbytes / (1024 * 1024),
            "mean": float(np.mean(embeddings)),
            "std": float(np.std(embeddings)),
            "min": float(np.min(embeddings)),
            "max": float(np.max(embeddings))
        }
        
        # Sparsity
        non_zero = np.count_nonzero(embeddings)
        total = embeddings.size
        metrics["sparsity"] = 1.0 - (non_zero / total) if total > 0 else 0.0
        
        return metrics
