"""
Embedding Generation Module

Handles generation of embeddings for text and other data types.

Key Features:
    - Text embedding generation
    - Multi-modal embedding support
    - Embedding optimization and fine-tuning
    - Batch embedding processing
    - Embedding quality assessment

Main Classes:
    - EmbeddingGenerator: Main embedding generation class
    - TextEmbedder: Text-specific embedding generator
    - MultiModalEmbedder: Multi-modal embedding support
    - EmbeddingOptimizer: Embedding optimization engine
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder
from .audio_embedder import AudioEmbedder
from .multimodal_embedder import MultimodalEmbedder
from .embedding_optimizer import EmbeddingOptimizer


class EmbeddingGenerator:
    """
    Embedding generation handler.
    
    • Generates embeddings for text and data
    • Supports multiple embedding models
    • Handles batch embedding processing
    • Optimizes embedding quality
    • Manages embedding metadata
    • Supports various embedding formats
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize embedding generator."""
        self.logger = get_logger("embedding_generator")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Initialize embedders
        self.text_embedder = TextEmbedder(**self.config.get("text", {}))
        self.image_embedder = ImageEmbedder(**self.config.get("image", {}))
        self.audio_embedder = AudioEmbedder(**self.config.get("audio", {}))
        self.multimodal_embedder = MultimodalEmbedder(**self.config.get("multimodal", {}))
        self.embedding_optimizer = EmbeddingOptimizer(**self.config.get("optimizer", {}))
        
        # Supported models
        self.supported_models = ["sentence-transformers", "openai", "bge", "clip"]
    
    def generate_embeddings(self, data: Union[str, Path, List], data_type: Optional[str] = None, **options) -> np.ndarray:
        """
        Generate embeddings for data.
        
        Args:
            data: Input data (text, file path, or list)
            data_type: Data type ("text", "image", "audio", auto-detected if None)
            **options: Generation options
            
        Returns:
            np.ndarray: Generated embeddings
        """
        # Auto-detect data type
        if data_type is None:
            data_type = self._detect_data_type(data)
        
        if data_type == "text":
            if isinstance(data, str):
                return self.text_embedder.embed_text(data, **options)
            elif isinstance(data, list):
                return self.text_embedder.embed_batch(data, **options)
        elif data_type == "image":
            if isinstance(data, (str, Path)):
                return self.image_embedder.embed_image(data, **options)
            elif isinstance(data, list):
                return self.image_embedder.embed_batch(data, **options)
        elif data_type == "audio":
            if isinstance(data, (str, Path)):
                return self.audio_embedder.embed_audio(data, **options)
            elif isinstance(data, list):
                return self.audio_embedder.embed_batch(data, **options)
        else:
            raise ProcessingError(f"Unsupported data type: {data_type}")
    
    def _detect_data_type(self, data: Union[str, Path, List]) -> str:
        """Detect data type from input."""
        if isinstance(data, list):
            if not data:
                return "text"
            # Check first item
            return self._detect_data_type(data[0])
        
        if isinstance(data, Path) or (isinstance(data, str) and Path(data).exists()):
            file_path = Path(data)
            suffix = file_path.suffix.lower()
            
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
            audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg']
            
            if suffix in image_extensions:
                return "image"
            elif suffix in audio_extensions:
                return "audio"
        
        return "text"
    
    def optimize_embeddings(self, embeddings: np.ndarray, **options) -> np.ndarray:
        """
        Optimize embedding quality and performance.
        
        Args:
            embeddings: Input embeddings
            **options: Optimization options
            
        Returns:
            np.ndarray: Optimized embeddings
        """
        return self.embedding_optimizer.compress(embeddings, **options)
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray, **options) -> float:
        """
        Compare embeddings for similarity.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            **options: Comparison options:
                - method: Similarity method ("cosine", "euclidean")
                
        Returns:
            float: Similarity score (0-1)
        """
        method = options.get("method", "cosine")
        
        if method == "cosine":
            return self._cosine_similarity(embedding1, embedding2)
        elif method == "euclidean":
            return self._euclidean_similarity(embedding1, embedding2)
        else:
            return self._cosine_similarity(embedding1, embedding2)
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def _euclidean_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute Euclidean similarity (converted to 0-1 scale)."""
        distance = np.linalg.norm(emb1 - emb2)
        # Normalize to 0-1 (simple approach)
        max_distance = np.linalg.norm(emb1) + np.linalg.norm(emb2)
        similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 0.0
        return max(0.0, min(1.0, similarity))
    
    def process_batch(self, data_items: List[Union[str, Path]], **options) -> Dict[str, Any]:
        """
        Process multiple data items for embedding generation.
        
        Args:
            data_items: List of data items
            **options: Processing options
            
        Returns:
            dict: Batch processing results
        """
        results = {
            "embeddings": [],
            "successful": [],
            "failed": []
        }
        
        for item in data_items:
            try:
                embedding = self.generate_embeddings(item, **options)
                results["embeddings"].append(embedding)
                results["successful"].append(str(item))
            except Exception as e:
                results["failed"].append({
                    "item": str(item),
                    "error": str(e)
                })
        
        results["total"] = len(data_items)
        results["success_count"] = len(results["successful"])
        results["failure_count"] = len(results["failed"])
        
        return results


# Re-export classes from other modules for convenience
from .text_embedder import TextEmbedder as TextEmbedderImpl
from .multimodal_embedder import MultimodalEmbedder as MultimodalEmbedderImpl
from .embedding_optimizer import EmbeddingOptimizer as EmbeddingOptimizerImpl

# Make classes available with same names
TextEmbedder = TextEmbedderImpl
MultiModalEmbedder = MultimodalEmbedderImpl
EmbeddingOptimizer = EmbeddingOptimizerImpl
