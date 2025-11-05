"""
Text embedder for Semantica framework.

This module provides text embedding generation using
sentence-transformers and other embedding models.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TextEmbedder:
    """Text embedding generator."""
    
    def __init__(self, **config):
        """
        Initialize text embedder.
        
        Args:
            **config: Configuration options:
                - model_name: Model name (default: "all-MiniLM-L6-v2")
                - device: Device ("cpu", "cuda")
                - normalize: Normalize embeddings (default: True)
        """
        self.logger = get_logger("text_embedder")
        self.config = config
        
        self.model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self.device = config.get("device", "cpu")
        self.normalize = config.get("normalize", True)
        
        # Initialize model
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.logger.info(f"Loaded sentence-transformers model: {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load model {self.model_name}: {e}")
        else:
            self.logger.warning("sentence-transformers not available. Using fallback embedding.")
    
    def embed_text(self, text: str, **options) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            **options: Embedding options
            
        Returns:
            np.ndarray: Text embedding vector
        """
        if not text:
            raise ProcessingError("Text cannot be empty")
        
        if self.model:
            return self._embed_with_model(text, **options)
        else:
            return self._embed_fallback(text, **options)
    
    def _embed_with_model(self, text: str, **options) -> np.ndarray:
        """Embed text using sentence-transformers model."""
        embeddings = self.model.encode(
            [text],
            normalize_embeddings=self.normalize,
            **options
        )
        
        return embeddings[0]
    
    def _embed_fallback(self, text: str, **options) -> np.ndarray:
        """Fallback embedding using simple hash-based method."""
        # Simple hash-based embedding (for testing without dependencies)
        import hashlib
        
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to numpy array (128-dimensional)
        embedding = np.frombuffer(hash_bytes + hash_bytes[:64], dtype=np.float32)[:128]
        
        # Normalize
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def embed_batch(self, texts: List[str], **options) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            **options: Embedding options
            
        Returns:
            np.ndarray: Array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if self.model:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                **options
            )
            return np.array(embeddings)
        else:
            return np.array([self._embed_fallback(text, **options) for text in texts])
    
    def embed_sentences(self, text: str, **options) -> List[np.ndarray]:
        """
        Generate embeddings for each sentence.
        
        Args:
            text: Input text
            **options: Embedding options
            
        Returns:
            list: List of sentence embeddings
        """
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if sentences:
            embeddings = self.embed_batch(sentences, **options)
            return [embeddings[i] for i in range(len(sentences))]
        
        return []
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            int: Embedding dimension
        """
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        else:
            return 128  # Fallback dimension
