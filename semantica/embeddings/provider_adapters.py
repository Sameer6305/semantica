"""
Provider adapters for Semantica framework.

This module provides adapters for various embedding providers
like OpenAI, BGE, and Llama.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger


class ProviderAdapter:
    """Base class for embedding provider adapters."""
    
    def __init__(self, **config):
        """Initialize provider adapter."""
        self.logger = get_logger("provider_adapter")
        self.config = config
    
    def embed(self, text: str, **options) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            **options: Embedding options
            
        Returns:
            np.ndarray: Embedding vector
        """
        raise NotImplementedError
    
    def embed_batch(self, texts: List[str], **options) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts
            **options: Embedding options
            
        Returns:
            np.ndarray: Array of embeddings
        """
        return np.array([self.embed(text, **options) for text in texts])


class OpenAIAdapter(ProviderAdapter):
    """OpenAI embedding API adapter."""
    
    def __init__(self, **config):
        """Initialize OpenAI adapter."""
        super().__init__(**config)
        
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model = config.get("model", "text-embedding-3-small")
        
        # Initialize client
        self.client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                self.logger.warning("OpenAI library not installed")
    
    def embed(self, text: str, **options) -> np.ndarray:
        """
        Generate embedding using OpenAI API.
        
        Args:
            text: Input text
            **options: Embedding options
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.client:
            raise ProcessingError("OpenAI client not initialized. Check API key.")
        
        try:
            response = self.client.embeddings.create(
                model=options.get("model", self.model),
                input=text
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to get OpenAI embedding: {e}")
            raise ProcessingError(f"Failed to get OpenAI embedding: {e}")


class BGEAdapter(ProviderAdapter):
    """BGE (BAAI General Embedding) model adapter."""
    
    def __init__(self, **config):
        """Initialize BGE adapter."""
        super().__init__(**config)
        
        self.model_name = config.get("model_name", "BAAI/bge-small-en-v1.5")
        self.model = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize BGE model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded BGE model: {self.model_name}")
        except ImportError:
            self.logger.warning("sentence-transformers not available for BGE")
        except Exception as e:
            self.logger.warning(f"Failed to load BGE model: {e}")
    
    def embed(self, text: str, **options) -> np.ndarray:
        """
        Generate embedding using BGE model.
        
        Args:
            text: Input text
            **options: Embedding options
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.model:
            raise ProcessingError("BGE model not initialized")
        
        try:
            embedding = self.model.encode([text], normalize_embeddings=True)[0]
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Failed to get BGE embedding: {e}")
            raise ProcessingError(f"Failed to get BGE embedding: {e}")


class LlamaAdapter(ProviderAdapter):
    """Llama embedding model adapter."""
    
    def __init__(self, **config):
        """Initialize Llama adapter."""
        super().__init__(**config)
        
        self.model_name = config.get("model_name")
        self.model = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Llama model."""
        # Note: Llama embedding models typically require custom setup
        # This is a placeholder for integration
        self.logger.warning("Llama adapter requires custom model setup")
    
    def embed(self, text: str, **options) -> np.ndarray:
        """
        Generate embedding using Llama model.
        
        Args:
            text: Input text
            **options: Embedding options
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self.model:
            raise ProcessingError("Llama model not initialized")
        
        # Placeholder - would require actual Llama model implementation
        # For now, return a placeholder embedding
        self.logger.warning("Llama adapter using placeholder implementation")
        
        # Generate a placeholder embedding (same dimension as typical embeddings)
        embedding_dim = 768  # Default Llama embedding dimension
        import numpy as np
        placeholder = np.random.normal(0, 0.1, embedding_dim).astype(np.float32)
        
        # Normalize
        norm = np.linalg.norm(placeholder)
        if norm > 0:
            placeholder = placeholder / norm
        
        return placeholder


class ProviderAdapterFactory:
    """Factory for creating provider adapters."""
    
    @staticmethod
    def create(provider: str, **config) -> ProviderAdapter:
        """
        Create provider adapter.
        
        Args:
            provider: Provider name ("openai", "bge", "llama")
            **config: Provider configuration
            
        Returns:
            ProviderAdapter: Provider adapter instance
        """
        providers = {
            "openai": OpenAIAdapter,
            "bge": BGEAdapter,
            "llama": LlamaAdapter
        }
        
        adapter_class = providers.get(provider.lower())
        if not adapter_class:
            raise ProcessingError(f"Unsupported provider: {provider}")
        
        return adapter_class(**config)
