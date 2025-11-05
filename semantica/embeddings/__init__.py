"""
Embeddings Generation Module

This module provides comprehensive embedding generation and management capabilities.

Exports:
    - EmbeddingGenerator: Main embedding generation class
    - TextEmbedder: Text embedding generation
    - ImageEmbedder: Image embedding generation
    - AudioEmbedder: Audio embedding generation
    - MultimodalEmbedder: Multi-modal embedding support
    - EmbeddingOptimizer: Embedding optimization and fine-tuning
    - ContextManager: Embedding context management
    - ProviderAdapters: Provider-specific adapters
"""

from .embedding_generator import EmbeddingGenerator
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder
from .audio_embedder import AudioEmbedder
from .multimodal_embedder import MultimodalEmbedder
from .embedding_optimizer import EmbeddingOptimizer
from .context_manager import ContextManager, ContextWindow
from .provider_adapters import (
    ProviderAdapter,
    OpenAIAdapter,
    BGEAdapter,
    LlamaAdapter,
    ProviderAdapterFactory,
)
from .pooling_strategies import (
    PoolingStrategy,
    MeanPooling,
    MaxPooling,
    CLSPooling,
    AttentionPooling,
    HierarchicalPooling,
    PoolingStrategyFactory,
)

__all__ = [
    "EmbeddingGenerator",
    "TextEmbedder",
    "ImageEmbedder",
    "AudioEmbedder",
    "MultimodalEmbedder",
    "EmbeddingOptimizer",
    "ContextManager",
    "ContextWindow",
    # Provider adapters
    "ProviderAdapter",
    "OpenAIAdapter",
    "BGEAdapter",
    "LlamaAdapter",
    "ProviderAdapterFactory",
    # Pooling strategies
    "PoolingStrategy",
    "MeanPooling",
    "MaxPooling",
    "CLSPooling",
    "AttentionPooling",
    "HierarchicalPooling",
    "PoolingStrategyFactory",
]
