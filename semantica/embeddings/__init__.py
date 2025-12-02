"""
Embeddings Generation Module

This module provides comprehensive embedding generation and management capabilities
for the Semantica framework, supporting text, image, audio, and multimodal content
with multiple embedding models, optimization strategies, and similarity calculations.

Algorithms Used:

Embedding Generation:
    - Sentence-transformers Encoding: Transformer-based sentence embedding generation using pre-trained models
    - CLIP Image Encoding: Vision transformer-based image embedding generation for cross-modal understanding
    - Librosa Audio Feature Extraction: Multi-feature audio analysis including:
      * MFCC (Mel-frequency Cepstral Coefficients): 13 coefficients for spectral representation
      * Chroma Features: 12-dimensional pitch class representation
      * Spectral Contrast: 7-dimensional spectral contrast features
      * Tonnetz: 6-dimensional tonal centroid features
      * Temporal Aggregation: Mean pooling across time frames for fixed-size vectors
    - Hash-based Fallback Embeddings: SHA-256 hash-based deterministic embeddings (128-dimensional)
    - Batch Processing: Vectorized processing for multiple items simultaneously
    - Data Type Auto-detection: File extension-based and content-based data type detection

Similarity Calculation:
    - Cosine Similarity: Dot product divided by vector norms for normalized similarity (0-1 range)
    - Euclidean Similarity: L2 norm distance converted to similarity score (0-1 range)
    - Cross-modal Similarity: Pairwise cosine similarity between different modalities (text-image, image-audio, etc.)

Embedding Optimization:
    - PCA (Principal Component Analysis): Variance-preserving dimension reduction using eigenvalue decomposition
    - Quantization: Bit-depth reduction (8-bit, 16-bit) for memory efficiency with dequantization
    - Dimension Truncation: Simple truncation to first N dimensions
    - Batch Normalization: L2 normalization across batch dimension for unit vectors

Pooling Strategies:
    - Mean Pooling: Arithmetic mean across embedding dimension
    - Max Pooling: Element-wise maximum across embedding dimension
    - CLS Token Pooling: First token/embedding extraction (for transformer models)
    - Attention-based Pooling: Softmax-weighted sum using dot product attention scores
    - Hierarchical Pooling: Two-level pooling (chunk-level then global-level mean pooling)

Multimodal Processing:
    - Dimension Alignment: Truncation/padding to align embedding dimensions across modalities
    - Concatenation: Vector concatenation for combined multimodal embeddings
    - Averaging: Mean aggregation for compact multimodal representation
    - Cross-modal Similarity: Pairwise similarity computation between different input modalities

Context Management:
    - Text Splitting: Sliding window text chunking with configurable window size
    - Sentence Boundary Preservation: Regex-based sentence boundary detection (., !, ?, newline) with minimum window size constraints
    - Overlapping Windows: Sliding window with configurable overlap for context continuity
    - Context Merging: Text concatenation and metadata aggregation from multiple context windows
    - LRU-style Context Cleanup: Oldest context removal when maximum context limit reached

Provider Adapters:
    - OpenAI API Integration: REST API-based embedding generation
    - BGE Model Integration: Sentence-transformers wrapper for BAAI General Embedding models
    - Sentence-transformers Integration: Hugging Face transformers-based embedding models

Key Features:
    - Text, image, audio, and multimodal embedding generation
    - Multiple embedding models and providers
    - Embedding optimization and compression
    - Similarity calculation and comparison
    - Pooling strategies for aggregation
    - Context window management for long texts
    - Method registry for extensibility
    - Configuration management with environment variables and config files

Main Classes:
    - EmbeddingGenerator: Main embedding generation handler
    - TextEmbedder: Text embedding generation
    - ImageEmbedder: Image embedding generation
    - AudioEmbedder: Audio embedding generation
    - MultimodalEmbedder: Multi-modal embedding support
    - EmbeddingOptimizer: Embedding optimization and fine-tuning
    - ContextManager: Embedding context management
    - MethodRegistry: Registry for custom embedding methods
    - EmbeddingsConfig: Configuration manager for embeddings module

Convenience Functions:
    - build: Generate embeddings from data in one call
    - generate_embeddings: Generate embeddings with method dispatch
    - embed_text: Text embedding wrapper
    - embed_image: Image embedding wrapper
    - embed_audio: Audio embedding wrapper
    - embed_multimodal: Multimodal embedding wrapper
    - optimize_embeddings: Embedding optimization wrapper
    - calculate_similarity: Similarity calculation wrapper
    - pool_embeddings: Pooling strategy wrapper

Example Usage:
    >>> from semantica.embeddings import build, embed_text, embed_image, calculate_similarity
    >>> # Using convenience function
    >>> result = build(data=["text1", "text2"], data_type="text")
    >>> # Using method functions
    >>> text_emb = embed_text("Hello world", method="sentence_transformers")
    >>> img_emb = embed_image("image.jpg", method="clip")
    >>> similarity = calculate_similarity(text_emb, img_emb, method="cosine")
    >>> # Using classes directly
    >>> from semantica.embeddings import EmbeddingGenerator
    >>> generator = EmbeddingGenerator()
    >>> embeddings = generator.generate_embeddings("Hello world", data_type="text")

Author: Semantica Contributors
License: MIT
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .audio_embedder import AudioEmbedder
from .config import EmbeddingsConfig, embeddings_config
from .context_manager import ContextManager, ContextWindow
from .embedding_generator import EmbeddingGenerator
from .embedding_optimizer import EmbeddingOptimizer
from .image_embedder import ImageEmbedder
from .methods import (
    calculate_similarity,
    embed_audio,
    embed_image,
    embed_multimodal,
    embed_text,
    generate_embeddings,
    get_embedding_method,
    list_available_methods,
    optimize_embeddings,
    pool_embeddings,
)
from .multimodal_embedder import MultimodalEmbedder
from .pooling_strategies import (
    AttentionPooling,
    CLSPooling,
    HierarchicalPooling,
    MaxPooling,
    MeanPooling,
    PoolingStrategy,
    PoolingStrategyFactory,
)
from .provider_adapters import (
    BGEAdapter,
    LlamaAdapter,
    OpenAIAdapter,
    ProviderAdapter,
    ProviderAdapterFactory,
)
from .registry import MethodRegistry, method_registry
from .text_embedder import TextEmbedder

__all__ = [
    # Core Classes
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
    # Registry and Methods
    "MethodRegistry",
    "method_registry",
    "generate_embeddings",
    "embed_text",
    "embed_image",
    "embed_audio",
    "embed_multimodal",
    "optimize_embeddings",
    "calculate_similarity",
    "pool_embeddings",
    "get_embedding_method",
    "list_available_methods",
    # Configuration
    "EmbeddingsConfig",
    "embeddings_config",
]
