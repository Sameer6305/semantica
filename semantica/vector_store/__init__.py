"""
Vector Store Management Module

This module provides comprehensive vector storage and retrieval capabilities.

Exports:
    - VectorStore: Main vector store interface
    - VectorIndexer: Vector indexing and search
    - VectorRetriever: Vector retrieval and similarity search
    - VectorManager: Vector store management and operations
    - FAISSAdapter: FAISS integration
    - PineconeAdapter: Pinecone integration
    - WeaviateAdapter: Weaviate integration
    - QdrantAdapter: Qdrant integration
    - MilvusAdapter: Milvus integration
    - HybridSearch: Hybrid vector and metadata search
    - MetadataStore: Metadata indexing and management
    - NamespaceManager: Namespace isolation and management
"""

from .vector_store import (
    VectorStore,
    VectorIndexer,
    VectorRetriever,
    VectorManager
)

from .faiss_adapter import (
    FAISSAdapter,
    FAISSIndex,
    FAISSSearch,
    FAISSIndexBuilder
)

from .pinecone_adapter import (
    PineconeAdapter,
    PineconeIndex,
    PineconeQuery,
    PineconeMetadata
)

from .weaviate_adapter import (
    WeaviateAdapter,
    WeaviateClient,
    WeaviateSchema,
    WeaviateQuery
)

from .qdrant_adapter import (
    QdrantAdapter,
    QdrantClient,
    QdrantCollection,
    QdrantSearch
)

from .milvus_adapter import (
    MilvusAdapter,
    MilvusClient,
    MilvusCollection,
    MilvusSearch
)

from .hybrid_search import (
    HybridSearch,
    MetadataFilter,
    SearchRanker
)

from .metadata_store import (
    MetadataStore,
    MetadataIndex,
    MetadataSchema
)

from .namespace_manager import (
    NamespaceManager,
    Namespace
)

__all__ = [
    # Core vector store
    "VectorStore",
    "VectorIndexer",
    "VectorRetriever",
    "VectorManager",
    # FAISS
    "FAISSAdapter",
    "FAISSIndex",
    "FAISSSearch",
    "FAISSIndexBuilder",
    # Pinecone
    "PineconeAdapter",
    "PineconeIndex",
    "PineconeQuery",
    "PineconeMetadata",
    # Weaviate
    "WeaviateAdapter",
    "WeaviateClient",
    "WeaviateSchema",
    "WeaviateQuery",
    # Qdrant
    "QdrantAdapter",
    "QdrantClient",
    "QdrantCollection",
    "QdrantSearch",
    # Milvus
    "MilvusAdapter",
    "MilvusClient",
    "MilvusCollection",
    "MilvusSearch",
    # Hybrid search
    "HybridSearch",
    "MetadataFilter",
    "SearchRanker",
    # Metadata store
    "MetadataStore",
    "MetadataIndex",
    "MetadataSchema",
    # Namespace manager
    "NamespaceManager",
    "Namespace",
]
