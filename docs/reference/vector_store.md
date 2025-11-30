# Vector Store

> **Unified vector database interface supporting FAISS, Pinecone, Weaviate, Qdrant, and Milvus with Hybrid Search.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **Multi-Backend Support**

    ---

    Seamlessly switch between FAISS (Local), Pinecone, Weaviate, Qdrant, and Milvus

-   :material-magnify-plus:{ .lg .middle } **Hybrid Search**

    ---

    Combine dense vector similarity with sparse keyword/metadata filtering

-   :material-filter:{ .lg .middle } **Metadata Filtering**

    ---

    Rich filtering capabilities (eq, ne, gt, lt, in, contains)

-   :material-layers-triple:{ .lg .middle } **Namespace Isolation**

    ---

    Multi-tenant support via isolated namespaces

-   :material-flash:{ .lg .middle } **Performance**

    ---

    Batch operations, index optimization, and caching

-   :material-cloud-upload:{ .lg .middle } **Cloud & Local**

    ---

    Support for both embedded (local) and cloud-native deployments

</div>

!!! tip "When to Use"
    - **Semantic Search**: Finding documents similar to a query
    - **RAG**: Retrieving context for LLM generation
    - **Memory**: Storing agent memories as embeddings
    - **Recommendation**: Finding similar items based on vector proximity

---

## ⚙️ Algorithms Used

### Similarity Metrics
- **Cosine Similarity**: `A · B / ||A|| ||B||` (Default for semantic search)
- **Euclidean Distance (L2)**: `||A - B||`
- **Dot Product**: `A · B` (Faster, requires normalized vectors)

### Indexing (FAISS)
- **Flat**: Exact search (brute force). High accuracy, slow for large datasets.
- **IVF (Inverted File)**: Partitions space into Voronoi cells. Faster search.
- **HNSW**: Hierarchical Navigable Small World graphs. Best trade-off for speed/accuracy.
- **PQ (Product Quantization)**: Compresses vectors for memory efficiency.

### Hybrid Search
- **Reciprocal Rank Fusion (RRF)**: Combines ranked lists from vector search and keyword search.
  `Score = 1 / (k + rank_vector) + 1 / (k + rank_keyword)`
- **Pre-filtering**: Apply metadata filters *before* vector search (supported by most backends).

---

## Main Classes

### VectorStore

The main facade for all vector operations.

**Methods:**

| Method | Description |
|--------|-------------|
| `store_vectors(vectors, metadata)` | Store embeddings |
| `search(query, k)` | Semantic search |
| `delete(ids)` | Remove vectors |

**Example:**

```python
from semantica.vector_store import VectorStore

# Initialize (defaults to FAISS)
store = VectorStore(backend="faiss", dimension=1536)

# Store
ids = store.store_vectors(
    vectors=[[0.1, 0.2, ...], ...],
    metadata=[{"text": "Hello"}, ...]
)

# Search
results = store.search(query_vector=[0.1, 0.2, ...], k=5)
```

### HybridSearch

Combines vector and metadata search.

**Methods:**

| Method | Description |
|--------|-------------|
| `search(query_vec, filter)` | Execute hybrid query |

**Example:**

```python
from semantica.vector_store import HybridSearch, MetadataFilter

searcher = HybridSearch(store)
filters = MetadataFilter().eq("category", "news").gt("date", "2023-01-01")

results = searcher.search(
    query_vector=emb,
    filter=filters,
    k=10
)
```

### Adapters

Backend-specific implementations:
- `FAISSAdapter`: Local, in-memory/disk.
- `PineconeAdapter`: Managed cloud service.
- `WeaviateAdapter`: Schema-aware vector DB.
- `QdrantAdapter`: Rust-based high-performance DB.
- `MilvusAdapter`: Scalable cloud-native DB.

---

## Convenience Functions

```python
from semantica.vector_store import store_vectors, search_vectors

# Quick usage (uses default configured backend)
store_vectors(embeddings, metadata)
results = search_vectors(query_embedding)
```

---

## Configuration

### Environment Variables

```bash
export VECTOR_STORE_BACKEND=pinecone
export PINECONE_API_KEY=sk-...
export PINECONE_ENV=us-west1-gcp
```

### YAML Configuration

```yaml
vector_store:
  backend: faiss # or pinecone, weaviate, etc.
  dimension: 1536
  metric: cosine
  
  faiss:
    index_type: HNSW
    
  pinecone:
    environment: us-west1-gcp
    index_name: my-index
```

---

## Integration Examples

### RAG Retrieval

```python
from semantica.embeddings import EmbeddingGenerator
from semantica.vector_store import VectorStore

# 1. Embed Query
embedder = EmbeddingGenerator()
query_vec = embedder.generate("What is the capital of France?")

# 2. Search
store = VectorStore()
results = store.search(query_vec, k=3)

# 3. Use Context
context = "\n".join([r.metadata['text'] for r in results])
print(f"Context: {context}")
```

---

## Best Practices

1.  **Normalize Vectors**: Always normalize vectors if using Cosine Similarity or Dot Product.
2.  **Use HNSW**: For FAISS, `HNSW` is usually the best default index type for performance/recall balance.
3.  **Batch Operations**: Use `store_vectors` with batches (e.g., 100 items) rather than one by one.
4.  **Filter First**: In hybrid search, restrictive filters significantly improve performance.

---

## Troubleshooting

**Issue**: `DimensionMismatchError`
**Solution**: Ensure your embedding model dimension (e.g., 1536 for OpenAI) matches the VectorStore dimension.

**Issue**: FAISS index not saved.
**Solution**: Call `store.save("index.faiss")` explicitly for local FAISS indices, or use a persistent backend like Pinecone/Qdrant.

---

## See Also

- [Embeddings Module](embeddings.md) - Generates the vectors
- [Context Module](context.md) - Uses vector store for memory
- [Ingest Module](ingest.md) - Source of data
