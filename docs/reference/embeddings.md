# Embeddings

> **Multi-modal embedding generation for Text, Images, and Audio.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-text-box:{ .lg .middle } **Text Embeddings**

    ---

    Sentence-transformers, OpenAI, and BGE model support

-   :material-image:{ .lg .middle } **Image Embeddings**

    ---

    CLIP-based vision embeddings for cross-modal search

-   :material-waveform:{ .lg .middle } **Audio Embeddings**

    ---

    MFCC, Chroma, and Spectral feature extraction via Librosa

-   :material-layers-search:{ .lg .middle } **Multimodal**

    ---

    Unified embedding space for searching across modalities

-   :material-compress:{ .lg .middle } **Optimization**

    ---

    PCA reduction, Quantization, and Pooling strategies

-   :material-text-short:{ .lg .middle } **Context Management**

    ---

    Sliding window chunking with sentence boundary preservation

</div>

!!! tip "When to Use"
    - **Semantic Search**: Converting text to vectors for similarity search
    - **Clustering**: Grouping similar documents or images
    - **Classification**: Using embeddings as features for ML models
    - **RAG**: Embedding chunks for retrieval

---

## ⚙️ Algorithms Used

### Generation
- **Transformer Encoding**: BERT/RoBERTa based models for text.
- **CLIP Encoding**: Vision Transformer (ViT) for images.
- **Feature Extraction**: Signal processing (MFCC, Spectral Contrast) for audio.

### Optimization
- **PCA**: Dimensionality reduction to reduce storage costs while preserving variance.
- **Quantization**: Converting float32 -> int8 for 4x memory savings.
- **Pooling**: Mean, Max, or CLS token pooling to aggregate token vectors into sentence vectors.

### Context
- **Sliding Window**: `[A, B, C], [B, C, D], ...` to capture context across boundaries.
- **Sentence Splitting**: Regex-based splitting to avoid breaking sentences mid-chunk.

---

## Main Classes

### EmbeddingGenerator

Unified interface for all modalities.

**Methods:**

| Method | Description |
|--------|-------------|
| `generate(data, type)` | Generate embedding |
| `process_batch(items)` | Batch generation |

**Example:**

```python
from semantica.embeddings import EmbeddingGenerator

gen = EmbeddingGenerator()
vec = gen.generate("Hello world", data_type="text")
```

### TextEmbedder

Specialized text embedding.

**Methods:**

| Method | Description |
|--------|-------------|
| `embed(text)` | Generate vector |
| `embed_batch(texts)` | Batch processing |

### ImageEmbedder

Specialized image embedding.

**Methods:**

| Method | Description |
|--------|-------------|
| `embed(image_path)` | Generate vector |

### EmbeddingOptimizer

Optimizes vectors.

**Methods:**

| Method | Description |
|--------|-------------|
| `reduce_dimension(vecs)` | Apply PCA |
| `quantize(vecs)` | Apply quantization |

---

## Convenience Functions

```python
from semantica.embeddings import EmbeddingGenerator, embed_text, calculate_similarity

# Generate embeddings
generator = EmbeddingGenerator()
emb1 = generator.generate_embeddings("text1", data_type="text")
emb2 = generator.generate_embeddings("text2", data_type="text")

# Similarity
score = calculate_similarity(emb1, emb2)
```

---

## Configuration

### Environment Variables

```bash
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export EMBEDDING_DEVICE=cuda
export OPENAI_API_KEY=sk-...
```

### YAML Configuration

```yaml
embeddings:
  text:
    model: all-MiniLM-L6-v2
    batch_size: 32
    
  image:
    model: clip-ViT-B-32
    
  optimization:
    quantize: false
```

---

## Integration Examples

### Multimodal Search

```python
from semantica.embeddings import MultimodalEmbedder
from semantica.vector_store import VectorStore

# 1. Embed Image and Text into same space
embedder = MultimodalEmbedder()
img_vec = embedder.embed_image("cat.jpg")
text_vec = embedder.embed_text("A cute kitten")

# 2. Store
store = VectorStore()
store.store_vectors([img_vec], metadata=[{"type": "image", "path": "cat.jpg"}])

# 3. Search with Text
results = store.search(text_vec, k=1)
print(f"Found: {results[0].metadata['path']}")
```

---

## Best Practices

1.  **Batch Processing**: Always use batch methods (`embed_batch`, `process_batch`) for >1 item. It's much faster on GPU.
2.  **Use Caching**: Embeddings are expensive to compute. Cache them if possible.
3.  **Match Dimensions**: Ensure your vector store is configured with the correct dimension for your chosen model.
4.  **Normalize**: L2 normalization is usually required for Cosine Similarity to work correctly.

---

## See Also

- [Vector Store Module](vector_store.md) - Storing the generated vectors
- [Ingest Module](ingest.md) - Loading data to embed
- [Pipeline Module](pipeline.md) - Orchestrating the embedding process
