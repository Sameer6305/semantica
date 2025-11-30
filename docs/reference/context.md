# Context

> **Context engineering and memory management system for intelligent agents using RAG and Knowledge Graphs.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-graph:{ .lg .middle } **Context Graph**

    ---

    Build dynamic context graphs from conversations and entities

-   :material-brain:{ .lg .middle } **Agent Memory**

    ---

    Persistent memory management with vector storage integration

-   :material-link-variant:{ .lg .middle } **Entity Linking**

    ---

    Link entities across documents and conversations

-   :material-magnify:{ .lg .middle } **Hybrid Retrieval**

    ---

    Retrieve context using Vector + Graph + Keyword search

-   :material-history:{ .lg .middle } **Conversation History**

    ---

    Manage and synthesize conversation history

-   :material-bullseye-arrow:{ .lg .middle } **Intent Analysis**

    ---

    Extract and track user intent and sentiment

</div>

!!! tip "When to Use"
    - **Agent Development**: When building agents that need long-term memory
    - **RAG Applications**: For advanced Retrieval-Augmented Generation
    - **Personalization**: To maintain user-specific context and preferences

---

## ⚙️ Algorithms Used

### Context Graph Construction
- **Graph Building**: Node/Edge construction from extracted entities
- **Graph Traversal**: BFS/DFS for multi-hop context discovery
- **Intent Extraction**: NLP-based intent classification
- **Sentiment Analysis**: Sentiment scoring and extraction

### Agent Memory
- **Vector Embedding**: Dense vector generation for memory items
- **Vector Search**: Cosine similarity search (k-NN)
- **Retention Policy**: Time-based decay and cleanup
- **Memory Indexing**: Deque-based sliding window for short-term memory

### Entity Linking
- **URI Generation**: Hash-based deterministic IDs
- **Text Similarity**: Jaccard/Levenshtein for name matching
- **Graph Lookup**: Entity resolution against Knowledge Graph
- **Bidirectional Linking**: Symmetric link creation

### Context Retrieval
- **Hybrid Scoring**: `α * VectorScore + β * GraphScore + γ * KeywordScore`
- **Graph Expansion**: Retrieving neighbors of retrieved entities
- **Deduplication**: Content-based result merging
- **Result Ranking**: Weighted aggregation of scores

---

## Main Classes

### ContextGraphBuilder

Builds and manages the context graph.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `build_from_entities(entities)` | Build graph from entities | Node creation |
| `add_conversation(conv)` | Add conversation data | Intent/Entity extraction |
| `get_subgraph(node_id)` | Get local context | BFS Traversal |

**Example:**

```python
from semantica.context import ContextGraphBuilder

builder = ContextGraphBuilder()
graph = builder.build_from_entities_and_relationships(
    entities=extracted_entities,
    relationships=extracted_rels
)
```

### AgentMemory

Manages persistent agent memory.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `store(text, metadata)` | Store memory item | Embedding + Vector Store |
| `retrieve(query)` | Retrieve relevant memories | Vector Similarity |
| `prune(days)` | Remove old memories | Time-based filtering |

**Example:**

```python
from semantica.context import AgentMemory

memory = AgentMemory(vector_store=vs, knowledge_graph=kg)
memory.store("User prefers Python over Java", metadata={"type": "preference"})
relevant = memory.retrieve("What language does the user like?")
```

### EntityLinker

Links entities across different contexts.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `link_entities(entities)` | Link list of entities | Similarity matching |
| `generate_uri(entity)` | Create unique ID | Hashing |
| `find_links(entity)` | Find related entities | Graph lookup |

### ContextRetriever

Orchestrates hybrid retrieval.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `retrieve(query)` | Get full context | Hybrid (Vector+Graph) |
| `retrieve_from_graph(query)` | Graph-only retrieval | Traversal |
| `retrieve_from_memory(query)` | Memory-only retrieval | Vector search |

---

## Convenience Functions

```python
from semantica.context import build_context

# Build context and memory in one go
context = build_context(
    entities=entities,
    relationships=relationships,
    vector_store=vs,
    knowledge_graph=kg,
    store_initial_memories=True
)
```

---

## Configuration

### Environment Variables

```bash
export CONTEXT_RETENTION_DAYS=30
export CONTEXT_MAX_HOPS=2
export CONTEXT_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### YAML Configuration

```yaml
context:
  retention_policy:
    max_days: 30
    max_items: 1000
    
  retrieval:
    hybrid_weights:
      vector: 0.6
      graph: 0.3
      keyword: 0.1
      
  graph:
    max_depth: 2
    include_attributes: true
```

---

## Integration Examples

### Chatbot with Memory

```python
from semantica.context import AgentMemory, ContextRetriever
from semantica.llm import LLMClient

# 1. Initialize
memory = AgentMemory(vector_store=vs)
retriever = ContextRetriever(memory=memory, graph=kg)
llm = LLMClient()

def chat(user_input):
    # 2. Retrieve Context
    context = retriever.retrieve(user_input)
    
    # 3. Generate Response
    response = llm.generate(user_input, context=context)
    
    # 4. Update Memory
    memory.store(f"User: {user_input}")
    memory.store(f"Agent: {response}")
    
    return response
```

---

## Best Practices

1.  **Prune Regularly**: Use retention policies to keep memory relevant and performant.
2.  **Use Hybrid Retrieval**: Relying solely on vector search misses structural relationships; use graph context too.
3.  **Enrich Metadata**: Store rich metadata (timestamp, source, type) with memories for better filtering.
4.  **Link Entities**: Ensure `EntityLinker` is used to connect mentions of the same entity across conversations.

---

## Troubleshooting

**Issue**: Retrieval returns irrelevant old memories.
**Solution**: Adjust retention policy or increase vector similarity threshold.

```python
memory = AgentMemory(
    retention_days=7,
    similarity_threshold=0.8
)
```

**Issue**: Context graph growing too large.
**Solution**: Use `prune_graph` or limit hop depth during retrieval.

---

## See Also

- [Vector Store Module](vector_store.md) - Underlying storage for memory
- [Knowledge Graph Module](kg.md) - Underlying graph structure
- [Embeddings Module](embeddings.md) - Vector generation
