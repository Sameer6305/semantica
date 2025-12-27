# Context Module Reference

> **The central nervous system for intelligent agents, managing memory, knowledge graphs, and context retrieval.**

---

## 🎯 System Overview

The **Context Module** provides agents with a persistent, searchable, and structured memory system. It is built on a **Synchronous Architecture 2.0**, ensuring predictable state management and compatibility with modern vector stores and graph databases.

### Key Capabilities

<div class="grid cards" markdown>

-   :material-brain:{ .lg .middle } **Hierarchical Memory**

    ---

    Mimics human memory with a fast, token-limited Short-Term buffer and infinite Long-Term vector storage.

-   :material-graph-outline:{ .lg .middle } **GraphRAG**

    ---

    Combines unstructured vector search with structured knowledge graph traversal for deep contextual understanding.

-   :material-scale-balance:{ .lg .middle } **Hybrid Retrieval**

    ---

    Intelligently blends Keyword (BM25), Vector (Dense), and Graph (Relational) scores for optimal relevance.

-   :material-lightning-bolt:{ .lg .middle } **Token Management**

    ---

    Automatic FIFO and importance-based pruning to keep context within LLM window limits.

-   :material-link-variant:{ .lg .middle } **Entity Linking**

    ---

    Resolves ambiguities by linking text mentions to unique entities in the knowledge graph.

</div>

!!! tip "When to Use"
    - **Memory Persistence**: Enabling agents to remember user preferences and history.
    - **Complex Retrieval**: When simple vector search fails to capture relationships.
    - **Knowledge Graph**: Building a structured world model from unstructured text.

---

## 🏗️ Architecture Components

### AgentContext (The Orchestrator)
The high-level facade that unifies all context operations. It routes data to the appropriate subsystems (Memory, Graph, Vector Store) and manages the lifecycle of context.

#### **Constructor Parameters**
*   `vector_store` (Required): The backing vector database instance (e.g., FAISS, Weaviate).
*   `knowledge_graph` (Optional): The graph store instance for structured knowledge.
*   `token_limit` (Default: `2000`): The maximum number of tokens allowed in short-term memory before pruning occurs.
*   `short_term_limit` (Default: `10`): The maximum number of distinct memory items in short-term memory.
*   `hybrid_alpha` (Default: `0.5`): The weighting factor for retrieval (0.0 = Pure Vector, 1.0 = Pure Graph).
*   `use_graph_expansion` (Default: `True`): Whether to fetch neighbors of retrieved nodes from the graph.

#### **Core Methods**

| Method | Description |
|--------|-------------|
| `store(content, ...)` | Writes information to memory. Handles auto-detection, write-through to vector store, and entity extraction. |
| `retrieve(query, ...)` | Fetches relevant context using hybrid search (Vector + Graph) and reranking. |
| `query_with_reasoning(query, llm_provider, ...)` | **GraphRAG with multi-hop reasoning**: Retrieves context, builds reasoning paths, and generates LLM-based natural language responses grounded in the knowledge graph. |

#### **Code Example**
```python
from semantica.context import AgentContext
from semantica.vector_store import VectorStore

# 1. Initialize
vs = VectorStore(backend="faiss", dimension=768)
context = AgentContext(
    vector_store=vs,
    token_limit=2000
)

# 2. Store Memory
context.store(
    "User is working on a React project.",
    conversation_id="session_1",
    user_id="user_123"
)

# 3. Retrieve Context
results = context.retrieve("What is the user building?")

# 4. Query with Reasoning (GraphRAG)
from semantica.llms import Groq
import os

llm_provider = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

result = context.query_with_reasoning(
    query="What IPs are associated with security alerts?",
    llm_provider=llm_provider,
    max_results=10,
    max_hops=2
)

print(f"Response: {result['response']}")
print(f"Reasoning Path: {result['reasoning_path']}")
print(f"Confidence: {result['confidence']:.3f}")
```

---

### AgentMemory (The Storage Engine)
Manages the storage and lifecycle of memory items. It implements the **Hierarchical Memory** pattern.

#### **Features & Functions**
*   **Short-Term Memory (Working Memory)**
    *   *Structure*: An in-memory list of recent `MemoryItem` objects.
    *   *Purpose*: Provides immediate context for the ongoing conversation.
    *   *Pruning Logic*:
        *   **FIFO**: Removes the oldest items first when limits are reached.
        *   **Token-Aware**: Calculates token counts to ensure the total buffer size stays under `token_limit`.
*   **Long-Term Memory (Episodic Memory)**
    *   *Structure*: Vector embeddings stored in the `vector_store`.
    *   *Purpose*: Persists history indefinitely for semantic retrieval.
    *   *Synchronization*: Automatically syncs with Short-term memory during `store()` operations.
*   **Retention Policy**
    *   *Time-Based*: Can automatically delete memories older than `retention_days`.
    *   *Count-Based*: Can limit the total number of memories to `max_memories`.

#### **Key Methods**

| Method | Description |
|--------|-------------|
| `store_vectors()` | Handles the low-level interaction with concrete Vector Store implementations. |
| `_prune_short_term_memory()` | Internal algorithm that enforces token and count limits. |
| `get_conversation_history()` | Retrieves a chronological list of interactions for a specific session. |

#### **Code Example**
```python
# Accessing via AgentContext
memory = context.memory

# Get conversation history
history = memory.get_conversation_history("session_1")
for item in history:
    print(f"[{item.timestamp}] {item.content}")

# Get statistics
stats = memory.get_statistics()
print(f"Stored Memories: {stats['total_memories']}")
```

---

### ContextGraph (The Knowledge Structure)
Manages the structured relationships between entities. It provides the "World Model" for the agent.

#### **Features & Functions**
*   **Dictionary-Based Interface**
    *   *Design*: Uses standard Python dictionaries for nodes and edges, removing dependencies on complex interface classes.
    *   *Benefit*: simpler serialization and easier integration with external APIs.
*   **Graph Traversal**
    *   *Adjacency List*: optimized internal structure for fast neighbor lookups.
    *   *Multi-Hop Search*: Can traverse `k` hops from a starting node to find indirect connections.
*   **Node & Edge Types**
    *   *Typed Schema*: Supports distinct types for nodes (e.g., "Person", "Concept") and edges (e.g., "KNOWS", "RELATED_TO").

#### **Key Methods**

| Method | Description |
|--------|-------------|
| `add_nodes(nodes)` | Bulk adds nodes using a list of dictionaries. |
| `add_edges(edges)` | Bulk adds edges using a list of dictionaries. |
| `get_neighbors(node_id, hops)` | Returns connected nodes within a specified distance. |
| `query(query_str)` | Performs keyword-based search specifically on graph nodes. |

#### **Code Example**
```python
from semantica.context import ContextGraph

graph = ContextGraph()

# Add Nodes
graph.add_nodes([
    {
        "id": "Python", 
        "type": "Language", 
        "properties": {"paradigm": "OO"}
    },
    {
        "id": "FastAPI", 
        "type": "Framework", 
        "properties": {"language": "Python"}
    }
])

# Add Edges
graph.add_edges([
    {
        "source_id": "FastAPI", 
        "target_id": "Python", 
        "type": "WRITTEN_IN"
    }
])

# Find Neighbors
neighbors = graph.get_neighbors("FastAPI", hops=1)
```

---

### Production Graph Store Integration

For production environments, you can replace the in-memory `ContextGraph` with a persistent `GraphStore` (Neo4j, FalkorDB) by passing it to the `knowledge_graph` parameter.

```python
from semantica.context import AgentContext
from semantica.graph_store import GraphStore

# 1. Initialize Persistent Graph Store (Neo4j)
gs = GraphStore(
    backend="neo4j",
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# 2. Initialize Agent Context with Persistent Graph
context = AgentContext(
    vector_store=vs,      # Your VectorStore instance
    knowledge_graph=gs,   # Your persistent GraphStore
    use_graph_expansion=True
)

# Now all graph operations (store, retrieve, build_graph) use Neo4j directly.
```

---

### ContextRetriever (The Search Engine)
The retrieval logic that powers the `retrieve()` command. It implements the **Hybrid Retrieval** algorithm.

#### **Retrieval Strategy**
1.  **Short-Term Check**: Scans the in-memory buffer for immediate, exact-match relevance.
2.  **Vector Search**: Queries the `vector_store` for semantically similar long-term memories.
3.  **Graph Expansion**:
    *   Identifies entities in the query.
    *   Finds those entities in the `ContextGraph`.
    *   Traverses edges to find related concepts that might not match keywords (e.g., finding "Python" when searching for "Coding").
4.  **Hybrid Scoring**:
    *   Formula: `Final_Score = (Vector_Score * (1 - α)) + (Graph_Score * α)`
    *   Allows tuning the balance between semantic similarity and structural relevance.

#### **Code Example**
```python
# The retriever is automatically used by AgentContext.retrieve()
# But can be accessed directly if needed:

retriever = context.retriever

# Perform a manual retrieval
results = retriever.retrieve(
    query="web frameworks",
    max_results=5
)
```

---

### GraphRAG with Multi-Hop Reasoning

The `query_with_reasoning()` method extends traditional retrieval by performing multi-hop graph traversal and generating natural language responses using LLMs. This enables deeper understanding of relationships and context-aware answer generation.

#### **How It Works**

1. **Context Retrieval**: Retrieves relevant context using hybrid search (vector + graph)
2. **Entity Extraction**: Extracts entities from query and retrieved context
3. **Multi-Hop Reasoning**: Traverses knowledge graph up to N hops to find related entities
4. **Reasoning Path Construction**: Builds reasoning chains showing entity relationships
5. **LLM Response Generation**: Generates natural language response grounded in graph context

#### **Key Features**

- **Multi-Hop Reasoning**: Traverses graph up to configurable hops (default: 2)
- **Reasoning Trace**: Shows entity relationship paths used in reasoning
- **Grounded Responses**: LLM generates answers citing specific graph entities
- **Multiple LLM Providers**: Supports Groq, OpenAI, HuggingFace, and LiteLLM (100+ LLMs)
- **Fallback Handling**: Returns context with reasoning path if LLM unavailable

#### **Method Signature**

```python
def query_with_reasoning(
    self,
    query: str,
    llm_provider: Any,  # LLM provider from semantica.llms
    max_results: int = 10,
    max_hops: int = 2,
    **kwargs
) -> Dict[str, Any]:
```

**Parameters:**
- `query` (str): User query
- `llm_provider`: LLM provider instance (from `semantica.llms`)
- `max_results` (int): Maximum context results to retrieve (default: 10)
- `max_hops` (int): Maximum graph traversal hops (default: 2)
- `**kwargs`: Additional retrieval options

**Returns:**
- `response` (str): Generated natural language answer
- `reasoning_path` (str): Multi-hop reasoning trace
- `sources` (List[Dict]): Retrieved context items used
- `confidence` (float): Overall confidence score
- `num_sources` (int): Number of sources retrieved
- `num_reasoning_paths` (int): Number of reasoning paths found

#### **Code Example**

```python
from semantica.context import AgentContext
from semantica.llms import Groq
from semantica.vector_store import VectorStore
import os

# Initialize context
context = AgentContext(
    vector_store=VectorStore(backend="faiss"),
    knowledge_graph=kg
)

# Configure LLM provider
llm_provider = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Query with reasoning
result = context.query_with_reasoning(
    query="What IPs are associated with security alerts?",
    llm_provider=llm_provider,
    max_results=10,
    max_hops=2
)

# Access results
print(f"Response: {result['response']}")
print(f"\nReasoning Path: {result['reasoning_path']}")
print(f"Confidence: {result['confidence']:.3f}")
```

#### **Using Different LLM Providers**

```python
# Groq
from semantica.llms import Groq
llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

# OpenAI
from semantica.llms import OpenAI
llm = OpenAI(model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))

# LiteLLM (100+ providers)
from semantica.llms import LiteLLM
llm = LiteLLM(model="anthropic/claude-sonnet-4-20250514")

# Use with query_with_reasoning
result = context.query_with_reasoning(
    query="Your question here",
    llm_provider=llm,
    max_hops=3
)
```

!!! tip "When to Use"
    - **Complex Queries**: When simple retrieval doesn't capture relationships
    - **Explainable AI**: When you need to show reasoning paths
    - **Multi-Hop Questions**: "What IPs are associated with alerts that affect users?"
    - **Grounded Responses**: When you need answers citing specific graph entities

---

### EntityLinker (The Connector)
Resolves text mentions to unique entities and assigns URIs.

#### **Key Methods**

| Method | Description |
|--------|-------------|
| `link_entities(source, target, type)` | Creates a link between two entities. |
| `assign_uri(entity_name, type)` | Generates a consistent URI for an entity. |

#### **Code Example**
```python
from semantica.context import EntityLinker

linker = EntityLinker(knowledge_graph=graph)

# Link two entities
linker.link_entities(
    source_entity_id="Python",
    target_entity_id="Programming",
    link_type="IS_A",
    confidence=0.95
)
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Global token limit
export CONTEXT_TOKEN_LIMIT=2000
```

### YAML Configuration

```yaml
context:
  short_term_limit: 10
  retrieval:
    hybrid_alpha: 0.5  # 0.0=Vector, 1.0=Graph
    max_expansion_hops: 2
```

---

## 📝 Data Structures

### MemoryItem
The fundamental unit of storage.
```python
@dataclass
class MemoryItem:
    content: str              # The actual text content
    timestamp: datetime       # When it was created
    metadata: Dict            # Arbitrary tags (user_id, source, etc.)
    embedding: List[float]    # The vector representation
    entities: List[Dict]      # Entities found in this content
```

### Graph Node (Dict Format)
```python
{
    "id": "node_unique_id",
    "type": "concept",
    "properties": {
        "content": "Description of the node",
        "weight": 1.0
    }
}
```

### Graph Edge (Dict Format)
```python
{
    "source_id": "origin_node",
    "target_id": "destination_node",
    "type": "related_to",
    "weight": 0.8
}
```

---

## 🧩 Advanced Usage

### Method Registry (Extensibility)
Register custom implementations for graph building, memory management, or retrieval.

#### **Code Example**
```python
from semantica.context import registry

def custom_graph_builder(entities, relationships):
    # Custom logic to build graph
    return "my_graph_structure"

# Register the new method
registry.register("graph", "custom_builder", custom_graph_builder)
```

### Configuration Manager
Programmatically manage configuration settings.

#### **Code Example**
```python
from semantica.context.config import context_config

# Update configuration at runtime
context_config.set("retention_days", 60)

## See Also
- [Vector Store](vector_store.md) - The long-term storage backend
- [Graph Store](graph_store.md) - The knowledge graph backend
- [Reasoning](reasoning.md) - Uses context for logic

## Cookbook
- [Context Module](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/19_Context_Module.ipynb)
- [Advanced Context Engineering](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/11_Advanced_Context_Engineering.ipynb)
