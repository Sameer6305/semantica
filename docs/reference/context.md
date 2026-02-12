# Context Module Reference

> **The central nervous system for intelligent agents, managing memory, knowledge graphs, context graphs, decision tracking, and advanced context retrieval with KG algorithms and vector store integration.**

---

## 🎯 System Overview

The **Context Module** provides agents with a persistent, searchable, and structured memory system with advanced decision tracking capabilities and **context graphs** for sophisticated knowledge representation, ensuring predictable state management and compatibility with modern vector stores and graph databases.

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

-   :material-gavel:{ .lg .middle } **Decision Tracking**

    ---

    Complete decision lifecycle management with precedent search, causal analysis, and policy compliance.

-   :material-chart-line:{ .lg .middle } **KG Algorithms**

    ---

    Advanced graph analytics including centrality, community detection, embeddings, and link prediction.

-   :material-magnify:{ .lg .middle } **Vector Store Features**

    ---

    Hybrid search with custom similarity weights and advanced filtering capabilities.

-   :material-graph:{ .lg .middle } **Context Graphs**

    ---

    Structured knowledge representation with entity relationships, decision history, and semantic context for sophisticated reasoning.

</div>

!!! tip "When to Use"
    - **Memory Persistence**: Enabling agents to remember user preferences and history.
    - **Complex Retrieval**: When simple vector search fails to capture relationships.
    - **Knowledge Graph**: Building a structured world model from unstructured text.
    - **Decision Management**: Tracking, analyzing, and learning from decisions.
    - **Advanced Analytics**: Understanding influence, patterns, and relationships in decisions.

---

## 🏗️ Architecture Components

### AgentContext (The Orchestrator)
The high-level facade that unifies all context operations. It routes data to the appropriate subsystems (Memory, Graph, Vector Store, Decision Tracking) and manages the lifecycle of context.

#### **Constructor Parameters**

- `vector_store` (Required): The backing vector database instance (e.g., FAISS, Weaviate)
- `knowledge_graph` (Optional): The graph store instance for structured knowledge
- `token_limit` (Default: `2000`): The maximum number of tokens allowed in short-term memory before pruning occurs
- `short_term_limit` (Default: `10`): The maximum number of distinct memory items in short-term memory
- `hybrid_alpha` (Default: `0.5`): The weighting factor for retrieval (`0.0` = Pure Vector, `1.0` = Pure Graph)
- `use_graph_expansion` (Default: `True`): Whether to fetch neighbors of retrieved nodes from the graph
- `enable_decision_tracking` (Default: `False`): Enable advanced decision tracking features
- `enable_advanced_analytics` (Default: `False`): Enable KG algorithms and analytics
- `enable_kg_algorithms` (Default: `False`): Enable knowledge graph algorithm integration
- `enable_vector_store_features` (Default: `False`): Enable advanced vector store features

#### **Core Methods**

| Method | Description |
|--------|-------------|
| `store(content, ...)` | Writes information to memory. Handles auto-detection, write-through to vector store, and entity extraction. |
| `retrieve(query, ...)` | Fetches relevant context using hybrid search (Vector + Graph) and reranking. |
| `query_with_reasoning(query, llm_provider, ...)` | **GraphRAG with multi-hop reasoning**: Retrieves context, builds reasoning paths, and generates LLM-based natural language responses grounded in the knowledge graph. |
| `record_decision(category, scenario, reasoning, outcome, confidence, ...)` | Records decisions with full context and metadata for tracking and analysis. |
| `find_precedents(scenario, category, ...)` | Finds similar decisions using advanced search capabilities. |
| `find_precedents_advanced(scenario, similarity_weights, ...)` | Enhanced precedent search with KG features and custom similarity weights. |
| `analyze_decision_influence(decision_id)` | Analyzes decision influence using KG algorithms and centrality measures. |
| `predict_decision_relationships(decision_id)` | Predicts relationships between decisions using link prediction algorithms. |
| `get_context_insights()` | Returns comprehensive system analytics and feature status. |
| `get_causal_chain(decision_id, direction, max_depth)` | Traces decision causality and influence chains. |

#### **Code Example**
```python
from semantica.context import AgentContext
from semantica.vector_store import VectorStore

# 1. Initialize with Advanced Features
vs = VectorStore(backend="faiss", dimension=768)
context = AgentContext(
    vector_store=vs,
    knowledge_graph=kg,
    enable_decision_tracking=True,
    enable_advanced_analytics=True,
    enable_kg_algorithms=True,
    enable_vector_store_features=True
)

# 2. Store Memory
context.store(
    "User is working on a React project.",
    conversation_id="session_1",
    user_id="user_123"
)

# 3. Record Decision
decision_id = context.record_decision(
    category="approval",
    scenario="Loan application for first-time homebuyer",
    reasoning="Strong credit score (750), stable employment, 20% down payment",
    outcome="approved",
    confidence=0.94,
    decision_maker="loan_officer_001"
)

# 4. Retrieve Context
results = context.retrieve("What is the user building?")

# 5. Find Similar Decisions
precedents = context.find_precedents_advanced(
    scenario="High-value credit application",
    category="approval",
    use_kg_features=True,
    similarity_weights={"semantic": 0.5, "structural": 0.3, "category": 0.2}
)

# 6. Analyze Decision Influence
influence = context.analyze_decision_influence(decision_id)

# 7. Get Context Insights
insights = context.get_context_insights()

# 8. Query with Reasoning (GraphRAG)
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

### Decision Tracking System

#### DecisionRecorder (The Decision Engine)
Records decisions with full context, policy applications, and provenance tracking.

**Key Methods:**
| Method | Description |
|--------|-------------|
| `record_decision(category, scenario, reasoning, outcome, confidence, ...)` | Records decisions with full context and metadata |
| `apply_policy(decision_id, policy_id)` | Applies policies to decisions and checks compliance |
| `create_approval_chain(decision_id, approvers)` | Creates multi-level approval workflows |
| `track_provenance(decision_id, source_info)` | Tracks decision provenance and lineage |

#### DecisionQuery (The Decision Search Engine)
Advanced decision querying with precedent search, filtering, and hybrid search operations.

**Key Methods:**
| Method | Description |
|--------|-------------|
| `find_precedents_hybrid(scenario, category, limit)` | Hybrid search with KG and vector store integration |
| `find_precedents_advanced(scenario, similarity_weights, ...)` | Enhanced search with custom similarity weights |
| `analyze_decision_influence(decision_id)` | Analyze decision influence using KG algorithms |
| `predict_decision_relationships(decision_id)` | Predict relationships between decisions |
| `multi_hop_reasoning(decision_id, max_hops)` | Multi-hop reasoning for complex relationships |
| `get_decision_statistics()` | Get comprehensive decision analytics |

#### CausalChainAnalyzer (The Influence Engine)
Analyzes decision causality, influence chains, and precedent relationships.

**Key Methods:**
| Method | Description |
|--------|-------------|
| `get_causal_chain(decision_id, direction, max_depth)` | Trace causal chains from decisions |
| `find_influenced_decisions(decision_id)` | Find decisions influenced by a decision |
| `find_influencing_decisions(decision_id)` | Find decisions that influenced a decision |
| `analyze_causal_impact(decision_id, max_depth)` | Analyze causal impact and scope |
| `calculate_influence_score(decision_id)` | Calculate decision influence scores |

#### PolicyEngine (The Governance Engine)
Policy management with versioning, compliance checking, and impact analysis.

**Key Methods:**
| Method | Description |
|--------|-------------|
| `create_policy(name, rules, category)` | Create new policies with rules and constraints |
| `check_compliance(decision_id, policy_id)` | Check decision compliance with policies |
| `analyze_impact(policy_id, time_range)` | Analyze policy impact on decisions |
| `get_violations(decision_id)` | Get policy violations for decisions |

#### **Decision Tracking Example**
```python
from semantica.context import DecisionRecorder, DecisionQuery, CausalChainAnalyzer, PolicyEngine

# Initialize decision tracking components
recorder = DecisionRecorder(graph_store=kg, vector_store=vs)
query = DecisionQuery(graph_store=kg, vector_store=vs)
analyzer = CausalChainAnalyzer(graph_store=kg)
policy_engine = PolicyEngine(graph_store=kg)

# Record a decision
decision_id = recorder.record_decision(
    category="loan_approval",
    scenario="Mortgage application for first-time homebuyer",
    reasoning="Strong credit score (750), stable employment, 20% down payment",
    outcome="approved",
    confidence=0.94,
    decision_maker="loan_officer_001"
)

# Find similar decisions (precedents)
precedents = query.find_precedents_hybrid(
    scenario="Mortgage application",
    category="loan_approval",
    limit=10
)

# Analyze decision influence
influence = analyzer.analyze_decision_influence(decision_id)

# Check policy compliance
compliance = policy_engine.check_compliance(decision_id, "lending_policy_001")

# Trace causal chain
causal_chain = analyzer.get_causal_chain(decision_id, "downstream", max_depth=3)
```

---

### Knowledge Graph Algorithm Integration

#### Supported KG Algorithms
- **Centrality Analysis**: Degree, betweenness, closeness, eigenvector centrality
- **Community Detection**: Modularity-based community identification
- **Node Embeddings**: Node2Vec embeddings for similarity analysis
- **Path Finding**: Shortest path and advanced path algorithms
- **Link Prediction**: Relationship prediction between entities
- **Similarity Calculation**: Multi-type similarity measures

#### Enhanced ContextGraph Features
```python
from semantica.context import ContextGraph

# Initialize with KG Algorithms
graph = ContextGraph(
    enable_advanced_analytics=True,
    enable_centrality_analysis=True,
    enable_community_detection=True,
    enable_node_embeddings=True
)

# Add nodes and edges
graph.add_node("Python", type="language", properties={"popularity": "high"})
graph.add_edge("Python", "Programming", type="related_to")

# Advanced analytics
centrality = graph.get_node_centrality("Python")
similar = graph.find_similar_nodes("Python", similarity_type="content")
analysis = graph.analyze_graph_with_kg()

# Decision integration
graph.add_decision(decision_id, decision_data)
precedents = graph.find_precedents("loan_approval")
```

---

### Vector Store Integration

#### Hybrid Search Features
- **Semantic + Structural Similarity**: Combined similarity scoring
- **Custom Similarity Weights**: Configurable similarity scoring
- **Advanced Precedent Search**: KG-enhanced similarity search
- **Multi-Embedding Support**: Multiple embedding types
- **Metadata Filtering**: Advanced filtering capabilities

#### Code Example
```python
# Hybrid search with custom weights
precedents = query.find_precedents_hybrid(
    scenario="Loan application",
    category="approval",
    limit=10,
    similarity_weights={
        "semantic": 0.6,
        "structural": 0.3,
        "category": 0.1
    }
)
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
Manages the structured relationships between entities. It provides the "World Model" for the agent with advanced KG algorithm integration and serves as the foundation for **Context Graphs** that enable sophisticated reasoning and decision analysis.

#### **What are Context Graphs?**
**Context Graphs** are structured representations of knowledge that capture:
- **Entity Relationships**: How concepts, people, and decisions are connected
- **Semantic Context**: The meaning and relevance of information within specific domains
- **Decision History**: How past decisions influence current and future choices
- **Knowledge Evolution**: How understanding grows and changes over time

#### **Key Features of Context Graphs**
*   **Dictionary-Based Interface**
    *   *Design*: Uses standard Python dictionaries for nodes and edges, removing dependencies on complex interface classes.
    *   *Benefit*: simpler serialization and easier integration with external APIs.
*   **Advanced Graph Traversal**
    *   *Adjacency List*: optimized internal structure for fast neighbor lookups.
    *   *Multi-Hop Search*: Can traverse `k` hops from a starting node to find indirect connections.
    *   *Path Finding*: Shortest path and advanced path algorithms for relationship discovery.
*   **Rich Node & Edge Types**
    *   *Typed Schema*: Supports distinct types for nodes (e.g., "Person", "Concept", "Decision") and edges (e.g., "KNOWS", "RELATED_TO", "INFLUENCES").
    *   *Metadata Support*: Rich properties and attributes for detailed context capture.
*   **Advanced Analytics Integration**
    *   *KG Algorithm Integration*: Centrality, community detection, embeddings, path finding
    *   *Decision Integration*: Store and analyze decisions in graph context
    *   *Similarity Analysis*: Advanced node similarity with multiple measures
    *   *Influence Analysis**: Track how decisions and entities influence each other

#### **Context Graph Use Cases**
- **Knowledge Management**: Build and query structured knowledge bases
- **Decision Support**: Trace decision precedents and influence patterns
- **Recommendation Systems**: Find related concepts and entities
- **Social Network Analysis**: Understand relationships and influence
- **Research Networks**: Map collaborations and citation patterns

#### **Key Methods**

| Method | Description |
|--------|-------------|
| `add_nodes(nodes)` | Bulk adds nodes using a list of dictionaries. |
| `add_edges(edges)` | Bulk adds edges using a list of dictionaries. |
| `get_neighbors(node_id, hops)` | Returns connected nodes within a specified distance. |
| `query(query_str)` | Performs keyword-based search specifically on graph nodes. |
| `analyze_graph_with_kg()` | Comprehensive graph analysis with KG algorithms. |
| `get_node_centrality(node_id)` | Get centrality measures for specific nodes. |
| `find_similar_nodes(node_id, similarity_type)` | Find similar nodes using advanced similarity. |
| `add_decision(decision_id, decision_data)` | Add decisions with full context integration. |
| `find_precedents(scenario, category)` | Find decision precedents using graph traversal. |
| `trace_influence_paths(entity_id, max_depth)` | Trace how influence propagates through the graph. |
| `get_graph_metrics()` | Get comprehensive graph statistics and health metrics. |

#### **Code Example**
```python
from semantica.context import ContextGraph

# Initialize Context Graph with advanced features
graph = ContextGraph(
    enable_advanced_analytics=True,
    enable_centrality_analysis=True,
    enable_community_detection=True,
    enable_node_embeddings=True
)

# Build Context Graph - Add entities and relationships
graph.add_nodes([
    {
        "id": "Python", 
        "type": "Language", 
        "properties": {
            "paradigm": "OO", 
            "popularity": "high",
            "domain": "programming"
        }
    },
    {
        "id": "FastAPI", 
        "type": "Framework", 
        "properties": {
            "language": "Python",
            "use_case": "web_api",
            "performance": "high"
        }
    },
    {
        "id": "DataScience", 
        "type": "Domain", 
        "properties": {
            "description": "Data analysis and machine learning",
            "tools": ["Python", "R", "SQL"]
        }
    }
])

# Create relationships in Context Graph
graph.add_edges([
    {
        "source_id": "FastAPI", 
        "target_id": "Python", 
        "type": "WRITTEN_IN",
        "properties": {"strength": 0.9}
    },
    {
        "source_id": "Python", 
        "target_id": "DataScience", 
        "type": "USED_IN",
        "properties": {"popularity": 0.95}
    },
    {
        "source_id": "FastAPI", 
        "target_id": "DataScience", 
        "type": "SUPPORTS",
        "properties": {"use_case": "api_for_ml"}
    }
])

# Advanced Context Graph Analytics
centrality = graph.get_node_centrality("Python")
similar = graph.find_similar_nodes("Python", similarity_type="content")
analysis = graph.analyze_graph_with_kg()

# Decision Integration in Context Graph
graph.add_decision("decision_001", {
    "category": "technology_choice",
    "scenario": "Framework selection for web API",
    "reasoning": "Python ecosystem with FastAPI provides best performance",
    "outcome": "selected_fastapi",
    "confidence": 0.92
})

# Find decision precedents in Context Graph
precedents = graph.find_precedents("technology_choice")

# Trace influence through Context Graph
influence_paths = graph.trace_influence_paths("Python", max_depth=3)
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

# 2. Initialize Agent Context with Persistent Graph and Advanced Features
context = AgentContext(
    vector_store=vs,      # Your VectorStore instance
    knowledge_graph=gs,   # Your persistent GraphStore
    enable_decision_tracking=True,
    enable_advanced_analytics=True,
    enable_kg_algorithms=True,
    enable_vector_store_features=True,
    use_graph_expansion=True
)

# Now all graph operations (store, retrieve, build_graph) use Neo4j directly.
```

---

### ContextRetriever (The Search Engine)
The retrieval logic that powers the `retrieve()` command. It implements the **Hybrid Retrieval** algorithm with advanced KG and vector store integration.

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
5.  **KG Algorithm Enhancement**: Uses centrality, community detection, and similarity for advanced ranking.

#### **Code Example**
```python
# The retriever is automatically used by AgentContext.retrieve()
# But can be accessed directly if needed:

retriever = context.retriever

# Perform a manual retrieval with advanced features
results = retriever.retrieve(
    query="web frameworks",
    max_results=5,
    use_kg_features=True,
    similarity_weights={"semantic": 0.7, "structural": 0.3}
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
6. **KG Algorithm Enhancement**: Uses centrality and community detection for enhanced reasoning

#### **Key Features**

- **Multi-Hop Reasoning**: Traverses graph up to configurable hops (default: 2)
- **Reasoning Trace**: Shows entity relationship paths used in reasoning
- **Grounded Responses**: LLM generates answers citing specific graph entities
- **Multiple LLM Providers**: Supports Groq, OpenAI, HuggingFace, and LiteLLM (100+ LLMs)
- **Fallback Handling**: Returns context with reasoning path if LLM unavailable
- **KG Algorithm Integration**: Uses centrality and community detection for enhanced reasoning

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

# Initialize context with advanced features
context = AgentContext(
    vector_store=VectorStore(backend="faiss"),
    knowledge_graph=kg,
    enable_advanced_analytics=True,
    enable_kg_algorithms=True
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
    - **Decision Analysis**: When analyzing decision influence and relationships

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

### Decision
The fundamental unit of decision tracking.
```python
@dataclass
class Decision:
    decision_id: str         # Unique decision identifier
    category: str            # Decision category (approval, rejection, etc.)
    scenario: str            # Decision scenario description
    reasoning: str           # Decision reasoning and explanation
    outcome: str             # Decision outcome
    confidence: float        # Confidence score (0-1)
    decision_maker: str      # Decision maker identifier
    timestamp: datetime      # When decision was made
    entities: List[str]      # Related entities
    metadata: Dict           # Additional decision metadata
    embedding: List[float]   # Decision embedding for similarity
```

### Graph Node (Dict Format)
```python
{
    "id": "node_unique_id",
    "type": "concept",
    "properties": {
        "content": "Description of the node",
        "weight": 1.0,
        "centrality": 0.85,
        "community": "cluster_1"
    }
}
```

### Graph Edge (Dict Format)
```python
{
    "source_id": "origin_node",
    "target_id": "destination_node",
    "type": "related_to",
    "weight": 0.8,
    "properties": {
        "similarity": 0.75,
        "confidence": 0.9
    }
}
```

---

## 🧩 Advanced Usage

### Context Graphs in Production

#### Building Domain-Specific Context Graphs

**Financial Services Context Graph**
```python
from semantica.context import ContextGraph

# Create financial context graph
financial_graph = ContextGraph(enable_advanced_analytics=True)

# Add financial entities
financial_graph.add_nodes([
    {
        "id": "customer_001",
        "type": "Customer",
        "properties": {
            "credit_score": 750,
            "risk_profile": "low",
            "account_type": "premium"
        }
    },
    {
        "id": "mortgage_product",
        "type": "Product",
        "properties": {
            "category": "loan",
            "interest_rate": 3.5,
            "max_amount": 500000
        }
    },
    {
        "id": "loan_officer_001",
        "type": "Agent",
        "properties": {
            "department": "lending",
            "experience_years": 5
        }
    }
])

# Add relationships
financial_graph.add_edges([
    {
        "source_id": "customer_001",
        "target_id": "mortgage_product",
        "type": "ELIGIBLE_FOR",
        "properties": {"confidence": 0.92}
    },
    {
        "source_id": "loan_officer_001",
        "target_id": "customer_001",
        "type": "SERVES",
        "properties": {"relationship_duration": "2_years"}
    }
])

# Analyze financial context
centrality = financial_graph.get_node_centrality("customer_001")
similar_customers = financial_graph.find_similar_nodes("customer_001")
```

**Healthcare Context Graph**
```python
# Create healthcare context graph
healthcare_graph = ContextGraph(enable_advanced_analytics=True)

# Add medical entities
healthcare_graph.add_nodes([
    {
        "id": "patient_001",
        "type": "Patient",
        "properties": {
            "condition": "diabetes_type_2",
            "age": 45,
            "risk_factors": ["obesity", "hypertension"]
        }
    },
    {
        "id": "metformin",
        "type": "Medication",
        "properties": {
            "class": "biguanide",
            "uses": ["diabetes_treatment", "pcos"]
        }
    },
    {
        "id": "dr_smith",
        "type": "Physician",
        "properties": {
            "specialty": "endocrinology",
            "hospital": "general_hospital"
        }
    }
])

# Add medical relationships
healthcare_graph.add_edges([
    {
        "source_id": "patient_001",
        "target_id": "metformin",
        "type": "PRESCRIBED",
        "properties": {"dosage": "500mg", "frequency": "twice_daily"}
    },
    {
        "source_id": "dr_smith",
        "target_id": "patient_001",
        "type": "TREATS",
        "properties": {"since": "2023-01-15"}
    }
])

# Analyze healthcare context
treatment_patterns = healthcare_graph.analyze_graph_with_kg()
similar_patients = healthcare_graph.find_similar_nodes("patient_001")
```

#### Context Graph Analytics and Insights

```python
# Get comprehensive graph insights
insights = graph.get_graph_metrics()
print(f"Graph Density: {insights['density']}")
print(f"Average Clustering: {insights['avg_clustering']}")
print(f"Number of Communities: {len(insights['communities'])}")

# Find influential nodes
influential_nodes = []
for node_id in graph.get_all_nodes():
    centrality = graph.get_node_centrality(node_id)
    if centrality['betweenness'] > 0.8:
        influential_nodes.append(node_id)

# Trace decision influence
decision_influence = graph.trace_influence_paths("decision_001", max_depth=3)
for path in decision_influence:
    print(f"Influence Path: {' -> '.join(path)}")
```

#### Context Graph Visualization

```python
# Export context graph for visualization
graph_data = graph.export_graph(format="networkx")

# Create visualization (requires matplotlib/networkx)
import matplotlib.pyplot as plt
import networkx as nx

G = nx.node_link_graph(graph_data)
pos = nx.spring_layout(G)

# Draw the context graph
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
         node_size=1000, font_size=8, edge_color='gray')
plt.title("Context Graph Visualization")
plt.show()
```

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
- [KG Algorithms](kg.md) - Knowledge graph algorithms and analytics
- [Decision Tracking](decision_tracking.md) - Decision management and analysis
- [Reasoning](reasoning.md) - Uses context for logic

## Cookbook

Interactive tutorials to learn context management, GraphRAG, and decision tracking:

- **[Context Module](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/19_Context_Module.ipynb)**: Practical guide to the context module for AI agents
  - **Topics**: Agent memory, context graph, hybrid retrieval, entity linking, decision tracking
  - **Difficulty**: Intermediate
  - **Use Cases**: Building stateful AI agents, persistent memory systems, decision management

- **[Advanced Context Engineering](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/11_Advanced_Context_Engineering.ipynb)**: Build a production-grade memory system for AI agents
  - **Topics**: Agent memory, GraphRAG, entity injection, lifecycle management, persistent stores, decision analytics
  - **Difficulty**: Advanced
  - **Use Cases**: Production agent systems, advanced memory management, decision analysis

- **[Decision Tracking with KG Algorithms](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/12_Decision_Tracking_KG.ipynb)**: Advanced decision tracking and analytics
  - **Topics**: Decision lifecycle, precedent search, causal analysis, KG algorithms, policy compliance
  - **Difficulty**: Advanced
  - **Use Cases**: Banking decisions, healthcare decisions, legal precedent analysis
