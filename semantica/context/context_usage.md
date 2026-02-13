# Context Module Usage Guide

This guide demonstrates how to use the Semantica context module for building context graphs, managing agent memory, retrieving context, linking entities, and decision tracking with hybrid search capabilities and advanced KG algorithm integration.

## Quick Imports

```python
# Core context classes
from semantica.context import AgentContext, ContextGraph, ContextRetriever, DecisionContext

# Memory management
from semantica.context import AgentMemory

# Entity linking
from semantica.context import EntityLinker

# Decision tracking with advanced features
from semantica.context import Decision, Policy, PolicyException, DecisionRecorder, DecisionQuery, CausalChainAnalyzer, PolicyEngine

# For vector storage (often used with context)
from semantica.vector_store import VectorStore
```

## Quick Example

```python
# Simple context setup
vector_store = VectorStore(backend="inmemory", dimension=384)
context = AgentContext(vector_store=vector_store)

# Store a memory
memory_id = context.store("User likes Python programming", conversation_id="conv1")

# Retrieve context
results = context.retrieve("Python programming", max_results=5)
print(f"Found {len(results)} results")
```

## Table of Contents

1. [High-Level Interface (Quick Start)](#high-level-interface-quick-start)
2. [Basic Usage](#basic-usage)
3. [Enhanced AgentContext with Decision Tracking and KG Algorithms](#enhanced-agentcontext-with-decision-tracking-and-kg-algorithms)
4. [Context Graph Construction](#context-graph-construction)
5. [Enhanced ContextGraph with KG Algorithms](#enhanced-contextgraph-with-kg-algorithms)
6. [Agent Memory Management](#agent-memory-management)
7. [Context Retrieval](#context-retrieval)
8. [Entity Linking](#entity-linking)
9. [Decision Tracking](#decision-tracking)
10. [Policy Exception Management](#policy-exception-management)
11. [Hybrid Search for Decisions](#hybrid-search-for-decisions)
12. [Context Graphs with KG Algorithms](#context-graphs-with-kg-algorithms)
13. [Advanced Decision Analytics](#advanced-decision-analytics)
14. [Production Examples](#production-examples)
15. [Explainable AI](#explainable-ai)

## High-Level Interface (Quick Start)

The `AgentContext` class provides a simplified, generic interface for common use cases. It integrates vector storage, knowledge graphs, and memory management into a unified system.

### Simple RAG (Vector Only)

```python
from semantica.context import AgentContext
from semantica.vector_store import VectorStore

# Initialize vector store
vs = VectorStore(backend="faiss", dimension=768)

# Initialize context
context = AgentContext(vector_store=vs)

# Store a memory
memory_id = context.store("User likes Python programming", conversation_id="conv1")

# Retrieve context
results = context.retrieve("Python programming", max_results=5)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']:.2f}")
```

### GraphRAG (Vector + Graph)

```python
from semantica.context import AgentContext, ContextGraph
from semantica.graph_store import GraphStore

# Initialize persistent knowledge graph (Recommended for production)
try:
    kg = GraphStore(backend="neo4j", uri="bolt://localhost:7687", user="neo4j", password="password")
    kg.connect()
except:
    print("Neo4j not available, falling back to in-memory graph")
    kg = ContextGraph()

# Initialize context with vector store and knowledge graph
context = AgentContext(vector_store=vs, knowledge_graph=kg)

# Store documents (auto-builds graph)
documents = [
    "Python is a programming language used for machine learning",
    "TensorFlow and PyTorch are popular ML frameworks",
    "Machine learning involves training models on data"
]

stats = context.store(
    documents,
    extract_entities=True,      # Extract entities from documents
    extract_relationships=True,  # Extract relationships
    link_entities=True          # Link entities across documents
)

print(f"Stored {stats['stored_count']} documents")
# Graph stats are available via the graph object directly or context stats
print(f"Graph nodes: {kg.stats()['node_count']}")

# Retrieve with graph context (auto-detects GraphRAG)
results = context.retrieve(
    "Python machine learning",
    use_graph=True,              # Explicitly use graph
    include_entities=True,       # Include related entities
    expand_graph=True            # Use graph expansion
)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Related entities: {len(result.get('related_entities', []))}")
```

### Agent Memory Management (Hierarchical)

The system uses a hierarchical memory structure with:
1.  **Short-Term Memory**: Fast, in-memory buffer with token and item count limits.
2.  **Long-Term Memory**: Persistent vector store.

```python
context = AgentContext(
    vector_store=vs,
    retention_days=30,
    short_term_limit=10,  # Max items in short-term buffer
    token_limit=2000      # Max tokens in short-term buffer
)

# Store multiple memories in a conversation
context.store("Hello, I'm interested in Python", conversation_id="conv1", user_id="user123")
context.store("What can you tell me about machine learning?", conversation_id="conv1", user_id="user123")

# Get conversation history
history = context.conversation(
    "conv1",
    reverse=True,           # Most recent first
    include_metadata=True   # Include full metadata
)

for item in history:
    print(f"{item['timestamp']}: {item['content']}")

# Delete old memories
deleted_count = context.forget(days_old=90)
print(f"Deleted {deleted_count} old memories")
```

### Persistence (Save/Load)

You can save the entire state of the agent (Memory, Graph, and Vector Index) to disk and reload it later.

```python
# Save state
context.save("./my_agent_state")

# Load state
new_context = AgentContext(vector_store=VectorStore(), knowledge_graph=ContextGraph())
new_context.load("./my_agent_state")
```

## Basic Usage

### Initialization with Backends

You can configure the `VectorStore` with different backends (`inmemory`, `faiss`, `chroma`, `qdrant`, `weaviate`, `milvus`) and embedding models (including FastEmbed).

```python
from semantica.context import AgentContext, ContextGraph
from semantica.vector_store import VectorStore

# Initialize Vector Store with FastEmbed
vs = VectorStore(backend="inmemory", dimension=384)
if hasattr(vs, "embedder") and vs.embedder:
    vs.embedder.set_text_model(method="fastembed", model_name="BAAI/bge-small-en-v1.5")

# Initialize Context Graph
kg = ContextGraph()

# Initialize Agent Context
context = AgentContext(vector_store=vs, knowledge_graph=kg)
```

### Enhanced AgentContext with Decision Tracking and KG Algorithms

The enhanced `AgentContext` supports advanced decision tracking, KG algorithm integration, and vector store features for production-grade context engineering.

```python
from semantica.context import AgentContext, ContextGraph
from semantica.vector_store import VectorStore
from semantica.graph_store import GraphStore  # For decision tracking

# Initialize Vector Store
vs = VectorStore(backend="inmemory", dimension=768)

# Initialize Graph Store (required for decision tracking)
# Note: Decision tracking requires a GraphStore with execute_query() support
gs = GraphStore(backend="neo4j", uri="bolt://localhost:7687")

# Initialize Context Graph with KG algorithms
kg = ContextGraph(
    enable_advanced_analytics=True,
    enable_centrality_analysis=True,
    enable_community_detection=True,
    enable_node_embeddings=True
)

# Initialize Enhanced Agent Context
context = AgentContext(
    vector_store=vs,
    knowledge_graph=kg,
    enable_decision_tracking=True,        # Enable decision lifecycle management
    enable_advanced_analytics=True,       # Enable KG algorithm integration
    enable_kg_algorithms=True,            # Enable centrality, community detection
    enable_vector_store_features=True     # Enable hybrid search capabilities
)

# Record a decision with full context
decision_id = context.record_decision(
    category="mortgage_approval",
    scenario="First-time homebuyer application",
    reasoning="Strong credit score, stable employment, low debt-to-income ratio",
    outcome="approved",
    confidence=0.94,
    decision_maker="loan_officer_001"
)

# Find similar decisions with KG-enhanced search
precedents = context.find_precedents_advanced(
    scenario="Mortgage application",
    use_kg_features=True,
    similarity_weights={"semantic": 0.5, "structural": 0.3, "category": 0.2}
)

# Analyze decision influence
influence = context.analyze_decision_influence(decision_id)
print(f"Influence score: {influence.get('influence_score', 0):.3f}")
print(f"Centrality measures: {influence.get('centrality_measures', {})}")

# Get comprehensive context insights
insights = context.get_context_insights()
print(f"Advanced features: {insights.get('advanced_features', {})}")
```

## Context Graph Construction

The `ContextGraph` class is an in-memory graph store.

### Building from Entities and Relationships

```python
from semantica.context import ContextGraph

graph = ContextGraph()

entities = [
    {"id": "e1", "text": "Python", "type": "PROGRAMMING_LANGUAGE"},
    {"id": "e2", "text": "Machine Learning", "type": "CONCEPT"},
    {"id": "e3", "text": "TensorFlow", "type": "FRAMEWORK"},
]

relationships = [
    {"source_id": "e1", "target_id": "e2", "type": "used_for", "confidence": 0.9},
    {"source_id": "e3", "target_id": "e2", "type": "implements", "confidence": 0.95},
]

graph_data = graph.build_from_entities_and_relationships(entities, relationships)

print(f"Nodes: {graph.stats()['node_count']}")
print(f"Edges: {graph.stats()['edge_count']}")
```

### Building from Conversations

```python
from semantica.context import ContextGraph

graph = ContextGraph()

conversations = [
    {
        "id": "conv1",
        "content": "User asked about Python programming",
        "entities": [
            {"id": "e1", "text": "Python", "type": "PROGRAMMING_LANGUAGE"}
        ],
        "relationships": []
    }
]

graph_data = graph.build_from_conversations(
    conversations,
    link_entities=True,
    extract_intents=True
)
```

### Adding Nodes and Edges Manually

```python
from semantica.context import ContextGraph

graph = ContextGraph()

# Add nodes
graph.add_node("node1", "entity", "Python programming", confidence=0.9)
graph.add_node("node2", "concept", "Machine Learning", confidence=0.95)

# Add edges
graph.add_edge("node1", "node2", "related_to", weight=0.9)

# Get neighbors
neighbors = graph.get_neighbors("node1", hops=2)
print(f"Neighbors: {neighbors}")

# Query graph
results = graph.query("Python") # Keyword search on nodes
```

### Graph Statistics and Analysis

```python
stats = graph.stats()
print(f"Node types: {stats['node_types']}")
print(f"Density: {stats['density']:.4f}")

# Find specific nodes/edges
entities = graph.find_nodes(node_type="entity")
relations = graph.find_edges(edge_type="related_to")
node = graph.find_node("node1")
```

### Enhanced ContextGraph with KG Algorithms

The enhanced `ContextGraph` supports advanced KG algorithms for centrality analysis, community detection, and node embeddings.

```python
from semantica.context import ContextGraph

# Initialize with KG algorithms enabled
graph = ContextGraph(
    enable_advanced_analytics=True,
    enable_centrality_analysis=True,
    enable_community_detection=True,
    enable_node_embeddings=True
)

# Add nodes and edges
graph.add_node("Python", "Language", {"popularity": "high"})
graph.add_node("FastAPI", "Framework", {"language": "Python"})
graph.add_node("Django", "Framework", {"language": "Python"})
graph.add_edge("FastAPI", "Python", "WRITTEN_IN")
graph.add_edge("Django", "Python", "WRITTEN_IN")

# Centrality analysis
centrality = graph.get_node_centrality("Python")
print(f"Python centrality: {centrality}")

# Find similar nodes using embeddings
similar_nodes = graph.find_similar_nodes("Python", similarity_type="content")
print(f"Similar nodes to Python: {[node['id'] for node in similar_nodes]}")

# Community detection
analysis = graph.analyze_graph_with_kg()
communities = analysis.get('community_analysis', {})
print(f"Communities found: {communities.get('num_communities', 0)}")

# Node embeddings
embeddings = graph.get_node_embeddings("Python")
print(f"Python embedding dimension: {len(embeddings) if embeddings else 0}")
```

## Agent Memory Management

The `AgentMemory` class handles short-term and long-term memory with hierarchical storage and token management.

### Storing and Retrieving

```python
from semantica.context import AgentMemory

memory = AgentMemory(
    vector_store=vs,
    knowledge_graph=kg,
    retention_policy="30_days",
    max_memory_size=10000,
    short_term_limit=20,    # 20 items max in short-term
    token_limit=4000        # 4000 tokens max in short-term
)

# Store (automatically updates short-term and long-term)
memory_id = memory.store(
    "User asked about Python programming",
    metadata={"conversation_id": "conv_123"}
)

# Store short-term only (fleeting thoughts)
temp_id = memory.store(
    "Just checking status...",
    skip_vector=True
)

# Retrieve
results = memory.retrieve(
    "Python programming",
    max_results=5,
    type="conversation"
)

# Conversation History
history = memory.get_conversation_history("conv_123")
```

## Context Retrieval

The `ContextRetriever` implements hybrid retrieval strategies.

### Hybrid Retrieval

```python
from semantica.context import ContextRetriever

retriever = ContextRetriever(
    memory_store=memory,
    knowledge_graph=kg,
    vector_store=vs,
    use_graph_expansion=True,
    max_expansion_hops=2,
    hybrid_alpha=0.5  # Balance between vector (0.0) and graph (1.0)
)

results = retriever.retrieve(
    "Python programming",
    max_results=5
)

for result in results:
    print(f"Content: {result.content}")
    print(f"Source: {result.source}") # 'vector', 'graph', or 'memory'
```

## Entity Linking

The `EntityLinker` helps resolve entities to canonical forms or URIs.

```python
from semantica.context import EntityLinker

linker = EntityLinker()
uri = linker.generate_uri("Python Programming Language")
print(uri) # e.g., "python_programming_language"

# Similarity matching
score = linker._calculate_text_similarity("Python", "Python Language")
```

## Decision Tracking

The `DecisionContext` class provides decision tracking capabilities with hybrid search, explainable AI, and KG algorithm integration.

### Basic Decision Recording

```python
from semantica.context import DecisionContext
from semantica.vector_store import VectorStore

# Initialize decision context
vector_store = VectorStore(backend="inmemory", dimension=384)
decision_context = DecisionContext(vector_store=vector_store, graph_store=None)

# Record a decision
decision_id = decision_context.record_decision(
    scenario="Credit limit increase for premium customer",
    reasoning="Excellent payment history and high credit score",
    outcome="approved",
    confidence=0.92,
    entities=["customer_123", "premium_segment", "credit_card"],
    category="credit_approval",
    amount=50000,
    risk_level="low"
)

print(f"Recorded decision: {decision_id}")
```

### Batch Decision Processing

```python
# Process multiple decisions
decisions = [
    {
        "scenario": "Credit limit increase request",
        "reasoning": "Good payment history",
        "outcome": "approved",
        "confidence": 0.85,
        "entities": ["customer_456"],
        "category": "credit_approval"
    },
    {
        "scenario": "Fraud detection alert",
        "reasoning": "Suspicious transaction pattern",
        "outcome": "blocked",
        "confidence": 0.95,
        "entities": ["transaction_789", "customer_456"],
        "category": "fraud_detection"
    }
]

decision_ids = []
for decision in decisions:
    decision_id = decision_context.record_decision(**decision)
    decision_ids.append(decision_id)

print(f"Processed {len(decision_ids)} decisions")
```

### Decision Context Retrieval

```python
# Get comprehensive decision context
context_info = decision_context.get_decision_context(
    decision_id,
    depth=2,
    include_entities=True,
    include_policies=True
)

print(f"Decision context: {len(context_info.related_entities)} entities")
print(f"Related relationships: {len(context_info.related_relationships)}")
```

### Policy Exception Management

The enhanced decision tracking system supports policy exceptions with proper audit trails.

```python
from semantica.context import PolicyException, PolicyEngine
from datetime import datetime

# Create a policy exception
exception = PolicyException(
    exception_id="exc_001",
    decision_id="decision_123",
    policy_id="lending_policy_v2",
    reason="Customer relationship exception - long-term premium client",
    approver="branch_manager_001",
    approval_timestamp=datetime.now(),
    justification="Customer has 10-year history with excellent payment record"
)

# Convert to dictionary for storage
exception_dict = exception.to_dict()
print(f"Exception recorded: {exception_dict['exception_id']}")

# Create exception from dictionary (e.g., when loading from database)
recreated_exception = PolicyException.from_dict(exception_dict)
print(f"Recreated exception: {recreated_exception.reason}")

# Policy engine can record exceptions in GraphStore
policy_engine = PolicyEngine(graph_store)
exception_id = policy_engine.record_exception(
    decision_id="decision_123",
    policy_id="lending_policy_v2",
    reason="Long-term customer relationship exception"
)
print(f"Policy exception recorded: {exception_id}")
```

## Hybrid Search for Decisions

The context retriever supports hybrid search combining semantic and structural embeddings.

### Finding Similar Decisions

```python
from semantica.context import ContextRetriever

# Initialize retriever with decision context
retriever = ContextRetriever(
    vector_store=vector_store,
    knowledge_graph=None
)

# Find similar decisions using hybrid search
precedents = decision_context.find_similar_decisions(
    scenario="Credit limit increase for good customer",
    limit=5,
    use_hybrid_search=True,
    semantic_weight=0.7,
    structural_weight=0.3
)

for precedent in precedents:
    print(f"Score: {precedent['score']:.3f}")
    print(f"Content: {precedent['content'][:100]}...")
    print(f"Entities: {precedent['related_entities']}")
```

### Decision Precedent Search

```python
# Search for decision precedents
precedents = retriever.retrieve_decision_precedents(
    query="Credit approval for premium customers",
    limit=10,
    use_hybrid_search=True,
    include_context=True
)

print(f"Found {len(precedents)} precedents")

for precedent in precedents:
    print(f"Scenario: {precedent['scenario']}")
    print(f"Outcome: {precedent['outcome']}")
    print(f"Confidence: {precedent['confidence']}")
```

### Query Decisions with Context

```python
# Query decisions with multi-hop context expansion
queried = retriever.query_decisions(
    query="High-risk credit decisions",
    max_hops=2,
    include_context=True,
    use_hybrid_search=True,
    filters={"category": "credit_approval", "risk_level": "high"}
)

print(f"Found {len(queried)} high-risk decisions")
```

## Explainable AI

The decision tracking system provides comprehensive explanations with path tracing and confidence scoring.

### Decision Explanations

```python
# Generate comprehensive decision explanation
explanation = decision_context.explain_decision(
    decision_id,
    include_paths=True,
    include_confidence=True,
    include_weights=True
)

print(f"Scenario: {explanation['scenario']}")
print(f"Reasoning: {explanation['reasoning']}")
print(f"Outcome: {explanation['outcome']}")
print(f"Confidence: {explanation['confidence']}")

# Available explanation components
components = [
    "scenario", "reasoning", "outcome", "confidence",
    "semantic_weight", "structural_weight", "embedding_info",
    "related_entities", "similar_decisions", "path_tracing"
]

for component in components:
    if component in explanation:
        print(f"{component}: {explanation[component]}")
```

### Path Tracing and Context

```python
# Get decision with path tracing
explanation = decision_context.explain_decision(
    decision_id,
    include_paths=True,
    max_depth=3
)

# Trace decision paths
if "path_tracing" in explanation:
    paths = explanation["path_tracing"]
    for path in paths:
        print(f"Path: {' -> '.join(path['entities'])}")
        print(f"Confidence: {path['confidence']}")
        print(f"Relationships: {path['relationships']}")
```

### Confidence and Weight Analysis

```python
# Analyze decision confidence and weights
explanation = decision_context.explain_decision(
    decision_id,
    include_confidence=True,
    include_weights=True
)

print(f"Decision confidence: {explanation['confidence']}")
print(f"Semantic weight: {explanation['semantic_weight']}")
print(f"Structural weight: {explanation['structural_weight']}")

# Check if structural embedding was used
if "has_structural_embedding" in explanation:
    has_structural = explanation["has_structural_embedding"]
    print(f"Structural embedding used: {has_structural}")
```

### Real-World Examples

```python
# Banking decision example
banking_decision = decision_context.record_decision(
    scenario="Mortgage application approval",
    reasoning="Strong credit score (750), stable employment, 20% down payment",
    outcome="approved",
    confidence=0.94,
    entities=["applicant_001", "mortgage_30yr", "property_main"],
    category="mortgage_approval",
    loan_amount=350000,
    credit_score=750
)

# Get banking decision explanation
banking_explanation = decision_context.explain_decision(banking_decision)
print(f"Banking decision: {banking_explanation['outcome']}")
print(f"Risk assessment: {banking_explanation['confidence']}")

# Insurance decision example
insurance_decision = decision_context.record_decision(
    scenario="Auto insurance claim approval",
    reasoning="Clear liability, reasonable repair costs, no prior claims",
    outcome="approved",
    confidence=0.96,
    entities=["claim_auto_001", "driver_safe", "policy_active"],
    category="auto_insurance",
    claim_amount=2500
)

# Find similar insurance decisions
insurance_precedents = decision_context.find_similar_decisions(
    scenario="Auto claim with clear liability",
    limit=5,
    filters={"category": "auto_insurance"}
)

print(f"Found {len(insurance_precedents)} similar insurance claims")
```

## Context Graphs with KG Algorithms

The context module now integrates with `semantica.kg` algorithms to provide advanced graph analytics, centrality measures, community detection, and node embeddings for comprehensive context graph analysis.

### Initializing Context Graph with KG Features

```python
from semantica.context import ContextGraph
from semantica.graph_store import GraphStore

# Context graph with KG algorithms
graph = ContextGraph(
    enable_advanced_analytics=True,      # Enable KG algorithms
    enable_centrality_analysis=True,     # Enable centrality measures
    enable_community_detection=True,     # Enable community detection
    enable_node_embeddings=True          # Enable Node2Vec embeddings
)

print(f"KG components initialized: {len(graph.kg_components)}")
```

### Graph Analytics with KG Algorithms

```python
# Comprehensive graph analysis
analysis = graph.analyze_graph_with_kg()

print(f"Graph metrics:")
print(f"  - Node count: {analysis['graph_metrics']['node_count']}")
print(f"  - Edge count: {analysis['graph_metrics']['edge_count']}")
print(f"  - Node types: {analysis['graph_metrics']['node_types']}")

# Centrality analysis
if 'centrality_analysis' in analysis:
    centrality = analysis['centrality_analysis']
    print(f"  - Centrality measures available for {len(centrality)} nodes")

# Community detection
if 'community_analysis' in analysis:
    communities = analysis['community_analysis']
    print(f"  - Found {communities['num_communities']} communities")
    print(f"  - Modularity: {communities['modularity']:.3f}")

# Node embeddings
if 'node_embeddings' in analysis:
    embeddings = analysis['node_embeddings']
    print(f"  - Generated embeddings for {len(embeddings)} nodes")
```

### Node Centrality Analysis

```python
# Get centrality measures for a specific node
node_id = "python_programming"
centrality_measures = graph.get_node_centrality(node_id)

print(f"Centrality measures for {node_id}:")
print(f"  - Degree centrality: {centrality_measures.get('degree_centrality', 0):.3f}")
print(f"  - Betweenness centrality: {centrality_measures.get('betweenness_centrality', 0):.3f}")
print(f"  - Closeness centrality: {centrality_measures.get('closeness_centrality', 0):.3f}")
print(f"  - Eigenvector centrality: {centrality_measures.get('eigenvector_centrality', 0):.3f}")
```

### Finding Similar Nodes with Advanced Similarity

```python
# Find similar nodes using different similarity measures
similar_nodes = graph.find_similar_nodes(
    node_id="python_programming",
    similarity_type="content",      # "content", "structural", "embedding"
    top_k=10
)

print(f"Similar nodes to 'python_programming':")
for node_id, similarity_score in similar_nodes:
    print(f"  - {node_id}: {similarity_score:.3f}")

# Structural similarity
structural_similar = graph.find_similar_nodes(
    node_id="python_programming",
    similarity_type="structural",
    top_k=5
)
```

### AgentContext with KG Features

```python
from semantica.context import AgentContext
from semantica.vector_store import VectorStore
from semantica.graph_store import GraphStore

# Initialize AgentContext with all KG features
vector_store = VectorStore(backend="inmemory", dimension=384)
knowledge_graph = GraphStore(backend="neo4j", uri="bolt://localhost:7687")

context = AgentContext(
    vector_store=vector_store,
    knowledge_graph=knowledge_graph,
    enable_decision_tracking=True,
    enable_advanced_analytics=True,      # Enable KG algorithms
    enable_kg_algorithms=True,          # Enable KG integration
    enable_vector_store_features=True    # Enable vector store features
)

print("AgentContext initialized with KG algorithms")
```

## Advanced Decision Analytics

The decision tracking system provides advanced analytics using KG algorithms for decision influence analysis, relationship prediction, and comprehensive insights.

### DecisionQuery with KG Integration

```python
from semantica.context import DecisionQuery

# Decision query with KG algorithms
query = DecisionQuery(
    graph_store=knowledge_graph,
    vector_store=vector_store,
    enable_advanced_analytics=True,
    enable_centrality_analysis=True,
    enable_community_detection=True,
    enable_node_embeddings=True
)

print(f"DecisionQuery with {len(query.kg_components)} KG components")
```

### Advanced Precedent Search with Custom Weights

```python
# Find precedents using advanced search with custom similarity weights
precedents = context.find_precedents_advanced(
    scenario="Credit limit increase for premium customer",
    category="credit_approval",
    limit=10,
    use_kg_features=True,
    similarity_weights={
        "semantic": 0.4,      # Vector similarity
        "structural": 0.3,    # Graph structure similarity
        "text": 0.2,         # Text overlap
        "category": 0.1      # Category matching
    }
)

print(f"Found {len(precedents)} precedents with advanced search")
for precedent in precedents:
    print(f"  - Score: {precedent.metadata.get('similarity_score', 0):.3f}")
    print(f"  - Scenario: {precedent.scenario[:100]}...")
```

### Decision Influence Analysis

```python
# Analyze decision influence using KG algorithms
decision_id = "decision_123"
influence_analysis = context.analyze_decision_influence(
    decision_id=decision_id,
    max_depth=3
)

print(f"Decision influence analysis for {decision_id}:")
print(f"  - Influence score: {influence_analysis.get('influence_score', 0):.3f}")

# Centrality measures
centrality = influence_analysis.get('centrality_measures', {})
print(f"  - Degree centrality: {centrality.get('degree_centrality', 0):.3f}")
print(f"  - Betweenness centrality: {centrality.get('betweenness_centrality', 0):.3f}")

# Community information
community = influence_analysis.get('community_info', {})
if community:
    print(f"  - Community ID: {community.get('community_id')}")
    print(f"  - Community size: {community.get('community_size')}")

# Related decisions
downstream = influence_analysis.get('downstream_decisions', [])
upstream = influence_analysis.get('upstream_decisions', [])
print(f"  - Downstream decisions: {len(downstream)}")
print(f"  - Upstream decisions: {len(upstream)}")
```

### Decision Relationship Prediction

```python
# Predict potential relationships for decisions
predictions = context.predict_decision_relationships(
    decision_id="decision_123",
    top_k=5
)

print(f"Predicted relationships for decision_123:")
for prediction in predictions:
    print(f"  - Target: {prediction.get('target', 'unknown')}")
    print(f"  - Score: {prediction.get('score', 0):.3f}")
    print(f"  - Type: {prediction.get('type', 'unknown')}")
```

### Context Graph Analysis

```python
# Analyze the entire context graph
graph_analysis = context.analyze_context_graph()

print("Context graph analysis:")
if 'error' not in graph_analysis:
    metrics = graph_analysis.get('graph_metrics', {})
    print(f"  - Nodes: {metrics.get('node_count', 0)}")
    print(f"  - Edges: {metrics.get('edge_count', 0)}")
    
    centrality = graph_analysis.get('centrality_analysis', {})
    print(f"  - Centrality analysis: {len(centrality)} nodes analyzed")
    
    communities = graph_analysis.get('community_analysis', {})
    print(f"  - Communities: {communities.get('num_communities', 0)}")
else:
    print(f"  - Error: {graph_analysis['error']}")
```

### Entity Similarity and Centrality

```python
# Find similar entities in the context graph
similar_entities = context.find_similar_entities(
    entity_id="python_programming",
    similarity_type="content",
    top_k=10
)

print(f"Similar entities to 'python_programming':")
for entity_id, similarity_score in similar_entities:
    print(f"  - {entity_id}: {similarity_score:.3f}")

# Get entity centrality measures
entity_centrality = context.get_entity_centrality("python_programming")
print(f"Entity centrality: {entity_centrality}")
```

### Comprehensive Context Insights

```python
# Get comprehensive insights about the context
insights = context.get_context_insights()

print("Context insights:")
print(f"  - Timestamp: {insights.get('timestamp')}")

# Memory statistics
memory_stats = insights.get('memory_stats', {})
print(f"  - Total memories: {memory_stats.get('total_items', 0)}")
print(f"  - Memory usage: {memory_stats.get('memory_usage', {})}")

# Decision statistics
decision_stats = insights.get('decision_stats', {})
if decision_stats:
    print(f"  - Total decisions: {decision_stats.get('total_decisions', 0)}")
    print(f"  - Decision categories: {decision_stats.get('categories', [])}")

# Advanced features status
features = insights.get('advanced_features', {})
print(f"  - KG algorithms enabled: {features.get('kg_algorithms_enabled', False)}")
print(f"  - Vector store features enabled: {features.get('vector_store_features_enabled', False)}")
print(f"  - Decision tracking enabled: {features.get('decision_tracking_enabled', False)}")
```

## Production Examples

### Banking Decision System with KG Analytics

```python
# Initialize banking decision system
banking_context = AgentContext(
    vector_store=VectorStore(backend="faiss", dimension=768),
    knowledge_graph=GraphStore(backend="neo4j", uri="bolt://localhost:7687"),
    enable_decision_tracking=True,
    enable_advanced_analytics=True,
    enable_kg_algorithms=True,
    enable_vector_store_features=True
)

# Record banking decisions with full context
decisions = [
    {
        "category": "mortgage_approval",
        "scenario": "Mortgage application for first-time homebuyer",
        "reasoning": "Strong credit score (750), stable employment, 20% down payment",
        "outcome": "approved",
        "confidence": 0.94,
        "decision_maker": "loan_officer_001",
        "amount": 350000,
        "credit_score": 750,
        "risk_level": "low"
    },
    {
        "category": "credit_card_approval",
        "scenario": "Premium credit card application",
        "reasoning": "Excellent credit history, high income, existing relationship",
        "outcome": "approved",
        "confidence": 0.96,
        "decision_maker": "credit_analyst_002",
        "credit_limit": 25000,
        "credit_score": 820,
        "risk_level": "very_low"
    }
]

# Process decisions
decision_ids = []
for decision_data in decisions:
    decision_id = banking_context.record_decision(**decision_data)
    decision_ids.append(decision_id)

# Analyze decision influence
for decision_id in decision_ids:
    influence = banking_context.analyze_decision_influence(decision_id)
    print(f"Decision {decision_id} influence: {influence.get('influence_score', 0):.3f}")

# Find similar decisions with KG features
similar_decisions = banking_context.find_precedents_advanced(
    scenario="High-value credit application",
    category="credit_approval",
    use_kg_features=True,
    similarity_weights={"semantic": 0.5, "structural": 0.3, "category": 0.2}
)

print(f"Found {len(similar_decisions)} similar decisions with KG analysis")

# Get comprehensive insights
insights = banking_context.get_context_insights()
print(f"Banking system insights: {insights.get('memory_stats', {})}")
```

### Healthcare Decision Support System

```python
# Healthcare decision system with advanced analytics
healthcare_context = AgentContext(
    vector_store=VectorStore(backend="chroma", dimension=1536),
    knowledge_graph=GraphStore(backend="neo4j"),
    enable_decision_tracking=True,
    enable_advanced_analytics=True,
    enable_kg_algorithms=True
)

# Record medical decisions
medical_decisions = [
    {
        "category": "treatment_approval",
        "scenario": "Approval for experimental cancer treatment",
        "reasoning": "Patient meets criteria, no alternative treatments available",
        "outcome": "approved",
        "confidence": 0.88,
        "decision_maker": "dr_smith",
        "patient_id": "patient_123",
        "condition": "stage_4_lung_cancer",
        "treatment_type": "immunotherapy"
    },
    {
        "category": "diagnostic_test",
        "scenario": "MRI scan authorization",
        "reasoning": "Symptoms indicate need for detailed imaging",
        "outcome": "approved",
        "confidence": 0.92,
        "decision_maker": "dr_jones",
        "patient_id": "patient_456",
        "test_type": "brain_mri",
        "urgency": "medium"
    }
]

# Process medical decisions
for decision in medical_decisions:
    decision_id = healthcare_context.record_decision(**decision)
    
    # Analyze decision influence in medical context
    influence = healthcare_context.analyze_decision_influence(decision_id)
    print(f"Medical decision influence: {influence.get('influence_score', 0):.3f}")

# Find similar treatment decisions
similar_treatments = healthcare_context.find_precedents_advanced(
    scenario="Cancer treatment approval",
    category="treatment_approval",
    use_kg_features=True
)

print(f"Found {len(similar_treatments)} similar treatment decisions")

# Analyze healthcare context graph
graph_analysis = healthcare_context.analyze_context_graph()
if 'error' not in graph_analysis:
    print(f"Healthcare graph: {graph_analysis.get('graph_metrics', {})}")
```

### E-commerce Personalization System

```python
# E-commerce system with KG recommendations
ecommerce_context = AgentContext(
    vector_store=VectorStore(backend="qdrant", dimension=1024),
    knowledge_graph=GraphStore(backend="neo4j"),
    enable_decision_tracking=True,
    enable_advanced_analytics=True,
    enable_kg_algorithms=True
)

# Record personalization decisions
personalization_decisions = [
    {
        "category": "product_recommendation",
        "scenario": "Premium product recommendation for VIP customer",
        "reasoning": "High purchase history, premium segment, similar preferences",
        "outcome": "recommended",
        "confidence": 0.91,
        "decision_maker": "recommendation_engine",
        "customer_id": "vip_customer_001",
        "product_category": "luxury_goods",
        "price_range": "high"
    },
    {
        "category": "pricing_decision",
        "scenario": "Dynamic pricing adjustment",
        "reasoning": "High demand, low inventory, competitor pricing",
        "outcome": "price_increased",
        "confidence": 0.87,
        "decision_maker": "pricing_algorithm",
        "product_id": "product_789",
        "original_price": 99.99,
        "new_price": 119.99
    }
]

# Process e-commerce decisions
for decision in personalization_decisions:
    decision_id = ecommerce_context.record_decision(**decision)
    
    # Find similar customers using KG features
    similar_customers = ecommerce_context.find_similar_entities(
        entity_id=decision.get('customer_id', ''),
        similarity_type="structural",
        top_k=5
    )
    print(f"Similar customers: {len(similar_customers)}")

# Get comprehensive e-commerce insights
insights = ecommerce_context.get_context_insights()
print(f"E-commerce system status: {insights.get('advanced_features', {})}")
```

### Backward Compatibility Examples

All existing code continues to work without changes:

```python
# Old API still works perfectly
context = AgentContext(vector_store=vector_store)
query = DecisionQuery(graph_store)
graph = ContextGraph()

# Store and retrieve as before
memory_id = context.store("User message", conversation_id="conv1")
results = context.retrieve("User query")

# Decision tracking with old API
if hasattr(context, 'record_decision'):
    decision_id = context.record_decision(
        category="test",
        scenario="Test scenario",
        reasoning="Test reasoning",
        outcome="approved",
        confidence=0.8
    )
```

## Summary

The context module provides:

- **Backward Compatibility**: All existing code works unchanged
- **KG Algorithm Integration**: Advanced graph analytics with centrality, community detection, embeddings
- **Vector Store Features**: Hybrid search combining semantic and structural similarity
- **Advanced Decision Analytics**: Influence analysis, relationship prediction, comprehensive insights
- **Production Ready**: Scalable architecture for real-world applications

Users can now build truly comprehensive context graphs with semantica, leveraging all advanced KG algorithms and vector store features while maintaining complete backward compatibility!
