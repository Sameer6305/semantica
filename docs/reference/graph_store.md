# Graph Store

> **Unified interface for Property Graph Databases (Neo4j, KuzuDB, FalkorDB).**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-database-search:{ .lg .middle } **Multi-Backend**

    ---

    Support for Neo4j (Enterprise), KuzuDB (Embedded), and FalkorDB (Redis-based)

-   :material-code-braces:{ .lg .middle } **Cypher Support**

    ---

    Execute standard Cypher queries across all supported backends

-   :material-graph:{ .lg .middle } **Graph Algorithms**

    ---

    Built-in support for PageRank, Community Detection, and Path Finding

-   :material-flash:{ .lg .middle } **Bulk Loading**

    ---

    Optimized batch processing for high-speed data ingestion

-   :material-lock:{ .lg .middle } **Transactions**

    ---

    ACID transaction support with rollback capabilities

-   :material-chart-bell-curve:{ .lg .middle } **Analytics**

    ---

    Centrality, Similarity, and Connectivity analysis

</div>

!!! tip "When to Use"
    - **Persistent Storage**: Storing the Knowledge Graph for long-term access
    - **Complex Queries**: Running multi-hop pattern matching queries
    - **Graph Analytics**: Performing global analysis on the graph structure
    - **Production**: Scaling to billions of nodes/edges (Neo4j/FalkorDB)

---

## ⚙️ Algorithms Used

### Query Execution
- **Cypher Translation**: Adapting queries for specific backend nuances (though most support OpenCypher).
- **Query Optimization**: Index utilization and execution plan analysis.

### Graph Analytics
- **PageRank**: Measuring node importance based on incoming links.
- **Louvain Modularity**: Detecting communities by optimizing modularity.
- **Shortest Path**: Dijkstra/A* for finding optimal routes.
- **Jaccard Similarity**: Measuring node similarity based on shared neighbors.

### Bulk Operations
- **Chunking**: Splitting large datasets into optimal batch sizes (e.g., 5000 records) to prevent memory overflow.
- **Parallel Loading**: Concurrent batch insertion (backend dependent).

---

## Main Classes

### GraphStore

The main facade for graph operations.

**Methods:**

| Method | Description |
|--------|-------------|
| `execute_query(query, params)` | Run Cypher query |
| `create_node(labels, props)` | Add node |
| `create_relationship(start, end, type)` | Add edge |

**Example:**

```python
from semantica.graph_store import GraphStore

store = GraphStore(backend="neo4j")
store.execute_query(
    "MATCH (n:Person {name: $name}) RETURN n",
    params={"name": "Alice"}
)
```

### Neo4jAdapter

Enterprise-grade backend.

**Features:**
- Bolt protocol support
- Cluster awareness
- APOC procedure integration

### KuzuAdapter

Embedded, in-process backend.

**Features:**
- No external server required
- Columnar storage for speed
- Zero-copy integration with Arrow

### FalkorDBAdapter

High-performance Redis module.

**Features:**
- Sparse matrix representation
- Ultra-low latency
- Redis protocol

---

## Convenience Functions

```python
from semantica.graph_store import execute_query, create_node

# Quick query
results = execute_query("MATCH (n) RETURN count(n) as count")

# Quick node creation
create_node(["Person"], {"name": "Bob"})
```

---

## Configuration

### Environment Variables

```bash
export GRAPH_STORE_BACKEND=neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

### YAML Configuration

```yaml
graph_store:
  backend: neo4j
  
  neo4j:
    uri: bolt://localhost:7687
    pool_size: 50
    
  kuzu:
    path: ./data/kuzu_db
    buffer_pool_size: 1024 # MB
```

---

## Integration Examples

### Hybrid Search (Vector + Graph)

```python
from semantica.graph_store import GraphStore
from semantica.vector_store import VectorStore

# 1. Find relevant nodes via Vector Search
vector_store = VectorStore()
results = vector_store.search(query_vec, k=5)
node_ids = [r.metadata['node_id'] for r in results]

# 2. Expand context via Graph Traversal
graph_store = GraphStore()
query = """
MATCH (n)-[r]-(m)
WHERE elementId(n) IN $ids
RETURN n, r, m
"""
subgraph = graph_store.execute_query(query, params={"ids": node_ids})
```

---

## Best Practices

1.  **Use Parameters**: Always use parameters in Cypher queries (`$name`) instead of string concatenation to prevent injection and improve caching.
2.  **Batch Writes**: Use `create_nodes` (plural) for bulk insertion instead of loop-inserting.
3.  **Create Indexes**: Ensure you have indexes on frequently queried properties (`id`, `name`).
4.  **Close Connections**: Use context managers (`with GraphStore() as store:`) or call `close()` to release resources.

---

## See Also

- [Knowledge Graph Module](kg.md) - Logical layer above Graph Store
- [Triple Store Module](triple_store.md) - RDF-based alternative
- [Visualization Module](visualization.md) - Visualizing query results
