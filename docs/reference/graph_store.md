# Graph Store Module

> **Store and query property graphs with support for Neo4j, KuzuDB, and FalkorDB backends.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-graph:{ .lg .middle } **Property Graphs**

    ---

    Store nodes and relationships with properties using industry-standard graph databases

-   :material-database-search:{ .lg .middle } **Cypher Queries**

    ---

    Full Cypher/OpenCypher query language support for powerful graph queries

-   :material-lightning-bolt:{ .lg .middle } **Multiple Backends**

    ---

    Neo4j, KuzuDB, and FalkorDB support for different use cases

-   :material-transit-connection:{ .lg .middle } **Graph Traversal**

    ---

    Efficient path finding and neighborhood traversal algorithms

-   :material-chart-bubble:{ .lg .middle } **Graph Analytics**

    ---

    Centrality, shortest path, and community detection algorithms

-   :material-cog-transfer:{ .lg .middle } **Batch Operations**

    ---

    Efficient bulk insert with transaction support and progress tracking

</div>

!!! tip "Choosing the Right Backend"
    - **Neo4j**: Production-grade, full-featured, best for enterprise applications
    - **KuzuDB**: Embedded database, excellent for analytics, no server required
    - **FalkorDB**: Ultra-fast, Redis-based, ideal for LLM applications and real-time queries

---

## ⚙️ Algorithms Used

### Graph Storage
- **Adjacency List**: Node-relationship storage pattern
- **Property Storage**: Key-value property management on nodes/relationships
- **Index Structures**: B-tree and hash indexes for fast lookups
- **Sparse Matrices**: FalkorDB uses sparse matrix representation for adjacency

### Query Processing
- **Pattern Matching**: Cypher MATCH clause processing
- **Join Algorithms**: Hash join, merge join for multi-pattern queries
- **Index Utilization**: Automatic index selection for optimal performance
- **Query Planning**: Cost-based query optimization

### Graph Analytics
- **Shortest Path**: Dijkstra's algorithm, BFS-based shortest path
- **Centrality**: Degree centrality, betweenness centrality, PageRank
- **Community Detection**: Label propagation, Louvain algorithm
- **Traversal**: BFS/DFS graph traversal with depth control

---

## Main Classes

### GraphStore

The main interface for working with property graph databases.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `create_node(labels, properties)` | Create a node | Index insertion |
| `create_nodes(nodes)` | Batch create nodes | Bulk insertion with transactions |
| `get_node(node_id)` | Get node by ID | Index lookup |
| `get_nodes(labels, properties, limit)` | Query nodes | Pattern matching |
| `update_node(node_id, properties, merge)` | Update node | Property merge/replace |
| `delete_node(node_id, detach)` | Delete node | Cascade or isolated delete |
| `create_relationship(start, end, type, props)` | Create relationship | Edge insertion |
| `get_relationships(node_id, type, direction)` | Query relationships | Pattern matching |
| `execute_query(cypher, params)` | Execute Cypher query | Full query processing |
| `shortest_path(start, end, type, max_depth)` | Find shortest path | Dijkstra/BFS |
| `get_neighbors(node_id, depth)` | Get neighborhood | BFS traversal |

**Supported Backends:**

| Backend | Query Language | Best For | Deployment |
|---------|---------------|----------|------------|
| **Neo4j** | Cypher | Enterprise, full features | Server/Cloud |
| **KuzuDB** | Cypher | Analytics, embedded | Embedded |
| **FalkorDB** | OpenCypher | LLM apps, real-time | Redis-based |

**Example:**

```python
from semantica.graph_store import GraphStore

# Initialize with Neo4j backend
store = GraphStore(
    backend="neo4j",
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Connect to database
store.connect()

# Create nodes
alice = store.create_node(
    labels=["Person"],
    properties={"name": "Alice", "age": 30}
)

bob = store.create_node(
    labels=["Person"],
    properties={"name": "Bob", "age": 25}
)

# Create relationship
store.create_relationship(
    start_node_id=alice["id"],
    end_node_id=bob["id"],
    rel_type="KNOWS",
    properties={"since": 2020}
)

# Query the graph
results = store.execute_query(
    "MATCH (p:Person) WHERE p.age > $min_age RETURN p.name, p.age",
    parameters={"min_age": 20}
)

for record in results["records"]:
    print(f"{record['p.name']} is {record['p.age']} years old")

# Find shortest path
path = store.shortest_path(alice["id"], bob["id"])
print(f"Path length: {path['length']}")

# Close connection
store.close()
```

---

### Neo4jAdapter

Direct Neo4j database adapter for advanced operations.

**Features:**
- Full Cypher query support
- ACID transactions with rollback
- Multi-database support
- Index and constraint management
- GDS (Graph Data Science) library integration

**Example:**

```python
from semantica.graph_store import Neo4jAdapter

adapter = Neo4jAdapter(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    database="neo4j"
)

adapter.connect()

# Create index for faster queries
adapter.create_index("Person", "name", index_type="btree")

# Execute complex query
results = adapter.execute_query("""
    MATCH path = shortestPath((a:Person)-[*]-(b:Person))
    WHERE a.name = 'Alice' AND b.name = 'Bob'
    RETURN path, length(path) as distance
""")

# Get statistics
stats = adapter.get_stats()
print(f"Nodes: {stats['node_count']}, Relationships: {stats['relationship_count']}")
```

---

### KuzuAdapter

Embedded graph database for high-performance analytics.

**Features:**
- No server required (embedded)
- Schema-based node and relationship tables
- High-performance analytical queries
- COPY FROM for bulk data loading
- Persistent and in-memory modes

**Example:**

```python
from semantica.graph_store import KuzuAdapter

adapter = KuzuAdapter(
    database_path="./my_graph_db",
    buffer_pool_size=268435456,  # 256MB
    max_num_threads=4
)

adapter.connect()

# Create node table with schema
adapter.create_node_table(
    "Person",
    properties={
        "id": "SERIAL",
        "name": "STRING",
        "age": "INT64"
    },
    primary_key="id"
)

# Create relationship table
adapter.create_rel_table(
    "KNOWS",
    from_table="Person",
    to_table="Person",
    properties={"since": "INT64"}
)

# Create nodes
adapter.create_node("Person", {"name": "Alice", "age": 30})

# Bulk load from CSV
adapter.bulk_load_nodes("Person", "people.csv", header=True)
```

---

### FalkorDBAdapter

Ultra-fast Redis-based graph database for real-time applications.

**Features:**
- Sparse matrix representation
- Linear algebra query optimization
- Multi-tenant graph support
- OpenCypher query language
- Redis-based persistence

**Example:**

```python
from semantica.graph_store import FalkorDBAdapter

adapter = FalkorDBAdapter(
    host="localhost",
    port=6379,
    graph_name="knowledge_graph"
)

adapter.connect()

# Select/create a graph
adapter.select_graph("MotoGP")

# Create nodes and relationships
rider = adapter.create_node(
    labels=["Rider"],
    properties={"name": "Valentino Rossi"}
)

team = adapter.create_node(
    labels=["Team"],
    properties={"name": "Yamaha"}
)

adapter.create_relationship(
    rider["id"],
    team["id"],
    "rides",
    properties={"since": 2004}
)

# Query
results = adapter.execute_query("""
    MATCH (r:Rider)-[:rides]->(t:Team)
    WHERE t.name = 'Yamaha'
    RETURN r.name
""")

# List all graphs
graphs = adapter.list_graphs()
```

---

## Convenience Functions

Quick access to graph operations without managing store instances:

```python
from semantica.graph_store import (
    # Node operations
    create_node,
    create_nodes,
    get_nodes,
    update_node,
    delete_node,
    
    # Relationship operations
    create_relationship,
    create_relationships,
    get_relationships,
    delete_relationship,
    
    # Query operations
    execute_query,
    
    # Analytics operations
    shortest_path,
    get_neighbors,
    run_analytics
)

# Create nodes using convenience functions
alice = create_node(labels=["Person"], properties={"name": "Alice"})
bob = create_node(labels=["Person"], properties={"name": "Bob"})

# Create relationship
rel = create_relationship(
    start_id=alice["id"],
    end_id=bob["id"],
    rel_type="KNOWS"
)

# Query
results = execute_query("MATCH (n:Person) RETURN n.name LIMIT 10")

# Analytics
path = shortest_path(alice["id"], bob["id"], max_depth=5)
neighbors = get_neighbors(alice["id"], depth=2)
```

---

## Configuration

### Environment Variables

```bash
# General settings
export GRAPH_STORE_DEFAULT_BACKEND=neo4j
export GRAPH_STORE_BATCH_SIZE=1000
export GRAPH_STORE_TIMEOUT=30

# Neo4j settings
export GRAPH_STORE_NEO4J_URI=bolt://localhost:7687
export GRAPH_STORE_NEO4J_USER=neo4j
export GRAPH_STORE_NEO4J_PASSWORD=password
export GRAPH_STORE_NEO4J_DATABASE=neo4j

# KuzuDB settings
export GRAPH_STORE_KUZU_DATABASE_PATH=./kuzu_db
export GRAPH_STORE_KUZU_BUFFER_POOL_SIZE=268435456

# FalkorDB settings
export GRAPH_STORE_FALKORDB_HOST=localhost
export GRAPH_STORE_FALKORDB_PORT=6379
export GRAPH_STORE_FALKORDB_GRAPH_NAME=default
```

### YAML Configuration

```yaml
# config.yaml - Graph Store Configuration

graph_store:
  backend: neo4j  # neo4j, kuzu, falkordb
  batch_size: 1000
  timeout: 30
  
  neo4j:
    uri: bolt://localhost:7687
    user: neo4j
    password: password
    database: neo4j
    encrypted: false
    
  kuzu:
    database_path: ./kuzu_db
    buffer_pool_size: 268435456
    max_num_threads: 4
    
  falkordb:
    host: localhost
    port: 6379
    password: null
    graph_name: default
```

---

## Backend Comparison

| Feature | Neo4j | KuzuDB | FalkorDB |
|---------|-------|--------|----------|
| **Query Language** | Cypher | Cypher | OpenCypher |
| **Deployment** | Server/Cloud | Embedded | Server (Redis) |
| **Schema** | Schema-optional | Schema-required | Schema-optional |
| **Transactions** | Full ACID | ACID | ACID |
| **Performance** | Excellent | Best for analytics | Ultra-fast |
| **Use Case** | Enterprise | Analytics/Embedded | Real-time/LLM |
| **Clustering** | Yes | No | Via Redis |
| **Graph Algorithms** | GDS Library | Built-in | Built-in |

---

## Docker Quick Start

### FalkorDB

```bash
docker run -p 6379:6379 -p 3000:3000 -it --rm \
    -v ./data:/var/lib/falkordb/data \
    falkordb/falkordb
```

Then open http://localhost:3000 for the web UI.

### Neo4j

```bash
docker run -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

Then open http://localhost:7474 for Neo4j Browser.

---

## Performance Tips

### Indexing

```python
# Create indexes for frequently queried properties
store.create_index("Person", "name", index_type="btree")
store.create_index("Document", "content", index_type="fulltext")
```

### Batch Operations

```python
# Use batch operations for bulk inserts
nodes = [
    {"labels": ["Person"], "properties": {"name": f"Person_{i}"}}
    for i in range(1000)
]
store.create_nodes(nodes)
```

### Query Optimization

```python
# Use parameterized queries
results = store.execute_query(
    "MATCH (p:Person) WHERE p.age > $min_age RETURN p",
    parameters={"min_age": 25}
)

# Limit results
results = store.get_nodes(labels=["Person"], limit=100)
```

---

## Integration Examples

### Knowledge Graph for RAG

```python
from semantica.graph_store import GraphStore
from semantica.embeddings import EmbeddingGenerator

# Create knowledge graph
store = GraphStore(backend="falkordb")
store.connect()

# Add entities and relationships
doc = store.create_node(["Document"], {"title": "AI Paper", "content": "..."})
concept = store.create_node(["Concept"], {"name": "Machine Learning"})
store.create_relationship(doc["id"], concept["id"], "MENTIONS")

# Query for retrieval
results = store.execute_query("""
    MATCH (d:Document)-[:MENTIONS]->(c:Concept)
    WHERE c.name CONTAINS 'Learning'
    RETURN d.title, d.content
""")
```

### Social Network Analysis

```python
from semantica.graph_store import GraphStore

store = GraphStore(backend="neo4j")
store.connect()

# Find influential users (high degree centrality)
results = store.execute_query("""
    MATCH (u:User)-[r:FOLLOWS]-()
    WITH u, count(r) as connections
    ORDER BY connections DESC
    RETURN u.name, connections
    LIMIT 10
""")

# Find communities
from semantica.graph_store import run_analytics
components = run_analytics("connected_components", labels=["User"])
```

---

## See Also

- [Vector Store Module](vector_store.md) - Store and search vector embeddings
- [Triple Store Module](triple_store.md) - RDF triple storage and SPARQL queries
- [Knowledge Graph Module](kg.md) - Build and analyze knowledge graphs
- [Visualization Module](visualization.md) - Visualize graph structures

