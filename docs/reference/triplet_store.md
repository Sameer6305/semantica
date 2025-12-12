# Triplet Store

> **Store and query RDF triplets with SPARQL support and semantic reasoning using industry-standard triplet stores.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } **RDF Storage**

    ---

    Store subject-predicate-object triplets in W3C-compliant RDF format

-   :material-code-braces:{ .lg .middle } **SPARQL Queries**

    ---

    Full W3C SPARQL 1.1 query language support for powerful semantic queries

-   :material-brain:{ .lg .middle } **Reasoning**

    ---

    RDFS and OWL reasoning for inference and knowledge discovery

-   :material-database-sync:{ .lg .middle } **Multiple Backends**

    ---

    Blazegraph, Apache Jena, RDF4J, and Virtuoso support

-   :material-link-variant:{ .lg .middle } **Federation**

    ---

    Query across multiple triplet stores with SPARQL federation

-   :material-upload-multiple:{ .lg .middle } **Bulk Loading**

    ---

    High-performance bulk data loading with progress tracking

</div>

!!! tip "Choosing the Right Backend"
    - **Blazegraph**: High-performance, excellent for large datasets, GPU acceleration
    - **Apache Jena**: Full-featured, TDB2 storage, SHACL validation
    - **RDF4J**: Java-based, excellent tooling, multiple storage backends
    - **Virtuoso**: Enterprise-grade, excellent performance, SQL integration

---

## ⚙️ Algorithms Used

### Query Algorithms
- **SPARQL Query Optimization**: Join reordering with selectivity estimation
- **Triple Pattern Matching**: Index-based lookup with B+ trees
- **Graph Pattern Matching**: Subgraph isomorphism with backtracking
- **Query Planning**: Cost-based optimization with statistics
- **Join Algorithms**: Hash join, merge join, nested loop join
- **Filter Pushdown**: Early filter application for performance

### Indexing
- **SPO Index**: Subject-Predicate-Object index for subject lookups
- **POS Index**: Predicate-Object-Subject index for predicate lookups
- **OSP Index**: Object-Subject-Predicate index for object lookups
- **Six-Index Scheme**: All permutations (SPO, SOP, PSO, POS, OSP, OPS) for optimal query performance
- **B+ Tree Indexing**: Efficient range queries and sorted access
- **Hash Indexing**: O(1) exact match lookups

### Reasoning Algorithms
- **RDFS Reasoning**: Subclass/subproperty inference, domain/range inference
- **OWL Reasoning**: Class hierarchy, property characteristics, cardinality constraints
- **Forward Chaining**: Materialization of inferred triples
- **Backward Chaining**: On-demand inference during query execution
- **Rule-Based Inference**: Custom SWRL rules

### Bulk Loading
- **Batch Processing**: Chunked triple insertion with configurable batch size
- **Parallel Loading**: Multi-threaded data loading
- **Index Building**: Deferred index construction for faster loading
- **Transaction Management**: Atomic batch commits with rollback support

---

## Main Classes

### TripletManager

Main coordinator for triplet store operations across multiple backends.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `register_store(store_id, backend, endpoint)` | Register triplet store | Store registration |
| `add_triple(triple, store_id)` | Add single triplet | Index insertion |
| `add_triples(triples, store_id)` | Batch add triplets | Bulk index insertion |
| `query(sparql, store_id)` | Execute SPARQL query | Query optimization + execution |
| `delete(pattern, store_id)` | Delete matching triplets | Pattern matching + deletion |
| `bulk_load(file_path, format, store_id)` | Bulk load from file | Streaming parser + batch insert |
| `get_stats(store_id)` | Get store statistics | Statistics collection |

**Example:**

```python
from semantica.triplet_store import TripletManager

# Initialize manager
manager = TripletManager()

# Register Blazegraph store
store = manager.register_store(
    store_id="main",
    backend="blazegraph",
    endpoint="http://localhost:9999/blazegraph/sparql"
)

# Add single triple
result = manager.add_triple(
    triple={
        "subject": "http://example.org/Alice",
        "predicate": "http://example.org/knows",
        "object": "http://example.org/Bob"
    },
    store_id="main"
)

# Add multiple triples
triples = [
    {
        "subject": "http://example.org/Alice",
        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "object": "http://example.org/Person"
    },
    {
        "subject": "http://example.org/Bob",
        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "object": "http://example.org/Person"
    }
]

manager.add_triples(triples, store_id="main")

# Query with SPARQL
results = manager.query("""
    PREFIX ex: <http://example.org/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    
    SELECT ?person ?friend WHERE {
        ?person rdf:type ex:Person .
        ?person ex:knows ?friend .
    }
""", store_id="main")

for row in results["results"]["bindings"]:
    print(f"{row['person']['value']} knows {row['friend']['value']}")

# Get statistics
stats = manager.get_stats("main")
print(f"Total triples: {stats['triple_count']}")
```

---

### QueryEngine

SPARQL query execution and optimization engine.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `execute(query, store)` | Execute SPARQL query | Parse + optimize + execute |
| `parse_query(sparql)` | Parse SPARQL syntax | SPARQL parser |
| `optimize_query(query)` | Optimize query plan | Join reordering + filter pushdown |
| `explain_query(query)` | Explain query plan | Query plan visualization |
| `validate_query(sparql)` | Validate SPARQL syntax | Syntax validation |

**SPARQL Query Types:**

| Query Type | Description | Use Case |
|------------|-------------|----------|
| **SELECT** | Retrieve variable bindings | Data retrieval |
| **CONSTRUCT** | Build RDF graph | Graph transformation |
| **ASK** | Boolean query | Existence check |
| **DESCRIBE** | Describe resources | Resource exploration |

**Example:**

```python
from semantica.triplet_store import QueryEngine, TripletManager

manager = TripletManager()
store = manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph/sparql")

engine = QueryEngine()

# SELECT query
select_query = """
    PREFIX ex: <http://example.org/>
    SELECT ?name ?age WHERE {
        ?person ex:name ?name .
        ?person ex:age ?age .
        FILTER (?age > 18)
    }
    ORDER BY DESC(?age)
    LIMIT 10
"""

results = engine.execute(select_query, store)

# CONSTRUCT query
construct_query = """
    PREFIX ex: <http://example.org/>
    CONSTRUCT {
        ?person ex:isAdult true .
    }
    WHERE {
        ?person ex:age ?age .
        FILTER (?age >= 18)
    }
"""

graph = engine.execute(construct_query, store)

# ASK query
ask_query = """
    PREFIX ex: <http://example.org/>
    ASK {
        ?person ex:name "Alice" .
    }
"""

exists = engine.execute(ask_query, store)
print(f"Alice exists: {exists}")

# Explain query plan
plan = engine.explain_query(select_query)
print(f"Query plan: {plan}")
```

---

### BulkLoader

High-performance bulk data loading with progress tracking.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `load(file_path, format, store)` | Load RDF file | Streaming parser + batch insert |
| `load_from_url(url, format, store)` | Load from URL | HTTP streaming + batch insert |
| `load_from_string(data, format, store)` | Load from string | String parser + batch insert |
| `get_progress()` | Get loading progress | Progress tracking |

**Supported Formats:**
- **RDF/XML**: W3C RDF/XML format
- **Turtle**: Terse RDF Triple Language
- **N-Triples**: Line-based triple format
- **N-Quads**: N-Triples with named graphs
- **JSON-LD**: JSON-based RDF format
- **TriG**: Turtle with named graphs

**Example:**

```python
from semantica.triplet_store import BulkLoader, TripletManager

manager = TripletManager()
store = manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph/sparql")

loader = BulkLoader(
    batch_size=10000,
    show_progress=True,
    parallel=True,
    n_jobs=4
)

# Load from file
progress = loader.load(
    file_path="knowledge_graph.ttl",
    format="turtle",
    store=store
)

print(f"Loaded {progress['triples_loaded']} triples in {progress['elapsed_time']:.2f}s")
print(f"Throughput: {progress['triples_per_second']:.0f} triples/sec")

# Load from URL
progress = loader.load_from_url(
    url="https://example.org/data.rdf",
    format="rdf/xml",
    store=store
)

# Load from string
rdf_data = """
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

ex:Alice rdf:type ex:Person ;
         ex:name "Alice" ;
         ex:age 30 .
"""

progress = loader.load_from_string(
    data=rdf_data,
    format="turtle",
    store=store
)
```

---

### Backend Adapters

#### BlazegraphAdapter

High-performance triplet store with GPU acceleration support.

**Features:**
- High-performance SPARQL query execution
- GPU acceleration for analytics
- Full-text search integration
- Geospatial query support
- High availability clustering

**Example:**

```python
from semantica.triplet_store import BlazegraphAdapter

adapter = BlazegraphAdapter(
    endpoint="http://localhost:9999/blazegraph/sparql",
    namespace="kb",  # Blazegraph namespace
    timeout=30
)

adapter.connect()

# Create namespace
adapter.create_namespace("my_kb", properties={
    "com.bigdata.rdf.store.AbstractTripleStore.textIndex": "true",
    "com.bigdata.rdf.store.AbstractTripleStore.geoSpatial": "true"
})

# Add triples
adapter.add_triple(
    subject="http://example.org/Alice",
    predicate="http://example.org/name",
    object_literal="Alice",
    object_datatype="http://www.w3.org/2001/XMLSchema#string"
)

# Full-text search
results = adapter.query("""
    PREFIX bds: <http://www.bigdata.com/rdf/search#>
    SELECT ?subject ?score WHERE {
        ?subject bds:search "machine learning" .
        ?subject bds:relevance ?score .
    }
    ORDER BY DESC(?score)
""")
```

---

#### JenaAdapter

Full-featured RDF framework with TDB2 storage.

**Features:**
- TDB2 native triplet store
- SHACL validation
- Inference engines (RDFS, OWL)
- Fuseki SPARQL server
- RDF/XML, Turtle, JSON-LD support

**Example:**

```python
from semantica.triplet_store import JenaAdapter

adapter = JenaAdapter(
    tdb_directory="./tdb2_data",
    inference="rdfs"  # rdfs, owl, or None
)

adapter.connect()

# Add triples with inference
adapter.add_triple(
    subject="http://example.org/Dog",
    predicate="http://www.w3.org/2000/01/rdf-schema#subClassOf",
    object="http://example.org/Animal"
)

adapter.add_triple(
    subject="http://example.org/Fido",
    predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    object="http://example.org/Dog"
)

# Query with inference (Fido is inferred to be an Animal)
results = adapter.query("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX ex: <http://example.org/>
    
    SELECT ?animal WHERE {
        ?animal rdf:type ex:Animal .
    }
""")

# SHACL validation
shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/> .

ex:PersonShape a sh:NodeShape ;
    sh:targetClass ex:Person ;
    sh:property [
        sh:path ex:name ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
    ] .
"""

validation_report = adapter.validate_shacl(shapes)
print(f"Valid: {validation_report['conforms']}")
```

---

#### RDF4JAdapter

Java-based RDF framework with multiple storage backends.

**Features:**
- Multiple storage backends (Memory, Native, HTTP)
- Transaction support with ACID guarantees
- SPARQL 1.1 Update support
- RDF Schema and OWL reasoning
- Repository federation

**Example:**

```python
from semantica.triplet_store import RDF4JAdapter

adapter = RDF4JAdapter(
    server_url="http://localhost:8080/rdf4j-server",
    repository_id="my_repo"
)

adapter.connect()

# Add triples with transaction
adapter.begin_transaction()
try:
    adapter.add_triple(
        subject="http://example.org/Alice",
        predicate="http://example.org/name",
        object_literal="Alice"
    )
    adapter.commit_transaction()
except Exception as e:
    adapter.rollback_transaction()

# Query with reasoning
results = adapter.query("""
    PREFIX ex: <http://example.org/>
    SELECT ?person WHERE {
        ?person ex:name ?name .
    }
""", enable_reasoning=True)
```

---

#### VirtuosoAdapter

Enterprise-grade RDF store with SQL integration.

**Features:**
- High-performance SPARQL execution
- SQL/SPARQL hybrid queries
- Quad store with named graphs
- Full-text indexing
- Geospatial support

**Example:**

```python
from semantica.triplet_store import VirtuosoAdapter

adapter = VirtuosoAdapter(
    host="localhost",
    port=1111,
    user="dba",
    password="dba"
)

adapter.connect()

# Add triples to named graph
graph_uri = "http://example.org/graph1"
adapter.create_graph(graph_uri)

adapter.add_triple(
    subject="http://example.org/Alice",
    predicate="http://example.org/name",
    object_literal="Alice",
    graph=graph_uri
)

# Query specific graph
results = adapter.query(f"""
    PREFIX ex: <http://example.org/>
    SELECT ?person ?name
    FROM <{graph_uri}>
    WHERE {{
        ?person ex:name ?name .
    }}
""")
```

---

## Convenience Functions

Quick access to triplet store operations:

```python
from semantica.triplet_store import (
    add_triple,
    add_triples,
    execute_query,
    bulk_load,
    export_graph,
    import_graph
)

# Add single triple
add_triple(
    subject="http://example.org/Alice",
    predicate="http://example.org/knows",
    object="http://example.org/Bob"
)

# Execute SPARQL query
results = execute_query("""
    SELECT ?s ?p ?o WHERE {
        ?s ?p ?o .
    }
    LIMIT 10
""")

# Bulk load
progress = bulk_load(
    file_path="data.ttl",
    format="turtle",
    batch_size=10000
)

# Export graph
export_graph(
    output_path="export.rdf",
    format="rdf/xml"
)
```

---

## Dataclasses

### TripletStore

Configuration dataclass for triplet store instances.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `store_id` | str | Unique store identifier |
| `store_type` | str | Backend type (blazegraph, jena, rdf4j, virtuoso) |
| `endpoint` | str | SPARQL endpoint URL |
| `config` | dict | Additional configuration options |

---

### QueryResult

Query execution result dataclass.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `variables` | List[str] | Query variable names |
| `bindings` | List[Dict] | Result bindings |
| `execution_time` | float | Query execution time (seconds) |
| `metadata` | Dict | Additional metadata (cached, optimized, etc.) |

---

### QueryPlan

Query execution plan dataclass.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `query` | str | Original SPARQL query |
| `optimized_query` | str | Optimized query |
| `estimated_cost` | float | Estimated execution cost |
| `execution_steps` | List[str] | Planned execution steps |

---

### LoadProgress

Bulk loading progress dataclass.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `loaded_triples` | int | Number of triples loaded |
| `total_triples` | int | Total triples to load |
| `failed_triples` | int | Number of failed triples |
| `progress_percentage` | float | Loading progress (0-100) |
| `elapsed_time` | float | Time elapsed (seconds) |
| `current_batch` | int | Current batch number |
| `total_batches` | int | Total number of batches |
| `metadata` | Dict | Additional metadata (throughput, ETA, etc.) |

---

## Configuration

### Environment Variables

```bash
# General settings
export TRIPLET_STORE_DEFAULT_BACKEND=blazegraph
export TRIPLET_STORE_BATCH_SIZE=10000
export TRIPLET_STORE_TIMEOUT=30

# Blazegraph settings
export TRIPLET_STORE_BLAZEGRAPH_ENDPOINT=http://localhost:9999/blazegraph/sparql
export TRIPLET_STORE_BLAZEGRAPH_NAMESPACE=kb

# Jena settings
export TRIPLET_STORE_JENA_TDB_DIRECTORY=./tdb2_data
export TRIPLET_STORE_JENA_INFERENCE=rdfs

# RDF4J settings
export TRIPLET_STORE_RDF4J_SERVER_URL=http://localhost:8080/rdf4j-server
export TRIPLET_STORE_RDF4J_REPOSITORY_ID=my_repo

# Virtuoso settings
export TRIPLET_STORE_VIRTUOSO_HOST=localhost
export TRIPLET_STORE_VIRTUOSO_PORT=1111
export TRIPLET_STORE_VIRTUOSO_USER=dba
export TRIPLET_STORE_VIRTUOSO_PASSWORD=dba
```

### YAML Configuration

```yaml
# config.yaml - Triplet Store Configuration

triplet_store:
  backend: blazegraph  # blazegraph, jena, rdf4j, virtuoso
  batch_size: 10000
  timeout: 30
  enable_reasoning: true
  
  blazegraph:
    endpoint: http://localhost:9999/blazegraph/sparql
    namespace: kb
    properties:
      textIndex: true
      geoSpatial: false
      
  jena:
    tdb_directory: ./tdb2_data
    inference: rdfs  # rdfs, owl, none
    unionDefaultGraph: true
    
  rdf4j:
    server_url: http://localhost:8080/rdf4j-server
    repository_id: my_repo
    
  virtuoso:
    host: localhost
    port: 1111
    user: dba
    password: dba
    graph_uri: http://example.org/graph
    
  query:
    optimize: true
    cache_enabled: true
    cache_size: 1000
    explain_plans: false
```

---

## Backend Comparison

| Feature | Blazegraph | Apache Jena | RDF4J | Virtuoso |
|---------|------------|-------------|-------|----------|
| **Performance** | Excellent | Good | Good | Excellent |
| **Scalability** | High | Medium | Medium | Very High |
| **SPARQL 1.1** | Full | Full | Full | Full |
| **Reasoning** | Limited | Full (RDFS/OWL) | Full | Full |
| **Full-Text** | Yes | Yes | Yes | Yes |
| **Geospatial** | Yes | Limited | Limited | Yes |
| **Clustering** | Yes | No | No | Yes |
| **License** | GPLv2 | Apache 2.0 | BSD | Commercial/GPL |
| **Best For** | Large datasets | Java apps | Java apps | Enterprise |

---

## Docker Quick Start

### Blazegraph

```bash
docker run -d -p 9999:9999 \
    -v ./blazegraph-data:/data \
    --name blazegraph \
    lyrasis/blazegraph:2.1.5

# Access UI at http://localhost:9999/blazegraph
```

### Apache Jena Fuseki

```bash
docker run -d -p 3030:3030 \
    -v ./fuseki-data:/fuseki \
    --name fuseki \
    stain/jena-fuseki

# Access UI at http://localhost:3030
```

### RDF4J Server

```bash
docker run -d -p 8080:8080 \
    -v ./rdf4j-data:/var/rdf4j \
    --name rdf4j \
    eclipse/rdf4j-workbench

# Access UI at http://localhost:8080/rdf4j-workbench
```

---

## Advanced SPARQL Examples

### Aggregation Queries

```sparql
PREFIX ex: <http://example.org/>

SELECT ?department (COUNT(?employee) as ?count) (AVG(?salary) as ?avg_salary)
WHERE {
    ?employee ex:worksIn ?department .
    ?employee ex:salary ?salary .
}
GROUP BY ?department
HAVING (COUNT(?employee) > 5)
ORDER BY DESC(?count)
```

### Property Paths

```sparql
PREFIX ex: <http://example.org/>

# Find all ancestors (transitive closure)
SELECT ?person ?ancestor
WHERE {
    ?person ex:hasParent+ ?ancestor .
}

# Find friends of friends
SELECT ?person ?friend_of_friend
WHERE {
    ?person ex:knows/ex:knows ?friend_of_friend .
    FILTER (?person != ?friend_of_friend)
}
```

### Federated Queries

```sparql
PREFIX ex: <http://example.org/>

SELECT ?person ?company ?stock_price
WHERE {
    # Local store
    ?person ex:worksFor ?company .
    
    # Remote store
    SERVICE <http://remote-store.example.org/sparql> {
        ?company ex:stockPrice ?stock_price .
    }
}
```

### Update Operations

```sparql
PREFIX ex: <http://example.org/>

# INSERT DATA
INSERT DATA {
    ex:Alice ex:knows ex:Bob .
    ex:Bob ex:knows ex:Charlie .
}

# DELETE/INSERT
DELETE {
    ?person ex:age ?old_age .
}
INSERT {
    ?person ex:age ?new_age .
}
WHERE {
    ?person ex:age ?old_age .
    BIND(?old_age + 1 AS ?new_age)
}
```

---

## Performance Tips

### Query Optimization

```python
# Use LIMIT for large result sets
query = """
    SELECT ?s ?p ?o WHERE {
        ?s ?p ?o .
    }
    LIMIT 1000
"""

# Use FILTER efficiently (after other patterns)
query = """
    SELECT ?person ?name WHERE {
        ?person rdf:type ex:Person .
        ?person ex:name ?name .
        FILTER (STRLEN(?name) > 5)  # Filter after pattern matching
    }
"""

# Use property paths wisely
query = """
    SELECT ?person ?ancestor WHERE {
        ?person ex:hasParent{1,3} ?ancestor .  # Limit path length
    }
"""
```

### Bulk Loading Optimization

```python
from semantica.triple_store import BulkLoader

loader = BulkLoader(
    batch_size=50000,      # Larger batches for better performance
    parallel=True,         # Enable parallel loading
    n_jobs=8,             # Use multiple cores
    disable_indexes=True,  # Disable indexes during load
    rebuild_indexes=True   # Rebuild after load
)

progress = loader.load("large_dataset.nt", format="ntriples")
```

### Indexing Strategy

```python
# Create selective indexes
adapter.create_index("predicate", ["http://example.org/name"])
adapter.create_index("object", ["http://example.org/Person"])

# Full-text index for specific predicates
adapter.create_fulltext_index([
    "http://example.org/description",
    "http://example.org/content"
])
```

---

## Integration Examples

### Knowledge Graph Export to RDF

```python
from semantica.kg import GraphBuilder
from semantica.triple_store import TripleManager, BulkLoader

# Build knowledge graph
builder = GraphBuilder()
kg = builder.build(entities, relationships)

# Convert to RDF triples
triples = []
for entity in kg.entities:
    triples.append({
        "subject": f"http://example.org/{entity.id}",
        "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "object": f"http://example.org/{entity.type}"
    })
    
    for prop, value in entity.properties.items():
        triples.append({
            "subject": f"http://example.org/{entity.id}",
            "predicate": f"http://example.org/{prop}",
            "object_literal": str(value)
        })

# Load into triple store
manager = TripleManager()
store = manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph/sparql")
manager.add_triples(triples, store_id="main")
```

### Semantic Search with SPARQL

```python
from semantica.triple_store import QueryEngine

engine = QueryEngine()

def semantic_search(keywords, limit=10):
    query = f"""
        PREFIX ex: <http://example.org/>
        PREFIX bds: <http://www.bigdata.com/rdf/search#>
        
        SELECT ?doc ?title ?score WHERE {{
            ?doc bds:search "{keywords}" .
            ?doc bds:relevance ?score .
            ?doc ex:title ?title .
        }}
        ORDER BY DESC(?score)
        LIMIT {limit}
    """
    
    return engine.execute(query, store)

results = semantic_search("machine learning", limit=5)
```

---

## Troubleshooting

### Common Issues

**Issue**: Slow query performance

```python
# Solution 1: Add indexes
adapter.create_index("predicate", ["http://example.org/knows"])

# Solution 2: Optimize query
# Bad: Cartesian product
query_bad = """
    SELECT ?s ?o WHERE {
        ?s ?p1 ?o1 .
        ?o ?p2 ?o2 .
    }
"""

# Good: Constrained join
query_good = """
    SELECT ?s ?o WHERE {
        ?s ex:knows ?o .
        ?o rdf:type ex:Person .
    }
"""

# Solution 3: Use LIMIT
results = execute_query(query + " LIMIT 1000")
```

**Issue**: Out of memory during bulk load

```python
# Solution: Use streaming and smaller batches
loader = BulkLoader(
    batch_size=10000,      # Smaller batches
    streaming=True,        # Stream from file
    clear_cache=True       # Clear cache between batches
)
```

**Issue**: SPARQL syntax errors

```python
# Solution: Validate query first
from semantica.triple_store import QueryEngine

engine = QueryEngine()
is_valid = engine.validate_query(sparql_query)

if not is_valid:
    errors = engine.get_syntax_errors(sparql_query)
    print(f"Syntax errors: {errors}")
```

---

## See Also
- [Knowledge Graph Module](kg.md) - Build and analyze knowledge graphs
- [Ontology Module](ontology.md) - Ontology generation and management
- [Graph Store Module](graph_store.md) - Property graph storage
- [Export Module](export.md) - Export to various formats

## Cookbook
- [Triple Store](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/20_Triple_Store.ipynb)
