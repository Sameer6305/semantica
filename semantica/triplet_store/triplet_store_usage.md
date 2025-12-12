# Triplet Store Module Usage Guide

This comprehensive guide demonstrates how to use the triplet store module for RDF data storage and querying, supporting multiple triplet store backends (Blazegraph, Jena, RDF4J, Virtuoso) with unified interfaces, SPARQL query execution, bulk loading, and query optimization.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Store Registration](#store-registration)
3. [CRUD Operations](#crud-operations)
4. [SPARQL Query Execution](#sparql-query-execution)
5. [Query Optimization](#query-optimization)
6. [Bulk Loading](#bulk-loading)
7. [Store Adapters](#store-adapters)
8. [Algorithms and Methods](#algorithms-and-methods)
9. [Configuration](#configuration)
10. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using TripletManager

```python
from semantica.triplet_store import TripletManager
from semantica.semantic_extract.triple_extractor import Triple

# Create triplet manager
manager = TripletManager()

# Register a store
store = manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")

# Add a triple
triple = Triple(
    subject="http://example.org/entity1",
    predicate="http://example.org/hasName",
    object="John Doe",
    confidence=0.9
)
result = manager.add_triple(triple, store_id="main")

print(f"Triple added: {result['success']}")
```

### Using Convenience Functions

```python
from semantica.triplet_store import register_store, add_triple, get_triples, execute_query
from semantica.semantic_extract.triple_extractor import Triple

# Register store
store = register_store("main", "blazegraph", "http://localhost:9999/blazegraph")

# Add triple
triple = Triple(
    subject="http://example.org/entity1",
    predicate="http://example.org/hasName",
    object="John Doe"
)
result = add_triple(triple, store_id="main")

# Get triples
triples = get_triples(subject="http://example.org/entity1", store_id="main")
print(f"Found {len(triples)} triples")
```

### Using QueryEngine

```python
from semantica.triplet_store import QueryEngine, BlazegraphAdapter

# Create query engine
engine = QueryEngine(enable_caching=True, enable_optimization=True)

# Create adapter
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

# Execute query
query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
result = engine.execute_query(query, adapter)

print(f"Found {len(result.bindings)} results")
print(f"Execution time: {result.execution_time:.2f}s")
```

## Store Registration

### Registering a Store

```python
from semantica.triplet_store import TripletManager

manager = TripletManager()

# Register Blazegraph store
blazegraph_store = manager.register_store(
    store_id="blazegraph",
    store_type="blazegraph",
    endpoint="http://localhost:9999/blazegraph",
    namespace="kb"
)

# Register Jena store
jena_store = manager.register_store(
    store_id="jena",
    store_type="jena",
    endpoint="http://localhost:3030/ds",
    dataset="default"
)

# Register RDF4J store
rdf4j_store = manager.register_store(
    store_id="rdf4j",
    store_type="rdf4j",
    endpoint="http://localhost:8080/rdf4j-server",
    repository="test"
)

# Register Virtuoso store
virtuoso_store = manager.register_store(
    store_id="virtuoso",
    store_type="virtuoso",
    endpoint="http://localhost:8890/sparql"
)
```

### Using Convenience Function

```python
from semantica.triplet_store import register_store

# Register store using convenience function
store = register_store(
    "main",
    "blazegraph",
    "http://localhost:9999/blazegraph",
    method="default"
)

print(f"Store registered: {store.store_id}")
print(f"Store type: {store.store_type}")
```

### Multiple Stores

```python
from semantica.triplet_store import TripletManager

manager = TripletManager()

# Register multiple stores
manager.register_store("primary", "blazegraph", "http://localhost:9999/blazegraph")
manager.register_store("secondary", "jena", "http://localhost:3030/ds")

# List all stores
store_ids = manager.list_stores()
print(f"Registered stores: {store_ids}")

# Get specific store
store = manager.get_store("primary")
print(f"Store endpoint: {store.endpoint}")
```

## CRUD Operations

### Adding Triples

```python
from semantica.triplet_store import TripletManager
from semantica.semantic_extract.triple_extractor import Triple

manager = TripletManager()
manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")

# Add single triple
triple = Triple(
    subject="http://example.org/entity1",
    predicate="http://example.org/hasName",
    object="John Doe",
    confidence=0.9
)
result = manager.add_triple(triple, store_id="main")
print(f"Added: {result['success']}")

# Add multiple triples
triples = [
    Triple("http://example.org/entity1", "http://example.org/hasAge", "30"),
    Triple("http://example.org/entity1", "http://example.org/hasCity", "New York"),
    Triple("http://example.org/entity2", "http://example.org/hasName", "Jane Smith")
]
result = manager.add_triples(triples, store_id="main", batch_size=1000)
print(f"Added {result['total_triples']} triples in {result['batches']} batches")
```

### Retrieving Triples

```python
from semantica.triplet_store import TripletManager

manager = TripletManager()
manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")

# Get all triples for a subject
triples = manager.get_triple(
    subject="http://example.org/entity1",
    store_id="main"
)
print(f"Found {len(triples)} triples for entity1")

# Get triples matching predicate
triples = manager.get_triple(
    predicate="http://example.org/hasName",
    store_id="main"
)
print(f"Found {len(triples)} triples with hasName predicate")

# Get specific triple
triples = manager.get_triple(
    subject="http://example.org/entity1",
    predicate="http://example.org/hasName",
    object="John Doe",
    store_id="main"
)
```

### Deleting Triples

```python
from semantica.triplet_store import TripletManager
from semantica.semantic_extract.triple_extractor import Triple

manager = TripletManager()
manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")

# Delete triple
triple = Triple(
    subject="http://example.org/entity1",
    predicate="http://example.org/hasName",
    object="John Doe"
)
result = manager.delete_triple(triple, store_id="main")
print(f"Deleted: {result['success']}")
```

### Updating Triples

```python
from semantica.triplet_store import TripletManager
from semantica.semantic_extract.triple_extractor import Triple

manager = TripletManager()
manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")

# Update triple (delete old, add new)
old_triple = Triple(
    subject="http://example.org/entity1",
    predicate="http://example.org/hasAge",
    object="30"
)
new_triple = Triple(
    subject="http://example.org/entity1",
    predicate="http://example.org/hasAge",
    object="31"
)
result = manager.update_triple(old_triple, new_triple, store_id="main")
print(f"Updated: {result['success']}")
```

## SPARQL Query Execution

### Basic Query Execution

```python
from semantica.triplet_store import QueryEngine, BlazegraphAdapter

# Create query engine
engine = QueryEngine(enable_caching=True)

# Create adapter
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

# Execute SELECT query
query = """
SELECT ?s ?p ?o
WHERE {
    ?s ?p ?o .
    ?s <http://example.org/hasName> ?o .
}
LIMIT 10
"""
result = engine.execute_query(query, adapter)

print(f"Variables: {result.variables}")
print(f"Results: {len(result.bindings)}")
for binding in result.bindings[:5]:
    print(binding)
```

### Using Convenience Function

```python
from semantica.triplet_store import execute_query, BlazegraphAdapter

adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
result = execute_query(query, adapter, method="default")

print(f"Found {len(result.bindings)} results")
```

### Query Result Processing

```python
from semantica.triplet_store import QueryEngine, BlazegraphAdapter

engine = QueryEngine()
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

query = "SELECT ?name ?age WHERE { ?s <http://example.org/hasName> ?name . ?s <http://example.org/hasAge> ?age }"
result = engine.execute_query(query, adapter)

# Process results
for binding in result.bindings:
    name = binding.get("name", {}).get("value", "")
    age = binding.get("age", {}).get("value", "")
    print(f"Name: {name}, Age: {age}")

print(f"Execution time: {result.execution_time:.2f}s")
print(f"Metadata: {result.metadata}")
```

### Query Caching

```python
from semantica.triplet_store import QueryEngine, BlazegraphAdapter

# Enable caching
engine = QueryEngine(enable_caching=True, cache_size=1000)
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"

# First execution (not cached)
result1 = engine.execute_query(query, adapter)
print(f"First execution: {result1.execution_time:.2f}s, Cached: {result1.metadata.get('cached', False)}")

# Second execution (cached)
result2 = engine.execute_query(query, adapter)
print(f"Second execution: {result2.execution_time:.2f}s, Cached: {result2.metadata.get('cached', False)}")

# Clear cache
engine.clear_cache()
```

## Query Optimization

### Basic Query Optimization

```python
from semantica.triplet_store import QueryEngine

engine = QueryEngine(enable_optimization=True)

# Original query
query = """
SELECT ?s ?p ?o
WHERE {
    ?s ?p ?o .
}
"""

# Optimize query
optimized = engine.optimize_query(query, add_limit=True, default_limit=1000)
print(f"Optimized query:\n{optimized}")
```

### Query Planning

```python
from semantica.triplet_store import QueryEngine

engine = QueryEngine(enable_optimization=True)

query = """
SELECT ?s ?p ?o
WHERE {
    ?s ?p ?o .
    FILTER(?o > 10)
}
ORDER BY ?s
LIMIT 100
"""

# Create query plan
plan = engine.plan_query(query)

print(f"Original query: {plan.query}")
print(f"Optimized query: {plan.optimized_query}")
print(f"Estimated cost: {plan.estimated_cost}")
print(f"Execution steps: {plan.execution_steps}")
```

### Query Statistics

```python
from semantica.triplet_store import QueryEngine, BlazegraphAdapter

engine = QueryEngine(enable_caching=True)
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

# Execute multiple queries
for i in range(10):
    query = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT {i * 10}"
    engine.execute_query(query, adapter)

# Get statistics
stats = engine.get_query_statistics()
print(f"Total queries: {stats['total_queries']}")
print(f"Average execution time: {stats['average_execution_time']:.2f}s")
print(f"Min execution time: {stats['min_execution_time']:.2f}s")
print(f"Max execution time: {stats['max_execution_time']:.2f}s")
print(f"Cache size: {stats['cache_size']}")
```

## Bulk Loading

### Basic Bulk Loading

```python
from semantica.triplet_store import BulkLoader, BlazegraphAdapter
from semantica.semantic_extract.triple_extractor import Triple

# Create bulk loader
loader = BulkLoader(batch_size=1000, max_retries=3)

# Create adapter
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

# Generate triples
triples = [
    Triple(f"http://example.org/entity{i}", "http://example.org/hasName", f"Entity {i}")
    for i in range(10000)
]

# Load triples
progress = loader.load_triples(triples, adapter)

print(f"Loaded: {progress.loaded_triples}/{progress.total_triples}")
print(f"Failed: {progress.failed_triples}")
print(f"Progress: {progress.progress_percentage:.1f}%")
print(f"Elapsed time: {progress.elapsed_time:.2f}s")
print(f"Throughput: {progress.metadata.get('throughput', 0):.0f} triples/sec")
```

### Progress Tracking

```python
from semantica.triplet_store import BulkLoader, BlazegraphAdapter, LoadProgress
from semantica.semantic_extract.triple_extractor import Triple

loader = BulkLoader(batch_size=1000)
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

# Progress callback
def progress_callback(progress: LoadProgress):
    print(f"Batch {progress.current_batch}/{progress.total_batches}: "
          f"{progress.progress_percentage:.1f}% "
          f"({progress.loaded_triples}/{progress.total_triples})")

# Load with progress callback
triples = [Triple(f"http://example.org/entity{i}", "http://example.org/hasName", f"Entity {i}")
           for i in range(5000)]
progress = loader.load_triples(triples, adapter, progress_callback=progress_callback)
```

### Pre-load Validation

```python
from semantica.triplet_store import BulkLoader
from semantica.semantic_extract.triple_extractor import Triple

loader = BulkLoader()

# Create triples (some invalid)
triples = [
    Triple("http://example.org/entity1", "http://example.org/hasName", "John"),  # Valid
    Triple("", "http://example.org/hasName", "Jane"),  # Invalid (empty subject)
    Triple("http://example.org/entity3", "", "Bob"),  # Invalid (empty predicate)
    Triple("http://example.org/entity4", "http://example.org/hasAge", "30"),  # Valid
]

# Validate before loading
validation = loader.validate_before_load(triples)

print(f"Valid: {validation['valid']}")
print(f"Errors: {validation['errors']}")
print(f"Warnings: {validation['warnings']}")
print(f"Valid triples: {validation['valid_triples']}/{validation['total_triples']}")
```

### Stream-based Loading

```python
from semantica.triplet_store import BulkLoader, BlazegraphAdapter
from semantica.semantic_extract.triple_extractor import Triple

loader = BulkLoader(batch_size=1000)
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

# Create stream of triples
def triple_stream():
    for i in range(10000):
        yield Triple(f"http://example.org/entity{i}", "http://example.org/hasName", f"Entity {i}")

# Load from stream
progress = loader.load_from_stream(triple_stream(), adapter)
print(f"Loaded {progress.loaded_triples} triples from stream")
```

## Store Adapters

### Blazegraph Adapter

```python
from semantica.triplet_store import BlazegraphAdapter
from semantica.semantic_extract.triple_extractor import Triple

# Create Blazegraph adapter
adapter = BlazegraphAdapter(
    endpoint="http://localhost:9999/blazegraph",
    namespace="kb",
    auth=("user", "password")  # Optional
)

# Add triples
triples = [
    Triple("http://example.org/entity1", "http://example.org/hasName", "John")
]
result = adapter.add_triples(triples)
print(f"Added: {result['success']}")

# Execute SPARQL query
query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
result = adapter.execute_sparql(query)
print(f"Found {len(result['bindings'])} results")
```

### Jena Adapter

```python
from semantica.triplet_store import JenaAdapter
from semantica.semantic_extract.triple_extractor import Triple

# Create Jena adapter (in-memory)
adapter = JenaAdapter()

# Or connect to Fuseki endpoint
adapter = JenaAdapter(
    endpoint="http://localhost:3030/ds",
    dataset="default",
    enable_inference=True
)

# Add triples
triples = [
    Triple("http://example.org/entity1", "http://example.org/hasName", "John")
]
result = adapter.add_triples(triples)

# Serialize to Turtle
turtle = adapter.serialize(format="turtle")
print(turtle)
```

### RDF4J Adapter

```python
from semantica.triplet_store import RDF4JAdapter
from semantica.semantic_extract.triple_extractor import Triple

# Create RDF4J adapter
adapter = RDF4JAdapter(
    endpoint="http://localhost:8080/rdf4j-server",
    repository="test"
)

# Add triples
triples = [
    Triple("http://example.org/entity1", "http://example.org/hasName", "John")
]
result = adapter.add_triples(triples)
```

### Virtuoso Adapter

```python
from semantica.triplet_store import VirtuosoAdapter
from semantica.semantic_extract.triple_extractor import Triple

# Create Virtuoso adapter
adapter = VirtuosoAdapter(
    endpoint="http://localhost:8890/sparql",
    user="dba",
    password="dba"
)

# Add triples
triples = [
    Triple("http://example.org/entity1", "http://example.org/hasName", "John")
]
result = adapter.add_triples(triples)
```

## Algorithms and Methods

### Triplet Store Management Algorithms

#### Store Registration
**Algorithm**: Store type detection and adapter factory pattern

1. **Store Type Detection**: Identify backend type (blazegraph, jena, rdf4j, virtuoso)
2. **Configuration Storage**: Store store configuration (endpoint, namespace, etc.)
3. **Adapter Factory**: Create appropriate adapter instance based on store type
4. **Default Store Selection**: Set first registered store as default if none specified

**Time Complexity**: O(1) for registration
**Space Complexity**: O(1) per store

```python
# Store registration
store = manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")
```

#### Adapter Pattern
**Algorithm**: Unified interface for multiple backends

1. **Interface Definition**: Common interface for all adapters (add_triple, execute_sparql, etc.)
2. **Backend-Specific Implementation**: Each adapter implements interface for its backend
3. **Adapter Instantiation**: Create adapter instance on-demand
4. **Operation Delegation**: Delegate operations to appropriate adapter

**Time Complexity**: O(1) for adapter creation
**Space Complexity**: O(1) per adapter

### CRUD Operations Algorithms

#### Triple Addition
**Algorithm**: Single and batch triple insertion

1. **Triple Validation**: Check required fields (subject, predicate, object), validate confidence (0-1)
2. **Adapter Selection**: Get adapter for specified store
3. **Operation Delegation**: Delegate to adapter's add_triple/add_triples method
4. **Result Processing**: Process and return operation result

**Time Complexity**: O(1) for single, O(n) for batch where n = triples
**Space Complexity**: O(1) for single, O(n) for batch

```python
# Triple addition
result = manager.add_triple(triple, store_id="main")
result = manager.add_triples(triples, store_id="main", batch_size=1000)
```

#### Triple Retrieval
**Algorithm**: Pattern-based triple retrieval

1. **Pattern Construction**: Build SPARQL query from subject/predicate/object patterns
2. **Query Execution**: Execute SPARQL query via adapter
3. **Result Binding Extraction**: Extract bindings from query result
4. **Triple Reconstruction**: Convert bindings to Triple objects

**Time Complexity**: O(n) where n = result count
**Space Complexity**: O(n) for results

```python
# Triple retrieval
triples = manager.get_triple(subject="http://example.org/entity1", store_id="main")
```

### Bulk Loading Algorithms

#### Batch Processing
**Algorithm**: Chunking algorithm for large datasets

1. **Batch Creation**: Divide triples into fixed-size batches
2. **Batch Processing**: Process each batch sequentially
3. **Progress Tracking**: Track loaded count, failed count, progress percentage
4. **Error Handling**: Retry failed batches with exponential backoff

**Time Complexity**: O(n) where n = total triples
**Space Complexity**: O(b) where b = batch size

```python
# Batch processing
progress = loader.load_triples(triples, adapter, batch_size=1000)
```

#### Progress Tracking
**Algorithm**: Load progress calculation

1. **Progress Calculation**: loaded_triples / total_triples * 100
2. **Elapsed Time Tracking**: Track time since start
3. **Estimated Remaining**: (elapsed / loaded) * (total - loaded)
4. **Throughput Calculation**: loaded_triples / elapsed_time

**Time Complexity**: O(1) per update
**Space Complexity**: O(1)

#### Retry Mechanism
**Algorithm**: Exponential backoff retry

1. **Retry Attempt**: Try batch load operation
2. **Failure Detection**: Catch exceptions
3. **Backoff Calculation**: delay = retry_delay * (attempt + 1)
4. **Max Retry Check**: Stop after max_retries attempts

**Time Complexity**: O(r) where r = max_retries
**Space Complexity**: O(1)

### SPARQL Query Execution Algorithms

#### Query Validation
**Algorithm**: Syntax validation and query type detection

1. **Keyword Checking**: Check for valid SPARQL keywords (SELECT, ASK, CONSTRUCT, etc.)
2. **Structure Validation**: Validate query structure
3. **Query Type Detection**: Identify query type (SELECT, ASK, CONSTRUCT, DESCRIBE, INSERT, DELETE)

**Time Complexity**: O(n) where n = query length
**Space Complexity**: O(1)

```python
# Query validation
is_valid = engine._validate_query(query)
```

#### Query Optimization
**Algorithm**: Basic query optimization

1. **Whitespace Normalization**: Remove unnecessary whitespace
2. **LIMIT Injection**: Add LIMIT clause if SELECT query doesn't have one
3. **Query Rewriting**: Simplify query structure

**Time Complexity**: O(n) where n = query length
**Space Complexity**: O(n) for optimized query

```python
# Query optimization
optimized = engine.optimize_query(query, add_limit=True, default_limit=1000)
```

#### Query Caching
**Algorithm**: MD5-based cache key generation and LRU-style eviction

1. **Cache Key Generation**: MD5 hash of normalized query
2. **Cache Lookup**: Check if query result is cached
3. **Cache Storage**: Store query result with cache key
4. **Cache Eviction**: Remove oldest entry when cache is full

**Time Complexity**: O(1) for lookup/storage
**Space Complexity**: O(c) where c = cache size

```python
# Query caching
result = engine.execute_query(query, adapter)  # Cached on second call
```

#### Cost Estimation
**Algorithm**: Heuristic-based cost calculation

1. **Base Cost**: Start with base cost of 1.0
2. **COUNT Detection**: Multiply by 2.0 if COUNT query
3. **Join Count**: Multiply by (1.0 + join_count * 0.1)
4. **DISTINCT Detection**: Multiply by 1.5 if DISTINCT

**Time Complexity**: O(n) where n = query length
**Space Complexity**: O(1)

```python
# Cost estimation
cost = engine._estimate_query_cost(query)
```

### Methods

#### TripletManager Methods

- `register_store(store_id, store_type, endpoint, **config)`: Register triplet store
- `add_triple(triple, store_id, **options)`: Add single triple
- `add_triples(triples, store_id, **options)`: Add multiple triples
- `get_triple(subject, predicate, object, store_id, **options)`: Get triples matching pattern
- `delete_triple(triple, store_id, **options)`: Delete triple
- `update_triple(old_triple, new_triple, store_id, **options)`: Update triple
- `get_store(store_id)`: Get store by ID
- `list_stores()`: List all store IDs

#### QueryEngine Methods

- `execute_query(query, store_adapter, **options)`: Execute SPARQL query
- `optimize_query(query, **options)`: Optimize SPARQL query
- `plan_query(query, **options)`: Create query execution plan
- `clear_cache()`: Clear query cache
- `get_query_statistics()`: Get query execution statistics

#### BulkLoader Methods

- `load_triples(triples, store_adapter, **options)`: Load triples in bulk
- `load_from_file(file_path, store_adapter, **options)`: Load triples from file
- `load_from_stream(triples_stream, store_adapter, **options)`: Load triples from stream
- `validate_before_load(triples, **options)`: Validate triples before loading

#### Convenience Functions

- `register_store(store_id, store_type, endpoint, method, **options)`: Register store wrapper
- `add_triple(triple, store_id, method, **options)`: Add triple wrapper
- `add_triples(triples, store_id, method, **options)`: Add triples wrapper
- `get_triples(subject, predicate, object, store_id, method, **options)`: Get triples wrapper
- `delete_triple(triple, store_id, method, **options)`: Delete triple wrapper
- `update_triple(old_triple, new_triple, store_id, method, **options)`: Update triple wrapper
- `execute_query(query, store_adapter, method, **options)`: Execute query wrapper
- `optimize_query(query, method, **options)`: Optimize query wrapper
- `bulk_load(triples, store_adapter, method, **options)`: Bulk load wrapper
- `validate_triples(triples, method, **options)`: Validate triples wrapper

## Dataclasses

### TripletStore

Configuration dataclass for triplet store instances.

```python
from semantica.triplet_store import TripletStore

store = TripletStore(
    store_id="main",
    store_type="blazegraph",
    endpoint="http://localhost:9999/blazegraph/sparql",
    config={
        "namespace": "kb",
        "timeout": 30
    }
)

print(f"Store ID: {store.store_id}")
print(f"Type: {store.store_type}")
```

**Attributes:**
- `store_id` (str): Unique store identifier
- `store_type` (str): Backend type (blazegraph, jena, rdf4j, virtuoso)
- `endpoint` (str): SPARQL endpoint URL
- `config` (dict): Additional configuration options

### QueryResult

Query execution result dataclass.

```python
from semantica.triplet_store import QueryEngine, QueryResult

engine = QueryEngine()
result: QueryResult = engine.execute_query(query, adapter)

print(f"Variables: {result.variables}")
print(f"Results: {len(result.bindings)}")
print(f"Execution time: {result.execution_time:.2f}s")
```

**Attributes:**
- `variables` (List[str]): Query variable names
- `bindings` (List[Dict]): Result bindings
- `execution_time` (float): Query execution time in seconds
- `metadata` (Dict): Additional metadata (cached, optimized, etc.)

### QueryPlan

Query execution plan dataclass.

```python
from semantica.triplet_store import QueryEngine, QueryPlan

engine = QueryEngine(enable_optimization=True)
plan: QueryPlan = engine.plan_query(query)

print(f"Estimated cost: {plan.estimated_cost}")
print(f"Execution steps: {plan.execution_steps}")
```

**Attributes:**
- `query` (str): Original SPARQL query
- `optimized_query` (str): Optimized query
- `estimated_cost` (float): Estimated execution cost
- `execution_steps` (List[str]): Planned execution steps

## Configuration

### Environment Variables

```bash
# Triplet store configuration
export TRIPLET_STORE_DEFAULT_STORE=main
export TRIPLET_STORE_BATCH_SIZE=1000
export TRIPLET_STORE_ENABLE_CACHING=true
export TRIPLET_STORE_CACHE_SIZE=1000
export TRIPLET_STORE_ENABLE_OPTIMIZATION=true
export TRIPLET_STORE_MAX_RETRIES=3
export TRIPLET_STORE_RETRY_DELAY=1.0
export TRIPLET_STORE_TIMEOUT=30

# Store endpoints
export TRIPLET_STORE_BLAZEGRAPH_ENDPOINT=http://localhost:9999/blazegraph
export TRIPLET_STORE_JENA_ENDPOINT=http://localhost:3030/ds
export TRIPLET_STORE_RDF4J_ENDPOINT=http://localhost:8080/rdf4j-server
export TRIPLET_STORE_VIRTUOSO_ENDPOINT=http://localhost:8890/sparql
```

### Programmatic Configuration

```python
from semantica.triplet_store.config import triplet_store_config

# Get configuration
batch_size = triplet_store_config.get("batch_size", default=1000)
enable_caching = triplet_store_config.get("enable_caching", default=True)

# Set configuration
triplet_store_config.set("batch_size", 2000)
triplet_store_config.set("enable_caching", False)

# Update with dictionary
triplet_store_config.update({
    "batch_size": 2000,
    "enable_caching": True,
    "cache_size": 2000
})
```

### Configuration File (YAML)

```yaml
# config.yaml
triplet_store:
  default_store: main
  batch_size: 1000
  enable_caching: true
  cache_size: 1000
  enable_optimization: true
  max_retries: 3
  retry_delay: 1.0
  timeout: 30
  blazegraph_endpoint: http://localhost:9999/blazegraph
  jena_endpoint: http://localhost:3030/ds
  rdf4j_endpoint: http://localhost:8080/rdf4j-server
  virtuoso_endpoint: http://localhost:8890/sparql
```

## Advanced Examples

### Complete Triplet Store Pipeline

```python
from semantica.triplet_store import (
    TripletManager,
    QueryEngine,
    BulkLoader,
    register_store,
    add_triples,
    execute_query
)
from semantica.semantic_extract.triple_extractor import Triple

# 1. Register store
store = register_store("main", "blazegraph", "http://localhost:9999/blazegraph")

# 2. Generate and add triples
triples = [
    Triple(f"http://example.org/entity{i}", "http://example.org/hasName", f"Entity {i}")
    for i in range(1000)
]
result = add_triples(triples, store_id="main", batch_size=100)

# 3. Execute queries
from semantica.triplet_store import BlazegraphAdapter
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")
query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
query_result = execute_query(query, adapter)

print(f"Added {result['total_triples']} triples")
print(f"Query returned {len(query_result.bindings)} results")
```

### Multi-Store Operations

```python
from semantica.triplet_store import TripletManager
from semantica.semantic_extract.triple_extractor import Triple

manager = TripletManager()

# Register multiple stores
manager.register_store("primary", "blazegraph", "http://localhost:9999/blazegraph")
manager.register_store("backup", "jena", "http://localhost:3030/ds")

# Add to primary store
triple = Triple("http://example.org/entity1", "http://example.org/hasName", "John")
manager.add_triple(triple, store_id="primary")

# Replicate to backup store
manager.add_triple(triple, store_id="backup")
```

### Query Optimization Workflow

```python
from semantica.triplet_store import QueryEngine, BlazegraphAdapter

engine = QueryEngine(enable_optimization=True, enable_caching=True)
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

# Original query
query = """
SELECT ?s ?p ?o
WHERE {
    ?s ?p ?o .
    FILTER(?o > 10)
}
"""

# Plan query
plan = engine.plan_query(query)
print(f"Estimated cost: {plan.estimated_cost}")
print(f"Execution steps: {plan.execution_steps}")

# Execute optimized query
result = engine.execute_query(query, adapter)
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Optimized: {result.metadata.get('optimized', False)}")
```

### Bulk Loading with Validation

```python
from semantica.triplet_store import BulkLoader, BlazegraphAdapter
from semantica.semantic_extract.triple_extractor import Triple

loader = BulkLoader(batch_size=1000, max_retries=3)
adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph")

# Generate triples
triples = [
    Triple(f"http://example.org/entity{i}", "http://example.org/hasName", f"Entity {i}")
    for i in range(10000)
]

# Validate before loading
validation = loader.validate_before_load(triples)
if validation['valid']:
    # Load triples
    progress = loader.load_triples(triples, adapter)
    print(f"Loaded {progress.loaded_triples}/{progress.total_triples} triples")
    print(f"Throughput: {progress.metadata.get('throughput', 0):.0f} triples/sec")
else:
    print(f"Validation failed: {validation['errors']}")
```

### Custom Method Registration

```python
from semantica.triplet_store.registry import method_registry
from semantica.triplet_store import add_triple

# Register custom add method
def custom_add_triple(triple, store_id=None, **options):
    # Custom logic
    print(f"Custom add: {triple.subject}")
    # Call default implementation
    from semantica.triplet_store.methods import _get_manager
    manager = _get_manager()
    return manager.add_triple(triple, store_id=store_id, **options)

method_registry.register("add", "custom", custom_add_triple)

# Use custom method
from semantica.triplet_store.methods import add_triple
result = add_triple(triple, store_id="main", method="custom")
```

## Best Practices

1. **Store Registration**:
   - Register stores before use
   - Use descriptive store IDs
   - Set appropriate default store
   - Configure store-specific options

2. **Triple Operations**:
   - Validate triples before adding
   - Use batch operations for multiple triples
   - Set appropriate batch sizes
   - Handle errors gracefully

3. **Query Execution**:
   - Enable caching for repeated queries
   - Use query optimization for complex queries
   - Monitor query performance
   - Use appropriate LIMIT clauses

4. **Bulk Loading**:
   - Validate triples before loading
   - Use appropriate batch sizes
   - Monitor progress for large loads
   - Handle retries appropriately

5. **Performance**:
   - Use batch operations when possible
   - Enable query caching
   - Optimize queries before execution
   - Monitor query statistics

6. **Error Handling**:
   - Validate triples before operations
   - Handle adapter errors gracefully
   - Use retry mechanisms for bulk operations
   - Log errors for debugging

7. **Configuration**:
   - Use environment variables for deployment
   - Use config files for development
   - Set appropriate defaults
   - Document configuration options

8. **Testing**:
   - Test with sample data first
   - Validate query results
   - Test error handling
   - Monitor performance

