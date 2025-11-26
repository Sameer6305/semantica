# Examples

Real-world examples and use cases for Semantica.

!!! tip "Interactive Learning"
    For hands-on interactive tutorials, check out our [Cookbook](cookbook.md) with Jupyter notebooks covering everything from basics to advanced use cases.

## Basic Examples

!!! note "Code Examples"
    All examples assume you have Semantica installed and imported. See the [Installation Guide](installation.md) if you need to set it up first.

### Example 1: Basic Knowledge Graph

Build a knowledge graph from a single document:

```python
from semantica import Semantica

semantica = Semantica()

# Build KG from PDF
result = semantica.build_knowledge_base(
    sources=["research_paper.pdf"],
    embeddings=True,
    graph=True
)

kg = result["knowledge_graph"]
print(f"Entities: {len(kg['entities'])}")
print(f"Relationships: {len(kg['relationships'])}")
```

### Example 2: Entity Extraction

Extract entities from text:

```python
from semantica import Semantica

semantica = Semantica()

text = """
Apple Inc. is a technology company founded by Steve Jobs.
The company is headquartered in Cupertino, California.
Tim Cook is the current CEO of Apple.
"""

entities = semantica.semantic_extract.extract_entities(text)
for entity in entities["entities"]:
    print(f"{entity['text']}: {entity['type']}")
```

**Output:**
```
Apple Inc.: ORGANIZATION
Steve Jobs: PERSON
Cupertino: LOCATION
California: LOCATION
Tim Cook: PERSON
```

### Example 3: Multi-Source Integration

Combine data from multiple sources:

```python
from semantica import Semantica

semantica = Semantica()

sources = [
    "documents/finance_report.pdf",
    "documents/market_analysis.docx",
    "https://example.com/news-article"
]

result = semantica.build_knowledge_base(sources)
kg = result["knowledge_graph"]

print(f"Unified knowledge graph with {len(kg['entities'])} entities")
```

### Example 4: Export Formats

Export knowledge graph to multiple formats:

```python
from semantica import Semantica

semantica = Semantica()
kg = semantica.kg.build_graph(["data.pdf"])

# Export to different formats
semantica.export.to_rdf(kg, "output.rdf")
semantica.export.to_json(kg, "output.json")
semantica.export.to_csv(kg, "output.csv")
semantica.export.to_owl(kg, "output.owl")
```

## Advanced Examples

### Example 5: Conflict Resolution

Resolve conflicts in data from multiple sources:

```python
from semantica import Semantica
from semantica.conflicts import ConflictResolver

semantica = Semantica()

# Build graph from multiple sources
result = semantica.build_knowledge_base([
    "source1.pdf",
    "source2.pdf",
    "source3.pdf"
])

# Detect conflicts
conflicts = semantica.kg.detect_conflicts(result["knowledge_graph"])

# Resolve conflicts
resolver = ConflictResolver(default_strategy="voting")
resolved = resolver.resolve_conflicts(conflicts)

print(f"Resolved {len(resolved)} conflicts")
```

### Example 6: Custom Configuration

Use custom configuration for specific use cases:

```python
from semantica import Semantica, Config

# Custom configuration
config = Config(
    embeddings=True,
    graph=True,
    normalize=True,
    conflict_resolution="highest_confidence"
)

semantica = Semantica(config=config)
result = semantica.build_knowledge_base(["document.pdf"])
```

### Example 7: Incremental Graph Building

Build knowledge graph incrementally:

```python
from semantica import Semantica

semantica = Semantica()

# Build graphs separately
kg1 = semantica.kg.build_graph(["source1.pdf"])
kg2 = semantica.kg.build_graph(["source2.pdf"])
kg3 = semantica.kg.build_graph(["source3.pdf"])

# Merge into unified graph
merged_kg = semantica.kg.merge([kg1, kg2, kg3])

print(f"Merged graph: {len(merged_kg['entities'])} entities")
```

### Example 8: Visualization

Create interactive visualizations:

```python
from semantica import Semantica

semantica = Semantica()

# Build graph
result = semantica.build_knowledge_base(["document.pdf"])
kg = result["knowledge_graph"]

# Visualize
semantica.kg.visualize(kg, output_path="graph.html")

# Also analyze
analysis = semantica.kg.analyze(kg)
print(f"Graph density: {analysis['density']}")
print(f"Connected components: {analysis['components']}")
```

### Example 9: Graph Store (Persistent Storage)

Store and query knowledge graphs in a persistent graph database:

```python
from semantica.graph_store import GraphStore

# Initialize with Neo4j
store = GraphStore(
    backend="neo4j",
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
store.connect()

# Create nodes
apple = store.create_node(
    labels=["Company"],
    properties={"name": "Apple Inc.", "founded": 1976}
)

tim_cook = store.create_node(
    labels=["Person"],
    properties={"name": "Tim Cook", "title": "CEO"}
)

# Create relationship
store.create_relationship(
    start_node_id=tim_cook["id"],
    end_node_id=apple["id"],
    rel_type="CEO_OF",
    properties={"since": 2011}
)

# Query with Cypher
results = store.execute_query("""
    MATCH (p:Person)-[:CEO_OF]->(c:Company)
    RETURN p.name, c.name
""")

print(f"Query results: {results}")
store.close()
```

### Example 10: Using KuzuDB (Embedded)

For embedded graph storage without external dependencies:

```python
from semantica.graph_store import GraphStore

# KuzuDB - no server required
store = GraphStore(backend="kuzu", database_path="./my_graph_db")
store.connect()

# Store your knowledge graph
node = store.create_node(["Entity"], {"name": "Test"})
neighbors = store.get_neighbors(node["id"], depth=2)
stats = store.get_stats()

print(f"Graph stats: {stats}")
store.close()
```

### Example 11: FalkorDB for Real-Time Applications

Ultra-fast graph queries for LLM applications:

```python
from semantica.graph_store import GraphStore

# FalkorDB - Redis-based, ultra-fast
store = GraphStore(
    backend="falkordb",
    host="localhost",
    port=6379,
    graph_name="knowledge_graph"
)
store.connect()

# Fast queries for RAG applications
results = store.execute_query("""
    MATCH (n)-[r]->(m)
    WHERE n.name CONTAINS $query
    RETURN n, r, m LIMIT 10
""", parameters={"query": "AI"})

store.close()
```

## Use Case Examples

### Research Paper Analysis

Extract knowledge from research papers:

```python
from semantica import Semantica

semantica = Semantica()

# Process research paper
result = semantica.build_knowledge_base([
    "papers/ai_research.pdf",
    "papers/ml_survey.pdf"
])

kg = result["knowledge_graph"]

# Find key concepts
concepts = [e for e in kg["entities"] if e["type"] == "CONCEPT"]
print(f"Found {len(concepts)} key concepts")
```

### Company Intelligence

Build knowledge graph from company documents:

```python
from semantica import Semantica

semantica = Semantica()

# Company documents
sources = [
    "company/annual_report.pdf",
    "company/press_releases/",
    "company/website_content.html"
]

result = semantica.build_knowledge_base(sources)
kg = result["knowledge_graph"]

# Export for analysis
semantica.export.to_json(kg, "company_intelligence.json")
```

### News Article Processing

Process and analyze news articles:

```python
from semantica import Semantica

semantica = Semantica()

# News articles
articles = [
    "https://example.com/article1",
    "https://example.com/article2",
    "https://example.com/article3"
]

result = semantica.build_knowledge_base(articles)
kg = result["knowledge_graph"]

# Extract key entities
people = [e for e in kg["entities"] if e["type"] == "PERSON"]
organizations = [e for e in kg["entities"] if e["type"] == "ORGANIZATION"]

print(f"People mentioned: {len(people)}")
print(f"Organizations: {len(organizations)}")
```

## Interactive Examples

For more interactive examples and tutorials, check out our [Cookbook](cookbook.md) with Jupyter notebooks covering:

- **Introduction**: Getting started tutorials
- **Advanced**: Advanced techniques and patterns
- **Use Cases**: Real-world applications in various domains

## More Resources

- **[Quick Start Guide](quickstart.md)** - Step-by-step tutorial
- **[API Reference](api.md)** - Complete API documentation
- **[Cookbook](cookbook.md)** - Interactive Jupyter notebooks
- **[Code Examples](../CodeExamples.md)** - Additional code samples
