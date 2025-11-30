# Export

> **Export knowledge graphs and data to multiple formats with W3C-compliant serialization.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-file-export:{ .lg .middle } **Multi-Format Export**

    ---

    Support for 10+ export formats including RDF, JSON, CSV, GraphML

-   :material-database-export:{ .lg .middle } **Graph Databases**

    ---

    Direct export to Neo4j, ArangoDB, Memgraph with Cypher generation

-   :material-code-json:{ .lg .middle } **RDF Serialization**

    ---

    W3C-compliant formats: Turtle, RDF/XML, JSON-LD, N-Triples

-   :material-cog:{ .lg .middle } **Custom Serializers**

    ---

    Extensible serialization framework for custom formats

-   :material-flash:{ .lg .middle } **Batch Export**

    ---

    Efficient large-scale data export with streaming support

-   :material-file-multiple:{ .lg .middle } **Multiple Outputs**

    ---

    Export to Cytoscape.js, D3.js, Gephi, Graphviz formats

</div>

!!! tip "Choosing Export Format"
    - **Development**: Use Turtle for human-readable RDF
    - **APIs**: Use JSON-LD for web services
    - **Visualization**: Use GraphML or Cytoscape.js
    - **Databases**: Use Neo4j Cypher or CSV for bulk import

---

## ⚙️ Algorithms Used

### Serialization Algorithms
- **RDF/XML Serialization**: W3C RDF/XML specification
- **Turtle Serialization**: Compact RDF format with prefix compression
- **JSON-LD Serialization**: JSON-based linked data with context
- **GraphML Generation**: XML-based graph format
- **Cypher Query Generation**: Neo4j query language generation

### Export Optimization
- **Streaming Export**: Memory-efficient export for large graphs
- **Batch Processing**: Chunked export with configurable batch sizes
- **Compression**: GZIP compression for large exports
- **Incremental Export**: Export only changed data

---

## Main Classes

### RDFExporter

Export knowledge graphs to RDF formats (Turtle, RDF/XML, JSON-LD, N-Triples).

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(graph, filename, format)` | Export to RDF format | RDF serialization with format-specific encoding |
| `serialize(graph, format)` | Serialize to string | In-memory RDF generation |
| `validate(rdf_data)` | Validate RDF syntax | RDF schema validation |
| `add_namespace(prefix, uri)` | Add namespace | Prefix registration |
| `set_base_uri(uri)` | Set base URI | Base URI configuration |

**Supported RDF Formats:**

| Format | Extension | Description | Use Case |
|--------|-----------|-------------|----------|
| **Turtle** | .ttl | Compact, human-readable | Development, debugging |
| **N-Triples** | .nt | Line-based, simple | Streaming, processing |
| **RDF/XML** | .rdf | XML-based, verbose | Legacy systems |
| **JSON-LD** | .jsonld | JSON with linked data | Web APIs, JavaScript |
| **N-Quads** | .nq | N-Triples with graphs | Named graphs |

**Example:**

```python
from semantica.export import RDFExporter

exporter = RDFExporter(
    base_uri="http://example.org/",
    namespaces={
        "ex": "http://example.org/",
        "foaf": "http://xmlns.com/foaf/0.1/"
    }
)

# Export to Turtle
exporter.export(
    graph=kg,
    filename="output.ttl",
    format="turtle"
)

# Export to JSON-LD
exporter.export(
    graph=kg,
    filename="output.jsonld",
    format="json-ld"
)
```

---

### JSONExporter

Export knowledge graphs to JSON formats including JSON-LD, Cytoscape.js, and D3.js.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(graph, filename, format)` | Export to JSON | JSON serialization with schema |
| `export_nodes(graph)` | Export nodes only | Node extraction and serialization |
| `export_edges(graph)` | Export edges only | Edge extraction and serialization |
| `export_cytoscape(graph)` | Export Cytoscape.js format | Cytoscape JSON generation |
| `export_d3(graph)` | Export D3.js format | D3 force-directed graph format |

**JSON Formats:**

- **Standard JSON**: Simple node/edge lists
- **JSON-LD**: Linked data with @context
- **Cytoscape.js**: For Cytoscape visualization
- **D3.js**: For D3 force-directed graphs
- **Neo4j JSON**: Neo4j-compatible format

**Example:**

```python
from semantica.export import JSONExporter

exporter = JSONExporter()

# Standard JSON export
exporter.export(kg, "output.json", format="standard")

# JSON-LD export
exporter.export(kg, "output.jsonld", format="json-ld")

# Cytoscape format
exporter.export_cytoscape(kg, "cytoscape.json")
```

---

### GraphExporter

Export to graph visualization formats (GraphML, GEXF, DOT, Pajek).

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(graph, filename, format)` | Export to graph format | Format-specific serialization |
| `to_graphml(graph, filename)` | Export to GraphML | XML-based graph format |
| `to_gexf(graph, filename)` | Export to GEXF | Gephi exchange format |
| `to_dot(graph, filename)` | Export to DOT | Graphviz format |
| `to_pajek(graph, filename)` | Export to Pajek | Pajek network format |

**Graph Formats:**

| Format | Tool | Use Case |
|--------|------|----------|
| **GraphML** | yEd, Gephi | General graph visualization |
| **GEXF** | Gephi | Network analysis |
| **DOT** | Graphviz | Diagram generation |
| **Pajek** | Pajek | Large network analysis |
| **GML** | Various | Graph Modeling Language |

**Example:**

```python
from semantica.export import GraphExporter

exporter = GraphExporter()

# Export to GraphML
exporter.to_graphml(kg, "graph.graphml")

# Export to DOT for Graphviz
exporter.to_dot(kg, "graph.dot")
```

---

### Neo4jExporter

Export directly to Neo4j graph database with Cypher query generation.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(graph, uri, username, password)` | Export to Neo4j | Cypher query execution |
| `generate_cypher(graph)` | Generate Cypher queries | CREATE/MERGE statement generation |
| `batch_import(graph, batch_size)` | Batch import | Chunked Cypher execution |
| `create_indexes(properties)` | Create indexes | Index creation for performance |
| `create_constraints(constraints)` | Create constraints | Uniqueness constraint creation |

**Cypher Generation:**
```cypher
// Node creation
CREATE (n:Person {name: "Steve Jobs", born: 1955})

// Relationship creation
MATCH (a:Person {name: "Steve Jobs"}), (b:Organization {name: "Apple Inc."})
CREATE (a)-[:FOUNDED]->(b)
```

**Example:**

```python
from semantica.export import Neo4jExporter

exporter = Neo4jExporter()

# Direct export to Neo4j
exporter.export(
    graph=kg,
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Generate Cypher queries
cypher_queries = exporter.generate_cypher(kg)
print(cypher_queries)
```

---

### CSVExporter

Export to CSV format for spreadsheets and database imports.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export_nodes(graph, filename)` | Export nodes to CSV | Node flattening and CSV writing |
| `export_edges(graph, filename)` | Export edges to CSV | Edge list CSV generation |
| `export_combined(graph, prefix)` | Export nodes + edges | Separate CSV files |
| `flatten_properties(properties)` | Flatten nested properties | Recursive property flattening |

**CSV Formats:**
- **Nodes CSV**: id, label, properties (flattened)
- **Edges CSV**: source, target, type, properties
- **Neo4j Import Format**: Neo4j-compatible CSV

**Example:**

```python
from semantica.export import CSVExporter

exporter = CSVExporter()

# Export nodes and edges separately
exporter.export_nodes(kg, "nodes.csv")
exporter.export_edges(kg, "edges.csv")

# Or combined export
exporter.export_combined(kg, prefix="graph")
# Creates: graph_nodes.csv, graph_edges.csv
```

---

### VectorExporter

Export vector embeddings to various formats.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(embeddings, filename, format)` | Export vectors | Format-specific vector serialization |
| `export_numpy(embeddings, filename)` | Export to NumPy | .npy format |
| `export_hdf5(embeddings, filename)` | Export to HDF5 | Hierarchical data format |
| `export_parquet(embeddings, filename)` | Export to Parquet | Columnar storage format |

---

## Configuration

```yaml
# config.yaml - Export Configuration

export:
  rdf:
    default_format: turtle
    base_uri: "http://example.org/"
    include_provenance: true
    validate_output: true
    
  json:
    pretty_print: true
    indent: 2
    ensure_ascii: false
    
  graph:
    include_metadata: true
    export_properties: true
    
  neo4j:
    batch_size: 1000
    create_indexes: true
    create_constraints: true
    
  csv:
    delimiter: ","
    quote_char: "\""
    encoding: "utf-8"
    include_header: true
```

---

## Performance Characteristics

### Export Speed

| Format | Small Graph | Large Graph | Compression |
|--------|-------------|-------------|-------------|
| JSON | Fast | Medium | Good with gzip |
| RDF/XML | Medium | Slow | Poor |
| Turtle | Fast | Fast | Good |
| CSV | Very Fast | Very Fast | Excellent |
| Neo4j | Medium | Medium | N/A |

### Memory Usage

- **Streaming Export**: Constant memory usage
- **Batch Export**: Memory proportional to batch size
- **Full Export**: Memory proportional to graph size

---

## See Also

- [Knowledge Graph Module](kg.md) - Build graphs
- [Visualization Module](visualization.md) - Visualize exported graphs
- [Core Module](core.md) - Framework orchestration


## Overview

- **Multi-Format Export**: Support for 10+ export formats
- **Graph Database Export**: Direct export to Neo4j, ArangoDB, Memgraph
- **RDF Serialization**: W3C-compliant RDF formats
- **Custom Serializers**: Extensible serialization framework
- **Batch Export**: Efficient large-scale data export

---

## Algorithms Used

### Serialization Algorithms
- **RDF/XML Serialization**: W3C RDF/XML specification
- **Turtle Serialization**: Compact RDF format with prefix compression
- **JSON-LD Serialization**: JSON-based linked data with context
- **GraphML Generation**: XML-based graph format
- **Cypher Query Generation**: Neo4j query language generation

### Export Optimization
- **Streaming Export**: Memory-efficient export for large graphs
- **Batch Processing**: Chunked export with configurable batch sizes
- **Compression**: GZIP compression for large exports
- **Incremental Export**: Export only changed data

---

## Main Classes

### RDFExporter


**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(graph, filename, format)` | Export to RDF format | RDF serialization with format-specific encoding |
| `serialize(graph, format)` | Serialize to string | In-memory RDF generation |
| `validate(rdf_data)` | Validate RDF syntax | RDF schema validation |
| `add_namespace(prefix, uri)` | Add namespace | Prefix registration |
| `set_base_uri(uri)` | Set base URI | Base URI configuration |

**Supported RDF Formats:**

| Format | Extension | Description | Use Case |
|--------|-----------|-------------|----------|
| **Turtle** | .ttl | Compact, human-readable | Development, debugging |
| **N-Triples** | .nt | Line-based, simple | Streaming, processing |
| **RDF/XML** | .rdf | XML-based, verbose | Legacy systems |
| **JSON-LD** | .jsonld | JSON with linked data | Web APIs, JavaScript |
| **N-Quads** | .nq | N-Triples with graphs | Named graphs |

**Example:**

```python
from semantica.export import RDFExporter

exporter = RDFExporter(
    base_uri="http://example.org/",
    namespaces={
        "ex": "http://example.org/",
        "foaf": "http://xmlns.com/foaf/0.1/"
    }
)

# Export to Turtle
exporter.export(
    graph=kg,
    filename="output.ttl",
    format="turtle"
)

# Export to JSON-LD
exporter.export(
    graph=kg,
    filename="output.jsonld",
    format="json-ld"
)
```

---

### JSONExporter


**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(graph, filename, format)` | Export to JSON | JSON serialization with schema |
| `export_nodes(graph)` | Export nodes only | Node extraction and serialization |
| `export_edges(graph)` | Export edges only | Edge extraction and serialization |
| `export_cytoscape(graph)` | Export Cytoscape.js format | Cytoscape JSON generation |
| `export_d3(graph)` | Export D3.js format | D3 force-directed graph format |

**JSON Formats:**

- **Standard JSON**: Simple node/edge lists
- **JSON-LD**: Linked data with @context
- **Cytoscape.js**: For Cytoscape visualization
- **D3.js**: For D3 force-directed graphs
- **Neo4j JSON**: Neo4j-compatible format

**Example:**

```python
from semantica.export import JSONExporter

exporter = JSONExporter()

# Standard JSON export
exporter.export(kg, "output.json", format="standard")

# JSON-LD export
exporter.export(kg, "output.jsonld", format="json-ld")

# Cytoscape format
exporter.export_cytoscape(kg, "cytoscape.json")
```

---

### GraphExporter


**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(graph, filename, format)` | Export to graph format | Format-specific serialization |
| `to_graphml(graph, filename)` | Export to GraphML | XML-based graph format |
| `to_gexf(graph, filename)` | Export to GEXF | Gephi exchange format |
| `to_dot(graph, filename)` | Export to DOT | Graphviz format |
| `to_pajek(graph, filename)` | Export to Pajek | Pajek network format |

**Graph Formats:**

| Format | Tool | Use Case |
|--------|------|----------|
| **GraphML** | yEd, Gephi | General graph visualization |
| **GEXF** | Gephi | Network analysis |
| **DOT** | Graphviz | Diagram generation |
| **Pajek** | Pajek | Large network analysis |
| **GML** | Various | Graph Modeling Language |

**Example:**

```python
from semantica.export import GraphExporter

exporter = GraphExporter()

# Export to GraphML
exporter.to_graphml(kg, "graph.graphml")

# Export to DOT for Graphviz
exporter.to_dot(kg, "graph.dot")
```

---

### Neo4jExporter


**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(graph, uri, username, password)` | Export to Neo4j | Cypher query execution |
| `generate_cypher(graph)` | Generate Cypher queries | CREATE/MERGE statement generation |
| `batch_import(graph, batch_size)` | Batch import | Chunked Cypher execution |
| `create_indexes(properties)` | Create indexes | Index creation for performance |
| `create_constraints(constraints)` | Create constraints | Uniqueness constraint creation |

**Cypher Generation:**
```cypher
// Node creation
CREATE (n:Person {name: "Steve Jobs", born: 1955})

// Relationship creation
MATCH (a:Person {name: "Steve Jobs"}), (b:Organization {name: "Apple Inc."})
CREATE (a)-[:FOUNDED]->(b)
```

**Example:**

```python
from semantica.export import Neo4jExporter

exporter = Neo4jExporter()

# Direct export to Neo4j
exporter.export(
    graph=kg,
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password"
)

# Generate Cypher queries
cypher_queries = exporter.generate_cypher(kg)
print(cypher_queries)
```

---

### CSVExporter


**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export_nodes(graph, filename)` | Export nodes to CSV | Node flattening and CSV writing |
| `export_edges(graph, filename)` | Export edges to CSV | Edge list CSV generation |
| `export_combined(graph, prefix)` | Export nodes + edges | Separate CSV files |
| `flatten_properties(properties)` | Flatten nested properties | Recursive property flattening |

**CSV Formats:**
- **Nodes CSV**: id, label, properties (flattened)
- **Edges CSV**: source, target, type, properties
- **Neo4j Import Format**: Neo4j-compatible CSV

**Example:**

```python
from semantica.export import CSVExporter

exporter = CSVExporter()

# Export nodes and edges separately
exporter.export_nodes(kg, "nodes.csv")
exporter.export_edges(kg, "edges.csv")

# Or combined export
exporter.export_combined(kg, prefix="graph")
# Creates: graph_nodes.csv, graph_edges.csv
```

---

### VectorExporter


**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `export(embeddings, filename, format)` | Export vectors | Format-specific vector serialization |
| `export_numpy(embeddings, filename)` | Export to NumPy | .npy format |
| `export_hdf5(embeddings, filename)` | Export to HDF5 | Hierarchical data format |
| `export_parquet(embeddings, filename)` | Export to Parquet | Columnar storage format |

---

## Configuration

```yaml
# config.yaml - Export Configuration

export:
  rdf:
    default_format: turtle
    base_uri: "http://example.org/"
    include_provenance: true
    validate_output: true
    
  json:
    pretty_print: true
    indent: 2
    ensure_ascii: false
    
  graph:
    include_metadata: true
    export_properties: true
    
  neo4j:
    batch_size: 1000
    create_indexes: true
    create_constraints: true
    
  csv:
    delimiter: ","
    quote_char: "\""
    encoding: "utf-8"
    include_header: true
```

---

## Performance Characteristics

### Export Speed

| Format | Small Graph | Large Graph | Compression |
|--------|-------------|-------------|-------------|
| JSON | Fast | Medium | Good with gzip |
| RDF/XML | Medium | Slow | Poor |
| Turtle | Fast | Fast | Good |
| CSV | Very Fast | Very Fast | Excellent |
| Neo4j | Medium | Medium | N/A |

### Memory Usage

- **Streaming Export**: Constant memory usage
- **Batch Export**: Memory proportional to batch size
- **Full Export**: Memory proportional to graph size

---

## See Also

- [Knowledge Graph Module](kg.md) - Build graphs
- [Visualization Module](visualization.md) - Visualize exported graphs
- [Core Module](core.md) - Framework orchestration
