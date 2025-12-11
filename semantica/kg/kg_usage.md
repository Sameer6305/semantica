# Knowledge Graph Module Usage Guide

This guide demonstrates how to use the knowledge graph module for building, analyzing, validating, and managing knowledge graphs, including temporal knowledge graphs, entity resolution, and graph analytics.

Note: For conflict detection and resolution, use the `semantica.conflicts` module.
For deduplication, use the `semantica.deduplication` module.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Knowledge Graph Building](#knowledge-graph-building)
3. [Graph Analysis](#graph-analysis)
4. [Entity Resolution](#entity-resolution)
5. [Graph Validation](#graph-validation)
6. [Centrality Calculation](#centrality-calculation)
7. [Community Detection](#community-detection)
8. [Connectivity Analysis](#connectivity-analysis)
9. [Temporal Queries](#temporal-queries)
10. [Provenance Tracking](#provenance-tracking)
11. [Using Methods](#using-methods)
12. [Using Registry](#using-registry)
13. [Configuration](#configuration)
14. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using Main Classes

```python
from semantica.kg import GraphBuilder, GraphAnalyzer

# Create graph builder
builder = GraphBuilder(
    merge_entities=True,
    entity_resolution_strategy="fuzzy",
    resolve_conflicts=True
)

# Build knowledge graph
kg = builder.build(sources)

# Analyze graph
analyzer = GraphAnalyzer()
analysis = analyzer.analyze_graph(kg)
```


## Knowledge Graph Building

### Basic Graph Building

```python
from semantica.kg import GraphBuilder

# Create graph builder
# Note: resolve_conflicts=True uses the basic resolution capabilities of ConflictDetector.
# For advanced conflict resolution, consider using the semantica.conflicts module directly.
builder = GraphBuilder(
    merge_entities=True,
    entity_resolution_strategy="fuzzy",
    resolve_conflicts=True,
    enable_temporal=False
)

# Build knowledge graph
kg = builder.build(sources)
```

### Temporal Knowledge Graph Building

```python
from semantica.kg import GraphBuilder

# Build temporal knowledge graph
builder = GraphBuilder(
    enable_temporal=True,
    temporal_granularity="day",
    track_history=True,
    version_snapshots=True
)

temporal_kg = builder.build(sources)

# Access temporal information
for rel in temporal_kg["relationships"]:
    if "valid_from" in rel:
        print(f"Relationship valid from: {rel['valid_from']}")
```

### Incremental Building

```python
from semantica.kg import GraphBuilder

builder = GraphBuilder(merge_entities=True)

# Build initial graph
initial_sources = [{"entities": [...], "relationships": [...]}]
kg = builder.build(initial_sources)

# Add more sources incrementally
new_sources = [{"entities": [...], "relationships": [...]}]
updated_kg = builder.build(new_sources)
```

### Building with Different Configurations

```python
from semantica.kg import GraphBuilder

# Default building
builder = GraphBuilder()
kg = builder.build(sources)

# Temporal building
temporal_builder = GraphBuilder(enable_temporal=True)
temporal_kg = temporal_builder.build(sources)

# Incremental building (same builder, multiple calls)
builder = GraphBuilder(merge_entities=True)
kg1 = builder.build(initial_sources)
kg2 = builder.build(additional_sources)
```

## Graph Analysis

### Comprehensive Analysis

```python
from semantica.kg import GraphAnalyzer

# Create analyzer and analyze graph
analyzer = GraphAnalyzer()
analysis = analyzer.analyze_graph(kg)

print(f"Nodes: {analysis['num_nodes']}")
print(f"Edges: {analysis['num_edges']}")
print(f"Density: {analysis['density']}")
```

### Centrality-Focused Analysis

```python
from semantica.kg import GraphAnalyzer, CentralityCalculator

# Analyze graph with centrality focus
analyzer = GraphAnalyzer()
analysis = analyzer.analyze_graph(kg)

# Calculate centrality separately
centrality_calc = CentralityCalculator()
degree_centrality = centrality_calc.calculate_degree_centrality(kg)

# Access centrality results
if "rankings" in degree_centrality:
    print("Top nodes by degree centrality:")
    for ranking in degree_centrality["rankings"][:5]:
        print(f"  {ranking['node']}: {ranking['score']}")
```

### Community-Focused Analysis

```python
from semantica.kg import CommunityDetector

# Detect communities
detector = CommunityDetector()
result = detector.detect_communities(kg, algorithm="louvain")

# Access community results
if "communities" in result:
    communities = result["communities"]
    print(f"Found {len(communities)} communities")
    for i, community in enumerate(communities):
        print(f"Community {i}: {len(community)} nodes")
```

### Different Types of Analysis

```python
from semantica.kg import GraphAnalyzer, CentralityCalculator, CommunityDetector, ConnectivityAnalyzer

# Default comprehensive analysis
analyzer = GraphAnalyzer()
analysis = analyzer.analyze_graph(kg)

# Centrality analysis
centrality_calc = CentralityCalculator()
centrality = centrality_calc.calculate_all_centrality(kg)

# Community analysis
community_detector = CommunityDetector()
communities = community_detector.detect_communities(kg, algorithm="louvain")

# Connectivity analysis
connectivity_analyzer = ConnectivityAnalyzer()
connectivity = connectivity_analyzer.analyze_connectivity(kg)
```

## Entity Resolution

### Fuzzy Matching Resolution

```python
from semantica.kg import EntityResolver

entities = [
    {"id": "1", "name": "Apple Inc.", "type": "Company"},
    {"id": "2", "name": "Apple", "type": "Company"},
    {"id": "3", "name": "Microsoft", "type": "Company"}
]

# Create resolver with fuzzy strategy
resolver = EntityResolver(strategy="fuzzy", similarity_threshold=0.8)
resolved = resolver.resolve_entities(entities)

print(f"Original: {len(entities)} entities")
print(f"Resolved: {len(resolved)} entities")
```

### Exact Matching Resolution

```python
from semantica.kg import EntityResolver

# Exact string matching
resolver = EntityResolver(strategy="exact")
resolved = resolver.resolve_entities(entities)
```

### Semantic Matching Resolution

```python
from semantica.kg import EntityResolver

# Semantic similarity matching
resolver = EntityResolver(strategy="semantic", similarity_threshold=0.9)
resolved = resolver.resolve_entities(entities)
```

### Different Resolution Strategies

```python
from semantica.kg import EntityResolver

# Fuzzy matching
fuzzy_resolver = EntityResolver(strategy="fuzzy", similarity_threshold=0.8)
fuzzy_resolved = fuzzy_resolver.resolve_entities(entities)

# Exact matching
exact_resolver = EntityResolver(strategy="exact")
exact_resolved = exact_resolver.resolve_entities(entities)

# Semantic matching
semantic_resolver = EntityResolver(strategy="semantic", similarity_threshold=0.9)
semantic_resolved = semantic_resolver.resolve_entities(entities)
```

## Graph Validation

### Comprehensive Validation

```python
from semantica.kg import GraphValidator

# Create validator and validate graph
validator = GraphValidator()
result = validator.validate(kg)

if result.valid:
    print("Graph is valid!")
else:
    print(f"Found {len(result.errors)} errors:")
    for error in result.errors:
        print(f"  - {error}")
    
    print(f"Found {len(result.warnings)} warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")
```

### Structure-Only Validation

```python
from semantica.kg import GraphValidator

# Validate structure only
validator = GraphValidator()
result = validator.validate(kg)  # Full validation includes structure
```

### Consistency Checking

```python
from semantica.kg import GraphValidator

# Check consistency only
validator = GraphValidator()
is_consistent = validator.check_consistency(kg)
```

### Different Validation Approaches

```python
from semantica.kg import GraphValidator

validator = GraphValidator()

# Full validation (includes structure and consistency)
full_result = validator.validate(kg)

# Consistency check only
is_consistent = validator.check_consistency(kg)
```

!!! note "Conflict Detection and Resolution"
    Conflict detection and resolution have been moved to the dedicated `semantica.conflicts` module.
    Please use `semantica.conflicts.ConflictDetector` and `semantica.conflicts.ConflictResolver` for these tasks.

## Centrality Calculation

### Degree Centrality

```python
from semantica.kg import CentralityCalculator

# Calculate degree centrality
calculator = CentralityCalculator()
result = calculator.calculate_degree_centrality(kg)

print("Top nodes by degree centrality:")
for ranking in result["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")
```

### Betweenness Centrality

```python
from semantica.kg import CentralityCalculator

# Calculate betweenness centrality
calculator = CentralityCalculator()
result = calculator.calculate_betweenness_centrality(kg)

print("Top nodes by betweenness centrality:")
for ranking in result["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")
```

### Closeness Centrality

```python
from semantica.kg import CentralityCalculator

# Calculate closeness centrality
calculator = CentralityCalculator()
result = calculator.calculate_closeness_centrality(kg)

print("Top nodes by closeness centrality:")
for ranking in result["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")
```

### Eigenvector Centrality

```python
from semantica.kg import CentralityCalculator

# Calculate eigenvector centrality
calculator = CentralityCalculator()
result = calculator.calculate_eigenvector_centrality(kg)

print("Top nodes by eigenvector centrality:")
for ranking in result["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")
```

### All Centrality Measures

```python
from semantica.kg import CentralityCalculator

# Calculate all centrality measures
calculator = CentralityCalculator()
result = calculator.calculate_all_centrality(kg)

for measure_type, measure_result in result["centrality_measures"].items():
    print(f"\n{measure_type.upper()} Centrality:")
    for ranking in measure_result["rankings"][:3]:
        print(f"  {ranking['node']}: {ranking['score']}")
```

### Different Centrality Measures

```python
from semantica.kg import CentralityCalculator

calculator = CentralityCalculator()

# Degree centrality
degree = calculator.calculate_degree_centrality(kg)

# Betweenness centrality
betweenness = calculator.calculate_betweenness_centrality(kg)

# Closeness centrality
closeness = calculator.calculate_closeness_centrality(kg)

# Eigenvector centrality
eigenvector = calculator.calculate_eigenvector_centrality(kg)

# All measures at once
all_centrality = calculator.calculate_all_centrality(kg)
```

## Community Detection

### Louvain Algorithm

```python
from semantica.kg import CommunityDetector

# Detect communities using Louvain algorithm
detector = CommunityDetector()
result = detector.detect_communities(kg, algorithm="louvain")

print(f"Found {len(result['communities'])} communities")
print(f"Modularity: {result['modularity']}")

for i, community in enumerate(result["communities"]):
    print(f"Community {i}: {len(community)} nodes")
```

### Leiden Algorithm

```python
from semantica.kg import CommunityDetector

# Detect communities using Leiden algorithm
detector = CommunityDetector()
result = detector.detect_communities(kg, algorithm="leiden", resolution=1.0)

print(f"Found {len(result['communities'])} communities")
```

### Overlapping Communities

```python
from semantica.kg import CommunityDetector

# Detect overlapping communities
detector = CommunityDetector()
result = detector.detect_communities(kg, algorithm="overlapping", k=3)

print(f"Found {len(result['communities'])} overlapping communities")
print(f"Nodes in multiple communities: {result.get('overlap_count', 0)}")
```

### Community Metrics

```python
from semantica.kg import CommunityDetector

detector = CommunityDetector()
communities = detector.detect_communities(kg, algorithm="louvain")

# Calculate community metrics
metrics = detector.calculate_community_metrics(kg, communities)

print(f"Number of communities: {metrics['num_communities']}")
print(f"Average community size: {metrics['avg_community_size']}")
print(f"Modularity: {metrics['modularity']}")

# Analyze community structure
structure = detector.analyze_community_structure(kg, communities)
print(f"Intra-community edges: {structure['intra_community_edges']}")
print(f"Inter-community edges: {structure['inter_community_edges']}")
```

### Different Community Detection Algorithms

```python
from semantica.kg import CommunityDetector

detector = CommunityDetector()

# Louvain algorithm
louvain_result = detector.detect_communities(kg, algorithm="louvain")

# Leiden algorithm
leiden_result = detector.detect_communities(kg, algorithm="leiden")

# Overlapping communities
overlapping_result = detector.detect_communities(kg, algorithm="overlapping", k=3)
```

## Connectivity Analysis

### Comprehensive Connectivity Analysis

```python
from semantica.kg import analyze_connectivity, ConnectivityAnalyzer

# Using convenience function
result = analyze_connectivity(kg, method="default")

print(f"Number of components: {result['num_components']}")
print(f"Is connected: {result['is_connected']}")
print(f"Density: {result['density']}")
print(f"Average degree: {result['avg_degree']}")

# Using class directly
analyzer = ConnectivityAnalyzer()
result = analyzer.analyze_connectivity(kg)
```

### Connected Components

```python
from semantica.kg import analyze_connectivity

# Find connected components
result = analyze_connectivity(kg, method="components")

print(f"Found {result['num_components']} connected components")
for i, component in enumerate(result["components"]):
    print(f"Component {i}: {len(component)} nodes")
```

### Shortest Paths

```python
from semantica.kg import analyze_connectivity, ConnectivityAnalyzer

# Find shortest path between two nodes
result = analyze_connectivity(
    kg,
    method="paths",
    source="node1",
    target="node2"
)

if result["exists"]:
    print(f"Path: {' -> '.join(result['path'])}")
    print(f"Distance: {result['distance']}")
else:
    print("No path found")

# Using class directly
analyzer = ConnectivityAnalyzer()
paths = analyzer.calculate_shortest_paths(kg, source="node1", target="node2")
```

### Bridge Detection

```python
from semantica.kg import analyze_connectivity

# Identify bridge edges
result = analyze_connectivity(kg, method="bridges")

print(f"Found {result['num_bridges']} bridge edges")
for bridge in result["bridge_edges"]:
    print(f"Bridge: {bridge['source']} -> {bridge['target']}")
```

### Different Connectivity Analysis Types

```python
from semantica.kg import ConnectivityAnalyzer

analyzer = ConnectivityAnalyzer()

# Default comprehensive analysis
connectivity = analyzer.analyze_connectivity(kg)

# Components only
components = analyzer.find_connected_components(kg)

# Path finding
paths = analyzer.calculate_shortest_paths(kg, source="A", target="B")

# Bridge detection
bridges = analyzer.identify_bridges(kg)
```

!!! note "Deduplication"
    Deduplication has been moved to the dedicated `semantica.deduplication` module.
    Please use `semantica.deduplication.DuplicateDetector` and `semantica.deduplication.EntityMerger` for these tasks.

## Temporal Queries

### Time-Point Queries

```python
from semantica.kg import TemporalGraphQuery

# Create query engine and query at specific time
query_engine = TemporalGraphQuery()
result = query_engine.query_at_time(kg, query="", at_time="2024-01-01")

print(f"Entities at time: {result['num_entities']}")
print(f"Relationships at time: {result['num_relationships']}")
```

### Time-Range Queries

```python
from semantica.kg import TemporalGraphQuery

# Query within time range
query_engine = TemporalGraphQuery()
result = query_engine.query_time_range(
    kg,
    query="",
    start_time="2024-01-01",
    end_time="2024-12-31",
    temporal_aggregation="union"
)

print(f"Relationships in range: {result['num_relationships']}")
```

### Temporal Pattern Detection

```python
from semantica.kg import TemporalGraphQuery

# Detect temporal patterns
query_engine = TemporalGraphQuery()
result = query_engine.query_temporal_pattern(kg, pattern="sequence", min_support=2)

print(f"Found {result['num_patterns']} temporal patterns")
```

### Graph Evolution Analysis

```python
from semantica.kg import TemporalGraphQuery

# Analyze graph evolution
query_engine = TemporalGraphQuery()
result = query_engine.analyze_evolution(
    kg,
    start_time="2024-01-01",
    end_time="2024-12-31",
    metrics=["count", "diversity", "stability"]
)

print(f"Relationship count: {result.get('count', 0)}")
print(f"Diversity: {result.get('diversity', 0)}")
print(f"Stability: {result.get('stability', 0)}")
```

### Temporal Path Finding

```python
from semantica.kg import TemporalGraphQuery

query_engine = TemporalGraphQuery()

# Find temporal paths
paths = query_engine.find_temporal_paths(
    kg,
    source="entity1",
    target="entity2",
    start_time="2024-01-01",
    end_time="2024-12-31"
)

print(f"Found {paths['num_paths']} temporal paths")
for path in paths["paths"]:
    print(f"Path: {' -> '.join(path['path'])}")
    print(f"Length: {path['length']}")
```

### Different Temporal Query Types

```python
from semantica.kg import TemporalGraphQuery

query_engine = TemporalGraphQuery()

# Time-point query
result = query_engine.query_at_time(kg, query="", at_time="2024-01-01")

# Time-range query
result = query_engine.query_time_range(kg, query="", start_time="2024-01-01", end_time="2024-12-31")

# Pattern detection
result = query_engine.query_temporal_pattern(kg, pattern="sequence")

# Evolution analysis
result = query_engine.analyze_evolution(kg)
```

## Provenance Tracking

### Tracking Entity Provenance

```python
from semantica.kg import ProvenanceTracker

tracker = ProvenanceTracker()

# Track entity provenance
tracker.track_entity(
    "entity_1",
    source="source_1",
    metadata={"confidence": 0.9, "extraction_method": "ner"}
)

# Track relationship provenance
tracker.track_relationship(
    "rel_1",
    source="source_2",
    metadata={"confidence": 0.85}
)
```

### Retrieving Provenance

```python
from semantica.kg import ProvenanceTracker

tracker = ProvenanceTracker()

# Get all sources for an entity
sources = tracker.get_all_sources("entity_1")
for source in sources:
    print(f"Source: {source['source']}")
    print(f"Timestamp: {source['timestamp']}")
    print(f"Metadata: {source['metadata']}")

# Get complete lineage
lineage = tracker.get_lineage("entity_1")
print(f"First seen: {lineage['first_seen']}")
print(f"Last updated: {lineage['last_updated']}")
print(f"Total sources: {len(lineage['sources'])}")
```

## Using Methods

### Method Functions

```python
from semantica.kg.methods import (
    build_kg,
    analyze_graph,
    resolve_entities,
    validate_graph,
    detect_conflicts,
    calculate_centrality,
    detect_communities,
    analyze_connectivity,
    deduplicate_graph,
    query_temporal
)

# Build knowledge graph
kg = build_kg(sources, method="default")

# Analyze graph
analysis = analyze_graph(kg, method="default")

# Resolve entities
resolved = resolve_entities(entities, method="fuzzy")

# Validate graph
result = validate_graph(kg, method="default")

# Detect conflicts
conflicts = detect_conflicts(kg, method="default")

# Calculate centrality
centrality = calculate_centrality(kg, method="degree")

# Detect communities
communities = detect_communities(kg, method="louvain")

# Analyze connectivity
connectivity = analyze_connectivity(kg, method="default")

# Deduplicate graph
deduplicated = deduplicate_graph(kg, method="default")

# Query temporal
temporal_result = query_temporal(kg, method="time_point", at_time="2024-01-01")
```

### Getting Methods

```python
from semantica.kg.methods import get_kg_method

# Get a specific method
build_method = get_kg_method("build", "default")
if build_method:
    kg = build_method(sources)
```

### Listing Available Methods

```python
from semantica.kg.methods import list_available_methods

# List all available methods
all_methods = list_available_methods()
print("Available methods:")
for task, methods in all_methods.items():
    print(f"  {task}: {methods}")

# List methods for a specific task
build_methods = list_available_methods("build")
print(f"Build methods: {build_methods}")
```

## Using Registry

### Registering Custom Methods

```python
from semantica.kg.registry import method_registry

def custom_build_method(sources, **kwargs):
    """Custom build method."""
    # Your custom implementation
    return {"entities": [], "relationships": [], "metadata": {}}

# Register custom method
method_registry.register("build", "custom_build", custom_build_method)

# Use custom method
from semantica.kg.methods import build_kg
kg = build_kg(sources, method="custom_build")
```

### Unregistering Methods

```python
from semantica.kg.registry import method_registry

# Unregister a method
method_registry.unregister("build", "custom_build")
```

### Listing Registered Methods

```python
from semantica.kg.registry import method_registry

# List all registered methods
all_methods = method_registry.list_all()
print(all_methods)

# List methods for a specific task
build_methods = method_registry.list_all("build")
print(build_methods)
```

## Configuration

### Environment Variables

```bash
# Set KG configuration via environment variables
export KG_MERGE_ENTITIES=true
export KG_RESOLUTION_STRATEGY=fuzzy
export KG_ENABLE_TEMPORAL=false
export KG_TEMPORAL_GRANULARITY=day
export KG_SIMILARITY_THRESHOLD=0.8
```

### Programmatic Configuration

```python
from semantica.kg.config import kg_config

# Set configuration programmatically
kg_config.set("merge_entities", True)
kg_config.set("resolution_strategy", "fuzzy")
kg_config.set("similarity_threshold", 0.8)

# Get configuration
merge_entities = kg_config.get("merge_entities", default=True)
strategy = kg_config.get("resolution_strategy", default="fuzzy")

# Set method-specific configuration
kg_config.set_method_config("build", merge_entities=True, resolve_conflicts=True)

# Get method-specific configuration
build_config = kg_config.get_method_config("build")
```

### Config Files

```yaml
# config.yaml
kg:
  merge_entities: true
  resolution_strategy: fuzzy
  enable_temporal: false
  temporal_granularity: day
  similarity_threshold: 0.8

kg_methods:
  build:
    merge_entities: true
    resolve_conflicts: true
  resolve:
    similarity_threshold: 0.8
```

```python
from semantica.kg.config import KGConfig

# Load from config file
kg_config = KGConfig(config_file="config.yaml")
```

## Advanced Examples

### Complete Knowledge Graph Pipeline

```python
from semantica.kg import (
    GraphBuilder,
    EntityResolver,
    GraphValidator,
    GraphAnalyzer,
    CentralityCalculator,
    CommunityDetector
)

# 1. Build knowledge graph
builder = GraphBuilder(merge_entities=True)
kg = builder.build(sources)

# 2. Resolve entities
resolver = EntityResolver(strategy="fuzzy", similarity_threshold=0.8)
entities = kg["entities"]
resolved_entities = resolver.resolve_entities(entities)
kg["entities"] = resolved_entities

# 3. Validate graph
validator = GraphValidator()
validation = validator.validate(kg)
if not validation.valid:
    print("Validation errors:", validation.errors)
    return

# 4. Analyze graph
analyzer = GraphAnalyzer()
analysis = analyzer.analyze_graph(kg)
print(f"Graph density: {analysis['density']}")
print(f"Average degree: {analysis['avg_degree']}")

# 5. Calculate centrality
centrality_calc = CentralityCalculator()
degree_centrality = centrality_calc.calculate_degree_centrality(kg)
print("Top 5 nodes by degree:")
for ranking in degree_centrality["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")

# 6. Detect communities
community_detector = CommunityDetector()
communities_result = community_detector.detect_communities(kg, algorithm="louvain")
print(f"Found {len(communities_result['communities'])} communities")
```

### Temporal Knowledge Graph Workflow

```python
from semantica.kg import (
    GraphBuilder,
    TemporalGraphQuery,
    TemporalVersionManager
)

# Build temporal knowledge graph
builder = GraphBuilder(
    enable_temporal=True,
    temporal_granularity="day",
    track_history=True
)
temporal_kg = builder.build(sources)

# Query at specific time point
query_engine = TemporalGraphQuery()
result = query_engine.query_at_time(
    temporal_kg,
    query="",
    at_time="2024-06-15"
)

# Query time range
range_result = query_engine.query_time_range(
    temporal_kg,
    query="",
    start_time="2024-01-01",
    end_time="2024-12-31"
)

# Analyze evolution
evolution = query_engine.analyze_evolution(
    temporal_kg,
    start_time="2024-01-01",
    end_time="2024-12-31",
    metrics=["count", "diversity"]
)

# Version management
version_manager = TemporalVersionManager()
version1 = version_manager.create_version(temporal_kg, version_label="v1.0")
version2 = version_manager.create_version(temporal_kg, version_label="v2.0")
comparison = version_manager.compare_versions(version1, version2)
```

### Graph Analytics Workflow

```python
from semantica.kg import (
    GraphAnalyzer,
    CentralityCalculator,
    CommunityDetector,
    ConnectivityAnalyzer
)

# Comprehensive analysis
analyzer = GraphAnalyzer()
analysis = analyzer.analyze_graph(kg)

# Centrality analysis
centrality_calc = CentralityCalculator()
all_centrality = centrality_calc.calculate_all_centrality(kg)

# Community detection
community_detector = CommunityDetector()
communities = community_detector.detect_communities(kg, algorithm="louvain")
metrics = community_detector.calculate_community_metrics(kg, communities)

# Connectivity analysis
connectivity_analyzer = ConnectivityAnalyzer()
connectivity = connectivity_analyzer.analyze_connectivity(kg)
components = connectivity_analyzer.find_connected_components(kg)
bridges = connectivity_analyzer.identify_bridges(kg)
```

### Entity Resolution

```python
from semantica.kg import EntityResolver

# Entity resolution
resolver = EntityResolver(strategy="fuzzy", similarity_threshold=0.8)
resolved = resolver.resolve_entities(kg["entities"])
```

!!! note "Deduplication and Conflict Resolution"
    For deduplication, use `semantica.deduplication.DuplicateDetector` and `semantica.deduplication.EntityMerger`.
    For conflict detection and resolution, use `semantica.conflicts.ConflictDetector` and `semantica.conflicts.ConflictResolver`.

This guide covers the main features and usage patterns of the knowledge graph module. For more detailed information, refer to the module documentation and API reference.

