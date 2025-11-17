# Knowledge Graph Module Usage Guide

This guide demonstrates how to use the knowledge graph module for building, analyzing, validating, and managing knowledge graphs, including temporal knowledge graphs, entity resolution, conflict detection, and graph analytics.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Knowledge Graph Building](#knowledge-graph-building)
3. [Graph Analysis](#graph-analysis)
4. [Entity Resolution](#entity-resolution)
5. [Graph Validation](#graph-validation)
6. [Conflict Detection](#conflict-detection)
7. [Centrality Calculation](#centrality-calculation)
8. [Community Detection](#community-detection)
9. [Connectivity Analysis](#connectivity-analysis)
10. [Deduplication](#deduplication)
11. [Temporal Queries](#temporal-queries)
12. [Provenance Tracking](#provenance-tracking)
13. [Using Methods](#using-methods)
14. [Using Registry](#using-registry)
15. [Configuration](#configuration)
16. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using the Convenience Function

```python
from semantica.kg import build

# Build knowledge graph from sources
sources = [
    {
        "entities": [
            {"id": "1", "name": "Alice", "type": "Person"},
            {"id": "2", "name": "Bob", "type": "Person"}
        ],
        "relationships": [
            {"source": "1", "target": "2", "type": "knows"}
        ]
    }
]

kg = build(
    sources=sources,
    merge_entities=True,
    resolve_conflicts=True
)

print(f"Built graph with {kg['metadata']['num_entities']} entities")
print(f"Relationships: {kg['metadata']['num_relationships']}")
```

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
from semantica.kg import build_kg, GraphBuilder

# Using convenience function
kg = build_kg(
    sources=sources,
    method="default",
    merge_entities=True,
    resolve_conflicts=True
)

# Using class directly
builder = GraphBuilder(
    merge_entities=True,
    entity_resolution_strategy="fuzzy",
    resolve_conflicts=True,
    enable_temporal=False
)

kg = builder.build(sources)
```

### Temporal Knowledge Graph Building

```python
from semantica.kg import build_kg

# Build temporal knowledge graph
temporal_kg = build_kg(
    sources=sources,
    method="temporal",
    enable_temporal=True,
    temporal_granularity="day",
    track_history=True,
    version_snapshots=True
)

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

### Using Build Methods

```python
from semantica.kg.methods import build_kg

# Default building
kg = build_kg(sources, method="default")

# Temporal building
temporal_kg = build_kg(sources, method="temporal", enable_temporal=True)

# Incremental building
kg = build_kg(sources, method="incremental")
```

## Graph Analysis

### Comprehensive Analysis

```python
from semantica.kg import analyze_graph, GraphAnalyzer

# Using convenience function
analysis = analyze_graph(kg, method="default")

print(f"Nodes: {analysis['num_nodes']}")
print(f"Edges: {analysis['num_edges']}")
print(f"Density: {analysis['density']}")

# Using class directly
analyzer = GraphAnalyzer()
analysis = analyzer.analyze_graph(kg)
```

### Centrality-Focused Analysis

```python
from semantica.kg import analyze_graph

# Focus on centrality metrics
analysis = analyze_graph(kg, method="centrality")

# Access centrality results
if "centrality" in analysis:
    degree_centrality = analysis["centrality"].get("degree", {})
    print("Top nodes by degree centrality:")
    for node, score in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {node}: {score}")
```

### Community-Focused Analysis

```python
from semantica.kg import analyze_graph

# Focus on community detection
analysis = analyze_graph(kg, method="community")

# Access community results
if "communities" in analysis:
    communities = analysis["communities"]
    print(f"Found {len(communities)} communities")
    for i, community in enumerate(communities):
        print(f"Community {i}: {len(community)} nodes")
```

### Using Analysis Methods

```python
from semantica.kg.methods import analyze_graph

# Default analysis
analysis = analyze_graph(kg, method="default")

# Centrality analysis
analysis = analyze_graph(kg, method="centrality")

# Community analysis
analysis = analyze_graph(kg, method="community")

# Connectivity analysis
analysis = analyze_graph(kg, method="connectivity")
```

## Entity Resolution

### Fuzzy Matching Resolution

```python
from semantica.kg import resolve_entities, EntityResolver

# Using convenience function
entities = [
    {"id": "1", "name": "Apple Inc.", "type": "Company"},
    {"id": "2", "name": "Apple", "type": "Company"},
    {"id": "3", "name": "Microsoft", "type": "Company"}
]

resolved = resolve_entities(entities, method="fuzzy", similarity_threshold=0.8)

print(f"Original: {len(entities)} entities")
print(f"Resolved: {len(resolved)} entities")

# Using class directly
resolver = EntityResolver(strategy="fuzzy", similarity_threshold=0.8)
resolved = resolver.resolve_entities(entities)
```

### Exact Matching Resolution

```python
from semantica.kg import resolve_entities

# Exact string matching
resolved = resolve_entities(entities, method="exact")
```

### Semantic Matching Resolution

```python
from semantica.kg import resolve_entities

# Semantic similarity matching
resolved = resolve_entities(entities, method="semantic", similarity_threshold=0.9)
```

### Using Resolution Methods

```python
from semantica.kg.methods import resolve_entities

# Fuzzy matching
resolved = resolve_entities(entities, method="fuzzy")

# Exact matching
resolved = resolve_entities(entities, method="exact")

# Semantic matching
resolved = resolve_entities(entities, method="semantic")
```

## Graph Validation

### Comprehensive Validation

```python
from semantica.kg import validate_graph, GraphValidator

# Using convenience function
result = validate_graph(kg, method="default")

if result.valid:
    print("Graph is valid!")
else:
    print(f"Found {len(result.errors)} errors:")
    for error in result.errors:
        print(f"  - {error}")
    
    print(f"Found {len(result.warnings)} warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")

# Using class directly
validator = GraphValidator()
result = validator.validate(kg)
```

### Structure-Only Validation

```python
from semantica.kg import validate_graph

# Validate structure only
result = validate_graph(kg, method="structure")
```

### Consistency Checking

```python
from semantica.kg import validate_graph, GraphValidator

# Check consistency only
is_consistent = validate_graph(kg, method="consistency")

# Using class directly
validator = GraphValidator()
is_consistent = validator.check_consistency(kg)
```

### Using Validation Methods

```python
from semantica.kg.methods import validate_graph

# Default validation
result = validate_graph(kg, method="default")

# Structure validation
result = validate_graph(kg, method="structure")

# Consistency check
is_consistent = validate_graph(kg, method="consistency")
```

## Conflict Detection

### Comprehensive Conflict Detection

```python
from semantica.kg import detect_conflicts, ConflictDetector

# Using convenience function
conflicts = detect_conflicts(kg, method="default")

print(f"Found {len(conflicts)} conflicts")
for conflict in conflicts:
    print(f"Type: {conflict['type']}")
    print(f"Property: {conflict['property']}")
    print(f"Conflicting values: {conflict['conflicting_values']}")

# Using class directly
detector = ConflictDetector()
conflicts = detector.detect_conflicts(kg)
```

### Value Conflict Detection

```python
from semantica.kg import detect_conflicts

# Detect value conflicts only
value_conflicts = detect_conflicts(kg, method="value")
```

### Relationship Conflict Detection

```python
from semantica.kg import detect_conflicts

# Detect relationship conflicts only
rel_conflicts = detect_conflicts(kg, method="relationship")
```

### Conflict Resolution

```python
from semantica.kg import ConflictDetector

detector = ConflictDetector()

# Detect conflicts
conflicts = detector.detect_conflicts(kg)

# Resolve conflicts
resolution = detector.resolve_conflicts(conflicts, strategy="highest_confidence")

print(f"Resolved: {resolution['resolved_count']}")
print(f"Unresolved: {resolution['unresolved_count']}")
```

### Using Conflict Detection Methods

```python
from semantica.kg.methods import detect_conflicts

# Default detection
conflicts = detect_conflicts(kg, method="default")

# Value conflicts
conflicts = detect_conflicts(kg, method="value")

# Relationship conflicts
conflicts = detect_conflicts(kg, method="relationship")
```

## Centrality Calculation

### Degree Centrality

```python
from semantica.kg import calculate_centrality, CentralityCalculator

# Using convenience function
result = calculate_centrality(kg, method="degree")

print("Top nodes by degree centrality:")
for ranking in result["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")

# Using class directly
calculator = CentralityCalculator()
result = calculator.calculate_degree_centrality(kg)
```

### Betweenness Centrality

```python
from semantica.kg import calculate_centrality

# Calculate betweenness centrality
result = calculate_centrality(kg, method="betweenness")

print("Top nodes by betweenness centrality:")
for ranking in result["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")
```

### Closeness Centrality

```python
from semantica.kg import calculate_centrality

# Calculate closeness centrality
result = calculate_centrality(kg, method="closeness")

print("Top nodes by closeness centrality:")
for ranking in result["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")
```

### Eigenvector Centrality

```python
from semantica.kg import calculate_centrality

# Calculate eigenvector centrality
result = calculate_centrality(kg, method="eigenvector")

print("Top nodes by eigenvector centrality:")
for ranking in result["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")
```

### All Centrality Measures

```python
from semantica.kg import calculate_centrality

# Calculate all centrality measures
result = calculate_centrality(kg, method="all")

for measure_type, measure_result in result["centrality_measures"].items():
    print(f"\n{measure_type.upper()} Centrality:")
    for ranking in measure_result["rankings"][:3]:
        print(f"  {ranking['node']}: {ranking['score']}")
```

### Using Centrality Methods

```python
from semantica.kg.methods import calculate_centrality

# Degree centrality
degree = calculate_centrality(kg, method="degree")

# Betweenness centrality
betweenness = calculate_centrality(kg, method="betweenness")

# Closeness centrality
closeness = calculate_centrality(kg, method="closeness")

# Eigenvector centrality
eigenvector = calculate_centrality(kg, method="eigenvector")

# All measures
all_centrality = calculate_centrality(kg, method="all")
```

## Community Detection

### Louvain Algorithm

```python
from semantica.kg import detect_communities, CommunityDetector

# Using convenience function
result = detect_communities(kg, method="louvain")

print(f"Found {len(result['communities'])} communities")
print(f"Modularity: {result['modularity']}")

for i, community in enumerate(result["communities"]):
    print(f"Community {i}: {len(community)} nodes")

# Using class directly
detector = CommunityDetector()
result = detector.detect_communities(kg, algorithm="louvain")
```

### Leiden Algorithm

```python
from semantica.kg import detect_communities

# Detect communities using Leiden algorithm
result = detect_communities(kg, method="leiden", resolution=1.0)

print(f"Found {len(result['communities'])} communities")
```

### Overlapping Communities

```python
from semantica.kg import detect_communities

# Detect overlapping communities
result = detect_communities(kg, method="overlapping", k=3)

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

### Using Community Detection Methods

```python
from semantica.kg.methods import detect_communities

# Louvain algorithm
communities = detect_communities(kg, method="louvain")

# Leiden algorithm
communities = detect_communities(kg, method="leiden")

# Overlapping communities
communities = detect_communities(kg, method="overlapping", k=3)
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

### Using Connectivity Methods

```python
from semantica.kg.methods import analyze_connectivity

# Default analysis
connectivity = analyze_connectivity(kg, method="default")

# Components only
components = analyze_connectivity(kg, method="components")

# Path finding
paths = analyze_connectivity(kg, method="paths", source="A", target="B")

# Bridge detection
bridges = analyze_connectivity(kg, method="bridges")
```

## Deduplication

### Entity Deduplication

```python
from semantica.kg import deduplicate_graph, Deduplicator

# Using convenience function
deduplicated = deduplicate_graph(kg, method="entities")

print(f"Original entities: {len(kg['entities'])}")
print(f"Deduplicated entities: {len(deduplicated['entities'])}")

# Using class directly
deduplicator = Deduplicator()
duplicate_groups = deduplicator.find_duplicates(kg["entities"])
merged_entities = deduplicator.merge_duplicates(duplicate_groups)
```

### Finding Duplicates

```python
from semantica.kg import Deduplicator

deduplicator = Deduplicator()

# Find duplicate groups
duplicate_groups = deduplicator.find_duplicates(kg["entities"])

print(f"Found {len(duplicate_groups)} duplicate groups")
for i, group in enumerate(duplicate_groups):
    print(f"Group {i}: {len(group)} duplicate entities")
```

### Merging Duplicates

```python
from semantica.kg import Deduplicator

deduplicator = Deduplicator()

# Find and merge duplicates
duplicate_groups = deduplicator.find_duplicates(kg["entities"])
merged_entities = deduplicator.merge_duplicates(duplicate_groups)

print(f"Merged {len(duplicate_groups)} groups into {len(merged_entities)} entities")
```

### Using Deduplication Methods

```python
from semantica.kg.methods import deduplicate_graph

# Default deduplication
deduplicated = deduplicate_graph(kg, method="default")

# Entity deduplication only
deduplicated = deduplicate_graph(kg, method="entities")
```

## Temporal Queries

### Time-Point Queries

```python
from semantica.kg import query_temporal, TemporalGraphQuery

# Using convenience function
result = query_temporal(
    kg,
    query="",
    method="time_point",
    at_time="2024-01-01"
)

print(f"Entities at time: {result['num_entities']}")
print(f"Relationships at time: {result['num_relationships']}")

# Using class directly
query_engine = TemporalGraphQuery()
result = query_engine.query_at_time(kg, "", at_time="2024-01-01")
```

### Time-Range Queries

```python
from semantica.kg import query_temporal

# Query within time range
result = query_temporal(
    kg,
    query="",
    method="time_range",
    start_time="2024-01-01",
    end_time="2024-12-31",
    temporal_aggregation="union"
)

print(f"Relationships in range: {result['num_relationships']}")
```

### Temporal Pattern Detection

```python
from semantica.kg import query_temporal

# Detect temporal patterns
result = query_temporal(
    kg,
    query="",
    method="pattern",
    pattern="sequence",
    min_support=2
)

print(f"Found {result['num_patterns']} temporal patterns")
```

### Graph Evolution Analysis

```python
from semantica.kg import query_temporal

# Analyze graph evolution
result = query_temporal(
    kg,
    query="",
    method="evolution",
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

### Using Temporal Query Methods

```python
from semantica.kg.methods import query_temporal

# Time-point query
result = query_temporal(kg, at_time="2024-01-01", method="time_point")

# Time-range query
result = query_temporal(kg, start_time="2024-01-01", end_time="2024-12-31", method="time_range")

# Pattern detection
result = query_temporal(kg, pattern="sequence", method="pattern")

# Evolution analysis
result = query_temporal(kg, method="evolution")
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
    build_kg,
    resolve_entities,
    validate_graph,
    detect_conflicts,
    analyze_graph,
    calculate_centrality,
    detect_communities
)

# 1. Build knowledge graph
kg = build_kg(sources, method="default")

# 2. Resolve entities
entities = kg["entities"]
resolved_entities = resolve_entities(entities, method="fuzzy")
kg["entities"] = resolved_entities

# 3. Validate graph
validation = validate_graph(kg, method="default")
if not validation.valid:
    print("Validation errors:", validation.errors)
    return

# 4. Detect conflicts
conflicts = detect_conflicts(kg, method="default")
if conflicts:
    print(f"Found {len(conflicts)} conflicts")
    # Resolve conflicts...

# 5. Analyze graph
analysis = analyze_graph(kg, method="default")
print(f"Graph density: {analysis['density']}")
print(f"Average degree: {analysis['avg_degree']}")

# 6. Calculate centrality
centrality = calculate_centrality(kg, method="degree")
print("Top 5 nodes by degree:")
for ranking in centrality["rankings"][:5]:
    print(f"  {ranking['node']}: {ranking['score']}")

# 7. Detect communities
communities = detect_communities(kg, method="louvain")
print(f"Found {len(communities['communities'])} communities")
```

### Temporal Knowledge Graph Workflow

```python
from semantica.kg import (
    build_kg,
    query_temporal,
    TemporalVersionManager
)

# Build temporal knowledge graph
temporal_kg = build_kg(
    sources,
    method="temporal",
    enable_temporal=True,
    temporal_granularity="day",
    track_history=True
)

# Query at specific time point
result = query_temporal(
    temporal_kg,
    method="time_point",
    at_time="2024-06-15"
)

# Query time range
range_result = query_temporal(
    temporal_kg,
    method="time_range",
    start_time="2024-01-01",
    end_time="2024-12-31"
)

# Analyze evolution
evolution = query_temporal(
    temporal_kg,
    method="evolution",
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

### Entity Resolution and Deduplication

```python
from semantica.kg import (
    EntityResolver,
    Deduplicator,
    deduplicate_graph
)

# Entity resolution
resolver = EntityResolver(strategy="fuzzy", similarity_threshold=0.8)
resolved = resolver.resolve_entities(kg["entities"])

# Deduplication
deduplicator = Deduplicator()
duplicate_groups = deduplicator.find_duplicates(kg["entities"])
merged_entities = deduplicator.merge_duplicates(duplicate_groups)

# Or use convenience function
deduplicated_kg = deduplicate_graph(kg, method="entities")
```

### Conflict Detection and Resolution

```python
from semantica.kg import ConflictDetector

detector = ConflictDetector()

# Detect conflicts
conflicts = detector.detect_conflicts(kg)

# Resolve conflicts
resolution = detector.resolve_conflicts(conflicts, strategy="highest_confidence")

# Apply resolutions
for resolved_conflict in resolution["resolved"]:
    conflict = resolved_conflict["conflict"]
    resolved_value = resolved_conflict["resolution"]
    # Apply resolution to graph...
```

This guide covers the main features and usage patterns of the knowledge graph module. For more detailed information, refer to the module documentation and API reference.

