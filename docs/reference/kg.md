# Knowledge Graph

> **High-level KG construction, management, and analysis system.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } **KG Construction**

    ---

    Build graphs from entities and relationships with automatic merging

-   :material-clock-time-four-outline:{ .lg .middle } **Temporal Graphs**

    ---

    Time-aware edges (`valid_from`, `valid_until`) and temporal queries

-   :material-account-multiple-check:{ .lg .middle } **Entity Resolution**

    ---

    Deduplicate entities using fuzzy matching and semantic similarity

-   :material-alert-decagram:{ .lg .middle } **Conflict Detection**

    ---

    Detect and resolve contradicting facts from multiple sources

-   :material-chart-network:{ .lg .middle } **Graph Analytics**

    ---

    Centrality, Community Detection, and Connectivity analysis

-   :material-history:{ .lg .middle } **Provenance**

    ---

    Track the source and lineage of every node and edge

</div>

!!! tip "When to Use"
    - **KG Building**: The primary module for assembling a KG from extracted data
    - **Data Cleaning**: Merging duplicates and resolving conflicts
    - **Analysis**: Understanding the structure and importance of nodes
    - **Time-Series**: Modeling how the graph evolves over time

---

## ⚙️ Algorithms Used

### Entity Resolution
- **Fuzzy Matching**: Levenshtein/Jaro-Winkler distance for string similarity.
- **Semantic Matching**: Cosine similarity of embeddings.
- **Transitive Merging**: If A=B and B=C, then A=B=C.

### Graph Analytics
- **Centrality**: Degree, Betweenness, Closeness, Eigenvector.
- **Communities**: Louvain, Leiden, K-Clique.
- **Connectivity**: Connected Components, Bridge Detection.

### Temporal Analysis
- **Time-Slicing**: Viewing the graph at a specific point in time.
- **Interval Algebra**: Allen's interval algebra for temporal reasoning (overlaps, during, before).

---

## Main Classes

### GraphBuilder

Constructs the KG from raw data.

**Methods:**

| Method | Description |
|--------|-------------|
| `build(sources)` | Build graph from inputs |
| `merge_entities()` | Run deduplication |

**Example:**

```python
from semantica.kg import GraphBuilder

builder = GraphBuilder(merge_entities=True)
kg = builder.build([source1, source2])
```

### GraphAnalyzer

Runs analytical algorithms.

**Methods:**

| Method | Description |
|--------|-------------|
| `centrality(method)` | Calculate importance |
| `communities(method)` | Find clusters |

### TemporalGraphQuery

Queries time-aware graphs.

**Methods:**

| Method | Description |
|--------|-------------|
| `at_time(timestamp)` | Graph state at T |
| `during(start, end)` | Graph state in interval |

---

## Convenience Functions

```python
from semantica.kg import GraphBuilder, analyze_graph

# Build using GraphBuilder
builder = GraphBuilder(resolve_conflicts=True)
kg = builder.build(sources)

# Analyze
stats = analyze_graph(kg)
print(f"Communities: {stats['communities']}")
```

---

## Configuration

### Environment Variables

```bash
export KG_MERGE_STRATEGY=fuzzy
export KG_TEMPORAL_GRANULARITY=day
export KG_CONFLICT_RESOLUTION=confidence
```

### YAML Configuration

```yaml
kg:
  resolution:
    threshold: 0.9
    strategy: semantic
    
  temporal:
    enabled: true
    default_validity: infinite
```

---

## Integration Examples

### Temporal Analysis Pipeline

```python
from semantica.kg import GraphBuilder, TemporalGraphQuery

# 1. Build Temporal Graph
builder = GraphBuilder(enable_temporal=True)
kg = builder.build(temporal_data)

# 2. Query Evolution
query = TemporalGraphQuery(kg)
snapshot_2020 = query.at_time("2020-01-01")
snapshot_2023 = query.at_time("2023-01-01")

# 3. Compare
diff = snapshot_2023.minus(snapshot_2020)
print(f"New nodes since 2020: {len(diff.nodes)}")
```

---

## Best Practices

1.  **Clean Data First**: Use `EntityResolver` aggressively to prevent "entity explosion" (too many duplicate nodes).
2.  **Use Provenance**: Always track sources (`track_history=True`) to debug where bad data came from.
3.  **Temporal Granularity**: Choose the right granularity (Day vs Second) to balance performance and precision.
4.  **Validate**: Run `GraphValidator` after building to ensure structural integrity.

---

## See Also

- [Graph Store Module](graph_store.md) - Persistence layer
- [Semantic Extract Module](semantic_extract.md) - Data source
- [Visualization Module](visualization.md) - Visualizing the KG
