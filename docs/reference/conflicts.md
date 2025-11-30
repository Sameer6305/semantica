# Conflicts

> **Comprehensive conflict detection and resolution system for managing data discrepancies across multiple sources.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-alert-decagram:{ .lg .middle } **Multi-Source Detection**

    ---

    Detect conflicts across values, types, relationships, and temporal data

-   :material-scale-balance:{ .lg .middle } **Resolution Strategies**

    ---

    Resolve using voting, credibility, recency, or confidence scores

-   :material-chart-line:{ .lg .middle } **Conflict Analysis**

    ---

    Analyze patterns, trends, and severity of data discrepancies

-   :material-source-branch:{ .lg .middle } **Source Tracking**

    ---

    Track data provenance and source credibility

-   :material-clipboard-check:{ .lg .middle } **Investigation Guides**

    ---

    Generate automated guides for manual conflict resolution

-   :material-history:{ .lg .middle } **Traceability**

    ---

    Maintain full traceability of resolution decisions

</div>

!!! tip "When to Use"
    - **Data Integration**: When merging data from multiple sources with overlapping entities
    - **Quality Assurance**: To identify inconsistent data in your knowledge graph
    - **Truth Maintenance**: To establish a "single source of truth" from noisy data

---

## ⚙️ Algorithms Used

### Conflict Detection
- **Value Comparison**: Equality checking with type normalization
- **Type Mismatch**: Entity type hierarchy validation
- **Temporal Analysis**: Timestamp comparison for time-based conflicts
- **Logical Consistency**: Rule-based validation (e.g., "Person cannot be Organization")
- **Severity Calculation**: Multi-factor scoring based on:
    - Property importance weights
    - Value difference magnitude
    - Number of conflicting sources

### Conflict Resolution
- **Voting (Majority Rule)**: `max(frequency(values))` using Counter
- **Credibility Weighted**: `Σ(value_i * source_credibility_i) / Σ(source_credibility)`
- **Temporal Selection**: Select value with latest timestamp (`max(timestamp)`)
- **Confidence Selection**: Select value with highest extraction confidence
- **Hybrid Resolution**: Waterfall approach (e.g., Voting -> Credibility -> Recency)

### Analysis & Tracking
- **Pattern Identification**: Frequency analysis of conflict types
- **Credibility Scoring**: Historical accuracy tracking per source
- **Traceability**: Graph-based lineage of values and decisions

---

## Main Classes

### ConflictDetector

Detects conflicts across entities and properties.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `detect_conflicts(entities)` | Detect all conflicts | Multi-pass detection |
| `detect_value_conflicts(entities, prop)` | Check specific property | Value comparison |
| `detect_type_conflicts(entities)` | Check entity types | Hierarchy validation |
| `detect_temporal_conflicts(entities)` | Check timestamps | Time-series analysis |

**Example:**

```python
from semantica.conflicts import ConflictDetector

detector = ConflictDetector()
conflicts = detector.detect_conflicts([
    {"id": "1", "name": "Apple", "source": "doc1"},
    {"id": "1", "name": "Apple Inc.", "source": "doc2"}
])

for conflict in conflicts:
    print(f"Conflict on {conflict.property_name}: {conflict.values}")
```

### ConflictResolver

Resolves detected conflicts using configured strategies.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `resolve_conflicts(conflicts)` | Resolve list of conflicts | Strategy pattern |
| `resolve_by_voting(conflict)` | Majority vote | Frequency counting |
| `resolve_by_credibility(conflict)` | Source credibility | Weighted average |
| `resolve_by_recency(conflict)` | Newest value | Timestamp comparison |

**Example:**

```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver(default_strategy="credibility_weighted")
results = resolver.resolve_conflicts(conflicts)

for result in results:
    print(f"Resolved {result.property}: {result.resolved_value}")
    print(f"Strategy used: {result.strategy}")
```

### SourceTracker

Tracks source information and credibility scores.

**Methods:**

| Method | Description |
|--------|-------------|
| `track_source(entity, source)` | Register source for entity |
| `get_source_credibility(source_id)` | Get current credibility score |
| `update_credibility(source_id, score)` | Update source score |

**Example:**

```python
from semantica.conflicts import SourceTracker

tracker = SourceTracker()
tracker.update_credibility("reliable_source", 0.9)
tracker.update_credibility("noisy_source", 0.4)
```

### InvestigationGuideGenerator

Generates human-readable guides for manual resolution.

**Methods:**

| Method | Description |
|--------|-------------|
| `generate_guide(conflict)` | Create investigation steps |
| `generate_checklist(conflicts)` | Create bulk checklist |

---

## Convenience Functions

```python
from semantica.conflicts import detect_and_resolve

# One-line detection and resolution
conflicts, results = detect_and_resolve(
    entities,
    property_name="revenue",
    resolution_strategy="credibility_weighted"
)
```

---

## Configuration

### Environment Variables

```bash
export CONFLICT_DEFAULT_STRATEGY=voting
export CONFLICT_SIMILARITY_THRESHOLD=0.85
export CONFLICT_AUTO_RESOLVE=true
```

### YAML Configuration

```yaml
conflicts:
  default_strategy: voting
  auto_resolve: true
  
  strategies:
    voting:
      min_votes: 2
    credibility:
      default_score: 0.5
      
  weights:
    name: 1.0
    description: 0.5
    date: 0.8
```

---

## Integration Examples

### Pipeline Integration

```python
from semantica.conflicts import detect_and_resolve
from semantica.ingest import Ingestor

# 1. Ingest from multiple sources
ingestor = Ingestor()
data1 = ingestor.ingest("source1.pdf")
data2 = ingestor.ingest("source2.html")

# 2. Combine entities (assuming same IDs)
combined_entities = data1.entities + data2.entities

# 3. Resolve conflicts
conflicts, resolutions = detect_and_resolve(
    combined_entities,
    resolution_strategy="credibility_weighted",
    source_credibility={
        "source1.pdf": 0.9,
        "source2.html": 0.6
    }
)

# 4. Apply resolutions
for resolution in resolutions:
    print(f"Final value for {resolution.entity_id}: {resolution.resolved_value}")
```

---

## Best Practices

1.  **Define Source Credibility**: Always assign credibility scores to your sources if possible.
2.  **Use Hybrid Strategies**: Voting is good for categorical data, Recency for temporal data.
3.  **Keep Humans in the Loop**: Use `InvestigationGuideGenerator` for high-severity conflicts.
4.  **Normalize First**: Ensure data is normalized (dates, numbers) before conflict detection to avoid false positives.

---

## Troubleshooting

**Issue**: Too many false positives on string fields.
**Solution**: Enable fuzzy matching or increase similarity threshold.

```python
detector = ConflictDetector(
    string_similarity_threshold=0.9,  # Stricter matching
    ignore_case=True
)
```

**Issue**: Resolution favoring wrong source.
**Solution**: Check and adjust source credibility scores.

```python
tracker.update_credibility("bad_source", 0.1)
```

---

## See Also

- [Deduplication Module](deduplication.md) - For merging duplicate entities
- [Normalize Module](normalize.md) - For pre-processing data
- [KG QA Module](kg_qa.md) - For overall graph quality
