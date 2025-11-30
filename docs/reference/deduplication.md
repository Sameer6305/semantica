# Deduplication

> **Advanced entity deduplication and resolution system for maintaining a clean, single-source-of-truth Knowledge Graph.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-content-duplicate:{ .lg .middle } **Duplicate Detection**

    ---

    Identify duplicates using multi-factor similarity metrics

-   :material-set-merge:{ .lg .middle } **Entity Merging**

    ---

    Merge entities with configurable strategies (Keep First, Most Complete, etc.)

-   :material-group:{ .lg .middle } **Clustering**

    ---

    Cluster similar entities for efficient batch processing

-   :material-calculator:{ .lg .middle } **Similarity Metrics**

    ---

    Levenshtein, Jaro-Winkler, Cosine, and Jaccard similarity support

-   :material-history:{ .lg .middle } **Provenance**

    ---

    Preserve data lineage and history during merges

-   :material-scale:{ .lg .middle } **Scalable**

    ---

    Batch processing and blocking for large datasets

</div>

!!! tip "When to Use"
    - **Data Ingestion**: Clean incoming data before adding to the graph
    - **Graph Maintenance**: Periodically clean up existing knowledge graphs
    - **Entity Resolution**: Resolve entities from different sources (e.g., "Apple" vs "Apple Inc.")

---

## ⚙️ Algorithms Used

### Similarity Calculation
- **Levenshtein Distance**: Edit distance for string difference
- **Jaro-Winkler**: String similarity with prefix weighting (good for names)
- **Cosine Similarity**: Vector similarity for embeddings
- **Jaccard Similarity**: Set overlap for properties/relationships
- **Multi-factor Aggregation**: Weighted sum of multiple metrics

### Duplicate Detection
- **Pairwise Comparison**: O(n²) comparison (for small sets)
- **Blocking/Indexing**: Reduce search space for large sets
- **Union-Find**: Disjoint set data structure for grouping duplicates
- **Confidence Scoring**: `0.0 - 1.0` probability score for duplicates

### Clustering
- **Hierarchical Clustering**: Agglomerative bottom-up clustering
- **Connected Components**: Graph-based cluster detection
- **Cluster Quality**: Cohesion and separation metrics

### Entity Merging
- **Strategy Pattern**: Pluggable merge logic
- **Property Union**: Combining unique properties
- **Relationship Merging**: Re-linking relationships to the merged entity

---

## Main Classes

### DuplicateDetector

Identifies potential duplicates in a dataset.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `detect_duplicates(entities)` | Find duplicate pairs | Pairwise/Blocking |
| `detect_duplicate_groups(entities)` | Find clusters of duplicates | Union-Find |

**Example:**

```python
from semantica.deduplication import DuplicateDetector

detector = DuplicateDetector(similarity_threshold=0.85)
duplicates = detector.detect_duplicates(entities)

for group in duplicates:
    print(f"Found group of {len(group)} duplicates")
```

### EntityMerger

Merges duplicate entities into a single canonical entity.

**Methods:**

| Method | Description | Strategy |
|--------|-------------|----------|
| `merge_duplicates(entities)` | Execute merge | Configured Strategy |
| `merge_group(group)` | Merge specific group | Configured Strategy |

**Strategies:**
- `KEEP_FIRST`: Keep the first entity encountered
- `KEEP_MOST_COMPLETE`: Keep entity with most properties
- `KEEP_HIGHEST_CONFIDENCE`: Keep entity with highest confidence score
- `MERGE_ALL`: Create new entity combining all info

**Example:**

```python
from semantica.deduplication import EntityMerger

merger = EntityMerger(strategy="keep_most_complete")
result = merger.merge_duplicates(entities)
```

### SimilarityCalculator

Calculates similarity between entities.

**Methods:**

| Method | Description |
|--------|-------------|
| `calculate(e1, e2)` | Get aggregate score |
| `string_similarity(s1, s2)` | Text comparison |

### ClusterBuilder

Builds clusters for batch processing.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `build_clusters(entities)` | Create clusters | Hierarchical/Graph |

---

## Convenience Functions

```python
from semantica.deduplication import deduplicate

# Detect and merge in one step
result = deduplicate(
    entities,
    similarity_threshold=0.8,
    merge_strategy="keep_most_complete"
)

print(f"Reduced {result['statistics']['reduction']} entities")
```

---

## Configuration

### Environment Variables

```bash
export DEDUP_SIMILARITY_THRESHOLD=0.8
export DEDUP_MERGE_STRATEGY=keep_most_complete
export DEDUP_BLOCKING_ENABLED=true
```

### YAML Configuration

```yaml
deduplication:
  thresholds:
    similarity: 0.8
    confidence: 0.7
    
  weights:
    name: 0.6
    type: 0.2
    attributes: 0.2
    
  blocking:
    enabled: true
    method: "token_blocking"
```

---

## Integration Examples

### Ingestion Pipeline

```python
from semantica.ingest import Ingestor
from semantica.deduplication import deduplicate
from semantica.kg import KnowledgeGraph

# 1. Ingest
ingestor = Ingestor()
raw_entities = ingestor.ingest_batch(files)

# 2. Deduplicate
dedup_result = deduplicate(
    raw_entities,
    similarity_threshold=0.85,
    merge_strategy="merge_all"
)

# 3. Load to KG
kg = KnowledgeGraph()
kg.add_entities(dedup_result['merged_entities'])
```

---

## Best Practices

1.  **Block First**: For >1000 entities, enable blocking to avoid O(n²) performance.
2.  **Tune Thresholds**: Start with 0.85 and adjust based on false positive/negative rates.
3.  **Preserve Provenance**: Keep `preserve_provenance=True` to track where merged data came from.
4.  **Normalize**: Run `normalize` module before deduplication for best results.

---

## Troubleshooting

**Issue**: Merging "Apple" and "Apple Pie" (False Positive).
**Solution**: Increase threshold or use Jaro-Winkler which penalizes prefix mismatches.

```python
detector = DuplicateDetector(
    similarity_method="jaro_winkler",
    similarity_threshold=0.9
)
```

**Issue**: Slow performance on large datasets.
**Solution**: Use `ClusterBuilder` with blocking.

---

## See Also

- [Conflicts Module](conflicts.md) - Handling conflicting values during merge
- [Normalize Module](normalize.md) - Pre-processing for better matching
- [Knowledge Graph Module](kg.md) - Target for deduplicated data
