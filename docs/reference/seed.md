# Seed

> **Seed data management system for initializing Knowledge Graphs from verified sources.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-database-import:{ .lg .middle } **Multi-Source Loading**

    ---

    Load seed data from CSV, JSON, Databases, and APIs

-   :material-graph-outline:{ .lg .middle } **Foundation Graph**

    ---

    Create reliable foundation graphs to bootstrap your KG

-   :material-merge:{ .lg .middle } **Data Integration**

    ---

    Merge seed data with extracted data using configurable strategies

-   :material-check-all:{ .lg .middle } **Validation**

    ---

    Validate seed data quality and schema compliance

-   :material-git:{ .lg .middle } **Versioning**

    ---

    Manage versions of seed data sources

-   :material-export:{ .lg .middle } **Export**

    ---

    Export seed data to standard formats

</div>

!!! tip "When to Use"
    - **Bootstrapping**: When starting a new KG and you have existing structured data (taxonomies, user lists, product catalogs).
    - **Reference Data**: To load immutable reference data (countries, codes, constants).
    - **Testing**: To load consistent test datasets for development.

---

## ⚙️ Algorithms Used

### Data Loading
- **Format Detection**: Auto-detection of CSV delimiters, JSON structure
- **Streaming**: Row-by-row processing for large files
- **Normalization**: Type conversion and encoding handling

### Integration & Merging
- **Seed-First Strategy**: Seed data overrides extracted data (Trust Seed)
- **Extracted-First Strategy**: Extracted data overrides seed (Trust Extraction)
- **Smart Merge**: Property-level merging with conflict resolution
- **ID Matching**: Entity resolution between seed and extracted entities

### Validation
- **Schema Validation**: Template-based structure checking
- **Constraint Checking**: Required field and type validation
- **Consistency Check**: Reference integrity (relationships point to existing entities)

---

## Main Classes

### SeedDataManager

Coordinator for all seed data operations.

**Methods:**

| Method | Description |
|--------|-------------|
| `register_source(name, type, path)` | Add data source |
| `create_foundation_graph()` | Build KG from sources |
| `validate_quality(graph)` | Check data quality |
| `integrate(seed_kg, extracted_kg)` | Merge graphs |

**Example:**

```python
from semantica.seed import SeedDataManager

manager = SeedDataManager()
manager.register_source("countries", "csv", "data/countries.csv")
foundation_kg = manager.create_foundation_graph()
```

### SeedDataSource

Data class defining a source.

**Attributes:**
- `name`: Source identifier
- `type`: `csv`, `json`, `api`, `sql`
- `path`: File path or connection string
- `config`: Parsing options

---

## Convenience Functions

```python
from semantica.seed import register_seed_source, create_foundation

# Quick setup
register_seed_source("products", "json", "products.json")
kg = create_foundation()
```

---

## Configuration

### Environment Variables

```bash
export SEED_DATA_DIR=./data/seed
export SEED_MERGE_STRATEGY=seed_first
```

### YAML Configuration

```yaml
seed:
  sources:
    - name: "employees"
      type: "csv"
      path: "./data/employees.csv"
      
  merge:
    strategy: "seed_first"
    
  validation:
    strict: true
```

---

## Integration Examples

### Bootstrapping a KG

```python
from semantica.seed import SeedDataManager
from semantica.ingest import Ingestor
from semantica.kg import KnowledgeGraph

# 1. Load Foundation (Seed)
seed_manager = SeedDataManager()
seed_manager.register_source("taxonomy", "json", "taxonomy.json")
foundation_kg = seed_manager.create_foundation_graph()

# 2. Ingest New Data
ingestor = Ingestor()
new_data = ingestor.ingest("news_articles.pdf")

# 3. Merge
final_kg = seed_manager.integrate_seed_extracted(
    seed_graph=foundation_kg,
    extracted_data=new_data,
    strategy="seed_first"  # Keep taxonomy strict
)
```

---

## Best Practices

1.  **Trust Seed Data**: Usually, seed data is verified. Use `seed_first` merge strategy.
2.  **Version Control**: Keep seed data files in version control (git).
3.  **Validate Schema**: Ensure seed data matches your target ontology.
4.  **Clean IDs**: Use consistent ID schemes in seed data to facilitate merging.

---

## See Also

- [Ingest Module](ingest.md) - Loading unstructured data
- [Knowledge Graph Module](kg.md) - The target graph structure
- [Deduplication Module](deduplication.md) - Handling duplicates during merge
