# Ontology

> **Automated ontology generation, validation, and management system.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-factory:{ .lg .middle } **Automated Generation**

    ---

    6-stage pipeline to generate OWL ontologies from raw data

-   :material-sitemap:{ .lg .middle } **Inference Engine**

    ---

    Infer classes, properties, and hierarchies from entity patterns

-   :material-check-decagram:{ .lg .middle } **Validation**

    ---

    Symbolic reasoning (HermiT/Pellet) for consistency checking

-   :material-recycle:{ .lg .middle } **Reuse Management**

    ---

    Import and align with standard ontologies (FOAF, Schema.org)

-   :material-chart-bar:{ .lg .middle } **Evaluation**

    ---

    Assess ontology quality using coverage, completeness, and granularity metrics

-   :material-file-code:{ .lg .middle } **OWL/RDF Export**

    ---

    Export to Turtle, RDF/XML, and JSON-LD formats

</div>

!!! tip "When to Use"
    - **Schema Design**: When defining the structure of your Knowledge Graph
    - **Data Modeling**: To formalize domain concepts and relationships
    - **Interoperability**: To ensure your data follows standard semantic web practices
    - **Validation**: To enforce constraints on your data

---

## ⚙️ Algorithms Used

### 6-Stage Generation Pipeline
1.  **Semantic Network Parsing**: Extract concepts and patterns from raw entity/relationship data.
2.  **YAML-to-Definition**: Transform patterns into intermediate class definitions.
3.  **Definition-to-Types**: Map definitions to OWL types (`owl:Class`, `owl:ObjectProperty`).
4.  **Hierarchy Generation**: Build taxonomy trees using transitive closure and cycle detection.
5.  **TTL Generation**: Serialize to Turtle format using `rdflib`.
6.  **Symbolic Validation**: Run reasoner to check for logical inconsistencies.

### Inference Algorithms
- **Class Inference**: Clustering entities by type and attribute similarity.
- **Property Inference**: Determining domain/range based on connected entity types.
- **Hierarchy Inference**: `A is_a B` detection based on subset relationships.

### Validation
- **Symbolic Reasoning**: Uses HermiT or Pellet to check satisfiability.
- **Constraint Checking**: Validates cardinality, domain, and range constraints.
- **Hallucination Detection**: LLM-based verification of generated concepts.

---

## Main Classes

### OntologyGenerator

Main entry point for the generation pipeline.

**Methods:**

| Method | Description |
|--------|-------------|
| `generate_ontology(data)` | Run full pipeline |
| `generate_from_schema(schema)` | Generate from explicit schema |

**Example:**

```python
from semantica.ontology import OntologyGenerator

generator = OntologyGenerator(base_uri="http://example.org/onto/")
ontology = generator.generate_ontology({
    "entities": entities,
    "relationships": relationships
})
print(ontology.serialize(format="turtle"))
```

### OntologyValidator

Validates ontology consistency.

**Methods:**

| Method | Description |
|--------|-------------|
| `validate(ontology)` | Run symbolic reasoner |
| `check_constraints(ontology)` | Check structural rules |

### OntologyEvaluator

Scores ontology quality.

**Methods:**

| Method | Description |
|--------|-------------|
| `evaluate(ontology)` | Calculate all metrics |
| `check_competency(questions)` | Verify coverage |

### ReuseManager

Manages external dependencies.

**Methods:**

| Method | Description |
|--------|-------------|
| `import_ontology(uri)` | Load external ontology |
| `align_concepts(source, target)` | Map equivalent classes |

---

## Convenience Functions

```python
from semantica.ontology import generate_ontology, validate_ontology

# Quick generation
onto = generate_ontology(data, method="default")

# Quick validation
is_valid, report = validate_ontology(onto)
```

---

## Configuration

### Environment Variables

```bash
export ONTOLOGY_BASE_URI="http://my-org.com/ontology/"
export ONTOLOGY_REASONER="hermit"
export ONTOLOGY_STRICT_MODE=true
```

### YAML Configuration

```yaml
ontology:
  base_uri: "http://example.org/"
  generation:
    min_class_size: 5
    infer_hierarchy: true
    
  validation:
    reasoner: hermit
    timeout: 60
```

---

## Integration Examples

### Schema-First Knowledge Graph

```python
from semantica.ontology import OntologyGenerator
from semantica.kg import KnowledgeGraph

# 1. Generate Ontology from Sample Data
generator = OntologyGenerator()
ontology = generator.generate_ontology(sample_data)

# 2. Initialize KG with Ontology
kg = KnowledgeGraph(schema=ontology)

# 3. Add Data (Validated against Ontology)
kg.add_entities(full_dataset)  # Will raise error if violates schema
```

---

## Best Practices

1.  **Reuse Standard Ontologies**: Don't reinvent `Person` or `Organization`; import FOAF or Schema.org using `ReuseManager`.
2.  **Validate Early**: Run validation during generation to catch logical errors before populating the graph.
3.  **Use Competency Questions**: Define what questions your ontology should answer and use `OntologyEvaluator` to verify.
4.  **Version Control**: Treat ontologies like code. Use `VersionManager` to track changes.

---

## See Also

- [Knowledge Graph Module](kg.md) - The instance data following the ontology
- [Reasoning Module](reasoning.md) - Uses the ontology for inference
- [Visualization Module](visualization.md) - Visualizing the class hierarchy
