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

---

## ⚙️ Algorithms Used

### 6-Stage Generation Pipeline
1.  **Semantic Network Parsing**: Extract concepts and patterns from raw entity/relationship data.
2.  **YAML-to-Definition**: Transform patterns into intermediate class definitions.
3.  **Definition-to-Types**: Map definitions to OWL types (`owl:Class`, `owl:ObjectProperty`).
4.  **Hierarchy Generation**: Build taxonomy trees using transitive closure and cycle detection.
5.  **TTL Generation**: Serialize to Turtle format using `rdflib`.

### Inference Algorithms
- **Class Inference**: Clustering entities by type and attribute similarity.
- **Property Inference**: Determining domain/range based on connected entity types.
- **Hierarchy Inference**: `A is_a B` detection based on subset relationships.

---

## Main Classes

### OntologyEngine

Unified orchestration for generation, inference, validation, OWL export, and evaluation.

**Methods:**

| Method | Description |
|--------|-------------|
| `from_data(data, **options)` | Generate ontology from structured data |
| `from_text(text, provider=None, model=None, **options)` | LLM-based generation from text |
| `validate(ontology, **options)` | Validate ontology consistency |
| `infer_classes(entities, **options)` | Infer classes from entities |
| `infer_properties(entities, relationships, classes, **options)` | Infer properties |
| `evaluate(ontology, **options)` | Evaluate ontology quality |
| `to_owl(ontology, format="turtle")` | Export OWL/RDF serialization |
| `export_owl(ontology, path, format="turtle")` | Save OWL to file |

**Quick Start:**

```python
from semantica.ontology import OntologyEngine

engine = OntologyEngine(base_uri="https://example.org/ontology/")

data = {"entities": entities, "relationships": relationships}
ontology = engine.from_data(data, name="MyOntology")

turtle = engine.to_owl(ontology, format="turtle")
```

### LLMOntologyGenerator

LLM-based ontology generation with multi-provider support (`openai`, `groq`, `deepseek`, `huggingface_llm`).

**Example:**

```python
from semantica.ontology import OntologyEngine

text = "Acme Corp. hired Alice in 2024. Alice works for Acme."
engine = OntologyEngine()

ontology = engine.from_text(
    text,
    provider="deepseek",
    model="deepseek-chat",
    name="EmploymentOntology",
    base_uri="https://example.org/employment/",
)
```

Environment variables:

```bash
export OPENAI_API_KEY=...
export GROQ_API_KEY=...
export DEEPSEEK_API_KEY=...
```

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

### OntologyEvaluator

Scores ontology quality.

**Methods:**

| Method | Description |
|--------|-------------|
| `evaluate_ontology(ontology)` | Calculate evaluation metrics |
| `calculate_coverage(ontology, questions)` | Verify coverage |

### ReuseManager

Manages external dependencies.

**Methods:**

| Method | Description |
|--------|-------------|
| `import_external_ontology(uri, ontology)` | Load and merge external ontology |
| `evaluate_alignment(uri, ontology)` | Assess alignment and compatibility |

---

## Unified Engine Examples

```python
from semantica.ontology import OntologyEngine

engine = OntologyEngine(base_uri="https://example.org/ontology/")

# Generate
ontology = engine.from_data({
    "entities": entities,
    "relationships": relationships,
})

# Validate
result = engine.validate(ontology, reasoner="hermit")
print("valid=", result.valid, "consistent=", result.consistent)

# Export
turtle = engine.to_owl(ontology, format="turtle")
```

---

## Configuration

### Environment Variables

```bash
export ONTOLOGY_BASE_URI="http://my-org.com/ontology/"
export ONTOLOGY_STRICT_MODE=true
```

### YAML Configuration

```yaml
ontology:
  base_uri: "http://example.org/"
  generation:
    min_class_size: 5
    infer_hierarchy: true
```

---

## Integration Examples

### Schema-First Knowledge Graph

```python
from semantica.ontology import OntologyEngine
from semantica.kg import KnowledgeGraph

# 1. Generate Ontology from Sample Data
engine = OntologyEngine()
ontology = engine.from_data(sample_data)

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

## Cookbook

- [Ontology](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/14_Ontology.ipynb)
- [Unstructured to Ontology](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/12_Unstructured_to_Ontology.ipynb)
