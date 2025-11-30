# Semantic Extract

> **Advanced information extraction system for Entities, Relations, Events, and Triples.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-account-search:{ .lg .middle } **NER**

    ---

    Extract Named Entities (Person, Org, Loc) with confidence scores

-   :material-relation-one-to-one:{ .lg .middle } **Relation Extraction**

    ---

    Identify relationships between entities (e.g., `founded_by`, `located_in`)

-   :material-calendar-clock:{ .lg .middle } **Event Detection**

    ---

    Detect events with temporal information and participants

-   :material-format-quote-close:{ .lg .middle } **Coreference**

    ---

    Resolve pronouns ("he", "it") to their entity references

-   :material-share-variant:{ .lg .middle } **Triple Extraction**

    ---

    Extract Subject-Predicate-Object triples for Knowledge Graphs

-   :material-robot:{ .lg .middle } **LLM Enhancement**

    ---

    Use LLMs to improve extraction quality and handle complex schemas

</div>

!!! tip "When to Use"
    - **KG Construction**: Converting unstructured text into structured graph data
    - **Text Analysis**: Identifying key actors and events in documents
    - **Search Indexing**: Extracting metadata for faceted search
    - **Data Enrichment**: Adding semantic tags to content

---

## ⚙️ Algorithms Used

### Named Entity Recognition (NER)
- **Transformer Models**: BERT/RoBERTa for token classification.
- **Regex Patterns**: Pattern matching for specific formats (Emails, IDs).
- **LLM Prompting**: Zero-shot extraction for custom entity types.

### Relation Extraction
- **Dependency Parsing**: Analyzing grammatical structure to find subject-verb-object paths.
- **Joint Extraction**: Extracting entities and relations simultaneously.
- **Semantic Role Labeling**: Identifying "Who did What to Whom".

### Coreference Resolution
- **Mention Detection**: Finding all potential references (nouns, pronouns).
- **Clustering**: Grouping mentions that refer to the same real-world entity.
- **Pronoun Resolution**: Mapping pronouns to the most likely antecedent.

### Triple Extraction
- **OpenIE**: Open Information Extraction for arbitrary relation strings.
- **Schema-Based**: Mapping extracted relations to a predefined ontology.
- **Reification**: Handling complex relations (time, location) by creating event nodes.

---

## Main Classes

### NamedEntityRecognizer

Coordinator for entity extraction.

**Methods:**

| Method | Description |
|--------|-------------|
| `extract_entities(text)` | Get list of entities |
| `add_custom_pattern(pattern)` | Add regex rule |

**Example:**

```python
from semantica.semantic_extract import NamedEntityRecognizer

ner = NamedEntityRecognizer()
entities = ner.extract_entities("Elon Musk leads SpaceX.")
# [Entity(text="Elon Musk", label="PERSON"), Entity(text="SpaceX", label="ORG")]
```

### RelationExtractor

Extracts relationships between entities.

**Methods:**

| Method | Description |
|--------|-------------|
| `extract_relations(text, entities)` | Find links |

**Example:**

```python
from semantica.semantic_extract import RelationExtractor

re = RelationExtractor()
relations = re.extract_relations(text, entities)
# [Relation(source="Elon Musk", target="SpaceX", type="leads")]
```

### EventDetector

Identifies events.

**Methods:**

| Method | Description |
|--------|-------------|
| `detect_events(text)` | Find events |

### TripleExtractor

Extracts RDF triples.

**Methods:**

| Method | Description |
|--------|-------------|
| `extract_triples(text)` | Get (S, P, O) tuples |

---

## Convenience Functions

```python
from semantica.semantic_extract import build

# All-in-one extraction
result = build(
    "Apple released the iPhone in 2007.",
    extract_entities=True,
    extract_relations=True,
    extract_events=True
)

print(result['triples'])
```

---

## Configuration

### Environment Variables

```bash
export NER_MODEL=dslim/bert-base-NER
export RELATION_MODEL=semantica/rel-extract-v1
export EXTRACT_CONFIDENCE_THRESHOLD=0.7
```

### YAML Configuration

```yaml
semantic_extract:
  ner:
    model: dslim/bert-base-NER
    min_confidence: 0.7
    
  relations:
    max_distance: 50 # tokens
    
  coreference:
    enabled: true
```

---

## Integration Examples

### KG Population Pipeline

```python
from semantica.semantic_extract import build
from semantica.kg import KnowledgeGraph

# 1. Extract
text = "Google was founded by Larry Page and Sergey Brin."
data = build(text, extract_triples=True)

# 2. Populate KG
kg = KnowledgeGraph()
for triple in data['triples']:
    kg.add_triple(
        subject=triple.subject,
        predicate=triple.predicate,
        object=triple.object
    )
```

---

## Best Practices

1.  **Resolve Coreferences**: Always run coreference resolution *before* relation extraction to link "He" to "John Doe".
2.  **Filter Low Confidence**: Set a confidence threshold (e.g., 0.7) to reduce noise.
3.  **Use Custom Patterns**: For domain-specific IDs (e.g., "Invoice #123"), regex is faster and more accurate than ML.
4.  **Batch Processing**: Use batch methods when processing large corpora.

---

## See Also

- [Parse Module](parse.md) - Prepares text for extraction
- [Ontology Module](ontology.md) - Defines the schema for extraction
- [Knowledge Graph Module](kg.md) - Stores the extracted data
