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

-   :material-graph:{ .lg .middle } **Semantic Networks**

    ---

    Build structured networks with nodes and edges from text

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

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `methods` | list | `["spacy"]` | Extraction methods to use |
| `confidence_threshold` | float | `0.5` | Minimum confidence score |
| `merge_overlapping` | bool | `True` | Merge overlapping entities |
| `include_standard_types` | bool | `True` | Include Person, Org, Location |

**Methods:**

| Method | Description |
|--------|-------------|
| `extract_entities(text)` | Get list of entities |
| `add_custom_pattern(pattern)` | Add regex rule |

**Example:**

```python
from semantica.semantic_extract import NamedEntityRecognizer

# Basic usage
ner = NamedEntityRecognizer()
entities = ner.extract_entities("Elon Musk leads SpaceX.")
# [Entity(text="Elon Musk", label="PERSON"), Entity(text="SpaceX", label="ORG")]

# With configuration
ner = NamedEntityRecognizer(
    methods=["spacy", "rule-based"],
    confidence_threshold=0.7,
    merge_overlapping=True
)
entities = ner.extract_entities("Apple Inc. was founded in 1976.")
```

### NERExtractor

Core entity extraction implementation used by notebooks and lower-level integrations.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str or list | `"ml"` | Method(s): "ml", "llm", "pattern", "regex", "huggingface" |
| `**config` | dict | `{}` | Method-specific config (e.g., `model`, `provider`) |

**Methods:**

| Method | Description |
|--------|-------------|
| `extract(text)` | Alias for `extract_entities`. Get list of entities. |
| `extract_entities(text)` | Get list of entities |

**Example:**

```python
from semantica.semantic_extract import NERExtractor

# 1. ML (spaCy) - Default
extractor = NERExtractor(method="ml", model="en_core_web_trf")
entities = extractor.extract("Elon Musk leads SpaceX.")

# 2. LLM (OpenAI/Gemini/etc)
extractor = NERExtractor(
    method="llm", 
    provider="openai", 
    model="gpt-4",
    temperature=0.0
)

# 3. Regex with custom patterns
patterns = {"CODE": r"[A-Z]{3}-\d{3}"}
extractor = NERExtractor(method="regex", patterns=patterns)

# 4. Ensemble (Multiple methods)
extractor = NERExtractor(method=["ml", "llm"], ensemble_voting=True)
```

### RelationExtractor

Extracts relationships between entities.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `relation_types` | list | `None` | Specific relation types to extract |
| `bidirectional` | bool | `False` | Extract bidirectional relations |
| `confidence_threshold` | float | `0.6` | Minimum confidence score |
| `max_distance` | int | `50` | Max token distance between entities |

**Methods:**

| Method | Description |
|--------|-------------|
| `extract(text, entities)` | Alias for `extract_relations`. Find links. |
| `extract_relations(text, entities)` | Find links |

**Example:**

```python
from semantica.semantic_extract import RelationExtractor, NamedEntityRecognizer

# First extract entities
ner = NamedEntityRecognizer()
text = "Elon Musk founded SpaceX in 2002."
entities = ner.extract_entities(text)

# Basic relation extraction
rel_extractor = RelationExtractor()
relations = rel_extractor.extract(text, entities=entities)
# [Relation(source="Elon Musk", target="SpaceX", type="founded")]

# With configuration
rel_extractor = RelationExtractor(
    relation_types=["founded", "leads", "works_at"],
    confidence_threshold=0.7,
    bidirectional=False
)
relations = rel_extractor.extract(text, entities=entities)
```

### CoreferenceResolver

Resolves pronoun references and entity coreferences.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str or list | `None` | Underlying NER method(s) |
| `**config` | dict | `{}` | Configuration for NER method |

**Methods:**

| Method | Description |
|--------|-------------|
| `resolve(text)` | Alias for `resolve_coreferences`. Get coreference chains. |
| `resolve_coreferences(text)` | Get coreference chains |
| `resolve_pronouns(text)` | Resolve pronouns to entities |

**Example:**

```python
from semantica.semantic_extract import CoreferenceResolver

resolver = CoreferenceResolver()
text = "Steve Jobs founded Apple. He was the CEO."

# Resolve references
chains = resolver.resolve(text)
# [CoreferenceChain(mentions=["Steve Jobs", "He"], representative="Steve Jobs")]
```

### EventDetector

Identifies events with temporal information and participants.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `event_types` | list | `None` | Specific event types to detect |
| `extract_participants` | bool | `True` | Extract event participants |
| `extract_location` | bool | `True` | Extract event locations |
| `extract_time` | bool | `True` | Extract temporal information |

**Methods:**

| Method | Description |
|--------|-------------|
| `detect_events(text)` | Find events |

**Example:**

```python
from semantica.semantic_extract import EventDetector

detector = EventDetector(
    event_types=["launch", "acquisition", "announcement"],
    extract_participants=True,
    extract_time=True
)
events = detector.detect_events("SpaceX launched Starship on March 14, 2024.")
```

### TripleExtractor

Extracts RDF triples (Subject-Predicate-Object).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_temporal` | bool | `False` | Include time information |
| `include_provenance` | bool | `False` | Track source sentences |
| `method` | str | `"pattern"` | Extraction method ("pattern", "rules", "huggingface", "llm") |

**Methods:**

| Method | Description |
|--------|-------------|
| `extract_triples(text)` | Get (S, P, O) tuples |

**Example:**

```python
from semantica.semantic_extract import TripleExtractor

extractor = TripleExtractor(
    include_temporal=True,
    include_provenance=True
)
triples = extractor.extract_triples("Steve Jobs founded Apple in 1976.")
# [Triple(subject="Steve Jobs", predicate="founded", object="Apple", temporal="1976")]
```

### SemanticNetworkExtractor

Extracts structured semantic networks with nodes and edges.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ner_method` | str | `None` | Method for node extraction |
| `relation_method` | str | `None` | Method for edge extraction |
| `**config` | dict | `{}` | Configuration for underlying extractors |

**Methods:**

| Method | Description |
|--------|-------------|
| `extract_network(text)` | Build network from text |
| `extract(text)` | Alias for `extract_network` |
| `export_to_yaml(network, path)` | Save network to YAML |

**Example:**

```python
from semantica.semantic_extract import SemanticNetworkExtractor

extractor = SemanticNetworkExtractor()
network = extractor.extract("Apple Inc. is located in Cupertino.")

# Analyze network
print(f"Nodes: {len(network.nodes)}")
print(f"Edges: {len(network.edges)}")
```

### LLMEnhancer

Enhances extraction results using Large Language Models.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | str | `"openai"` | LLM provider ("openai", "gemini", "anthropic", etc.) |
| `**config` | dict | `{}` | Model config (model name, api_key, etc.) |

**Methods:**

| Method | Description |
|--------|-------------|
| `enhance_entities(text, entities)` | Improve entity accuracy and details |
| `enhance_relations(text, relations)` | Improve relation detection |

**Example:**

```python
from semantica.semantic_extract import LLMEnhancer

enhancer = LLMEnhancer(provider="openai", model="gpt-4")
enhanced_entities = enhancer.enhance_entities(text, entities)
```

---

## Usage Examples

```python
from semantica.semantic_extract import (
    NamedEntityRecognizer, 
    RelationExtractor,
    TripleExtractor,
    EventDetector,
    CoreferenceResolver,
    SemanticNetworkExtractor
)

text = "Apple released the iPhone in 2007. Steve Jobs announced it at Macworld."

# Extract entities with confidence filtering
ner = NamedEntityRecognizer(confidence_threshold=0.7)
entities = ner.extract_entities(text)

# Resolve coreferences (recommended before relation extraction)
coref = CoreferenceResolver()
resolved = coref.resolve(text)

# Extract relations
rel_extractor = RelationExtractor(confidence_threshold=0.6)
relations = rel_extractor.extract_relations(text, entities=entities)

# Extract triples for KG
triple_extractor = TripleExtractor(include_temporal=True)
triples = triple_extractor.extract_triples(text)

# Detect events
event_detector = EventDetector(extract_time=True)
events = event_detector.detect_events(text)

# Extract semantic network
network_extractor = SemanticNetworkExtractor()
network = network_extractor.extract(text)

print(f"Entities: {len(entities)}")
print(f"Relations: {len(relations)}")
print(f"Triples: {len(triples)}")
print(f"Events: {len(events)}")
print(f"Network Nodes: {len(network.nodes)}")
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
from semantica.semantic_extract import NamedEntityRecognizer, RelationExtractor, TripleExtractor
from semantica.kg import GraphBuilder

# 1. Extract
text = "Google was founded by Larry Page and Sergey Brin."
ner = NamedEntityRecognizer()
entities = ner.extract_entities(text)
triple_extractor = TripleExtractor()
triples = triple_extractor.extract_triples(text)

# 2. Populate KG using GraphBuilder
builder = GraphBuilder()
sources = [{
    "entities": entities,
    "relationships": [{"source": t.subject, "target": t.object, "type": t.predicate} for t in triples]
}]
kg = builder.build(sources)
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

## Cookbook

- [Entity Extraction](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/05_Entity_Extraction.ipynb)
- [Relation Extraction](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/06_Relation_Extraction.ipynb)
- [Advanced Extraction](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/01_Advanced_Extraction.ipynb)
