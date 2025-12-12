# Semantic Extract Module Usage Guide

This comprehensive guide demonstrates how to use the semantic extraction module for extracting entities, relations, events, triples, and semantic networks from text. The module provides multiple extraction methods including pattern-based, ML-based, and LLM-based approaches, with support for custom method registration and configuration management.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Entity Extraction](#entity-extraction)
3. [Relation Extraction](#relation-extraction)
4. [Triple Extraction](#triple-extraction)
5. [Event Detection](#event-detection)
6. [Coreference Resolution](#coreference-resolution)
7. [Semantic Analysis](#semantic-analysis)
8. [Semantic Networks](#semantic-networks)
9. [Using Registry](#using-registry)
10. [Configuration](#configuration)
11. [Advanced Examples](#advanced-examples)

## Basic Usage

```python
from semantica.semantic_extract import NamedEntityRecognizer, RelationExtractor

text = "Apple Inc. was founded by Steve Jobs in 1976. The company is headquartered in Cupertino, California."

# Extract entities
ner = NamedEntityRecognizer()
entities = ner.extract_entities(text)
print(f"Entities: {entities}")

# Extract relations
rel_extractor = RelationExtractor()
relations = rel_extractor.extract(text, entities=entities)
print(f"Relations: {relations}")

print(f"Extracted {len(entities)} entities and {len(relations)} relations")
```


## Entity Extraction

### Basic Entity Extraction

```python
from semantica.semantic_extract import NamedEntityRecognizer

ner = NamedEntityRecognizer()
entities = ner.extract_entities("Apple Inc. was founded by Steve Jobs in 1976.")

for entity in entities:
    print(f"{entity.text} ({entity.type}) - Confidence: {entity.confidence:.2f}")
```

### Different Entity Extraction Methods

```python
from semantica.semantic_extract import NERExtractor

text = "Apple Inc. was founded by Steve Jobs in 1976."

# Pattern-based extraction
extractor = NERExtractor(method="pattern")
entities = extractor.extract(text)
print(f"Pattern method: {len(entities)} entities")

# Regex-based extraction
extractor = NERExtractor(method="regex")
entities = extractor.extract(text)
print(f"Regex method: {len(entities)} entities")

# ML-based extraction (spaCy)
extractor = NERExtractor(method="ml")
entities = extractor.extract(text)
print(f"ML method: {len(entities)} entities")

# HuggingFace model extraction
extractor = NERExtractor(method="huggingface")
entities = extractor.extract(text, model="dslim/bert-base-NER")
print(f"HuggingFace method: {len(entities)} entities")

# LLM-based extraction
extractor = NERExtractor(method="llm")
entities = extractor.extract(text, provider="openai", model="gpt-4")
print(f"LLM method: {len(entities)} entities")
```

### Using NERExtractor Directly

```python
from semantica.semantic_extract import NERExtractor

# 1. Standard ML (spaCy)
extractor = NERExtractor(method="ml", model="en_core_web_trf")
entities = extractor.extract(text)

# 2. LLM-based extraction
extractor = NERExtractor(
    method="llm", 
    provider="openai", 
    model="gpt-4",
    temperature=0.0
)
entities = extractor.extract(text)

# 3. Regex with custom patterns
extractor = NERExtractor(
    method="regex", 
    patterns={"CODE": r"[A-Z]{3}-\d{3}"}
)
entities = extractor.extract(text)

for entity in entities:
    print(f"Entity: {entity.text}")
    print(f"  Type: {entity.label}")
    print(f"  Confidence: {entity.confidence}")
```

### Entity Classification and Confidence Scoring

```python
from semantica.semantic_extract import EntityClassifier, EntityConfidenceScorer

classifier = EntityClassifier()
scorer = EntityConfidenceScorer()

entity = {"text": "Apple Inc.", "type": "ORG"}

# Classify entity
classification = classifier.classify(entity)
print(f"Classification: {classification}")

# Score confidence
confidence = scorer.score(entity)
print(f"Confidence: {confidence:.2f}")
```

## Relation Extraction

### Basic Relation Extraction

```python
from semantica.semantic_extract import RelationExtractor

extractor = RelationExtractor()
text = "Steve Jobs founded Apple Inc. in 1976."

relations = extractor.extract(text, entities=entities)

for relation in relations:
    print(f"{relation.subject} --[{relation.predicate}]--> {relation.object}")
    print(f"  Confidence: {relation.confidence:.2f}")
```

### Different Relation Extraction Methods

```python
from semantica.semantic_extract import RelationExtractor

text = "Steve Jobs founded Apple Inc."

# Pattern-based extraction
extractor = RelationExtractor(method="pattern")
relations = extractor.extract(text, entities=entities)

# Dependency parsing-based
extractor = RelationExtractor(method="dependency")
relations = extractor.extract(text, entities=entities)

# Co-occurrence based
extractor = RelationExtractor(method="cooccurrence")
relations = extractor.extract(text, entities=entities)

# HuggingFace model
extractor = RelationExtractor(method="huggingface")
relations = extractor.extract(text, entities=entities, model="microsoft/DialoGPT-medium")

# LLM-based
extractor = RelationExtractor(method="llm")
relations = extractor.extract(text, entities=entities, provider="openai")
```

### Relation Types

```python
from semantica.semantic_extract import Relation

# Create relation manually
relation = Relation(
    subject="Steve Jobs",
    predicate="founded",
    object="Apple Inc.",
    confidence=0.95,
    metadata={"source": "text", "position": 0}
)

print(f"Relation: {relation.subject} --[{relation.predicate}]--> {relation.object}")
```

## Triple Extraction

### Basic Triple Extraction

```python
from semantica.semantic_extract import TripleExtractor

extractor = TripleExtractor()
text = "Apple Inc. was founded by Steve Jobs in 1976."

triples = extractor.extract_triples(text)

for triple in triples:
    print(f"({triple.subject}, {triple.predicate}, {triple.object})")
    print(f"  Confidence: {triple.confidence:.2f}")
```

### Different Triple Extraction Methods

```python
from semantica.semantic_extract import TripleExtractor

text = "Apple Inc. was founded by Steve Jobs in 1976."

# Pattern-based
extractor = TripleExtractor(method="pattern")
triples = extractor.extract_triples(text)

# Rules-based
extractor = TripleExtractor(method="rules")
triples = extractor.extract_triples(text)

# HuggingFace model
extractor = TripleExtractor(method="huggingface")
triples = extractor.extract_triples(text, model="t5-base")

# LLM-based
extractor = TripleExtractor(method="llm")
triples = extractor.extract_triples(text, provider="openai", model="gpt-4")
```

### RDF Serialization

```python
from semantica.semantic_extract import TripleExtractor, RDFSerializer

extractor = TripleExtractor()
triples = extractor.extract_triples(text)

# Serialize to RDF
serializer = RDFSerializer()
rdf_output = serializer.serialize(triples, format="turtle")
print(rdf_output)

# Serialize to JSON-LD
jsonld_output = serializer.serialize(triples, format="json-ld")
print(jsonld_output)
```

### Triple Validation

```python
from semantica.semantic_extract import TripleValidator, TripleQualityChecker

validator = TripleValidator()
quality_checker = TripleQualityChecker()

for triple in triples:
    # Validate triple
    is_valid = validator.validate(triple)
    print(f"Valid: {is_valid}")
    
    # Check quality
    quality = quality_checker.check_quality(triple)
    print(f"Quality score: {quality:.2f}")
```

## Event Detection

### Basic Event Detection

```python
from semantica.semantic_extract import EventDetector

detector = EventDetector()
text = "Apple Inc. was founded in 1976. The company launched the iPhone in 2007."

events = detector.detect_events(text)

for event in events:
    print(f"Event: {event.text}")
    print(f"  Type: {event.type}")
    print(f"  Trigger: {event.trigger}")
    print(f"  Participants: {event.participants}")
    print(f"  Temporal: {event.temporal}")
```

### Event Classification

```python
from semantica.semantic_extract import EventClassifier

classifier = EventClassifier()
event = {"text": "Apple Inc. was founded", "trigger": "founded"}

event_type = classifier.classify(event)
print(f"Event type: {event_type}")
```

### Temporal Event Processing

```python
from semantica.semantic_extract import TemporalEventProcessor

processor = TemporalEventProcessor()
events = detector.detect_events(text)

# Process temporal information
processed_events = processor.process(events)

for event in processed_events:
    print(f"Event: {event.text}")
    print(f"  Time: {event.temporal.get('time')}")
    print(f"  Duration: {event.temporal.get('duration')}")
```

## Coreference Resolution

### Basic Coreference Resolution

```python
from semantica.semantic_extract import CoreferenceResolver

resolver = CoreferenceResolver()
text = "Apple Inc. was founded in 1976. The company is headquartered in Cupertino."

coreferences = resolver.resolve(text)

for chain in coreferences:
    print(f"Coreference chain: {chain.mentions}")
    print(f"  Representative: {chain.representative}")
```

### Pronoun Resolution

```python
from semantica.semantic_extract import PronounResolver

pronoun_resolver = PronounResolver()
text = "Steve Jobs founded Apple. He was the CEO."

resolved = pronoun_resolver.resolve(text)
print(f"Resolved text: {resolved}")
```

### Entity Coreference Detection

```python
from semantica.semantic_extract import EntityCoreferenceDetector

detector = EntityCoreferenceDetector()
entities = [
    {"text": "Apple Inc.", "type": "ORG"},
    {"text": "Apple", "type": "ORG"},
    {"text": "the company", "type": "ORG"}
]

coreferences = detector.detect(entities)
print(f"Found {len(coreferences)} coreference chains")
```

## Semantic Analysis

### Basic Semantic Analysis

```python
from semantica.semantic_extract import SemanticAnalyzer

analyzer = SemanticAnalyzer()
text = "Apple Inc. was founded by Steve Jobs in 1976."

analysis = analyzer.analyze(text)

print(f"Semantic roles: {analysis.roles}")
print(f"Clusters: {analysis.clusters}")
```

### Semantic Role Labeling

```python
from semantica.semantic_extract import RoleLabeler

labeler = RoleLabeler()
text = "Steve Jobs founded Apple Inc."

roles = labeler.label(text)

for role in roles:
    print(f"Role: {role.role}")
    print(f"  Argument: {role.argument}")
    print(f"  Type: {role.type}")
```

### Semantic Clustering

```python
from semantica.semantic_extract import SemanticClusterer

clusterer = SemanticClusterer()
entities = [
    {"text": "Apple Inc.", "type": "ORG"},
    {"text": "Microsoft", "type": "ORG"},
    {"text": "Google", "type": "ORG"}
]

clusters = clusterer.cluster(entities)

for cluster in clusters:
    print(f"Cluster: {cluster.label}")
    print(f"  Entities: {[e['text'] for e in cluster.entities]}")
```

### Similarity Analysis

```python
from semantica.semantic_extract import SimilarityAnalyzer

similarity_analyzer = SimilarityAnalyzer()
entity1 = {"text": "Apple Inc.", "type": "ORG"}
entity2 = {"text": "Apple", "type": "ORG"}

similarity = similarity_analyzer.analyze(entity1, entity2)
print(f"Similarity: {similarity:.2f}")
```

## Semantic Networks

### Building Semantic Networks

```python
from semantica.semantic_extract import SemanticNetworkExtractor

extractor = SemanticNetworkExtractor()
text = "Apple Inc. was founded by Steve Jobs in 1976. The company is headquartered in Cupertino."

network = extractor.extract(text)

print(f"Nodes: {len(network.nodes)}")
print(f"Edges: {len(network.edges)}")

for node in network.nodes:
    print(f"Node: {node.label} ({node.type})")

for edge in network.edges:
    print(f"Edge: {edge.source} --[{edge.relation}]--> {edge.target}")
```

### Semantic Network Components

```python
from semantica.semantic_extract import SemanticNode, SemanticEdge

# Create semantic node
node = SemanticNode(
    label="Apple Inc.",
    type="ORG",
    properties={"founded": 1976}
)

# Create semantic edge
edge = SemanticEdge(
    source="Steve Jobs",
    target="Apple Inc.",
    relation="founded",
    weight=0.95
)

print(f"Node: {node.label}")
print(f"Edge: {edge.source} --[{edge.relation}]--> {edge.target}")
```


## Using Registry

### Registering Custom Methods

```python
from semantica.semantic_extract.registry import method_registry

# Custom entity extraction method
def custom_entity_extraction(text, **kwargs):
    # Your custom extraction logic
    entities = []
    # ... extraction code ...
    return entities

# Register custom method
method_registry.register("entity", "custom_method", custom_entity_extraction)

# Use custom method
from semantica.semantic_extract import NERExtractor
extractor = NERExtractor(method="custom_method")
entities = extractor.extract(text)
```

### Listing Registered Methods

```python
from semantica.semantic_extract.registry import method_registry

# List all registered methods
all_methods = method_registry.list_all()
print("Registered methods:", all_methods)

# List methods for specific task
entity_methods = method_registry.list_all("entity")
print("Entity methods:", entity_methods)
```

## Configuration

### Using Configuration Manager

```python
from semantica.semantic_extract.config import config

# Get API key
api_key = config.get_api_key("openai")
print(f"OpenAI API key: {api_key[:10]}...")

# Set provider configuration
config.set_provider("openai", api_key="sk-...", model="gpt-4")

# Get provider configuration
provider_config = config.get_provider_config("openai")
print(f"Provider config: {provider_config}")
```

### Environment Variables

```bash
# Set API keys
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export ANTHROPIC_API_KEY=...
```

### Configuration File

```yaml
# config.yaml
openai:
  api_key: sk-...
  model: gpt-4

gemini:
  api_key: ...
  model: gemini-pro
```

```python
from semantica.semantic_extract.config import Config

# Load from config file
config = Config(config_file="config.yaml")
api_key = config.get_api_key("openai")
```

## Advanced Examples

### Complete Extraction Pipeline

```python
from semantica.semantic_extract import (
    NamedEntityRecognizer,
    RelationExtractor,
    TripleExtractor,
    EventDetector,
    CoreferenceResolver
)

text = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
The company launched the iPhone in 2007. It is headquartered in Cupertino, California.
"""

# Step 1: Extract entities
ner = NamedEntityRecognizer()
entities = ner.extract_entities(text)
print(f"Entities: {len(entities)}")

# Step 2: Resolve coreferences
resolver = CoreferenceResolver()
resolved_text = resolver.resolve(text)
print(f"Resolved coreferences")

# Step 3: Extract relations
rel_extractor = RelationExtractor()
relations = rel_extractor.extract_relations(resolved_text, entities=entities)
print(f"Relations: {len(relations)}")

# Step 4: Extract triples
triple_extractor = TripleExtractor()
triples = triple_extractor.extract_triples(resolved_text)
print(f"Triples: {len(triples)}")

# Step 5: Detect events
event_detector = EventDetector()
events = event_detector.detect_events(resolved_text)
print(f"Events: {len(events)}")
```

### LLM-Enhanced Extraction

```python
from semantica.semantic_extract import LLMEnhancer

enhancer = LLMEnhancer(provider="openai", model="gpt-4")

# Enhance existing extraction
entities = ner.extract_entities(text)
enhanced_entities = enhancer.enhance_entities(entities, text)

for original, enhanced in zip(entities, enhanced_entities):
    print(f"Original: {original.text}")
    print(f"Enhanced: {enhanced.text}")
    print(f"  Additional info: {enhanced.metadata}")
```

### Extraction Validation

```python
from semantica.semantic_extract import ExtractionValidator

validator = ExtractionValidator()

# Validate extraction results
result = {
    "entities": entities,
    "relations": relations,
    "triples": triples
}

validation = validator.validate(result)

print(f"Validation passed: {validation.passed}")
print(f"Quality score: {validation.quality_score:.2f}")
print(f"Issues: {validation.issues}")
```

### Building Knowledge Graph from Extraction

```python
from semantica.semantic_extract import NamedEntityRecognizer, RelationExtractor, TripleExtractor
from semantica.kg import GraphBuilder

# Extract all information
ner = NamedEntityRecognizer()
entities = ner.extract_entities(text)

rel_extractor = RelationExtractor()
relations = rel_extractor.extract_relations(text, entities=entities)

triple_extractor = TripleExtractor()
triples = triple_extractor.extract_triples(text, entities=entities, relationships=relations)

# Build knowledge graph
graph_builder = GraphBuilder()
knowledge_graph = graph_builder.build({
    "entities": entities,
    "relations": relations,
    "triples": triples
})

print(f"Knowledge graph nodes: {len(knowledge_graph.nodes)}")
print(f"Knowledge graph edges: {len(knowledge_graph.edges)}")
```

### Batch Processing

```python
from semantica.semantic_extract import NamedEntityRecognizer, RelationExtractor

texts = [
    "Apple Inc. was founded in 1976.",
    "Microsoft was founded in 1975.",
    "Google was founded in 1998."
]

# Process multiple texts
ner = NamedEntityRecognizer()
rel_extractor = RelationExtractor()

all_entities = []
all_relations = []

for text in texts:
    entities = ner.extract_entities(text)
    relations = rel_extractor.extract_relations(text, entities=entities)
    all_entities.extend(entities)
    all_relations.extend(relations)

print(f"Total entities: {len(all_entities)}")
print(f"Total relations: {len(all_relations)}")
```

## Best Practices

1. **Choose appropriate extraction methods**: Use pattern-based for simple cases, LLM-based for complex extraction
2. **Resolve coreferences first**: Resolve coreferences before relation extraction for better accuracy
3. **Validate extraction results**: Always validate extraction results for quality assurance
4. **Use appropriate providers**: Choose LLM providers based on your needs (cost, speed, accuracy)
5. **Batch processing**: Process multiple texts together for efficiency
6. **Combine methods**: Use multiple extraction methods and combine results for better coverage
7. **Customize extraction**: Register custom methods for domain-specific extraction needs

