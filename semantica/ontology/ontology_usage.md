# Ontology Management Module Usage Guide

This comprehensive guide demonstrates how to use the ontology management module for ontology generation, class/property inference, validation, evaluation, OWL generation, requirements specification, reuse management, versioning, namespace management, and associative class creation.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Ontology Generation](#ontology-generation)
3. [Class Inference](#class-inference)
4. [Property Inference](#property-inference)
5. [Ontology Validation](#ontology-validation)
6. [OWL/RDF Generation](#owlrdf-generation)
7. [Ontology Evaluation](#ontology-evaluation)
8. [Requirements Specification](#requirements-specification)
9. [Ontology Reuse](#ontology-reuse)
10. [Version Management](#version-management)
11. [Namespace Management](#namespace-management)
12. [Associative Classes](#associative-classes)
13. [Using Methods](#using-methods)
14. [Using Registry](#using-registry)
15. [Configuration](#configuration)
16. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using the Convenience Functions

```python
from semantica.ontology import generate_ontology, infer_classes, validate_ontology

# Generate ontology
data = {
    "entities": [
        {"type": "Person", "name": "John"},
        {"type": "Organization", "name": "Acme Corp"}
    ],
    "relationships": [
        {"type": "worksFor", "source": "John", "target": "Acme Corp"}
    ]
}

ontology = generate_ontology(data, method="default")
print(f"Generated ontology: {ontology['name']}")

# Infer classes
entities = [{"type": "Person", "name": "John"}]
classes = infer_classes(entities, method="default")
print(f"Inferred {len(classes)} classes")

# Validate ontology
result = validate_ontology(ontology, method="default")
if result.valid:
    print("Ontology is valid")
```

### Using Main Classes

```python
from semantica.ontology import OntologyGenerator, ClassInferrer, PropertyGenerator

# Create generators
generator = OntologyGenerator(base_uri="https://example.org/ontology/")
inferrer = ClassInferrer()
prop_gen = PropertyGenerator()

# Generate ontology
ontology = generator.generate_ontology(data)

# Infer classes
classes = inferrer.infer_classes(entities, build_hierarchy=True)

# Infer properties
properties = prop_gen.infer_properties(entities, relationships, classes)
```

## Ontology Generation

### Basic Ontology Generation

```python
from semantica.ontology import generate_ontology, OntologyGenerator

# Using convenience function
data = {
    "entities": [
        {"type": "Person", "name": "John", "age": 30},
        {"type": "Person", "name": "Jane", "age": 25},
        {"type": "Organization", "name": "Acme Corp"}
    ],
    "relationships": [
        {"type": "worksFor", "source": "John", "target": "Acme Corp"},
        {"type": "worksFor", "source": "Jane", "target": "Acme Corp"}
    ]
}

ontology = generate_ontology(data, method="default", name="PersonOrgOntology")
print(f"URI: {ontology['uri']}")
print(f"Classes: {len(ontology['classes'])}")
print(f"Properties: {len(ontology['properties'])}")

# Using class directly
generator = OntologyGenerator(base_uri="https://example.org/ontology/")
ontology = generator.generate_ontology(data, name="PersonOrgOntology")
```

### 6-Stage Pipeline

The ontology generation uses a 6-stage pipeline:

1. **Semantic Network Parsing**: Extract domain concepts
2. **YAML-to-Definition**: Transform to class definitions
3. **Definition-to-Types**: Map to OWL types
4. **Hierarchy Generation**: Build taxonomic structures
5. **TTL Generation**: Generate OWL/Turtle syntax
6. **Symbolic Validation**: HermiT/Pellet reasoning

```python
from semantica.ontology import OntologyGenerator

generator = OntologyGenerator(base_uri="https://example.org/ontology/")

# Generate with full pipeline
ontology = generator.generate_ontology(
    data,
    name="MyOntology",
    build_hierarchy=True
)

# Access pipeline stages
print(f"Stage 1: Parsed {len(ontology.get('metadata', {}).get('concept_count', 0))} concepts")
print(f"Stage 2-3: Generated {len(ontology['classes'])} classes and {len(ontology['properties'])} properties")
print(f"Stage 4: Built hierarchy with {len([c for c in ontology['classes'] if 'parent' in c])} parent-child relationships")
```

### Generation Options

```python
from semantica.ontology import generate_ontology

# Generate with custom options
ontology = generate_ontology(
    data,
    method="default",
    name="CustomOntology",
    base_uri="https://example.org/ontology/",
    build_hierarchy=True,
    min_occurrences=3
)
```

## Class Inference

### Basic Class Inference

```python
from semantica.ontology import infer_classes, ClassInferrer

entities = [
    {"type": "Person", "name": "John", "age": 30},
    {"type": "Person", "name": "Jane", "age": 25},
    {"type": "Organization", "name": "Acme Corp"},
    {"type": "Organization", "name": "Tech Inc"}
]

# Using convenience function
classes = infer_classes(entities, method="default", build_hierarchy=True)

for cls in classes:
    print(f"Class: {cls['name']}, Instances: {cls.get('entity_count', 0)}")

# Using class directly
inferrer = ClassInferrer(min_occurrences=2, similarity_threshold=0.8)
classes = inferrer.infer_classes(entities, build_hierarchy=True)
```

### Hierarchy Building

```python
from semantica.ontology import ClassInferrer

inferrer = ClassInferrer()

# Infer classes with hierarchy
classes = inferrer.infer_classes(entities, build_hierarchy=True)

# Build hierarchy separately
hierarchical = inferrer.build_class_hierarchy(classes)

# Validate classes
validation = inferrer.validate_classes(classes)
if validation.get("valid"):
    print("Classes are valid")
```

### Class Inference Options

```python
from semantica.ontology import infer_classes

# Custom inference options
classes = infer_classes(
    entities,
    method="default",
    build_hierarchy=True,
    min_occurrences=3,
    similarity_threshold=0.85
)
```

## Property Inference

### Basic Property Inference

```python
from semantica.ontology import infer_properties, PropertyGenerator

entities = [
    {"type": "Person", "name": "John", "age": 30, "email": "john@example.com"},
    {"type": "Person", "name": "Jane", "age": 25, "email": "jane@example.com"}
]

relationships = [
    {"type": "worksFor", "source": "John", "target": "Acme Corp"},
    {"type": "knows", "source": "John", "target": "Jane"}
]

classes = [
    {"name": "Person", "uri": "https://example.org/ontology/Person"},
    {"name": "Organization", "uri": "https://example.org/ontology/Organization"}
]

# Using convenience function
properties = infer_properties(entities, relationships, classes, method="default")

for prop in properties:
    print(f"Property: {prop['name']}, Type: {prop['type']}")

# Using class directly
prop_gen = PropertyGenerator()
properties = prop_gen.infer_properties(entities, relationships, classes)
```

### Object vs Data Properties

```python
from semantica.ontology import PropertyGenerator

prop_gen = PropertyGenerator()

# Infer all properties
properties = prop_gen.infer_properties(entities, relationships, classes)

# Separate object and data properties
object_props = [p for p in properties if p['type'] == 'object']
data_props = [p for p in properties if p['type'] == 'data']

print(f"Object properties: {len(object_props)}")
print(f"Data properties: {len(data_props)}")

# Validate properties
validation = prop_gen.validate_properties(properties)
```

## Ontology Validation

### Basic Validation

```python
from semantica.ontology import validate_ontology, OntologyValidator

# Using convenience function
result = validate_ontology(ontology, method="default")

if result.valid:
    print("Ontology is valid")
if result.consistent:
    print("Ontology is consistent")
if result.errors:
    print(f"Errors: {len(result.errors)}")
if result.warnings:
    print(f"Warnings: {len(result.warnings)}")

# Using class directly
validator = OntologyValidator(reasoner="hermit")
result = validator.validate_ontology(ontology)
```

### Validation with Reasoners

```python
from semantica.ontology import validate_ontology

# HermiT reasoner
result = validate_ontology(ontology, method="hermit", reasoner="hermit")

# Pellet reasoner
result = validate_ontology(ontology, method="pellet", reasoner="pellet")

# Auto-select reasoner
result = validate_ontology(ontology, method="default", reasoner="auto")
```

### Validation Options

```python
from semantica.ontology import validate_ontology

# Custom validation options
result = validate_ontology(
    ontology,
    method="default",
    check_consistency=True,
    check_satisfiability=True
)
```

## OWL/RDF Generation

### Basic OWL Generation

```python
from semantica.ontology import generate_owl, OWLGenerator

# Using convenience function
turtle = generate_owl(ontology, format="turtle", method="default")
print(turtle)

# Using class directly
generator = OWLGenerator()
turtle = generator.generate_owl(ontology, format="turtle")
```

### Different Formats

```python
from semantica.ontology import generate_owl

# Turtle format
turtle = generate_owl(ontology, format="turtle", method="default")

# RDF/XML format
rdfxml = generate_owl(ontology, format="rdfxml", method="default")

# JSON-LD format
jsonld = generate_owl(ontology, format="json-ld", method="default")

# N3 format
n3 = generate_owl(ontology, format="n3", method="default")
```

### Export to File

```python
from semantica.ontology import OWLGenerator

generator = OWLGenerator()

# Export to file
generator.export_owl(ontology, "ontology.ttl", format="turtle")
generator.export_owl(ontology, "ontology.rdf", format="rdfxml")
generator.export_owl(ontology, "ontology.jsonld", format="json-ld")
```

## Ontology Evaluation

### Basic Evaluation

```python
from semantica.ontology import evaluate_ontology, OntologyEvaluator

competency_questions = [
    "Who are the employees of an organization?",
    "What organizations does a person work for?",
    "What is the age of a person?"
]

# Using convenience function
result = evaluate_ontology(
    ontology,
    competency_questions=competency_questions,
    method="default"
)

print(f"Coverage score: {result.coverage_score:.2f}")
print(f"Completeness score: {result.completeness_score:.2f}")
print(f"Gaps: {len(result.gaps)}")
print(f"Suggestions: {len(result.suggestions)}")

# Using class directly
evaluator = OntologyEvaluator()
result = evaluator.evaluate_ontology(ontology, competency_questions=competency_questions)
```

### Coverage and Completeness

```python
from semantica.ontology import OntologyEvaluator

evaluator = OntologyEvaluator()

# Evaluate coverage
coverage = evaluator.calculate_coverage(ontology, competency_questions)

# Evaluate completeness
completeness = evaluator.calculate_completeness(ontology)

# Identify gaps
gaps = evaluator.identify_gaps(ontology, competency_questions)

# Generate report
report = evaluator.generate_report(ontology)
```

## Requirements Specification

### Creating Requirements Spec

```python
from semantica.ontology import create_requirements_spec, RequirementsSpecManager

# Using convenience function
spec = create_requirements_spec(
    "PersonOntology",
    "Model person-related concepts and relationships",
    "Person, Organization, Role entities and their relationships",
    method="default",
    domain="human resources",
    stakeholders=["domain experts", "developers"],
    use_cases=["employee management", "organizational structure"]
)

# Using class directly
manager = RequirementsSpecManager()
spec = manager.create_spec(
    "PersonOntology",
    "Model person-related concepts",
    "Person entities"
)
```

### Adding Competency Questions

```python
from semantica.ontology import add_competency_question, CompetencyQuestionsManager

# Using convenience function
cq = add_competency_question(
    "PersonOntology",
    "Who are the employees of a given organization?",
    category="organizational",
    method="default",
    priority=1
)

# Using class directly
manager = CompetencyQuestionsManager()
cq = manager.add_question(
    "Who are the employees of a given organization?",
    category="organizational",
    priority=1
)
```

### Requirements Traceability

```python
from semantica.ontology import RequirementsSpecManager

manager = RequirementsSpecManager()

# Trace requirements to ontology
trace = manager.trace_requirements("PersonOntology", ontology)

# Validate against requirements
validation = manager.validate_against_spec("PersonOntology", ontology)
```

## Ontology Reuse

### Researching Ontologies

```python
from semantica.ontology import research_ontology, ReuseManager

# Using convenience function
info = research_ontology("http://xmlns.com/foaf/0.1/", method="default")

if info:
    print(f"Name: {info.get('name')}")
    print(f"Description: {info.get('description')}")

# Using class directly
manager = ReuseManager()
info = manager.research_ontology("http://xmlns.com/foaf/0.1/")
```

### Importing External Ontologies

```python
from semantica.ontology import import_external_ontology, ReuseManager

# Using convenience function
updated = import_external_ontology(
    "http://xmlns.com/foaf/0.1/",
    ontology,
    method="default"
)

# Using class directly
manager = ReuseManager()
updated = manager.import_external_ontology("http://xmlns.com/foaf/0.1/", ontology)
```

### Alignment Evaluation

```python
from semantica.ontology import ReuseManager

manager = ReuseManager()

# Evaluate alignment
alignment = manager.evaluate_alignment(
    "http://xmlns.com/foaf/0.1/",
    ontology
)

print(f"Compatibility score: {alignment.get('compatibility_score', 0):.2f}")
print(f"Decision: {alignment.get('decision')}")
```

## Version Management

### Creating Versions

```python
from semantica.ontology import create_version, VersionManager

# Using convenience function
version = create_version(
    "1.0",
    ontology,
    method="default",
    changes=["Added Person class", "Added worksFor property"]
)

print(f"Version: {version.version}")
print(f"IRI: {version.ontology_iri}")

# Using class directly
manager = VersionManager(base_uri="https://example.org/ontology/")
version = manager.create_version("1.0", ontology, changes=["Added Person class"])
```

### Version Comparison

```python
from semantica.ontology import VersionManager

manager = VersionManager()

# Compare versions
comparison = manager.compare_versions("1.0", "2.0")

# Get version diff
diff = manager.get_version_diff("1.0", "2.0")

# Migrate ontology
migrated = manager.migrate_ontology("1.0", "2.0", ontology)
```

## Namespace Management

### Managing Namespaces

```python
from semantica.ontology import manage_namespace, NamespaceManager

# Using convenience function
class_iri = manage_namespace(
    "generate_class_iri",
    class_name="Person",
    base_uri="https://example.org/ontology/"
)
print(f"Class IRI: {class_iri}")

property_iri = manage_namespace(
    "generate_property_iri",
    property_name="hasName",
    base_uri="https://example.org/ontology/"
)
print(f"Property IRI: {property_iri}")

# Using class directly
manager = NamespaceManager(base_uri="https://example.org/ontology/")
class_iri = manager.generate_class_iri("Person")
property_iri = manager.generate_property_iri("hasName")
```

### Registering Namespaces

```python
from semantica.ontology import NamespaceManager

manager = NamespaceManager()

# Register custom namespace
manager.register_namespace("ex", "https://example.org/")

# Get base URI
base_uri = manager.get_base_uri()

# Get all namespaces
namespaces = manager.get_all_namespaces()
```

## Associative Classes

### Creating Associative Classes

```python
from semantica.ontology import create_associative_class, AssociativeClassBuilder

# Using convenience function
position = create_associative_class(
    "Position",
    ["Person", "Organization", "Role"],
    method="default",
    properties={"startDate": "xsd:date", "endDate": "xsd:date"},
    temporal=True
)

print(f"Associative class: {position.name}")
print(f"Connects: {position.connects}")

# Using class directly
builder = AssociativeClassBuilder()
position = builder.create_associative_class(
    "Position",
    ["Person", "Organization", "Role"],
    properties={"startDate": "xsd:date"},
    temporal=True
)
```

### Position Classes

```python
from semantica.ontology import AssociativeClassBuilder

builder = AssociativeClassBuilder()

# Create position class
position = builder.create_position_class(
    person_class="Person",
    organization_class="Organization",
    role_class="Role"
)

# Validate associative class
validation = builder.validate_associative_class(position)
```

## Using Methods

### Getting Available Methods

```python
from semantica.ontology.methods import get_ontology_method, list_available_methods

# List all available methods
all_methods = list_available_methods()
print("Available methods:", all_methods)

# List methods for specific task
generate_methods = list_available_methods("generate")
print("Generation methods:", generate_methods)

# Get specific method
generate_method = get_ontology_method("generate", "default")
if generate_method:
    ontology = generate_method(data)
```

### Method Examples

```python
from semantica.ontology.methods import (
    generate_ontology,
    infer_classes,
    infer_properties,
    validate_ontology,
    generate_owl,
    evaluate_ontology,
    create_requirements_spec,
    add_competency_question,
    research_ontology,
    import_external_ontology,
    create_version,
    manage_namespace,
    create_associative_class
)

# Generate ontology
ontology = generate_ontology(data, method="default")

# Infer classes
classes = infer_classes(entities, method="default")

# Infer properties
properties = infer_properties(entities, relationships, classes, method="default")

# Validate ontology
result = validate_ontology(ontology, method="default")

# Generate OWL
turtle = generate_owl(ontology, format="turtle", method="default")

# Evaluate ontology
evaluation = evaluate_ontology(ontology, competency_questions=["Who are the employees?"], method="default")

# Create requirements spec
spec = create_requirements_spec("PersonOntology", "Purpose", "Scope", method="default")

# Add competency question
cq = add_competency_question("PersonOntology", "Who are the employees?", method="default")

# Research ontology
info = research_ontology("http://xmlns.com/foaf/0.1/", method="default")

# Import external ontology
updated = import_external_ontology("http://xmlns.com/foaf/0.1/", ontology, method="default")

# Create version
version = create_version("1.0", ontology, method="default")

# Manage namespace
class_iri = manage_namespace("generate_class_iri", class_name="Person", method="default")

# Create associative class
position = create_associative_class("Position", ["Person", "Organization"], method="default")
```

## Using Registry

### Registering Custom Methods

```python
from semantica.ontology.registry import method_registry

# Custom ontology generation method
def custom_ontology_generation(data, **kwargs):
    """Custom generation logic."""
    # Your custom generation code
    return {"uri": "custom", "name": "CustomOntology", "classes": [], "properties": []}

# Register custom method
method_registry.register("generate", "custom", custom_ontology_generation)

# Use custom method
from semantica.ontology.methods import get_ontology_method
custom_method = get_ontology_method("generate", "custom")
ontology = custom_method(data)
```

### Listing Registered Methods

```python
from semantica.ontology.registry import method_registry

# List all registered methods
all_methods = method_registry.list_all()
print("Registered methods:", all_methods)

# List methods for specific task
generate_methods = method_registry.list_all("generate")
print("Generation methods:", generate_methods)

validate_methods = method_registry.list_all("validate")
print("Validation methods:", validate_methods)
```

### Unregistering Methods

```python
from semantica.ontology.registry import method_registry

# Unregister a method
method_registry.unregister("generate", "custom")

# Clear all methods for a task
method_registry.clear("generate")

# Clear all methods
method_registry.clear()
```

## Configuration

### Using Configuration Manager

```python
from semantica.ontology.config import ontology_config

# Get configuration values
base_uri = ontology_config.get("base_uri", default="https://semantica.dev/ontology/")
reasoner = ontology_config.get("reasoner", default="auto")
format = ontology_config.get("format", default="turtle")
min_occurrences = ontology_config.get("min_occurrences", default=2)

# Set configuration values
ontology_config.set("base_uri", "https://example.org/ontology/")
ontology_config.set("reasoner", "hermit")

# Method-specific configuration
ontology_config.set_method_config("generate", base_uri="https://example.org/ontology/", min_occurrences=3)
generate_config = ontology_config.get_method_config("generate")

# Get all configuration
all_config = ontology_config.get_all()
print("All config:", all_config)
```

### Environment Variables

```bash
# Set environment variables
export ONTOLOGY_BASE_URI=https://example.org/ontology/
export ONTOLOGY_REASONER=hermit
export ONTOLOGY_FORMAT=turtle
export ONTOLOGY_MIN_OCCURRENCES=3
export ONTOLOGY_SIMILARITY_THRESHOLD=0.85
export ONTOLOGY_CHECK_CONSISTENCY=true
export ONTOLOGY_CHECK_SATISFIABILITY=true
```

### Configuration File

```yaml
# config.yaml
ontology:
  base_uri: https://example.org/ontology/
  reasoner: hermit
  format: turtle
  min_occurrences: 3
  similarity_threshold: 0.85
  check_consistency: true
  check_satisfiability: true

ontology_methods:
  generate:
    base_uri: https://example.org/ontology/
    min_occurrences: 3
  validate:
    reasoner: hermit
    check_consistency: true
  owl:
    format: turtle
```

```python
from semantica.ontology.config import OntologyConfig

# Load from config file
config = OntologyConfig(config_file="config.yaml")
base_uri = config.get("base_uri")
```

## Advanced Examples

### Complete Ontology Generation Pipeline

```python
from semantica.ontology import (
    generate_ontology,
    infer_classes,
    infer_properties,
    validate_ontology,
    generate_owl,
    evaluate_ontology
)

# Step 1: Generate ontology
data = {
    "entities": [
        {"type": "Person", "name": "John", "age": 30},
        {"type": "Person", "name": "Jane", "age": 25},
        {"type": "Organization", "name": "Acme Corp"}
    ],
    "relationships": [
        {"type": "worksFor", "source": "John", "target": "Acme Corp"}
    ]
}

ontology = generate_ontology(data, method="default", name="PersonOrgOntology")

# Step 2: Validate ontology
result = validate_ontology(ontology, method="default")
if not result.valid:
    print(f"Validation errors: {result.errors}")

# Step 3: Generate OWL
turtle = generate_owl(ontology, format="turtle", method="default")

# Step 4: Evaluate ontology
competency_questions = ["Who are the employees of an organization?"]
evaluation = evaluate_ontology(ontology, competency_questions=competency_questions, method="default")

print(f"Coverage: {evaluation.coverage_score:.2f}")
print(f"Completeness: {evaluation.completeness_score:.2f}")
```

### Custom Ontology Workflow

```python
from semantica.ontology import (
    OntologyGenerator,
    ClassInferrer,
    PropertyGenerator,
    OntologyValidator,
    OWLGenerator
)

# Create generators with custom config
generator = OntologyGenerator(
    base_uri="https://example.org/ontology/",
    min_occurrences=3
)

inferrer = ClassInferrer(min_occurrences=3, similarity_threshold=0.85)
prop_gen = PropertyGenerator()
validator = OntologyValidator(reasoner="hermit")
owl_gen = OWLGenerator()

# Generate ontology
ontology = generator.generate_ontology(data, name="CustomOntology")

# Infer classes
classes = inferrer.infer_classes(entities, build_hierarchy=True)

# Infer properties
properties = prop_gen.infer_properties(entities, relationships, classes)

# Validate
result = validator.validate_ontology(ontology)

# Generate OWL
turtle = owl_gen.generate_owl(ontology, format="turtle")
```

### Requirements-Driven Development

```python
from semantica.ontology import (
    create_requirements_spec,
    add_competency_question,
    generate_ontology,
    evaluate_ontology
)

# Step 1: Create requirements specification
spec = create_requirements_spec(
    "PersonOntology",
    "Model person-related concepts",
    "Person, Organization, Role entities",
    method="default"
)

# Step 2: Add competency questions
add_competency_question("PersonOntology", "Who are the employees?", category="organizational", method="default")
add_competency_question("PersonOntology", "What organizations does a person work for?", category="organizational", method="default")

# Step 3: Generate ontology
ontology = generate_ontology(data, method="default")

# Step 4: Evaluate against requirements
competency_questions = ["Who are the employees?", "What organizations does a person work for?"]
evaluation = evaluate_ontology(ontology, competency_questions=competency_questions, method="default")

# Step 5: Refine based on gaps
if evaluation.gaps:
    print(f"Gaps identified: {evaluation.gaps}")
    print(f"Suggestions: {evaluation.suggestions}")
```

### Version Management Workflow

```python
from semantica.ontology import (
    generate_ontology,
    create_version,
    VersionManager
)

# Generate initial ontology
ontology_v1 = generate_ontology(data, method="default", name="PersonOntology")

# Create version 1.0
version1 = create_version("1.0", ontology_v1, changes=["Initial version"], method="default")

# Modify ontology (add new class)
ontology_v2 = ontology_v1.copy()
ontology_v2["classes"].append({"name": "Role", "uri": "https://example.org/ontology/Role"})

# Create version 2.0
version2 = create_version("2.0", ontology_v2, changes=["Added Role class"], method="default")

# Compare versions
manager = VersionManager()
comparison = manager.compare_versions("1.0", "2.0")
print(f"Changes: {comparison.get('changes', [])}")
```

### Integration with Knowledge Graph

```python
from semantica.ontology import generate_ontology, generate_owl
from semantica.kg import build

# Build knowledge graph
kg = build(sources=[{"entities": entities, "relationships": relationships}])

# Generate ontology from KG
ontology = generate_ontology({
    "entities": kg.get("entities", []),
    "relationships": kg.get("relationships", [])
}, method="default")

# Generate OWL
turtle = generate_owl(ontology, format="turtle", method="default")

# Export for use in KG
with open("ontology.ttl", "w") as f:
    f.write(turtle)
```

## Best Practices

1. **Ontology Generation**: Always validate generated ontologies before use
2. **Class Inference**: Use appropriate minimum occurrence thresholds to avoid noise
3. **Property Inference**: Validate domain/range constraints for properties
4. **Validation**: Use symbolic reasoners (HermiT/Pellet) for consistency checking
5. **OWL Generation**: Prefer Turtle format for readability, RDF/XML for compatibility
6. **Evaluation**: Define competency questions early in the development process
7. **Requirements**: Trace requirements to ontology elements for maintainability
8. **Reuse**: Research existing ontologies before creating new ones
9. **Versioning**: Use version-aware IRIs following best practices
10. **Namespaces**: Use stable, resolvable URIs for namespaces
11. **Associative Classes**: Use for complex relationships with properties
12. **Configuration**: Use configuration files for consistent settings across environments
13. **Error Handling**: Always handle ValidationError and ProcessingError exceptions
14. **Method Registry**: Register custom methods for domain-specific ontology needs

