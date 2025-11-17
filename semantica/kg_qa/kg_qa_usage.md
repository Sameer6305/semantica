# Knowledge Graph Quality Assurance Module Usage Guide

This comprehensive guide demonstrates how to use the knowledge graph quality assurance module for quality assessment, validation, reporting, and automated fixing of knowledge graphs.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Quality Assessment](#quality-assessment)
3. [Quality Reporting](#quality-reporting)
4. [Consistency Checking](#consistency-checking)
5. [Completeness Validation](#completeness-validation)
6. [Quality Metrics](#quality-metrics)
7. [Validation Engine](#validation-engine)
8. [Automated Fixes](#automated-fixes)
9. [Using Methods](#using-methods)
10. [Using Registry](#using-registry)
11. [Configuration](#configuration)
12. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using the Convenience Functions

```python
from semantica.kg_qa import assess_quality, generate_quality_report

# Assess overall quality
knowledge_graph = {
    "entities": [
        {"id": "1", "name": "Alice", "type": "Person"},
        {"id": "2", "name": "Bob", "type": "Person"}
    ],
    "relationships": [
        {"source": "1", "target": "2", "type": "knows"}
    ]
}

score = assess_quality(knowledge_graph, method="default")
print(f"Overall quality score: {score:.2f}")

# Generate quality report
schema = {
    "constraints": {
        "Person": {
            "required_props": ["id", "name", "type"]
        }
    }
}

report = generate_quality_report(knowledge_graph, schema, method="default")
print(f"Overall score: {report.overall_score:.2f}")
print(f"Completeness: {report.completeness_score:.2f}")
print(f"Consistency: {report.consistency_score:.2f}")
print(f"Issues found: {len(report.issues)}")
```

### Using Main Classes

```python
from semantica.kg_qa import KGQualityAssessor, QualityMetrics, ValidationEngine

# Create quality assessor
assessor = KGQualityAssessor()

# Assess overall quality
score = assessor.assess_overall_quality(knowledge_graph)
print(f"Quality score: {score:.2f}")

# Generate quality report
report = assessor.generate_quality_report(knowledge_graph, schema)

# Identify quality issues
issues = assessor.identify_quality_issues(knowledge_graph, schema)
print(f"Found {len(issues)} quality issues")
```

## Quality Assessment

### Basic Quality Assessment

```python
from semantica.kg_qa import assess_quality, KGQualityAssessor

# Using convenience function
score = assess_quality(knowledge_graph, method="default")

# Using class directly
assessor = KGQualityAssessor()
score = assessor.assess_overall_quality(knowledge_graph)
```

### Comprehensive Assessment

```python
from semantica.kg_qa import assess_quality

# Comprehensive assessment with all metrics
score = assess_quality(
    knowledge_graph,
    method="comprehensive"
)
```

### Quick Assessment

```python
from semantica.kg_qa import assess_quality

# Quick assessment with basic metrics
score = assess_quality(
    knowledge_graph,
    method="quick"
)
```

## Quality Reporting

### Basic Report Generation

```python
from semantica.kg_qa import generate_quality_report, QualityReporter

# Using convenience function
report = generate_quality_report(
    knowledge_graph,
    schema,
    method="default"
)

print(f"Overall score: {report.overall_score:.2f}")
print(f"Completeness: {report.completeness_score:.2f}")
print(f"Consistency: {report.consistency_score:.2f}")

# Using class directly
reporter = QualityReporter()
quality_metrics = {
    "overall": 0.85,
    "completeness": 0.90,
    "consistency": 0.80
}
report = reporter.generate_report(knowledge_graph, quality_metrics)
```

### Detailed Report

```python
from semantica.kg_qa import generate_quality_report

# Generate detailed report with all issues
report = generate_quality_report(
    knowledge_graph,
    schema,
    method="detailed"
)

# Access issues
for issue in report.issues:
    print(f"Issue: {issue.id}")
    print(f"  Type: {issue.type}")
    print(f"  Severity: {issue.severity}")
    print(f"  Description: {issue.description}")
```

### Summary Report

```python
from semantica.kg_qa import generate_quality_report

# Generate summary report only
report = generate_quality_report(
    knowledge_graph,
    schema,
    method="summary"
)
```

### Exporting Reports

```python
from semantica.kg_qa import export_report, generate_quality_report

# Generate report
report = generate_quality_report(knowledge_graph, schema)

# Export to JSON
json_report = export_report(report, format="json")
print(json_report)

# Export to YAML
yaml_report = export_report(report, format="yaml")
print(yaml_report)

# Export using QualityReporter directly
from semantica.kg_qa import QualityReporter
reporter = QualityReporter()
json_output = reporter.export_report(report, format="json")
```

### Issue Identification

```python
from semantica.kg_qa import identify_quality_issues

# Identify all quality issues
issues = identify_quality_issues(knowledge_graph, schema, method="default")

for issue in issues:
    print(f"Issue ID: {issue['id']}")
    print(f"  Type: {issue['type']}")
    print(f"  Severity: {issue['severity']}")
    print(f"  Description: {issue['description']}")
```

### Issue Tracking

```python
from semantica.kg_qa import IssueTracker, QualityIssue

tracker = IssueTracker()

# Add issues
issue = QualityIssue(
    id="missing_property_1",
    type="completeness",
    severity="medium",
    description="Entity missing required property 'email'",
    entity_id="1"
)
tracker.add_issue(issue)

# List all issues
all_issues = tracker.list_issues()
print(f"Total issues: {len(all_issues)}")

# Filter by severity
high_issues = tracker.list_issues(severity="high")
print(f"High severity issues: {len(high_issues)}")

# Get specific issue
issue = tracker.get_issue("missing_property_1")

# Resolve issue
tracker.resolve_issue("missing_property_1")
```

## Consistency Checking

### Logical Consistency

```python
from semantica.kg_qa import check_consistency, ConsistencyChecker

# Using convenience function
is_consistent = check_consistency(
    knowledge_graph,
    consistency_type="logical",
    method="default"
)
print(f"Logically consistent: {is_consistent}")

# Using class directly
checker = ConsistencyChecker()
is_consistent = checker.check_logical_consistency(knowledge_graph)
```

### Temporal Consistency

```python
from semantica.kg_qa import check_consistency

# Check temporal consistency
is_consistent = check_consistency(
    knowledge_graph,
    consistency_type="temporal",
    method="default"
)
print(f"Temporally consistent: {is_consistent}")
```

### Hierarchical Consistency

```python
from semantica.kg_qa import check_consistency

# Check hierarchical consistency
is_consistent = check_consistency(
    knowledge_graph,
    consistency_type="hierarchical",
    method="default"
)
print(f"Hierarchically consistent: {is_consistent}")
```

### All Consistency Checks

```python
from semantica.kg_qa import check_consistency

# Check all consistency types
consistency_results = check_consistency(
    knowledge_graph,
    consistency_type="all",
    method="default"
)

print(f"Logical: {consistency_results['logical']}")
print(f"Temporal: {consistency_results['temporal']}")
print(f"Hierarchical: {consistency_results['hierarchical']}")
```

### Using Consistency Metrics

```python
from semantica.kg_qa import ConsistencyMetrics

metrics = ConsistencyMetrics()

# Calculate logical consistency score
logical_score = metrics.calculate_logical_consistency(knowledge_graph)
print(f"Logical consistency score: {logical_score:.2f}")

# Calculate temporal consistency score
temporal_score = metrics.calculate_temporal_consistency(knowledge_graph)
print(f"Temporal consistency score: {temporal_score:.2f}")

# Calculate hierarchical consistency score
hierarchical_score = metrics.calculate_hierarchical_consistency(knowledge_graph)
print(f"Hierarchical consistency score: {hierarchical_score:.2f}")
```

## Completeness Validation

### Entity Completeness

```python
from semantica.kg_qa import validate_completeness, CompletenessValidator

entities = [
    {"id": "1", "name": "Alice", "type": "Person"},
    {"id": "2", "name": "Bob", "type": "Person"}
]

schema = {
    "constraints": {
        "Person": {
            "required_props": ["id", "name", "type", "email"]
        }
    }
}

# Using convenience function
is_complete = validate_completeness(
    entities=entities,
    schema=schema,
    completeness_type="entity",
    method="default"
)
print(f"Entities complete: {is_complete}")

# Using class directly
validator = CompletenessValidator()
is_complete = validator.validate_entity_completeness(entities, schema)
```

### Relationship Completeness

```python
from semantica.kg_qa import validate_completeness

relationships = [
    {"source": "1", "target": "2", "type": "knows"},
    {"source": "2", "target": "1", "type": "knows"}
]

# Validate relationship completeness
is_complete = validate_completeness(
    relationships=relationships,
    schema=schema,
    completeness_type="relationship",
    method="default"
)
print(f"Relationships complete: {is_complete}")
```

### Property Completeness

```python
from semantica.kg_qa import validate_completeness

properties = {
    "Person": {
        "id": True,
        "name": True,
        "type": True,
        "email": False  # Missing
    }
}

# Validate property completeness
is_complete = validate_completeness(
    properties=properties,
    schema=schema,
    completeness_type="property",
    method="default"
)
print(f"Properties complete: {is_complete}")
```

### All Completeness Checks

```python
from semantica.kg_qa import validate_completeness

# Validate all completeness types
completeness_results = validate_completeness(
    entities=entities,
    relationships=relationships,
    properties=properties,
    schema=schema,
    completeness_type="all",
    method="default"
)

print(f"Entity completeness: {completeness_results['entity']}")
print(f"Relationship completeness: {completeness_results['relationship']}")
print(f"Property completeness: {completeness_results['property']}")
```

### Using Completeness Metrics

```python
from semantica.kg_qa import CompletenessMetrics

metrics = CompletenessMetrics()

# Calculate entity completeness score
entity_score = metrics.calculate_entity_completeness(entities, schema)
print(f"Entity completeness score: {entity_score:.2f}")

# Calculate relationship completeness score
rel_score = metrics.calculate_relationship_completeness(relationships, schema)
print(f"Relationship completeness score: {rel_score:.2f}")

# Calculate property completeness score
prop_score = metrics.calculate_property_completeness(properties, schema)
print(f"Property completeness score: {prop_score:.2f}")
```

## Quality Metrics

### Overall Quality Score

```python
from semantica.kg_qa import calculate_quality_metrics, QualityMetrics

# Using convenience function
overall_score = calculate_quality_metrics(
    knowledge_graph,
    metrics_type="overall",
    method="default"
)
print(f"Overall quality score: {overall_score:.2f}")

# Using class directly
metrics = QualityMetrics()
overall_score = metrics.calculate_overall_score(knowledge_graph)
```

### Entity Quality Score

```python
from semantica.kg_qa import calculate_quality_metrics

entities = [
    {"id": "1", "name": "Alice", "type": "Person"},
    {"id": "2", "name": "Bob", "type": "Person"}
]

# Calculate entity quality
entity_score = calculate_quality_metrics(
    {"entities": entities},
    metrics_type="entity",
    method="default"
)
print(f"Entity quality score: {entity_score:.2f}")
```

### Relationship Quality Score

```python
from semantica.kg_qa import calculate_quality_metrics

relationships = [
    {"source": "1", "target": "2", "type": "knows"}
]

# Calculate relationship quality
rel_score = calculate_quality_metrics(
    {"relationships": relationships},
    metrics_type="relationship",
    method="default"
)
print(f"Relationship quality score: {rel_score:.2f}")
```

### All Quality Metrics

```python
from semantica.kg_qa import calculate_quality_metrics

# Calculate all metrics
all_metrics = calculate_quality_metrics(
    knowledge_graph,
    metrics_type="all",
    method="default"
)

print(f"Overall: {all_metrics['overall']:.2f}")
print(f"Entity: {all_metrics['entity']:.2f}")
print(f"Relationship: {all_metrics['relationship']:.2f}")
```

### Using Quality Metrics Directly

```python
from semantica.kg_qa import QualityMetrics

metrics = QualityMetrics()

# Calculate overall score
overall = metrics.calculate_overall_score(knowledge_graph)

# Calculate entity quality
entities = getattr(knowledge_graph, "entities", [])
entity_quality = metrics.calculate_entity_quality(entities)

# Calculate relationship quality
relationships = getattr(knowledge_graph, "relationships", [])
rel_quality = metrics.calculate_relationship_quality(relationships)

print(f"Overall: {overall:.2f}")
print(f"Entity: {entity_quality:.2f}")
print(f"Relationship: {rel_quality:.2f}")
```

## Validation Engine

### Basic Validation

```python
from semantica.kg_qa import validate_graph, ValidationEngine

# Using convenience function
result = validate_graph(knowledge_graph, method="default")

if result.valid:
    print("Graph is valid")
else:
    print(f"Validation failed with {len(result.errors)} errors")
    for error in result.errors:
        print(f"  Error: {error}")

# Using class directly
engine = ValidationEngine()
result = engine.validate(knowledge_graph)
```

### Custom Rule Validation

```python
from semantica.kg_qa import validate_graph, ValidationEngine

# Define custom validation rule
def check_entity_ids(knowledge_graph):
    """Check that all entities have valid IDs."""
    entities = getattr(knowledge_graph, "entities", [])
    errors = []
    for entity in entities:
        if "id" not in entity or not entity["id"]:
            errors.append(f"Entity missing ID: {entity}")
    return {"error": errors[0] if errors else None}

# Validate with custom rule
result = validate_graph(
    knowledge_graph,
    rules=[check_entity_ids],
    method="custom"
)

# Using class directly
engine = ValidationEngine()
engine.add_rule(check_entity_ids)
result = engine.validate(knowledge_graph)
```

### Constraint-Based Validation

```python
from semantica.kg_qa import validate_graph, ConstraintValidator

constraints = {
    "entities": {
        "Person": {
            "required_props": ["id", "name", "type"]
        }
    },
    "relationships": {
        "knows": {
            "domain": "Person",
            "range": "Person"
        }
    }
}

# Validate with constraints
validator = ConstraintValidator()
result = validator.validate_constraints(knowledge_graph, constraints)

if result.valid:
    print("Graph satisfies all constraints")
else:
    print(f"Constraint violations: {len(result.errors)}")
```

### Rule Management

```python
from semantica.kg_qa import ValidationEngine

engine = ValidationEngine()

# Add validation rule
def check_required_fields(knowledge_graph):
    """Check required fields."""
    # Validation logic
    return {"error": None}

engine.add_rule(check_required_fields)

# Validate with stored rules
result = engine.validate(knowledge_graph)

# Remove rule
engine.remove_rule(check_required_fields)
```

### Rule Validator

```python
from semantica.kg_qa import RuleValidator

validator = RuleValidator()

# Validate against specific rule
result = validator.validate_rule(knowledge_graph, "entity_id_rule")

# Validate against multiple rules
results = validator.validate_all_rules(
    knowledge_graph,
    ["rule1", "rule2", "rule3"]
)

for rule_name, result in results.items():
    print(f"{rule_name}: {'Valid' if result.valid else 'Invalid'}")
```

## Automated Fixes

### Fix Duplicates

```python
from semantica.kg_qa import fix_issues, AutomatedFixer

# Using convenience function
result = fix_issues(
    knowledge_graph,
    fix_type="duplicates",
    method="default"
)

if result.success:
    print(f"Fixed {result.fixed_count} duplicate(s)")
else:
    print(f"Fix failed: {result.errors}")

# Using class directly
fixer = AutomatedFixer()
result = fixer.fix_duplicates(knowledge_graph)
```

### Fix Inconsistencies

```python
from semantica.kg_qa import fix_issues

# Fix logical inconsistencies
result = fix_issues(
    knowledge_graph,
    fix_type="inconsistencies",
    method="default"
)

print(f"Fixed {result.fixed_count} inconsistency(ies)")
```

### Fix Missing Properties

```python
from semantica.kg_qa import fix_issues

schema = {
    "constraints": {
        "Person": {
            "required_props": ["id", "name", "type", "email"]
        }
    }
}

# Fix missing required properties
result = fix_issues(
    knowledge_graph,
    fix_type="missing_properties",
    schema=schema,
    method="default"
)

print(f"Added {result.fixed_count} missing property(ies)")
```

### Apply All Fixes

```python
from semantica.kg_qa import fix_issues

# Apply all automated fixes
result = fix_issues(
    knowledge_graph,
    fix_type="all",
    schema=schema,
    method="default"
)

print(f"Total fixes applied: {result.fixed_count}")
print(f"Fixes: {result.metadata.get('fixes_applied', [])}")
```

### Using Auto Merger

```python
from semantica.kg_qa import AutoMerger

merger = AutoMerger()

# Merge duplicate entities
result = merger.merge_duplicate_entities(knowledge_graph)
print(f"Merged {result.fixed_count} duplicate entity(ies)")

# Merge duplicate relationships
result = merger.merge_duplicate_relationships(knowledge_graph)
print(f"Merged {result.fixed_count} duplicate relationship(s)")

# Merge conflicting properties
result = merger.merge_conflicting_properties(knowledge_graph)
print(f"Resolved {result.fixed_count} property conflict(s)")
```

### Using Auto Resolver

```python
from semantica.kg_qa import AutoResolver

resolver = AutoResolver()

# Resolve conflicts
result = resolver.resolve_conflicts(knowledge_graph)
print(f"Resolved {result.fixed_count} conflict(s)")

# Resolve disagreements
result = resolver.resolve_disagreements(knowledge_graph)
print(f"Resolved {result.fixed_count} disagreement(s)")

# Resolve inconsistencies
result = resolver.resolve_inconsistencies(knowledge_graph)
print(f"Resolved {result.fixed_count} inconsistency(ies)")
```

## Using Methods

### Getting Available Methods

```python
from semantica.kg_qa.methods import get_qa_method, list_available_methods

# List all available methods
all_methods = list_available_methods()
print("Available methods:", all_methods)

# List methods for specific task
assess_methods = list_available_methods("assess")
print("Assessment methods:", assess_methods)

# Get specific method
assess_method = get_qa_method("assess", "default")
if assess_method:
    score = assess_method(knowledge_graph)
```

### Method Examples

```python
from semantica.kg_qa.methods import (
    assess_quality,
    generate_quality_report,
    identify_quality_issues,
    check_consistency,
    validate_completeness,
    calculate_quality_metrics,
    validate_graph,
    export_report,
    fix_issues
)

# Quality assessment
score = assess_quality(knowledge_graph, method="default")

# Report generation
report = generate_quality_report(knowledge_graph, schema, method="default")

# Issue identification
issues = identify_quality_issues(knowledge_graph, schema, method="default")

# Consistency checking
is_consistent = check_consistency(knowledge_graph, consistency_type="logical")

# Completeness validation
is_complete = validate_completeness(entities, schema, completeness_type="entity")

# Quality metrics
metrics = calculate_quality_metrics(knowledge_graph, metrics_type="overall")

# Graph validation
result = validate_graph(knowledge_graph, method="default")

# Report export
json_report = export_report(report, format="json")

# Automated fixes
fix_result = fix_issues(knowledge_graph, fix_type="duplicates")
```

## Using Registry

### Registering Custom Methods

```python
from semantica.kg_qa.registry import method_registry

# Custom quality assessment method
def custom_assessment(knowledge_graph, **kwargs):
    """Custom assessment logic."""
    # Your custom assessment code
    score = 0.85  # Calculate score
    return score

# Register custom method
method_registry.register("assess", "custom_assessment", custom_assessment)

# Use custom method
from semantica.kg_qa.methods import get_qa_method
custom_method = get_qa_method("assess", "custom_assessment")
score = custom_method(knowledge_graph)
```

### Listing Registered Methods

```python
from semantica.kg_qa.registry import method_registry

# List all registered methods
all_methods = method_registry.list_all()
print("Registered methods:", all_methods)

# List methods for specific task
assess_methods = method_registry.list_all("assess")
print("Assessment methods:", assess_methods)

consistency_methods = method_registry.list_all("consistency")
print("Consistency methods:", consistency_methods)
```

### Unregistering Methods

```python
from semantica.kg_qa.registry import method_registry

# Unregister a method
method_registry.unregister("assess", "custom_assessment")

# Clear all methods for a task
method_registry.clear("assess")

# Clear all methods
method_registry.clear()
```

## Configuration

### Using Configuration Manager

```python
from semantica.kg_qa.config import kg_qa_config

# Get configuration values
quality_threshold = kg_qa_config.get("quality_threshold", default=0.7)
consistency_threshold = kg_qa_config.get("consistency_threshold", default=0.8)
completeness_threshold = kg_qa_config.get("completeness_threshold", default=0.8)

# Set configuration values
kg_qa_config.set("quality_threshold", 0.75)
kg_qa_config.set("consistency_threshold", 0.85)

# Method-specific configuration
kg_qa_config.set_method_config("assess", quality_threshold=0.8)
assess_config = kg_qa_config.get_method_config("assess")

# Get all configuration
all_config = kg_qa_config.get_all()
print("All config:", all_config)
```

### Environment Variables

```bash
# Set environment variables
export KG_QA_QUALITY_THRESHOLD=0.75
export KG_QA_CONSISTENCY_THRESHOLD=0.85
export KG_QA_COMPLETENESS_THRESHOLD=0.80
export KG_QA_ENABLE_AUTO_FIX=true
export KG_QA_REPORT_FORMAT=json
```

### Configuration File

```yaml
# config.yaml
kg_qa:
  quality_threshold: 0.75
  consistency_threshold: 0.85
  completeness_threshold: 0.80
  enable_auto_fix: true
  report_format: json

kg_qa_methods:
  assess:
    quality_threshold: 0.8
  report:
    format: json
  consistency:
    threshold: 0.85
```

```python
from semantica.kg_qa.config import KGQAConfig

# Load from config file
config = KGQAConfig(config_file="config.yaml")
quality_threshold = config.get("quality_threshold")
```

## Advanced Examples

### Complete Quality Assurance Pipeline

```python
from semantica.kg_qa import (
    assess_quality,
    generate_quality_report,
    check_consistency,
    validate_completeness,
    fix_issues
)

# Step 1: Assess overall quality
score = assess_quality(knowledge_graph, method="default")
print(f"Initial quality score: {score:.2f}")

# Step 2: Generate comprehensive report
report = generate_quality_report(knowledge_graph, schema, method="detailed")
print(f"Report generated with {len(report.issues)} issues")

# Step 3: Check consistency
consistency_results = check_consistency(knowledge_graph, consistency_type="all")
print(f"Consistency checks: {consistency_results}")

# Step 4: Validate completeness
completeness_results = validate_completeness(
    entities=entities,
    relationships=relationships,
    schema=schema,
    completeness_type="all"
)
print(f"Completeness checks: {completeness_results}")

# Step 5: Apply automated fixes
fix_result = fix_issues(knowledge_graph, fix_type="all", schema=schema)
print(f"Fixed {fix_result.fixed_count} issue(s)")

# Step 6: Re-assess quality
final_score = assess_quality(knowledge_graph, method="default")
print(f"Final quality score: {final_score:.2f}")
print(f"Improvement: {final_score - score:.2f}")
```

### Custom Validation Rules

```python
from semantica.kg_qa import ValidationEngine

engine = ValidationEngine()

# Rule 1: Check entity IDs
def validate_entity_ids(kg):
    entities = getattr(kg, "entities", [])
    errors = []
    for entity in entities:
        if "id" not in entity:
            errors.append(f"Entity missing ID: {entity}")
    return {"error": errors[0] if errors else None}

# Rule 2: Check relationship references
def validate_relationship_refs(kg):
    entities = getattr(kg, "entities", [])
    relationships = getattr(kg, "relationships", [])
    entity_ids = {e.get("id") for e in entities}
    errors = []
    for rel in relationships:
        source = rel.get("source") or rel.get("subject")
        target = rel.get("target") or rel.get("object")
        if source not in entity_ids:
            errors.append(f"Relationship references unknown source: {source}")
        if target not in entity_ids:
            errors.append(f"Relationship references unknown target: {target}")
    return {"error": errors[0] if errors else None}

# Add rules
engine.add_rule(validate_entity_ids)
engine.add_rule(validate_relationship_refs)

# Validate
result = engine.validate(knowledge_graph)
if not result.valid:
    print(f"Validation failed: {result.errors}")
```

### Quality Monitoring Workflow

```python
from semantica.kg_qa import (
    KGQualityAssessor,
    IssueTracker,
    QualityReporter
)

assessor = KGQualityAssessor()
tracker = IssueTracker()
reporter = QualityReporter()

# Assess quality
score = assessor.assess_overall_quality(knowledge_graph)

# Generate report
report = assessor.generate_quality_report(knowledge_graph, schema)

# Track issues
for issue in report.issues:
    tracker.add_issue(issue)

# Monitor high-severity issues
high_issues = tracker.list_issues(severity="high")
print(f"High severity issues: {len(high_issues)}")

# Export report for monitoring
json_report = reporter.export_report(report, format="json")
# Save to file or send to monitoring system
```

### Batch Quality Assessment

```python
from semantica.kg_qa import assess_quality, generate_quality_report

knowledge_graphs = [
    {"entities": [...], "relationships": [...]},
    {"entities": [...], "relationships": [...]},
    {"entities": [...], "relationships": [...]}
]

# Assess multiple graphs
results = []
for kg in knowledge_graphs:
    score = assess_quality(kg, method="default")
    report = generate_quality_report(kg, schema, method="default")
    results.append({
        "score": score,
        "issues": len(report.issues),
        "completeness": report.completeness_score,
        "consistency": report.consistency_score
    })

# Analyze results
avg_score = sum(r["score"] for r in results) / len(results)
total_issues = sum(r["issues"] for r in results)
print(f"Average quality score: {avg_score:.2f}")
print(f"Total issues across graphs: {total_issues}")
```

### Integration with KG Module

```python
from semantica.kg import build
from semantica.kg_qa import assess_quality, generate_quality_report, fix_issues

# Build knowledge graph
kg = build(sources, merge_entities=True, resolve_conflicts=True)

# Assess quality
score = assess_quality(kg, method="default")
print(f"Quality after building: {score:.2f}")

# Generate report
schema = {
    "constraints": {
        "Person": {"required_props": ["id", "name", "type"]}
    }
}
report = generate_quality_report(kg, schema, method="default")

# Fix issues if needed
if score < 0.7:
    fix_result = fix_issues(kg, fix_type="all", schema=schema)
    print(f"Fixed {fix_result.fixed_count} issue(s)")
    
    # Re-assess
    final_score = assess_quality(kg, method="default")
    print(f"Quality after fixes: {final_score:.2f}")
```

### Custom Quality Metrics

```python
from semantica.kg_qa import QualityMetrics, CompletenessMetrics, ConsistencyMetrics

# Calculate individual metrics
quality_metrics = QualityMetrics()
completeness_metrics = CompletenessMetrics()
consistency_metrics = ConsistencyMetrics()

# Overall quality
overall = quality_metrics.calculate_overall_score(knowledge_graph)

# Entity quality
entities = getattr(knowledge_graph, "entities", [])
entity_quality = quality_metrics.calculate_entity_quality(entities)

# Relationship quality
relationships = getattr(knowledge_graph, "relationships", [])
rel_quality = quality_metrics.calculate_relationship_quality(relationships)

# Completeness
entity_completeness = completeness_metrics.calculate_entity_completeness(entities, schema)

# Consistency
logical_consistency = consistency_metrics.calculate_logical_consistency(knowledge_graph)
temporal_consistency = consistency_metrics.calculate_temporal_consistency(knowledge_graph)
hierarchical_consistency = consistency_metrics.calculate_hierarchical_consistency(knowledge_graph)

print(f"Overall: {overall:.2f}")
print(f"Entity Quality: {entity_quality:.2f}")
print(f"Relationship Quality: {rel_quality:.2f}")
print(f"Entity Completeness: {entity_completeness:.2f}")
print(f"Logical Consistency: {logical_consistency:.2f}")
```

### Improvement Suggestions

```python
from semantica.kg_qa import ImprovementSuggestions, generate_quality_report

# Generate report
report = generate_quality_report(knowledge_graph, schema, method="default")

# Generate improvement suggestions
suggestions_gen = ImprovementSuggestions()
suggestions = suggestions_gen.generate_suggestions(report)

print("Improvement Suggestions:")
for suggestion in suggestions:
    print(f"  - {suggestion}")
```

## Best Practices

1. **Regular Quality Assessment**: Assess quality regularly during knowledge graph construction and updates
2. **Schema-Based Validation**: Always provide schema for completeness validation
3. **Threshold Configuration**: Set appropriate thresholds based on your use case
4. **Issue Tracking**: Use IssueTracker to monitor and resolve quality issues over time
5. **Automated Fixes**: Use automated fixes for common issues, but review results
6. **Custom Rules**: Create custom validation rules for domain-specific requirements
7. **Report Export**: Export reports regularly for quality monitoring and auditing
8. **Integration**: Integrate QA checks into your KG construction pipeline
9. **Method Registry**: Register custom methods for domain-specific quality assessment
10. **Configuration Management**: Use configuration files for consistent QA settings across environments

