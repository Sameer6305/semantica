# KG QA

> **Knowledge Graph Quality Assurance system for validation, metrics, and automated repair.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-check-decagram:{ .lg .middle } **Quality Metrics**

    ---

    Calculate Completeness, Consistency, and Accuracy scores

-   :material-shield-check:{ .lg .middle } **Validation Engine**

    ---

    Validate against schema constraints and custom rules

-   :material-wrench:{ .lg .middle } **Automated Fixes**

    ---

    Auto-repair duplicates, missing fields, and inconsistencies

-   :material-file-document-edit:{ .lg .middle } **Reporting**

    ---

    Generate detailed quality reports (JSON, HTML, YAML)

-   :material-relation-many-to-many:{ .lg .middle } **Consistency**

    ---

    Check logical, temporal, and hierarchical consistency

-   :material-lightbulb:{ .lg .middle } **Suggestions**

    ---

    Get actionable improvement suggestions

</div>

!!! tip "When to Use"
    - **Pre-Deployment**: Validate graph quality before production use
    - **Monitoring**: Continuous quality monitoring of live graphs
    - **Debugging**: Identify and fix issues in problematic graphs

---

## ⚙️ Algorithms Used

### Quality Metrics
- **Weighted Averaging**: `Score = w1*Completeness + w2*Consistency`
- **Normalization**: Min-max scaling of scores to `0.0 - 1.0`
- **Completeness Ratio**: `PresentProperties / RequiredProperties`

### Consistency Checking
- **Logical Consistency**: Contradiction detection (e.g., A > B and B > A)
- **Temporal Consistency**: Time range validation (Start < End)
- **Hierarchical Consistency**: Cycle detection in taxonomy (DFS)
- **Domain/Range**: Type compatibility checking for relationships

### Automated Fixes
- **Duplicate Merging**: Using Deduplication module strategies
- **Conflict Resolution**: Using Conflicts module strategies
- **Default Injection**: Filling missing required fields with defaults
- **Inference**: Inferring missing types or links based on topology

---

## Main Classes

### KGQualityAssessor

Coordinator for overall quality assessment.

**Methods:**

| Method | Description |
|--------|-------------|
| `assess_quality(kg)` | Calculate all metrics |
| `generate_report(kg)` | Create full report |

**Example:**

```python
from semantica.kg_qa import KGQualityAssessor

assessor = KGQualityAssessor()
score = assessor.assess_overall_quality(kg)
print(f"Graph Quality Score: {score}")
```

### ConsistencyChecker

Validates graph consistency.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `check_logical(kg)` | Logical rules | Rule Engine |
| `check_temporal(kg)` | Time validity | Range Check |
| `check_hierarchical(kg)` | Cycles/Tree | DFS |

### CompletenessValidator

Checks for missing data.

**Methods:**

| Method | Description |
|--------|-------------|
| `validate_entities(kg)` | Check entity fields |
| `validate_schema(kg)` | Check schema compliance |

### AutomatedFixer

Applies automatic repairs.

**Methods:**

| Method | Description |
|--------|-------------|
| `fix_issues(kg, issues)` | Fix reported issues |
| `merge_duplicates(kg)` | Fix duplicates |
| `resolve_conflicts(kg)` | Fix conflicts |

---

## Convenience Functions

```python
from semantica.kg_qa import assess_quality, generate_quality_report, fix_issues

# 1. Assess
score = assess_quality(kg)

# 2. Report
report = generate_quality_report(kg, schema=my_schema)

# 3. Fix
fixed_kg = fix_issues(kg, report.issues)
```

---

## Configuration

### Environment Variables

```bash
export KG_QA_MIN_SCORE=0.7
export KG_QA_STRICT_MODE=true
```

### YAML Configuration

```yaml
kg_qa:
  thresholds:
    overall: 0.7
    completeness: 0.8
    consistency: 0.9
    
  weights:
    completeness: 0.6
    consistency: 0.4
    
  auto_fix:
    enabled: true
    strategies:
      duplicates: merge
      missing_fields: default
```

---

## Integration Examples

### CI/CD Pipeline

```python
from semantica.kg_qa import assess_quality

def validate_graph_deployment(kg):
    score = assess_quality(kg)
    
    if score < 0.8:
        raise ValueError(f"Quality score {score} too low for deployment!")
        
    print("Graph passed quality checks.")
```

---

## Best Practices

1.  **Define Schema**: QA is most effective when validated against a strict schema (Ontology).
2.  **Run Regularly**: Graph quality degrades over time; run QA jobs periodically.
3.  **Review Fixes**: Automated fixes are powerful but verify them for critical data.
4.  **Handle Warnings**: Don't ignore warnings; they often indicate creeping data quality issues.

---

## See Also

- [Ontology Module](ontology.md) - Defining schemas for validation
- [Deduplication Module](deduplication.md) - Used for fixing duplicates
- [Conflicts Module](conflicts.md) - Used for resolving inconsistencies
