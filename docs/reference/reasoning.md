# Reasoning

> **Advanced inference engine supporting Rule-based, SPARQL, Abductive, and Deductive reasoning strategies.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } **Rule-Based Inference**

    ---

    Forward and Backward chaining with Rete algorithm optimization

-   :material-database-search:{ .lg .middle } **SPARQL Reasoning**

    ---

    Query expansion and property chain inference

-   :material-lightbulb-question:{ .lg .middle } **Abductive Reasoning**

    ---

    Generate hypotheses to explain observations (Sherlock Holmes style)

-   :material-check-decagram:{ .lg .middle } **Deductive Reasoning**

    ---

    Logical proof generation and theorem proving

-   :material-text-box-search:{ .lg .middle } **Explanation**

    ---

    Generate natural language explanations for inferred facts

-   :material-flash:{ .lg .middle } **Rete Algorithm**

    ---

    High-performance pattern matching for large rule sets

</div>

!!! tip "When to Use"
    - **Inference**: Deriving new facts from existing data (e.g., `Parent(A,B) & Parent(B,C) -> Grandparent(A,C)`)
    - **Query Expansion**: Finding results that aren't explicitly stored but implied
    - **Hypothesis Generation**: Finding potential causes for an observed event
    - **Validation**: Checking logical consistency of the knowledge graph

---

## ⚙️ Algorithms Used

### Rule-Based Inference
- **Forward Chaining**: Data-driven. Apply rules to facts to derive new facts until saturation.
- **Backward Chaining**: Goal-driven. Start from a goal and work backward to find supporting facts.
- **Bidirectional Chaining**: Meet-in-the-middle strategy for complex paths.

### Rete Algorithm
- **Alpha Nodes**: Filter facts by single attributes (e.g., `type=Person`).
- **Beta Nodes**: Join results from Alpha nodes (e.g., `Person.id == Parent.child_id`).
- **Memory**: Stores partial matches to avoid re-computation.
- **Conflict Resolution**: Priority-based selection when multiple rules match.

### SPARQL Reasoning
- **Query Rewriting**: Modifying queries to include inferred patterns.
- **Property Paths**: Handling transitive relationships (`foaf:knows+`).
- **Materialization**: Pre-computing inferred triples for fast read performance.

### Abductive Reasoning
- **Hypothesis Generation**: Finding rules where the conclusion matches the observation.
- **Ranking**: Scoring hypotheses by Simplicity, Plausibility, and Coverage.
- **Consistency Check**: Ensuring hypotheses don't contradict known facts.

---

## Main Classes

### InferenceEngine

General-purpose inference engine supporting multiple strategies.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `infer(facts, rules)` | Derive new facts | Forward Chaining |
| `query(goal, rules)` | Check if goal is true | Backward Chaining |

**Example:**

```python
from semantica.reasoning import InferenceEngine, Rule

rules = [
    Rule("Grandparent", "Parent(x, y) & Parent(y, z) -> Grandparent(x, z)")
]
facts = ["Parent(Alice, Bob)", "Parent(Bob, Charlie)"]

engine = InferenceEngine()
inferred = engine.infer(facts, rules)
# Result: ["Grandparent(Alice, Charlie)"]
```

### ReteEngine

High-performance pattern matching engine.

**Methods:**

| Method | Description |
|--------|-------------|
| `build_network(rules)` | Compile rule into network |
| `add_fact(fact)` | Propagate fact through network |
| `match_patterns()` | Get triggered rules |

### SPARQLReasoner

SPARQL-based reasoner for RDF graphs.

**Methods:**

| Method | Description |
|--------|-------------|
| `expand_query(query)` | Rewrite query with inference |
| `infer_results(result)` | Add inferred triples to result |

### AbductiveReasoner

Generates explanations for observations.

**Methods:**

| Method | Description |
|--------|-------------|
| `generate_hypotheses(observations)` | Generate hypotheses |
| `rank_hypotheses(hyps)` | Score and sort |

**Example:**

```python
from semantica.reasoning import AbductiveReasoner

reasoner = AbductiveReasoner(rules)
hypotheses = reasoner.generate_hypotheses(["Pavement is wet"])
# Result: ["It rained", "Sprinkler was on"]
```

### ExplanationGenerator

Explains *why* a fact was inferred.

**Methods:**

| Method | Description |
|--------|-------------|
| `generate_explanation(fact)` | Generate reasoning trace |
| `show_reasoning_path(trace)` | Graph visualization |

---

## Convenience Functions

```python
from semantica.reasoning import forward_chain, backward_chain, generate_explanation

# Quick inference
new_facts = forward_chain(facts, rules)

# Explain result
explanation = generate_explanation(new_facts[0], rules)
print(explanation.text)
```

---

## Configuration

### Environment Variables

```bash
export REASONING_MAX_ITERATIONS=100
export REASONING_STRATEGY=rete
export REASONING_TIMEOUT=30
```

### YAML Configuration

```yaml
reasoning:
  default_strategy: rete
  max_depth: 10
  
  rete:
    node_sharing: true
    
  abductive:
    max_hypotheses: 5
```

---

## Integration Examples

### Knowledge Graph Enrichment

```python
from semantica.reasoning import InferenceEngine, Rule
from semantica.kg import KnowledgeGraph

# 1. Define Ontology Rules
rules = [
    Rule("SymmetricSibling", "Sibling(x, y) -> Sibling(y, x)"),
    Rule("TransitiveAncestor", "Ancestor(x, y) & Ancestor(y, z) -> Ancestor(x, z)")
]

# 2. Load Graph
kg = KnowledgeGraph()
facts = kg.get_all_triples()

# 3. Run Inference
engine = InferenceEngine()
inferred_triples = engine.infer(facts, rules)

# 4. Update Graph
kg.add_triples(inferred_triples)
```

---

## Best Practices

1.  **Limit Recursion**: Be careful with recursive rules (e.g., `A(x,y) -> A(y,x)`) which can cause infinite loops in naive implementations.
2.  **Use Rete for Scale**: For >100 rules or >10k facts, always use the Rete engine.
3.  **Materialize vs. Query**: Materialize (pre-compute) for read-heavy workloads; Query-rewrite for write-heavy workloads.
4.  **Validate Rules**: Ensure rules are logically consistent to avoid exploding the fact space.

---

## See Also

- [Ontology Module](ontology.md) - Source of schema-based rules
- [Triplet Store Module](triplet_store.md) - Backend for SPARQL reasoning
- [Modules Guide](../modules.md#quality-assurance) - Consistency checking overview

## Cookbook

- [Reasoning and Inference](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/08_Reasoning_and_Inference.ipynb)
