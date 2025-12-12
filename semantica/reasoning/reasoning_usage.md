# Reasoning and Inference Module Usage Guide

This comprehensive guide demonstrates how to use the reasoning and inference module for rule-based inference, SPARQL reasoning, Rete algorithm matching, abductive and deductive reasoning, rule management, and explanation generation.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Rule-Based Inference](#rule-based-inference)
3. [SPARQL Reasoning](#sparql-reasoning)
4. [Rete Algorithm](#rete-algorithm)
5. [Abductive Reasoning](#abductive-reasoning)
6. [Deductive Reasoning](#deductive-reasoning)
7. [Rule Management](#rule-management)
8. [Explanation Generation](#explanation-generation)
9. [Algorithms and Methods](#algorithms-and-methods)
10. [Configuration](#configuration)
11. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using Main Classes

```python
from semantica.reasoning import InferenceEngine, SPARQLReasoner, ReteEngine

# Create inference engine
engine = InferenceEngine()

# Add facts
engine.add_fact("Person(John)")
engine.add_fact("Person(Jane)")

# Add rule
rule = engine.add_rule("IF Person(?x) THEN Human(?x)")

# Perform forward chaining
results = engine.forward_chain()

print(f"Inferred {len(results)} new facts")
```

### Using SPARQL Reasoner

```python
from semantica.reasoning import SPARQLReasoner

# Create SPARQL reasoner
reasoner = SPARQLReasoner(triplet_store=kg)

# Execute query
query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
result = reasoner.execute_query(query)

print(f"Found {len(result.bindings)} results")
```

### Using Rete Engine

```python
from semantica.reasoning import ReteEngine, Fact

# Create Rete engine
rete = ReteEngine()

# Add rules
rete.add_rule(rule1)
rete.add_rule(rule2)

# Add facts
fact = Fact("f1", "Person", ["John"])
rete.add_fact(fact)

# Get matches
matches = rete.match_patterns()
print(f"Found {len(matches)} matches")
```

## Rule-Based Inference

### Forward Chaining

Forward chaining is a data-driven inference strategy that starts with known facts and applies rules to derive new conclusions.

```python
from semantica.reasoning import InferenceEngine, InferenceStrategy

# Create inference engine
engine = InferenceEngine(strategy=InferenceStrategy.FORWARD)

# Add initial facts
engine.add_fact("Person(John)")
engine.add_fact("Person(Jane)")
engine.add_fact("WorksFor(John, Acme)")

# Add rules
engine.add_rule("IF Person(?x) THEN Human(?x)")
engine.add_rule("IF WorksFor(?x, ?y) AND Person(?x) THEN Employee(?x, ?y)")

# Perform forward chaining
results = engine.forward_chain()

for result in results:
    print(f"Inferred: {result.conclusion}")
    print(f"Using rule: {result.rule_used.name}")
    print(f"Confidence: {result.confidence}")
```

### Backward Chaining

Backward chaining is a goal-driven inference strategy that starts with a goal and works backwards to find supporting facts.

```python
from semantica.reasoning import InferenceEngine

engine = InferenceEngine()

# Add facts
engine.add_fact("Person(John)")
engine.add_fact("WorksFor(John, Acme)")

# Add rules
engine.add_rule("IF Person(?x) THEN Human(?x)")
engine.add_rule("IF WorksFor(?x, ?y) AND Person(?x) THEN Employee(?x, ?y)")

# Perform backward chaining to prove goal
goal = "Employee(John, Acme)"
result = engine.backward_chain(goal)

if result:
    print(f"Goal '{goal}' is provable")
    print(f"Proof: {result.premises} -> {result.conclusion}")
else:
    print(f"Goal '{goal}' cannot be proven")
```

### Bidirectional Chaining

Bidirectional chaining combines forward and backward chaining for more efficient inference.

```python
from semantica.reasoning import InferenceEngine, InferenceStrategy

engine = InferenceEngine(strategy=InferenceStrategy.BIDIRECTIONAL)

# Add facts and rules
engine.add_fact("Person(John)")
engine.add_rule("IF Person(?x) THEN Human(?x)")

# Perform bidirectional inference
goal = "Human(John)"
results = engine.infer(goal)

print(f"Found {len(results)} inference paths")
```

### Inference with Custom Rules

```python
from semantica.reasoning import InferenceEngine, Rule, RuleType

engine = InferenceEngine()

# Create custom rule
rule = Rule(
    rule_id="r1",
    name="TransitiveLocation",
    conditions=["LocatedIn(?x, ?y)", "LocatedIn(?y, ?z)"],
    conclusion="LocatedIn(?x, ?z)",
    rule_type=RuleType.IMPLICATION,
    confidence=0.9,
    priority=1
)

engine.add_rule(rule)

# Add facts
engine.add_fact("LocatedIn(Paris, France)")
engine.add_fact("LocatedIn(France, Europe)")

# Infer
results = engine.forward_chain()
# Will infer: LocatedIn(Paris, Europe)
```

## SPARQL Reasoning

### Basic SPARQL Query

```python
from semantica.reasoning import SPARQLReasoner

# Create reasoner with knowledge graph
reasoner = SPARQLReasoner(triplet_store=kg)

# Execute SPARQL query
query = """
SELECT ?person ?company
WHERE {
    ?person :worksFor ?company .
    ?person :type :Person .
}
"""

result = reasoner.execute_query(query)

for binding in result.bindings:
    print(f"Person: {binding.get('person')}, Company: {binding.get('company')}")
```

### Query Expansion with Inference Rules

```python
from semantica.reasoning import SPARQLReasoner

reasoner = SPARQLReasoner(triplet_store=kg, enable_inference=True)

# Add inference rule
reasoner.add_inference_rule("IF ?x :type :Company THEN ?x :type :Organization")

# Original query
query = "SELECT ?x WHERE { ?x :type :Organization }"

# Query is automatically expanded with inference rules
result = reasoner.execute_query(query)

# Results include both explicit :Organization types and inferred from :Company
```



### SPARQL with RDF Inference

```python
from semantica.reasoning import SPARQLReasoner

reasoner = SPARQLReasoner(
    triplet_store=kg,
    enable_inference=True,
    inference_rules=["rdfs:subClassOf", "rdfs:subPropertyOf"]
)

# Query will use RDFS inference
query = """
SELECT ?person
WHERE {
    ?person a :Employee .
}
"""

# Will also match :Person if :Employee rdfs:subClassOf :Person
result = reasoner.execute_query(query)
```

## Rete Algorithm

### Building Rete Network

```python
from semantica.reasoning import ReteEngine, Rule, RuleType

# Create Rete engine
rete = ReteEngine()

# Define rules
rule1 = Rule(
    rule_id="r1",
    name="PersonRule",
    conditions=["Person(?x)"],
    conclusion="Human(?x)",
    rule_type=RuleType.IMPLICATION
)

rule2 = Rule(
    rule_id="r2",
    name="EmployeeRule",
    conditions=["WorksFor(?x, ?y)", "Person(?x)"],
    conclusion="Employee(?x, ?y)",
    rule_type=RuleType.IMPLICATION
)

# Build network from rules
rete.build_network([rule1, rule2])

print(f"Built network with {len(rete.network)} nodes")
```

### Pattern Matching with Rete

```python
from semantica.reasoning import ReteEngine, Fact

rete = ReteEngine()
rete.build_network([rule1, rule2])

# Add facts
fact1 = Fact("f1", "Person", ["John"])
fact2 = Fact("f2", "WorksFor", ["John", "Acme"])

rete.add_fact(fact1)
rete.add_fact(fact2)

# Get matches
matches = rete.match_patterns()

for match in matches:
    print(f"Rule: {match.rule.name}")
    print(f"Facts: {[f.fact_id for f in match.facts]}")
    print(f"Bindings: {match.bindings}")
```

### Incremental Fact Processing

```python
from semantica.reasoning import ReteEngine, Fact

rete = ReteEngine()
rete.build_network([rule1, rule2])

# Add facts incrementally
fact1 = Fact("f1", "Person", ["John"])
rete.add_fact(fact1)
# Get matches
matches = rete.match_patterns()

fact2 = Fact("f2", "WorksFor", ["John", "Acme"])
rete.add_fact(fact2)
# Now includes matches for rule2
matches2 = rete.match_patterns()

# Reset engine (clears facts and matches)
rete.reset()
```

### Rete Network Optimization

```python
from semantica.reasoning import ReteEngine

rete = ReteEngine()

# Build network
rete.build_network(rules)

# Network automatically optimizes:
# - Shared alpha nodes for common conditions
# - Efficient join ordering
# - Redundant node elimination

print(f"Network nodes: {len(rete.network)}")
print(f"Alpha nodes: {sum(1 for n in rete.network.values() if isinstance(n, AlphaNode))}")
print(f"Beta nodes: {sum(1 for n in rete.network.values() if isinstance(n, BetaNode))}")
```

## Abductive Reasoning

### Hypothesis Generation

```python
from semantica.reasoning import AbductiveReasoner, Observation

# Create abductive reasoner
reasoner = AbductiveReasoner()

# Define observation
observation = Observation(
    observation_id="obs1",
    description="Person X is in location Y",
    facts=["LocatedIn(PersonX, LocationY)"]
)

# Generate hypotheses
hypotheses = reasoner.generate_hypotheses([observation])

for hypothesis in hypotheses:
    print(f"Hypothesis: {hypothesis.explanation}")
    print(f"Confidence: {hypothesis.confidence}")
    print(f"Coverage: {hypothesis.coverage}")
    print(f"Simplicity: {hypothesis.simplicity}")
```

### Hypothesis Ranking

```python
from semantica.reasoning import AbductiveReasoner, HypothesisRanking

reasoner = AbductiveReasoner()

# Generate hypotheses
hypotheses = reasoner.generate_hypotheses(observations)

# Rank by simplicity
ranked_simplicity = reasoner.rank_hypotheses(
    hypotheses,
    ranking_strategy=HypothesisRanking.SIMPLICITY
)

# Rank by plausibility
ranked_plausibility = reasoner.rank_hypotheses(
    hypotheses,
    ranking_strategy=HypothesisRanking.PLAUSIBILITY
)

# Rank by coverage
ranked_coverage = reasoner.rank_hypotheses(
    hypotheses,
    ranking_strategy=HypothesisRanking.COVERAGE
)

# Get best hypothesis
best = ranked_plausibility[0]
print(f"Best explanation: {best.explanation}")
```

### Finding Best Explanation

```python
from semantica.reasoning import AbductiveReasoner, Observation

reasoner = AbductiveReasoner()

# Find best explanation for observation
observation = Observation(
    observation_id="obs1",
    description="The ground is wet",
    facts=["Wet(Ground)"]
)

explanations = reasoner.find_explanations([observation])

for explanation in explanations:
    print(f"Observation: {explanation.observation.description}")
    print(f"Best hypothesis: {explanation.best_hypothesis.explanation}")
    print(f"Confidence: {explanation.confidence}")
```

### Abductive Reasoning with Rules

```python
from semantica.reasoning import AbductiveReasoner, Observation, Rule, RuleType

reasoner = AbductiveReasoner()

# Add rules that can explain observations
rule = Rule(
    rule_id="r1",
    name="RainRule",
    conditions=["Raining"],
    conclusion="Wet(Ground)",
    rule_type=RuleType.IMPLICATION
)

reasoner.rule_manager.add_rule(rule)

# Generate hypotheses
observation = Observation("obs1", "The ground is wet", ["Wet(Ground)"])
hypotheses = reasoner.generate_hypotheses([observation])

# Will generate hypothesis: "Raining" explains "Wet(Ground)"
```

## Deductive Reasoning

### Basic Deductive Inference

```python
from semantica.reasoning import DeductiveReasoner, Premise

# Create deductive reasoner
reasoner = DeductiveReasoner()

# Define premises
premises = [
    Premise("p1", "All humans are mortal", confidence=1.0),
    Premise("p2", "Socrates is human", confidence=1.0)
]

# Apply logical inference
conclusions = reasoner.apply_logic(premises)

for conclusion in conclusions:
    print(f"Conclusion: {conclusion.statement}")
    print(f"From premises: {[p.statement for p in conclusion.premises]}")
    print(f"Confidence: {conclusion.confidence}")
```

### Proof Generation

```python
from semantica.reasoning import DeductiveReasoner, Premise

reasoner = DeductiveReasoner()

# Define premises
premises = [
    Premise("p1", "All humans are mortal"),
    Premise("p2", "Socrates is human")
]

# Generate proof for conclusion
conclusion_statement = "Socrates is mortal"
proof = reasoner.prove_theorem(conclusion_statement)

if proof:
    print(f"Theorem: {proof.theorem}")
    print(f"Valid: {proof.valid}")
    print(f"Proof steps: {len(proof.steps)}")
    for step in proof.steps:
        print(f"  Step: {step.statement}")
        print(f"    Rule: {step.rule_applied.name if step.rule_applied else 'N/A'}")
```

### Theorem Proving

```python
from semantica.reasoning import DeductiveReasoner

reasoner = DeductiveReasoner()

# Prove theorem
theorem = "Socrates is mortal"
proof = reasoner.prove_theorem(theorem)

if proof and proof.valid:
    print(f"Theorem '{theorem}' is provable")
    print(f"Proof steps: {len(proof.steps)}")
    for i, step in enumerate(proof.steps, 1):
        print(f"{i}. {step.statement}")
else:
    print(f"Theorem '{theorem}' cannot be proven")
```

### Logical Argument Construction

```python
from semantica.reasoning import DeductiveReasoner, Premise, Argument

reasoner = DeductiveReasoner()

# Build argument
premises = [
    Premise("p1", "If it rains, the ground gets wet"),
    Premise("p2", "It is raining")
]

conclusion = reasoner.apply_logic(premises)[0]

# Validate argument
argument = Argument(
    argument_id="arg1",
    premises=premises,
    conclusion=conclusion
)

is_valid = reasoner.validate_argument(argument)
print(f"Argument is valid: {is_valid}")
```

## Rule Management

### Defining Rules

```python
from semantica.reasoning import RuleManager, RuleType

# Create rule manager
manager = RuleManager()

# Define rule using natural language
rule = manager.define_rule(
    "IF Person(?x) AND WorksFor(?x, ?y) THEN Employee(?x, ?y)",
    name="EmployeeRule",
    confidence=0.9,
    priority=1
)

print(f"Rule ID: {rule.rule_id}")
print(f"Conditions: {rule.conditions}")
print(f"Conclusion: {rule.conclusion}")
```

### Rule Types

```python
from semantica.reasoning import RuleManager, RuleType

manager = RuleManager()

# Implication rule (IF-THEN)
implication = manager.define_rule(
    "IF Person(?x) THEN Human(?x)",
    rule_type=RuleType.IMPLICATION
)

# Equivalence rule (IFF)
equivalence = manager.define_rule(
    "Person(?x) IFF Human(?x)",
    rule_type=RuleType.EQUIVALENCE
)

# Constraint rule
constraint = manager.define_rule(
    "IF Person(?x) THEN Age(?x) >= 0",
    rule_type=RuleType.CONSTRAINT
)

# Transformation rule
transformation = manager.define_rule(
    "IF Name(?x, ?n) THEN NormalizedName(?x, ?n.lower())",
    rule_type=RuleType.TRANSFORMATION
)
```

### Rule Execution

```python
from semantica.reasoning import RuleManager

manager = RuleManager()

# Add rule
rule = manager.define_rule("IF Person(?x) THEN Human(?x)")

# Execute rule with facts
facts = ["Person(John)", "Person(Jane)"]
execution = manager.execute_rule(rule, facts)

print(f"Execution successful: {execution.success}")
print(f"Output fact: {execution.output_fact}")
print(f"Execution time: {execution.execution_time}s")
```

### Rule Validation

```python
from semantica.reasoning import RuleManager

manager = RuleManager()

# Define rule
rule = manager.define_rule("IF Person(?x) THEN Human(?x)")

# Validate rule
validation = manager.validate_rule(rule)

if validation["valid"]:
    print("Rule is valid")
else:
    print(f"Validation errors: {validation['errors']}")
    print(f"Warnings: {validation.get('warnings', [])}")
```

### Rule Execution History

```python
from semantica.reasoning import RuleManager

manager = RuleManager()

# Execute multiple rules
for rule in rules:
    manager.execute_rule(rule, facts)

# Get execution history
history = manager.get_execution_history()

print(f"Total executions: {len(history)}")
for execution in history:
    print(f"Rule: {execution.rule_id}")
    print(f"Success: {execution.success}")
    print(f"Time: {execution.execution_time}s")
```

## Explanation Generation

### Generating Explanations

```python
from semantica.reasoning import ExplanationGenerator, InferenceEngine

# Create inference engine and perform inference
engine = InferenceEngine()
results = engine.forward_chain()

# Generate explanation for inference result
generator = ExplanationGenerator()
explanation = generator.generate_explanation(results[0])

print(f"Explanation type: {explanation.explanation_type}")
print(f"Conclusion: {explanation.conclusion}")
print(f"Natural language: {explanation.natural_language}")
```

### Reasoning Path Generation

```python
from semantica.reasoning import ExplanationGenerator

generator = ExplanationGenerator()

# Generate reasoning path
path = generator.show_reasoning_path(inference_result)

print(f"Path ID: {path.path_id}")
print(f"Steps: {len(path.steps)}")
print(f"Start facts: {path.start_facts}")
print(f"End conclusion: {path.end_conclusion}")
print(f"Total confidence: {path.total_confidence}")

for i, step in enumerate(path.steps, 1):
    print(f"\nStep {i}:")
    print(f"  Description: {step.description}")
    print(f"  Rule: {step.rule_applied.name if step.rule_applied else 'N/A'}")
    print(f"  Input facts: {step.input_facts}")
    print(f"  Output fact: {step.output_fact}")
```

### Justification Creation

```python
from semantica.reasoning import ExplanationGenerator

generator = ExplanationGenerator()

# Create justification
justification = generator.justify_conclusion(conclusion, reasoning_path)

print(f"Justification ID: {justification.justification_id}")
print(f"Conclusion: {justification.conclusion}")
print(f"Supporting evidence: {justification.supporting_evidence}")
print(f"Confidence: {justification.confidence}")
print(f"Explanation: {justification.explanation_text}")
```

### Natural Language Explanations

```python
from semantica.reasoning import ExplanationGenerator

generator = ExplanationGenerator(generate_nl=True, detail_level="detailed")

# Generate natural language explanation
explanation = generator.generate_explanation(inference_result)

print("Natural Language Explanation:")
print(explanation.natural_language)

# Generate for different detail levels
simple_explanation = generator.generate_explanation(
    inference_result,
    detail_level="simple"
)

verbose_explanation = generator.generate_explanation(
    inference_result,
    detail_level="verbose"
)
```

### Explanation for Different Reasoning Types

```python
from semantica.reasoning import ExplanationGenerator

generator = ExplanationGenerator()

# Explain inference result
inference_explanation = generator.generate_explanation(inference_result)

# Explain proof
proof_explanation = generator.generate_explanation(proof)

# Explain abductive explanation
abductive_explanation = generator.generate_explanation(abductive_result)

# Each explanation type has appropriate structure
print(f"Inference explanation: {inference_explanation.explanation_type}")
print(f"Proof explanation: {proof_explanation.explanation_type}")
print(f"Abductive explanation: {abductive_explanation.explanation_type}")
```

## Algorithms and Methods

### Rule-Based Inference Algorithms

#### Forward Chaining
**Algorithm**: Iterative rule application with fact propagation

1. **Initialization**: Start with initial fact set
2. **Iteration**: While new facts are derived:
   - For each rule:
     - Check if all conditions are satisfied by current facts
     - If yes, add conclusion to fact set
     - Mark as new fact
3. **Termination**: Stop when no new facts are derived or max iterations reached
4. **Conflict Resolution**: If multiple rules can fire, select by priority

**Time Complexity**: O(r × f × i) where r = rules, f = facts, i = iterations
**Space Complexity**: O(f) for fact storage

```python
# Forward chaining example
engine = InferenceEngine()
engine.add_facts(["A", "B"])
engine.add_rule("IF A AND B THEN C")
engine.add_rule("IF C THEN D")
results = engine.forward_chain()
# Results: [C, D]
```

#### Backward Chaining
**Algorithm**: Goal-driven recursive proof search

1. **Goal Check**: If goal is in facts, return success
2. **Rule Selection**: Find rules whose conclusions match goal
3. **Recursive Proof**: For each applicable rule:
   - Recursively prove each premise
   - If all premises proven, return success
4. **Backtracking**: If proof fails, try next rule
5. **Termination**: Return when goal proven or all paths exhausted

**Time Complexity**: O(r^d) where r = rules, d = proof depth
**Space Complexity**: O(d) for recursion stack

```python
# Backward chaining example
engine = InferenceEngine()
engine.add_facts(["A", "B"])
engine.add_rule("IF A AND B THEN C")
engine.add_rule("IF C THEN D")
result = engine.backward_chain("D")
# Proves: D <- C <- (A AND B)
```

#### Bidirectional Chaining
**Algorithm**: Meet-in-the-middle strategy

1. **Forward Pass**: Start from facts, apply rules forward
2. **Backward Pass**: Start from goal, apply rules backward
3. **Intersection**: Check if forward and backward paths meet
4. **Path Construction**: Combine forward and backward paths
5. **Optimization**: Use heuristics to guide search

**Time Complexity**: O(r × f + r^d) combining forward and backward
**Space Complexity**: O(f + d) for both passes

### Rete Algorithm

#### Network Construction
**Algorithm**: Build Rete network from rules

1. **Alpha Node Creation**: Create one alpha node per unique condition pattern
2. **Beta Node Joins**: Create beta nodes to join alpha node outputs
3. **Terminal Node Activation**: Create terminal node for each rule
4. **Network Optimization**: Share common alpha nodes, eliminate redundancy

**Time Complexity**: O(r × c) where r = rules, c = conditions per rule
**Space Complexity**: O(r × c) for network storage

```python
# Rete network construction
rete = ReteEngine()
rete.build_network([rule1, rule2])
# Creates optimized network with shared nodes
```

#### Pattern Matching
**Algorithm**: Incremental fact processing through network

1. **Fact Addition**: Add fact to working memory
2. **Alpha Matching**: Match fact against alpha node conditions
3. **Beta Joins**: Join alpha matches through beta nodes
4. **Terminal Activation**: Activate terminal nodes when all conditions matched
5. **Match Collection**: Collect all terminal node activations

**Time Complexity**: O(n × m) where n = facts, m = network nodes
**Space Complexity**: O(n + m) for matches and network

```python
# Pattern matching
rete.add_fact(Fact("f1", "Person", ["John"]))
matches = rete.match_patterns()
# Efficiently finds all rule matches
```

#### Join Operations
**Algorithm**: Beta node join with unification

1. **Left Input**: Receive matches from left child
2. **Right Input**: Receive matches from right child
3. **Unification**: Check variable bindings are consistent
4. **Join**: Create joined matches with unified bindings
5. **Propagation**: Send joined matches to children

**Time Complexity**: O(l × r) where l = left matches, r = right matches
**Space Complexity**: O(l × r) for join results

### SPARQL Reasoning Algorithms

#### Query Expansion
**Algorithm**: Add inferred patterns to SPARQL query

1. **Rule Conversion**: Convert inference rules to SPARQL patterns
2. **Pattern Matching**: Find query patterns that match rule conditions
3. **Pattern Addition**: Add inferred patterns to WHERE clause
4. **Optimization**: Remove redundant patterns

**Time Complexity**: O(r × q) where r = rules, q = query patterns
**Space Complexity**: O(q) for expanded query

```python
# Query expansion
reasoner = SPARQLReasoner(enable_inference=True)
reasoner.add_inference_rule("IF ?x :type :Company THEN ?x :type :Organization")
expanded = reasoner.expand_query("SELECT ?x WHERE { ?x :type :Organization }")
# Adds inferred :Company -> :Organization pattern
```

#### Query Optimization
**Algorithm**: Optimize SPARQL query execution

1. **Plan Generation**: Create logical query plan
2. **Join Ordering**: Order joins by selectivity
3. **Filter Pushdown**: Move filters as early as possible
4. **Projection Pushdown**: Select only needed variables

**Time Complexity**: O(q²) for join ordering
**Space Complexity**: O(q) for query plan

### Abductive Reasoning Algorithms

#### Hypothesis Generation
**Algorithm**: Generate candidate hypotheses from rules

1. **Observation Analysis**: Analyze observation to find explainable parts
2. **Rule Matching**: Find rules whose conclusions match observation
3. **Hypothesis Construction**: Create hypothesis from rule premises
4. **Hypothesis Expansion**: Generate variations of hypotheses

**Time Complexity**: O(r × o) where r = rules, o = observations
**Space Complexity**: O(h) where h = hypotheses

```python
# Hypothesis generation
reasoner = AbductiveReasoner()
hypotheses = reasoner.generate_hypotheses(observations)
# Generates all possible explanations
```

#### Hypothesis Ranking
**Algorithm**: Rank hypotheses by multiple criteria

1. **Simplicity Scoring**: Score by number of conditions (inverse)
2. **Plausibility Assessment**: Score by confidence and evidence
3. **Consistency Checking**: Check consistency with known facts
4. **Coverage Calculation**: Measure how well hypothesis explains observation
5. **Weighted Combination**: Combine scores with weights

**Time Complexity**: O(h × f) where h = hypotheses, f = facts
**Space Complexity**: O(h) for rankings

```python
# Hypothesis ranking
ranked = reasoner.rank_hypotheses(
    hypotheses,
    ranking_strategy=HypothesisRanking.PLAUSIBILITY
)
# Ranks by plausibility score
```

### Deductive Reasoning Algorithms

#### Logical Inference
**Algorithm**: Apply logical inference rules

1. **Rule Matching**: Find applicable inference rules
2. **Premise Matching**: Match rule premises to known facts
3. **Conclusion Generation**: Generate conclusion from matched rule
4. **Confidence Calculation**: Calculate confidence from premises

**Inference Rules**:
- Modus Ponens: If P→Q and P, then Q
- Modus Tollens: If P→Q and ¬Q, then ¬P
- Syllogism: If A→B and B→C, then A→C

**Time Complexity**: O(r × p) where r = rules, p = premises
**Space Complexity**: O(c) where c = conclusions

```python
# Logical inference
reasoner = DeductiveReasoner()
conclusions = reasoner.apply_logic(premises)
# Applies modus ponens, syllogism, etc.
```

#### Proof Generation
**Algorithm**: Construct proof tree for conclusion

1. **Goal Decomposition**: Break goal into subgoals
2. **Subgoal Resolution**: Recursively prove subgoals
3. **Proof Tree Construction**: Build tree of inference steps
4. **Proof Validation**: Verify proof correctness

**Time Complexity**: O(r^d) where d = proof depth
**Space Complexity**: O(d) for proof tree

```python
# Proof generation
proof = reasoner.prove_theorem(conclusion)
# Constructs step-by-step proof
```

### Rule Management Algorithms

#### Rule Parsing
**Algorithm**: Parse natural language rule definitions

1. **Pattern Matching**: Match rule syntax patterns (IF-THEN, IFF, etc.)
2. **Condition Extraction**: Extract conditions from rule string
3. **Conclusion Extraction**: Extract conclusion from rule string
4. **Type Detection**: Detect rule type from syntax

**Time Complexity**: O(n) where n = rule string length
**Space Complexity**: O(c) where c = conditions

```python
# Rule parsing
rule = manager.define_rule("IF Person(?x) THEN Human(?x)")
# Parses and creates Rule object
```

#### Rule Validation
**Algorithm**: Validate rule structure and semantics

1. **Syntax Validation**: Check rule syntax correctness
2. **Semantic Validation**: Check rule semantics
3. **Consistency Checking**: Check consistency with existing rules
4. **Completeness Checking**: Check if rule is well-formed

**Time Complexity**: O(r) where r = existing rules
**Space Complexity**: O(1)

### Explanation Generation Algorithms

#### Reasoning Path Construction
**Algorithm**: Build step-by-step reasoning path

1. **Step Extraction**: Extract inference steps from result
2. **Path Building**: Build path from initial facts to conclusion
3. **Step Linking**: Link steps by input/output relationships
4. **Path Optimization**: Simplify path for readability

**Time Complexity**: O(s) where s = steps
**Space Complexity**: O(s) for path storage

```python
# Reasoning path construction
path = generator.generate_reasoning_path(inference_result)
# Builds complete reasoning chain
```

#### Natural Language Generation
**Algorithm**: Convert reasoning path to natural language

1. **Template Selection**: Select appropriate explanation template
2. **Step Narration**: Convert each step to natural language
3. **Path Description**: Describe overall reasoning path
4. **Conclusion Summary**: Summarize final conclusion

**Time Complexity**: O(s) where s = steps
**Space Complexity**: O(n) where n = natural language text

### Methods

#### InferenceEngine Methods

- `infer(query, **options)`: General inference with strategy selection
- `forward_chain(facts, **options)`: Forward chaining inference
- `backward_chain(goal, **options)`: Backward chaining inference
- `add_rule(rule_definition, **options)`: Add inference rule
- `add_fact(fact)`: Add fact to knowledge base
- `add_facts(facts)`: Add multiple facts

#### SPARQLReasoner Methods

- `query(sparql_query, **options)`: Execute SPARQL query
- `expand_query(query, **options)`: Expand query with inference rules
- `add_inference_rule(rule)`: Add inference rule for query expansion
- `optimize_query(query)`: Optimize SPARQL query

#### ReteEngine Methods

- `build_network(rules)`: Build Rete network from rules
- `add_rule(rule)`: Add rule to network
- `add_fact(fact)`: Add fact and propagate through network
- `match_patterns()`: Get all rule matches
- `match_patterns(facts)`: Match patterns using Rete algorithm

#### AbductiveReasoner Methods

- `generate_hypotheses(observations, **options)`: Generate candidate hypotheses
- `rank_hypotheses(hypotheses, ranking_strategy)`: Rank hypotheses
- `find_explanations(observations, **options)`: Find best explanations
- `evaluate_hypothesis(hypothesis, facts)`: Evaluate hypothesis quality

#### DeductiveReasoner Methods

- `apply_logic(premises, **options)`: Apply logical inference rules
- `prove_theorem(theorem)`: Prove theorem
- `prove_theorem(theorem, **options)`: Prove logical theorem
- `validate_argument(argument)`: Validate logical argument

#### RuleManager Methods

- `define_rule(rule_definition, **options)`: Define new rule
- `add_rule(rule)`: Add rule to manager
- `execute_rule(rule, facts)`: Execute rule with facts
- `validate_rule(rule)`: Validate rule structure
- `get_execution_history()`: Get rule execution history
- `get_all_rules()`: Get all registered rules

#### ExplanationGenerator Methods

- `generate_explanation(reasoning, **options)`: Generate explanation for result
- `generate_reasoning_path(reasoning)`: Generate reasoning path
- `create_justification(conclusion, path)`: Create justification
- `generate_natural_language(explanation)`: Generate natural language explanation

## Configuration

### Environment Variables

```bash
# Reasoning configuration
export REASONING_DEFAULT_STRATEGY=forward
export REASONING_MAX_ITERATIONS=100
export REASONING_CONFIDENCE_THRESHOLD=0.7
export REASONING_ENABLE_EXPLANATION=true
export REASONING_RETE_OPTIMIZATION=true

# SPARQL reasoning configuration
export REASONING_SPARQL_ENDPOINT=http://localhost:3030/ds
export REASONING_ENABLE_INFERENCE=true
export REASONING_QUERY_CACHE_SIZE=1000

# Abductive reasoning configuration
export REASONING_MAX_HYPOTHESES=10
export REASONING_RANKING_STRATEGY=plausibility

# Explanation configuration
export REASONING_GENERATE_NL=true
export REASONING_DETAIL_LEVEL=detailed
```

### Programmatic Configuration

```python
from semantica.reasoning import InferenceEngine, SPARQLReasoner

# Configure inference engine
engine = InferenceEngine(
    strategy="forward",
    max_iterations=100,
    confidence_threshold=0.7
)

# Configure SPARQL reasoner
reasoner = SPARQLReasoner(
    triplet_store=kg,
    enable_inference=True,
    query_cache_size=1000
)
```

### Configuration File (YAML)

```yaml
# config.yaml
reasoning:
  default_strategy: forward
  max_iterations: 100
  confidence_threshold: 0.7
  enable_explanation: true
  rete_optimization: true

reasoning_sparql:
  endpoint: http://localhost:3030/ds
  enable_inference: true
  query_cache_size: 1000

reasoning_abductive:
  max_hypotheses: 10
  ranking_strategy: plausibility

reasoning_explanation:
  generate_nl: true
  detail_level: detailed
```

## Advanced Examples

### Complete Knowledge Graph Reasoning Pipeline

```python
from semantica.reasoning import (
    InferenceEngine,
    SPARQLReasoner,
    ExplanationGenerator
)

# 1. Set up inference engine
engine = InferenceEngine(strategy="forward")

# 2. Add facts from knowledge graph
kg_facts = extract_facts_from_kg(kg)
engine.add_facts(kg_facts)

# 3. Add domain rules
rules = [
    "IF Person(?x) AND WorksFor(?x, ?y) THEN Employee(?x, ?y)",
    "IF Employee(?x, ?y) AND Company(?y) THEN WorksAt(?x, ?y)",
    "IF Person(?x) THEN Human(?x)"
]
for rule in rules:
    engine.add_rule(rule)

# 4. Perform inference
results = engine.forward_chain()

# 5. Generate explanations
generator = ExplanationGenerator()
for result in results:
    explanation = generator.generate_explanation(result)
    print(f"Conclusion: {result.conclusion}")
    print(f"Explanation: {explanation.natural_language}")

# 6. Query with SPARQL reasoning
sparql_reasoner = SPARQLReasoner(triplet_store=kg, enable_inference=True)
query_result = sparql_reasoner.execute_query("SELECT ?x WHERE { ?x :type :Employee }")
```

### Abductive Explanation System

```python
from semantica.reasoning import AbductiveReasoner, Observation

# Create reasoner
reasoner = AbductiveReasoner(
    max_hypotheses=10,
    ranking_strategy=HypothesisRanking.PLAUSIBILITY
)

# Define observations
observations = [
    Observation("obs1", "The ground is wet", ["Wet(Ground)"]),
    Observation("obs2", "The car won't start", ["NotWorking(Car)"])
]

# Find best explanations
explanations = reasoner.find_explanations(observations)

for explanation in explanations:
    print(f"Observation: {explanation.observation.description}")
    print(f"Best explanation: {explanation.best_hypothesis.explanation}")
    print(f"Confidence: {explanation.confidence}")
    print(f"Alternative hypotheses: {len(explanation.hypotheses) - 1}")
```

### Rete-Based Rule System

```python
from semantica.reasoning import ReteEngine, Fact, Rule, RuleType

# Create Rete engine
rete = ReteEngine()

# Define complex rules
rules = [
    Rule("r1", "PersonRule", ["Person(?x)"], "Human(?x)", RuleType.IMPLICATION),
    Rule("r2", "EmployeeRule", ["WorksFor(?x, ?y)", "Person(?x)"], "Employee(?x, ?y)", RuleType.IMPLICATION),
    Rule("r3", "ManagerRule", ["Manages(?x, ?y)", "Employee(?x, ?y)"], "Manager(?x)", RuleType.IMPLICATION)
]

# Build network
rete.build_network(rules)

# Add facts incrementally
facts = [
    Fact("f1", "Person", ["John"]),
    Fact("f2", "WorksFor", ["John", "Acme"]),
    Fact("f3", "Manages", ["John", "Team1"])
]

for fact in facts:
    rete.add_fact(fact)
    matches = rete.match_patterns()
    print(f"After adding {fact.fact_id}: {len(matches)} matches")
```

### Theorem Proving System

```python
from semantica.reasoning import DeductiveReasoner, Premise

reasoner = DeductiveReasoner()

# Define axioms
axioms = [
    Premise("ax1", "All humans are mortal"),
    Premise("ax2", "Socrates is human")
]

# Prove theorem
theorem = "Socrates is mortal"
proof = reasoner.prove_theorem(theorem)

if proof and proof.valid:
    print(f"Theorem '{theorem}' is provable")
    print("Proof steps:")
    for i, step in enumerate(proof.steps, 1):
        print(f"{i}. {step.statement}")
        if step.rule_applied:
            print(f"   (Using rule: {step.rule_applied.name})")
```

### Integration with Knowledge Graph

```python
from semantica.reasoning import InferenceEngine, SPARQLReasoner
from semantica.kg import build

# Build knowledge graph
kg = build(sources=[...])

# Create SPARQL reasoner with KG
reasoner = SPARQLReasoner(triplet_store=kg, enable_inference=True)

# Add inference rules
reasoner.add_inference_rule("IF ?x :type :Company THEN ?x :type :Organization")
reasoner.add_inference_rule("IF ?x :subClassOf ?y AND ?z :type ?x THEN ?z :type ?y")

# Query with reasoning
query = """
SELECT ?person ?company
WHERE {
    ?person :worksFor ?company .
    ?company :type :Organization .
}
"""

result = reasoner.execute_query(query)
# Results include both explicit and inferred relationships
```

## Best Practices

1. **Rule Design**:
   - Keep rules focused and single-purpose
   - Use clear, descriptive rule names
   - Set appropriate confidence scores
   - Use priority to control rule execution order

2. **Inference Strategy Selection**:
   - Use forward chaining for data-driven inference
   - Use backward chaining for goal-driven queries
   - Use bidirectional for complex reasoning tasks

3. **Rete Algorithm**:
   - Build network once and reuse for multiple fact sets
   - Use incremental fact processing for efficiency
   - Optimize network by sharing common conditions

4. **SPARQL Reasoning**:
   - Enable inference for query expansion
   - Use query optimization for performance
   - Cache frequently used queries

5. **Abductive Reasoning**:
   - Set appropriate max_hypotheses limit
   - Choose ranking strategy based on use case
   - Consider multiple ranking criteria

6. **Deductive Reasoning**:
   - Structure premises clearly
   - Use appropriate logical inference rules
   - Validate proofs before accepting conclusions

7. **Explanation Generation**:
   - Generate explanations for important conclusions
   - Use appropriate detail level for audience
   - Validate explanation correctness

8. **Performance**:
   - Limit max_iterations for forward chaining
   - Use Rete algorithm for large rule sets
   - Cache query results when possible
   - Optimize rule order by selectivity

9. **Error Handling**:
   - Validate rules before adding
   - Check fact consistency
   - Handle inference failures gracefully

10. **Testing**:
    - Test rules with known fact sets
    - Verify inference results
    - Validate explanations
    - Test edge cases and boundary conditions

