"""
Inference engine for Semantica framework.

This module provides rule-based inference capabilities
for knowledge graph reasoning and analysis.
"""

from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from .rule_manager import RuleManager, Rule


class InferenceStrategy(Enum):
    """Inference strategies."""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class InferenceResult:
    """Inference result."""
    conclusion: Any
    premises: List[Any] = field(default_factory=list)
    rule_used: Optional[Rule] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InferenceEngine:
    """
    Rule-based inference engine.
    
    • Rule-based inference and reasoning
    • Forward and backward chaining
    • Rule management and execution
    • Performance optimization
    • Error handling and recovery
    • Custom rule support
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize inference engine.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - strategy: Inference strategy (forward, backward, bidirectional)
                - max_iterations: Maximum inference iterations
        """
        self.logger = get_logger("inference_engine")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.rule_manager = RuleManager(**self.config)
        self.strategy = InferenceStrategy(self.config.get("strategy", "forward"))
        self.max_iterations = self.config.get("max_iterations", 100)
        
        self.facts: Set[Any] = set()
        self.inferred_facts: List[InferenceResult] = []
    
    def add_rule(
        self,
        rule_definition: str,
        **options
    ) -> Rule:
        """
        Add inference rule to engine.
        
        Args:
            rule_definition: Rule definition string or Rule object
            **options: Additional options
        
        Returns:
            Created rule
        """
        if isinstance(rule_definition, str):
            rule = self.rule_manager.define_rule(rule_definition, **options)
        else:
            rule = rule_definition
        
        self.rule_manager.add_rule(rule)
        self.logger.debug(f"Added rule: {rule.name}")
        
        return rule
    
    def add_fact(self, fact: Any) -> None:
        """
        Add fact to knowledge base.
        
        Args:
            fact: Fact to add
        """
        self.facts.add(fact)
        self.logger.debug(f"Added fact: {fact}")
    
    def add_facts(self, facts: List[Any]) -> None:
        """
        Add multiple facts.
        
        Args:
            facts: List of facts
        """
        for fact in facts:
            self.add_fact(fact)
    
    def forward_chain(
        self,
        facts: Optional[List[Any]] = None,
        **options
    ) -> List[InferenceResult]:
        """
        Perform forward chaining inference.
        
        Args:
            facts: Optional initial facts
            **options: Additional options
        
        Returns:
            List of inference results
        """
        if facts:
            self.add_facts(facts)
        
        new_facts = True
        iterations = 0
        results = []
        
        while new_facts and iterations < self.max_iterations:
            new_facts = False
            iterations += 1
            
            # Get all rules
            rules = self.rule_manager.get_all_rules()
            
            for rule in rules:
                # Check if rule can fire
                if self._can_rule_fire(rule):
                    # Apply rule
                    result = self._apply_rule(rule)
                    if result:
                        results.append(result)
                        self.inferred_facts.append(result)
                        self.add_fact(result.conclusion)
                        new_facts = True
        
        self.logger.info(f"Forward chaining completed: {len(results)} inferences in {iterations} iterations")
        
        return results
    
    def backward_chain(
        self,
        goal: Any,
        **options
    ) -> Optional[InferenceResult]:
        """
        Perform backward chaining inference.
        
        Args:
            goal: Goal to prove
            **options: Additional options
        
        Returns:
            Inference result or None
        """
        # Check if goal is already a fact
        if goal in self.facts:
            return InferenceResult(conclusion=goal, confidence=1.0)
        
        # Find rules that can prove the goal
        rules = self.rule_manager.get_all_rules()
        applicable_rules = [r for r in rules if self._rule_concludes(r, goal)]
        
        for rule in applicable_rules:
            # Try to prove premises
            premises = []
            all_premises_proven = True
            
            for premise in rule.conditions:
                premise_result = self.backward_chain(premise, **options)
                if premise_result:
                    premises.append(premise_result.conclusion)
                else:
                    all_premises_proven = False
                    break
            
            if all_premises_proven:
                # All premises proven, rule can fire
                result = self._apply_rule(rule, premises)
                if result:
                    return result
        
        return None
    
    def _can_rule_fire(self, rule: Rule) -> bool:
        """Check if rule can fire (all conditions met)."""
        for condition in rule.conditions:
            if condition not in self.facts:
                return False
        return True
    
    def _rule_concludes(self, rule: Rule, goal: Any) -> bool:
        """Check if rule concludes the goal."""
        return rule.conclusion == goal
    
    def _apply_rule(
        self,
        rule: Rule,
        premises: Optional[List[Any]] = None
    ) -> Optional[InferenceResult]:
        """Apply rule and return inference result."""
        if premises is None:
            premises = list(rule.conditions)
        
        result = InferenceResult(
            conclusion=rule.conclusion,
            premises=premises,
            rule_used=rule,
            confidence=rule.confidence,
            metadata={
                "rule_name": rule.name,
                "rule_id": rule.rule_id
            }
        )
        
        return result
    
    def infer(
        self,
        query: Any,
        **options
    ) -> List[InferenceResult]:
        """
        Perform inference based on strategy.
        
        Args:
            query: Query or goal
            **options: Additional options
        
        Returns:
            List of inference results
        """
        if self.strategy == InferenceStrategy.FORWARD:
            return self.forward_chain(**options)
        elif self.strategy == InferenceStrategy.BACKWARD:
            result = self.backward_chain(query, **options)
            return [result] if result else []
        else:  # BIDIRECTIONAL
            forward_results = self.forward_chain(**options)
            backward_result = self.backward_chain(query, **options)
            if backward_result:
                forward_results.append(backward_result)
            return forward_results
    
    def get_facts(self) -> Set[Any]:
        """Get all facts."""
        return set(self.facts)
    
    def get_inferred_facts(self) -> List[InferenceResult]:
        """Get all inferred facts."""
        return list(self.inferred_facts)
    
    def clear_facts(self) -> None:
        """Clear all facts."""
        self.facts.clear()
        self.inferred_facts.clear()
    
    def reset(self) -> None:
        """Reset inference engine."""
        self.clear_facts()
        self.rule_manager.clear_rules()
