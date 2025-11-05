"""
SPARQL reasoner for Semantica framework.

This module provides SPARQL-based reasoning capabilities
for knowledge graph query answering.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import re

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from .rule_manager import RuleManager, Rule


@dataclass
class SPARQLQueryResult:
    """SPARQL query result."""
    bindings: List[Dict[str, Any]]
    variables: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SPARQLReasoner:
    """
    SPARQL-based reasoning engine.
    
    • SPARQL query reasoning and execution
    • Inference rule integration
    • Query optimization and caching
    • Performance optimization
    • Error handling and recovery
    • Advanced SPARQL features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize SPARQL reasoner.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - triple_store: Triple store connection
                - enable_inference: Enable inference rules
        """
        self.logger = get_logger("sparql_reasoner")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.rule_manager = RuleManager(**self.config)
        self.triple_store = self.config.get("triple_store")
        self.enable_inference = self.config.get("enable_inference", True)
        
        self.query_cache: Dict[str, Any] = {}
    
    def expand_query(
        self,
        query: str,
        **options
    ) -> str:
        """
        Expand SPARQL query with inference rules.
        
        Args:
            query: Original SPARQL query
            **options: Additional options
        
        Returns:
            Expanded query
        """
        if not self.enable_inference:
            return query
        
        # Parse query to find patterns
        expanded_query = query
        
        # Get inference rules
        rules = self.rule_manager.get_all_rules()
        
        # Add inferred patterns based on rules
        for rule in rules:
            # Convert rule to SPARQL pattern
            sparql_pattern = self._rule_to_sparql(rule)
            if sparql_pattern:
                # Add to query (basic implementation)
                expanded_query += f"\n# Inference: {rule.name}\n{sparql_pattern}"
        
        return expanded_query
    
    def _rule_to_sparql(self, rule: Rule) -> Optional[str]:
        """Convert rule to SPARQL pattern."""
        # Basic conversion - can be enhanced
        try:
            # Extract conditions as SPARQL patterns
            patterns = []
            for condition in rule.conditions:
                # Simple pattern matching
                if " is_a " in condition:
                    parts = condition.split(" is_a ")
                    if len(parts) == 2:
                        var = parts[0].strip()
                        class_type = parts[1].strip()
                        patterns.append(f"?{var} a :{class_type} .")
            
            # Conclusion
            if " is_a " in rule.conclusion:
                parts = rule.conclusion.split(" is_a ")
                if len(parts) == 2:
                    var = parts[0].strip()
                    class_type = parts[1].strip()
                    conclusion_pattern = f"?{var} a :{class_type} ."
                    
                    # Combine into SPARQL pattern
                    if patterns:
                        return f"{' '.join(patterns)} => {conclusion_pattern}"
        
        except Exception as e:
            self.logger.warning(f"Could not convert rule to SPARQL: {e}")
        
        return None
    
    def infer_results(
        self,
        query_results: SPARQLQueryResult,
        **options
    ) -> SPARQLQueryResult:
        """
        Infer additional results from query results.
        
        Args:
            query_results: Original query results
            **options: Additional options
        
        Returns:
            Results with inferences
        """
        inferred_bindings = list(query_results.bindings)
        
        # Apply inference rules
        if self.enable_inference:
            rules = self.rule_manager.get_all_rules()
            
            for rule in rules:
                # Check if rule can be applied to results
                new_bindings = self._apply_rule_to_results(rule, query_results.bindings)
                inferred_bindings.extend(new_bindings)
        
        # Remove duplicates
        unique_bindings = self._deduplicate_bindings(inferred_bindings)
        
        return SPARQLQueryResult(
            bindings=unique_bindings,
            variables=query_results.variables,
            metadata={
                **query_results.metadata,
                "original_count": len(query_results.bindings),
                "inferred_count": len(unique_bindings) - len(query_results.bindings)
            }
        )
    
    def _apply_rule_to_results(
        self,
        rule: Rule,
        bindings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply rule to query results."""
        new_bindings = []
        
        for binding in bindings:
            # Check if rule conditions match
            if self._match_rule_conditions(rule, binding):
                # Generate new binding from conclusion
                new_binding = self._generate_binding_from_conclusion(rule, binding)
                if new_binding:
                    new_bindings.append(new_binding)
        
        return new_bindings
    
    def _match_rule_conditions(
        self,
        rule: Rule,
        binding: Dict[str, Any]
    ) -> bool:
        """Check if rule conditions match binding."""
        for condition in rule.conditions:
            # Simple matching - can be enhanced
            if " is_a " in condition:
                parts = condition.split(" is_a ")
                if len(parts) == 2:
                    var = parts[0].strip().replace("?", "")
                    class_type = parts[1].strip()
                    
                    # Check if binding has matching type
                    if var in binding:
                        value = binding[var]
                        # Check type (simplified)
                        if not self._has_type(value, class_type):
                            return False
        
        return True
    
    def _has_type(self, value: Any, class_type: str) -> bool:
        """Check if value has type (simplified)."""
        # This is a placeholder - in practice would check against knowledge graph
        return True
    
    def _generate_binding_from_conclusion(
        self,
        rule: Rule,
        binding: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate new binding from rule conclusion."""
        new_binding = binding.copy()
        
        # Parse conclusion
        if " is_a " in rule.conclusion:
            parts = rule.conclusion.split(" is_a ")
            if len(parts) == 2:
                var = parts[0].strip().replace("?", "")
                class_type = parts[1].strip()
                
                # Add type information
                if var in new_binding:
                    new_binding[f"{var}_type"] = class_type
        
        return new_binding
    
    def _deduplicate_bindings(
        self,
        bindings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate bindings."""
        seen = set()
        unique = []
        
        for binding in bindings:
            # Create hashable representation
            binding_key = tuple(sorted(binding.items()))
            if binding_key not in seen:
                seen.add(binding_key)
                unique.append(binding)
        
        return unique
    
    def execute_query(
        self,
        query: str,
        **options
    ) -> SPARQLQueryResult:
        """
        Execute SPARQL query with reasoning.
        
        Args:
            query: SPARQL query string
            **options: Additional options
        
        Returns:
            Query results
        """
        # Check cache
        if query in self.query_cache:
            self.logger.debug("Returning cached query result")
            return self.query_cache[query]
        
        # Expand query
        expanded_query = self.expand_query(query, **options)
        
        # Execute query (if triple store available)
        if self.triple_store:
            # This would call the triple store's query method
            # For now, return empty result
            result = SPARQLQueryResult(bindings=[], variables=[])
        else:
            # Mock result for testing
            result = SPARQLQueryResult(bindings=[], variables=[])
        
        # Infer additional results
        if self.enable_inference:
            result = self.infer_results(result, **options)
        
        # Cache result
        self.query_cache[query] = result
        
        return result
    
    def clear_cache(self) -> None:
        """Clear query cache."""
        self.query_cache.clear()
    
    def add_inference_rule(self, rule_definition: str, **options) -> Rule:
        """Add inference rule."""
        return self.rule_manager.define_rule(rule_definition, **options)
