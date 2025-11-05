"""
Rule manager for Semantica framework.

This module provides rule management and execution
for reasoning and inference operations.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import re

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


class RuleType(Enum):
    """Rule types."""
    IMPLICATION = "implication"
    EQUIVALENCE = "equivalence"
    CONSTRAINT = "constraint"
    TRANSFORMATION = "transformation"


@dataclass
class Rule:
    """Rule definition."""
    rule_id: str
    name: str
    conditions: List[Any]
    conclusion: Any
    rule_type: RuleType = RuleType.IMPLICATION
    confidence: float = 1.0
    priority: int = 0
    handler: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleExecution:
    """Rule execution record."""
    rule_id: str
    executed_at: str
    input_facts: List[Any]
    output_fact: Any
    success: bool
    execution_time: float = 0.0
    error: Optional[str] = None


class RuleManager:
    """
    Rule management system.
    
    • Rule creation and management
    • Rule validation and testing
    • Rule execution and scheduling
    • Performance optimization
    • Error handling and recovery
    • Rule versioning and updates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize rule manager.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("rule_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.rules: Dict[str, Rule] = {}
        self.execution_history: List[RuleExecution] = []
        self.rule_counter = 0
    
    def define_rule(
        self,
        rule_definition: str,
        **options
    ) -> Rule:
        """
        Define new inference rule.
        
        Args:
            rule_definition: Rule definition string (e.g., "IF ?x is_a Company THEN ?x is_a Organization")
            **options: Additional options
        
        Returns:
            Created rule
        """
        # Parse rule definition
        parsed = self._parse_rule_definition(rule_definition)
        
        # Create rule
        self.rule_counter += 1
        rule = Rule(
            rule_id=f"rule_{self.rule_counter}",
            name=options.get("name", f"Rule {self.rule_counter}"),
            conditions=parsed["conditions"],
            conclusion=parsed["conclusion"],
            rule_type=RuleType(parsed.get("type", "implication")),
            confidence=options.get("confidence", 1.0),
            priority=options.get("priority", 0),
            handler=options.get("handler"),
            metadata=options.get("metadata", {})
        )
        
        # Validate rule
        validation = self.validate_rule(rule)
        if not validation["valid"]:
            raise ValidationError(f"Rule validation failed: {validation['errors']}")
        
        return rule
    
    def _parse_rule_definition(self, definition: str) -> Dict[str, Any]:
        """Parse rule definition string."""
        # Simple pattern matching
        # Format: "IF condition1 AND condition2 THEN conclusion"
        # or "IF condition THEN conclusion"
        
        definition = definition.strip()
        
        # Check for IF-THEN pattern
        if_match = re.match(r'IF\s+(.+?)\s+THEN\s+(.+)$', definition, re.IGNORECASE)
        if if_match:
            conditions_str = if_match.group(1)
            conclusion_str = if_match.group(2)
            
            # Split conditions by AND
            conditions = [c.strip() for c in re.split(r'\s+AND\s+', conditions_str, flags=re.IGNORECASE)]
            
            return {
                "conditions": conditions,
                "conclusion": conclusion_str.strip(),
                "type": "implication"
            }
        
        # Check for equivalence pattern
        equiv_match = re.match(r'(.+?)\s+IFF\s+(.+)$', definition, re.IGNORECASE)
        if equiv_match:
            left = equiv_match.group(1).strip()
            right = equiv_match.group(2).strip()
            
            return {
                "conditions": [left],
                "conclusion": right,
                "type": "equivalence"
            }
        
        # Fallback: treat as simple condition -> conclusion
        if "->" in definition:
            parts = definition.split("->", 1)
            return {
                "conditions": [parts[0].strip()],
                "conclusion": parts[1].strip(),
                "type": "implication"
            }
        
        raise ValidationError(f"Could not parse rule definition: {definition}")
    
    def add_rule(self, rule: Rule) -> None:
        """
        Add rule to manager.
        
        Args:
            rule: Rule object
        """
        self.rules[rule.rule_id] = rule
        self.logger.debug(f"Added rule: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str) -> None:
        """
        Remove rule.
        
        Args:
            rule_id: Rule identifier
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.debug(f"Removed rule: {rule_id}")
    
    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get rule by ID."""
        return self.rules.get(rule_id)
    
    def get_all_rules(self) -> List[Rule]:
        """Get all rules."""
        return list(self.rules.values())
    
    def validate_rule(self, rule: Rule) -> Dict[str, Any]:
        """
        Validate rule syntax and logic.
        
        Args:
            rule: Rule to validate
        
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check required fields
        if not rule.conditions:
            errors.append("Rule missing conditions")
        if not rule.conclusion:
            errors.append("Rule missing conclusion")
        if rule.confidence < 0 or rule.confidence > 1:
            errors.append("Rule confidence must be between 0 and 1")
        
        # Check for circular dependencies (basic)
        if rule.conclusion in rule.conditions:
            warnings.append("Rule conclusion appears in conditions (potential circularity)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def execute_rule(
        self,
        rule: Rule,
        facts: List[Any],
        **options
    ) -> Optional[Any]:
        """
        Execute rule with given facts.
        
        Args:
            rule: Rule to execute
            facts: Input facts
            **options: Additional options
        
        Returns:
            Rule execution result or None
        """
        import time
        start_time = time.time()
        
        try:
            # Check if conditions are met
            conditions_met = all(condition in facts for condition in rule.conditions)
            
            if not conditions_met:
                return None
            
            # Execute rule
            if rule.handler:
                result = rule.handler(facts, **options)
            else:
                result = rule.conclusion
            
            execution_time = time.time() - start_time
            
            # Record execution
            execution = RuleExecution(
                rule_id=rule.rule_id,
                executed_at=datetime.now().isoformat(),
                input_facts=facts,
                output_fact=result,
                success=True,
                execution_time=execution_time
            )
            self.execution_history.append(execution)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            execution = RuleExecution(
                rule_id=rule.rule_id,
                executed_at=datetime.now().isoformat(),
                input_facts=facts,
                output_fact=None,
                success=False,
                execution_time=execution_time,
                error=str(e)
            )
            self.execution_history.append(execution)
            
            self.logger.error(f"Rule execution failed: {e}")
            return None
    
    def track_execution(self, rule: Rule) -> List[RuleExecution]:
        """
        Track rule execution history.
        
        Args:
            rule: Rule to track
        
        Returns:
            Execution history
        """
        return [e for e in self.execution_history if e.rule_id == rule.rule_id]
    
    def get_execution_stats(self, rule_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Args:
            rule_id: Optional rule ID filter
        
        Returns:
            Execution statistics
        """
        if rule_id:
            executions = [e for e in self.execution_history if e.rule_id == rule_id]
        else:
            executions = self.execution_history
        
        if not executions:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "average_execution_time": 0.0
            }
        
        successful = sum(1 for e in executions if e.success)
        failed = len(executions) - successful
        avg_time = sum(e.execution_time for e in executions) / len(executions)
        
        return {
            "total_executions": len(executions),
            "successful_executions": successful,
            "failed_executions": failed,
            "average_execution_time": avg_time,
            "success_rate": successful / len(executions) if executions else 0.0
        }
    
    def clear_rules(self) -> None:
        """Clear all rules."""
        self.rules.clear()
        self.rule_counter = 0
    
    def clear_execution_history(self) -> None:
        """Clear execution history."""
        self.execution_history.clear()
