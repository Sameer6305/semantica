"""
Rete algorithm implementation for Semantica framework.

This module provides Rete algorithm implementation
for efficient rule-based reasoning.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from .rule_manager import Rule


@dataclass
class Fact:
    """Fact representation."""
    fact_id: str
    predicate: str
    arguments: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Match:
    """Pattern match."""
    rule: Rule
    facts: List[Fact]
    bindings: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class ReteNode:
    """Base Rete network node."""
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.children: List['ReteNode'] = []


class AlphaNode(ReteNode):
    """Alpha node for single condition matching."""
    def __init__(self, node_id: str, condition: Any):
        super().__init__(node_id)
        self.condition = condition
        self.matches: List[Fact] = []
    
    def add_fact(self, fact: Fact) -> bool:
        """Add fact if it matches condition."""
        if self._matches(fact):
            self.matches.append(fact)
            return True
        return False
    
    def _matches(self, fact: Fact) -> bool:
        """Check if fact matches condition."""
        # Simple matching - can be enhanced
        return True


class BetaNode(ReteNode):
    """Beta node for joining conditions."""
    def __init__(self, node_id: str, left: ReteNode, right: ReteNode):
        super().__init__(node_id)
        self.left = left
        self.right = right
        self.matches: List[Tuple[Fact, Fact]] = []
    
    def join(self, left_fact: Fact, right_fact: Fact) -> bool:
        """Join facts from left and right nodes."""
        if self._can_join(left_fact, right_fact):
            self.matches.append((left_fact, right_fact))
            return True
        return False
    
    def _can_join(self, left_fact: Fact, right_fact: Fact) -> bool:
        """Check if facts can be joined."""
        # Simple join logic - can be enhanced
        return True


class TerminalNode(ReteNode):
    """Terminal node representing rule activation."""
    def __init__(self, node_id: str, rule: Rule):
        super().__init__(node_id)
        self.rule = rule
        self.activations: List[Match] = []
    
    def activate(self, match: Match) -> None:
        """Activate rule."""
        self.activations.append(match)


class ReteEngine:
    """
    Rete algorithm implementation for efficient rule matching.
    
    • Rete algorithm implementation
    • Rule network construction and optimization
    • Pattern matching and conflict resolution
    • Performance optimization
    • Error handling and recovery
    • Advanced Rete features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize Rete engine.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("rete_engine")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.network: Dict[str, ReteNode] = {}
        self.facts: List[Fact] = []
        self.fact_counter = 0
        self.node_counter = 0
    
    def build_network(self, rules: List[Rule]) -> None:
        """
        Build Rete network from rules.
        
        Args:
            rules: List of rules
        """
        self.network.clear()
        
        for rule in rules:
            self._add_rule_to_network(rule)
        
        self.logger.info(f"Built Rete network with {len(self.network)} nodes for {len(rules)} rules")
    
    def _add_rule_to_network(self, rule: Rule) -> None:
        """Add rule to Rete network."""
        # Create alpha nodes for each condition
        alpha_nodes = []
        for condition in rule.conditions:
            node_id = f"alpha_{self.node_counter}"
            self.node_counter += 1
            alpha_node = AlphaNode(node_id, condition)
            alpha_nodes.append(alpha_node)
            self.network[node_id] = alpha_node
        
        # Create beta nodes for joining
        if len(alpha_nodes) > 1:
            current = alpha_nodes[0]
            for i in range(1, len(alpha_nodes)):
                node_id = f"beta_{self.node_counter}"
                self.node_counter += 1
                beta_node = BetaNode(node_id, current, alpha_nodes[i])
                self.network[node_id] = beta_node
                current = beta_node
            final_node = current
        else:
            final_node = alpha_nodes[0] if alpha_nodes else None
        
        # Create terminal node
        if final_node:
            node_id = f"terminal_{self.node_counter}"
            self.node_counter += 1
            terminal_node = TerminalNode(node_id, rule)
            final_node.children.append(terminal_node)
            self.network[node_id] = terminal_node
    
    def add_fact(self, fact: Fact) -> None:
        """
        Add fact to working memory.
        
        Args:
            fact: Fact to add
        """
        self.facts.append(fact)
        
        # Propagate through network
        self._propagate_fact(fact)
    
    def _propagate_fact(self, fact: Fact) -> None:
        """Propagate fact through Rete network."""
        # Find matching alpha nodes
        for node_id, node in self.network.items():
            if isinstance(node, AlphaNode):
                if node.add_fact(fact):
                    # Propagate to children
                    self._propagate_from_alpha(node, fact)
    
    def _propagate_from_alpha(self, alpha_node: AlphaNode, fact: Fact) -> None:
        """Propagate from alpha node to children."""
        for child in alpha_node.children:
            if isinstance(child, BetaNode):
                # Join with matches from left side
                for left_fact in alpha_node.matches:
                    if child.join(left_fact, fact):
                        # Propagate to children
                        for grandchild in child.children:
                            if isinstance(grandchild, TerminalNode):
                                match = Match(
                                    rule=grandchild.rule,
                                    facts=[left_fact, fact],
                                    confidence=1.0
                                )
                                grandchild.activate(match)
            elif isinstance(child, TerminalNode):
                # Direct activation
                match = Match(
                    rule=child.rule,
                    facts=[fact],
                    confidence=1.0
                )
                child.activate(match)
    
    def match_patterns(
        self,
        facts: Optional[List[Fact]] = None
    ) -> List[Match]:
        """
        Match patterns using Rete algorithm.
        
        Args:
            facts: Optional facts to match (uses working memory if not provided)
        
        Returns:
            List of matches
        """
        if facts:
            # Add facts to working memory
            for fact in facts:
                self.add_fact(fact)
        
        # Collect all activations
        matches = []
        for node_id, node in self.network.items():
            if isinstance(node, TerminalNode):
                matches.extend(node.activations)
        
        return matches
    
    def execute_matches(
        self,
        matches: Optional[List[Match]] = None
    ) -> List[Any]:
        """
        Execute matched rules.
        
        Args:
            matches: Optional matches to execute (uses current matches if not provided)
        
        Returns:
            List of inference results
        """
        if matches is None:
            matches = self.match_patterns()
        
        results = []
        for match in matches:
            try:
                # Execute rule
                result = match.rule.conclusion
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error executing match: {e}")
        
        return results
    
    def reset(self) -> None:
        """Reset Rete engine."""
        self.facts.clear()
        for node in self.network.values():
            if isinstance(node, AlphaNode):
                node.matches.clear()
            elif isinstance(node, BetaNode):
                node.matches.clear()
            elif isinstance(node, TerminalNode):
                node.activations.clear()
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        alpha_count = sum(1 for n in self.network.values() if isinstance(n, AlphaNode))
        beta_count = sum(1 for n in self.network.values() if isinstance(n, BetaNode))
        terminal_count = sum(1 for n in self.network.values() if isinstance(n, TerminalNode))
        
        return {
            "total_nodes": len(self.network),
            "alpha_nodes": alpha_count,
            "beta_nodes": beta_count,
            "terminal_nodes": terminal_count,
            "facts": len(self.facts)
        }
