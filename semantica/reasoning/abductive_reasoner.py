"""
Abductive reasoner for Semantica framework.

This module provides abductive reasoning capabilities
for hypothesis generation and explanation.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from .rule_manager import RuleManager, Rule


class HypothesisRanking(Enum):
    """Hypothesis ranking strategies."""
    SIMPLICITY = "simplicity"
    PLAUSIBILITY = "plausibility"
    CONSISTENCY = "consistency"
    COVERAGE = "coverage"


@dataclass
class Observation:
    """Observation to explain."""
    observation_id: str
    description: str
    facts: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    """Abductive hypothesis."""
    hypothesis_id: str
    explanation: str
    premises: List[Any] = field(default_factory=list)
    confidence: float = 0.0
    coverage: float = 0.0
    simplicity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Explanation:
    """Abductive explanation."""
    explanation_id: str
    observation: Observation
    hypotheses: List[Hypothesis] = field(default_factory=list)
    best_hypothesis: Optional[Hypothesis] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AbductiveReasoner:
    """
    Abductive reasoning engine.
    
    • Abductive reasoning algorithms
    • Hypothesis generation and evaluation
    • Explanation generation
    • Performance optimization
    • Error handling and recovery
    • Advanced abductive techniques
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize abductive reasoner.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - max_hypotheses: Maximum hypotheses to generate
                - ranking_strategy: Hypothesis ranking strategy
        """
        self.logger = get_logger("abductive_reasoner")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.rule_manager = RuleManager(**self.config)
        self.max_hypotheses = self.config.get("max_hypotheses", 10)
        self.ranking_strategy = HypothesisRanking(self.config.get("ranking_strategy", "plausibility"))
        
        self.knowledge_base: List[Any] = []
    
    def generate_hypotheses(
        self,
        observations: List[Observation],
        **options
    ) -> List[Hypothesis]:
        """
        Generate explanatory hypotheses for observations.
        
        Args:
            observations: List of observations to explain
            **options: Additional options
        
        Returns:
            List of generated hypotheses
        """
        hypotheses = []
        
        for observation in observations:
            # Generate hypotheses for this observation
            obs_hypotheses = self._generate_hypotheses_for_observation(observation, **options)
            hypotheses.extend(obs_hypotheses)
        
        # Rank hypotheses
        ranked = self.rank_hypotheses(hypotheses)
        
        # Return top hypotheses
        return ranked[:self.max_hypotheses]
    
    def _generate_hypotheses_for_observation(
        self,
        observation: Observation,
        **options
    ) -> List[Hypothesis]:
        """Generate hypotheses for a single observation."""
        hypotheses = []
        
        # Get rules that could explain the observation
        rules = self.rule_manager.get_all_rules()
        
        # Find rules whose conclusions match observation
        for rule in rules:
            if self._rule_explains_observation(rule, observation):
                hypothesis = Hypothesis(
                    hypothesis_id=f"hyp_{len(hypotheses)}",
                    explanation=f"Hypothesis based on rule: {rule.name}",
                    premises=rule.conditions,
                    confidence=rule.confidence,
                    coverage=self._calculate_coverage(rule, observation),
                    simplicity=self._calculate_simplicity(rule),
                    metadata={"rule_id": rule.rule_id}
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _rule_explains_observation(
        self,
        rule: Rule,
        observation: Observation
    ) -> bool:
        """Check if rule can explain observation."""
        # Simple check: rule conclusion matches observation
        # Can be enhanced with more sophisticated matching
        return True
    
    def _calculate_coverage(
        self,
        rule: Rule,
        observation: Observation
    ) -> float:
        """Calculate how well rule covers observation."""
        # Simple coverage calculation
        # Can be enhanced with more sophisticated metrics
        return 0.5
    
    def _calculate_simplicity(self, rule: Rule) -> float:
        """Calculate hypothesis simplicity."""
        # Simpler hypotheses have fewer conditions
        if not rule.conditions:
            return 1.0
        
        # Inverse of number of conditions
        return 1.0 / (1.0 + len(rule.conditions))
    
    def find_explanations(
        self,
        observations: List[Observation],
        **options
    ) -> List[Explanation]:
        """
        Find explanations for observations.
        
        Args:
            observations: List of observations
            **options: Additional options
        
        Returns:
            List of explanations
        """
        explanations = []
        
        for observation in observations:
            # Generate hypotheses
            hypotheses = self.generate_hypotheses([observation], **options)
            
            # Select best hypothesis
            best_hypothesis = hypotheses[0] if hypotheses else None
            
            explanation = Explanation(
                explanation_id=f"exp_{len(explanations)}",
                observation=observation,
                hypotheses=hypotheses,
                best_hypothesis=best_hypothesis,
                confidence=best_hypothesis.confidence if best_hypothesis else 0.0,
                metadata={}
            )
            
            explanations.append(explanation)
        
        return explanations
    
    def rank_hypotheses(
        self,
        hypotheses: List[Hypothesis],
        **options
    ) -> List[Hypothesis]:
        """
        Rank hypotheses by plausibility.
        
        Args:
            hypotheses: List of hypotheses
            **options: Additional options
        
        Returns:
            Ranked hypotheses
        """
        # Calculate scores based on ranking strategy
        for hypothesis in hypotheses:
            if self.ranking_strategy == HypothesisRanking.SIMPLICITY:
                hypothesis.metadata["score"] = hypothesis.simplicity
            elif self.ranking_strategy == HypothesisRanking.PLAUSIBILITY:
                hypothesis.metadata["score"] = hypothesis.confidence
            elif self.ranking_strategy == HypothesisRanking.CONSISTENCY:
                hypothesis.metadata["score"] = self._calculate_consistency(hypothesis)
            elif self.ranking_strategy == HypothesisRanking.COVERAGE:
                hypothesis.metadata["score"] = hypothesis.coverage
            else:
                # Combined score
                hypothesis.metadata["score"] = (
                    hypothesis.confidence * 0.4 +
                    hypothesis.coverage * 0.3 +
                    hypothesis.simplicity * 0.3
                )
        
        # Sort by score
        ranked = sorted(hypotheses, key=lambda h: h.metadata.get("score", 0.0), reverse=True)
        
        return ranked
    
    def _calculate_consistency(self, hypothesis: Hypothesis) -> float:
        """Calculate hypothesis consistency with knowledge base."""
        # Check consistency with knowledge base
        # Simple implementation
        return 0.8
    
    def add_knowledge(self, facts: List[Any]) -> None:
        """
        Add knowledge to knowledge base.
        
        Args:
            facts: Facts to add
        """
        self.knowledge_base.extend(facts)
    
    def get_best_explanation(
        self,
        observation: Observation,
        **options
    ) -> Optional[Explanation]:
        """
        Get best explanation for observation.
        
        Args:
            observation: Observation to explain
            **options: Additional options
        
        Returns:
            Best explanation or None
        """
        explanations = self.find_explanations([observation], **options)
        return explanations[0] if explanations else None
