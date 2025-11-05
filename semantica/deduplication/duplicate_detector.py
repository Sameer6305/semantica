"""
Duplicate Detector for Semantica framework.

Detects duplicate entities and relationships in knowledge graphs
using similarity thresholds and clustering algorithms.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .similarity_calculator import SimilarityCalculator, SimilarityResult


@dataclass
class DuplicateCandidate:
    """Duplicate candidate representation."""
    
    entity1: Dict[str, Any]
    entity2: Dict[str, Any]
    similarity_score: float
    confidence: float
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateGroup:
    """Group of duplicate entities."""
    
    entities: List[Dict[str, Any]]
    similarity_scores: Dict[Tuple[str, str], float] = field(default_factory=dict)
    representative: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DuplicateDetector:
    """
    Duplicate detection engine.
    
    • Detects duplicate entities using similarity
    • Identifies duplicate relationships
    • Uses cluster-based duplicate identification
    • Supports batch duplicate detection
    • Provides incremental duplicate detection
    • Scores confidence for duplicate candidates
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize duplicate detector."""
        self.logger = get_logger("duplicate_detector")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.similarity_calculator = SimilarityCalculator(**config.get("similarity", {}))
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.use_clustering = config.get("use_clustering", True)
    
    def detect_duplicates(
        self,
        entities: List[Dict[str, Any]],
        **options
    ) -> List[DuplicateCandidate]:
        """
        Detect duplicate entities.
        
        Args:
            entities: List of entities to check
            **options: Detection options
            
        Returns:
            List of duplicate candidates
        """
        threshold = options.get("threshold", self.similarity_threshold)
        candidates = []
        
        # Calculate similarity for all pairs
        similarities = self.similarity_calculator.batch_calculate_similarity(
            entities,
            threshold=threshold
        )
        
        for entity1, entity2, score in similarities:
            candidate = self._create_duplicate_candidate(entity1, entity2, score)
            
            if candidate.confidence >= self.confidence_threshold:
                candidates.append(candidate)
        
        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)
        
        return candidates
    
    def detect_duplicate_groups(
        self,
        entities: List[Dict[str, Any]],
        **options
    ) -> List[DuplicateGroup]:
        """
        Detect groups of duplicate entities.
        
        Args:
            entities: List of entities
            **options: Detection options
            
        Returns:
            List of duplicate groups
        """
        candidates = self.detect_duplicates(entities, **options)
        
        # Build groups using union-find approach
        groups = self._build_duplicate_groups(candidates)
        
        # Calculate group metrics
        for group in groups:
            group.confidence = self._calculate_group_confidence(group)
            group.representative = self._select_representative(group)
        
        return groups
    
    def detect_relationship_duplicates(
        self,
        relationships: List[Dict[str, Any]],
        **options
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Detect duplicate relationships.
        
        Args:
            relationships: List of relationships
            **options: Detection options
            
        Returns:
            List of duplicate relationship pairs
        """
        duplicates = []
        threshold = options.get("threshold", 0.9)
        
        for i in range(len(relationships)):
            for j in range(i + 1, len(relationships)):
                rel1 = relationships[i]
                rel2 = relationships[j]
                
                if self._relationships_are_duplicates(rel1, rel2, threshold):
                    duplicates.append((rel1, rel2))
        
        return duplicates
    
    def incremental_detect(
        self,
        new_entities: List[Dict[str, Any]],
        existing_entities: List[Dict[str, Any]],
        **options
    ) -> List[DuplicateCandidate]:
        """
        Incremental duplicate detection for new entities.
        
        Args:
            new_entities: New entities to check
            existing_entities: Existing entities to compare against
            **options: Detection options
            
        Returns:
            List of duplicate candidates
        """
        candidates = []
        threshold = options.get("threshold", self.similarity_threshold)
        
        for new_entity in new_entities:
            for existing_entity in existing_entities:
                similarity = self.similarity_calculator.calculate_similarity(
                    new_entity,
                    existing_entity
                )
                
                if similarity.score >= threshold:
                    candidate = self._create_duplicate_candidate(
                        new_entity,
                        existing_entity,
                        similarity.score
                    )
                    
                    if candidate.confidence >= self.confidence_threshold:
                        candidates.append(candidate)
        
        return candidates
    
    def _create_duplicate_candidate(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        similarity_score: float
    ) -> DuplicateCandidate:
        """Create duplicate candidate from similarity result."""
        reasons = []
        confidence = similarity_score
        
        # Check name similarity
        name1 = entity1.get("name", "").lower()
        name2 = entity2.get("name", "").lower()
        if name1 == name2:
            reasons.append("exact_name_match")
            confidence += 0.1
        
        # Check property matches
        props1 = entity1.get("properties", {})
        props2 = entity2.get("properties", {})
        
        common_props = set(props1.keys()) & set(props2.keys())
        if common_props:
            prop_matches = sum(
                1 for prop in common_props
                if props1.get(prop) == props2.get(prop)
            )
            if prop_matches > 0:
                reasons.append(f"{prop_matches}_property_matches")
                confidence += 0.05 * prop_matches
        
        # Check type match
        if entity1.get("type") == entity2.get("type"):
            reasons.append("same_type")
            confidence += 0.05
        
        confidence = min(1.0, confidence)
        
        return DuplicateCandidate(
            entity1=entity1,
            entity2=entity2,
            similarity_score=similarity_score,
            confidence=confidence,
            reasons=reasons
        )
    
    def _build_duplicate_groups(
        self,
        candidates: List[DuplicateCandidate]
    ) -> List[DuplicateGroup]:
        """Build duplicate groups from candidates."""
        # Union-find structure
        entity_to_group = {}
        groups = []
        
        for candidate in candidates:
            entity1_id = candidate.entity1.get("id") or id(candidate.entity1)
            entity2_id = candidate.entity2.get("id") or id(candidate.entity2)
            
            group1 = entity_to_group.get(entity1_id)
            group2 = entity_to_group.get(entity2_id)
            
            if group1 is None and group2 is None:
                # Create new group
                group = DuplicateGroup(
                    entities=[candidate.entity1, candidate.entity2],
                    similarity_scores={(entity1_id, entity2_id): candidate.similarity_score}
                )
                groups.append(group)
                entity_to_group[entity1_id] = group
                entity_to_group[entity2_id] = group
            elif group1 is not None and group2 is None:
                # Add entity2 to group1
                if candidate.entity2 not in group1.entities:
                    group1.entities.append(candidate.entity2)
                group1.similarity_scores[(entity1_id, entity2_id)] = candidate.similarity_score
                entity_to_group[entity2_id] = group1
            elif group1 is None and group2 is not None:
                # Add entity1 to group2
                if candidate.entity1 not in group2.entities:
                    group2.entities.append(candidate.entity1)
                group2.similarity_scores[(entity1_id, entity2_id)] = candidate.similarity_score
                entity_to_group[entity1_id] = group2
            elif group1 != group2:
                # Merge groups
                group1.entities.extend([e for e in group2.entities if e not in group1.entities])
                group1.similarity_scores.update(group2.similarity_scores)
                group1.similarity_scores[(entity1_id, entity2_id)] = candidate.similarity_score
                
                # Update references
                for entity in group2.entities:
                    entity_id = entity.get("id") or id(entity)
                    entity_to_group[entity_id] = group1
                
                groups.remove(group2)
        
        return groups
    
    def _calculate_group_confidence(self, group: DuplicateGroup) -> float:
        """Calculate confidence for duplicate group."""
        if not group.similarity_scores:
            return 0.0
        
        scores = list(group.similarity_scores.values())
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Boost confidence for larger groups
        size_factor = min(1.0, len(group.entities) / 5.0)
        
        return avg_score * (0.8 + 0.2 * size_factor)
    
    def _select_representative(self, group: DuplicateGroup) -> Dict[str, Any]:
        """Select representative entity from group."""
        if not group.entities:
            return None
        
        # Select entity with most properties/relationships
        best_entity = max(
            group.entities,
            key=lambda e: len(e.get("properties", {})) + len(e.get("relationships", []))
        )
        
        return best_entity
    
    def _relationships_are_duplicates(
        self,
        rel1: Dict[str, Any],
        rel2: Dict[str, Any],
        threshold: float
    ) -> bool:
        """Check if two relationships are duplicates."""
        # Exact match
        if (rel1.get("subject") == rel2.get("subject") and
            rel1.get("predicate") == rel2.get("predicate") and
            rel1.get("object") == rel2.get("object")):
            return True
        
        # Similarity check
        similarity = self.similarity_calculator.calculate_string_similarity(
            str(rel1.get("predicate", "")),
            str(rel2.get("predicate", ""))
        )
        
        return similarity >= threshold
