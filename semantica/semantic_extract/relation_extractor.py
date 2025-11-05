"""
Relation extraction for Semantica framework.

This module provides relationship detection and extraction
between entities in text documents.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .ner_extractor import Entity


@dataclass
class Relation:
    """Relation representation."""
    
    subject: Entity
    predicate: str
    object: Entity
    confidence: float = 1.0
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RelationExtractor:
    """Relation extractor for entity relationships."""
    
    def __init__(self, **config):
        """
        Initialize relation extractor.
        
        Args:
            **config: Configuration options:
                - min_confidence: Minimum confidence threshold
                - use_llm: Use LLM for relation extraction (default: False)
        """
        self.logger = get_logger("relation_extractor")
        self.config = config
        
        self.min_confidence = config.get("min_confidence", 0.5)
        self.use_llm = config.get("use_llm", False)
        
        # Common relation patterns
        self.relation_patterns = {
            "founded_by": [
                r"(?P<subject>\w+)\s+(?:was\s+)?founded\s+by\s+(?P<object>\w+(?:\s+\w+)*)",
                r"(?P<object>\w+(?:\s+\w+)*)\s+founded\s+(?P<subject>\w+)"
            ],
            "located_in": [
                r"(?P<subject>\w+)\s+is\s+located\s+in\s+(?P<object>\w+)",
                r"(?P<subject>\w+)\s+in\s+(?P<object>\w+)"
            ],
            "works_for": [
                r"(?P<subject>\w+)\s+works?\s+for\s+(?P<object>\w+)",
                r"(?P<subject>\w+)\s+is\s+an?\s+employee\s+of\s+(?P<object>\w+)"
            ],
            "born_in": [
                r"(?P<subject>\w+)\s+was\s+born\s+in\s+(?P<object>\w+)",
                r"(?P<subject>\w+)\s+born\s+in\s+(?P<object>\w+)"
            ]
        }
    
    def extract_relations(self, text: str, entities: List[Entity], **options) -> List[Relation]:
        """
        Extract relations between entities.
        
        Args:
            text: Input text
            entities: List of extracted entities
            **options: Extraction options
            
        Returns:
            list: List of extracted relations
        """
        if not text or not entities:
            return []
        
        min_confidence = options.get("min_confidence", self.min_confidence)
        
        # Pattern-based extraction
        relations = self._extract_with_patterns(text, entities)
        
        # Filter by confidence
        relations = [r for r in relations if r.confidence >= min_confidence]
        
        return relations
    
    def _extract_with_patterns(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using pattern matching."""
        relations = []
        
        # Create entity lookup by text
        entity_map = {e.text.lower(): e for e in entities}
        
        # Check each relation pattern
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    subject_text = match.group("subject")
                    object_text = match.group("object")
                    
                    subject_entity = entity_map.get(subject_text.lower())
                    object_entity = entity_map.get(object_text.lower())
                    
                    if subject_entity and object_entity:
                        # Get context around the match
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end]
                        
                        relations.append(Relation(
                            subject=subject_entity,
                            predicate=relation_type,
                            object=object_entity,
                            confidence=0.7,  # Pattern-based confidence
                            context=context,
                            metadata={
                                "extraction_method": "pattern",
                                "pattern": pattern
                            }
                        ))
        
        # Co-occurrence based relations (entities close to each other)
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Check if entities are close in text
                distance = abs(entity1.end_char - entity2.start_char)
                if distance < 100:  # Within 100 characters
                    # Simple relation based on proximity
                    start = min(entity1.start_char, entity2.start_char)
                    end = max(entity1.end_char, entity2.end_char)
                    context = text[max(0, start-30):min(len(text), end+30)]
                    
                    relations.append(Relation(
                        subject=entity1,
                        predicate="related_to",
                        object=entity2,
                        confidence=0.5,  # Lower confidence for co-occurrence
                        context=context,
                        metadata={
                            "extraction_method": "co_occurrence",
                            "distance": distance
                        }
                    ))
        
        return relations
    
    def classify_relations(self, relations: List[Relation]) -> Dict[str, List[Relation]]:
        """
        Classify relations by type.
        
        Args:
            relations: List of relations
            
        Returns:
            dict: Relations grouped by predicate type
        """
        classified = {}
        for relation in relations:
            if relation.predicate not in classified:
                classified[relation.predicate] = []
            classified[relation.predicate].append(relation)
        
        return classified
    
    def validate_relations(self, relations: List[Relation]) -> List[Relation]:
        """
        Validate relations for consistency.
        
        Args:
            relations: List of relations
            
        Returns:
            list: Validated relations
        """
        valid_relations = []
        
        for relation in relations:
            # Basic validation
            if not relation.subject or not relation.object:
                continue
            
            if not relation.predicate:
                continue
            
            # Check if subject and object are different
            if relation.subject.text == relation.object.text:
                continue
            
            valid_relations.append(relation)
        
        return valid_relations
