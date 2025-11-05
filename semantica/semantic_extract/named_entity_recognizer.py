"""
Named Entity Recognition Module

Handles extraction and recognition of named entities from text.

Key Features:
    - Multi-type entity recognition
    - Entity classification and categorization
    - Entity confidence scoring
    - Custom entity type support
    - Batch entity processing

Main Classes:
    - NamedEntityRecognizer: Main NER class
    - EntityClassifier: Entity classification engine
    - EntityConfidenceScorer: Confidence scoring system
    - CustomEntityDetector: Custom entity detection
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .ner_extractor import Entity, NERExtractor


class NamedEntityRecognizer:
    """
    Named entity recognition handler.
    
    • Extracts named entities from text
    • Classifies entities by type and category
    • Provides confidence scores for entities
    • Supports custom entity types
    • Handles multiple languages and domains
    • Processes batch text collections
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize named entity recognizer."""
        self.logger = get_logger("named_entity_recognizer")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Use NERExtractor for actual extraction
        self.ner_extractor = NERExtractor(**self.config.get("ner", {}))
        self.entity_classifier = EntityClassifier(**self.config.get("classifier", {}))
        self.confidence_scorer = EntityConfidenceScorer(**self.config.get("scorer", {}))
    
    def extract_entities(self, text: str, **options) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            **options: Extraction options
            
        Returns:
            list: List of extracted entities
        """
        return self.ner_extractor.extract_entities(text, **options)
    
    def classify_entities(self, entities: List[Entity], **context) -> Dict[str, List[Entity]]:
        """
        Classify entities by type and category.
        
        Args:
            entities: List of entities
            **context: Context information
            
        Returns:
            dict: Entities grouped by type
        """
        return self.entity_classifier.classify_entities(entities, **context)
    
    def score_confidence(self, entities: List[Entity], **options) -> List[Entity]:
        """
        Calculate confidence scores for entities.
        
        Args:
            entities: List of entities
            **options: Scoring options
            
        Returns:
            list: Entities with confidence scores
        """
        return self.confidence_scorer.score_entities(entities, **options)
    
    def process_batch(self, texts: List[str], **options) -> List[List[Entity]]:
        """
        Process multiple texts for entity extraction.
        
        Args:
            texts: List of input texts
            **options: Processing options
            
        Returns:
            list: List of entity lists for each text
        """
        return self.ner_extractor.extract_entities_batch(texts, **options)


class EntityClassifier:
    """Entity classification engine."""
    
    def __init__(self, **config):
        """Initialize entity classifier."""
        self.logger = get_logger("entity_classifier")
        self.config = config
        
        # Entity type mappings
        self.type_hierarchy = {
            "PERSON": ["PERSON", "PER"],
            "ORG": ["ORG", "ORGANIZATION"],
            "GPE": ["GPE", "LOCATION", "LOC"],
            "DATE": ["DATE", "TIME"],
            "MONEY": ["MONEY", "CURRENCY"],
            "PERCENT": ["PERCENT", "PERCENTAGE"]
        }
    
    def classify_entity_type(self, entity: Entity, **context) -> str:
        """
        Classify entity by type.
        
        Args:
            entity: Entity to classify
            **context: Context information
            
        Returns:
            str: Entity type
        """
        # Normalize entity type
        label = entity.label.upper()
        
        # Check type hierarchy
        for canonical_type, variants in self.type_hierarchy.items():
            if label in variants:
                return canonical_type
        
        return label
    
    def disambiguate_entity(self, entity: Entity, candidates: List[Entity], **context) -> Optional[Entity]:
        """
        Disambiguate entity among candidates.
        
        Args:
            entity: Entity to disambiguate
            candidates: Candidate entities
            **context: Context information
            
        Returns:
            Entity: Best matching entity or None
        """
        if not candidates:
            return None
        
        # Simple disambiguation by type match
        entity_type = entity.label
        matching = [c for c in candidates if c.label == entity_type]
        
        if matching:
            # Return first match with highest confidence
            return max(matching, key=lambda e: e.confidence)
        
        return candidates[0] if candidates else None
    
    def classify_entities(self, entities: List[Entity], **context) -> Dict[str, List[Entity]]:
        """
        Classify entities by type.
        
        Args:
            entities: List of entities
            **context: Context information
            
        Returns:
            dict: Entities grouped by type
        """
        classified = {}
        for entity in entities:
            entity_type = self.classify_entity_type(entity, **context)
            if entity_type not in classified:
                classified[entity_type] = []
            classified[entity_type].append(entity)
        
        return classified


class EntityConfidenceScorer:
    """Confidence scoring system."""
    
    def __init__(self, **config):
        """Initialize confidence scorer."""
        self.logger = get_logger("entity_confidence_scorer")
        self.config = config
    
    def score_entities(self, entities: List[Entity], **options) -> List[Entity]:
        """
        Calculate confidence scores for entities.
        
        Args:
            entities: List of entities
            **options: Scoring options
            
        Returns:
            list: Entities with updated confidence scores
        """
        for entity in entities:
            if entity.confidence == 1.0:  # Only recalculate if needed
                entity.confidence = self._calculate_confidence(entity, **options)
        
        return entities
    
    def _calculate_confidence(self, entity: Entity, **options) -> float:
        """
        Calculate confidence score for entity.
        
        Args:
            entity: Entity to score
            **options: Scoring options
            
        Returns:
            float: Confidence score (0-1)
        """
        score = 1.0
        
        # Length-based scoring
        text_length = len(entity.text)
        if text_length < 2:
            score *= 0.7
        elif text_length > 50:
            score *= 0.9
        
        # Capitalization check (for PERSON, ORG)
        if entity.label in ["PERSON", "ORG", "GPE"]:
            if not entity.text[0].isupper():
                score *= 0.8
        
        # Type-specific scoring
        if entity.label == "DATE":
            # Check if looks like date
            if any(char.isdigit() for char in entity.text):
                score *= 1.1  # Boost for date-like patterns
        
        return min(1.0, max(0.0, score))


class CustomEntityDetector:
    """Custom entity detection."""
    
    def __init__(self, **config):
        """Initialize custom entity detector."""
        self.logger = get_logger("custom_entity_detector")
        self.config = config
        
        self.custom_patterns = config.get("patterns", {})
    
    def detect_custom_entities(self, text: str, entity_type: str) -> List[Entity]:
        """
        Detect custom entities using patterns.
        
        Args:
            text: Input text
            entity_type: Custom entity type
            
        Returns:
            list: List of detected entities
        """
        entities = []
        
        if entity_type in self.custom_patterns:
            import re
            pattern = self.custom_patterns[entity_type]
            
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(0),
                    label=entity_type,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.8,
                    metadata={"extraction_method": "custom_pattern"}
                ))
        
        return entities
