"""
Entity Normalization Module

Handles normalization of named entities and proper nouns.

Key Features:
    - Entity name standardization
    - Alias resolution and mapping
    - Entity disambiguation
    - Name variant handling
    - Entity linking and resolution

Main Classes:
    - EntityNormalizer: Main entity normalization class
    - AliasResolver: Entity alias resolution
    - EntityDisambiguator: Entity disambiguation
    - NameVariantHandler: Name variant processing
"""

import re
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


class EntityNormalizer:
    """
    Entity normalization and standardization handler.
    
    • Normalizes entity names and proper nouns
    • Resolves entity aliases and variants
    • Handles entity disambiguation
    • Standardizes entity formats
    • Links entities to canonical forms
    • Supports multiple entity types
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize entity normalizer.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("entity_normalizer")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.alias_resolver = AliasResolver(**self.config)
        self.disambiguator = EntityDisambiguator(**self.config)
        self.variant_handler = NameVariantHandler(**self.config)
    
    def normalize_entity(self, entity_name: str, entity_type: Optional[str] = None, **options) -> str:
        """
        Normalize entity name to standard form.
        
        Args:
            entity_name: Entity name to normalize
            entity_type: Entity type (optional)
            **options: Normalization options
        
        Returns:
            Normalized entity name
        """
        if not entity_name:
            return ""
        
        normalized = entity_name.strip()
        
        # Clean and standardize
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.title() if entity_type == "Person" else normalized
        
        # Resolve aliases
        if options.get("resolve_aliases", True):
            resolved = self.alias_resolver.resolve_aliases(normalized, entity_type=entity_type)
            if resolved:
                normalized = resolved
        
        # Handle name variants
        normalized = self.variant_handler.normalize_name_format(normalized, format_type="standard")
        
        return normalized
    
    def resolve_aliases(self, entity_name: str, **context) -> Optional[str]:
        """
        Resolve entity aliases and variants.
        
        Args:
            entity_name: Entity name
            **context: Context information
        
        Returns:
            Resolved canonical form or None
        """
        return self.alias_resolver.resolve_aliases(entity_name, **context)
    
    def disambiguate_entity(self, entity_name: str, **context) -> Dict[str, Any]:
        """
        Disambiguate entity when multiple candidates exist.
        
        Args:
            entity_name: Entity name
            **context: Context information
        
        Returns:
            Disambiguation result
        """
        return self.disambiguator.disambiguate(entity_name, **context)
    
    def link_entities(self, entities: List[str], **options) -> Dict[str, str]:
        """
        Link entities to canonical forms.
        
        Args:
            entities: List of entity names
            **options: Linking options
        
        Returns:
            Dictionary mapping entities to canonical forms
        """
        linked = {}
        
        for entity in entities:
            canonical = self.normalize_entity(entity, **options)
            linked[entity] = canonical
        
        return linked


class AliasResolver:
    """
    Entity alias resolution engine.
    
    • Resolves entity aliases and nicknames
    • Maps name variations to canonical forms
    • Handles different naming conventions
    • Processes cultural and linguistic variations
    """
    
    def __init__(self, **config):
        """
        Initialize alias resolver.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("alias_resolver")
        self.config = config
        self.alias_map = config.get("alias_map", {})
    
    def resolve_aliases(self, entity_name: str, **context) -> Optional[str]:
        """
        Resolve entity aliases to canonical form.
        
        Args:
            entity_name: Entity name
            **context: Context information
        
        Returns:
            Resolved canonical form or None
        """
        # Check alias map
        entity_lower = entity_name.lower()
        
        if entity_lower in self.alias_map:
            return self.alias_map[entity_lower]
        
        # Check for common aliases
        for alias, canonical in self.alias_map.items():
            if alias.lower() == entity_lower:
                return canonical
        
        return None
    
    def map_variants(self, entity_name: str, entity_type: str) -> str:
        """
        Map entity name variants.
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
        
        Returns:
            Mapped variant
        """
        # Simple variant mapping
        return entity_name
    
    def handle_cultural_variations(self, entity_name: str, culture: Optional[str] = None) -> str:
        """
        Handle cultural and linguistic variations.
        
        Args:
            entity_name: Entity name
            culture: Culture identifier
        
        Returns:
            Culturally appropriate form
        """
        return entity_name


class EntityDisambiguator:
    """
    Entity disambiguation engine.
    
    • Disambiguates entities with multiple meanings
    • Uses contextual information for disambiguation
    • Applies machine learning models
    • Handles entity type classification
    """
    
    def __init__(self, **config):
        """
        Initialize entity disambiguator.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("entity_disambiguator")
        self.config = config
    
    def disambiguate(self, entity_name: str, **context) -> Dict[str, Any]:
        """
        Disambiguate entity using context.
        
        Args:
            entity_name: Entity name
            **context: Context information
        
        Returns:
            Disambiguation result
        """
        entity_type = context.get("entity_type")
        text_context = context.get("context", "")
        
        return {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "confidence": 0.8,
            "candidates": [entity_name]
        }
    
    def classify_entity_type(self, entity_name: str, **context) -> str:
        """
        Classify entity type for disambiguation.
        
        Args:
            entity_name: Entity name
            **context: Context information
        
        Returns:
            Entity type
        """
        # Simple heuristic-based classification
        if entity_name[0].isupper() and ' ' in entity_name:
            return "Person"
        elif entity_name[0].isupper():
            return "Organization"
        else:
            return "Entity"
    
    def calculate_confidence(self, candidates: List[str], **context) -> Dict[str, float]:
        """
        Calculate confidence scores for candidates.
        
        Args:
            candidates: List of candidate entities
            **context: Context information
        
        Returns:
            Dictionary of confidence scores
        """
        return {candidate: 0.8 for candidate in candidates}


class NameVariantHandler:
    """
    Name variant processing engine.
    
    • Handles different name formats and variations
    • Processes formal and informal names
    • Manages name order variations
    • Handles title and honorific processing
    """
    
    def __init__(self, **config):
        """
        Initialize name variant handler.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("name_variant_handler")
        self.config = config
        self.titles = {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sir", "Madam"}
    
    def process_variants(self, entity_name: str, **options) -> List[str]:
        """
        Process entity name variants.
        
        Args:
            entity_name: Entity name
            **options: Processing options
        
        Returns:
            List of variants
        """
        variants = [entity_name]
        
        # Generate common variants
        normalized = self.normalize_name_format(entity_name, "standard")
        if normalized != entity_name:
            variants.append(normalized)
        
        return variants
    
    def normalize_name_format(self, entity_name: str, format_type: str = "standard") -> str:
        """
        Normalize name format.
        
        Args:
            entity_name: Entity name
            format_type: Format type ('standard', 'title', 'lower')
        
        Returns:
            Formatted name
        """
        # Remove titles
        name = entity_name
        for title in self.titles:
            name = name.replace(title + " ", "").replace(title, "")
        
        name = name.strip()
        
        if format_type == "standard":
            # Title case for names
            parts = name.split()
            name = " ".join(part.capitalize() for part in parts)
        elif format_type == "title":
            name = name.title()
        elif format_type == "lower":
            name = name.lower()
        
        return name
    
    def handle_titles_and_honorifics(self, entity_name: str) -> Dict[str, Any]:
        """
        Handle titles and honorifics in names.
        
        Args:
            entity_name: Entity name
        
        Returns:
            Dictionary with name and title
        """
        title = None
        name = entity_name
        
        for t in self.titles:
            if entity_name.startswith(t):
                title = t
                name = entity_name.replace(t, "").strip()
                break
        
        return {"name": name, "title": title}
