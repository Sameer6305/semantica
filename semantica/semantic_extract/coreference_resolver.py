"""
Coreference Resolution Module

Handles resolution of coreferences and pronoun references.

Key Features:
    - Pronoun resolution
    - Entity coreference detection
    - Coreference chain construction
    - Ambiguity resolution
    - Cross-document coreference

Main Classes:
    - CoreferenceResolver: Main coreference resolution class
    - PronounResolver: Pronoun resolution engine
    - EntityCoreferenceDetector: Entity coreference detection
    - CoreferenceChainBuilder: Coreference chain construction
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .ner_extractor import Entity


@dataclass
class Mention:
    """Mention representation."""
    
    text: str
    start_char: int
    end_char: int
    mention_type: str  # pronoun, entity, nominal
    entity_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoreferenceChain:
    """Coreference chain representation."""
    
    mentions: List[Mention]
    representative: Mention
    entity_type: Optional[str] = None


class CoreferenceResolver:
    """Coreference resolution handler."""
    
    def __init__(self, config=None, **kwargs):
        """Initialize coreference resolver."""
        self.logger = get_logger("coreference_resolver")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.pronoun_resolver = PronounResolver(**self.config.get("pronoun", {}))
        self.entity_detector = EntityCoreferenceDetector(**self.config.get("entity", {}))
        self.chain_builder = CoreferenceChainBuilder(**self.config.get("chain", {}))
    
    def resolve_coreferences(self, text: str, **options) -> List[CoreferenceChain]:
        """
        Resolve coreferences in text.
        
        Args:
            text: Input text
            **options: Resolution options
            
        Returns:
            list: List of coreference chains
        """
        # Extract mentions
        mentions = self._extract_mentions(text)
        
        # Resolve pronouns
        pronoun_resolutions = self.pronoun_resolver.resolve_pronouns(text, mentions, **options)
        
        # Detect entity coreferences
        entity_corefs = self.entity_detector.detect_entity_coreferences(text, mentions, **options)
        
        # Build chains
        chains = self.chain_builder.build_coreference_chains(mentions, **options)
        
        return chains
    
    def _extract_mentions(self, text: str) -> List[Mention]:
        """Extract all mentions from text."""
        mentions = []
        
        # Extract pronouns
        pronoun_patterns = {
            "he": r"\bhe\b",
            "she": r"\bshe\b",
            "it": r"\bit\b",
            "they": r"\bthey\b",
            "his": r"\bhis\b",
            "her": r"\bher\b",
            "their": r"\btheir\b"
        }
        
        for pronoun, pattern in pronoun_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                mentions.append(Mention(
                    text=match.group(0),
                    start_char=match.start(),
                    end_char=match.end(),
                    mention_type="pronoun",
                    metadata={"pronoun": pronoun}
                ))
        
        return mentions
    
    def build_coreference_chains(self, mentions: List[Mention], **options) -> List[CoreferenceChain]:
        """
        Build coreference chains from mentions.
        
        Args:
            mentions: List of mentions
            **options: Chain building options
            
        Returns:
            list: List of coreference chains
        """
        return self.chain_builder.build_coreference_chains(mentions, **options)
    
    def resolve_pronouns(self, text: str, **options) -> List[Tuple[str, str]]:
        """
        Resolve pronoun references in text.
        
        Args:
            text: Input text
            **options: Resolution options
            
        Returns:
            list: List of (pronoun, antecedent) tuples
        """
        mentions = self._extract_mentions(text)
        return self.pronoun_resolver.resolve_pronouns(text, mentions, **options)
    
    def detect_entity_coreferences(self, text: str, entities: List[Entity], **options) -> List[CoreferenceChain]:
        """
        Detect entity coreferences in text.
        
        Args:
            text: Input text
            entities: List of entities
            **options: Detection options
            
        Returns:
            list: List of coreference chains
        """
        # Convert entities to mentions
        mentions = [
            Mention(
                text=e.text,
                start_char=e.start_char,
                end_char=e.end_char,
                mention_type="entity",
                metadata={"entity_label": e.label}
            )
            for e in entities
        ]
        
        return self.entity_detector.detect_entity_coreferences(text, mentions, **options)


class PronounResolver:
    """Pronoun resolution engine."""
    
    def __init__(self, **config):
        """Initialize pronoun resolver."""
        self.logger = get_logger("pronoun_resolver")
        self.config = config
    
    def resolve_pronouns(
        self,
        text: str,
        mentions: List[Mention],
        **options
    ) -> List[Tuple[str, str]]:
        """
        Resolve pronoun references in text.
        
        Args:
            text: Input text
            mentions: List of mentions
            **options: Resolution options
            
        Returns:
            list: List of (pronoun, antecedent) tuples
        """
        resolutions = []
        
        # Get pronouns and entities
        pronouns = [m for m in mentions if m.mention_type == "pronoun"]
        entities = [m for m in mentions if m.mention_type == "entity" or m.mention_type == "nominal"]
        
        # Simple resolution: find closest preceding entity
        for pronoun in pronouns:
            # Find preceding entities
            preceding = [e for e in entities if e.end_char < pronoun.start_char]
            
            if preceding:
                # Take closest
                antecedent = max(preceding, key=lambda e: e.end_char)
                resolutions.append((pronoun.text, antecedent.text))
        
        return resolutions


class EntityCoreferenceDetector:
    """Entity coreference detection."""
    
    def __init__(self, **config):
        """Initialize entity coreference detector."""
        self.logger = get_logger("entity_coreference_detector")
        self.config = config
    
    def detect_entity_coreferences(
        self,
        text: str,
        mentions: List[Mention],
        **options
    ) -> List[CoreferenceChain]:
        """
        Detect entity coreferences in text.
        
        Args:
            text: Input text
            mentions: List of mentions
            **options: Detection options
            
        Returns:
            list: List of coreference chains
        """
        chains = []
        
        # Group mentions by text similarity
        entity_mentions = [m for m in mentions if m.mention_type in ["entity", "nominal"]]
        
        # Simple grouping by exact text match
        text_groups = {}
        for mention in entity_mentions:
            key = mention.text.lower()
            if key not in text_groups:
                text_groups[key] = []
            text_groups[key].append(mention)
        
        # Create chains from groups
        for key, group_mentions in text_groups.items():
            if len(group_mentions) > 1:
                # Use first mention as representative
                representative = group_mentions[0]
                chain = CoreferenceChain(
                    mentions=group_mentions,
                    representative=representative
                )
                chains.append(chain)
        
        return chains


class CoreferenceChainBuilder:
    """Coreference chain construction."""
    
    def __init__(self, **config):
        """Initialize coreference chain builder."""
        self.logger = get_logger("coreference_chain_builder")
        self.config = config
    
    def build_coreference_chains(self, mentions: List[Mention], **options) -> List[CoreferenceChain]:
        """
        Build coreference chains from mentions.
        
        Args:
            mentions: List of mentions
            **options: Chain building options
            
        Returns:
            list: List of coreference chains
        """
        chains = []
        
        # Simple implementation: group by text similarity
        processed = set()
        
        for mention in mentions:
            if mention.text.lower() in processed:
                continue
            
            # Find similar mentions
            similar = [
                m for m in mentions
                if m.text.lower() == mention.text.lower() or
                self._similar_mentions(mention.text, m.text)
            ]
            
            if len(similar) > 1:
                processed.add(mention.text.lower())
                
                # Representative is first (leftmost) mention
                representative = min(similar, key=lambda m: m.start_char)
                
                chain = CoreferenceChain(
                    mentions=similar,
                    representative=representative,
                    entity_type=similar[0].metadata.get("entity_label")
                )
                chains.append(chain)
        
        return chains
    
    def _similar_mentions(self, text1: str, text2: str) -> bool:
        """Check if two mentions are similar."""
        t1_lower = text1.lower()
        t2_lower = text2.lower()
        
        # Exact match
        if t1_lower == t2_lower:
            return True
        
        # One contains the other
        if t1_lower in t2_lower or t2_lower in t1_lower:
            return True
        
        # Word overlap
        words1 = set(t1_lower.split())
        words2 = set(t2_lower.split())
        overlap = len(words1 & words2) / max(len(words1), len(words2)) if words1 or words2 else 0
        
        return overlap > 0.7
