"""
LLM-based extraction enhancer for Semantica framework.

This module provides LLM-based extraction capabilities
using OpenAI, Anthropic, and other language models.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .ner_extractor import Entity
from .relation_extractor import Relation


@dataclass
class LLMResponse:
    """LLM response representation."""
    
    content: str
    model: str
    usage: Dict[str, Any]
    metadata: Dict[str, Any]


class LLMEnhancer:
    """LLM-based extraction enhancer."""
    
    def __init__(self, **config):
        """
        Initialize LLM enhancer.
        
        Args:
            **config: Configuration options:
                - provider: LLM provider ("openai", "anthropic", default: "openai")
                - model: Model name (default: "gpt-3.5-turbo")
                - api_key: API key (from environment if not provided)
                - temperature: Temperature for generation
        """
        self.logger = get_logger("llm_enhancer")
        self.config = config
        
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.temperature = config.get("temperature", 0.3)
        
        # Initialize client
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider."""
        try:
            if self.provider == "openai":
                import os
                from openai import OpenAI
                api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = OpenAI(api_key=api_key)
                else:
                    self.logger.warning("OpenAI API key not found. LLM features disabled.")
            elif self.provider == "anthropic":
                import os
                from anthropic import Anthropic
                api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = Anthropic(api_key=api_key)
                else:
                    self.logger.warning("Anthropic API key not found. LLM features disabled.")
            else:
                self.logger.warning(f"Unsupported LLM provider: {self.provider}")
        except ImportError:
            self.logger.warning(f"Required library for {self.provider} not installed. LLM features disabled.")
    
    def enhance_entities(self, text: str, entities: List[Entity], **options) -> List[Entity]:
        """
        Enhance entity extraction using LLM.
        
        Args:
            text: Input text
            entities: Pre-extracted entities
            **options: Enhancement options
            
        Returns:
            list: Enhanced entities
        """
        if not self.client:
            self.logger.warning("LLM client not available. Returning original entities.")
            return entities
        
        prompt = self._build_entity_prompt(text, entities)
        
        try:
            response = self._call_llm(prompt, **options)
            enhanced_entities = self._parse_entity_response(response, entities)
            return enhanced_entities
        except Exception as e:
            self.logger.error(f"Failed to enhance entities with LLM: {e}")
            return entities
    
    def enhance_relations(self, text: str, relations: List[Relation], **options) -> List[Relation]:
        """
        Enhance relation extraction using LLM.
        
        Args:
            text: Input text
            relations: Pre-extracted relations
            **options: Enhancement options
            
        Returns:
            list: Enhanced relations
        """
        if not self.client:
            self.logger.warning("LLM client not available. Returning original relations.")
            return relations
        
        prompt = self._build_relation_prompt(text, relations)
        
        try:
            response = self._call_llm(prompt, **options)
            enhanced_relations = self._parse_relation_response(response, relations)
            return enhanced_relations
        except Exception as e:
            self.logger.error(f"Failed to enhance relations with LLM: {e}")
            return relations
    
    def _call_llm(self, prompt: str, **options) -> str:
        """Call LLM API."""
        if self.provider == "openai" and self.client:
            response = self.client.chat.completions.create(
                model=options.get("model", self.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=options.get("temperature", self.temperature)
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic" and self.client:
            response = self.client.messages.create(
                model=options.get("model", self.model),
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        raise ProcessingError("LLM client not available")
    
    def _build_entity_prompt(self, text: str, entities: List[Entity]) -> str:
        """Build prompt for entity enhancement."""
        entities_str = "\n".join([f"- {e.text} ({e.label})" for e in entities])
        
        return f"""Analyze the following text and enhance the entity extraction:

Text:
{text}

Extracted Entities:
{entities_str}

Please:
1. Verify each entity is correctly identified
2. Suggest any missing entities
3. Improve entity type classifications
4. Provide confidence scores

Return the enhanced entity list in JSON format."""
    
    def _build_relation_prompt(self, text: str, relations: List[Relation]) -> str:
        """Build prompt for relation enhancement."""
        relations_str = "\n".join([
            f"- {r.subject.text} --[{r.predicate}]--> {r.object.text}"
            for r in relations
        ])
        
        return f"""Analyze the following text and enhance the relation extraction:

Text:
{text}

Extracted Relations:
{relations_str}

Please:
1. Verify each relation is correct
2. Suggest any missing relations
3. Improve relation type classifications
4. Provide confidence scores

Return the enhanced relation list in JSON format."""
    
    def _parse_entity_response(self, response: str, original_entities: List[Entity]) -> List[Entity]:
        """Parse LLM response for entities."""
        # Simplified parsing - in practice would parse JSON
        # For now, return original entities
        return original_entities
    
    def _parse_relation_response(self, response: str, original_relations: List[Relation]) -> List[Relation]:
        """Parse LLM response for relations."""
        # Simplified parsing - in practice would parse JSON
        # For now, return original relations
        return original_relations
