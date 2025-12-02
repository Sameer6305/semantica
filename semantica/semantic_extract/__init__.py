"""
Semantic Extraction Module

This module provides comprehensive semantic extraction capabilities for knowledge
engineering, enabling extraction of entities, relations, events, triples, and
semantic networks from text.

Key Features:
    - Named entity recognition (NER) with multiple implementations
    - Relationship extraction between entities
    - Event detection and temporal processing
    - Coreference resolution for pronouns and entity references
    - RDF triple extraction and serialization
    - Semantic analysis and role labeling
    - Semantic network construction
    - LLM-based extraction enhancement
    - Extraction validation and quality assessment

Main Classes:
    - NamedEntityRecognizer: Main NER coordinator (confidence_threshold, merge_overlapping)
    - NERExtractor: Core NER implementation
    - RelationExtractor: Relationship extraction (confidence_threshold, bidirectional)
    - EventDetector: Event detection and classification (extract_participants, extract_time)
    - CoreferenceResolver: Coreference resolution
    - TripleExtractor: RDF triple extraction (include_temporal, include_provenance)
    - SemanticAnalyzer: Semantic analysis engine
    - SemanticNetworkExtractor: Semantic network construction
    - LLMEnhancer: LLM-based enhancement
    - ExtractionValidator: Quality validation

Example Usage:
    >>> from semantica.semantic_extract import NamedEntityRecognizer
    >>> ner = NamedEntityRecognizer(confidence_threshold=0.7)
    >>> entities = ner.extract_entities("Steve Jobs founded Apple.")
    
    >>> from semantica.semantic_extract import RelationExtractor
    >>> rel_extractor = RelationExtractor(confidence_threshold=0.6)
    >>> relations = rel_extractor.extract_relations(text, entities=entities)
    
    >>> from semantica.semantic_extract import TripleExtractor
    >>> triple_extractor = TripleExtractor(include_temporal=True)
    >>> triples = triple_extractor.extract_triples(text)

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union

from .config import Config, config
from .coreference_resolver import (
    CoreferenceChain,
    CoreferenceChainBuilder,
    CoreferenceResolver,
    EntityCoreferenceDetector,
    Mention,
    PronounResolver,
)
from .event_detector import (
    Event,
    EventClassifier,
    EventDetector,
    EventRelationshipExtractor,
    TemporalEventProcessor,
)
from .extraction_validator import ExtractionValidator, ValidationResult
from .llm_enhancer import LLMEnhancer, LLMResponse
from .methods import get_entity_method, get_relation_method, get_triple_method
from .named_entity_recognizer import (
    CustomEntityDetector,
    EntityClassifier,
    EntityConfidenceScorer,
    NamedEntityRecognizer,
)
from .ner_extractor import Entity, NERExtractor
from .providers import (
    AnthropicProvider,
    BaseProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceLLMProvider,
    HuggingFaceModelLoader,
    OllamaProvider,
    OpenAIProvider,
    create_provider,
)
from .registry import (
    MethodRegistry,
    ProviderRegistry,
    method_registry,
    provider_registry,
)
from .relation_extractor import Relation, RelationExtractor
from .semantic_analyzer import (
    RoleLabeler,
    SemanticAnalyzer,
    SemanticCluster,
    SemanticClusterer,
    SemanticRole,
    SimilarityAnalyzer,
)
from .semantic_network_extractor import (
    SemanticEdge,
    SemanticNetwork,
    SemanticNetworkExtractor,
    SemanticNode,
)
from .triple_extractor import (
    RDFSerializer,
    Triple,
    TripleExtractor,
    TripleQualityChecker,
    TripleValidator,
)

__all__ = [
    # Named Entity Recognition
    "NamedEntityRecognizer",
    "EntityClassifier",
    "EntityConfidenceScorer",
    "CustomEntityDetector",
    "NERExtractor",
    "Entity",
    # Relation Extraction
    "RelationExtractor",
    "Relation",
    # Event Detection
    "EventDetector",
    "Event",
    "EventClassifier",
    "TemporalEventProcessor",
    "EventRelationshipExtractor",
    # Coreference Resolution
    "CoreferenceResolver",
    "Mention",
    "CoreferenceChain",
    "PronounResolver",
    "EntityCoreferenceDetector",
    "CoreferenceChainBuilder",
    # Triple Extraction
    "TripleExtractor",
    "Triple",
    "TripleValidator",
    "RDFSerializer",
    "TripleQualityChecker",
    # Semantic Analysis
    "SemanticAnalyzer",
    "SemanticRole",
    "SemanticCluster",
    "SimilarityAnalyzer",
    "RoleLabeler",
    "SemanticClusterer",
    # Semantic Network
    "SemanticNetworkExtractor",
    "SemanticNode",
    "SemanticEdge",
    "SemanticNetwork",
    # LLM Enhancement
    "LLMEnhancer",
    "LLMResponse",
    # Validation
    "ExtractionValidator",
    "ValidationResult",
    # Providers
    "BaseProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "GroqProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "HuggingFaceLLMProvider",
    "HuggingFaceModelLoader",
    "create_provider",
    # Registry
    "ProviderRegistry",
    "MethodRegistry",
    "provider_registry",
    "method_registry",
    # Config
    "Config",
    "config",
    # Methods
    "get_entity_method",
    "get_relation_method",
    "get_triple_method",
]
