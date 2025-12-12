"""
Extraction Methods Module

This module provides all extraction methods as simple, reusable functions for
entities, relations, and triples. It supports multiple extraction approaches
ranging from simple pattern matching to advanced LLM-based extraction.

Supported Methods:

Entity Extraction:
    - "pattern": Pattern-based entity extraction using regex
    - "regex": Advanced regex-based entity extraction with custom patterns
    - "rules": Rule-based entity extraction using linguistic rules
    - "ml": ML-based entity extraction using spaCy
    - "huggingface": HuggingFace NER model extraction
    - "llm": LLM-based entity extraction

Relation Extraction:
    - "pattern": Pattern-based relation extraction
    - "regex": Advanced regex-based relation extraction
    - "cooccurrence": Co-occurrence based relation detection
    - "dependency": Dependency parsing-based relation extraction
    - "huggingface": HuggingFace relation extraction models
    - "llm": LLM-based relation extraction

Triple Extraction:
    - "pattern": Pattern-based triple extraction
    - "rules": Rule-based triple extraction
    - "huggingface": HuggingFace triplet extraction models
    - "llm": LLM-based triple extraction

Algorithms Used:

Entity Extraction:
    - Regular Expression Matching: Finite automata-based pattern matching
    - Rule-based NLP: Linguistic rule application and pattern matching
    - Neural NER: CNN/Transformer-based named entity recognition (spaCy)
    - Transformer Token Classification: BERT/RoBERTa for token-level classification
    - LLM Generation: Transformer-based language models for entity extraction

Relation Extraction:
    - Pattern Matching: Regex and string pattern matching algorithms
    - Co-occurrence Analysis: Proximity-based entity relationship detection
    - Dependency Parsing: Transition-based and graph-based parsing algorithms
    - Sequence Classification: Transformer-based relation classification
    - LLM Generation: Language model-based relation extraction

Triple Extraction:
    - Pattern Matching: Subject-predicate-object pattern extraction
    - Rule-based Extraction: Linguistic rule application
    - Seq2Seq Models: Encoder-decoder transformer models for triplet generation
    - LLM Generation: Structured output generation from language models

Key Features:
    - Multiple extraction methods for entities:
        * Pattern-based: Simple regex pattern matching
        * Regex-based: Advanced regex with custom patterns
        * Rules-based: Linguistic rule-based extraction
        * ML-based: spaCy-based machine learning extraction
        * HuggingFace: Custom HuggingFace NER models
        * LLM-based: Large language model extraction
    - Multiple extraction methods for relations:
        * Pattern-based: Pattern matching for common relations
        * Regex-based: Advanced regex relation extraction
        * Co-occurrence: Proximity-based relation detection
        * Dependency: Dependency parsing-based extraction
        * HuggingFace: Custom HuggingFace relation models
        * LLM-based: LLM-powered relation extraction
    - Multiple extraction methods for triples:
        * Pattern-based: Pattern matching for triple extraction
        * Rules-based: Rule-based triple extraction
        * HuggingFace: Custom HuggingFace triplet models
        * LLM-based: LLM-powered triple extraction
    - Method dispatchers with registry support
    - Custom method registration capability
    - Consistent interface across all methods

Main Functions:
    - extract_entities_pattern: Pattern-based entity extraction
    - extract_entities_regex: Regex-based entity extraction
    - extract_entities_rules: Rule-based entity extraction
    - extract_entities_ml: ML-based (spaCy) entity extraction
    - extract_entities_huggingface: HuggingFace model entity extraction
    - extract_entities_llm: LLM-based entity extraction
    - extract_relations_pattern: Pattern-based relation extraction
    - extract_relations_regex: Regex-based relation extraction
    - extract_relations_cooccurrence: Co-occurrence relation extraction
    - extract_relations_dependency: Dependency parsing relation extraction
    - extract_relations_huggingface: HuggingFace relation extraction
    - extract_relations_llm: LLM-based relation extraction
    - extract_triples_pattern: Pattern-based triple extraction
    - extract_triples_rules: Rule-based triple extraction
    - extract_triples_huggingface: HuggingFace triple extraction
    - extract_triples_llm: LLM-based triple extraction
    - get_entity_method: Get entity extraction method by name
    - get_relation_method: Get relation extraction method by name
    - get_triple_method: Get triple extraction method by name

Example Usage:
    >>> from semantica.semantic_extract.methods import get_entity_method
    >>> extract_fn = get_entity_method("llm")
    >>> entities = extract_fn("Apple Inc. was founded in 1976.", provider="openai")

Author: Semantica Contributors
License: MIT
"""

import re
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .ner_extractor import Entity
from .providers import HuggingFaceModelLoader, create_provider
from .registry import method_registry
from .relation_extractor import Relation
from .triple_extractor import Triple

logger = get_logger("methods")

# Try to import spaCy
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ============================================================================
# Entity Extraction Methods
# ============================================================================


def extract_entities_pattern(text: str, **kwargs) -> List[Entity]:
    """Pattern-based entity extraction using regex."""
    entities = []

    patterns = {
        "PERSON": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
        "ORG": r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company))\b",
        "GPE": r"\b([A-Z][a-z]+\s*(?:City|State|Country|Nation))\b",
        "DATE": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b",
    }

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            entities.append(
                Entity(
                    text=match.group(1) if match.groups() else match.group(0),
                    label=label,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.7,
                    metadata={"extraction_method": "pattern"},
                )
            )

    return entities


def extract_entities_regex(
    text: str, patterns: Optional[Dict[str, str]] = None, **kwargs
) -> List[Entity]:
    """Advanced regex-based entity extraction with custom patterns."""
    entities = []

    if patterns is None:
        patterns = {
            "PERSON": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
            "ORG": r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company|Corporation))\b",
            "GPE": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            "DATE": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4})\b",
            "MONEY": r"\b(\$[\d,]+(?:\.\d{2})?)\b",
            "PERCENT": r"\b(\d+(?:\.\d+)?%)\b",
        }

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append(
                Entity(
                    text=match.group(1) if match.groups() else match.group(0),
                    label=label,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.75,
                    metadata={"extraction_method": "regex", "pattern": pattern},
                )
            )

    return entities


def extract_entities_rules(text: str, **kwargs) -> List[Entity]:
    """Rule-based entity extraction using linguistic rules."""
    entities = []
    words = text.split()

    # Rule: Capitalized words at sentence start are likely entities
    sentences = re.split(r"[.!?]+", text)
    char_offset = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            char_offset += 1
            continue

        words_in_sent = sentence.split()
        if words_in_sent:
            first_word = words_in_sent[0]
            if first_word and first_word[0].isupper() and len(first_word) > 2:
                start = text.find(first_word, char_offset)
                if start >= 0:
                    entities.append(
                        Entity(
                            text=first_word,
                            label="PERSON",  # Default assumption
                            start_char=start,
                            end_char=start + len(first_word),
                            confidence=0.6,
                            metadata={
                                "extraction_method": "rules",
                                "rule": "sentence_start",
                            },
                        )
                    )

        char_offset += len(sentence) + 1

    return entities


def extract_entities_ml(
    text: str, model: str = "en_core_web_sm", **kwargs
) -> List[Entity]:
    """ML-based entity extraction using spaCy."""
    if not SPACY_AVAILABLE:
        logger.warning("spaCy not available, falling back to pattern extraction")
        return extract_entities_pattern(text, **kwargs)

    try:
        nlp = spacy.load(model)
    except OSError:
        logger.warning(f"spaCy model {model} not found, using en_core_web_sm")
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model not available, falling back to pattern extraction"
            )
            return extract_entities_pattern(text, **kwargs)

    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        confidence = 1.0
        if hasattr(ent, "confidence"):
            confidence = ent.confidence
        elif hasattr(ent, "score"):
            confidence = ent.score

        entities.append(
            Entity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=confidence,
                metadata={
                    "extraction_method": "ml",
                    "model": model,
                    "lemma": ent.lemma_ if hasattr(ent, "lemma_") else ent.text,
                },
            )
        )

    return entities


def extract_entities_huggingface(
    text: str,
    model: str = "dslim/bert-base-NER",
    device: Optional[str] = None,
    **kwargs,
) -> List[Entity]:
    """HuggingFace entity extraction."""
    loader = HuggingFaceModelLoader(device=device)
    model_obj = loader.load_ner_model(model)
    results = loader.extract_entities(model_obj, text)

    entities = []
    for result in results:
        if isinstance(result, dict):
            entities.append(
                Entity(
                    text=result.get("word", result.get("entity", "")),
                    label=result.get("entity_group", result.get("label", "UNKNOWN")),
                    start_char=result.get("start", 0),
                    end_char=result.get("end", 0),
                    confidence=result.get("score", 1.0),
                    metadata={"model": model, "extraction_method": "huggingface"},
                )
            )

    return entities


def extract_entities_llm(
    text: str, provider: str = "openai", model: Optional[str] = None, **kwargs
) -> List[Entity]:
    """LLM-based entity extraction."""
    # Support llm_model parameter to disambiguate from ML model
    if "llm_model" in kwargs:
        model = kwargs.pop("llm_model")

    llm = create_provider(provider, model=model, **kwargs)

    if not llm.is_available():
        raise ProcessingError(f"{provider} provider not available")

    prompt = f"""Extract named entities from the following text. 
Return JSON format: [{{"text": "...", "label": "PERSON|ORG|GPE|DATE|...", "start": 0, "end": 10}}]

Text: {text}"""

    try:
        result = llm.generate_structured(prompt)
        entities = []

        if isinstance(result, list):
            for item in result:
                entities.append(
                    Entity(
                        text=item.get("text", ""),
                        label=item.get("label", "UNKNOWN"),
                        start_char=item.get("start", 0),
                        end_char=item.get("end", 0),
                        confidence=item.get("confidence", 0.9),
                        metadata={
                            "provider": provider,
                            "model": model,
                            "extraction_method": "llm",
                        },
                    )
                )
        elif isinstance(result, dict) and "entities" in result:
            for item in result["entities"]:
                entities.append(
                    Entity(
                        text=item.get("text", ""),
                        label=item.get("label", "UNKNOWN"),
                        start_char=item.get("start", 0),
                        end_char=item.get("end", 0),
                        confidence=item.get("confidence", 0.9),
                        metadata={
                            "provider": provider,
                            "model": model,
                            "extraction_method": "llm",
                        },
                    )
                )

        return entities
    except Exception as e:
        logger.error(f"LLM entity extraction failed: {e}")
        return []


# ============================================================================
# Relation Extraction Methods
# ============================================================================


def extract_relations_pattern(
    text: str, entities: List[Entity], **kwargs
) -> List[Relation]:
    """Pattern-based relation extraction."""
    relations = []

    relation_patterns = {
        "founded_by": [
            r"(?P<subject>\w+)\s+(?:was\s+)?founded\s+by\s+(?P<object>\w+(?:\s+\w+)*)",
            r"(?P<object>\w+(?:\s+\w+)*)\s+founded\s+(?P<subject>\w+)",
        ],
        "located_in": [
            r"(?P<subject>\w+)\s+is\s+located\s+in\s+(?P<object>\w+)",
            r"(?P<subject>\w+)\s+in\s+(?P<object>\w+)",
        ],
        "works_for": [
            r"(?P<subject>\w+)\s+works?\s+for\s+(?P<object>\w+)",
            r"(?P<subject>\w+)\s+is\s+an?\s+employee\s+of\s+(?P<object>\w+)",
        ],
        "born_in": [
            r"(?P<subject>\w+)\s+was\s+born\s+in\s+(?P<object>\w+)",
            r"(?P<subject>\w+)\s+born\s+in\s+(?P<object>\w+)",
        ],
    }

    entity_map = {e.text.lower(): e for e in entities}

    for relation_type, patterns in relation_patterns.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject_text = match.group("subject")
                object_text = match.group("object")

                subject_entity = entity_map.get(subject_text.lower())
                object_entity = entity_map.get(object_text.lower())

                if subject_entity and object_entity:
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]

                    relations.append(
                        Relation(
                            subject=subject_entity,
                            predicate=relation_type,
                            object=object_entity,
                            confidence=0.7,
                            context=context,
                            metadata={
                                "extraction_method": "pattern",
                                "pattern": pattern,
                            },
                        )
                    )

    return relations


def extract_relations_regex(
    text: str,
    entities: List[Entity],
    patterns: Optional[Dict[str, List[str]]] = None,
    **kwargs,
) -> List[Relation]:
    """Advanced regex-based relation extraction."""
    if patterns is None:
        patterns = {
            "founded_by": [
                r"(?P<subject>\w+)\s+(?:was\s+)?founded\s+by\s+(?P<object>\w+(?:\s+\w+)*)"
            ],
            "located_in": [r"(?P<subject>\w+)\s+is\s+located\s+in\s+(?P<object>\w+)"],
        }

    relations = []
    entity_map = {e.text.lower(): e for e in entities}

    for relation_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subject_text = match.group("subject")
                object_text = match.group("object")

                subject_entity = entity_map.get(subject_text.lower())
                object_entity = entity_map.get(object_text.lower())

                if subject_entity and object_entity:
                    relations.append(
                        Relation(
                            subject=subject_entity,
                            predicate=relation_type,
                            object=object_entity,
                            confidence=0.75,
                            context=text[
                                max(0, match.start() - 30) : min(
                                    len(text), match.end() + 30
                                )
                            ],
                            metadata={"extraction_method": "regex"},
                        )
                    )

    return relations


def extract_relations_cooccurrence(
    text: str, entities: List[Entity], **kwargs
) -> List[Relation]:
    """Co-occurrence based relation extraction."""
    relations = []

    for i, entity1 in enumerate(entities):
        for entity2 in entities[i + 1 :]:
            distance = abs(entity1.end_char - entity2.start_char)
            if distance < 100:  # Within 100 characters
                start = min(entity1.start_char, entity2.start_char)
                end = max(entity1.end_char, entity2.end_char)
                context = text[max(0, start - 30) : min(len(text), end + 30)]

                relations.append(
                    Relation(
                        subject=entity1,
                        predicate="related_to",
                        object=entity2,
                        confidence=0.5,
                        context=context,
                        metadata={
                            "extraction_method": "co_occurrence",
                            "distance": distance,
                        },
                    )
                )

    return relations


def extract_relations_dependency(
    text: str, entities: List[Entity], model: str = "en_core_web_sm", **kwargs
) -> List[Relation]:
    """Dependency parsing based relation extraction."""
    if not SPACY_AVAILABLE:
        logger.warning("spaCy not available, falling back to pattern extraction")
        return extract_relations_pattern(text, entities, **kwargs)

    try:
        nlp = spacy.load(model)
    except OSError:
        logger.warning(f"spaCy model {model} not found")
        return extract_relations_pattern(text, entities, **kwargs)

    doc = nlp(text)
    relations = []
    entity_map = {e.text.lower(): e for e in entities}

    for token in doc:
        # Look for subject-verb-object patterns
        if token.dep_ == "nsubj" or token.dep_ == "nsubjpass":
            subject = token.text
            verb = token.head
            # Find object
            for child in verb.children:
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    obj = child.text

                    subject_entity = entity_map.get(subject.lower())
                    object_entity = entity_map.get(obj.lower())

                    if subject_entity and object_entity:
                        relations.append(
                            Relation(
                                subject=subject_entity,
                                predicate=verb.lemma_,
                                object=object_entity,
                                confidence=0.8,
                                context=text[
                                    max(0, token.idx - 30) : min(
                                        len(text), child.idx + len(obj) + 30
                                    )
                                ],
                                metadata={
                                    "extraction_method": "dependency",
                                    "dependency_path": f"{token.dep_} -> {child.dep_}",
                                },
                            )
                        )

    return relations


def extract_relations_huggingface(
    text: str,
    entities: List[Entity],
    model: str,
    device: Optional[str] = None,
    **kwargs,
) -> List[Relation]:
    """HuggingFace relation extraction."""
    loader = HuggingFaceModelLoader(device=device)
    model_obj = loader.load_relation_model(model)

    # This is simplified - actual implementation would depend on model architecture
    results = loader.extract_relations(model_obj, text, entities)

    relations = []
    # Parse results based on model output format
    # This is a placeholder - actual parsing would depend on the model
    return relations


def extract_relations_llm(
    text: str,
    entities: List[Entity],
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> List[Relation]:
    """LLM-based relation extraction."""
    llm = create_provider(provider, model=model, **kwargs)

    if not llm.is_available():
        raise ProcessingError(f"{provider} provider not available")

    entities_str = ", ".join([f"{e.text} ({e.label})" for e in entities])
    prompt = f"""Extract relations between entities from the following text.

Text: {text}
Entities: {entities_str}

Return JSON format: [{{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}]"""

    try:
        result = llm.generate_structured(prompt)
        relations = []

        if isinstance(result, list):
            for item in result:
                # Find matching entities
                subject_text = item.get("subject", "")
                object_text = item.get("object", "")

                subject_entity = next(
                    (e for e in entities if e.text.lower() == subject_text.lower()),
                    None,
                )
                object_entity = next(
                    (e for e in entities if e.text.lower() == object_text.lower()), None
                )

                if subject_entity and object_entity:
                    relations.append(
                        Relation(
                            subject=subject_entity,
                            predicate=item.get("predicate", "related_to"),
                            object=object_entity,
                            confidence=item.get("confidence", 0.9),
                            context=text,
                            metadata={
                                "provider": provider,
                                "model": model,
                                "extraction_method": "llm",
                            },
                        )
                    )

        return relations
    except Exception as e:
        logger.error(f"LLM relation extraction failed: {e}")
        return []


# ============================================================================
# Triple Extraction Methods
# ============================================================================


def extract_triples_pattern(
    text: str,
    entities: Optional[List[Entity]] = None,
    relations: Optional[List[Relation]] = None,
    **kwargs,
) -> List[Triple]:
    """Pattern-based triple extraction."""
    triples = []

    if relations:
        # Convert relations to triples
        for relation in relations:
            triples.append(
                Triple(
                    subject=relation.subject.text,
                    predicate=relation.predicate,
                    object=relation.object.text,
                    confidence=relation.confidence,
                    metadata={"extraction_method": "pattern", **relation.metadata},
                )
            )
    elif entities:
        # Simple triple extraction from entities
        # Look for subject-verb-object patterns
        pattern = r"(?P<subject>\w+)\s+(?P<predicate>\w+)\s+(?P<object>\w+)"
        for match in re.finditer(pattern, text):
            subject_text = match.group("subject")
            predicate_text = match.group("predicate")
            object_text = match.group("object")

            subject_entity = next(
                (e for e in entities if e.text.lower() == subject_text.lower()), None
            )
            object_entity = next(
                (e for e in entities if e.text.lower() == object_text.lower()), None
            )

            if subject_entity and object_entity:
                triples.append(
                    Triple(
                        subject=subject_entity.text,
                        predicate=predicate_text,
                        object=object_entity.text,
                        confidence=0.7,
                        metadata={"extraction_method": "pattern"},
                    )
                )

    return triples


def extract_triples_rules(
    text: str, entities: Optional[List[Entity]] = None, **kwargs
) -> List[Triple]:
    """Rule-based triple extraction."""
    triples = []

    if not entities:
        return triples

    # Rule: Look for verb patterns between entities
    sentences = re.split(r"[.!?]+", text)
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            if word.lower() in ["is", "was", "has", "founded", "located"]:
                # Look for entities before and after
                if i > 0 and i < len(words) - 1:
                    before = " ".join(words[:i])
                    after = " ".join(words[i + 1 :])

                    subject_entity = next(
                        (e for e in entities if e.text.lower() in before.lower()), None
                    )
                    object_entity = next(
                        (e for e in entities if e.text.lower() in after.lower()), None
                    )

                    if subject_entity and object_entity:
                        triples.append(
                            Triple(
                                subject=subject_entity.text,
                                predicate=word,
                                object=object_entity.text,
                                confidence=0.7,
                                metadata={"extraction_method": "rules"},
                            )
                        )

    return triples


def extract_triples_huggingface(
    text: str, model: str, device: Optional[str] = None, **kwargs
) -> List[Triple]:
    """HuggingFace triple extraction."""
    loader = HuggingFaceModelLoader(device=device)
    model_obj = loader.load_triplet_model(model)
    results = loader.extract_triplets(model_obj, text)

    triples = []
    for result in results:
        # Parse result based on model output format
        # This is a placeholder - actual parsing would depend on the model
        if "triplet" in result:
            # Parse triplet string (format depends on model)
            pass

    return triples


def extract_triples_llm(
    text: str,
    entities: Optional[List[Entity]] = None,
    relations: Optional[List[Relation]] = None,
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs,
) -> List[Triple]:
    """LLM-based triple extraction."""
    llm = create_provider(provider, model=model, **kwargs)

    if not llm.is_available():
        raise ProcessingError(f"{provider} provider not available")

    prompt = f"""Extract RDF triples (subject-predicate-object) from the following text.

Text: {text}

Return JSON format: [{{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}]"""

    try:
        result = llm.generate_structured(prompt)
        triples = []

        if isinstance(result, list):
            for item in result:
                triples.append(
                    Triple(
                        subject=item.get("subject", ""),
                        predicate=item.get("predicate", ""),
                        object=item.get("object", ""),
                        confidence=item.get("confidence", 0.9),
                        metadata={
                            "provider": provider,
                            "model": model,
                            "extraction_method": "llm",
                        },
                    )
                )

        return triples
    except Exception as e:
        logger.error(f"LLM triple extraction failed: {e}")
        return []


# ============================================================================
# Method Dispatchers
# ============================================================================


def get_entity_method(method_name: str):
    """Get entity extraction method - checks registry for custom methods."""
    # Check registry first
    custom_method = method_registry.get("entity", method_name)
    if custom_method:
        return custom_method

    # Built-in methods
    builtin = {
        "pattern": extract_entities_pattern,
        "regex": extract_entities_regex,
        "rules": extract_entities_rules,
        "ml": extract_entities_ml,
        "huggingface": extract_entities_huggingface,
        "llm": extract_entities_llm,
    }

    method_func = builtin.get(method_name)
    if not method_func:
        raise ValueError(
            f"Unknown method: {method_name}. Register custom method or use built-in: {list(builtin.keys())}"
        )

    return method_func


def get_relation_method(method_name: str):
    """Get relation extraction method - checks registry for custom methods."""
    # Check registry first
    custom_method = method_registry.get("relation", method_name)
    if custom_method:
        return custom_method

    # Built-in methods
    builtin = {
        "pattern": extract_relations_pattern,
        "regex": extract_relations_regex,
        "cooccurrence": extract_relations_cooccurrence,
        "dependency": extract_relations_dependency,
        "huggingface": extract_relations_huggingface,
        "llm": extract_relations_llm,
    }

    method_func = builtin.get(method_name)
    if not method_func:
        raise ValueError(
            f"Unknown method: {method_name}. Register custom method or use built-in: {list(builtin.keys())}"
        )

    return method_func


def get_triple_method(method_name: str):
    """Get triple extraction method - checks registry for custom methods."""
    # Check registry first
    custom_method = method_registry.get("triple", method_name)
    if custom_method:
        return custom_method

    # Built-in methods
    builtin = {
        "pattern": extract_triples_pattern,
        "rules": extract_triples_rules,
        "huggingface": extract_triples_huggingface,
        "llm": extract_triples_llm,
    }

    method_func = builtin.get(method_name)
    if not method_func:
        raise ValueError(
            f"Unknown method: {method_name}. Register custom method or use built-in: {list(builtin.keys())}"
        )

    return method_func
