"""
RDF Triple Extraction Module

This module provides comprehensive RDF triple extraction capabilities, enabling
conversion of entities and relations into RDF triples using multiple extraction
methods, with validation and serialization support.

Supported Methods:
    - "pattern": Pattern-based triple extraction from relations (default)
    - "rules": Rule-based triple extraction using linguistic rules
    - "huggingface": Custom HuggingFace triplet extraction models
    - "llm": LLM-based triple extraction using various providers

Algorithms Used:
    - Pattern Matching: Regex-based subject-predicate-object extraction
    - Rule-based Extraction: Linguistic rule application for triple formation
    - Sequence-to-Sequence Models: Transformer-based seq2seq for triplet generation
    - Large Language Models: GPT, Claude, Gemini for structured triple extraction
    - RDF Serialization: Graph serialization algorithms (Turtle, N-Triples, JSON-LD)
    - URI Normalization: String normalization and URI formatting algorithms

Key Features:
    - Multiple extraction methods:
        * Pattern-based: Pattern matching for triple extraction (default)
        * Rules-based: Rule-based triple extraction
        * HuggingFace: Custom HuggingFace triplet models
        * LLM-based: LLM-powered triple extraction
    - Fallback chain support: Try methods in order until one succeeds
    - RDF triple generation from entities and relations
    - Subject-predicate-object extraction
    - Triple validation and quality checking
    - RDF serialization (Turtle, N-Triples, JSON-LD, RDF/XML)
    - Batch triple processing
    - URI formatting and normalization
    - Quality assessment and scoring

Main Classes:
    - TripleExtractor: Main triple extraction coordinator with method selection
    - TripleValidator: Triple validation engine
    - RDFSerializer: RDF serialization handler
    - TripleQualityChecker: Triple quality assessment
    - Triple: RDF triple representation dataclass

Example Usage:
    >>> from semantica.semantic_extract import TripleExtractor
    >>> # Using pattern method (default)
    >>> extractor = TripleExtractor(method="pattern")
    >>> triples = extractor.extract_triples(text, entities, relations)
    >>> 
    >>> # Using LLM method
    >>> extractor = TripleExtractor(method="llm", provider="openai", llm_model="gpt-4")
    >>> triples = extractor.extract_triples(text, entities, relations)
    >>> 
    >>> # Using HuggingFace model
    >>> extractor = TripleExtractor(method="huggingface", huggingface_model="custom/triplet-model")
    >>> triples = extractor.extract_triples(text)
    >>> 
    >>> # Serialize to RDF
    >>> rdf_turtle = extractor.serialize_triples(triples, format="turtle")
    >>> validated = extractor.validate_triples(triples)

Author: Semantica Contributors
License: MIT
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .ner_extractor import Entity
from .relation_extractor import Relation


@dataclass
class Triple:
    """RDF triple representation."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TripleExtractor:
    """RDF triple extraction handler."""

    def __init__(
        self,
        method: Union[str, List[str]] = "pattern",
        include_temporal: bool = False,
        include_provenance: bool = False,
        config=None,
        **kwargs
    ):
        """
        Initialize triple extractor.

        Args:
            method: Extraction method(s). Can be:
                - "pattern": Pattern-based extraction (default)
                - "rules": Rule-based extraction
                - "huggingface": HuggingFace model
                - "llm": LLM-based extraction
                - List of methods for fallback chain
            include_temporal: Whether to include temporal information in triples
            include_provenance: Whether to track source sentences for provenance
            config: Legacy config dict (deprecated, use kwargs)
            **kwargs: Configuration options:
                - model: Model name (for HuggingFace methods)
                - huggingface_model: HuggingFace model name
                - provider: LLM provider (for LLM method)
                - llm_model: LLM model name
                - device: Device for HuggingFace models
                - min_confidence: Minimum confidence threshold
                - validate: Enable validation (default: True)
        """
        self.logger = get_logger("triple_extractor")
        self.config = config or {}
        self.config.update(kwargs)
        self.progress_tracker = get_progress_tracker()

        # Store parameters
        self.include_temporal = include_temporal
        self.include_provenance = include_provenance

        # Method configuration
        self.method = method if isinstance(method, list) else [method]
        self.min_confidence = self.config.get("min_confidence", 0.5)
        self.validate_triples = self.config.get("validate", True)

        self.triple_validator = TripleValidator(**self.config.get("validator", {}))
        self.rdf_serializer = RDFSerializer(**self.config.get("serializer", {}))
        self.quality_checker = TripleQualityChecker(**self.config.get("quality", {}))

        self.supported_formats = ["turtle", "ntriples", "jsonld", "xml"]

    def extract_triples(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        relationships: Optional[List[Relation]] = None,
        **options,
    ) -> List[Triple]:
        """
        Extract RDF triples from text.

        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            relationships: Pre-extracted relations (optional)
            **options: Extraction options

        Returns:
            list: List of extracted triples
        """
        from .methods import get_triple_method

        tracking_id = self.progress_tracker.start_tracking(
            module="semantic_extract",
            submodule="TripleExtractor",
            message="Extracting RDF triples from text",
        )

        try:
            from .ner_extractor import NERExtractor
            from .relation_extractor import RelationExtractor

            # Extract entities if not provided
            if entities is None:
                self.progress_tracker.update_tracking(
                    tracking_id, message="Extracting entities..."
                )
                ner = NERExtractor(**self.config.get("ner", {}))
                entities = ner.extract_entities(text)

            # Extract relations if not provided
            if relationships is None:
                self.progress_tracker.update_tracking(
                    tracking_id, message="Extracting relations..."
                )
                rel_extractor = RelationExtractor(**self.config.get("relation", {}))
                relationships = rel_extractor.extract_relations(text, entities)

            # Use method-based extraction
            methods = options.get("method", self.method)
            if isinstance(methods, str):
                methods = [methods]

            # Merge config with options
            all_options = {**self.config, **options}

            # Try each method in order (fallback chain)
            all_triples = []
            for method_name in methods:
                try:
                    self.progress_tracker.update_tracking(
                        tracking_id,
                        message=f"Extracting triples using {method_name}...",
                    )
                    method_func = get_triple_method(method_name)

                    # Prepare method-specific options
                    method_options = all_options.copy()
                    if method_name == "huggingface":
                        method_options["model"] = all_options.get(
                            "huggingface_model", all_options.get("model")
                        )
                        method_options["device"] = all_options.get("device")
                    elif method_name == "llm":
                        method_options["provider"] = all_options.get(
                            "provider", "openai"
                        )
                        method_options["model"] = all_options.get(
                            "llm_model", all_options.get("model")
                        )

                    triples = method_func(
                        text,
                        entities=entities,
                        relations=relationships,
                        **method_options,
                    )

                    # Filter by confidence
                    min_conf = options.get("min_confidence", self.min_confidence)
                    filtered = [t for t in triples if t.confidence >= min_conf]

                    if filtered:
                        all_triples.append((method_name, filtered))

                        # If not using ensemble, return first successful result
                        if len(methods) == 1:
                            result = filtered
                            if options.get("validate", self.validate_triples):
                                result = self.triple_validator.validate_triples(result)
                            self.progress_tracker.stop_tracking(
                                tracking_id,
                                status="completed",
                                message=f"Extracted {len(result)} triples using {method_name}",
                            )
                            return result

                except Exception as e:
                    self.logger.warning(f"Method {method_name} failed: {e}")
                    continue

            # Use first successful method or fallback to relation conversion
            if all_triples:
                triples = all_triples[0][1]
            else:
                # Fallback: Convert relations to triples
                self.progress_tracker.update_tracking(
                    tracking_id,
                    message=f"Converting {len(relationships)} relations to triples...",
                )
                triples = []
                for relation in relationships:
                    triple = Triple(
                        subject=self._format_uri(relation.subject.text),
                        predicate=self._format_uri(relation.predicate),
                        object=self._format_uri(relation.object.text),
                        confidence=relation.confidence,
                        metadata={"context": relation.context, **relation.metadata},
                    )
                    triples.append(triple)

            # Validate triples
            if options.get("validate", self.validate_triples):
                self.progress_tracker.update_tracking(
                    tracking_id, message="Validating triples..."
                )
                triples = self.triple_validator.validate_triples(triples)

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Extracted {len(triples)} triples",
            )
            return triples

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def _format_uri(self, value: str) -> str:
        """Format value as URI."""
        # Simple URI formatting
        if value.startswith("http://") or value.startswith("https://"):
            return value

        # Format as local URI
        formatted = quote(value.replace(" ", "_"), safe="")
        return f"http://example.org/{formatted}"

    def validate_triples(self, triples: List[Triple], **criteria) -> List[Triple]:
        """
        Validate triple quality and consistency.

        Args:
            triples: List of triples
            **criteria: Validation criteria

        Returns:
            list: Validated triples
        """
        return self.triple_validator.validate_triples(triples, **criteria)

    def serialize_triples(
        self, triples: List[Triple], format: str = "turtle", **options
    ) -> str:
        """
        Serialize triples to RDF format.

        Args:
            triples: List of triples
            format: RDF format (turtle, ntriples, jsonld, xml)
            **options: Serialization options

        Returns:
            str: Serialized RDF
        """
        return self.rdf_serializer.serialize_to_rdf(triples, format, **options)

    def process_batch(self, texts: List[str], **options) -> List[List[Triple]]:
        """
        Process multiple texts for triple extraction.

        Args:
            texts: List of input texts
            **options: Processing options

        Returns:
            list: List of triple lists for each text
        """
        return [self.extract_triples(text, **options) for text in texts]


class TripleValidator:
    """Triple validation engine."""

    def __init__(self, **config):
        """Initialize triple validator."""
        self.logger = get_logger("triple_validator")
        self.config = config

    def validate_triple(self, triple: Triple, **criteria) -> bool:
        """
        Validate individual triple.

        Args:
            triple: Triple to validate
            **criteria: Validation criteria

        Returns:
            bool: True if valid
        """
        # Check structure
        if not triple.subject or not triple.predicate or not triple.object:
            return False

        # Check confidence
        min_confidence = criteria.get("min_confidence", 0.5)
        if triple.confidence < min_confidence:
            return False

        return True

    def validate_triples(self, triples: List[Triple], **criteria) -> List[Triple]:
        """
        Validate list of triples.

        Args:
            triples: List of triples
            **criteria: Validation criteria

        Returns:
            list: Valid triples
        """
        return [t for t in triples if self.validate_triple(t, **criteria)]

    def check_triple_consistency(self, triples: List[Triple]) -> Dict[str, Any]:
        """
        Check consistency among triples.

        Args:
            triples: List of triples

        Returns:
            dict: Consistency report
        """
        issues = []

        # Check for contradictory triples
        # (simplified - would need domain knowledge for full implementation)

        return {
            "total_triples": len(triples),
            "issues": issues,
            "consistent": len(issues) == 0,
        }


class RDFSerializer:
    """RDF serialization handler."""

    def __init__(self, **config):
        """Initialize RDF serializer."""
        self.logger = get_logger("rdf_serializer")
        self.config = config

    def serialize_to_rdf(
        self, triples: List[Triple], format: str = "turtle", **options
    ) -> str:
        """
        Serialize triples to RDF format.

        Args:
            triples: List of triples
            format: RDF format
            **options: Serialization options

        Returns:
            str: Serialized RDF
        """
        if format == "turtle":
            return self._serialize_turtle(triples, **options)
        elif format == "ntriples":
            return self._serialize_ntriples(triples, **options)
        elif format == "jsonld":
            return self._serialize_jsonld(triples, **options)
        elif format == "xml":
            return self._serialize_xml(triples, **options)
        else:
            raise ValidationError(f"Unsupported RDF format: {format}")

    def _serialize_turtle(self, triples: List[Triple], **options) -> str:
        """Serialize to Turtle format."""
        lines = ["@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."]

        for triple in triples:
            line = f"{triple.subject} <{triple.predicate}> {triple.object} ."
            lines.append(line)

        return "\n".join(lines)

    def _serialize_ntriples(self, triples: List[Triple], **options) -> str:
        """Serialize to N-Triples format."""
        lines = []
        for triple in triples:
            line = f"<{triple.subject}> <{triple.predicate}> <{triple.object}> ."
            lines.append(line)
        return "\n".join(lines)

    def _serialize_jsonld(self, triples: List[Triple], **options) -> str:
        """Serialize to JSON-LD format."""
        import json

        graph = []
        for triple in triples:
            graph.append({"@id": triple.subject, triple.predicate: triple.object})

        return json.dumps({"@graph": graph}, indent=2)

    def _serialize_xml(self, triples: List[Triple], **options) -> str:
        """Serialize to RDF/XML format."""
        lines = [
            '<?xml version="1.0"?>',
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">',
        ]

        for triple in triples:
            lines.append(f'  <rdf:Description rdf:about="{triple.subject}">')
            lines.append(
                f"    <{triple.predicate}>{triple.object}</{triple.predicate}>"
            )
            lines.append("  </rdf:Description>")

        lines.append("</rdf:RDF>")
        return "\n".join(lines)


class TripleQualityChecker:
    """Triple quality assessment engine."""

    def __init__(self, **config):
        """Initialize triple quality checker."""
        self.logger = get_logger("triple_quality_checker")
        self.config = config

    def assess_triple_quality(self, triple: Triple, **metrics) -> Dict[str, Any]:
        """
        Assess quality of individual triple.

        Args:
            triple: Triple to assess
            **metrics: Quality metrics

        Returns:
            dict: Quality assessment
        """
        return {
            "confidence": triple.confidence,
            "completeness": 1.0
            if triple.subject and triple.predicate and triple.object
            else 0.0,
            "quality_score": triple.confidence,
        }

    def calculate_quality_scores(
        self, triples: List[Triple], **options
    ) -> Dict[str, Any]:
        """
        Calculate quality scores for triples.

        Args:
            triples: List of triples
            **options: Quality options

        Returns:
            dict: Quality scores
        """
        if not triples:
            return {}

        scores = [self.assess_triple_quality(t)["quality_score"] for t in triples]

        return {
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "high_quality": len([s for s in scores if s >= 0.8]),
            "medium_quality": len([s for s in scores if 0.5 <= s < 0.8]),
            "low_quality": len([s for s in scores if s < 0.5]),
        }
