"""
RDF Triple Extraction Module

Handles extraction of RDF triples from text and structured data.

Key Features:
    - RDF triple generation
    - Subject-predicate-object extraction
    - Triple validation and quality checking
    - RDF serialization support
    - Batch triple processing

Main Classes:
    - TripleExtractor: Main triple extraction class
    - TripleValidator: Triple validation engine
    - RDFSerializer: RDF serialization handler
    - TripleQualityChecker: Triple quality assessment
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
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
    
    def __init__(self, config=None, **kwargs):
        """Initialize triple extractor."""
        self.logger = get_logger("triple_extractor")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.triple_validator = TripleValidator(**self.config.get("validator", {}))
        self.rdf_serializer = RDFSerializer(**self.config.get("serializer", {}))
        self.quality_checker = TripleQualityChecker(**self.config.get("quality", {}))
        
        self.supported_formats = ["turtle", "ntriples", "jsonld", "xml"]
    
    def extract_triples(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        relationships: Optional[List[Relation]] = None,
        **options
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
        from .ner_extractor import NERExtractor
        from .relation_extractor import RelationExtractor
        
        # Extract entities if not provided
        if entities is None:
            ner = NERExtractor(**self.config.get("ner", {}))
            entities = ner.extract_entities(text)
        
        # Extract relations if not provided
        if relationships is None:
            rel_extractor = RelationExtractor(**self.config.get("relation", {}))
            relationships = rel_extractor.extract_relations(text, entities)
        
        # Convert relations to triples
        triples = []
        for relation in relationships:
            triple = Triple(
                subject=self._format_uri(relation.subject.text),
                predicate=self._format_uri(relation.predicate),
                object=self._format_uri(relation.object.text),
                confidence=relation.confidence,
                metadata={
                    "context": relation.context,
                    **relation.metadata
                }
            )
            triples.append(triple)
        
        # Validate triples
        if options.get("validate", True):
            triples = self.triple_validator.validate_triples(triples)
        
        return triples
    
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
    
    def serialize_triples(self, triples: List[Triple], format: str = "turtle", **options) -> str:
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
            "consistent": len(issues) == 0
        }


class RDFSerializer:
    """RDF serialization handler."""
    
    def __init__(self, **config):
        """Initialize RDF serializer."""
        self.logger = get_logger("rdf_serializer")
        self.config = config
    
    def serialize_to_rdf(self, triples: List[Triple], format: str = "turtle", **options) -> str:
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
            graph.append({
                "@id": triple.subject,
                triple.predicate: triple.object
            })
        
        return json.dumps({"@graph": graph}, indent=2)
    
    def _serialize_xml(self, triples: List[Triple], **options) -> str:
        """Serialize to RDF/XML format."""
        lines = ['<?xml version="1.0"?>', '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">']
        
        for triple in triples:
            lines.append(f'  <rdf:Description rdf:about="{triple.subject}">')
            lines.append(f'    <{triple.predicate}>{triple.object}</{triple.predicate}>')
            lines.append('  </rdf:Description>')
        
        lines.append('</rdf:RDF>')
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
            "completeness": 1.0 if triple.subject and triple.predicate and triple.object else 0.0,
            "quality_score": triple.confidence
        }
    
    def calculate_quality_scores(self, triples: List[Triple], **options) -> Dict[str, Any]:
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
            "low_quality": len([s for s in scores if s < 0.5])
        }
