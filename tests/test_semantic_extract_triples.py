
import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantica.semantic_extract.triple_extractor import (
    TripleExtractor, Triple, TripleValidator, RDFSerializer, TripleQualityChecker
)
from semantica.semantic_extract.ner_extractor import Entity
from semantica.semantic_extract.relation_extractor import Relation

class TestSemanticExtractTriples(unittest.TestCase):

    def setUp(self):
        self.entities = [
            Entity(text="Steve Jobs", label="PERSON", start_char=0, end_char=10),
            Entity(text="Apple", label="ORG", start_char=19, end_char=24)
        ]
        self.relations = [
            Relation(
                subject=self.entities[0],
                predicate="founded",
                object=self.entities[1],
                confidence=0.9,
                context="Steve Jobs founded Apple."
            )
        ]
        self.triples = [
            Triple(subject="Steve_Jobs", predicate="founded", object="Apple", confidence=0.9),
            Triple(subject="Apple", predicate="located_in", object="Cupertino", confidence=0.8)
        ]

    # --- Triple Extractor Tests ---

    def test_triple_extractor_init(self):
        """Test TripleExtractor initialization"""
        extractor = TripleExtractor()
        self.assertIsNotNone(extractor.triple_validator)
        self.assertIsNotNone(extractor.rdf_serializer)
        self.assertIsNotNone(extractor.quality_checker)

    def test_triple_extractor_extract_from_relations(self):
        """Test extracting triples by converting relations (fallback/default)"""
        extractor = TripleExtractor(method=[]) # No specific method, force fallback
        
        # Mocking progress tracker to avoid console clutter/errors
        extractor.progress_tracker = MagicMock()
        
        triples = extractor.extract_triples(
            text="Steve Jobs founded Apple.",
            entities=self.entities,
            relationships=self.relations
        )
        
        self.assertEqual(len(triples), 1)
        # Predicate is formatted as URI
        self.assertTrue(triples[0].predicate.endswith("founded") or triples[0].predicate == "founded")
        # Check URI formatting (simple implementation in _format_uri)
        # "Steve Jobs" -> "Steve_Jobs", prepended with http://example.org/ if not http
        self.assertIn("Steve_Jobs", triples[0].subject)

    # --- Triple Validator Tests ---

    def test_triple_validator_valid(self):
        """Test TripleValidator with valid triple"""
        validator = TripleValidator()
        triple = Triple(subject="S", predicate="P", object="O", confidence=0.9)
        self.assertTrue(validator.validate_triple(triple))

    def test_triple_validator_invalid_structure(self):
        """Test TripleValidator with missing fields"""
        validator = TripleValidator()
        triple = Triple(subject="", predicate="P", object="O") # Empty subject
        self.assertFalse(validator.validate_triple(triple))

    def test_triple_validator_low_confidence(self):
        """Test TripleValidator confidence threshold"""
        validator = TripleValidator()
        triple = Triple(subject="S", predicate="P", object="O", confidence=0.4)
        self.assertFalse(validator.validate_triple(triple, min_confidence=0.5))

    # --- RDF Serializer Tests ---

    def test_rdf_serializer_turtle(self):
        """Test RDF serialization to Turtle"""
        serializer = RDFSerializer()
        output = serializer.serialize_to_rdf(self.triples, format="turtle")
        self.assertIn("@prefix", output)
        self.assertIn("Steve_Jobs", output)
        self.assertIn("founded", output)
        self.assertIn("Apple", output)
        self.assertTrue(output.strip().endswith("."))

    def test_rdf_serializer_ntriples(self):
        """Test RDF serialization to N-Triples"""
        serializer = RDFSerializer()
        output = serializer.serialize_to_rdf(self.triples, format="ntriples")
        self.assertNotIn("@prefix", output)
        self.assertIn("<Steve_Jobs>", output)
        self.assertIn("<founded>", output)

    def test_rdf_serializer_jsonld(self):
        """Test RDF serialization to JSON-LD"""
        serializer = RDFSerializer()
        output = serializer.serialize_to_rdf(self.triples, format="jsonld")
        import json
        data = json.loads(output)
        self.assertIn("@graph", data)
        self.assertEqual(len(data["@graph"]), 2)

    def test_rdf_serializer_xml(self):
        """Test RDF serialization to XML"""
        serializer = RDFSerializer()
        output = serializer.serialize_to_rdf(self.triples, format="xml")
        self.assertIn("rdf:RDF", output)
        self.assertIn("rdf:Description", output)

    # --- Triple Quality Checker Tests ---

    def test_triple_quality_checker_assess(self):
        """Test TripleQualityChecker assessment"""
        checker = TripleQualityChecker()
        triple = Triple(subject="S", predicate="P", object="O", confidence=0.85)
        assessment = checker.assess_triple_quality(triple)
        
        self.assertEqual(assessment["confidence"], 0.85)
        self.assertEqual(assessment["completeness"], 1.0)
        self.assertEqual(assessment["quality_score"], 0.85)

    def test_triple_quality_checker_stats(self):
        """Test TripleQualityChecker statistics"""
        checker = TripleQualityChecker()
        stats = checker.calculate_quality_scores(self.triples)
        
        # Implementation returns average_score, min_score, max_score, high_quality, medium_quality, low_quality
        self.assertIn("average_score", stats)
        self.assertIn("high_quality", stats) # 0.9 and 0.8 are >= 0.8
        self.assertEqual(stats["high_quality"], 2)

if __name__ == '__main__':
    unittest.main()
