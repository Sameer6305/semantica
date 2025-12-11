import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from semantica.semantic_extract.ner_extractor import NERExtractor
from semantica.semantic_extract.relation_extractor import RelationExtractor
from semantica.semantic_extract.triple_extractor import TripleExtractor
from semantica.semantic_extract.named_entity_recognizer import Entity
from semantica.semantic_extract.relation_extractor import Relation

class TestExtractors(unittest.TestCase):
    
    def test_ner_extractor_initialization(self):
        """Test NERExtractor initialization and circular import resolution"""
        try:
            extractor = NERExtractor(method="pattern")
            self.assertIsNotNone(extractor)
        except ImportError as e:
            self.fail(f"NERExtractor initialization failed with ImportError: {e}")

    def test_relation_extractor_initialization(self):
        """Test RelationExtractor initialization and circular import resolution"""
        try:
            extractor = RelationExtractor(method="pattern")
            self.assertIsNotNone(extractor)
        except ImportError as e:
            self.fail(f"RelationExtractor initialization failed with ImportError: {e}")

    def test_triple_extractor_initialization(self):
        """Test TripleExtractor initialization and circular import resolution"""
        try:
            extractor = TripleExtractor(method="pattern")
            self.assertIsNotNone(extractor)
        except ImportError as e:
            self.fail(f"TripleExtractor initialization failed with ImportError: {e}")

    @patch("semantica.semantic_extract.methods.get_entity_method")
    def test_ner_extraction(self, mock_get_method):
        """Test NER extraction call"""
        mock_method = MagicMock()
        mock_method.extract_entities.return_value = []
        mock_get_method.return_value = mock_method
        
        extractor = NERExtractor(method="pattern")
        entities = extractor.extract_entities("Test text")
        
        self.assertIsInstance(entities, list)
        mock_get_method.assert_called()

    @patch("semantica.semantic_extract.methods.get_relation_method")
    def test_relation_extraction(self, mock_get_method):
        """Test relation extraction call"""
        mock_method = MagicMock()
        mock_method.extract_relations.return_value = []
        mock_get_method.return_value = mock_method
        
        extractor = RelationExtractor(method="pattern")
        entities = [Entity(text="A", label="PERSON", start_char=0, end_char=1), Entity(text="B", label="PERSON", start_char=5, end_char=6)]
        relations = extractor.extract_relations("A knows B", entities)
        
        self.assertIsInstance(relations, list)
        mock_get_method.assert_called()

    @patch("semantica.semantic_extract.methods.get_triple_method")
    def test_triple_extraction(self, mock_get_method):
        """Test triple extraction call"""
        mock_method = MagicMock()
        mock_method.extract_triples.return_value = []
        mock_get_method.return_value = mock_method
        
        extractor = TripleExtractor(method="pattern")
        entities = [Entity(text="A", label="PERSON", start_char=0, end_char=1)]
        relations = [Relation(subject=entities[0], object=entities[0], predicate="knows")]
        
        triples = extractor.extract_triples("A knows A", entities, relations)
        
        self.assertIsInstance(triples, list)
        mock_get_method.assert_called()

if __name__ == "__main__":
    unittest.main()
