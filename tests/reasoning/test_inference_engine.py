import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from semantica.reasoning.inference_engine import InferenceEngine, InferenceStrategy
from semantica.reasoning.rule_manager import Rule, RuleType

class TestInferenceEngine(unittest.TestCase):
    def setUp(self):
        self.mock_tracker_patcher = patch("semantica.utils.progress_tracker.get_progress_tracker")
        self.mock_get_tracker = self.mock_tracker_patcher.start()
        self.mock_tracker = MagicMock()
        self.mock_get_tracker.return_value = self.mock_tracker

    def tearDown(self):
        self.mock_tracker_patcher.stop()

    def test_initialization(self):
        engine = InferenceEngine()
        self.assertEqual(engine.strategy, InferenceStrategy.FORWARD)
        self.assertEqual(len(engine.facts), 0)
        self.assertEqual(len(engine.unhashable_facts), 0)

    def test_add_hashable_facts(self):
        engine = InferenceEngine()
        engine.add_fact("fact1")
        engine.add_fact(("fact", "2"))
        
        self.assertEqual(len(engine.facts), 2)
        self.assertIn("fact1", engine.facts)
        self.assertEqual(len(engine.unhashable_facts), 0)

    def test_add_unhashable_facts(self):
        engine = InferenceEngine()
        # Dict is unhashable
        fact1 = {"subject": "s", "predicate": "p", "object": "o"}
        fact2 = ["list", "is", "unhashable"]
        
        engine.add_fact(fact1)
        engine.add_fact(fact2)
        
        self.assertEqual(len(engine.facts), 0)
        self.assertEqual(len(engine.unhashable_facts), 2)
        self.assertIn(fact1, engine.unhashable_facts)

    def test_mixed_facts_retrieval(self):
        engine = InferenceEngine()
        engine.add_fact("hashable")
        engine.add_fact({"unhashable": True})
        
        facts = engine.get_facts()
        self.assertEqual(len(facts), 2)
        self.assertIn("hashable", facts)
        self.assertIn({"unhashable": True}, facts)

    def test_rule_execution_hashable(self):
        engine = InferenceEngine()
        engine.add_fact("A")
        
        # Rule: IF A THEN B
        engine.add_rule("IF A THEN B")
        
        results = engine.infer(None, strategy=InferenceStrategy.FORWARD)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].conclusion, "B")
        self.assertIn("B", engine.facts)

    def test_rule_execution_unhashable(self):
        engine = InferenceEngine()
        fact_a = {"id": "A"}
        engine.add_fact(fact_a)
        
        # Rule that depends on unhashable fact
        # Note: The simple string parser in RuleManager might not handle dict string representation perfectly
        # So we construct Rule object manually for this test to avoid parsing issues
        
        rule = Rule(
            rule_id="r1",
            name="Test Rule",
            conditions=[fact_a],
            conclusion="B",
            rule_type=RuleType.IMPLICATION
        )
        engine.rule_manager.add_rule(rule)
        
        results = engine.infer(None, strategy=InferenceStrategy.FORWARD)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].conclusion, "B")

    def test_backward_chaining_unhashable(self):
        engine = InferenceEngine(strategy="backward")
        fact_a = {"id": "A"}
        engine.add_fact(fact_a)
        
        # Goal is the unhashable fact itself
        result = engine.infer(fact_a)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].conclusion, fact_a)
        self.assertEqual(result[0].confidence, 1.0)

if __name__ == "__main__":
    unittest.main()
