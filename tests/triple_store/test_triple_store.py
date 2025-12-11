import unittest
from unittest.mock import MagicMock, patch
from semantica.triple_store.triple_manager import TripleManager, TripleStore
from semantica.triple_store.query_engine import QueryEngine, QueryResult
from semantica.semantic_extract.triple_extractor import Triple

class TestTripleStore(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()
        self.mock_tracker = MagicMock()
        
        self.logger_patcher = patch('semantica.triple_store.triple_manager.get_logger', return_value=self.mock_logger)
        self.tracker_patcher = patch('semantica.triple_store.triple_manager.get_progress_tracker', return_value=self.mock_tracker)
        self.logger_patcher_qe = patch('semantica.triple_store.query_engine.get_logger', return_value=self.mock_logger)
        self.tracker_patcher_qe = patch('semantica.triple_store.query_engine.get_progress_tracker', return_value=self.mock_tracker)
        
        self.logger_patcher.start()
        self.tracker_patcher.start()
        self.logger_patcher_qe.start()
        self.tracker_patcher_qe.start()

    def tearDown(self):
        self.logger_patcher.stop()
        self.tracker_patcher.stop()
        self.logger_patcher_qe.stop()
        self.tracker_patcher_qe.stop()

    def test_triple_manager_init(self):
        manager = TripleManager(default_store="main")
        self.assertEqual(manager.default_store_id, "main")
        self.assertEqual(manager.stores, {})

    def test_register_store(self):
        manager = TripleManager()
        store = manager.register_store("main", "blazegraph", "http://localhost:9999")
        self.assertIsInstance(store, TripleStore)
        self.assertEqual(store.store_id, "main")
        self.assertEqual(store.store_type, "blazegraph")
        self.assertEqual(store.endpoint, "http://localhost:9999")
        self.assertIn("main", manager.stores)

    @patch('semantica.triple_store.triple_manager.TripleManager._get_adapter')
    def test_add_triple(self, mock_get_adapter):
        manager = TripleManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.add_triple.return_value = {"status": "success"}
        
        triple = Triple(subject="s", predicate="p", object="o")
        result = manager.add_triple(triple, store_id="main")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["store_id"], "main")
        mock_adapter.add_triple.assert_called_once_with(triple)

    @patch('semantica.triple_store.triple_manager.TripleManager._get_adapter')
    def test_add_triples(self, mock_get_adapter):
        manager = TripleManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.add_triples.return_value = {"status": "success"}
        
        triples = [
            Triple(subject="s1", predicate="p1", object="o1"),
            Triple(subject="s2", predicate="p2", object="o2")
        ]
        
        result = manager.add_triples(triples, store_id="main", batch_size=2)
        
        self.assertTrue(result["success"])
        self.assertEqual(result["total_triples"], 2)
        mock_adapter.add_triples.assert_called()

    @patch('semantica.triple_store.triple_manager.TripleManager._get_adapter')
    def test_get_triple(self, mock_get_adapter):
        manager = TripleManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        expected_triples = [Triple(subject="s", predicate="p", object="o")]
        mock_adapter.get_triples.return_value = expected_triples
        
        result = manager.get_triple(subject="s", store_id="main")
        
        self.assertEqual(result, expected_triples)
        mock_adapter.get_triples.assert_called_once_with("s", None, None)

    @patch('semantica.triple_store.triple_manager.TripleManager._get_adapter')
    def test_delete_triple(self, mock_get_adapter):
        manager = TripleManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.delete_triple.return_value = {"status": "deleted"}
        
        triple = Triple(subject="s", predicate="p", object="o")
        result = manager.delete_triple(triple, store_id="main")
        
        self.assertTrue(result["success"])
        mock_adapter.delete_triple.assert_called_once_with(triple)

    @patch('semantica.triple_store.triple_manager.TripleManager._get_adapter')
    def test_update_triple(self, mock_get_adapter):
        manager = TripleManager()
        manager.register_store("main", "blazegraph", "http://localhost:9999")
        
        mock_adapter = MagicMock()
        mock_get_adapter.return_value = mock_adapter
        mock_adapter.delete_triple.return_value = {"status": "deleted"}
        mock_adapter.add_triple.return_value = {"status": "added"}
        
        old_triple = Triple(subject="s", predicate="p", object="o_old")
        new_triple = Triple(subject="s", predicate="p", object="o_new")
        
        result = manager.update_triple(old_triple, new_triple, store_id="main")
        
        self.assertTrue(result["success"])
        mock_adapter.delete_triple.assert_called_once_with(old_triple)
        mock_adapter.add_triple.assert_called_once_with(new_triple)

    def test_query_engine_init(self):
        engine = QueryEngine(enable_caching=True)
        self.assertTrue(engine.enable_caching)
        self.assertEqual(engine.query_cache, {})

if __name__ == '__main__':
    unittest.main()
