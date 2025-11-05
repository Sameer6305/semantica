"""
Apache Jena adapter for Semantica framework.

This module provides Apache Jena integration for RDF storage
and SPARQL querying.
"""

from typing import Any, Dict, List, Optional

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..semantic_extract.triple_extractor import Triple

# Optional Jena imports
try:
    from rdflib import Graph, Namespace, URIRef, Literal, RDF
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
    HAS_JENA_RDFLIB = True
except ImportError:
    HAS_JENA_RDFLIB = False
    Graph = None
    RDF = None


class JenaAdapter:
    """
    Apache Jena adapter for triple store operations.
    
    • Jena connection and configuration
    • SPARQL query execution
    • Model and dataset management
    • Inference and reasoning support
    • Performance optimization
    • Error handling and recovery
    """
    
    def __init__(self, **config):
        """
        Initialize Jena adapter.
        
        Args:
            **config: Configuration options:
                - endpoint: Jena Fuseki endpoint (optional)
                - dataset: Dataset name
                - enable_inference: Enable inference (default: False)
        """
        self.logger = get_logger("jena_adapter")
        self.config = config
        
        self.endpoint = config.get("endpoint")
        self.dataset = config.get("dataset", "default")
        self.enable_inference = config.get("enable_inference", False)
        
        self.graph: Optional[Graph] = None
        self._initialize_graph()
    
    def _initialize_graph(self) -> None:
        """Initialize RDF graph."""
        if HAS_JENA_RDFLIB:
            if self.endpoint:
                # Use SPARQL store for remote endpoint
                try:
                    store = SPARQLStore(query_endpoint=self.endpoint)
                    self.graph = Graph(store=store)
                except Exception as e:
                    self.logger.warning(f"Could not initialize SPARQL store: {e}")
                    self.graph = Graph()
            else:
                # Use in-memory graph
                self.graph = Graph()
        else:
            self.logger.warning("rdflib not available. Jena adapter will use basic operations.")
            self.graph = None
    
    def create_model(self, **options) -> Dict[str, Any]:
        """
        Create and manage RDF models.
        
        Args:
            **options: Model options
        
        Returns:
            Model information
        """
        if self.graph is None:
            self._initialize_graph()
        
        return {
            "model_id": self.dataset,
            "endpoint": self.endpoint,
            "triple_count": len(self.graph) if self.graph else 0
        }
    
    def add_triples(
        self,
        triples: List[Triple],
        **options
    ) -> Dict[str, Any]:
        """
        Add triples to model.
        
        Args:
            triples: List of triples
            **options: Additional options
        
        Returns:
            Operation status
        """
        if not self.graph:
            raise ProcessingError("Graph not initialized")
        
        try:
            added_count = 0
            for triple in triples:
                try:
                    subject = URIRef(triple.subject)
                    predicate = URIRef(triple.predicate)
                    obj = URIRef(triple.object) if triple.object.startswith("http") else Literal(triple.object)
                    
                    self.graph.add((subject, predicate, obj))
                    added_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to add triple: {e}")
            
            return {
                "success": True,
                "added": added_count,
                "total": len(triples)
            }
        except Exception as e:
            self.logger.error(f"Failed to add triples: {e}")
            raise ProcessingError(f"Failed to add triples: {e}")
    
    def add_triple(self, triple: Triple, **options) -> Dict[str, Any]:
        """Add single triple."""
        return self.add_triples([triple], **options)
    
    def get_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        **options
    ) -> List[Triple]:
        """Get triples matching criteria."""
        if not self.graph:
            return []
        
        try:
            # Build SPARQL query
            query_parts = []
            if subject:
                query_parts.append(f"?s = <{subject}>")
            if predicate:
                query_parts.append(f"?p = <{predicate}>")
            if object:
                query_parts.append(f"?o = <{object}>")
            
            where_clause = " ".join(query_parts) if query_parts else ""
            query = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o {where_clause} }}"
            
            results = self.graph.query(query)
            
            triples = []
            for row in results:
                triples.append(Triple(
                    subject=str(row.s),
                    predicate=str(row.p),
                    object=str(row.o),
                    metadata={"source": "jena"}
                ))
            
            return triples
        except Exception as e:
            self.logger.error(f"Failed to get triples: {e}")
            return []
    
    def delete_triple(self, triple: Triple, **options) -> Dict[str, Any]:
        """Delete triple."""
        if not self.graph:
            raise ProcessingError("Graph not initialized")
        
        try:
            subject = URIRef(triple.subject)
            predicate = URIRef(triple.predicate)
            obj = URIRef(triple.object) if triple.object.startswith("http") else Literal(triple.object)
            
            self.graph.remove((subject, predicate, obj))
            
            return {"success": True}
        except Exception as e:
            self.logger.error(f"Failed to delete triple: {e}")
            raise ProcessingError(f"Failed to delete triple: {e}")
    
    def run_inference(
        self,
        model: Optional[Any] = None,
        **options
    ) -> Dict[str, Any]:
        """
        Execute inference rules.
        
        Args:
            model: Optional model (uses default if not provided)
            **options: Inference options
        
        Returns:
            Inference results
        """
        if not self.enable_inference:
            self.logger.warning("Inference not enabled")
            return {"success": False, "message": "Inference not enabled"}
        
        # Basic inference would require OWL reasoner
        # This is a placeholder implementation
        self.logger.info("Inference would be executed here with OWL reasoner")
        
        return {
            "success": True,
            "inferred_triples": 0,
            "message": "Inference placeholder"
        }
    
    def execute_sparql(
        self,
        query: str,
        **options
    ) -> Dict[str, Any]:
        """
        Execute SPARQL query.
        
        Args:
            query: SPARQL query string
            **options: Additional options
        
        Returns:
            Query results
        """
        if not self.graph:
            raise ProcessingError("Graph not initialized")
        
        try:
            results = self.graph.query(query)
            
            bindings = []
            variables = []
            
            if results.vars:
                variables = [str(v) for v in results.vars]
                
                for row in results:
                    binding = {}
                    for var in results.vars:
                        value = getattr(row, str(var))
                        if value:
                            binding[str(var)] = {"value": str(value), "type": "uri" if isinstance(value, URIRef) else "literal"}
                    bindings.append(binding)
            
            return {
                "success": True,
                "bindings": bindings,
                "variables": variables,
                "metadata": {"query": query}
            }
        except Exception as e:
            self.logger.error(f"SPARQL query failed: {e}")
            raise ProcessingError(f"SPARQL query failed: {e}")
    
    def serialize(self, format: str = "turtle", **options) -> str:
        """
        Serialize graph to RDF format.
        
        Args:
            format: RDF format (turtle, rdfxml, n3)
            **options: Serialization options
        
        Returns:
            Serialized RDF string
        """
        if not self.graph:
            return ""
        
        try:
            return self.graph.serialize(format=format)
        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            return ""
