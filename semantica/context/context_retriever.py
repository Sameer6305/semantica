"""
Context Retriever for Agents

Retrieves relevant context for agents from memory, knowledge graphs,
and vector stores to inform decision-making.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


@dataclass
class RetrievedContext:
    """Retrieved context item."""
    content: str
    score: float
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    related_relationships: List[Dict[str, Any]] = field(default_factory=list)


class ContextRetriever:
    """
    Context retriever for hybrid retrieval.
    
    • Retrieve context from multiple sources
    • Hybrid retrieval (vector + graph)
    • Context relevance ranking
    • Context aggregation and synthesis
    • Ontology-aware context retrieval
    • Real-time context updates
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize context retriever.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - memory_store: Memory store instance
                - knowledge_graph: Knowledge graph instance
                - vector_store: Vector store instance
                - use_graph_expansion: Use graph expansion (default: True)
                - max_expansion_hops: Maximum graph expansion hops (default: 2)
                - hybrid_alpha: Weight for hybrid retrieval (0=vector only, 1=graph only, default: 0.5)
        """
        self.logger = get_logger("context_retriever")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.memory_store = self.config.get("memory_store")
        self.knowledge_graph = self.config.get("knowledge_graph")
        self.vector_store = self.config.get("vector_store")
        
        self.use_graph_expansion = self.config.get("use_graph_expansion", True)
        self.max_expansion_hops = self.config.get("max_expansion_hops", 2)
        self.hybrid_alpha = self.config.get("hybrid_alpha", 0.5)
    
    def retrieve(
        self,
        query: str,
        max_results: int = 5,
        use_graph_expansion: Optional[bool] = None,
        min_relevance_score: float = 0.0,
        **options
    ) -> List[RetrievedContext]:
        """
        Retrieve relevant context for query.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            use_graph_expansion: Use graph expansion (overrides config)
            min_relevance_score: Minimum relevance score
            **options: Additional options:
                - entity_ids: Filter by entity IDs
                - node_types: Filter by node types
                - max_hops: Maximum expansion hops
                
        Returns:
            List of retrieved context items
        """
        use_expansion = use_graph_expansion if use_graph_expansion is not None else self.use_graph_expansion
        
        all_results = []
        
        # Vector-based retrieval
        vector_results = self._retrieve_from_vector(query, max_results * 2)
        all_results.extend(vector_results)
        
        # Graph-based retrieval
        if self.knowledge_graph and use_expansion:
            graph_results = self._retrieve_from_graph(
                query,
                max_results * 2,
                max_hops=options.get("max_hops", self.max_expansion_hops)
            )
            all_results.extend(graph_results)
        
        # Memory-based retrieval
        if self.memory_store:
            memory_results = self._retrieve_from_memory(query, max_results * 2)
            all_results.extend(memory_results)
        
        # Combine and rank results
        ranked_results = self._rank_and_merge(all_results, query)
        
        # Filter by minimum score
        filtered_results = [
            r for r in ranked_results
            if r.score >= min_relevance_score
        ]
        
        # Return top results
        return filtered_results[:max_results]
    
    def _retrieve_from_vector(
        self,
        query: str,
        max_results: int
    ) -> List[RetrievedContext]:
        """Retrieve from vector store."""
        if not self.vector_store:
            return []
        
        try:
            # Simulate vector search (actual implementation would use vector_store)
            results = []
            
            # If vector_store has a search method
            if hasattr(self.vector_store, "search"):
                search_results = self.vector_store.search(
                    query=query,
                    top_k=max_results
                )
                
                for result in search_results:
                    results.append(RetrievedContext(
                        content=result.get("content", ""),
                        score=result.get("score", 0.0),
                        source=result.get("source"),
                        metadata=result.get("metadata", {})
                    ))
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Vector retrieval failed: {e}")
            return []
    
    def _retrieve_from_graph(
        self,
        query: str,
        max_results: int,
        max_hops: int = 2
    ) -> List[RetrievedContext]:
        """Retrieve from knowledge graph."""
        if not self.knowledge_graph:
            return []
        
        results = []
        
        try:
            # Extract entities from query (simplified)
            query_lower = query.lower()
            
            # Search for nodes matching query
            nodes = self.knowledge_graph.get("nodes", [])
            
            for node in nodes:
                node_content = node.get("content", "").lower()
                node_type = node.get("type", "")
                
                # Simple keyword matching
                if any(word in node_content for word in query_lower.split()):
                    score = self._calculate_graph_relevance(node, query)
                    
                    # Get related entities
                    related_entities = self._get_related_entities(
                        node.get("id"),
                        max_hops=max_hops
                    )
                    
                    results.append(RetrievedContext(
                        content=node.get("content", ""),
                        score=score,
                        source=f"graph:{node.get('id')}",
                        metadata={
                            "node_type": node_type,
                            "node_id": node.get("id"),
                            **node.get("metadata", {})
                        },
                        related_entities=related_entities
                    ))
            
            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results[:max_results]
            
        except Exception as e:
            self.logger.warning(f"Graph retrieval failed: {e}")
            return []
    
    def _retrieve_from_memory(
        self,
        query: str,
        max_results: int
    ) -> List[RetrievedContext]:
        """Retrieve from memory store."""
        if not self.memory_store:
            return []
        
        try:
            # If memory_store has a retrieve method
            if hasattr(self.memory_store, "retrieve"):
                memory_results = self.memory_store.retrieve(
                    query=query,
                    max_results=max_results
                )
                
                results = []
                for result in memory_results:
                    results.append(RetrievedContext(
                        content=result.get("content", ""),
                        score=result.get("score", 0.0),
                        source=result.get("source"),
                        metadata=result.get("metadata", {})
                    ))
                
                return results
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Memory retrieval failed: {e}")
            return []
    
    def _rank_and_merge(
        self,
        results: List[RetrievedContext],
        query: str
    ) -> List[RetrievedContext]:
        """Rank and merge results from multiple sources."""
        # Deduplicate by content
        seen_content = {}
        
        for result in results:
            content_key = result.content[:100]  # First 100 chars as key
            
            if content_key not in seen_content:
                seen_content[content_key] = result
            else:
                # Merge scores (take maximum or average)
                existing = seen_content[content_key]
                existing.score = max(existing.score, result.score)
                
                # Merge metadata
                existing.metadata.update(result.metadata)
                
                # Merge related entities
                existing_entity_ids = {e.get("id") for e in existing.related_entities}
                for entity in result.related_entities:
                    if entity.get("id") not in existing_entity_ids:
                        existing.related_entities.append(entity)
        
        # Sort by score
        ranked = sorted(seen_content.values(), key=lambda x: x.score, reverse=True)
        
        return ranked
    
    def _calculate_graph_relevance(self, node: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for graph node."""
        content = node.get("content", "").lower()
        query_lower = query.lower()
        
        # Simple word overlap
        query_words = set(query_lower.split())
        content_words = set(content.split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        return overlap / len(query_words)
    
    def _get_related_entities(
        self,
        node_id: str,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Get related entities from graph."""
        if not self.knowledge_graph:
            return []
        
        related = []
        visited = set()
        current_level = {node_id}
        
        for hop in range(max_hops):
            next_level = set()
            
            for current_id in current_level:
                if current_id in visited:
                    continue
                visited.add(current_id)
                
                # Find edges
                edges = self.knowledge_graph.get("edges", [])
                for edge in edges:
                    if edge.get("source") == current_id:
                        target_id = edge.get("target")
                        if target_id not in visited:
                            next_level.add(target_id)
                            
                            # Get node info
                            nodes = self.knowledge_graph.get("nodes", [])
                            for node in nodes:
                                if node.get("id") == target_id:
                                    related.append({
                                        "id": target_id,
                                        "type": node.get("type"),
                                        "content": node.get("content"),
                                        "relationship": edge.get("type"),
                                        "hop": hop + 1
                                    })
                    
                    elif edge.get("target") == current_id:
                        source_id = edge.get("source")
                        if source_id not in visited:
                            next_level.add(source_id)
                            
                            nodes = self.knowledge_graph.get("nodes", [])
                            for node in nodes:
                                if node.get("id") == source_id:
                                    related.append({
                                        "id": source_id,
                                        "type": node.get("type"),
                                        "content": node.get("content"),
                                        "relationship": edge.get("type"),
                                        "hop": hop + 1
                                    })
            
            current_level = next_level
        
        return related
