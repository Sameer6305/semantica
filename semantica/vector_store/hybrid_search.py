"""
Hybrid search for Semantica framework.

This module provides combined vector and metadata search
capabilities for enhanced retrieval.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


class MetadataFilter:
    """Metadata filter builder."""
    
    def __init__(self):
        """Initialize metadata filter."""
        self.conditions: List[Dict[str, Any]] = []
    
    def add_condition(
        self,
        field: str,
        operator: str,
        value: Any
    ) -> "MetadataFilter":
        """
        Add filter condition.
        
        Args:
            field: Field name
            operator: Operator ("eq", "ne", "gt", "gte", "lt", "lte", "in", "contains")
            value: Value to filter by
            
        Returns:
            Self for chaining
        """
        self.conditions.append({
            "field": field,
            "operator": operator,
            "value": value
        })
        return self
    
    def eq(self, field: str, value: Any) -> "MetadataFilter":
        """Add equality condition."""
        return self.add_condition(field, "eq", value)
    
    def ne(self, field: str, value: Any) -> "MetadataFilter":
        """Add not-equal condition."""
        return self.add_condition(field, "ne", value)
    
    def gt(self, field: str, value: Any) -> "MetadataFilter":
        """Add greater-than condition."""
        return self.add_condition(field, "gt", value)
    
    def gte(self, field: str, value: Any) -> "MetadataFilter":
        """Add greater-than-or-equal condition."""
        return self.add_condition(field, "gte", value)
    
    def lt(self, field: str, value: Any) -> "MetadataFilter":
        """Add less-than condition."""
        return self.add_condition(field, "lt", value)
    
    def lte(self, field: str, value: Any) -> "MetadataFilter":
        """Add less-than-or-equal condition."""
        return self.add_condition(field, "lte", value)
    
    def contains(self, field: str, value: Any) -> "MetadataFilter":
        """Add contains condition."""
        return self.add_condition(field, "contains", value)
    
    def in_list(self, field: str, values: List[Any]) -> "MetadataFilter":
        """Add in-list condition."""
        return self.add_condition(field, "in", values)
    
    def matches(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches all conditions."""
        for condition in self.conditions:
            field = condition["field"]
            operator = condition["operator"]
            value = condition["value"]
            
            if field not in metadata:
                return False
            
            field_value = metadata[field]
            
            if operator == "eq" and field_value != value:
                return False
            elif operator == "ne" and field_value == value:
                return False
            elif operator == "gt" and not (field_value > value):
                return False
            elif operator == "gte" and not (field_value >= value):
                return False
            elif operator == "lt" and not (field_value < value):
                return False
            elif operator == "lte" and not (field_value <= value):
                return False
            elif operator == "contains":
                if isinstance(field_value, str) and isinstance(value, str):
                    if value not in field_value:
                        return False
                elif isinstance(field_value, list):
                    if value not in field_value:
                        return False
                else:
                    return False
            elif operator == "in" and field_value not in value:
                return False
        
        return True


class SearchRanker:
    """Search result ranker."""
    
    def __init__(self, strategy: str = "reciprocal_rank_fusion"):
        """Initialize search ranker."""
        self.strategy = strategy
        self.logger = get_logger("search_ranker")
    
    def reciprocal_rank_fusion(
        self,
        results: List[List[Dict[str, Any]]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        
        Args:
            results: List of result lists from different sources
            k: RRF constant
            
        Returns:
            Fused and ranked results
        """
        scores: Dict[str, float] = {}
        
        for result_list in results:
            for rank, result in enumerate(result_list, start=1):
                result_id = result.get("id", str(id(result)))
                score = 1.0 / (k + rank)
                scores[result_id] = scores.get(result_id, 0.0) + score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Reconstruct results
        fused_results = []
        result_map = {}
        for result_list in results:
            for result in result_list:
                result_id = result.get("id", str(id(result)))
                result_map[result_id] = result
        
        for result_id, score in ranked:
            if result_id in result_map:
                result = result_map[result_id].copy()
                result["score"] = score
                fused_results.append(result)
        
        return fused_results
    
    def weighted_average(
        self,
        results: List[List[Dict[str, Any]]],
        weights: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Weighted average fusion.
        
        Args:
            results: List of result lists
            weights: Weights for each result list
            
        Returns:
            Fused results
        """
        if len(weights) != len(results):
            weights = [1.0 / len(results)] * len(results)
        
        scores: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        
        for weight, result_list in zip(weights, results):
            for result in result_list:
                result_id = result.get("id", str(id(result)))
                score = result.get("score", 0.0) * weight
                
                if result_id not in scores:
                    scores[result_id] = (0.0, result)
                
                scores[result_id] = (
                    scores[result_id][0] + score,
                    scores[result_id][1]
                )
        
        # Sort by score
        ranked = sorted(scores.values(), key=lambda x: x[0], reverse=True)
        
        fused_results = []
        for score, result in ranked:
            result_copy = result.copy()
            result_copy["score"] = score
            fused_results.append(result_copy)
        
        return fused_results
    
    def rank(
        self,
        results: List[List[Dict[str, Any]]],
        **options
    ) -> List[Dict[str, Any]]:
        """
        Rank and fuse results.
        
        Args:
            results: List of result lists
            **options: Ranking options
            
        Returns:
            Fused and ranked results
        """
        if self.strategy == "reciprocal_rank_fusion":
            k = options.get("k", 60)
            return self.reciprocal_rank_fusion(results, k)
        elif self.strategy == "weighted_average":
            weights = options.get("weights", [1.0 / len(results)] * len(results))
            return self.weighted_average(results, weights)
        else:
            return self.reciprocal_rank_fusion(results)


class HybridSearch:
    """
    Hybrid search combining vector similarity and metadata filtering.
    
    • Vector similarity search
    • Metadata filtering and querying
    • Result fusion and ranking
    • Performance optimization
    • Error handling and recovery
    • Advanced search strategies
    """
    
    def __init__(self, **config):
        """Initialize hybrid search."""
        self.logger = get_logger("hybrid_search")
        self.config = config
        self.ranker = SearchRanker(config.get("ranking_strategy", "reciprocal_rank_fusion"))
    
    def search(
        self,
        query_vector: np.ndarray,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        vector_ids: List[str],
        k: int = 10,
        metadata_filter: Optional[MetadataFilter] = None,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search.
        
        Args:
            query_vector: Query vector
            vectors: List of vectors to search
            metadata: List of metadata dictionaries
            vector_ids: Vector IDs
            k: Number of results
            metadata_filter: Optional metadata filter
            **options: Additional options
            
        Returns:
            List of search results
        """
        if not vectors or not metadata:
            return []
        
        # Filter by metadata first
        if metadata_filter:
            filtered_indices = [
                i for i, meta in enumerate(metadata)
                if metadata_filter.matches(meta)
            ]
            
            filtered_vectors = [vectors[i] for i in filtered_indices]
            filtered_metadata = [metadata[i] for i in filtered_indices]
            filtered_ids = [vector_ids[i] for i in filtered_indices]
        else:
            filtered_vectors = vectors
            filtered_metadata = metadata
            filtered_ids = vector_ids
        
        if not filtered_vectors:
            return []
        
        # Perform vector similarity search
        vector_results = self._vector_search(
            query_vector,
            filtered_vectors,
            filtered_ids,
            k * 2,  # Get more results for ranking
            **options
        )
        
        # Add metadata to results
        for result in vector_results:
            result_id = result.get("id")
            if result_id in filtered_ids:
                idx = filtered_ids.index(result_id)
                result["metadata"] = filtered_metadata[idx]
        
        # Rank and return top k
        return vector_results[:k]
    
    def _vector_search(
        self,
        query_vector: np.ndarray,
        vectors: List[np.ndarray],
        vector_ids: List[str],
        k: int,
        **options
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        if not vectors:
            return []
        
        # Convert to numpy
        if isinstance(vectors[0], list):
            vectors = np.array(vectors)
        else:
            vectors = np.vstack(vectors)
        
        if isinstance(query_vector, list):
            query_vector = np.array(query_vector)
        
        # Calculate cosine similarity
        query_norm = np.linalg.norm(query_vector)
        vector_norms = np.linalg.norm(vectors, axis=1)
        
        similarities = np.dot(vectors, query_vector) / (vector_norms * query_norm + 1e-8)
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "id": vector_ids[idx],
                "score": float(similarities[idx]),
                "distance": 1.0 - float(similarities[idx])
            })
        
        return results
    
    def multi_source_search(
        self,
        query_vector: np.ndarray,
        sources: List[Dict[str, Any]],
        k: int = 10,
        **options
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple sources and fuse results.
        
        Args:
            query_vector: Query vector
            sources: List of source dictionaries with 'vectors', 'metadata', 'ids'
            k: Number of results
            **options: Additional options
            
        Returns:
            Fused search results
        """
        all_results = []
        
        for source in sources:
            source_results = self.search(
                query_vector,
                source.get("vectors", []),
                source.get("metadata", []),
                source.get("ids", []),
                k=k,
                metadata_filter=source.get("filter"),
                **options
            )
            all_results.append(source_results)
        
        # Fuse results using ranker
        fused_results = self.ranker.rank(all_results, **options)
        
        return fused_results[:k]
    
    def filter_by_metadata(
        self,
        results: List[Dict[str, Any]],
        metadata_filter: MetadataFilter
    ) -> List[Dict[str, Any]]:
        """
        Filter results by metadata.
        
        Args:
            results: Search results
            metadata_filter: Metadata filter
            
        Returns:
            Filtered results
        """
        filtered = []
        for result in results:
            metadata = result.get("metadata", {})
            if metadata_filter.matches(metadata):
                filtered.append(result)
        
        return filtered
