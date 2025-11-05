"""
Cluster Builder for Deduplication

Builds clusters of similar entities for batch deduplication
using clustering algorithms and similarity graphs.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .similarity_calculator import SimilarityCalculator


@dataclass
class Cluster:
    """Entity cluster representation."""
    
    cluster_id: str
    entities: List[Dict[str, Any]]
    centroid: Optional[Dict[str, Any]] = None
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterResult:
    """Cluster building result."""
    
    clusters: List[Cluster]
    unclustered: List[Dict[str, Any]] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClusterBuilder:
    """
    Cluster building engine.
    
    • Builds entity clusters using similarity graphs
    • Supports cluster-based deduplication workflows
    • Assesses cluster quality
    • Uses hierarchical clustering for large datasets
    • Supports incremental cluster updates
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize cluster builder."""
        self.logger = get_logger("cluster_builder")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.similarity_calculator = SimilarityCalculator(**config.get("similarity", {}))
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.min_cluster_size = config.get("min_cluster_size", 2)
        self.max_cluster_size = config.get("max_cluster_size", 100)
        self.use_hierarchical = config.get("use_hierarchical", False)
    
    def build_clusters(
        self,
        entities: List[Dict[str, Any]],
        **options
    ) -> ClusterResult:
        """
        Build clusters of similar entities.
        
        Args:
            entities: List of entities to cluster
            **options: Clustering options
            
        Returns:
            ClusterResult with clusters and metrics
        """
        threshold = options.get("threshold", self.similarity_threshold)
        
        if self.use_hierarchical:
            clusters = self._hierarchical_clustering(entities, threshold)
        else:
            clusters = self._graph_based_clustering(entities, threshold)
        
        # Filter clusters by size
        valid_clusters = [
            c for c in clusters
            if self.min_cluster_size <= len(c.entities) <= self.max_cluster_size
        ]
        
        # Find unclustered entities
        clustered_entity_ids = set()
        for cluster in valid_clusters:
            for entity in cluster.entities:
                entity_id = entity.get("id") or id(entity)
                clustered_entity_ids.add(entity_id)
        
        unclustered = [
            e for e in entities
            if (e.get("id") or id(e)) not in clustered_entity_ids
        ]
        
        # Calculate quality metrics
        quality_metrics = self._calculate_cluster_quality(valid_clusters)
        
        return ClusterResult(
            clusters=valid_clusters,
            unclustered=unclustered,
            quality_metrics=quality_metrics
        )
    
    def _graph_based_clustering(
        self,
        entities: List[Dict[str, Any]],
        threshold: float
    ) -> List[Cluster]:
        """Build clusters using graph-based approach."""
        # Build similarity graph
        similarity_pairs = self.similarity_calculator.batch_calculate_similarity(
            entities,
            threshold=threshold
        )
        
        # Union-find to build clusters
        entity_to_cluster = {}
        clusters_dict = {}
        cluster_id_counter = 0
        
        for entity1, entity2, score in similarity_pairs:
            entity1_id = entity1.get("id") or id(entity1)
            entity2_id = entity2.get("id") or id(entity2)
            
            cluster1 = entity_to_cluster.get(entity1_id)
            cluster2 = entity_to_cluster.get(entity2_id)
            
            if cluster1 is None and cluster2 is None:
                # Create new cluster
                cluster_id = f"cluster_{cluster_id_counter}"
                cluster_id_counter += 1
                
                cluster = Cluster(
                    cluster_id=cluster_id,
                    entities=[entity1, entity2],
                    metadata={"similarity_scores": {(entity1_id, entity2_id): score}}
                )
                clusters_dict[cluster_id] = cluster
                entity_to_cluster[entity1_id] = cluster_id
                entity_to_cluster[entity2_id] = cluster_id
            elif cluster1 is not None and cluster2 is None:
                # Add entity2 to cluster1
                clusters_dict[cluster1].entities.append(entity2)
                entity_to_cluster[entity2_id] = cluster1
            elif cluster1 is None and cluster2 is not None:
                # Add entity1 to cluster2
                clusters_dict[cluster2].entities.append(entity1)
                entity_to_cluster[entity1_id] = cluster2
            elif cluster1 != cluster2:
                # Merge clusters
                cluster1_obj = clusters_dict[cluster1]
                cluster2_obj = clusters_dict[cluster2]
                
                cluster1_obj.entities.extend(cluster2_obj.entities)
                cluster1_obj.metadata.get("similarity_scores", {}).update(
                    cluster2_obj.metadata.get("similarity_scores", {})
                )
                cluster1_obj.metadata["similarity_scores"][(entity1_id, entity2_id)] = score
                
                # Update references
                for entity in cluster2_obj.entities:
                    entity_id = entity.get("id") or id(entity)
                    entity_to_cluster[entity_id] = cluster1
                
                del clusters_dict[cluster2]
        
        return list(clusters_dict.values())
    
    def _hierarchical_clustering(
        self,
        entities: List[Dict[str, Any]],
        threshold: float
    ) -> List[Cluster]:
        """Build clusters using hierarchical clustering."""
        # Simplified hierarchical clustering
        # Start with each entity as its own cluster
        clusters = [
            Cluster(
                cluster_id=f"cluster_{i}",
                entities=[entity],
                metadata={}
            )
            for i, entity in enumerate(entities)
        ]
        
        # Merge clusters based on similarity
        merged = True
        while merged:
            merged = False
            best_merge = None
            best_similarity = threshold
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Calculate cluster similarity
                    similarity = self._cluster_similarity(clusters[i], clusters[j])
                    
                    if similarity >= best_similarity:
                        best_similarity = similarity
                        best_merge = (i, j)
            
            if best_merge:
                i, j = best_merge
                # Merge clusters
                merged_cluster = Cluster(
                    cluster_id=clusters[i].cluster_id,
                    entities=clusters[i].entities + clusters[j].entities,
                    metadata={"merge_similarity": best_similarity}
                )
                clusters[i] = merged_cluster
                clusters.pop(j)
                merged = True
        
        return clusters
    
    def _cluster_similarity(self, cluster1: Cluster, cluster2: Cluster) -> float:
        """Calculate similarity between two clusters."""
        # Average similarity between all pairs
        similarities = []
        
        for entity1 in cluster1.entities:
            for entity2 in cluster2.entities:
                similarity = self.similarity_calculator.calculate_similarity(
                    entity1,
                    entity2
                )
                similarities.append(similarity.score)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_cluster_quality(self, clusters: List[Cluster]) -> Dict[str, Any]:
        """Calculate quality metrics for clusters."""
        if not clusters:
            return {
                "average_size": 0,
                "average_quality": 0.0,
                "total_clusters": 0
            }
        
        # Calculate cluster quality scores
        for cluster in clusters:
            cluster.quality_score = self._cluster_quality_score(cluster)
        
        avg_quality = sum(c.quality_score for c in clusters) / len(clusters)
        avg_size = sum(len(c.entities) for c in clusters) / len(clusters)
        
        return {
            "average_size": avg_size,
            "average_quality": avg_quality,
            "total_clusters": len(clusters),
            "high_quality_clusters": len([c for c in clusters if c.quality_score >= 0.7])
        }
    
    def _cluster_quality_score(self, cluster: Cluster) -> float:
        """Calculate quality score for a cluster."""
        if len(cluster.entities) < 2:
            return 0.0
        
        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(cluster.entities)):
            for j in range(i + 1, len(cluster.entities)):
                similarity = self.similarity_calculator.calculate_similarity(
                    cluster.entities[i],
                    cluster.entities[j]
                )
                similarities.append(similarity.score)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Size factor (prefer medium-sized clusters)
        size_factor = 1.0
        size = len(cluster.entities)
        if size < self.min_cluster_size or size > self.max_cluster_size:
            size_factor = 0.8
        
        return avg_similarity * size_factor
    
    def update_clusters(
        self,
        existing_clusters: List[Cluster],
        new_entities: List[Dict[str, Any]],
        **options
    ) -> ClusterResult:
        """
        Incrementally update clusters with new entities.
        
        Args:
            existing_clusters: Existing clusters
            new_entities: New entities to add
            **options: Update options
            
        Returns:
            Updated ClusterResult
        """
        threshold = options.get("threshold", self.similarity_threshold)
        
        # Try to add new entities to existing clusters
        for entity in new_entities:
            best_cluster = None
            best_similarity = threshold
            
            for cluster in existing_clusters:
                # Calculate similarity to cluster centroid or average
                similarity = self._entity_cluster_similarity(entity, cluster)
                
                if similarity >= best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster
            
            if best_cluster:
                best_cluster.entities.append(entity)
            else:
                # Create new singleton cluster
                new_cluster = Cluster(
                    cluster_id=f"cluster_{len(existing_clusters)}",
                    entities=[entity]
                )
                existing_clusters.append(new_cluster)
        
        # Rebuild all clusters
        all_entities = []
        for cluster in existing_clusters:
            all_entities.extend(cluster.entities)
        
        return self.build_clusters(all_entities, **options)
    
    def _entity_cluster_similarity(
        self,
        entity: Dict[str, Any],
        cluster: Cluster
    ) -> float:
        """Calculate similarity between entity and cluster."""
        if not cluster.entities:
            return 0.0
        
        # Average similarity to all entities in cluster
        similarities = [
            self.similarity_calculator.calculate_similarity(entity, e).score
            for e in cluster.entities
        ]
        
        return sum(similarities) / len(similarities) if similarities else 0.0
