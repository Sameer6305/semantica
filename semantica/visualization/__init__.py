"""
Visualization Module

This module provides comprehensive visualization capabilities for all knowledge artifacts
created by the Semantica framework, including knowledge graphs, ontologies, embeddings,
semantic networks, quality metrics, analytics results, and temporal graphs with interactive
and static output formats.

Key Features:
    - Interactive and static visualization outputs (HTML, PNG, SVG, PDF)
    - Knowledge graph network visualizations with multiple layout algorithms
    - Ontology hierarchy and structure visualizations
    - Embedding dimensionality reduction and clustering visualizations
    - Semantic network visualizations
    - Quality metrics dashboards and issue tracking
    - Graph analytics visualizations (centrality, communities, connectivity)
    - Temporal graph timeline and evolution visualizations
    - Customizable color schemes and layout algorithms

Main Classes:
    - KGVisualizer: Knowledge graph network and community visualizations
    - OntologyVisualizer: Ontology hierarchy, properties, and structure visualizations
    - EmbeddingVisualizer: Vector embedding projections, similarity, and clustering
    - SemanticNetworkVisualizer: Semantic network structure and type distributions
    - QualityVisualizer: Quality metrics dashboards, completeness, and consistency
    - AnalyticsVisualizer: Graph analytics, centrality rankings, and metrics dashboards
    - TemporalVisualizer: Temporal timeline, patterns, and snapshot comparisons

Example Usage:
    >>> from semantica.visualization import KGVisualizer
    >>> viz = KGVisualizer(layout="force", color_scheme="vibrant")
    >>> fig = viz.visualize_network(graph, output="interactive")
    >>> viz.visualize_communities(graph, communities, file_path="communities.html")
    >>> 
    >>> from semantica.visualization import EmbeddingVisualizer
    >>> emb_viz = EmbeddingVisualizer()
    >>> fig = emb_viz.visualize_2d_projection(embeddings, labels, method="umap")
    >>> emb_viz.visualize_clustering(embeddings, cluster_labels, file_path="clusters.png")
    >>> 
    >>> from semantica.visualization import OntologyVisualizer
    >>> ont_viz = OntologyVisualizer()
    >>> fig = ont_viz.visualize_hierarchy(ontology, output="interactive")

Author: Semantica Contributors
License: MIT
"""

from .kg_visualizer import KGVisualizer
from .ontology_visualizer import OntologyVisualizer
from .embedding_visualizer import EmbeddingVisualizer
from .semantic_network_visualizer import SemanticNetworkVisualizer
from .quality_visualizer import QualityVisualizer
from .analytics_visualizer import AnalyticsVisualizer
from .temporal_visualizer import TemporalVisualizer

__all__ = [
    "KGVisualizer",
    "OntologyVisualizer",
    "EmbeddingVisualizer",
    "SemanticNetworkVisualizer",
    "QualityVisualizer",
    "AnalyticsVisualizer",
    "TemporalVisualizer",
]

