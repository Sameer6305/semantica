"""
Knowledge Graph Visualizer

This module provides visualization capabilities for knowledge graphs including
network graphs, community views, centrality visualizations, and entity/relationship analysis.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ..utils.logging import get_logger
from ..utils.exceptions import ProcessingError
from .utils.layout_algorithms import ForceDirectedLayout, HierarchicalLayout, CircularLayout
from .utils.color_schemes import ColorPalette, ColorScheme
from .utils.export_formats import export_plotly_figure, export_matplotlib_figure, save_html


class KGVisualizer:
    """
    Knowledge graph visualizer.
    
    Provides various visualization methods for knowledge graphs including:
    - Interactive network graphs
    - Community visualizations
    - Centrality visualizations
    - Entity/relationship distribution charts
    """
    
    def __init__(self, **config):
        """
        Initialize knowledge graph visualizer.
        
        Args:
            **config: Configuration options:
                - layout: Layout algorithm ("force", "hierarchical", "circular")
                - color_scheme: Color scheme name
                - node_size: Base node size
                - edge_width: Edge width
        """
        self.logger = get_logger("kg_visualizer")
        self.config = config
        
        self.layout_type = config.get("layout", "force")
        color_scheme_name = config.get("color_scheme", "default")
        try:
            self.color_scheme = ColorScheme[color_scheme_name.upper()]
        except (KeyError, AttributeError):
            self.color_scheme = ColorScheme.DEFAULT
        self.node_size = config.get("node_size", 10)
        self.edge_width = config.get("edge_width", 1)
        
        # Initialize layout algorithms
        self.force_layout = ForceDirectedLayout(**config)
        self.hierarchical_layout = HierarchicalLayout(**config)
        self.circular_layout = CircularLayout(**config)
    
    def visualize_network(
        self,
        graph: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize knowledge graph as interactive network.
        
        Args:
            graph: Knowledge graph dictionary with entities and relationships
            output: Output type ("interactive", "html", "png", "svg")
            file_path: Output file path (required for non-interactive)
            **options: Additional visualization options
            
        Returns:
            Plotly figure (if interactive) or None
        """
        self.logger.info("Visualizing knowledge graph network")
        
        # Extract entities and relationships
        entities = graph.get("entities", [])
        relationships = graph.get("relationships", [])
        
        if not entities:
            raise ProcessingError("No entities found in graph")
        
        # Build node and edge lists
        nodes = self._extract_nodes(entities)
        edges = self._extract_edges(relationships, entities)
        
        return self._visualize_network_plotly(nodes, edges, output, file_path, **options)
    
    def visualize_communities(
        self,
        graph: Dict[str, Any],
        communities: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize knowledge graph with community coloring.
        
        Args:
            graph: Knowledge graph dictionary
            communities: Community detection results with node_assignments
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing knowledge graph communities")
        
        entities = graph.get("entities", [])
        relationships = graph.get("relationships", [])
        
        # Get community assignments
        node_assignments = communities.get("node_assignments", {})
        num_communities = communities.get("num_communities", len(set(node_assignments.values())))
        
        # Get colors for communities
        community_colors = ColorPalette.get_community_colors(num_communities, self.color_scheme)
        
        # Build nodes with community colors
        nodes = self._extract_nodes(entities)
        for node in nodes:
            entity_id = node.get("id", "")
            community_id = node_assignments.get(entity_id, 0)
            node["community"] = community_id
            node["color"] = community_colors[community_id % len(community_colors)]
        
        edges = self._extract_edges(relationships, entities)
        
        return self._visualize_network_plotly(nodes, edges, output, file_path, community_colors=community_colors, **options)
    
    def visualize_centrality(
        self,
        graph: Dict[str, Any],
        centrality: Dict[str, Any],
        centrality_type: str = "degree",
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize knowledge graph with centrality-based node sizing/coloring.
        
        Args:
            graph: Knowledge graph dictionary
            centrality: Centrality calculation results
            centrality_type: Type of centrality ("degree", "betweenness", "closeness", "eigenvector")
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info(f"Visualizing knowledge graph with {centrality_type} centrality")
        
        entities = graph.get("entities", [])
        relationships = graph.get("relationships", [])
        
        # Get centrality scores
        centrality_scores = centrality.get("centrality", {})
        if not centrality_scores:
            # Try to get from rankings
            rankings = centrality.get("rankings", [])
            centrality_scores = {r["node"]: r["score"] for r in rankings}
        
        # Build nodes with centrality information
        nodes = self._extract_nodes(entities)
        max_centrality = max(centrality_scores.values()) if centrality_scores else 1.0
        
        for node in nodes:
            entity_id = node.get("id", "")
            score = centrality_scores.get(entity_id, 0.0)
            node["centrality"] = score
            node["size"] = self.node_size * (1 + 5 * (score / max_centrality)) if max_centrality > 0 else self.node_size
        
        edges = self._extract_edges(relationships, entities)
        
        return self._visualize_network_plotly(nodes, edges, output, file_path, size_by_centrality=True, **options)
    
    def visualize_entity_types(
        self,
        graph: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize entity type distribution.
        
        Args:
            graph: Knowledge graph dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing entity type distribution")
        
        entities = graph.get("entities", [])
        
        # Count entity types
        type_counts = {}
        for entity in entities:
            entity_type = entity.get("type") or entity.get("entity_type") or "Unknown"
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        fig = px.bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            labels={"x": "Entity Type", "y": "Count"},
            title="Entity Type Distribution"
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_relationship_matrix(
        self,
        graph: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize relationship frequency matrix between entity types.
        
        Args:
            graph: Knowledge graph dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing relationship matrix")
        
        entities = graph.get("entities", [])
        relationships = graph.get("relationships", [])
        
        # Build entity type map
        entity_type_map = {}
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id", "")
            entity_type = entity.get("type") or entity.get("entity_type", "Unknown")
            entity_type_map[entity_id] = entity_type
        
        # Count relationships between types
        type_matrix = {}
        for rel in relationships:
            source_id = rel.get("source") or rel.get("subject", "")
            target_id = rel.get("target") or rel.get("object", "")
            
            source_type = entity_type_map.get(source_id, "Unknown")
            target_type = entity_type_map.get(target_id, "Unknown")
            
            key = (source_type, target_type)
            type_matrix[key] = type_matrix.get(key, 0) + 1
        
        # Convert to matrix format
        source_types = sorted(set(t[0] for t in type_matrix.keys()))
        target_types = sorted(set(t[1] for t in type_matrix.keys()))
        
        matrix = np.zeros((len(source_types), len(target_types)))
        for i, source_type in enumerate(source_types):
            for j, target_type in enumerate(target_types):
                matrix[i, j] = type_matrix.get((source_type, target_type), 0)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=target_types,
            y=source_types,
            colorscale='Viridis',
            text=matrix,
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title="Relationship Frequency Matrix",
            xaxis_title="Target Entity Type",
            yaxis_title="Source Entity Type"
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def _extract_nodes(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract nodes from entities."""
        nodes = []
        for entity in entities:
            node = {
                "id": entity.get("id") or entity.get("entity_id", ""),
                "label": entity.get("text") or entity.get("label") or entity.get("name", ""),
                "type": entity.get("type") or entity.get("entity_type", "entity"),
                "metadata": entity.get("metadata", {})
            }
            nodes.append(node)
        return nodes
    
    def _extract_edges(self, relationships: List[Dict[str, Any]], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract edges from relationships."""
        edges = []
        entity_map = {e.get("id") or e.get("entity_id", ""): e for e in entities}
        
        for rel in relationships:
            source_id = rel.get("source") or rel.get("subject", "")
            target_id = rel.get("target") or rel.get("object", "")
            
            if source_id in entity_map and target_id in entity_map:
                edge = {
                    "source": source_id,
                    "target": target_id,
                    "label": rel.get("type") or rel.get("relationship_type") or rel.get("predicate", ""),
                    "metadata": rel.get("metadata", {})
                }
                edges.append(edge)
        
        return edges
    
    def _visualize_network_plotly(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        output: str,
        file_path: Optional[Path],
        **options
    ) -> Optional[Any]:
        """Create Plotly network visualization."""
        # Compute layout
        node_ids = [n["id"] for n in nodes]
        edge_tuples = [(e["source"], e["target"]) for e in edges]
        
        if self.layout_type == "hierarchical":
            pos = self.hierarchical_layout.compute_layout(node_ids, edge_tuples, **options)
        elif self.layout_type == "circular":
            pos = self.circular_layout.compute_layout(node_ids, edge_tuples, **options)
        else:
            pos = self.force_layout.compute_layout(node_ids, edge_tuples, **options)
        
        # Get entity type colors
        entity_types = list(set(n.get("type", "entity") for n in nodes))
        type_colors = ColorPalette.get_entity_type_colors(entity_types, self.color_scheme)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        for edge in edges:
            source_pos = pos.get(edge["source"])
            target_pos = pos.get(edge["target"])
            if source_pos and target_pos:
                edge_x.extend([source_pos[0], target_pos[0], None])
                edge_y.extend([source_pos[1], target_pos[1], None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=self.edge_width, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces
        node_x = [pos[n["id"]][0] for n in nodes if n["id"] in pos]
        node_y = [pos[n["id"]][1] for n in nodes if n["id"] in pos]
        node_text = [n["label"] for n in nodes if n["id"] in pos]
        node_colors = [type_colors.get(n.get("type", "entity"), "#888") for n in nodes if n["id"] in pos]
        node_sizes = [n.get("size", self.node_size) for n in nodes if n["id"] in pos]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Knowledge Graph Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None

