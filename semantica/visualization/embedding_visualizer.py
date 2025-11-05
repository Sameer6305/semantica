"""
Embedding Visualizer

This module provides visualization capabilities for embeddings including
2D/3D projections, similarity heatmaps, clustering visualizations, and multi-modal comparisons.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None

from ..utils.logging import get_logger
from ..utils.exceptions import ProcessingError
from .utils.color_schemes import ColorPalette, ColorScheme
from .utils.export_formats import export_plotly_figure, export_matplotlib_figure


class EmbeddingVisualizer:
    """
    Embedding visualizer.
    
    Provides visualization methods for embeddings including:
    - 2D/3D dimensionality reduction projections
    - Similarity heatmaps
    - Clustering visualizations
    - Multi-modal embedding comparisons
    """
    
    def __init__(self, **config):
        """
        Initialize embedding visualizer.
        
        Args:
            **config: Configuration options:
                - color_scheme: Color scheme name
                - point_size: Point size for scatter plots
        """
        self.logger = get_logger("embedding_visualizer")
        self.config = config
        
        color_scheme_name = config.get("color_scheme", "default")
        try:
            self.color_scheme = ColorScheme[color_scheme_name.upper()]
        except (KeyError, AttributeError):
            self.color_scheme = ColorScheme.DEFAULT
        self.point_size = config.get("point_size", 5)
    
    def visualize_2d_projection(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = "umap",
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize embeddings in 2D using dimensionality reduction.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            labels: Optional labels for coloring points
            method: Reduction method ("umap", "tsne", "pca")
            output: Output type ("interactive", "html", "png", "svg")
            file_path: Output file path
            **options: Additional options:
                - n_components: Number of components (default: 2)
                - perplexity: Perplexity for t-SNE
                - n_neighbors: Number of neighbors for UMAP
                
        Returns:
            Visualization figure or None
        """
        self.logger.info(f"Visualizing 2D projection using {method}")
        
        if embeddings.shape[1] <= 2:
            # Already 2D or less, use directly
            projected = embeddings[:, :2]
        else:
            projected = self._reduce_dimensions(embeddings, method=method, n_components=2, **options)
        
        return self._visualize_2d_plotly(projected, labels, output, file_path, **options)
    
    def visualize_3d_projection(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        method: str = "umap",
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize embeddings in 3D using dimensionality reduction.
        
        Args:
            embeddings: Embedding matrix (n_samples, n_features)
            labels: Optional labels for coloring points
            method: Reduction method ("umap", "tsne", "pca")
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info(f"Visualizing 3D projection using {method}")
        
        if embeddings.shape[1] <= 3:
            projected = embeddings[:, :3]
        else:
            projected = self._reduce_dimensions(embeddings, method=method, n_components=3, **options)
        
        return self._visualize_3d_plotly(projected, labels, output, file_path, **options)
    
    def visualize_similarity_heatmap(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[str]] = None,
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize similarity heatmap between embeddings.
        
        Args:
            embeddings: Embedding matrix
            labels: Optional labels for axis
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing similarity heatmap")
        
        # Calculate cosine similarity
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(normalized, normalized.T)
        
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            colorscale='Viridis',
            text=similarity_matrix,
            texttemplate='%{text:.2f}',
            textfont={"size": 8}
        ))
        
        if labels:
            fig.update_layout(
                xaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels),
                yaxis=dict(tickmode='array', tickvals=list(range(len(labels))), ticktext=labels)
            )
        
        fig.update_layout(
            title="Embedding Similarity Heatmap",
            xaxis_title="Embedding Index",
            yaxis_title="Embedding Index",
            width=800,
            height=800
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_clustering(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        method: str = "umap",
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize embeddings with cluster coloring.
        
        Args:
            embeddings: Embedding matrix
            cluster_labels: Cluster assignments for each embedding
            method: Reduction method for 2D projection
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing embedding clusters")
        
        # Project to 2D
        if embeddings.shape[1] <= 2:
            projected = embeddings[:, :2]
        else:
            projected = self._reduce_dimensions(embeddings, method=method, n_components=2, **options)
        
        num_clusters = len(set(cluster_labels))
        cluster_colors = ColorPalette.get_community_colors(num_clusters, self.color_scheme)
        
        colors = [cluster_colors[label % num_clusters] for label in cluster_labels]
        
        fig = go.Figure(data=go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode='markers',
            marker=dict(
                size=self.point_size,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=[f"Cluster {label}" for label in cluster_labels],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Embedding Clusters",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=800,
            height=600
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_multimodal_comparison(
        self,
        text_embeddings: Optional[np.ndarray] = None,
        image_embeddings: Optional[np.ndarray] = None,
        audio_embeddings: Optional[np.ndarray] = None,
        method: str = "umap",
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize multi-modal embeddings side by side.
        
        Args:
            text_embeddings: Text embeddings
            image_embeddings: Image embeddings
            audio_embeddings: Audio embeddings
            method: Reduction method
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing multi-modal embedding comparison")
        
        # Collect embeddings and labels
        all_embeddings = []
        all_labels = []
        all_types = []
        
        if text_embeddings is not None:
            all_embeddings.append(text_embeddings)
            all_labels.extend([f"Text {i}" for i in range(len(text_embeddings))])
            all_types.extend(["text"] * len(text_embeddings))
        
        if image_embeddings is not None:
            all_embeddings.append(image_embeddings)
            all_labels.extend([f"Image {i}" for i in range(len(image_embeddings))])
            all_types.extend(["image"] * len(image_embeddings))
        
        if audio_embeddings is not None:
            all_embeddings.append(audio_embeddings)
            all_labels.extend([f"Audio {i}" for i in range(len(audio_embeddings))])
            all_types.extend(["audio"] * len(audio_embeddings))
        
        if not all_embeddings:
            raise ProcessingError("No embeddings provided")
        
        # Concatenate embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        # Project to 2D
        if combined_embeddings.shape[1] <= 2:
            projected = combined_embeddings[:, :2]
        else:
            projected = self._reduce_dimensions(combined_embeddings, method=method, n_components=2, **options)
        
        # Color by type
        type_colors = {"text": "#1f77b4", "image": "#ff7f0e", "audio": "#2ca02c"}
        colors = [type_colors.get(t, "#888") for t in all_types]
        
        fig = go.Figure(data=go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode='markers',
            marker=dict(
                size=self.point_size,
                color=colors,
                line=dict(width=1, color='black')
            ),
            text=all_labels,
            hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Multi-Modal Embedding Comparison",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=800,
            height=600
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_quality_metrics(
        self,
        embeddings: np.ndarray,
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize embedding quality metrics (norms, distributions).
        
        Args:
            embeddings: Embedding matrix
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing embedding quality metrics")
        
        # Calculate norms
        norms = np.linalg.norm(embeddings, axis=1)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Embedding Norm Distribution", "Norm Statistics"),
            specs=[[{"type": "histogram"}, {"type": "bar"}]]
        )
        
        # Norm distribution
        fig.add_trace(
            go.Histogram(x=norms, nbinsx=30, name="Norm Distribution"),
            row=1, col=1
        )
        
        # Statistics
        stats = {
            "Mean": np.mean(norms),
            "Std": np.std(norms),
            "Min": np.min(norms),
            "Max": np.max(norms)
        }
        
        fig.add_trace(
            go.Bar(x=list(stats.keys()), y=list(stats.values()), name="Statistics"),
            row=1, col=2
        )
        
        fig.update_layout(title="Embedding Quality Metrics")
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def _reduce_dimensions(
        self,
        embeddings: np.ndarray,
        method: str = "umap",
        n_components: int = 2,
        **options
    ) -> np.ndarray:
        """Reduce embedding dimensions using specified method."""
        if method == "pca":
            pca = PCA(n_components=n_components, **options)
            return pca.fit_transform(embeddings)
        
        elif method == "tsne":
            perplexity = options.get("perplexity", min(30, len(embeddings) - 1))
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, **options)
            return tsne.fit_transform(embeddings)
        
        elif method == "umap":
            if umap is not None:
                n_neighbors = options.get("n_neighbors", min(15, len(embeddings) - 1))
                reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, **options)
                return reducer.fit_transform(embeddings)
            else:
                # Fallback to PCA if UMAP not available
                self.logger.warning("UMAP not available, using PCA. Install with: pip install umap-learn")
                pca = PCA(n_components=n_components)
                return pca.fit_transform(embeddings)
        
        else:
            # Fallback to PCA
            self.logger.warning(f"Method {method} not available, using PCA")
            pca = PCA(n_components=n_components)
            return pca.fit_transform(embeddings)
    
    def _visualize_2d_plotly(
        self,
        projected: np.ndarray,
        labels: Optional[List[str]],
        output: str,
        file_path: Optional[Path],
        **options
    ) -> Optional[Any]:
        """Create 2D Plotly visualization."""
        if labels:
            fig = go.Figure(data=go.Scatter(
                x=projected[:, 0],
                y=projected[:, 1],
                mode='markers+text',
                text=labels,
                textposition="top center",
                marker=dict(size=self.point_size, color='lightblue', line=dict(width=1, color='darkblue'))
            ))
        else:
            fig = go.Figure(data=go.Scatter(
                x=projected[:, 0],
                y=projected[:, 1],
                mode='markers',
                marker=dict(size=self.point_size, color='lightblue', line=dict(width=1, color='darkblue'))
            ))
        
        fig.update_layout(
            title="Embedding 2D Projection",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=800,
            height=600
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def _visualize_3d_plotly(
        self,
        projected: np.ndarray,
        labels: Optional[List[str]],
        output: str,
        file_path: Optional[Path],
        **options
    ) -> Optional[Any]:
        """Create 3D Plotly visualization."""
        if labels:
            fig = go.Figure(data=go.Scatter3d(
                x=projected[:, 0],
                y=projected[:, 1],
                z=projected[:, 2],
                mode='markers+text',
                text=labels,
                textposition="top center",
                marker=dict(size=self.point_size, color='lightblue', line=dict(width=1, color='darkblue'))
            ))
        else:
            fig = go.Figure(data=go.Scatter3d(
                x=projected[:, 0],
                y=projected[:, 1],
                z=projected[:, 2],
                mode='markers',
                marker=dict(size=self.point_size, color='lightblue', line=dict(width=1, color='darkblue'))
            ))
        
        fig.update_layout(
            title="Embedding 3D Projection",
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3"
            ),
            width=800,
            height=600
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None

