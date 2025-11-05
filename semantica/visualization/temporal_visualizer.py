"""
Temporal Graph Visualizer

This module provides visualization capabilities for temporal knowledge graphs including
timeline views, temporal network animations, snapshot comparisons, and temporal pattern visualizations.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..utils.logging import get_logger
from ..utils.exceptions import ProcessingError
from .utils.color_schemes import ColorPalette, ColorScheme
from .utils.export_formats import export_plotly_figure


class TemporalVisualizer:
    """
    Temporal graph visualizer.
    
    Provides visualization methods for temporal graphs including:
    - Timeline views
    - Temporal network animations
    - Snapshot comparisons
    - Temporal pattern visualizations
    """
    
    def __init__(self, **config):
        """Initialize temporal visualizer."""
        self.logger = get_logger("temporal_visualizer")
        self.config = config
        color_scheme_name = config.get("color_scheme", "default")
        try:
            self.color_scheme = ColorScheme[color_scheme_name.upper()]
        except (KeyError, AttributeError):
            self.color_scheme = ColorScheme.DEFAULT
    
    def visualize_timeline(
        self,
        temporal_data: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize timeline of entity/relationship changes.
        
        Args:
            temporal_data: Temporal data dictionary with timestamps and changes
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing temporal timeline")
        
        # Extract timeline data
        events = temporal_data.get("events", [])
        timestamps = temporal_data.get("timestamps", [])
        
        if not events:
            raise ProcessingError("No temporal events found")
        
        # Build timeline
        event_types = []
        event_times = []
        event_labels = []
        
        for event in events:
            event_time = event.get("timestamp") or event.get("time", "")
            event_type = event.get("type") or event.get("event_type", "change")
            event_label = event.get("label") or event.get("entity", "")
            
            event_times.append(event_time)
            event_types.append(event_type)
            event_labels.append(event_label)
        
        # Create Gantt chart style timeline
        fig = go.Figure()
        
        # Group by type
        type_colors = {}
        unique_types = list(set(event_types))
        colors = ColorPalette.get_colors(self.color_scheme, len(unique_types))
        for i, t in enumerate(unique_types):
            type_colors[t] = colors[i]
        
        for event_type in unique_types:
            mask = [t == event_type for t in event_types]
            type_times = [event_times[i] for i in range(len(event_times)) if mask[i]]
            type_labels = [event_labels[i] for i in range(len(event_labels)) if mask[i]]
            
            fig.add_trace(go.Scatter(
                x=type_times,
                y=[event_type] * len(type_times),
                mode='markers',
                name=event_type,
                marker=dict(size=10, color=type_colors[event_type]),
                text=type_labels,
                hovertemplate='%{text}<br>Time: %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Temporal Timeline",
            xaxis_title="Time",
            yaxis_title="Event Type",
            width=1200,
            height=600
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_temporal_patterns(
        self,
        patterns: List[Dict[str, Any]],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize detected temporal patterns.
        
        Args:
            patterns: List of temporal patterns
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing temporal patterns")
        
        if not patterns:
            raise ProcessingError("No temporal patterns found")
        
        # Extract pattern data
        pattern_types = [p.get("pattern_type", "Unknown") for p in patterns]
        start_times = [p.get("start_time", "") for p in patterns]
        end_times = [p.get("end_time", "") for p in patterns]
        entities = [p.get("entities", []) for p in patterns]
        
        # Create timeline visualization
        fig = go.Figure()
        
        for i, pattern in enumerate(patterns):
            pattern_type = pattern.get("pattern_type", "Unknown")
            start_time = pattern.get("start_time", "")
            end_time = pattern.get("end_time", "")
            pattern_entities = pattern.get("entities", [])
            
            # Create a bar for the pattern duration
            fig.add_trace(go.Scatter(
                x=[start_time, end_time],
                y=[i, i],
                mode='lines+markers',
                name=pattern_type,
                line=dict(width=10, color=ColorPalette.get_color_by_index(self.color_scheme, i)),
                text=f"{pattern_type}: {len(pattern_entities)} entities",
                hovertemplate='%{text}<br>Start: %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Temporal Patterns",
            xaxis_title="Time",
            yaxis_title="Pattern",
            width=1200,
            height=max(400, len(patterns) * 50)
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_snapshot_comparison(
        self,
        snapshots: Dict[str, Dict[str, Any]],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize comparison of graph snapshots at different times.
        
        Args:
            snapshots: Dictionary mapping timestamps to graph snapshots
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing snapshot comparison")
        
        timestamps = sorted(snapshots.keys())
        
        # Extract metrics for each snapshot
        metrics_over_time = {
            "entities": [],
            "relationships": [],
            "density": []
        }
        
        for timestamp in timestamps:
            snapshot = snapshots[timestamp]
            entities = snapshot.get("entities", [])
            relationships = snapshot.get("relationships", [])
            
            metrics_over_time["entities"].append(len(entities))
            metrics_over_time["relationships"].append(len(relationships))
            
            # Calculate density
            num_nodes = len(entities)
            num_edges = len(relationships)
            max_edges = num_nodes * (num_nodes - 1) / 2 if num_nodes > 1 else 0
            density = num_edges / max_edges if max_edges > 0 else 0
            metrics_over_time["density"].append(density)
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=metrics_over_time["entities"],
            mode='lines+markers',
            name='Entities',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=metrics_over_time["relationships"],
            mode='lines+markers',
            name='Relationships',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=metrics_over_time["density"],
            mode='lines+markers',
            name='Density',
            line=dict(color='green', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Graph Evolution Over Time",
            xaxis_title="Time",
            yaxis_title="Count",
            yaxis2=dict(title="Density", overlaying='y', side='right'),
            width=1200,
            height=600
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_version_history(
        self,
        version_history: List[Dict[str, Any]],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize version history tree.
        
        Args:
            version_history: List of version information
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing version history")
        
        # Build tree structure
        versions = sorted(version_history, key=lambda v: v.get("version", ""))
        
        # Create timeline
        version_names = [v.get("version", f"v{i}") for i, v in enumerate(versions)]
        version_dates = [v.get("date", v.get("timestamp", "")) for v in versions]
        version_changes = [v.get("changes", v.get("description", "")) for v in versions]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=version_dates,
            y=[i for i in range(len(versions))],
            mode='lines+markers+text',
            text=version_names,
            textposition="middle right",
            line=dict(color='blue', width=2),
            marker=dict(size=10, color='red'),
            hovertemplate='Version: %{text}<br>Date: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Version History",
            xaxis_title="Date",
            yaxis_title="Version",
            yaxis=dict(tickmode='array', tickvals=list(range(len(versions))), ticktext=version_names),
            width=1200,
            height=max(400, len(versions) * 50)
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None
    
    def visualize_metrics_evolution(
        self,
        metrics_history: Dict[str, List[float]],
        timestamps: List[str],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Optional[Any]:
        """
        Visualize metrics evolution over time.
        
        Args:
            metrics_history: Dictionary mapping metric names to time series
            timestamps: List of timestamps
            output: Output type
            file_path: Output file path
            **options: Additional options
            
        Returns:
            Visualization figure or None
        """
        self.logger.info("Visualizing metrics evolution")
        
        fig = go.Figure()
        
        colors = ColorPalette.get_colors(self.color_scheme, len(metrics_history))
        
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(color=colors[i], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Metrics Evolution Over Time",
            xaxis_title="Time",
            yaxis_title="Metric Value",
            width=1200,
            height=600,
            hovermode='x unified'
        )
        
        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(fig, file_path, format=output if output != "interactive" else "html")
            return None

