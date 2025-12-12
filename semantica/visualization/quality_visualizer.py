"""
Quality Metrics Visualizer Module

This module provides comprehensive visualization capabilities for knowledge graph quality
metrics in the Semantica framework, including quality dashboards, completeness metrics,
consistency analysis, issue tracking, and quality score distributions.

Key Features:
    - Comprehensive quality dashboards with gauge indicators
    - Quality score distribution histograms
    - Quality issue analysis by type and severity
    - Completeness metrics visualization
    - Consistency heatmap visualization
    - Support for QualityReport objects and dictionaries
    - Interactive and static output formats

Main Classes:
    - QualityVisualizer: Main quality metrics visualizer coordinator

Example Usage:
    >>> from semantica.visualization import QualityVisualizer
    >>> viz = QualityVisualizer(color_scheme="default")
    >>> fig = viz.visualize_dashboard(quality_report, output="interactive")
    >>> viz.visualize_score_distribution(quality_scores, file_path="scores.png")
    >>> viz.visualize_issues(quality_report, output="interactive")
    >>> viz.visualize_completeness_metrics(completeness_metrics, file_path="completeness.html")
    >>> viz.visualize_consistency_heatmap(consistency_data, output="interactive")

Author: Semantica Contributors
License: MIT
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    px = None
    go = None
    make_subplots = None

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .utils.color_schemes import ColorScheme
from .utils.export_formats import export_plotly_figure


class QualityVisualizer:
    """
    Quality metrics visualizer.

    Provides visualization methods for quality metrics including:
    - Quality dashboards
    - Completeness metrics
    - Consistency analysis
    - Issue tracking
    """

    def __init__(self, **config):
        """Initialize quality visualizer."""
        self.logger = get_logger("quality_visualizer")
        self.config = config
        self.progress_tracker = get_progress_tracker()
        color_scheme_name = config.get("color_scheme", "default")
        try:
            self.color_scheme = ColorScheme[color_scheme_name.upper()]
        except (KeyError, AttributeError):
            self.color_scheme = ColorScheme.DEFAULT

    def _check_dependencies(self):
        """Check if dependencies are available."""
        if px is None or go is None:
            raise ProcessingError(
                "Plotly is required for quality visualization. "
                "Install with: pip install plotly"
            )

    def visualize_dashboard(
        self,
        quality_report: Any,
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options,
    ) -> Optional[Any]:
        """
        Visualize comprehensive quality dashboard.

        Args:
            quality_report: QualityReport object or dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options

        Returns:
            Visualization figure or None
        """
        self._check_dependencies()
        tracking_id = self.progress_tracker.start_tracking(
            module="visualization",
            submodule="QualityVisualizer",
            message="Visualizing quality dashboard",
        )

        try:
            self.logger.info("Visualizing quality dashboard")

            # Extract metrics
            self.progress_tracker.update_tracking(
                tracking_id, message="Extracting quality metrics..."
            )
            if hasattr(quality_report, "overall_score"):
                overall_score = quality_report.overall_score
                consistency_score = quality_report.consistency_score
                completeness_score = quality_report.completeness_score
            elif isinstance(quality_report, dict):
                overall_score = quality_report.get("overall_score", 0.0)
                consistency_score = quality_report.get("consistency_score", 0.0)
                completeness_score = quality_report.get("completeness_score", 0.0)
            else:
                overall_score = 0.0
                consistency_score = 0.0
                completeness_score = 0.0

            # Create dashboard
            self.progress_tracker.update_tracking(
                tracking_id, message="Creating dashboard visualization..."
            )
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Overall Quality Score",
                    "Consistency Score",
                    "Completeness Score",
                    "Quality Breakdown",
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "bar"}],
                ],
            )

            # Overall score
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=overall_score * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Overall Quality"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "gray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                ),
                row=1,
                col=1,
            )

            # Consistency score
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=consistency_score * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Consistency"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkgreen"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "gray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                ),
                row=1,
                col=2,
            )

            # Completeness score
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=completeness_score * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Completeness"},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkorange"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "gray"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                ),
                row=2,
                col=1,
            )

            # Quality breakdown
            metrics = {
                "Overall": overall_score * 100,
                "Consistency": consistency_score * 100,
                "Completeness": completeness_score * 100,
            }

            fig.add_trace(
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color="lightblue",
                ),
                row=2,
                col=2,
            )

            fig.update_layout(title="Knowledge Graph Quality Dashboard", height=800)

            if output == "interactive":
                self.progress_tracker.stop_tracking(
                    tracking_id,
                    status="completed",
                    message="Quality dashboard visualization generated",
                )
                return fig
            elif file_path:
                export_plotly_figure(
                    fig, file_path, format=output if output != "interactive" else "html"
                )
                self.progress_tracker.stop_tracking(
                    tracking_id,
                    status="completed",
                    message=f"Quality dashboard saved to {file_path}",
                )
                return None
        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def visualize_score_distribution(
        self,
        quality_scores: List[float],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options,
    ) -> Optional[Any]:
        """
        Visualize quality score distribution.

        Args:
            quality_scores: List of quality scores
            output: Output type
            file_path: Output file path
            **options: Additional options

        Returns:
            Visualization figure or None
        """
        self._check_dependencies()
        self.logger.info("Visualizing quality score distribution")

        fig = go.Figure(
            data=[
                go.Histogram(
                    x=quality_scores,
                    nbinsx=30,
                    marker_color="lightblue",
                    marker_line_color="darkblue",
                    marker_line_width=1,
                )
            ]
        )

        fig.update_layout(
            title="Quality Score Distribution",
            xaxis_title="Quality Score",
            yaxis_title="Frequency",
            width=800,
            height=600,
        )

        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(
                fig, file_path, format=output if output != "interactive" else "html"
            )
            return None

    def visualize_issues(
        self,
        quality_report: Any,
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options,
    ) -> Optional[Any]:
        """
        Visualize quality issues.

        Args:
            quality_report: QualityReport with issues
            output: Output type
            file_path: Output file path
            **options: Additional options

        Returns:
            Visualization figure or None
        """
        self._check_dependencies()
        self.logger.info("Visualizing quality issues")

        # Extract issues
        issues = []
        if hasattr(quality_report, "issues"):
            issues = quality_report.issues
        elif isinstance(quality_report, dict):
            issues = quality_report.get("issues", [])

        if not issues:
            self.logger.warning("No issues found in quality report")
            return None

        # Count issues by type and severity
        issue_counts = {}
        severity_counts = {}

        for issue in issues:
            issue_type = (
                issue.type if hasattr(issue, "type") else issue.get("type", "Unknown")
            )
            severity = (
                issue.severity
                if hasattr(issue, "severity")
                else issue.get("severity", "Unknown")
            )

            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Issues by Type", "Issues by Severity"),
            specs=[[{"type": "bar"}, {"type": "pie"}]],
        )

        # Issues by type
        fig.add_trace(
            go.Bar(
                x=list(issue_counts.keys()),
                y=list(issue_counts.values()),
                marker_color="lightcoral",
                name="Issues",
            ),
            row=1,
            col=1,
        )

        # Issues by severity
        fig.add_trace(
            go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                name="Severity",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(title="Quality Issues Analysis", height=600)

        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(
                fig, file_path, format=output if output != "interactive" else "html"
            )
            return None

    def visualize_completeness_metrics(
        self,
        completeness_metrics: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options,
    ) -> Optional[Any]:
        """
        Visualize completeness metrics.

        Args:
            completeness_metrics: Completeness metrics dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options

        Returns:
            Visualization figure or None
        """
        self._check_dependencies()
        self.logger.info("Visualizing completeness metrics")

        # Extract metrics
        entity_completeness = completeness_metrics.get("entity_completeness", 0.0)
        property_completeness = completeness_metrics.get("property_completeness", 0.0)
        relationship_completeness = completeness_metrics.get(
            "relationship_completeness", 0.0
        )

        metrics = {
            "Entity Completeness": entity_completeness * 100,
            "Property Completeness": property_completeness * 100,
            "Relationship Completeness": relationship_completeness * 100,
        }

        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color="lightgreen",
                    text=[f"{v:.1f}%" for v in metrics.values()],
                    textposition="auto",
                )
            ]
        )

        fig.update_layout(
            title="Completeness Metrics",
            xaxis_title="Metric",
            yaxis_title="Completeness (%)",
            yaxis_range=[0, 100],
            width=800,
            height=500,
        )

        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(
                fig, file_path, format=output if output != "interactive" else "html"
            )
            return None

    def visualize_consistency_heatmap(
        self,
        consistency_data: Dict[str, Any],
        output: str = "interactive",
        file_path: Optional[Union[str, Path]] = None,
        **options,
    ) -> Optional[Any]:
        """
        Visualize consistency heatmap.

        Args:
            consistency_data: Consistency data dictionary
            output: Output type
            file_path: Output file path
            **options: Additional options

        Returns:
            Visualization figure or None
        """
        self._check_dependencies()
        if np is None:
            raise ProcessingError(
                "NumPy is required for consistency heatmap visualization. "
                "Install with: pip install numpy"
            )

        self.logger.info("Visualizing consistency heatmap")

        # Extract consistency matrix
        matrix = consistency_data.get("consistency_matrix", [])
        labels = consistency_data.get("labels", [])

        if not matrix:
            raise ProcessingError("No consistency matrix found")

        matrix = np.array(matrix)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=labels if labels else None,
                y=labels if labels else None,
                colorscale="RdYlGn",
                text=matrix,
                texttemplate="%{text:.2f}",
                textfont={"size": 8},
            )
        )

        fig.update_layout(
            title="Consistency Heatmap",
            xaxis_title="Entity/Type",
            yaxis_title="Entity/Type",
            width=800,
            height=800,
        )

        if output == "interactive":
            return fig
        elif file_path:
            export_plotly_figure(
                fig, file_path, format=output if output != "interactive" else "html"
            )
            return None
