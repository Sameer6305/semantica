"""
Export utilities for visualizations.

This module provides functions for exporting visualizations to various formats.
"""

from typing import Optional, Union
from pathlib import Path

from ...utils.logging import get_logger
from ...utils.exceptions import ProcessingError


def export_plotly_figure(fig, file_path: Union[str, Path], format: str = "html", **options) -> None:
    """
    Export Plotly figure to file.
    
    Args:
        fig: Plotly figure object
        file_path: Output file path
        format: Export format ("html", "png", "svg", "pdf", "json")
        **options: Additional export options
    """
    logger = get_logger("export_plotly")
    file_path = Path(file_path)
    
    try:
        if format == "html":
            fig.write_html(str(file_path), **options)
        elif format == "png":
            fig.write_image(str(file_path), **options)
        elif format == "svg":
            fig.write_image(str(file_path), format="svg", **options)
        elif format == "pdf":
            fig.write_image(str(file_path), format="pdf", **options)
        elif format == "json":
            fig.write_json(str(file_path), **options)
        else:
            raise ProcessingError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported Plotly figure to {file_path}")
    except Exception as e:
        logger.error(f"Failed to export Plotly figure: {e}")
        raise ProcessingError(f"Failed to export Plotly figure: {e}")


def export_matplotlib_figure(fig, file_path: Union[str, Path], format: str = "png", **options) -> None:
    """
    Export Matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        file_path: Output file path
        format: Export format ("png", "svg", "pdf", "jpg")
        **options: Additional export options (dpi, bbox_inches, etc.)
    """
    logger = get_logger("export_matplotlib")
    file_path = Path(file_path)
    
    try:
        dpi = options.get("dpi", 300)
        bbox_inches = options.get("bbox_inches", "tight")
        
        fig.savefig(
            str(file_path),
            format=format,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **{k: v for k, v in options.items() if k not in ["dpi", "bbox_inches"]}
        )
        
        logger.info(f"Exported Matplotlib figure to {file_path}")
    except Exception as e:
        logger.error(f"Failed to export Matplotlib figure: {e}")
        raise ProcessingError(f"Failed to export Matplotlib figure: {e}")


def save_html(html_content: str, file_path: Union[str, Path]) -> None:
    """
    Save HTML content to file.
    
    Args:
        html_content: HTML content string
        file_path: Output file path
    """
    logger = get_logger("save_html")
    file_path = Path(file_path)
    
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save HTML: {e}")
        raise ProcessingError(f"Failed to save HTML: {e}")

