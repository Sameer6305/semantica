"""
Report generator for Semantica framework.

This module provides report generation capabilities
for analysis results and framework metrics.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime
import json

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


class ReportGenerator:
    """
    Report generator for analysis results and metrics.
    
    • Report generation functionality
    • Analysis result formatting
    • Framework metrics reporting
    • Performance optimization
    • Error handling and recovery
    • Advanced reporting features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize report generator.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - format: Report format ('html', 'markdown', 'json', 'text')
                - include_charts: Include charts/visualizations (default: False)
                - template: Custom report template
        """
        self.logger = get_logger("report_generator")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.format = self.config.get("format", "markdown")
        self.include_charts = self.config.get("include_charts", False)
        self.template = self.config.get("template")
    
    def generate_report(
        self,
        data: Dict[str, Any],
        file_path: Optional[Union[str, Path]] = None,
        format: Optional[str] = None,
        **options
    ) -> Union[str, None]:
        """
        Generate report from data.
        
        Args:
            data: Report data dictionary
            file_path: Optional output file path
            format: Report format ('html', 'markdown', 'json', 'text')
            **options: Additional options
            
        Returns:
            Report string if file_path not provided, None otherwise
        """
        report_format = format or self.format
        
        if report_format == "markdown":
            report = self._generate_markdown(data, **options)
        elif report_format == "html":
            report = self._generate_html(data, **options)
        elif report_format == "json":
            report = self._generate_json(data, **options)
        elif report_format == "text":
            report = self._generate_text(data, **options)
        else:
            raise ValidationError(f"Unsupported report format: {report_format}")
        
        if file_path:
            file_path = Path(file_path)
            ensure_directory(file_path.parent)
            
            encoding = options.get("encoding", "utf-8")
            with open(file_path, "w", encoding=encoding) as f:
                f.write(report)
            
            self.logger.info(f"Generated report ({report_format}) to: {file_path}")
            return None
        
        return report
    
    def generate_quality_report(
        self,
        quality_metrics: Dict[str, Any],
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Union[str, None]:
        """
        Generate quality assurance report.
        
        Args:
            quality_metrics: Quality metrics dictionary
            file_path: Optional output file path
            **options: Additional options
            
        Returns:
            Report string if file_path not provided, None otherwise
        """
        report_data = {
            "title": "Quality Assurance Report",
            "generated_at": datetime.now().isoformat(),
            "metrics": quality_metrics,
            "summary": self._generate_quality_summary(quality_metrics)
        }
        
        return self.generate_report(report_data, file_path=file_path, **options)
    
    def generate_analysis_report(
        self,
        analysis_results: Dict[str, Any],
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Union[str, None]:
        """
        Generate analysis report.
        
        Args:
            analysis_results: Analysis results dictionary
            file_path: Optional output file path
            **options: Additional options
            
        Returns:
            Report string if file_path not provided, None otherwise
        """
        report_data = {
            "title": "Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "analysis": analysis_results,
            "summary": self._generate_analysis_summary(analysis_results)
        }
        
        return self.generate_report(report_data, file_path=file_path, **options)
    
    def generate_metrics_report(
        self,
        metrics: Dict[str, Any],
        file_path: Optional[Union[str, Path]] = None,
        **options
    ) -> Union[str, None]:
        """
        Generate metrics report.
        
        Args:
            metrics: Metrics dictionary
            file_path: Optional output file path
            **options: Additional options
            
        Returns:
            Report string if file_path not provided, None otherwise
        """
        report_data = {
            "title": "Framework Metrics Report",
            "generated_at": datetime.now().isoformat(),
            "metrics": metrics,
            "summary": self._generate_metrics_summary(metrics)
        }
        
        return self.generate_report(report_data, file_path=file_path, **options)
    
    def _generate_markdown(self, data: Dict[str, Any], **options) -> str:
        """Generate Markdown report."""
        lines = []
        
        # Title
        title = data.get("title", "Report")
        lines.append(f"# {title}")
        lines.append("")
        
        # Generated at
        if "generated_at" in data:
            lines.append(f"**Generated:** {data['generated_at']}")
            lines.append("")
        
        # Summary
        if "summary" in data:
            lines.append("## Summary")
            lines.append("")
            summary = data["summary"]
            if isinstance(summary, dict):
                for key, value in summary.items():
                    lines.append(f"- **{key}**: {value}")
            else:
                lines.append(str(summary))
            lines.append("")
        
        # Metrics
        if "metrics" in data:
            lines.append("## Metrics")
            lines.append("")
            metrics = data["metrics"]
            self._add_dict_to_markdown(metrics, lines, level=2)
        
        # Analysis
        if "analysis" in data:
            lines.append("## Analysis")
            lines.append("")
            analysis = data["analysis"]
            self._add_dict_to_markdown(analysis, lines, level=2)
        
        return "\n".join(lines)
    
    def _generate_html(self, data: Dict[str, Any], **options) -> str:
        """Generate HTML report."""
        lines = ['<!DOCTYPE html>']
        lines.append('<html lang="en">')
        lines.append("<head>")
        lines.append('  <meta charset="UTF-8">')
        lines.append('  <meta name="viewport" content="width=device-width, initial-scale=1.0">')
        title = data.get("title", "Report")
        lines.append(f'  <title>{title}</title>')
        lines.append('  <style>')
        lines.append('    body { font-family: Arial, sans-serif; margin: 20px; }')
        lines.append('    h1 { color: #333; }')
        lines.append('    h2 { color: #666; margin-top: 30px; }')
        lines.append('    table { border-collapse: collapse; width: 100%; margin: 20px 0; }')
        lines.append('    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
        lines.append('    th { background-color: #f2f2f2; }')
        lines.append('  </style>')
        lines.append("</head>")
        lines.append("<body>")
        
        # Title
        title = data.get("title", "Report")
        lines.append(f"  <h1>{title}</h1>")
        
        # Generated at
        if "generated_at" in data:
            lines.append(f'  <p><strong>Generated:</strong> {data["generated_at"]}</p>')
        
        # Summary
        if "summary" in data:
            lines.append("  <h2>Summary</h2>")
            summary = data["summary"]
            if isinstance(summary, dict):
                lines.append("  <ul>")
                for key, value in summary.items():
                    lines.append(f"    <li><strong>{key}:</strong> {value}</li>")
                lines.append("  </ul>")
            else:
                lines.append(f"  <p>{summary}</p>")
        
        # Metrics
        if "metrics" in data:
            lines.append("  <h2>Metrics</h2>")
            metrics = data["metrics"]
            self._add_dict_to_html(metrics, lines)
        
        lines.append("</body>")
        lines.append("</html>")
        
        return "\n".join(lines)
    
    def _generate_json(self, data: Dict[str, Any], **options) -> str:
        """Generate JSON report."""
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def _generate_text(self, data: Dict[str, Any], **options) -> str:
        """Generate plain text report."""
        lines = []
        
        # Title
        title = data.get("title", "Report")
        lines.append("=" * 60)
        lines.append(title)
        lines.append("=" * 60)
        lines.append("")
        
        # Generated at
        if "generated_at" in data:
            lines.append(f"Generated: {data['generated_at']}")
            lines.append("")
        
        # Summary
        if "summary" in data:
            lines.append("SUMMARY")
            lines.append("-" * 60)
            summary = data["summary"]
            if isinstance(summary, dict):
                for key, value in summary.items():
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(str(summary))
            lines.append("")
        
        # Metrics
        if "metrics" in data:
            lines.append("METRICS")
            lines.append("-" * 60)
            metrics = data["metrics"]
            self._add_dict_to_text(metrics, lines, indent=2)
        
        return "\n".join(lines)
    
    def _add_dict_to_markdown(self, data: Dict[str, Any], lines: List[str], level: int = 2) -> None:
        """Add dictionary to markdown."""
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{'#' * level} {key}")
                lines.append("")
                self._add_dict_to_markdown(value, lines, level + 1)
            elif isinstance(value, list):
                lines.append(f"**{key}:**")
                for item in value:
                    if isinstance(item, dict):
                        lines.append("- " + ", ".join(f"{k}: {v}" for k, v in item.items()))
                    else:
                        lines.append(f"- {item}")
                lines.append("")
            else:
                lines.append(f"- **{key}**: {value}")
    
    def _add_dict_to_html(self, data: Dict[str, Any], lines: List[str]) -> None:
        """Add dictionary to HTML."""
        lines.append("  <table>")
        lines.append("    <tr><th>Key</th><th>Value</th></tr>")
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value_str = json.dumps(value, indent=2)
            else:
                value_str = str(value)
            
            lines.append(f"    <tr><td>{key}</td><td>{value_str}</td></tr>")
        
        lines.append("  </table>")
    
    def _add_dict_to_text(self, data: Dict[str, Any], lines: List[str], indent: int = 0) -> None:
        """Add dictionary to text."""
        prefix = " " * indent
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                self._add_dict_to_text(value, lines, indent + 2)
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        item_str = ", ".join(f"{k}={v}" for k, v in item.items())
                        lines.append(f"{prefix}  - {item_str}")
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
    
    def _generate_quality_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality summary."""
        score = metrics.get("score") or metrics.get("overall_score", 0.0)
        
        return {
            "Overall Score": f"{score:.2f}",
            "Status": "PASS" if score >= 0.7 else "FAIL",
            "Completeness": metrics.get("completeness", "N/A"),
            "Accuracy": metrics.get("accuracy", "N/A"),
            "Consistency": metrics.get("consistency", "N/A")
        }
    
    def _generate_analysis_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary."""
        return {
            "Analysis Type": analysis.get("type", "Unknown"),
            "Items Analyzed": analysis.get("total_count", 0),
            "Status": analysis.get("status", "Completed")
        }
    
    def _generate_metrics_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate metrics summary."""
        return {
            "Total Items": metrics.get("total_items", 0),
            "Success Rate": f"{metrics.get('success_rate', 0.0):.2%}",
            "Average Confidence": f"{metrics.get('average_confidence', 0.0):.2f}"
        }
