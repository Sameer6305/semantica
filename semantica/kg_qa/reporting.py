"""
Quality Reporting Module

Generates quality reports and tracks issues.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.logging import get_logger


@dataclass
class QualityIssue:
    """Quality issue representation."""
    
    id: str
    type: str
    severity: str
    description: str
    entity_id: Optional[str] = None
    relationship_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Quality report representation."""
    
    timestamp: datetime
    overall_score: float
    completeness_score: float
    consistency_score: float
    issues: List[QualityIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QualityReporter:
    """
    Quality reporter.
    
    Generates quality reports for Knowledge Graphs.
    """
    
    def __init__(self, **kwargs):
        """Initialize quality reporter."""
        self.logger = get_logger("quality_reporter")
        self.config = kwargs
    
    def generate_report(
        self,
        knowledge_graph: Any,
        quality_metrics: Dict[str, float]
    ) -> QualityReport:
        """
        Generate quality report.
        
        Args:
            knowledge_graph: Knowledge graph instance
            quality_metrics: Quality metrics dictionary
            
        Returns:
            Quality report
        """
        issues = self._identify_issues(knowledge_graph, quality_metrics)
        recommendations = self._generate_recommendations(issues)
        
        report = QualityReport(
            timestamp=datetime.now(),
            overall_score=quality_metrics.get("overall", 0.0),
            completeness_score=quality_metrics.get("completeness", 0.0),
            consistency_score=quality_metrics.get("consistency", 0.0),
            issues=issues,
            recommendations=recommendations
        )
        
        return report
    
    def export_report(
        self,
        report: QualityReport,
        format: str = "json"
    ) -> str:
        """
        Export report to specified format.
        
        Args:
            report: Quality report
            format: Export format (json, yaml, html)
            
        Returns:
            Exported report string
        """
        if format == "json":
            import json
            return json.dumps({
                "timestamp": report.timestamp.isoformat(),
                "overall_score": report.overall_score,
                "completeness_score": report.completeness_score,
                "consistency_score": report.consistency_score,
                "issues": [
                    {
                        "id": issue.id,
                        "type": issue.type,
                        "severity": issue.severity,
                        "description": issue.description
                    }
                    for issue in report.issues
                ],
                "recommendations": report.recommendations
            }, indent=2)
        
        elif format == "yaml":
            import yaml
            return yaml.dump({
                "timestamp": report.timestamp.isoformat(),
                "overall_score": report.overall_score,
                "issues": [
                    {
                        "id": issue.id,
                        "type": issue.type,
                        "description": issue.description
                    }
                    for issue in report.issues
                ]
            })
        
        else:
            return str(report)
    
    def _identify_issues(
        self,
        knowledge_graph: Any,
        metrics: Dict[str, float]
    ) -> List[QualityIssue]:
        """Identify quality issues."""
        issues = []
        
        # Check for low scores
        if metrics.get("overall", 1.0) < 0.7:
            issues.append(QualityIssue(
                id="low_overall_score",
                type="quality",
                severity="high",
                description="Overall quality score is below threshold"
            ))
        
        if metrics.get("completeness", 1.0) < 0.8:
            issues.append(QualityIssue(
                id="low_completeness",
                type="completeness",
                severity="medium",
                description="Completeness score is below threshold"
            ))
        
        return issues
    
    def _generate_recommendations(
        self,
        issues: List[QualityIssue]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        for issue in issues:
            if issue.type == "completeness":
                recommendations.append(
                    "Add missing required properties to entities"
                )
            elif issue.type == "consistency":
                recommendations.append(
                    "Resolve consistency violations in the knowledge graph"
                )
        
        return recommendations


class IssueTracker:
    """
    Issue tracker.
    
    Tracks and manages quality issues.
    """
    
    def __init__(self, **kwargs):
        """Initialize issue tracker."""
        self.logger = get_logger("issue_tracker")
        self.config = kwargs
        self.issues: Dict[str, QualityIssue] = {}
    
    def add_issue(self, issue: QualityIssue) -> None:
        """
        Add an issue.
        
        Args:
            issue: Quality issue
        """
        self.issues[issue.id] = issue
    
    def get_issue(self, issue_id: str) -> Optional[QualityIssue]:
        """
        Get issue by ID.
        
        Args:
            issue_id: Issue ID
            
        Returns:
            Quality issue or None
        """
        return self.issues.get(issue_id)
    
    def list_issues(
        self,
        severity: Optional[str] = None
    ) -> List[QualityIssue]:
        """
        List issues, optionally filtered by severity.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of issues
        """
        issues = list(self.issues.values())
        
        if severity:
            issues = [i for i in issues if i.severity == severity]
        
        return issues
    
    def resolve_issue(self, issue_id: str) -> bool:
        """
        Mark issue as resolved.
        
        Args:
            issue_id: Issue ID
            
        Returns:
            True if resolved, False otherwise
        """
        if issue_id in self.issues:
            del self.issues[issue_id]
            return True
        return False


class ImprovementSuggestions:
    """
    Improvement suggestions generator.
    
    Generates suggestions for improving Knowledge Graph quality.
    """
    
    def __init__(self, **kwargs):
        """Initialize improvement suggestions generator."""
        self.logger = get_logger("improvement_suggestions")
        self.config = kwargs
    
    def generate_suggestions(
        self,
        quality_report: QualityReport
    ) -> List[str]:
        """
        Generate improvement suggestions.
        
        Args:
            quality_report: Quality report
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Based on issues
        for issue in quality_report.issues:
            if issue.type == "completeness":
                suggestions.append(
                    f"Improve completeness for {issue.description}"
                )
            elif issue.type == "consistency":
                suggestions.append(
                    f"Resolve consistency issue: {issue.description}"
                )
        
        # Based on scores
        if quality_report.overall_score < 0.7:
            suggestions.append("Overall quality needs improvement")
        
        if quality_report.completeness_score < 0.8:
            suggestions.append("Add missing required properties")
        
        return suggestions

