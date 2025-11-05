"""
Ontology Documentation Manager

Provides comprehensive ontology documentation including metadata,
descriptions, version information, and contributor tracking.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


@dataclass
class OntologyDocumentation:
    """Ontology documentation structure."""
    name: str
    description: str
    purpose: str
    scope: str
    authors: List[str] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)
    created_at: str = ""
    version: str = "1.0"
    license: str = ""
    namespace: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class OntologyDocumentationManager:
    """
    Ontology documentation management system.
    
    • Ontology metadata management (description, purpose, scope)
    • Author and contributor tracking
    • Creation date and version tracking
    • License and IRI namespace management
    • Documentation generation and export
    • Documentation validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize documentation manager.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("ontology_documentation_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.documentation: Dict[str, OntologyDocumentation] = {}
    
    def create_documentation(
        self,
        name: str,
        description: str,
        purpose: str,
        scope: str,
        **options
    ) -> OntologyDocumentation:
        """
        Create ontology documentation.
        
        Args:
            name: Ontology name
            description: Description
            purpose: Purpose statement
            scope: Scope description
            **options: Additional options:
                - authors: List of authors
                - contributors: List of contributors
                - version: Version string
                - license: License information
                - namespace: Namespace URI
        
        Returns:
            Created documentation
        """
        doc = OntologyDocumentation(
            name=name,
            description=description,
            purpose=purpose,
            scope=scope,
            authors=options.get("authors", []),
            contributors=options.get("contributors", []),
            created_at=datetime.now().isoformat(),
            version=options.get("version", "1.0"),
            license=options.get("license", ""),
            namespace=options.get("namespace", ""),
            metadata=options.get("metadata", {})
        )
        
        self.documentation[name] = doc
        self.logger.info(f"Created documentation for: {name}")
        
        return doc
    
    def add_author(self, ontology_name: str, author: str) -> None:
        """Add author to documentation."""
        if ontology_name not in self.documentation:
            raise ValidationError(f"Documentation not found: {ontology_name}")
        
        doc = self.documentation[ontology_name]
        if author not in doc.authors:
            doc.authors.append(author)
    
    def add_contributor(self, ontology_name: str, contributor: str) -> None:
        """Add contributor to documentation."""
        if ontology_name not in self.documentation:
            raise ValidationError(f"Documentation not found: {ontology_name}")
        
        doc = self.documentation[ontology_name]
        if contributor not in doc.contributors:
            doc.contributors.append(contributor)
    
    def generate_markdown(
        self,
        ontology_name: str,
        ontology: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate Markdown documentation.
        
        Args:
            ontology_name: Ontology name
            ontology: Optional ontology dictionary
        
        Returns:
            Markdown documentation string
        """
        if ontology_name not in self.documentation:
            raise ValidationError(f"Documentation not found: {ontology_name}")
        
        doc = self.documentation[ontology_name]
        
        lines = []
        lines.append(f"# {doc.name}")
        lines.append("")
        lines.append(f"**Version:** {doc.version}")
        lines.append(f"**Namespace:** {doc.namespace}")
        lines.append("")
        
        lines.append("## Description")
        lines.append("")
        lines.append(doc.description)
        lines.append("")
        
        lines.append("## Purpose")
        lines.append("")
        lines.append(doc.purpose)
        lines.append("")
        
        lines.append("## Scope")
        lines.append("")
        lines.append(doc.scope)
        lines.append("")
        
        if doc.authors:
            lines.append("## Authors")
            lines.append("")
            for author in doc.authors:
                lines.append(f"- {author}")
            lines.append("")
        
        if doc.contributors:
            lines.append("## Contributors")
            lines.append("")
            for contributor in doc.contributors:
                lines.append(f"- {contributor}")
            lines.append("")
        
        if doc.license:
            lines.append("## License")
            lines.append("")
            lines.append(doc.license)
            lines.append("")
        
        if ontology:
            classes = ontology.get("classes", [])
            properties = ontology.get("properties", [])
            
            lines.append("## Classes")
            lines.append("")
            for cls in classes:
                lines.append(f"### {cls.get('name', 'Unknown')}")
                if cls.get("comment"):
                    lines.append(f"{cls['comment']}")
                lines.append("")
            
            lines.append("## Properties")
            lines.append("")
            for prop in properties:
                lines.append(f"### {prop.get('name', 'Unknown')}")
                lines.append(f"**Type:** {prop.get('type', 'unknown')}")
                if prop.get("comment"):
                    lines.append(f"{prop['comment']}")
                lines.append("")
        
        return "\n".join(lines)
    
    def export_documentation(
        self,
        ontology_name: str,
        file_path: Union[str, Path],
        format: str = "markdown",
        ontology: Optional[Dict[str, Any]] = None,
        **options
    ) -> None:
        """
        Export documentation to file.
        
        Args:
            ontology_name: Ontology name
            file_path: Output file path
            format: Export format ('markdown', 'html')
            ontology: Optional ontology dictionary
            **options: Additional options
        """
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        if format == "markdown":
            content = self.generate_markdown(ontology_name, ontology)
        else:
            raise ValidationError(f"Unsupported format: {format}")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        self.logger.info(f"Exported documentation to: {file_path}")
    
    def validate_documentation(self, ontology_name: str) -> Dict[str, Any]:
        """
        Validate documentation completeness.
        
        Args:
            ontology_name: Ontology name
        
        Returns:
            Validation results
        """
        if ontology_name not in self.documentation:
            raise ValidationError(f"Documentation not found: {ontology_name}")
        
        doc = self.documentation[ontology_name]
        errors = []
        warnings = []
        
        if not doc.description:
            errors.append("Documentation missing description")
        if not doc.purpose:
            errors.append("Documentation missing purpose")
        if not doc.scope:
            errors.append("Documentation missing scope")
        if not doc.authors:
            warnings.append("Documentation has no authors")
        if not doc.namespace:
            warnings.append("Documentation missing namespace")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def get_documentation(self, ontology_name: str) -> Optional[OntologyDocumentation]:
        """Get documentation by name."""
        return self.documentation.get(ontology_name)
