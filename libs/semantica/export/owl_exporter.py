"""
OWL Exporter for Semantic Modeling

Exports ontologies and knowledge graphs to OWL (Web Ontology Language) format
for semantic modeling and ontology representation.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


class OWLExporter:
    """
    OWL exporter for semantic modeling.
    
    • Exports ontologies to OWL/OWL-XML format
    • Handles class definitions and hierarchies
    • Manages object and data properties
    • Supports OWL 2.0 features
    • Generates OWL-XML serialization
    • Validates OWL structure
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize OWL exporter.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - ontology_uri: Base ontology URI
                - version: Ontology version
                - format: Export format ('owl-xml', 'turtle')
        """
        self.logger = get_logger("owl_exporter")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.ontology_uri = self.config.get("ontology_uri", "https://semantica.dev/ontology/")
        self.version = self.config.get("version", "1.0")
        self.format = self.config.get("format", "owl-xml")
    
    def export(
        self,
        ontology: Dict[str, Any],
        file_path: Union[str, Path],
        format: Optional[str] = None,
        **options
    ) -> None:
        """
        Export ontology to OWL format.
        
        Args:
            ontology: Ontology dictionary with classes, properties, etc.
            file_path: Output file path
            format: Export format ('owl-xml', 'turtle')
            **options: Additional options
        """
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        export_format = format or self.format
        
        if export_format == "owl-xml":
            owl_content = self._export_owl_xml(ontology, **options)
        elif export_format == "turtle":
            owl_content = self._export_owl_turtle(ontology, **options)
        else:
            raise ValidationError(f"Unsupported OWL format: {export_format}")
        
        encoding = options.get("encoding", "utf-8")
        with open(file_path, "w", encoding=encoding) as f:
            f.write(owl_content)
        
        self.logger.info(f"Exported OWL ({export_format}) to: {file_path}")
    
    def export_ontology(
        self,
        ontology: Dict[str, Any],
        file_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export complete ontology to OWL.
        
        Args:
            ontology: Ontology dictionary
            file_path: Output file path
            **options: Additional options
        """
        self.export(ontology, file_path, **options)
    
    def _export_owl_xml(self, ontology: Dict[str, Any], **options) -> str:
        """Export to OWL-XML format."""
        ontology_uri = ontology.get("uri") or self.ontology_uri
        ontology_name = ontology.get("name", "SemanticaOntology")
        version = ontology.get("version") or self.version
        
        lines = ['<?xml version="1.0"?>']
        lines.append('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"')
        lines.append('         xmlns:owl="http://www.w3.org/2002/07/owl#"')
        lines.append('         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"')
        lines.append('         xmlns:xsd="http://www.w3.org/2001/XMLSchema#">')
        lines.append("")
        
        # Ontology declaration
        lines.append(f'  <owl:Ontology rdf:about="{ontology_uri}">')
        lines.append(f'    <rdfs:label>{ontology_name}</rdfs:label>')
        lines.append(f'    <owl:versionInfo>{version}</owl:versionInfo>')
        if ontology.get("description"):
            lines.append(f'    <rdfs:comment>{ontology.get("description")}</rdfs:comment>')
        lines.append("  </owl:Ontology>")
        lines.append("")
        
        # Classes
        classes = ontology.get("classes", [])
        for cls in classes:
            class_uri = cls.get("uri") or cls.get("id", "")
            class_name = cls.get("name") or cls.get("label", "")
            
            lines.append(f'  <owl:Class rdf:about="{class_uri}">')
            lines.append(f'    <rdfs:label>{class_name}</rdfs:label>')
            
            if cls.get("comment"):
                lines.append(f'    <rdfs:comment>{cls.get("comment")}</rdfs:comment>')
            
            # Subclass relationships
            if cls.get("subClassOf"):
                parent = cls.get("subClassOf")
                lines.append(f'    <rdfs:subClassOf rdf:resource="{parent}"/>')
            
            # Equivalent classes
            if cls.get("equivalentClass"):
                equiv = cls.get("equivalentClass")
                lines.append(f'    <owl:equivalentClass rdf:resource="{equiv}"/>')
            
            lines.append("  </owl:Class>")
            lines.append("")
        
        # Object properties
        object_properties = ontology.get("object_properties", [])
        for prop in object_properties:
            prop_uri = prop.get("uri") or prop.get("id", "")
            prop_name = prop.get("name") or prop.get("label", "")
            
            lines.append(f'  <owl:ObjectProperty rdf:about="{prop_uri}">')
            lines.append(f'    <rdfs:label>{prop_name}</rdfs:label>')
            
            if prop.get("comment"):
                lines.append(f'    <rdfs:comment>{prop.get("comment")}</rdfs:comment>')
            
            # Domain
            if prop.get("domain"):
                domain = prop.get("domain")
                if isinstance(domain, list):
                    for d in domain:
                        lines.append(f'    <rdfs:domain rdf:resource="{d}"/>')
                else:
                    lines.append(f'    <rdfs:domain rdf:resource="{domain}"/>')
            
            # Range
            if prop.get("range"):
                range_val = prop.get("range")
                if isinstance(range_val, list):
                    for r in range_val:
                        lines.append(f'    <rdfs:range rdf:resource="{r}"/>')
                else:
                    lines.append(f'    <rdfs:range rdf:resource="{range_val}"/>')
            
            lines.append("  </owl:ObjectProperty>")
            lines.append("")
        
        # Data properties
        data_properties = ontology.get("data_properties", [])
        for prop in data_properties:
            prop_uri = prop.get("uri") or prop.get("id", "")
            prop_name = prop.get("name") or prop.get("label", "")
            
            lines.append(f'  <owl:DatatypeProperty rdf:about="{prop_uri}">')
            lines.append(f'    <rdfs:label>{prop_name}</rdfs:label>')
            
            if prop.get("comment"):
                lines.append(f'    <rdfs:comment>{prop.get("comment")}</rdfs:comment>')
            
            # Domain
            if prop.get("domain"):
                domain = prop.get("domain")
                lines.append(f'    <rdfs:domain rdf:resource="{domain}"/>')
            
            # Range
            if prop.get("range"):
                range_type = prop.get("range", "xsd:string")
                lines.append(f'    <rdfs:range rdf:resource="http://www.w3.org/2001/XMLSchema#{range_type}"/>')
            
            lines.append("  </owl:DatatypeProperty>")
            lines.append("")
        
        lines.append("</rdf:RDF>")
        return "\n".join(lines)
    
    def _export_owl_turtle(self, ontology: Dict[str, Any], **options) -> str:
        """Export to OWL Turtle format."""
        ontology_uri = ontology.get("uri") or self.ontology_uri
        ontology_name = ontology.get("name", "SemanticaOntology")
        version = ontology.get("version") or self.version
        
        lines = []
        
        # Namespace declarations
        lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
        lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
        lines.append("@prefix owl: <http://www.w3.org/2002/07/owl#> .")
        lines.append("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
        lines.append(f"@prefix ont: <{ontology_uri}> .")
        lines.append("")
        
        # Ontology declaration
        lines.append(f"<{ontology_uri}> a owl:Ontology ;")
        lines.append(f'    rdfs:label "{ontology_name}" ;')
        lines.append(f'    owl:versionInfo "{version}" .')
        if ontology.get("description"):
            lines.append(f'    rdfs:comment "{ontology.get("description")}" ;')
        lines.append("")
        
        # Classes
        classes = ontology.get("classes", [])
        for cls in classes:
            class_uri = cls.get("uri") or cls.get("id", "")
            class_name = cls.get("name") or cls.get("label", "")
            
            lines.append(f"<{class_uri}> a owl:Class ;")
            lines.append(f'    rdfs:label "{class_name}" .')
            
            if cls.get("comment"):
                lines.append(f'    rdfs:comment "{cls.get("comment")}" ;')
            
            if cls.get("subClassOf"):
                parent = cls.get("subClassOf")
                lines.append(f'    rdfs:subClassOf <{parent}> ;')
            
            # Remove trailing semicolon and add period
            if lines[-1].endswith(" ;"):
                lines[-1] = lines[-1].rstrip(" ;") + " ."
            else:
                lines.append(" .")
            lines.append("")
        
        # Object properties
        object_properties = ontology.get("object_properties", [])
        for prop in object_properties:
            prop_uri = prop.get("uri") or prop.get("id", "")
            prop_name = prop.get("name") or prop.get("label", "")
            
            lines.append(f"<{prop_uri}> a owl:ObjectProperty ;")
            lines.append(f'    rdfs:label "{prop_name}" .')
            
            if prop.get("domain"):
                domain = prop.get("domain")
                if isinstance(domain, list):
                    for d in domain:
                        lines.append(f'    rdfs:domain <{d}> ;')
                else:
                    lines.append(f'    rdfs:domain <{domain}> ;')
            
            if prop.get("range"):
                range_val = prop.get("range")
                if isinstance(range_val, list):
                    for r in range_val:
                        lines.append(f'    rdfs:range <{r}> ;')
                else:
                    lines.append(f'    rdfs:range <{range_val}> ;')
            
            if lines[-1].endswith(" ;"):
                lines[-1] = lines[-1].rstrip(" ;") + " ."
            lines.append("")
        
        return "\n".join(lines)
    
    def export_classes(
        self,
        classes: List[Dict[str, Any]],
        file_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export classes to OWL format.
        
        Args:
            classes: List of class definitions
            file_path: Output file path
            **options: Additional options
        """
        ontology = {
            "classes": classes,
            "uri": options.get("ontology_uri", self.ontology_uri),
            "name": options.get("ontology_name", "SemanticaOntology")
        }
        self.export(ontology, file_path, **options)
    
    def export_properties(
        self,
        properties: List[Dict[str, Any]],
        file_path: Union[str, Path],
        property_type: str = "object",
        **options
    ) -> None:
        """
        Export properties to OWL format.
        
        Args:
            properties: List of property definitions
            file_path: Output file path
            property_type: Property type ('object', 'data')
            **options: Additional options
        """
        ontology = {
            "uri": options.get("ontology_uri", self.ontology_uri),
            "name": options.get("ontology_name", "SemanticaOntology")
        }
        
        if property_type == "object":
            ontology["object_properties"] = properties
        else:
            ontology["data_properties"] = properties
        
        self.export(ontology, file_path, **options)

