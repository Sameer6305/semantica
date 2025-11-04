"""
RDF Export Module

Handles export of data to RDF formats.

Key Features:
    - RDF serialization and export
    - Multiple RDF format support
    - Namespace management
    - RDF validation and quality checking
    - Batch RDF export processing

Main Classes:
    - RDFExporter: Main RDF export class
    - RDFSerializer: RDF serialization engine
    - RDFValidator: RDF validation engine
    - NamespaceManager: RDF namespace management
"""

from typing import Any, Dict, List, Optional, Set, Union
from pathlib import Path

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


class NamespaceManager:
    """
    RDF namespace management engine.
    
    • Manages RDF namespaces
    • Handles namespace declarations
    • Processes namespace conflicts
    • Manages namespace metadata
    """
    
    def __init__(self, **config):
        """
        Initialize namespace manager.
        
        • Setup namespace handling
        • Configure namespace resolution
        • Initialize conflict resolution
        • Setup metadata management
        """
        self.logger = get_logger("namespace_manager")
        self.namespaces: Dict[str, str] = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "semantica": "https://semantica.dev/ns#"
        }
        self.config = config or {}
    
    def extract_namespaces(self, rdf_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Extract namespaces from RDF data.
        
        • Identify namespace usage
        • Extract namespace declarations
        • Analyze namespace requirements
        • Return namespace information
        """
        extracted = {}
        
        # Check for @context in JSON-LD
        if "@context" in rdf_data:
            context = rdf_data["@context"]
            if isinstance(context, dict):
                for prefix, uri in context.items():
                    if not prefix.startswith("@"):
                        extracted[prefix] = uri
        
        return extracted
    
    def generate_namespace_declarations(self, namespaces: Dict[str, str], format: str = "turtle") -> str:
        """
        Generate namespace declarations.
        
        • Create namespace declarations
        • Handle namespace formatting
        • Optimize namespace usage
        • Return namespace declarations
        """
        declarations = []
        
        if format == "turtle":
            for prefix, uri in namespaces.items():
                declarations.append(f"@prefix {prefix}: <{uri}> .")
        elif format == "rdfxml":
            for prefix, uri in namespaces.items():
                declarations.append(f'xmlns:{prefix}="{uri}"')
        elif format == "jsonld":
            # JSON-LD uses @context
            return ""  # Handled separately
        
        return "\n".join(declarations)
    
    def resolve_namespace_conflicts(self, namespaces: Dict[str, str]) -> Dict[str, str]:
        """
        Resolve namespace conflicts.
        
        • Identify namespace conflicts
        • Apply conflict resolution
        • Handle namespace mapping
        • Return resolved namespaces
        """
        resolved = {}
        seen_uris = {}
        
        for prefix, uri in namespaces.items():
            if uri in seen_uris:
                # Conflict - use existing prefix or create new one
                existing_prefix = seen_uris[uri]
                resolved[prefix] = uri
                if prefix != existing_prefix:
                    self.logger.warning(f"Namespace conflict: {prefix} and {existing_prefix} map to {uri}")
            else:
                resolved[prefix] = uri
                seen_uris[uri] = prefix
        
        return resolved


class RDFSerializer:
    """
    RDF serialization engine.
    
    • Serializes RDF data to various formats
    • Handles format-specific serialization
    • Manages RDF encoding
    • Processes RDF metadata
    """
    
    def __init__(self, **config):
        """
        Initialize RDF serializer.
        
        • Setup serialization engines
        • Configure format handlers
        • Initialize encoding tools
        • Setup metadata processing
        """
        self.logger = get_logger("rdf_serializer")
        self.config = config or {}
        self.namespace_manager = NamespaceManager()
    
    def serialize_to_turtle(self, rdf_data: Dict[str, Any], **options) -> str:
        """
        Serialize RDF to Turtle format.
        
        • Convert RDF to Turtle syntax
        • Handle Turtle-specific formatting
        • Optimize Turtle output
        • Return Turtle serialization
        """
        lines = []
        
        # Generate namespace declarations
        namespaces = self.namespace_manager.extract_namespaces(rdf_data)
        if namespaces:
            ns_declarations = self.namespace_manager.generate_namespace_declarations(namespaces, "turtle")
            lines.append(ns_declarations)
            lines.append("")
        
        # Convert entities to triples
        entities = rdf_data.get("entities", [])
        for entity in entities:
            entity_id = entity.get("id") or f"semantica:entity_{hash(entity.get('text', ''))}"
            entity_type = entity.get("type", "semantica:Entity")
            text = entity.get("text") or entity.get("label", "")
            
            lines.append(f"<{entity_id}> a <{entity_type}> ;")
            lines.append(f'    semantica:text "{text}" ;')
            lines.append(f'    semantica:confidence {entity.get("confidence", 1.0)} .')
            lines.append("")
        
        # Convert relationships to triples
        relationships = rdf_data.get("relationships", [])
        for rel in relationships:
            source_id = rel.get("source_id") or rel.get("source")
            target_id = rel.get("target_id") or rel.get("target")
            rel_type = rel.get("type", "semantica:related_to")
            
            lines.append(f"<{source_id}> <{rel_type}> <{target_id}> .")
        
        return "\n".join(lines)
    
    def serialize_to_rdfxml(self, rdf_data: Dict[str, Any], **options) -> str:
        """
        Serialize RDF to RDF/XML format.
        
        • Convert RDF to RDF/XML syntax
        • Handle XML-specific formatting
        • Optimize XML structure
        • Return RDF/XML serialization
        """
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"')
        lines.append('         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"')
        lines.append('         xmlns:semantica="https://semantica.dev/ns#">')
        lines.append("")
        
        # Convert entities
        entities = rdf_data.get("entities", [])
        for entity in entities:
            entity_id = entity.get("id") or f"semantica:entity_{hash(entity.get('text', ''))}"
            entity_type = entity.get("type", "semantica:Entity")
            text = entity.get("text") or entity.get("label", "")
            
            lines.append(f'  <rdf:Description rdf:about="{entity_id}">')
            lines.append(f'    <rdf:type rdf:resource="{entity_type}"/>')
            lines.append(f'    <semantica:text>{text}</semantica:text>')
            lines.append(f'    <semantica:confidence>{entity.get("confidence", 1.0)}</semantica:confidence>')
            lines.append("  </rdf:Description>")
            lines.append("")
        
        # Convert relationships
        relationships = rdf_data.get("relationships", [])
        for rel in relationships:
            source_id = rel.get("source_id") or rel.get("source")
            target_id = rel.get("target_id") or rel.get("target")
            rel_type = rel.get("type", "semantica:related_to")
            
            lines.append(f'  <rdf:Description rdf:about="{source_id}">')
            lines.append(f'    <{rel_type} rdf:resource="{target_id}"/>')
            lines.append("  </rdf:Description>")
            lines.append("")
        
        lines.append("</rdf:RDF>")
        return "\n".join(lines)
    
    def serialize_to_jsonld(self, rdf_data: Dict[str, Any], **options) -> str:
        """
        Serialize RDF to JSON-LD format.
        
        • Convert RDF to JSON-LD syntax
        • Handle JSON-specific formatting
        • Optimize JSON structure
        • Return JSON-LD serialization
        """
        import json
        
        jsonld = {
            "@context": {
                "@vocab": "https://semantica.dev/vocab/",
                "semantica": "https://semantica.dev/ns#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            "@graph": []
        }
        
        # Convert entities
        entities = rdf_data.get("entities", [])
        for entity in entities:
            jsonld["@graph"].append({
                "@id": entity.get("id") or f"semantica:entity/{entity.get('text', '')}",
                "@type": entity.get("type", "semantica:Entity"),
                "semantica:text": entity.get("text") or entity.get("label", ""),
                "semantica:confidence": entity.get("confidence", 1.0)
            })
        
        # Convert relationships
        relationships = rdf_data.get("relationships", [])
        for rel in relationships:
            jsonld["@graph"].append({
                "@id": rel.get("id") or f"semantica:rel/{rel.get('source_id')}_{rel.get('target_id')}",
                "@type": "semantica:Relationship",
                "semantica:source": {"@id": rel.get("source_id")},
                "semantica:target": {"@id": rel.get("target_id")},
                "semantica:type": rel.get("type", "related_to")
            })
        
        return json.dumps(jsonld, indent=2, ensure_ascii=False)


class RDFValidator:
    """
    RDF validation engine.
    
    • Validates RDF data quality
    • Checks RDF syntax and structure
    • Validates namespace usage
    • Handles RDF consistency checking
    """
    
    def __init__(self, **config):
        """
        Initialize RDF validator.
        
        • Setup validation rules
        • Configure syntax checkers
        • Initialize consistency checkers
        • Setup quality assessment
        """
        self.logger = get_logger("rdf_validator")
        self.config = config or {}
    
    def validate_rdf_syntax(self, rdf_data: Dict[str, Any], format: str = "turtle") -> Dict[str, Any]:
        """
        Validate RDF syntax for specified format.
        
        • Check syntax correctness
        • Validate format-specific rules
        • Handle syntax errors
        • Return validation results
        """
        errors = []
        warnings = []
        
        # Basic structure validation
        if not isinstance(rdf_data, dict):
            errors.append("RDF data must be a dictionary")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Check for required fields
        if "entities" not in rdf_data and "relationships" not in rdf_data:
            warnings.append("No entities or relationships found in RDF data")
        
        # Validate entities
        entities = rdf_data.get("entities", [])
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                errors.append(f"Entity {i} is not a dictionary")
                continue
            
            if "id" not in entity and "text" not in entity:
                warnings.append(f"Entity {i} missing 'id' or 'text'")
        
        # Validate relationships
        relationships = rdf_data.get("relationships", [])
        for i, rel in enumerate(relationships):
            if not isinstance(rel, dict):
                errors.append(f"Relationship {i} is not a dictionary")
                continue
            
            if "source_id" not in rel and "target_id" not in rel:
                errors.append(f"Relationship {i} missing 'source_id' or 'target_id'")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def validate_namespace_usage(self, rdf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate RDF namespace usage.
        
        • Check namespace declarations
        • Validate namespace references
        • Handle namespace conflicts
        • Return namespace validation
        """
        issues = []
        
        # Check for @context in JSON-LD
        if "@context" in rdf_data:
            context = rdf_data["@context"]
            if not isinstance(context, dict):
                issues.append("@context must be a dictionary")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def check_rdf_consistency(self, rdf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check RDF consistency and coherence.
        
        • Analyze RDF structure
        • Check logical consistency
        • Validate RDF relationships
        • Return consistency report
        """
        issues = []
        
        # Check relationship references
        relationships = rdf_data.get("relationships", [])
        entities = rdf_data.get("entities", [])
        entity_ids = {e.get("id") for e in entities if e.get("id")}
        
        for rel in relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            
            if source_id and source_id not in entity_ids:
                issues.append(f"Relationship references non-existent source entity: {source_id}")
            
            if target_id and target_id not in entity_ids:
                issues.append(f"Relationship references non-existent target entity: {target_id}")
        
        return {
            "consistent": len(issues) == 0,
            "issues": issues
        }


class RDFExporter:
    """
    RDF export and serialization handler.
    
    • Exports data to RDF formats
    • Handles RDF serialization
    • Manages RDF namespaces
    • Validates RDF output
    • Supports multiple RDF formats
    • Handles batch RDF export
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize RDF exporter.
        
        • Setup RDF serialization
        • Configure export formats
        • Initialize validation tools
        • Setup namespace management
        • Configure batch processing
        """
        self.logger = get_logger("rdf_exporter")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.serializer = RDFSerializer()
        self.validator = RDFValidator()
        self.namespace_manager = NamespaceManager()
        
        self.supported_formats = ["turtle", "rdfxml", "jsonld", "ntriples", "n3"]
    
    def export_to_rdf(self, data: Dict[str, Any], format: str = "turtle", **options) -> str:
        """
        Export data to RDF format.
        
        • Convert data to RDF
        • Apply format-specific serialization
        • Handle namespace declarations
        • Validate RDF output
        • Return RDF export
        """
        if format not in self.supported_formats:
            raise ValidationError(f"Unsupported RDF format: {format}")
        
        # Validate input
        validation = self.validator.validate_rdf_syntax(data, format)
        if not validation["valid"]:
            self.logger.warning(f"RDF validation issues: {validation['errors']}")
        
        # Serialize
        if format == "turtle":
            return self.serializer.serialize_to_turtle(data, **options)
        elif format == "rdfxml":
            return self.serializer.serialize_to_rdfxml(data, **options)
        elif format == "jsonld":
            return self.serializer.serialize_to_jsonld(data, **options)
        else:
            raise ValidationError(f"Format {format} not yet implemented")
    
    def export(
        self,
        data: Dict[str, Any],
        file_path: Union[str, Path],
        format: str = "turtle",
        **options
    ) -> None:
        """
        Export data to RDF file.
        
        Args:
            data: Data to export
            file_path: Output file path
            format: RDF format ('turtle', 'rdfxml', 'jsonld')
            **options: Additional options
        """
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        rdf_content = self.export_to_rdf(data, format=format, **options)
        
        encoding = options.get("encoding", "utf-8")
        
        with open(file_path, "w", encoding=encoding) as f:
            f.write(rdf_content)
        
        self.logger.info(f"Exported RDF ({format}) to: {file_path}")
    
    def serialize_rdf(self, rdf_data: Dict[str, Any], format: str = "turtle", **options) -> str:
        """
        Serialize RDF data to specified format.
        
        • Convert RDF to serialized format
        • Apply format-specific rules
        • Handle encoding and formatting
        • Return serialized RDF
        """
        return self.export_to_rdf(rdf_data, format=format, **options)
    
    def validate_rdf(self, rdf_data: Dict[str, Any], **options) -> Dict[str, Any]:
        """
        Validate RDF data quality and structure.
        
        • Check RDF syntax and structure
        • Validate namespace usage
        • Check RDF consistency
        • Return validation results
        """
        syntax_validation = self.validator.validate_rdf_syntax(rdf_data)
        namespace_validation = self.validator.validate_namespace_usage(rdf_data)
        consistency_check = self.validator.check_rdf_consistency(rdf_data)
        
        return {
            "syntax": syntax_validation,
            "namespaces": namespace_validation,
            "consistency": consistency_check,
            "overall_valid": (
                syntax_validation["valid"] and
                namespace_validation["valid"] and
                consistency_check["consistent"]
            )
        }
    
    def manage_namespaces(self, rdf_data: Dict[str, Any], **namespaces: str) -> Dict[str, Any]:
        """
        Manage RDF namespaces and declarations.
        
        • Extract namespace requirements
        • Generate namespace declarations
        • Handle namespace conflicts
        • Return namespace management results
        """
        # Extract existing namespaces
        extracted = self.namespace_manager.extract_namespaces(rdf_data)
        
        # Merge with provided namespaces
        all_namespaces = {**extracted, **namespaces}
        
        # Resolve conflicts
        resolved = self.namespace_manager.resolve_namespace_conflicts(all_namespaces)
        
        return {
            "namespaces": resolved,
            "declarations": self.namespace_manager.generate_namespace_declarations(resolved, "turtle")
        }
