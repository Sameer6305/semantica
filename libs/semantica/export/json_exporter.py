"""
JSON exporter for Semantica framework.

This module provides JSON and JSON-LD export capabilities
for knowledge graphs and semantic data.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory, write_json_file


class JSONExporter:
    """
    JSON exporter for knowledge graphs and semantic data.
    
    • JSON and JSON-LD export functionality
    • Knowledge graph serialization
    • Metadata and provenance export
    • Performance optimization
    • Error handling and recovery
    • Advanced JSON features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize JSON exporter.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - indent: JSON indentation (default: 2)
                - ensure_ascii: Ensure ASCII encoding (default: False)
                - format: Export format ('json', 'json-ld')
        """
        self.logger = get_logger("json_exporter")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.indent = self.config.get("indent", 2)
        self.ensure_ascii = self.config.get("ensure_ascii", False)
        self.format = self.config.get("format", "json")
    
    def export(
        self,
        data: Any,
        file_path: Union[str, Path],
        format: Optional[str] = None,
        **options
    ) -> None:
        """
        Export data to JSON file.
        
        Args:
            data: Data to export
            file_path: Output file path
            format: Export format ('json', 'json-ld')
            **options: Additional options:
                - include_metadata: Include metadata (default: True)
                - include_provenance: Include provenance (default: True)
        """
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        export_format = format or self.format
        
        if export_format == "json-ld":
            json_data = self._convert_to_jsonld(data, **options)
        else:
            json_data = self._convert_to_json(data, **options)
        
        write_json_file(
            json_data,
            file_path,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii
        )
        
        self.logger.info(f"Exported JSON to: {file_path}")
    
    def export_knowledge_graph(
        self,
        knowledge_graph: Dict[str, Any],
        file_path: Union[str, Path],
        format: Optional[str] = None,
        **options
    ) -> None:
        """
        Export knowledge graph to JSON/JSON-LD.
        
        Args:
            knowledge_graph: Knowledge graph dictionary
            file_path: Output file path
            format: Export format ('json', 'json-ld')
            **options: Additional options
        """
        export_format = format or self.format
        
        if export_format == "json-ld":
            json_data = self._convert_kg_to_jsonld(knowledge_graph, **options)
        else:
            json_data = self._convert_kg_to_json(knowledge_graph, **options)
        
        self.export(json_data, file_path, format=export_format, **options)
    
    def export_entities(
        self,
        entities: List[Dict[str, Any]],
        file_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export entities to JSON.
        
        Args:
            entities: List of entity dictionaries
            file_path: Output file path
            **options: Additional options
        """
        json_data = {
            "@context": {
                "@vocab": "https://semantica.dev/vocab/",
                "entities": {
                    "@id": "semantica:entities",
                    "@container": "@list"
                }
            },
            "entities": entities,
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "entity_count": len(entities),
                **options.get("metadata", {})
            }
        }
        
        self.export(json_data, file_path, **options)
    
    def export_relationships(
        self,
        relationships: List[Dict[str, Any]],
        file_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export relationships to JSON.
        
        Args:
            relationships: List of relationship dictionaries
            file_path: Output file path
            **options: Additional options
        """
        json_data = {
            "@context": {
                "@vocab": "https://semantica.dev/vocab/",
                "relationships": {
                    "@id": "semantica:relationships",
                    "@container": "@list"
                }
            },
            "relationships": relationships,
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "relationship_count": len(relationships),
                **options.get("metadata", {})
            }
        }
        
        self.export(json_data, file_path, **options)
    
    def _convert_to_json(self, data: Any, **options) -> Dict[str, Any]:
        """Convert data to JSON format."""
        if isinstance(data, dict):
            result = dict(data)
            
            # Add metadata if requested
            if options.get("include_metadata", True):
                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["exported_at"] = datetime.now().isoformat()
            
            return result
        elif isinstance(data, list):
            return {
                "data": data,
                "count": len(data),
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    **options.get("metadata", {})
                }
            }
        else:
            return {"value": data}
    
    def _convert_to_jsonld(self, data: Any, **options) -> Dict[str, Any]:
        """Convert data to JSON-LD format."""
        jsonld = {
            "@context": {
                "@vocab": "https://semantica.dev/vocab/",
                "semantica": "https://semantica.dev/ns#"
            }
        }
        
        if isinstance(data, dict):
            if "entities" in data or "relationships" in data:
                # Knowledge graph structure
                jsonld.update(self._convert_kg_to_jsonld(data, **options))
            else:
                # Generic data
                jsonld["@graph"] = [data]
        elif isinstance(data, list):
            jsonld["@graph"] = data
        else:
            jsonld["@value"] = data
        
        # Add metadata
        jsonld["@id"] = f"https://semantica.dev/data/{datetime.now().isoformat()}"
        jsonld["semantica:exportedAt"] = datetime.now().isoformat()
        
        return jsonld
    
    def _convert_kg_to_json(self, kg: Dict[str, Any], **options) -> Dict[str, Any]:
        """Convert knowledge graph to JSON."""
        result = {
            "entities": kg.get("entities", []),
            "relationships": kg.get("relationships", []),
            "nodes": kg.get("nodes", []),
            "edges": kg.get("edges", []),
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                **kg.get("metadata", {}),
                **options.get("metadata", {})
            }
        }
        
        # Add statistics if available
        if "statistics" in kg:
            result["statistics"] = kg["statistics"]
        
        return result
    
    def _convert_kg_to_jsonld(self, kg: Dict[str, Any], **options) -> Dict[str, Any]:
        """Convert knowledge graph to JSON-LD."""
        jsonld = {
            "@context": {
                "@vocab": "https://semantica.dev/vocab/",
                "semantica": "https://semantica.dev/ns#",
                "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#"
            },
            "@id": f"https://semantica.dev/graph/{datetime.now().isoformat()}",
            "@type": "semantica:KnowledgeGraph"
        }
        
        # Convert entities
        entities = kg.get("entities", [])
        if entities:
            jsonld["semantica:entities"] = [
                self._entity_to_jsonld(e) for e in entities
            ]
        
        # Convert relationships
        relationships = kg.get("relationships", [])
        if relationships:
            jsonld["semantica:relationships"] = [
                self._relationship_to_jsonld(r) for r in relationships
            ]
        
        # Add metadata
        jsonld["semantica:exportedAt"] = datetime.now().isoformat()
        if "metadata" in kg:
            jsonld["semantica:metadata"] = kg["metadata"]
        
        return jsonld
    
    def _entity_to_jsonld(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Convert entity to JSON-LD format."""
        jsonld = {
            "@id": entity.get("id") or f"semantica:entity/{entity.get('text', 'unknown')}",
            "@type": entity.get("type") or "semantica:Entity",
            "semantica:text": entity.get("text") or entity.get("label", ""),
            "semantica:confidence": entity.get("confidence", 1.0)
        }
        
        if "metadata" in entity:
            jsonld["semantica:metadata"] = entity["metadata"]
        
        return jsonld
    
    def _relationship_to_jsonld(self, rel: Dict[str, Any]) -> Dict[str, Any]:
        """Convert relationship to JSON-LD format."""
        jsonld = {
            "@id": rel.get("id") or f"semantica:rel/{rel.get('source_id')}_{rel.get('target_id')}",
            "@type": "semantica:Relationship",
            "semantica:type": rel.get("type", "related_to"),
            "semantica:source": {
                "@id": rel.get("source_id") or rel.get("source")
            },
            "semantica:target": {
                "@id": rel.get("target_id") or rel.get("target")
            },
            "semantica:confidence": rel.get("confidence", 1.0)
        }
        
        if "metadata" in rel:
            jsonld["semantica:metadata"] = rel["metadata"]
        
        return jsonld
