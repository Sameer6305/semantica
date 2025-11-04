"""
CSV exporter for Semantica framework.

This module provides CSV export capabilities
for tabular data and structured information.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import csv
from collections import defaultdict

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


class CSVExporter:
    """
    CSV exporter for knowledge graphs and structured data.
    
    • CSV export functionality
    • Tabular data serialization
    • Metadata and schema export
    • Performance optimization
    • Error handling and recovery
    • Advanced CSV features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize CSV exporter.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - delimiter: CSV delimiter (default: ',')
                - encoding: File encoding (default: 'utf-8')
                - include_header: Include header row (default: True)
        """
        self.logger = get_logger("csv_exporter")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.delimiter = self.config.get("delimiter", ",")
        self.encoding = self.config.get("encoding", "utf-8")
        self.include_header = self.config.get("include_header", True)
    
    def export(
        self,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        file_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export data to CSV file.
        
        Args:
            data: Data to export (list of dicts or dict with keys as separate files)
            file_path: Output file path
            **options: Additional options:
                - fieldnames: Custom field names
                - mode: Write mode ('w' or 'a')
        """
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        # Handle different data structures
        if isinstance(data, dict):
            # Export each key as separate CSV file
            for key, value in data.items():
                if isinstance(value, list):
                    output_path = file_path.parent / f"{file_path.stem}_{key}.csv"
                    self._write_csv(value, output_path, **options)
        elif isinstance(data, list):
            # Single CSV file
            self._write_csv(data, file_path, **options)
        else:
            raise ValidationError(f"Unsupported data type: {type(data)}")
        
        self.logger.info(f"Exported CSV to: {file_path}")
    
    def export_entities(
        self,
        entities: List[Dict[str, Any]],
        file_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export entities to CSV.
        
        Args:
            entities: List of entity dictionaries
            file_path: Output file path
            **options: Additional options
        """
        if not entities:
            raise ValidationError("No entities to export")
        
        # Normalize entity data
        normalized_entities = []
        for entity in entities:
            normalized = {
                "id": entity.get("id") or entity.get("entity_id", ""),
                "text": entity.get("text") or entity.get("label") or entity.get("name", ""),
                "type": entity.get("type") or entity.get("entity_type", ""),
                "confidence": entity.get("confidence", ""),
                "start": entity.get("start") or entity.get("start_offset", ""),
                "end": entity.get("end") or entity.get("end_offset", ""),
            }
            
            # Add metadata as JSON string
            if "metadata" in entity:
                import json
                normalized["metadata"] = json.dumps(entity["metadata"])
            
            normalized_entities.append(normalized)
        
        self._write_csv(normalized_entities, file_path, **options)
    
    def export_relationships(
        self,
        relationships: List[Dict[str, Any]],
        file_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export relationships to CSV.
        
        Args:
            relationships: List of relationship dictionaries
            file_path: Output file path
            **options: Additional options
        """
        if not relationships:
            raise ValidationError("No relationships to export")
        
        # Normalize relationship data
        normalized_rels = []
        for rel in relationships:
            normalized = {
                "id": rel.get("id", ""),
                "source_id": rel.get("source_id") or rel.get("source", ""),
                "target_id": rel.get("target_id") or rel.get("target", ""),
                "type": rel.get("type") or rel.get("relationship_type", ""),
                "confidence": rel.get("confidence", ""),
            }
            
            # Add metadata as JSON string
            if "metadata" in rel:
                import json
                normalized["metadata"] = json.dumps(rel["metadata"])
            
            normalized_rels.append(normalized)
        
        self._write_csv(normalized_rels, file_path, **options)
    
    def export_knowledge_graph(
        self,
        knowledge_graph: Dict[str, Any],
        base_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export knowledge graph to CSV files.
        
        Args:
            knowledge_graph: Knowledge graph dictionary
            base_path: Base path for output files
            **options: Additional options
        """
        base_path = Path(base_path)
        
        # Export entities
        entities = knowledge_graph.get("entities", [])
        if entities:
            entities_path = base_path.parent / f"{base_path.stem}_entities.csv"
            self.export_entities(entities, entities_path, **options)
        
        # Export relationships
        relationships = knowledge_graph.get("relationships", [])
        if relationships:
            rels_path = base_path.parent / f"{base_path.stem}_relationships.csv"
            self.export_relationships(relationships, rels_path, **options)
        
        # Export nodes if available
        nodes = knowledge_graph.get("nodes", [])
        if nodes:
            nodes_path = base_path.parent / f"{base_path.stem}_nodes.csv"
            self._write_csv(nodes, nodes_path, **options)
        
        # Export edges if available
        edges = knowledge_graph.get("edges", [])
        if edges:
            edges_path = base_path.parent / f"{base_path.stem}_edges.csv"
            self._write_csv(edges, edges_path, **options)
    
    def _write_csv(
        self,
        data: List[Dict[str, Any]],
        file_path: Path,
        **options
    ) -> None:
        """Write data to CSV file."""
        if not data:
            raise ValidationError("No data to write")
        
        fieldnames = options.get("fieldnames")
        if not fieldnames:
            # Get all unique keys from data
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            fieldnames = sorted(list(fieldnames))
        
        mode = options.get("mode", "w")
        
        try:
            with open(file_path, mode, newline='', encoding=self.encoding) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    delimiter=self.delimiter,
                    extrasaction='ignore'
                )
                
                if self.include_header and mode == 'w':
                    writer.writeheader()
                
                for row in data:
                    # Convert values to strings
                    row_str = {
                        k: str(v) if v is not None else ""
                        for k, v in row.items()
                        if k in fieldnames
                    }
                    writer.writerow(row_str)
        
        except Exception as e:
            raise ProcessingError(f"Failed to write CSV file: {e}") from e
