"""
Graph exporter for Semantica framework.

This module provides graph format export capabilities
for knowledge graphs and network data.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


class GraphExporter:
    """
    Graph exporter for knowledge graphs and network data.
    
    • Graph format export functionality
    • Network data serialization
    • Graph visualization export
    • Performance optimization
    • Error handling and recovery
    • Advanced graph features
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize graph exporter.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - format: Export format ('graphml', 'gexf', 'json', 'dot')
                - include_attributes: Include node/edge attributes (default: True)
        """
        self.logger = get_logger("graph_exporter")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.format = self.config.get("format", "json")
        self.include_attributes = self.config.get("include_attributes", True)
    
    def export(
        self,
        graph_data: Dict[str, Any],
        file_path: Union[str, Path],
        format: Optional[str] = None,
        **options
    ) -> None:
        """
        Export graph to file.
        
        Args:
            graph_data: Graph data dictionary
            file_path: Output file path
            format: Export format ('graphml', 'gexf', 'json', 'dot')
            **options: Additional options
        """
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        export_format = format or self.format
        
        if export_format == "json":
            self._export_json(graph_data, file_path, **options)
        elif export_format == "graphml":
            self._export_graphml(graph_data, file_path, **options)
        elif export_format == "gexf":
            self._export_gexf(graph_data, file_path, **options)
        elif export_format == "dot":
            self._export_dot(graph_data, file_path, **options)
        else:
            raise ValidationError(f"Unsupported graph format: {export_format}")
        
        self.logger.info(f"Exported graph ({export_format}) to: {file_path}")
    
    def export_knowledge_graph(
        self,
        knowledge_graph: Dict[str, Any],
        file_path: Union[str, Path],
        format: Optional[str] = None,
        **options
    ) -> None:
        """
        Export knowledge graph to graph format.
        
        Args:
            knowledge_graph: Knowledge graph dictionary
            file_path: Output file path
            format: Export format
            **options: Additional options
        """
        # Convert knowledge graph to graph format
        graph_data = self._convert_kg_to_graph(knowledge_graph)
        self.export(graph_data, file_path, format=format, **options)
    
    def _convert_kg_to_graph(self, kg: Dict[str, Any]) -> Dict[str, Any]:
        """Convert knowledge graph to graph format."""
        nodes = []
        edges = []
        
        # Convert entities to nodes
        entities = kg.get("entities", [])
        for entity in entities:
            node = {
                "id": entity.get("id") or entity.get("entity_id"),
                "label": entity.get("text") or entity.get("label") or entity.get("name", ""),
                "type": entity.get("type") or entity.get("entity_type", "entity")
            }
            
            if self.include_attributes:
                node["attributes"] = {
                    "confidence": entity.get("confidence", 1.0),
                    **entity.get("metadata", {})
                }
            
            nodes.append(node)
        
        # Convert relationships to edges
        relationships = kg.get("relationships", [])
        for rel in relationships:
            edge = {
                "source": rel.get("source_id") or rel.get("source"),
                "target": rel.get("target_id") or rel.get("target"),
                "type": rel.get("type") or rel.get("relationship_type", "related_to")
            }
            
            if self.include_attributes:
                edge["attributes"] = {
                    "confidence": rel.get("confidence", 1.0),
                    **rel.get("metadata", {})
                }
            
            edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": kg.get("metadata", {})
        }
    
    def _export_json(self, graph_data: Dict[str, Any], file_path: Path, **options) -> None:
        """Export to JSON format."""
        from ..utils.helpers import write_json_file
        
        write_json_file(graph_data, file_path, indent=2)
    
    def _export_graphml(self, graph_data: Dict[str, Any], file_path: Path, **options) -> None:
        """Export to GraphML format."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"')
        lines.append('         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"')
        lines.append('         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns')
        lines.append('         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">')
        lines.append("")
        
        # Define attributes
        lines.append('  <key id="type" for="node" attr.name="type" attr.type="string"/>')
        lines.append('  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>')
        lines.append("")
        
        # Graph
        lines.append('  <graph id="G" edgedefault="directed">')
        lines.append("")
        
        # Nodes
        nodes = graph_data.get("nodes", [])
        for node in nodes:
            node_id = node.get("id", "")
            label = node.get("label", "")
            node_type = node.get("type", "")
            
            lines.append(f'    <node id="{node_id}">')
            lines.append(f'      <data key="label">{label}</data>')
            lines.append(f'      <data key="type">{node_type}</data>')
            
            if self.include_attributes and "attributes" in node:
                attrs = node["attributes"]
                if "confidence" in attrs:
                    lines.append(f'      <data key="confidence">{attrs["confidence"]}</data>')
            
            lines.append("    </node>")
        
        lines.append("")
        
        # Edges
        edges = graph_data.get("edges", [])
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            edge_type = edge.get("type", "")
            
            lines.append(f'    <edge source="{source}" target="{target}">')
            lines.append(f'      <data key="label">{edge_type}</data>')
            
            if self.include_attributes and "attributes" in edge:
                attrs = edge["attributes"]
                if "confidence" in attrs:
                    lines.append(f'      <data key="confidence">{attrs["confidence"]}</data>')
            
            lines.append("    </edge>")
        
        lines.append("  </graph>")
        lines.append("</graphml>")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def _export_gexf(self, graph_data: Dict[str, Any], file_path: Path, **options) -> None:
        """Export to GEXF format."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">')
        lines.append('  <graph mode="static" defaultedgetype="directed">')
        lines.append("    <nodes>")
        
        # Nodes
        nodes = graph_data.get("nodes", [])
        for node in nodes:
            node_id = node.get("id", "")
            label = node.get("label", "")
            
            lines.append(f'      <node id="{node_id}" label="{label}">')
            
            if self.include_attributes and "attributes" in node:
                lines.append("        <attvalues>")
                attrs = node["attributes"]
                for key, value in attrs.items():
                    lines.append(f'          <attvalue for="{key}" value="{value}"/>')
                lines.append("        </attvalues>")
            
            lines.append("      </node>")
        
        lines.append("    </nodes>")
        lines.append("    <edges>")
        
        # Edges
        edges = graph_data.get("edges", [])
        for i, edge in enumerate(edges):
            source = edge.get("source", "")
            target = edge.get("target", "")
            edge_type = edge.get("type", "")
            
            lines.append(f'      <edge id="{i}" source="{source}" target="{target}" label="{edge_type}">')
            
            if self.include_attributes and "attributes" in edge:
                lines.append("        <attvalues>")
                attrs = edge["attributes"]
                for key, value in attrs.items():
                    lines.append(f'          <attvalue for="{key}" value="{value}"/>')
                lines.append("        </attvalues>")
            
            lines.append("      </edge>")
        
        lines.append("    </edges>")
        lines.append("  </graph>")
        lines.append("</gexf>")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def _export_dot(self, graph_data: Dict[str, Any], file_path: Path, **options) -> None:
        """Export to DOT format (Graphviz)."""
        lines = ["digraph G {"]
        lines.append("  rankdir=LR;")
        lines.append("")
        
        # Nodes
        nodes = graph_data.get("nodes", [])
        for node in nodes:
            node_id = node.get("id", "").replace('"', '\\"')
            label = node.get("label", "").replace('"', '\\"')
            node_type = node.get("type", "")
            
            attributes = [f'label="{label}"']
            if node_type:
                attributes.append(f'type="{node_type}"')
            
            attrs_str = ", ".join(attributes)
            lines.append(f'  "{node_id}" [{attrs_str}];')
        
        lines.append("")
        
        # Edges
        edges = graph_data.get("edges", [])
        for edge in edges:
            source = edge.get("source", "").replace('"', '\\"')
            target = edge.get("target", "").replace('"', '\\"')
            edge_type = edge.get("type", "").replace('"', '\\"')
            
            attributes = [f'label="{edge_type}"']
            attrs_str = ", ".join(attributes)
            lines.append(f'  "{source}" -> "{target}" [{attrs_str}];')
        
        lines.append("}")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
