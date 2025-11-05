"""
Semantic Network Extractor for Semantica framework.

Extracts structured semantic networks from documents as part of the
6-stage ontology generation pipeline (Stage 2: semantic network extraction).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .ner_extractor import Entity
from .relation_extractor import Relation


@dataclass
class SemanticNode:
    """Semantic network node representation."""
    
    id: str
    label: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticEdge:
    """Semantic network edge representation."""
    
    source: str
    target: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticNetwork:
    """Semantic network representation."""
    
    nodes: List[SemanticNode]
    edges: List[SemanticEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticNetworkExtractor:
    """Semantic network extractor for structured networks."""
    
    def __init__(self, **config):
        """
        Initialize semantic network extractor.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("semantic_network_extractor")
        self.config = config
    
    def extract_network(
        self,
        text: str,
        entities: Optional[List[Entity]] = None,
        relations: Optional[List[Relation]] = None,
        **options
    ) -> SemanticNetwork:
        """
        Extract semantic network from text.
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            relations: Pre-extracted relations (optional)
            **options: Extraction options
            
        Returns:
            SemanticNetwork: Extracted semantic network
        """
        from .ner_extractor import NERExtractor
        from .relation_extractor import RelationExtractor
        
        # Extract entities if not provided
        if entities is None:
            ner = NERExtractor(**self.config.get("ner", {}))
            entities = ner.extract_entities(text)
        
        # Extract relations if not provided
        if relations is None:
            rel_extractor = RelationExtractor(**self.config.get("relation", {}))
            relations = rel_extractor.extract_relations(text, entities)
        
        # Build network
        network = self._build_network(entities, relations)
        
        return network
    
    def _build_network(self, entities: List[Entity], relations: List[Relation]) -> SemanticNetwork:
        """Build semantic network from entities and relations."""
        nodes = []
        edges = []
        node_map = {}
        
        # Create nodes from entities
        for entity in entities:
            node_id = f"entity_{len(nodes)}"
            node_map[entity.text] = node_id
            
            node = SemanticNode(
                id=node_id,
                label=entity.text,
                type=entity.label,
                properties={
                    "start_char": entity.start_char,
                    "end_char": entity.end_char,
                    "confidence": entity.confidence
                },
                metadata=entity.metadata
            )
            nodes.append(node)
        
        # Create edges from relations
        for relation in relations:
            subject_id = node_map.get(relation.subject.text)
            object_id = node_map.get(relation.object.text)
            
            if subject_id and object_id:
                edge = SemanticEdge(
                    source=subject_id,
                    target=object_id,
                    label=relation.predicate,
                    properties={
                        "confidence": relation.confidence,
                        "context": relation.context
                    },
                    metadata=relation.metadata
                )
                edges.append(edge)
        
        return SemanticNetwork(
            nodes=nodes,
            edges=edges,
            metadata={
                "node_count": len(nodes),
                "edge_count": len(edges),
                "entity_types": list(set(e.label for e in entities)),
                "relation_types": list(set(r.predicate for r in relations))
            }
        )
    
    def export_to_yaml(self, network: SemanticNetwork, file_path: Optional[str] = None) -> str:
        """
        Export semantic network to YAML format.
        
        Args:
            network: Semantic network
            file_path: Optional file path to save
            
        Returns:
            str: YAML representation
        """
        yaml_data = {
            "network": {
                "nodes": [
                    {
                        "id": node.id,
                        "label": node.label,
                        "type": node.type,
                        "properties": node.properties,
                        "metadata": node.metadata
                    }
                    for node in network.nodes
                ],
                "edges": [
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "label": edge.label,
                        "properties": edge.properties,
                        "metadata": edge.metadata
                    }
                    for edge in network.edges
                ],
                "metadata": network.metadata
            }
        }
        
        yaml_str = yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    def analyze_network(self, network: SemanticNetwork) -> Dict[str, Any]:
        """
        Analyze semantic network structure.
        
        Args:
            network: Semantic network
            
        Returns:
            dict: Network analysis
        """
        # Count node types
        node_types = {}
        for node in network.nodes:
            node_types[node.type] = node_types.get(node.type, 0) + 1
        
        # Count relation types
        relation_types = {}
        for edge in network.edges:
            relation_types[edge.label] = relation_types.get(edge.label, 0) + 1
        
        # Calculate connectivity
        node_degrees = {}
        for edge in network.edges:
            node_degrees[edge.source] = node_degrees.get(edge.source, 0) + 1
            node_degrees[edge.target] = node_degrees.get(edge.target, 0) + 1
        
        avg_degree = sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0
        
        return {
            "node_count": len(network.nodes),
            "edge_count": len(network.edges),
            "node_types": node_types,
            "relation_types": relation_types,
            "average_degree": avg_degree,
            "connectivity": "sparse" if avg_degree < 2 else "moderate" if avg_degree < 5 else "dense"
        }
