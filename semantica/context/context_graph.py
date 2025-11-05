"""
Context Graph Builder for Semantica framework.

Formalizes context as a graph of connections, turning context
from intuition into infrastructure. Context is connection.
"""

from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict

from .entity_linker import EntityLinker
from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.types import EntityDict, RelationshipDict


@dataclass
class ContextNode:
    """Context graph node."""
    node_id: str
    node_type: str  # "entity", "concept", "document", "intent", etc.
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextEdge:
    """Context graph edge."""
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextGraphBuilder:
    """
    Context graph builder for formalizing context as connections.
    
    • Builds context graphs from entities and relationships
    • Creates meaningful connections between concepts
    • Assigns URLs/URIs to entities for web-like context
    • Formalizes context into graph structure
    • Supports ontology-based context graphs
    • Enables context traversal and querying
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize context graph builder.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - extract_entities: Extract entities from content (default: True)
                - extract_relationships: Extract relationships (default: True)
                - link_external_entities: Link to external entities (default: True)
                - entity_linker: Entity linker instance
        """
        self.logger = get_logger("context_graph_builder")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.extract_entities = self.config.get("extract_entities", True)
        self.extract_relationships = self.config.get("extract_relationships", True)
        self.link_external_entities = self.config.get("link_external_entities", True)
        
        self.entity_linker = self.config.get("entity_linker") or EntityLinker()
        
        # Graph structure
        self.nodes: Dict[str, ContextNode] = {}
        self.edges: List[ContextEdge] = []
        
        # Indexes
        self.node_type_index: Dict[str, Set[str]] = defaultdict(set)
        self.edge_type_index: Dict[str, List[ContextEdge]] = defaultdict(list)
    
    def build_from_conversations(
        self,
        conversations: List[Union[str, Dict[str, Any]]],
        link_entities: bool = True,
        extract_intents: bool = False,
        extract_sentiments: bool = False,
        **options
    ) -> Dict[str, Any]:
        """
        Build context graph from conversations.
        
        Args:
            conversations: List of conversation files or dictionaries
            link_entities: Link entities across conversations
            extract_intents: Extract conversation intents
            extract_sentiments: Extract sentiment information
            **options: Additional options
            
        Returns:
            Context graph dictionary
        """
        self.logger.info(f"Building context graph from {len(conversations)} conversations")
        
        # Process each conversation
        for conv in conversations:
            if isinstance(conv, str):
                # Load conversation from file
                conv_data = self._load_conversation(conv)
            else:
                conv_data = conv
            
            self._process_conversation(
                conv_data,
                extract_intents=extract_intents,
                extract_sentiments=extract_sentiments
            )
        
        # Link entities if requested
        if link_entities:
            self._link_entities()
        
        # Build graph structure
        graph = self._build_graph_structure()
        
        self.logger.info(
            f"Built context graph: {len(self.nodes)} nodes, {len(self.edges)} edges"
        )
        
        return graph
    
    def build_from_entities_and_relationships(
        self,
        entities: List[EntityDict],
        relationships: List[RelationshipDict],
        **options
    ) -> Dict[str, Any]:
        """
        Build context graph from entities and relationships.
        
        Args:
            entities: List of entities
            relationships: List of relationships
            **options: Additional options
            
        Returns:
            Context graph dictionary
        """
        # Add entities as nodes
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            if not entity_id:
                continue
            
            node = ContextNode(
                node_id=entity_id,
                node_type="entity",
                content=entity.get("text") or entity.get("label") or entity.get("name", ""),
                metadata={
                    "type": entity.get("type") or entity.get("entity_type"),
                    "confidence": entity.get("confidence", 1.0),
                    **entity.get("metadata", {})
                },
                properties=entity.get("properties", {})
            )
            
            self.nodes[entity_id] = node
            self.node_type_index["entity"].add(entity_id)
            
            # Assign URI if entity linker available
            if self.entity_linker:
                self.entity_linker.assign_uri(
                    entity_id,
                    node.content,
                    node.metadata.get("type")
                )
        
        # Add relationships as edges
        for rel in relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            
            if not source_id or not target_id:
                continue
            
            # Ensure source and target nodes exist
            if source_id not in self.nodes:
                self.nodes[source_id] = ContextNode(
                    node_id=source_id,
                    node_type="entity",
                    content=source_id
                )
            
            if target_id not in self.nodes:
                self.nodes[target_id] = ContextNode(
                    node_id=target_id,
                    node_type="entity",
                    content=target_id
                )
            
            edge = ContextEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=rel.get("type") or rel.get("relationship_type", "related_to"),
                weight=rel.get("confidence", 1.0),
                metadata=rel.get("metadata", {})
            )
            
            self.edges.append(edge)
            self.edge_type_index[edge.edge_type].append(edge)
        
        return self._build_graph_structure()
    
    def add_node(
        self,
        node_id: str,
        node_type: str,
        content: str,
        **metadata
    ) -> bool:
        """
        Add node to context graph.
        
        Args:
            node_id: Node identifier
            node_type: Node type
            content: Node content
            **metadata: Additional metadata
            
        Returns:
            True if node added successfully
        """
        node = ContextNode(
            node_id=node_id,
            node_type=node_type,
            content=content,
            metadata=metadata
        )
        
        self.nodes[node_id] = node
        self.node_type_index[node_type].add(node_id)
        
        self.logger.debug(f"Added node: {node_id} ({node_type})")
        return True
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "related_to",
        weight: float = 1.0,
        **metadata
    ) -> bool:
        """
        Add edge to context graph.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Edge type
            weight: Edge weight
            **metadata: Additional metadata
            
        Returns:
            True if edge added successfully
        """
        # Ensure nodes exist
        if source_id not in self.nodes:
            self.add_node(source_id, "entity", source_id)
        
        if target_id not in self.nodes:
            self.add_node(target_id, "entity", target_id)
        
        edge = ContextEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata
        )
        
        self.edges.append(edge)
        self.edge_type_index[edge_type].append(edge)
        
        self.logger.debug(f"Added edge: {source_id} --{edge_type}--> {target_id}")
        return True
    
    def _process_conversation(
        self,
        conv_data: Dict[str, Any],
        extract_intents: bool = False,
        extract_sentiments: bool = False
    ) -> None:
        """Process a conversation and add to graph."""
        # Extract conversation ID
        conv_id = conv_data.get("id") or f"conv_{hash(str(conv_data)) % 10000}"
        
        # Add conversation as node
        self.add_node(
            conv_id,
            "conversation",
            conv_data.get("content", "") or conv_data.get("summary", ""),
            **{
                "timestamp": conv_data.get("timestamp"),
                "participants": conv_data.get("participants", []),
                **conv_data.get("metadata", {})
            }
        )
        
        # Extract entities
        if self.extract_entities:
            entities = conv_data.get("entities", [])
            for entity in entities:
                entity_id = entity.get("id") or entity.get("entity_id")
                if entity_id:
                    self.add_node(
                        entity_id,
                        "entity",
                        entity.get("text") or entity.get("label", ""),
                        **{
                            "type": entity.get("type"),
                            "confidence": entity.get("confidence", 1.0)
                        }
                    )
                    
                    # Link entity to conversation
                    self.add_edge(conv_id, entity_id, "mentions", weight=entity.get("confidence", 1.0))
        
        # Extract relationships
        if self.extract_relationships:
            relationships = conv_data.get("relationships", [])
            for rel in relationships:
                source_id = rel.get("source_id")
                target_id = rel.get("target_id")
                
                if source_id and target_id:
                    self.add_edge(
                        source_id,
                        target_id,
                        rel.get("type", "related_to"),
                        weight=rel.get("confidence", 1.0)
                    )
        
        # Extract intents
        if extract_intents:
            intents = conv_data.get("intents", [])
            for intent in intents:
                intent_id = f"{conv_id}_intent_{len(intents)}"
                self.add_node(
                    intent_id,
                    "intent",
                    intent.get("text", ""),
                    **{"confidence": intent.get("confidence", 1.0)}
                )
                self.add_edge(conv_id, intent_id, "has_intent")
        
        # Extract sentiments
        if extract_sentiments:
            sentiment = conv_data.get("sentiment")
            if sentiment:
                sentiment_id = f"{conv_id}_sentiment"
                self.add_node(
                    sentiment_id,
                    "sentiment",
                    sentiment.get("label", ""),
                    **{"score": sentiment.get("score", 0.0)}
                )
                self.add_edge(conv_id, sentiment_id, "has_sentiment")
    
    def _link_entities(self) -> None:
        """Link entities across the graph."""
        if not self.entity_linker:
            return
        
        # Get all entity nodes
        entity_nodes = [
            (node_id, node)
            for node_id, node in self.nodes.items()
            if node.node_type == "entity"
        ]
        
        # Link similar entities
        for i, (node_id1, node1) in enumerate(entity_nodes):
            for node_id2, node2 in entity_nodes[i+1:]:
                # Check if similar
                similarity = self.entity_linker._calculate_text_similarity(
                    node1.content.lower(),
                    node2.content.lower()
                )
                
                if similarity >= self.entity_linker.similarity_threshold:
                    # Link entities
                    self.entity_linker.link_entities(
                        node_id1,
                        node_id2,
                        link_type="same_as" if similarity >= 0.9 else "related_to",
                        confidence=similarity
                    )
                    
                    # Add edge to graph
                    self.add_edge(
                        node_id1,
                        node_id2,
                        "same_as" if similarity >= 0.9 else "related_to",
                        weight=similarity
                    )
    
    def _load_conversation(self, file_path: str) -> Dict[str, Any]:
        """Load conversation from file."""
        from ..utils.helpers import read_json_file
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            raise ProcessingError(f"Conversation file not found: {file_path}")
        
        return read_json_file(path)
    
    def _build_graph_structure(self) -> Dict[str, Any]:
        """Build graph structure dictionary."""
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    "type": node.node_type,
                    "content": node.content,
                    "metadata": node.metadata,
                    "properties": node.properties
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "type": edge.edge_type,
                    "weight": edge.weight,
                    "metadata": edge.metadata
                }
                for edge in self.edges
            ],
            "statistics": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "node_types": {
                    node_type: len(node_ids)
                    for node_type, node_ids in self.node_type_index.items()
                },
                "edge_types": {
                    edge_type: len(edges)
                    for edge_type, edges in self.edge_type_index.items()
                }
            }
        }
    
    def get_neighbors(self, node_id: str, max_hops: int = 1) -> List[str]:
        """
        Get neighbor nodes.
        
        Args:
            node_id: Node identifier
            max_hops: Maximum number of hops
            
        Returns:
            List of neighbor node IDs
        """
        neighbors = set()
        current_level = {node_id}
        
        for hop in range(max_hops):
            next_level = set()
            
            for current_id in current_level:
                # Find outgoing edges
                for edge in self.edges:
                    if edge.source_id == current_id:
                        next_level.add(edge.target_id)
                        neighbors.add(edge.target_id)
                
                # Find incoming edges
                for edge in self.edges:
                    if edge.target_id == current_id:
                        next_level.add(edge.source_id)
                        neighbors.add(edge.source_id)
            
            current_level = next_level
        
        return list(neighbors)
    
    def query(
        self,
        node_type: Optional[str] = None,
        edge_type: Optional[str] = None,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Query graph nodes and edges.
        
        Args:
            node_type: Filter by node type
            edge_type: Filter by edge type
            **filters: Additional filters
            
        Returns:
            List of matching nodes
        """
        results = []
        
        for node_id, node in self.nodes.items():
            # Filter by node type
            if node_type and node.node_type != node_type:
                continue
            
            # Filter by metadata
            match = True
            for key, value in filters.items():
                if key in node.metadata and node.metadata[key] != value:
                    match = False
                    break
            
            if match:
                results.append({
                    "id": node_id,
                    "type": node.node_type,
                    "content": node.content,
                    "metadata": node.metadata
                })
        
        return results
