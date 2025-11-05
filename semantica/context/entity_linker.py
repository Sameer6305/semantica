"""
Entity Linker for Context Engineering

Links entities across sources to build the web of context,
assigning each entity a URL and connecting them meaningfully.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import quote
import hashlib

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.types import EntityDict


@dataclass
class EntityLink:
    """Entity link definition."""
    source_entity_id: str
    target_entity_id: str
    link_type: str  # "same_as", "related_to", "part_of", etc.
    confidence: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkedEntity:
    """Linked entity with context."""
    entity_id: str
    uri: str
    text: str
    type: str
    linked_entities: List[EntityLink] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


class EntityLinker:
    """
    Entity linker for context engineering.
    
    • Links entities across different sources
    • Assigns unique identifiers (URLs/URIs) to entities
    • Creates semantic connections between entities
    • Builds entity connection web
    • Supports cross-document entity linking
    • Enables entity disambiguation and resolution
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize entity linker.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - knowledge_graph: Knowledge graph for entity lookup
                - similarity_threshold: Similarity threshold for linking (default: 0.8)
                - base_uri: Base URI for entity URIs
                - enable_cross_document_linking: Enable cross-document linking (default: True)
        """
        self.logger = get_logger("entity_linker")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.knowledge_graph = self.config.get("knowledge_graph")
        self.similarity_threshold = self.config.get("similarity_threshold", 0.8)
        self.base_uri = self.config.get("base_uri", "https://semantica.dev/entity/")
        self.enable_cross_document_linking = self.config.get("enable_cross_document_linking", True)
        
        # Entity registry: entity_id -> URI
        self.entity_registry: Dict[str, str] = {}
        
        # Entity links: entity_id -> List[EntityLink]
        self.entity_links: Dict[str, List[EntityLink]] = {}
    
    def assign_uri(
        self,
        entity_id: str,
        entity_text: Optional[str] = None,
        entity_type: Optional[str] = None
    ) -> str:
        """
        Assign unique URI to entity.
        
        Args:
            entity_id: Entity identifier
            entity_text: Entity text/name
            entity_type: Entity type
            
        Returns:
            Entity URI
        """
        if entity_id in self.entity_registry:
            return self.entity_registry[entity_id]
        
        # Generate URI from entity information
        if entity_text:
            # Create URI-safe identifier
            uri_safe = quote(entity_text.lower().replace(" ", "_"), safe="")
            uri = f"{self.base_uri}{uri_safe}"
        else:
            # Use hash of entity_id
            entity_hash = hashlib.md5(entity_id.encode()).hexdigest()[:8]
            uri = f"{self.base_uri}{entity_hash}"
        
        # Add type if available
        if entity_type:
            uri = f"{uri}#{entity_type.lower()}"
        
        self.entity_registry[entity_id] = uri
        self.logger.debug(f"Assigned URI {uri} to entity {entity_id}")
        
        return uri
    
    def link(
        self,
        text: str,
        entities: Optional[List[EntityDict]] = None,
        context: Optional[List[Dict[str, Any]]] = None
    ) -> List[LinkedEntity]:
        """
        Link entities in text to knowledge graph.
        
        Args:
            text: Input text
            entities: List of entities to link
            context: Additional context information
            
        Returns:
            List of linked entities
        """
        if not entities:
            entities = []
        
        linked_entities = []
        
        for entity in entities:
            entity_id = entity.get("id") or entity.get("entity_id")
            entity_text = entity.get("text") or entity.get("label") or entity.get("name", "")
            entity_type = entity.get("type") or entity.get("entity_type", "UNKNOWN")
            
            if not entity_id:
                # Generate entity_id if not present
                entity_id = self._generate_entity_id(entity_text, entity_type)
            
            # Assign URI
            uri = self.assign_uri(entity_id, entity_text, entity_type)
            
            # Find linked entities
            linked_entity_links = self._find_linked_entities(
                entity_id,
                entity_text,
                entity_type,
                entities,
                context
            )
            
            linked_entity = LinkedEntity(
                entity_id=entity_id,
                uri=uri,
                text=entity_text,
                type=entity_type,
                linked_entities=linked_entity_links,
                context=entity.get("metadata", {}),
                confidence=entity.get("confidence", 1.0)
            )
            
            linked_entities.append(linked_entity)
            
            # Store links
            if entity_id not in self.entity_links:
                self.entity_links[entity_id] = []
            self.entity_links[entity_id].extend(linked_entity_links)
        
        return linked_entities
    
    def link_entities(
        self,
        entity1_id: str,
        entity2_id: str,
        link_type: str = "related_to",
        confidence: float = 1.0,
        source: Optional[str] = None,
        **metadata
    ) -> bool:
        """
        Create explicit link between two entities.
        
        Args:
            entity1_id: First entity ID
            entity2_id: Second entity ID
            link_type: Link type
            confidence: Link confidence
            source: Source of the link
            **metadata: Additional metadata
            
        Returns:
            True if link created successfully
        """
        link = EntityLink(
            source_entity_id=entity1_id,
            target_entity_id=entity2_id,
            link_type=link_type,
            confidence=confidence,
            source=source,
            metadata=metadata
        )
        
        if entity1_id not in self.entity_links:
            self.entity_links[entity1_id] = []
        
        self.entity_links[entity1_id].append(link)
        
        # Create bidirectional link if appropriate
        if link_type in ["related_to", "same_as"]:
            reverse_link = EntityLink(
                source_entity_id=entity2_id,
                target_entity_id=entity1_id,
                link_type=link_type,
                confidence=confidence,
                source=source,
                metadata=metadata
            )
            
            if entity2_id not in self.entity_links:
                self.entity_links[entity2_id] = []
            self.entity_links[entity2_id].append(reverse_link)
        
        self.logger.debug(f"Linked {entity1_id} --{link_type}--> {entity2_id}")
        return True
    
    def get_entity_links(self, entity_id: str) -> List[EntityLink]:
        """
        Get all links for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            List of entity links
        """
        return self.entity_links.get(entity_id, [])
    
    def get_entity_uri(self, entity_id: str) -> Optional[str]:
        """
        Get URI for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            Entity URI or None if not found
        """
        return self.entity_registry.get(entity_id)
    
    def find_similar_entities(
        self,
        entity_text: str,
        entity_type: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Find similar entities in knowledge graph.
        
        Args:
            entity_text: Entity text to search for
            entity_type: Optional entity type filter
            threshold: Similarity threshold (uses default if None)
            
        Returns:
            List of (entity_id, similarity_score) tuples
        """
        threshold = threshold or self.similarity_threshold
        
        if not self.knowledge_graph:
            return []
        
        # Simple text-based similarity (can be enhanced with embeddings)
        similar_entities = []
        
        entities = self.knowledge_graph.get("entities", [])
        
        for entity in entities:
            if entity_type and entity.get("type") != entity_type:
                continue
            
            entity_text2 = entity.get("text") or entity.get("label") or entity.get("name", "")
            
            # Calculate simple similarity (can be enhanced)
            similarity = self._calculate_text_similarity(entity_text.lower(), entity_text2.lower())
            
            if similarity >= threshold:
                entity_id = entity.get("id") or entity.get("entity_id")
                if entity_id:
                    similar_entities.append((entity_id, similarity))
        
        # Sort by similarity
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        
        return similar_entities
    
    def _find_linked_entities(
        self,
        entity_id: str,
        entity_text: str,
        entity_type: str,
        all_entities: List[EntityDict],
        context: Optional[List[Dict[str, Any]]]
    ) -> List[EntityLink]:
        """Find entities linked to this entity."""
        links = []
        
        # Find similar entities in knowledge graph
        if self.knowledge_graph:
            similar = self.find_similar_entities(entity_text, entity_type)
            for similar_id, similarity in similar:
                if similar_id != entity_id:
                    links.append(EntityLink(
                        source_entity_id=entity_id,
                        target_entity_id=similar_id,
                        link_type="same_as" if similarity >= 0.9 else "related_to",
                        confidence=similarity,
                        metadata={"similarity": similarity}
                    ))
        
        # Find relationships in context
        if context:
            for ctx in context:
                relationships = ctx.get("relationships", [])
                for rel in relationships:
                    source_id = rel.get("source_id")
                    target_id = rel.get("target_id")
                    
                    if source_id == entity_id:
                        links.append(EntityLink(
                            source_entity_id=entity_id,
                            target_entity_id=target_id,
                            link_type=rel.get("type", "related_to"),
                            confidence=rel.get("confidence", 1.0),
                            source=ctx.get("source")
                        ))
                    elif target_id == entity_id:
                        links.append(EntityLink(
                            source_entity_id=entity_id,
                            target_entity_id=source_id,
                            link_type=rel.get("type", "related_to"),
                            confidence=rel.get("confidence", 1.0),
                            source=ctx.get("source")
                        ))
        
        return links
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _generate_entity_id(self, text: str, entity_type: str) -> str:
        """Generate entity ID from text and type."""
        entity_hash = hashlib.md5(f"{text}_{entity_type}".encode()).hexdigest()[:12]
        return f"{entity_type.lower()}_{entity_hash}"
    
    def build_entity_web(self) -> Dict[str, Any]:
        """
        Build entity connection web.
        
        Returns:
            Entity web dictionary
        """
        web = {
            "entities": {},
            "links": [],
            "statistics": {
                "total_entities": len(self.entity_registry),
                "total_links": sum(len(links) for links in self.entity_links.values())
            }
        }
        
        # Add entities
        for entity_id, uri in self.entity_registry.items():
            web["entities"][entity_id] = {
                "uri": uri,
                "links": len(self.entity_links.get(entity_id, []))
            }
        
        # Add links
        for entity_id, links in self.entity_links.items():
            for link in links:
                web["links"].append({
                    "source": link.source_entity_id,
                    "target": link.target_entity_id,
                    "type": link.link_type,
                    "confidence": link.confidence
                })
        
        return web
