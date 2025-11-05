"""
Agent Memory Manager

Manages agent memory and context retrieval, integrating RAG
with knowledge graphs to give agents persistent context.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.types import EntityDict, RelationshipDict


@dataclass
class MemoryItem:
    """Memory item structure."""
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    entities: List[EntityDict] = field(default_factory=list)
    relationships: List[RelationshipDict] = field(default_factory=list)
    embedding: Optional[Any] = None
    memory_id: Optional[str] = None


class AgentMemory:
    """
    Agent memory manager with RAG integration.
    
    • Persistent memory storage for agents
    • Context retrieval from vector stores
    • Knowledge graph context integration
    • Conversation history management
    • Context accumulation over time
    • Memory retrieval for agent decision-making
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize agent memory.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - vector_store: Vector store instance
                - knowledge_graph: Knowledge graph instance
                - retention_policy: Memory retention policy (e.g., "30_days", "unlimited")
                - max_memory_size: Maximum number of memory items
                - embedding_model: Embedding model for memory items
        """
        self.logger = get_logger("agent_memory")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.vector_store = self.config.get("vector_store")
        self.knowledge_graph = self.config.get("knowledge_graph")
        
        self.retention_policy = self.config.get("retention_policy", "unlimited")
        self.max_memory_size = self.config.get("max_memory_size", 10000)
        
        # In-memory storage
        self.memory_items: Dict[str, MemoryItem] = {}
        self.memory_index: deque = deque(maxlen=self.max_memory_size)
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "items_by_type": {},
            "last_accessed": None
        }
    
    def store(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[List[EntityDict]] = None,
        relationships: Optional[List[RelationshipDict]] = None,
        **options
    ) -> str:
        """
        Store memory item.
        
        Args:
            content: Memory content
            metadata: Additional metadata
            entities: Related entities
            relationships: Related relationships
            **options: Additional options:
                - memory_id: Custom memory ID
                - timestamp: Custom timestamp
                
        Returns:
            Memory ID
        """
        memory_id = options.get("memory_id") or self._generate_memory_id()
        timestamp = options.get("timestamp") or datetime.now()
        
        # Create memory item
        memory_item = MemoryItem(
            content=content,
            timestamp=timestamp,
            metadata=metadata or {},
            entities=entities or [],
            relationships=relationships or [],
            memory_id=memory_id
        )
        
        # Generate embedding if vector store available
        if self.vector_store:
            try:
                embedding = self._generate_embedding(content)
                memory_item.embedding = embedding
                
                # Store in vector store
                if hasattr(self.vector_store, "add"):
                    self.vector_store.add(
                        id=memory_id,
                        vector=embedding,
                        metadata={
                            "content": content,
                            "timestamp": timestamp.isoformat(),
                            **metadata or {}
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Failed to generate embedding: {e}")
        
        # Store in memory
        self.memory_items[memory_id] = memory_item
        self.memory_index.append(memory_id)
        
        # Update knowledge graph if available
        if self.knowledge_graph and entities:
            self._update_knowledge_graph(entities, relationships)
        
        # Update statistics
        self.stats["total_items"] += 1
        item_type = metadata.get("type", "general") if metadata else "general"
        self.stats["items_by_type"][item_type] = self.stats["items_by_type"].get(item_type, 0) + 1
        
        self.logger.debug(f"Stored memory item: {memory_id}")
        
        # Apply retention policy
        self._apply_retention_policy()
        
        return memory_id
    
    def retrieve(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.0,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_score: Minimum relevance score
            **filters: Additional filters:
                - type: Filter by memory type
                - start_date: Filter by start date
                - end_date: Filter by end date
                
        Returns:
            List of retrieved memory items
        """
        results = []
        
        # Vector-based retrieval
        if self.vector_store:
            try:
                if hasattr(self.vector_store, "search"):
                    vector_results = self.vector_store.search(
                        query=query,
                        top_k=max_results * 2
                    )
                    
                    for result in vector_results:
                        memory_id = result.get("id")
                        if memory_id in self.memory_items:
                            memory_item = self.memory_items[memory_id]
                            
                            # Apply filters
                            if not self._matches_filters(memory_item, filters):
                                continue
                            
                            results.append({
                                "memory_id": memory_id,
                                "content": memory_item.content,
                                "score": result.get("score", 0.0),
                                "timestamp": memory_item.timestamp.isoformat(),
                                "metadata": memory_item.metadata,
                                "entities": memory_item.entities,
                                "relationships": memory_item.relationships
                            })
            except Exception as e:
                self.logger.warning(f"Vector retrieval failed: {e}")
        
        # Fallback to keyword search if no vector store
        if not results:
            results = self._keyword_search(query, max_results, filters)
        
        # Sort by score and return top results
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        filtered_results = [r for r in results if r.get("score", 0.0) >= min_score]
        
        self.stats["last_accessed"] = datetime.now().isoformat()
        
        return filtered_results[:max_results]
    
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific memory item.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            Memory item dictionary or None if not found
        """
        if memory_id not in self.memory_items:
            return None
        
        memory_item = self.memory_items[memory_id]
        
        return {
            "memory_id": memory_id,
            "content": memory_item.content,
            "timestamp": memory_item.timestamp.isoformat(),
            "metadata": memory_item.metadata,
            "entities": memory_item.entities,
            "relationships": memory_item.relationships
        }
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete memory item.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            True if deleted successfully
        """
        if memory_id not in self.memory_items:
            return False
        
        # Remove from vector store
        if self.vector_store and hasattr(self.vector_store, "delete"):
            try:
                self.vector_store.delete(memory_id)
            except Exception as e:
                self.logger.warning(f"Failed to delete from vector store: {e}")
        
        # Remove from memory
        del self.memory_items[memory_id]
        
        # Remove from index
        if memory_id in self.memory_index:
            self.memory_index.remove(memory_id)
        
        self.stats["total_items"] = max(0, self.stats["total_items"] - 1)
        
        self.logger.debug(f"Deleted memory item: {memory_id}")
        return True
    
    def clear_memory(self, **filters) -> int:
        """
        Clear memory items matching filters.
        
        Args:
            **filters: Filter criteria
            
        Returns:
            Number of items deleted
        """
        deleted_count = 0
        memory_ids_to_delete = []
        
        for memory_id, memory_item in self.memory_items.items():
            if self._matches_filters(memory_item, filters):
                memory_ids_to_delete.append(memory_id)
        
        for memory_id in memory_ids_to_delete:
            if self.delete_memory(memory_id):
                deleted_count += 1
        
        return deleted_count
    
    def get_conversation_history(
        self,
        conversation_id: Optional[str] = None,
        max_items: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Optional conversation ID filter
            max_items: Maximum number of items
            
        Returns:
            List of conversation items
        """
        history = []
        
        for memory_id in list(self.memory_index)[-max_items:]:
            memory_item = self.memory_items.get(memory_id)
            if not memory_item:
                continue
            
            # Filter by conversation ID if provided
            if conversation_id:
                item_conv_id = memory_item.metadata.get("conversation_id")
                if item_conv_id != conversation_id:
                    continue
            
            # Check if it's a conversation item
            if memory_item.metadata.get("type") == "conversation":
                history.append({
                    "memory_id": memory_id,
                    "content": memory_item.content,
                    "timestamp": memory_item.timestamp.isoformat(),
                    "metadata": memory_item.metadata
                })
        
        return history
    
    def _generate_memory_id(self) -> str:
        """Generate unique memory ID."""
        import hashlib
        import time
        
        timestamp = str(time.time())
        random_str = str(hash(str(self.memory_items)) % 10000)
        memory_hash = hashlib.md5(f"{timestamp}_{random_str}".encode()).hexdigest()[:12]
        
        return f"mem_{memory_hash}"
    
    def _generate_embedding(self, content: str) -> Any:
        """Generate embedding for content."""
        # This would use an embedding model
        # For now, return placeholder
        if hasattr(self.vector_store, "embed"):
            return self.vector_store.embed(content)
        return None
    
    def _update_knowledge_graph(
        self,
        entities: List[EntityDict],
        relationships: Optional[List[RelationshipDict]]
    ) -> None:
        """Update knowledge graph with new entities and relationships."""
        if not self.knowledge_graph:
            return
        
        # Add entities to graph
        graph_entities = self.knowledge_graph.get("entities", [])
        existing_ids = {e.get("id") for e in graph_entities}
        
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id and entity_id not in existing_ids:
                graph_entities.append(entity)
        
        self.knowledge_graph["entities"] = graph_entities
        
        # Add relationships
        if relationships:
            graph_relationships = self.knowledge_graph.get("relationships", [])
            graph_relationships.extend(relationships)
            self.knowledge_graph["relationships"] = graph_relationships
    
    def _matches_filters(self, memory_item: MemoryItem, filters: Dict[str, Any]) -> bool:
        """Check if memory item matches filters."""
        # Filter by type
        if "type" in filters:
            item_type = memory_item.metadata.get("type")
            if item_type != filters["type"]:
                return False
        
        # Filter by date range
        if "start_date" in filters:
            start_date = filters["start_date"]
            if isinstance(start_date, str):
                from dateutil.parser import parse
                start_date = parse(start_date)
            if memory_item.timestamp < start_date:
                return False
        
        if "end_date" in filters:
            end_date = filters["end_date"]
            if isinstance(end_date, str):
                from dateutil.parser import parse
                end_date = parse(end_date)
            if memory_item.timestamp > end_date:
                return False
        
        return True
    
    def _keyword_search(
        self,
        query: str,
        max_results: int,
        filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Fallback keyword search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for memory_id, memory_item in self.memory_items.items():
            if not self._matches_filters(memory_item, filters):
                continue
            
            content_lower = memory_item.content.lower()
            content_words = set(content_lower.split())
            
            # Calculate simple word overlap score
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / len(query_words)
                results.append({
                    "memory_id": memory_id,
                    "content": memory_item.content,
                    "score": score,
                    "timestamp": memory_item.timestamp.isoformat(),
                    "metadata": memory_item.metadata,
                    "entities": memory_item.entities,
                    "relationships": memory_item.relationships
                })
        
        return results
    
    def _apply_retention_policy(self) -> None:
        """Apply memory retention policy."""
        if self.retention_policy == "unlimited":
            return
        
        # Parse retention policy
        if isinstance(self.retention_policy, str) and "_days" in self.retention_policy:
            try:
                days = int(self.retention_policy.replace("_days", ""))
            except ValueError:
                days = 30
        else:
            days = 30
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Delete old items
        memory_ids_to_delete = []
        for memory_id, memory_item in self.memory_items.items():
            if memory_item.timestamp < cutoff_date:
                memory_ids_to_delete.append(memory_id)
        
        for memory_id in memory_ids_to_delete:
            self.delete_memory(memory_id)
        
        if memory_ids_to_delete:
            self.logger.info(f"Deleted {len(memory_ids_to_delete)} items based on retention policy")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "current_items": len(self.memory_items),
            "max_size": self.max_memory_size,
            "retention_policy": self.retention_policy
        }
