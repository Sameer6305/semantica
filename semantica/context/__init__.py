"""
Context Engineering Module

This module provides context engineering infrastructure for agents,
formalizing context as a graph of connections to enable meaningful
agent understanding and memory.

Context engineering layers:
- Prompting: Natural-language programming for agent goals
- Memory: RAG with vector databases for context retrieval
- Tools: MCP servers for consistent tool access
- Graphs: Knowledge graphs for formalized context as connections
"""

from .context_graph import ContextGraphBuilder, ContextNode, ContextEdge
from .entity_linker import EntityLinker, EntityLink, LinkedEntity
from .agent_memory import AgentMemory, MemoryItem
from .context_retriever import ContextRetriever, RetrievedContext

__all__ = [
    "ContextGraphBuilder",
    "ContextNode",
    "ContextEdge",
    "EntityLinker",
    "EntityLink",
    "LinkedEntity",
    "AgentMemory",
    "MemoryItem",
    "ContextRetriever",
    "RetrievedContext",
]
