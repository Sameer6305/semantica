"""
Triple store module for Semantica framework.

This module provides triple store integration and management
for RDF data storage and querying.

Exports:
    - TripleManager: Triple store management and CRUD operations
    - BlazegraphAdapter: Blazegraph integration
    - JenaAdapter: Apache Jena integration
    - RDF4JAdapter: Eclipse RDF4J integration
    - VirtuosoAdapter: Virtuoso RDF store integration
    - QueryEngine: SPARQL query execution and optimization
    - BulkLoader: High-volume data loading
"""

from .triple_manager import (
    TripleManager,
    TripleStore
)
from .blazegraph_adapter import (
    BlazegraphAdapter
)
from .jena_adapter import (
    JenaAdapter
)
from .rdf4j_adapter import (
    RDF4JAdapter
)
from .virtuoso_adapter import (
    VirtuosoAdapter
)
from .query_engine import (
    QueryEngine,
    QueryResult,
    QueryPlan
)
from .bulk_loader import (
    BulkLoader,
    LoadProgress
)

__all__ = [
    # Triple management
    "TripleManager",
    "TripleStore",
    
    # Store adapters
    "BlazegraphAdapter",
    "JenaAdapter",
    "RDF4JAdapter",
    "VirtuosoAdapter",
    
    # Query engine
    "QueryEngine",
    "QueryResult",
    "QueryPlan",
    
    # Bulk loading
    "BulkLoader",
    "LoadProgress",
]
