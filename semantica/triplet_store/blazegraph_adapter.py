"""
Blazegraph Adapter Module

This module provides Blazegraph integration for RDF storage and SPARQL
querying, enabling connection to Blazegraph instances with namespace
management and bulk loading capabilities.

Key Features:
    - Blazegraph connection and authentication
    - SPARQL query execution
    - Bulk data loading and management
    - Namespace and graph management
    - REST API integration
    - Performance optimization

Main Classes:
    - BlazegraphAdapter: Main Blazegraph integration adapter

Example Usage:
    >>> from semantica.triplet_store import BlazegraphAdapter
    >>> adapter = BlazegraphAdapter(endpoint="http://localhost:9999/blazegraph", namespace="kb")
    >>> result = adapter.execute_sparql(sparql_query)
    >>> load_result = adapter.bulk_load(triples)
    >>> namespace_result = adapter.create_namespace("new_namespace")

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from ..semantic_extract.triple_extractor import Triple
from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker


class BlazegraphAdapter:
    """
    Blazegraph triplet store adapter.

    • Blazegraph connection and authentication
    • SPARQL query execution
    • Bulk data loading and management
    • Namespace and graph management
    • Performance optimization
    • Error handling and recovery
    """

    def __init__(self, endpoint: str, **config):
        """
        Initialize Blazegraph adapter.

        Args:
            endpoint: Blazegraph endpoint URL
            **config: Additional configuration:
                - namespace: Namespace name (default: "kb")
                - username: Username for authentication
                - password: Password for authentication
                - timeout: Request timeout (default: 30)
        """
        self.logger = get_logger("blazegraph_adapter")
        self.config = config
        self.progress_tracker = get_progress_tracker()

        self.endpoint = endpoint.rstrip("/")
        self.namespace = config.get("namespace", "kb")
        self.username = config.get("username")
        self.password = config.get("password")
        self.timeout = config.get("timeout", 30)

        self.connected = False
        self._connect()

    def _connect(self) -> None:
        """Connect to Blazegraph instance."""
        try:
            # Test connection
            sparql_endpoint = self._get_sparql_endpoint()
            response = requests.get(
                sparql_endpoint,
                params={"query": "SELECT * WHERE { ?s ?p ?o } LIMIT 1"},
                timeout=self.timeout,
                auth=(self.username, self.password)
                if self.username and self.password
                else None,
            )

            if response.status_code == 200:
                self.connected = True
                self.logger.info(f"Connected to Blazegraph: {self.endpoint}")
            else:
                self.logger.warning(
                    f"Blazegraph connection test failed: {response.status_code}"
                )
        except Exception as e:
            self.logger.warning(f"Could not connect to Blazegraph: {e}")

    def _get_sparql_endpoint(self) -> str:
        """Get SPARQL endpoint URL."""
        return urljoin(self.endpoint, f"/blazegraph/namespace/{self.namespace}/sparql")

    def _get_update_endpoint(self) -> str:
        """Get SPARQL Update endpoint URL."""
        return urljoin(self.endpoint, f"/blazegraph/namespace/{self.namespace}/sparql")

    def execute_sparql(self, query: str, **options) -> Dict[str, Any]:
        """
        Execute SPARQL query.

        Args:
            query: SPARQL query string
            **options: Additional options

        Returns:
            Query results
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="triplet_store",
            submodule="BlazegraphAdapter",
            message="Executing SPARQL query on Blazegraph",
        )

        try:
            if not self.connected:
                self.progress_tracker.stop_tracking(
                    tracking_id, status="failed", message="Not connected to Blazegraph"
                )
                raise ProcessingError("Not connected to Blazegraph")

            sparql_endpoint = self._get_sparql_endpoint()

            self.progress_tracker.update_tracking(
                tracking_id, message="Sending query to Blazegraph endpoint..."
            )
            response = requests.post(
                sparql_endpoint,
                data={"query": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout,
                auth=(self.username, self.password)
                if self.username and self.password
                else None,
            )

            response.raise_for_status()

            # Parse JSON response
            self.progress_tracker.update_tracking(
                tracking_id, message="Parsing query results..."
            )
            result_data = response.json()

            result = {
                "success": True,
                "bindings": result_data.get("results", {}).get("bindings", []),
                "variables": result_data.get("head", {}).get("vars", []),
                "metadata": {"query": query, "endpoint": sparql_endpoint},
            }

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Query executed: {len(result['bindings'])} results",
            )
            return result
        except Exception as e:
            self.logger.error(f"SPARQL query failed: {e}")
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise ProcessingError(f"SPARQL query failed: {e}")

    def bulk_load(self, triples: List[Triple], **options) -> Dict[str, Any]:
        """
        Load triples in bulk.

        Args:
            triples: List of triples
            **options: Additional options:
                - format: RDF format (turtle, ntriples, rdfxml)
                - graph: Named graph URI

        Returns:
            Load status
        """
        if not self.connected:
            raise ProcessingError("Not connected to Blazegraph")

        # Convert triples to RDF format
        format = options.get("format", "turtle")
        rdf_data = self._triples_to_rdf(triples, format)

        # Upload endpoint
        upload_endpoint = urljoin(
            self.endpoint, f"/blazegraph/namespace/{self.namespace}/sparql"
        )

        try:
            # Use SPARQL INSERT for bulk loading
            graph = options.get("graph", "")
            graph_clause = f"GRAPH <{graph}>" if graph else ""

            # Build INSERT query
            insert_data = self._build_insert_data(triples)
            query = f"INSERT DATA {graph_clause} {{ {insert_data} }}"

            response = requests.post(
                upload_endpoint,
                data={"update": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout * 2,  # Longer timeout for bulk operations
                auth=(self.username, self.password)
                if self.username and self.password
                else None,
            )

            response.raise_for_status()

            return {
                "success": True,
                "triples_loaded": len(triples),
                "namespace": self.namespace,
            }
        except Exception as e:
            self.logger.error(f"Bulk load failed: {e}")
            raise ProcessingError(f"Bulk load failed: {e}")

    def _triples_to_rdf(self, triples: List[Triple], format: str = "turtle") -> str:
        """Convert triples to RDF format."""
        if format == "turtle":
            lines = []
            for triple in triples:
                lines.append(
                    f"<{triple.subject}> <{triple.predicate}> <{triple.object}> ."
                )
            return "\n".join(lines)
        else:
            # For other formats, use simple turtle conversion
            return self._triples_to_rdf(triples, "turtle")

    def _build_insert_data(self, triples: List[Triple]) -> str:
        """Build SPARQL INSERT DATA clause."""
        lines = []
        for triple in triples:
            lines.append(f"<{triple.subject}> <{triple.predicate}> <{triple.object}> .")
        return " ".join(lines)

    def add_triple(self, triple: Triple, **options) -> Dict[str, Any]:
        """Add single triple."""
        return self.bulk_load([triple], **options)

    def add_triples(self, triples: List[Triple], **options) -> Dict[str, Any]:
        """Add multiple triples."""
        return self.bulk_load(triples, **options)

    def get_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        **options,
    ) -> List[Triple]:
        """Get triples matching criteria."""
        # Build SPARQL query
        where_clauses = []
        if subject:
            where_clauses.append(f"?s = <{subject}>")
        if predicate:
            where_clauses.append(f"?p = <{predicate}>")
        if object:
            where_clauses.append(f"?o = <{object}>")

        where_clause = " ".join(where_clauses) if where_clauses else ""
        query = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o {where_clause} }}"

        result = self.execute_sparql(query, **options)

        # Convert bindings to triples
        triples = []
        for binding in result["bindings"]:
            triples.append(
                Triple(
                    subject=binding.get("s", {}).get("value", ""),
                    predicate=binding.get("p", {}).get("value", ""),
                    object=binding.get("o", {}).get("value", ""),
                    metadata={"source": "blazegraph"},
                )
            )

        return triples

    def delete_triple(self, triple: Triple, **options) -> Dict[str, Any]:
        """Delete triple."""
        if not self.connected:
            raise ProcessingError("Not connected to Blazegraph")

        update_endpoint = self._get_update_endpoint()

        query = f"DELETE DATA {{ <{triple.subject}> <{triple.predicate}> <{triple.object}> }}"

        try:
            response = requests.post(
                update_endpoint,
                data={"update": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout,
                auth=(self.username, self.password)
                if self.username and self.password
                else None,
            )

            response.raise_for_status()

            return {"success": True}
        except Exception as e:
            self.logger.error(f"Delete triple failed: {e}")
            raise ProcessingError(f"Delete triple failed: {e}")

    def create_namespace(self, namespace: str, **options) -> Dict[str, Any]:
        """
        Create new namespace.

        Args:
            namespace: Namespace name
            **options: Additional options

        Returns:
            Operation status
        """
        # Blazegraph namespace creation via REST API
        create_endpoint = urljoin(self.endpoint, "/blazegraph/namespace")

        try:
            response = requests.post(
                create_endpoint,
                json={"namespace": namespace, **options},
                timeout=self.timeout,
                auth=(self.username, self.password)
                if self.username and self.password
                else None,
            )

            response.raise_for_status()

            return {"success": True, "namespace": namespace}
        except Exception as e:
            self.logger.error(f"Create namespace failed: {e}")
            raise ProcessingError(f"Create namespace failed: {e}")
