"""
Virtuoso Adapter Module

This module provides Virtuoso RDF store integration for RDF storage and
SPARQL querying, supporting cluster connections and query optimization.

Key Features:
    - Virtuoso connection and authentication
    - SPARQL query execution
    - Bulk data loading and management
    - Graph and namespace management
    - Cluster connection support
    - Query optimization

Main Classes:
    - VirtuosoAdapter: Main Virtuoso integration adapter

Example Usage:
    >>> from semantica.triplet_store import VirtuosoAdapter
    >>> adapter = VirtuosoAdapter(endpoint="http://localhost:8890/sparql", username="dba", password="dba")
    >>> result = adapter.execute_sparql(sparql_query)
    >>> load_result = adapter.bulk_load(triples, graph="http://example.org/graph")
    >>> cluster_status = adapter.connect_cluster(cluster_config)

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from ..semantic_extract.triple_extractor import Triple
from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


class VirtuosoAdapter:
    """
    Virtuoso RDF store adapter.

    • Virtuoso connection and authentication
    • SPARQL query execution
    • Bulk data loading and management
    • Graph and namespace management
    • Performance optimization
    • Error handling and recovery
    """

    def __init__(self, endpoint: str, **config):
        """
        Initialize Virtuoso adapter.

        Args:
            endpoint: Virtuoso endpoint URL
            **config: Additional configuration:
                - username: Username for authentication
                - password: Password for authentication
                - timeout: Request timeout (default: 30)
                - graph: Default graph URI
        """
        self.logger = get_logger("virtuoso_adapter")
        self.config = config
        self.progress_tracker = get_progress_tracker()

        self.endpoint = endpoint.rstrip("/")
        self.username = config.get("username", "dba")
        self.password = config.get("password", "dba")
        self.timeout = config.get("timeout", 30)
        self.default_graph = config.get("graph", "")

        self.connected = False
        self._connect()

    def _connect(self) -> None:
        """Connect to Virtuoso instance."""
        try:
            # Test connection with simple query
            sparql_endpoint = self._get_sparql_endpoint()
            response = requests.get(
                sparql_endpoint,
                params={"query": "SELECT * WHERE { ?s ?p ?o } LIMIT 1"},
                timeout=self.timeout,
                auth=(self.username, self.password),
            )

            if response.status_code == 200:
                self.connected = True
                self.logger.info(f"Connected to Virtuoso: {self.endpoint}")
            else:
                self.logger.warning(
                    f"Virtuoso connection test failed: {response.status_code}"
                )
        except Exception as e:
            self.logger.warning(f"Could not connect to Virtuoso: {e}")

    def _get_sparql_endpoint(self) -> str:
        """Get SPARQL endpoint URL."""
        return urljoin(self.endpoint, "/sparql")

    def _get_sparql_update_endpoint(self) -> str:
        """Get SPARQL Update endpoint URL."""
        return urljoin(self.endpoint, "/sparql-auth")

    def connect_cluster(
        self, cluster_config: Dict[str, Any], **options
    ) -> Dict[str, Any]:
        """
        Connect to Virtuoso cluster.

        Args:
            cluster_config: Cluster configuration
            **options: Additional options

        Returns:
            Connection status
        """
        # Virtuoso cluster connection
        # This would typically involve multiple endpoints
        endpoints = cluster_config.get("endpoints", [self.endpoint])

        connections = []
        for endpoint in endpoints:
            try:
                adapter = VirtuosoAdapter(endpoint, **cluster_config)
                if adapter.connected:
                    connections.append(endpoint)
            except Exception as e:
                self.logger.warning(f"Failed to connect to {endpoint}: {e}")

        return {
            "success": len(connections) > 0,
            "connected_endpoints": connections,
            "total_endpoints": len(endpoints),
        }

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
            submodule="VirtuosoAdapter",
            message="Executing SPARQL query on Virtuoso",
        )

        try:
            if not self.connected:
                self.progress_tracker.stop_tracking(
                    tracking_id, status="failed", message="Not connected to Virtuoso"
                )
                raise ProcessingError("Not connected to Virtuoso")

            sparql_endpoint = self._get_sparql_endpoint()

            self.progress_tracker.update_tracking(
                tracking_id, message="Sending query to Virtuoso endpoint..."
            )
            response = requests.get(
                sparql_endpoint,
                params={"query": query, "default-graph-uri": self.default_graph}
                if self.default_graph
                else {"query": query},
                headers={"Accept": "application/sparql-results+json"},
                timeout=self.timeout,
                auth=(self.username, self.password),
            )

            response.raise_for_status()

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

    def optimize_queries(self, queries: List[str], **options) -> List[str]:
        """
        Optimize query performance.

        Args:
            queries: List of SPARQL queries
            **options: Optimization options

        Returns:
            List of optimized queries
        """
        optimized = []

        for query in queries:
            # Basic optimization: remove unnecessary whitespace
            opt_query = " ".join(query.split())

            # Add LIMIT if not present (for SELECT queries)
            if "SELECT" in query.upper() and "LIMIT" not in query.upper():
                opt_query += " LIMIT 1000"

            optimized.append(opt_query)

        return optimized

    def bulk_load(self, triples: List[Triple], **options) -> Dict[str, Any]:
        """
        Load triples in bulk.

        Args:
            triples: List of triples
            **options: Additional options:
                - graph: Named graph URI
                - format: RDF format (turtle, ntriples)

        Returns:
            Load status
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="triplet_store",
            submodule="VirtuosoAdapter",
            message=f"Bulk loading {len(triples)} triples to Virtuoso",
        )

        try:
            if not self.connected:
                self.progress_tracker.stop_tracking(
                    tracking_id, status="failed", message="Not connected to Virtuoso"
                )
                raise ProcessingError("Not connected to Virtuoso")

            graph = options.get("graph", self.default_graph)
            format = options.get("format", "turtle")

            # Convert triples to RDF
            self.progress_tracker.update_tracking(
                tracking_id, message="Converting triples to RDF format..."
            )
            rdf_data = self._triples_to_rdf(triples, format)

            # Use SPARQL INSERT for bulk loading
            update_endpoint = self._get_sparql_update_endpoint()

            graph_clause = f"GRAPH <{graph}>" if graph else ""
            insert_data = self._build_insert_data(triples)
            query = f"INSERT DATA {graph_clause} {{ {insert_data} }}"

            self.progress_tracker.update_tracking(
                tracking_id, message="Sending bulk load request..."
            )
            response = requests.post(
                update_endpoint,
                data={"query": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout * 2,
                auth=(self.username, self.password),
            )

            response.raise_for_status()

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Bulk loaded {len(triples)} triples",
            )
            return {"success": True, "triples_loaded": len(triples), "graph": graph}
        except Exception as e:
            self.logger.error(f"Bulk load failed: {e}")
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
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
        else:  # ntriples
            lines = []
            for triple in triples:
                lines.append(
                    f"<{triple.subject}> <{triple.predicate}> <{triple.object}> ."
                )
            return "\n".join(lines)

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
                    metadata={"source": "virtuoso"},
                )
            )

        return triples

    def delete_triple(self, triple: Triple, **options) -> Dict[str, Any]:
        """Delete triple."""
        if not self.connected:
            raise ProcessingError("Not connected to Virtuoso")

        update_endpoint = self._get_sparql_update_endpoint()

        query = f"DELETE DATA {{ <{triple.subject}> <{triple.predicate}> <{triple.object}> }}"

        try:
            response = requests.post(
                update_endpoint,
                data={"query": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout,
                auth=(self.username, self.password),
            )

            response.raise_for_status()

            return {"success": True}
        except Exception as e:
            self.logger.error(f"Delete triple failed: {e}")
            raise ProcessingError(f"Delete triple failed: {e}")
