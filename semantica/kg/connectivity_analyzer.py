"""
Connectivity Analyzer Module

Handles connectivity analysis for knowledge graphs including
connected components, shortest paths, and bridge identification.

Key Features:
    - Graph connectivity analysis
    - Connected components detection
    - Shortest path calculation
    - Bridge identification
    - Connectivity metrics and statistics

Main Classes:
    - ConnectivityAnalyzer: Main connectivity analysis engine
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque

from ..utils.logging import get_logger


class ConnectivityAnalyzer:
    """
    Connectivity analysis engine.
    
    • Analyzes graph connectivity
    • Calculates connectivity metrics
    • Identifies connected components
    • Processes path analysis
    
    Attributes:
        • connectivity_algorithms: Available connectivity algorithms
        • analysis_config: Configuration for connectivity analysis
        • component_detector: Connected components detection engine
        • path_analyzer: Path analysis and calculation engine
        
    Methods:
        • analyze_connectivity(): Analyze graph connectivity
        • find_connected_components(): Find connected components
        • calculate_shortest_paths(): Calculate shortest paths
        • identify_bridges(): Identify bridge edges
    """
    
    def __init__(self, **config):
        """
        Initialize connectivity analyzer.
        
        • Setup connectivity algorithms
        • Configure component detection
        • Initialize path analysis
        • Setup metric calculation
        """
        self.logger = get_logger("connectivity_analyzer")
        self.connectivity_algorithms = [
            "dfs", "bfs", "tarjan", "kosaraju"
        ]
        self.analysis_config = config.get("analysis_config", {})
        self.config = config
        
        # Try to use networkx if available
        try:
            import networkx as nx
            self.nx = nx
            self.use_networkx = True
        except ImportError:
            self.nx = None
            self.use_networkx = False
            self.logger.warning("NetworkX not available, using basic implementations")
    
    def analyze_connectivity(self, graph):
        """
        Analyze graph connectivity.
        
        • Calculate connectivity metrics
        • Identify connected components
        • Analyze graph structure
        • Return connectivity analysis
        
        Args:
            graph: Input graph for connectivity analysis
            
        Returns:
            dict: Comprehensive connectivity analysis results
        """
        self.logger.info("Analyzing graph connectivity")
        
        components_result = self.find_connected_components(graph)
        metrics = self.calculate_connectivity_metrics(graph)
        
        return {
            **components_result,
            **metrics,
            "is_connected": components_result.get("num_components", 0) == 1
        }
    
    def find_connected_components(self, graph):
        """
        Find connected components in graph.
        
        • Identify disconnected subgraphs
        • Calculate component sizes
        • Analyze component structure
        • Return component information
        
        Args:
            graph: Input graph for component analysis
            
        Returns:
            dict: Connected components analysis results
        """
        self.logger.info("Finding connected components")
        
        adjacency = self._build_adjacency(graph)
        visited = set()
        components = []
        
        # DFS to find components
        for node in adjacency:
            if node not in visited:
                component = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        for neighbor in adjacency.get(current, []):
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if component:
                    components.append(component)
        
        # Calculate statistics
        component_sizes = [len(comp) for comp in components]
        
        return {
            "components": components,
            "num_components": len(components),
            "component_sizes": component_sizes,
            "largest_component_size": max(component_sizes) if component_sizes else 0,
            "smallest_component_size": min(component_sizes) if component_sizes else 0
        }
    
    def calculate_shortest_paths(self, graph, source=None, target=None):
        """
        Calculate shortest paths in graph.
        
        • Find shortest paths between nodes
        • Calculate path lengths
        • Handle weighted and unweighted graphs
        • Return path information
        
        Args:
            graph: Input graph for path analysis
            source: Source node for path calculation
            target: Target node for path calculation
            
        Returns:
            dict: Shortest path analysis results
        """
        self.logger.info(f"Calculating shortest paths from {source} to {target}")
        
        adjacency = self._build_adjacency(graph)
        
        if source is None or target is None:
            # Calculate all pairs shortest paths
            return self._calculate_all_pairs_shortest_paths(adjacency)
        
        # Single pair shortest path
        path, distance = self._bfs_shortest_path(adjacency, source, target)
        
        return {
            "source": source,
            "target": target,
            "path": path,
            "distance": distance,
            "exists": path is not None
        }
    
    def identify_bridges(self, graph):
        """
        Identify bridge edges in graph.
        
        • Find edges whose removal disconnects graph
        • Calculate bridge importance
        • Analyze bridge impact
        • Return bridge information
        
        Args:
            graph: Input graph for bridge analysis
            
        Returns:
            dict: Bridge identification and analysis results
        """
        self.logger.info("Identifying bridge edges")
        
        adjacency = self._build_adjacency(graph)
        bridges = []
        
        # Get all edges
        edges = set()
        for source, targets in adjacency.items():
            for target in targets:
                edge = tuple(sorted([source, target]))
                edges.add(edge)
        
        # Check each edge
        for edge in edges:
            source, target = edge
            
            # Remove edge temporarily
            temp_adjacency = {k: [v for v in vs if v != target] for k, vs in adjacency.items()}
            temp_adjacency[source] = [v for v in temp_adjacency.get(source, []) if v != target]
            
            # Check connectivity
            components = self._find_components(temp_adjacency)
            
            # If more components, it's a bridge
            if len(components) > 1:
                bridges.append(edge)
        
        return {
            "bridges": bridges,
            "num_bridges": len(bridges),
            "bridge_edges": [{"source": s, "target": t} for s, t in bridges]
        }
    
    def calculate_connectivity_metrics(self, graph):
        """
        Calculate comprehensive connectivity metrics.
        
        • Calculate connectivity statistics
        • Analyze graph structure metrics
        • Compute connectivity indices
        • Return connectivity metrics
        
        Args:
            graph: Input graph for metrics calculation
            
        Returns:
            dict: Connectivity metrics and statistics
        """
        self.logger.info("Calculating connectivity metrics")
        
        adjacency = self._build_adjacency(graph)
        nodes = list(adjacency.keys())
        n = len(nodes)
        
        # Count edges
        total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
        
        # Calculate density
        max_edges = n * (n - 1) / 2 if n > 1 else 0
        density = total_edges / max_edges if max_edges > 0 else 0.0
        
        # Average degree
        degrees = [len(adjacency.get(node, [])) for node in nodes]
        avg_degree = sum(degrees) / n if n > 0 else 0.0
        
        return {
            "num_nodes": n,
            "num_edges": total_edges,
            "density": density,
            "avg_degree": avg_degree,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0
        }
    
    def analyze_graph_structure(self, graph):
        """
        Analyze overall graph structure and connectivity.
        
        • Analyze graph topology
        • Calculate structural metrics
        • Identify structural patterns
        • Return structure analysis
        
        Args:
            graph: Input graph for structure analysis
            
        Returns:
            dict: Graph structure analysis results
        """
        self.logger.info("Analyzing graph structure")
        
        connectivity = self.analyze_connectivity(graph)
        metrics = self.calculate_connectivity_metrics(graph)
        bridges = self.identify_bridges(graph)
        
        return {
            **connectivity,
            **metrics,
            **bridges,
            "structure_type": self._classify_structure(connectivity, metrics)
        }
    
    def _build_adjacency(self, graph) -> Dict[str, List[str]]:
        """Build adjacency list from graph."""
        adjacency = defaultdict(list)
        
        # Extract relationships
        relationships = []
        if hasattr(graph, "relationships"):
            relationships = graph.relationships
        elif hasattr(graph, "get_relationships"):
            relationships = graph.get_relationships()
        elif isinstance(graph, dict):
            relationships = graph.get("relationships", [])
        
        # Build adjacency
        for rel in relationships:
            source = rel.get("source") or rel.get("subject")
            target = rel.get("target") or rel.get("object")
            
            if source and target:
                if target not in adjacency[source]:
                    adjacency[source].append(target)
                if source not in adjacency[target]:
                    adjacency[target].append(source)
        
        return dict(adjacency)
    
    def _bfs_shortest_path(self, adjacency: Dict[str, List[str]], source: str, target: str) -> Tuple[Optional[List[str]], int]:
        """Find shortest path using BFS."""
        if source == target:
            return [source], 0
        
        queue = deque([(source, [source])])
        visited = {source}
        
        while queue:
            node, path = queue.popleft()
            
            for neighbor in adjacency.get(node, []):
                if neighbor == target:
                    return path + [target], len(path)
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None, -1
    
    def _calculate_all_pairs_shortest_paths(self, adjacency: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate all pairs shortest paths."""
        nodes = list(adjacency.keys())
        distances = {}
        paths = {}
        
        for source in nodes:
            distances[source] = {}
            paths[source] = {}
            
            for target in nodes:
                if source == target:
                    distances[source][target] = 0
                    paths[source][target] = [source]
                else:
                    path, distance = self._bfs_shortest_path(adjacency, source, target)
                    distances[source][target] = distance
                    paths[source][target] = path
        
        return {
            "distances": distances,
            "paths": paths,
            "avg_path_length": self._calculate_avg_path_length(distances)
        }
    
    def _calculate_avg_path_length(self, distances: Dict[str, Dict[str, int]]) -> float:
        """Calculate average path length."""
        total = 0
        count = 0
        
        for source_distances in distances.values():
            for distance in source_distances.values():
                if distance > 0:
                    total += distance
                    count += 1
        
        return total / count if count > 0 else 0.0
    
    def _find_components(self, adjacency: Dict[str, List[str]]) -> List[List[str]]:
        """Find connected components using DFS."""
        visited = set()
        components = []
        
        for node in adjacency:
            if node not in visited:
                component = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        
                        for neighbor in adjacency.get(current, []):
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                if component:
                    components.append(component)
        
        return components
    
    def _classify_structure(self, connectivity: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Classify graph structure type."""
        num_components = connectivity.get("num_components", 1)
        density = metrics.get("density", 0.0)
        
        if num_components > 1:
            return "disconnected"
        elif density > 0.5:
            return "dense"
        elif density < 0.1:
            return "sparse"
        else:
            return "moderate"
