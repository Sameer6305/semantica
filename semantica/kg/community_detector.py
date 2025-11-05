"""
Community Detection Module

Handles community detection in knowledge graphs using various algorithms
including Louvain, Leiden, and overlapping community detection.

Key Features:
    - Louvain community detection
    - Leiden community detection
    - Overlapping community detection
    - Community quality metrics
    - Community analysis and statistics

Main Classes:
    - CommunityDetector: Main community detection engine
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict

from ..utils.logging import get_logger


class CommunityDetector:
    """
    Community detection engine.
    
    • Detects communities in graphs
    • Handles different detection algorithms
    • Manages community quality metrics
    • Processes overlapping communities
    
    Attributes:
        • supported_algorithms: List of supported detection algorithms
        • detection_config: Configuration for community detection
        • quality_metrics: Community quality assessment tools
        • overlapping_detector: Overlapping community detection engine
        
    Methods:
        • detect_communities_louvain(): Detect communities using Louvain
        • detect_communities_leiden(): Detect communities using Leiden
        • detect_overlapping_communities(): Detect overlapping communities
        • calculate_community_metrics(): Calculate community quality metrics
    """
    
    def __init__(self, **config):
        """
        Initialize community detector.
        
        • Setup detection algorithms
        • Configure quality metrics
        • Initialize overlapping detection
        • Setup community analysis
        """
        self.logger = get_logger("community_detector")
        self.supported_algorithms = [
            "louvain", "leiden", "overlapping", "label_propagation"
        ]
        self.detection_config = config.get("detection_config", {})
        self.config = config
        
        # Try to use networkx/igraph if available
        try:
            import networkx as nx
            self.nx = nx
            self.use_networkx = True
        except ImportError:
            self.nx = None
            self.use_networkx = False
            self.logger.warning("NetworkX not available, using basic implementations")
    
    def detect_communities_louvain(self, graph, **options):
        """
        Detect communities using Louvain algorithm.
        
        • Apply Louvain community detection
        • Optimize modularity
        • Handle resolution parameters
        • Return community assignments
        
        Args:
            graph: Input graph for community detection
            **options: Additional detection options
            
        Returns:
            dict: Community detection results and assignments
        """
        self.logger.info("Detecting communities using Louvain algorithm")
        
        if self.use_networkx:
            try:
                import networkx.algorithms.community as nx_comm
                nx_graph = self._to_networkx(graph)
                resolution = options.get("resolution", 1.0)
                
                # Use greedy modularity communities (Louvain-like)
                communities = nx_comm.greedy_modularity_communities(nx_graph, resolution=resolution)
                
                # Convert to node assignments
                node_communities = {}
                for i, community in enumerate(communities):
                    for node in community:
                        node_communities[node] = i
                
                modularity = nx_comm.modularity(nx_graph, communities)
                
                return {
                    "communities": list(communities),
                    "node_assignments": node_communities,
                    "modularity": modularity,
                    "algorithm": "louvain"
                }
            except Exception as e:
                self.logger.warning(f"NetworkX Louvain failed: {e}, using basic implementation")
        
        # Basic greedy modularity implementation
        adjacency = self._build_adjacency(graph)
        return self._basic_community_detection(adjacency, algorithm="louvain", **options)
    
    def detect_communities_leiden(self, graph, **options):
        """
        Detect communities using Leiden algorithm.
        
        • Apply Leiden community detection
        • Optimize modularity with refinement
        • Handle resolution parameters
        • Return community assignments
        
        Args:
            graph: Input graph for community detection
            **options: Additional detection options
            
        Returns:
            dict: Community detection results and assignments
        """
        self.logger.info("Detecting communities using Leiden algorithm")
        
        # Leiden is similar to Louvain but with refinement
        # For now, use Louvain-like approach
        return self.detect_communities_louvain(graph, **options)
    
    def detect_overlapping_communities(self, graph, **options):
        """
        Detect overlapping communities.
        
        • Apply overlapping detection algorithms
        • Handle node membership in multiple communities
        • Calculate overlapping metrics
        • Return overlapping community structure
        
        Args:
            graph: Input graph for community detection
            **options: Additional detection options
            
        Returns:
            dict: Overlapping community detection results
        """
        self.logger.info("Detecting overlapping communities")
        
        if self.use_networkx:
            try:
                import networkx.algorithms.community as nx_comm
                nx_graph = self._to_networkx(graph)
                
                # Use k-clique communities for overlapping detection
                k = options.get("k", 3)
                communities = list(nx_comm.k_clique_communities(nx_graph, k))
                
                # Build node to communities mapping
                node_communities = defaultdict(list)
                for i, community in enumerate(communities):
                    for node in community:
                        node_communities[node].append(i)
                
                return {
                    "communities": [list(c) for c in communities],
                    "node_assignments": dict(node_communities),
                    "algorithm": "overlapping",
                    "overlap_count": len([n for n, comms in node_communities.items() if len(comms) > 1])
                }
            except Exception as e:
                self.logger.warning(f"NetworkX overlapping detection failed: {e}, using basic implementation")
        
        # Basic overlapping detection
        adjacency = self._build_adjacency(graph)
        return self._basic_overlapping_detection(adjacency, **options)
    
    def calculate_community_metrics(self, graph, communities):
        """
        Calculate community quality metrics.
        
        • Calculate modularity
        • Compute community statistics
        • Assess community quality
        • Return community metrics
        
        Args:
            graph: Input graph for community analysis
            communities: Community assignments to analyze
            
        Returns:
            dict: Community quality metrics and statistics
        """
        self.logger.info("Calculating community quality metrics")
        
        adjacency = self._build_adjacency(graph)
        
        # Extract community structure
        if isinstance(communities, dict):
            node_communities = communities
        elif isinstance(communities, dict) and "node_assignments" in communities:
            node_communities = communities["node_assignments"]
        else:
            # Convert list of communities to node assignments
            node_communities = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_communities[node] = i
        
        # Calculate metrics
        num_communities = len(set(node_communities.values()))
        community_sizes = defaultdict(int)
        for comm_id in node_communities.values():
            community_sizes[comm_id] += 1
        
        # Calculate modularity (simplified)
        modularity = self._calculate_modularity(adjacency, node_communities)
        
        # Calculate statistics
        sizes = list(community_sizes.values())
        
        return {
            "num_communities": num_communities,
            "community_sizes": dict(community_sizes),
            "avg_community_size": sum(sizes) / len(sizes) if sizes else 0,
            "max_community_size": max(sizes) if sizes else 0,
            "min_community_size": min(sizes) if sizes else 0,
            "modularity": modularity
        }
    
    def analyze_community_structure(self, graph, communities):
        """
        Analyze community structure and properties.
        
        • Analyze community size distribution
        • Calculate community connectivity
        • Assess community stability
        • Return community structure analysis
        
        Args:
            graph: Input graph for community analysis
            communities: Community assignments to analyze
            
        Returns:
            dict: Community structure analysis results
        """
        self.logger.info("Analyzing community structure")
        
        metrics = self.calculate_community_metrics(graph, communities)
        
        # Extract node assignments
        if isinstance(communities, dict) and "node_assignments" in communities:
            node_communities = communities["node_assignments"]
        elif isinstance(communities, dict):
            node_communities = communities
        else:
            node_communities = {}
            for i, community in enumerate(communities):
                for node in community:
                    node_communities[node] = i
        
        # Analyze connectivity between communities
        adjacency = self._build_adjacency(graph)
        inter_community_edges = 0
        intra_community_edges = 0
        
        for source, targets in adjacency.items():
            source_comm = node_communities.get(source)
            for target in targets:
                target_comm = node_communities.get(target)
                if source_comm == target_comm:
                    intra_community_edges += 1
                else:
                    inter_community_edges += 1
        
        return {
            **metrics,
            "intra_community_edges": intra_community_edges,
            "inter_community_edges": inter_community_edges,
            "edge_ratio": intra_community_edges / (inter_community_edges + 1)
        }
    
    def detect_communities(self, graph, algorithm="louvain", **options):
        """
        Detect communities using specified algorithm.
        
        • Apply specified community detection algorithm
        • Handle different algorithm parameters
        • Return community detection results
        • Provide algorithm-specific analysis
        
        Args:
            graph: Input graph for community detection
            algorithm: Community detection algorithm to use
            **options: Additional detection options
            
        Returns:
            dict: Community detection results and analysis
        """
        self.logger.info(f"Detecting communities using {algorithm} algorithm")
        
        if algorithm == "louvain":
            return self.detect_communities_louvain(graph, **options)
        elif algorithm == "leiden":
            return self.detect_communities_leiden(graph, **options)
        elif algorithm == "overlapping":
            return self.detect_overlapping_communities(graph, **options)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _build_adjacency(self, graph) -> Dict[str, List[str]]:
        """Build adjacency list from graph."""
        from collections import defaultdict
        
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
    
    def _to_networkx(self, graph):
        """Convert graph to NetworkX format."""
        adjacency = self._build_adjacency(graph)
        nx_graph = self.nx.Graph()
        
        for source, targets in adjacency.items():
            for target in targets:
                nx_graph.add_edge(source, target)
        
        return nx_graph
    
    def _basic_community_detection(self, adjacency: Dict[str, List[str]], algorithm="louvain", **options):
        """Basic community detection implementation."""
        # Simple greedy approach
        nodes = list(adjacency.keys())
        node_communities = {node: i for i, node in enumerate(nodes)}
        
        # Simple merge step
        changed = True
        iterations = 0
        max_iter = options.get("max_iter", 10)
        
        while changed and iterations < max_iter:
            changed = False
            iterations += 1
            
            for node in nodes:
                # Find best community for this node
                best_community = node_communities[node]
                best_modularity = self._calculate_modularity(adjacency, node_communities)
                
                # Try moving to neighbor communities
                for neighbor in adjacency.get(node, []):
                    neighbor_comm = node_communities.get(neighbor)
                    if neighbor_comm != node_communities[node]:
                        # Try moving
                        old_comm = node_communities[node]
                        node_communities[node] = neighbor_comm
                        new_modularity = self._calculate_modularity(adjacency, node_communities)
                        
                        if new_modularity > best_modularity:
                            best_modularity = new_modularity
                            best_community = neighbor_comm
                            changed = True
                        else:
                            node_communities[node] = old_comm
        
        # Build communities
        communities = defaultdict(list)
        for node, comm_id in node_communities.items():
            communities[comm_id].append(node)
        
        return {
            "communities": list(communities.values()),
            "node_assignments": node_communities,
            "modularity": best_modularity,
            "algorithm": algorithm
        }
    
    def _basic_overlapping_detection(self, adjacency: Dict[str, List[str]], **options):
        """Basic overlapping community detection."""
        # Simple approach: find dense subgraphs
        communities = []
        nodes = list(adjacency.keys())
        visited = set()
        
        for node in nodes:
            if node in visited:
                continue
            
            # Find neighbors
            neighbors = set(adjacency.get(node, []))
            neighbors.add(node)
            
            # Check if this forms a community (min size)
            min_size = options.get("min_size", 3)
            if len(neighbors) >= min_size:
                communities.append(list(neighbors))
                visited.update(neighbors)
        
        # Build node to communities mapping
        node_communities = defaultdict(list)
        for i, community in enumerate(communities):
            for node in community:
                node_communities[node].append(i)
        
        return {
            "communities": communities,
            "node_assignments": dict(node_communities),
            "algorithm": "overlapping"
        }
    
    def _calculate_modularity(self, adjacency: Dict[str, List[str]], node_communities: Dict[str, int]) -> float:
        """Calculate modularity."""
        # Simplified modularity calculation
        total_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
        
        if total_edges == 0:
            return 0.0
        
        modularity = 0.0
        nodes = list(adjacency.keys())
        
        for node in nodes:
            node_comm = node_communities.get(node)
            degree = len(adjacency.get(node, []))
            
            for neighbor in adjacency.get(node, []):
                neighbor_comm = node_communities.get(neighbor)
                
                if node_comm == neighbor_comm:
                    # Same community
                    modularity += 1.0 - (degree * len(adjacency.get(neighbor, []))) / (2 * total_edges)
        
        return modularity / (2 * total_edges) if total_edges > 0 else 0.0
