"""
Graph Analytics Module

Handles comprehensive graph analytics including centrality measures, community detection, and connectivity analysis.

Key Features:
    - Centrality measures (degree, betweenness, closeness, eigenvector)
    - Community detection algorithms
    - Connectivity analysis
    - Graph metrics and statistics
    - Path analysis and shortest paths

Main Classes:
    - GraphAnalyzer: Main graph analytics class
"""

from .centrality_calculator import CentralityCalculator
from .community_detector import CommunityDetector
from .connectivity_analyzer import ConnectivityAnalyzer


class GraphAnalyzer:
    """
    Comprehensive graph analytics handler.
    
    • Performs graph analytics and metrics calculation
    • Calculates centrality measures for nodes
    • Detects communities and clusters in graphs
    • Analyzes graph connectivity and structure
    • Provides graph statistics and insights
    • Supports various graph algorithms
    
    Attributes:
        • centrality_calculator: Centrality measures calculator
        • community_detector: Community detection engine
        • connectivity_analyzer: Connectivity analysis engine
        • metrics_calculator: Graph metrics calculator
        • supported_algorithms: List of supported algorithms
        
    Methods:
        • analyze_graph(): Perform comprehensive graph analysis
        • calculate_centrality(): Calculate centrality measures
        • detect_communities(): Detect graph communities
        • analyze_connectivity(): Analyze graph connectivity
    """
    
    def __init__(
        self,
        config=None,
        enable_temporal=False,
        temporal_granularity="day",
        **kwargs
    ):
        """
        Initialize graph analyzer.
        
        • Setup graph analysis algorithms
        • Configure centrality calculations
        • Initialize community detection
        • Setup connectivity analysis
        • Configure metrics calculation
        • Enable temporal analysis if requested
        
        Args:
            config: Configuration dictionary
            enable_temporal: Enable temporal graph analysis
            temporal_granularity: Time granularity for temporal analysis
            **kwargs: Additional configuration options
        """
        self.config = config or {}
        self.enable_temporal = enable_temporal
        self.temporal_granularity = temporal_granularity
        self.centrality_calculator = CentralityCalculator(**self.config)
        self.community_detector = CommunityDetector(**self.config)
        self.connectivity_analyzer = ConnectivityAnalyzer(**self.config)
        
        # Initialize graph analysis components
        from ..utils.logging import get_logger
        self.logger = get_logger("graph_analyzer")
    
    def analyze_graph(self, graph, **options):
        """
        Perform comprehensive graph analysis.
        
        • Calculate graph metrics and statistics
        • Analyze graph structure and properties
        • Identify key nodes and relationships
        • Detect patterns and anomalies
        • Return comprehensive analysis results
        """
        self.logger.info("Performing comprehensive graph analysis")
        
        # Calculate centrality
        centrality = self.calculate_centrality(graph, **options)
        
        # Detect communities
        communities = self.detect_communities(graph, **options)
        
        # Analyze connectivity
        connectivity = self.analyze_connectivity(graph, **options)
        
        # Compute metrics
        metrics = self.compute_metrics(graph=graph, **options)
        
        return {
            "centrality": centrality,
            "communities": communities,
            "connectivity": connectivity,
            "metrics": metrics
        }
    
    def calculate_centrality(self, graph, centrality_type="degree", **options):
        """
        Calculate centrality measures for graph nodes.
        
        • Apply centrality algorithms
        • Calculate centrality scores
        • Rank nodes by centrality
        • Handle different centrality types
        • Return centrality results
        """
        return self.centrality_calculator.calculate_all_centrality(
            graph, centrality_types=[centrality_type]
        )
    
    def detect_communities(self, graph, algorithm="louvain", **options):
        """
        Detect communities in graph.
        
        • Apply community detection algorithms
        • Identify community structures
        • Calculate community metrics
        • Handle overlapping communities
        • Return community detection results
        """
        return self.community_detector.detect_communities(
            graph, algorithm=algorithm, **options
        )
    
    def analyze_connectivity(self, graph, **options):
        """
        Analyze graph connectivity and structure.
        
        • Calculate connectivity metrics
        • Identify connected components
        • Analyze path lengths and distances
        • Detect bottlenecks and bridges
        • Return connectivity analysis
        """
        return self.connectivity_analyzer.analyze_connectivity(graph, **options)
    
    def compute_metrics(self, graph=None, at_time=None, time_range=None, **options):
        """
        Compute comprehensive graph metrics.
        
        • Calculate graph statistics
        • Compute structural metrics
        • Analyze graph properties
        • Support temporal metrics if temporal enabled
        • Return metrics dictionary
        
        Args:
            graph: Graph to analyze (if not provided, uses stored graph)
            at_time: Calculate metrics at specific time point (temporal graphs)
            time_range: Calculate metrics for time range (temporal graphs)
            **options: Additional metric calculation options
            
        Returns:
            Dictionary of graph metrics
        """
        if graph is None:
            return {}
        
        # Get connectivity metrics
        connectivity_metrics = self.connectivity_analyzer.calculate_connectivity_metrics(graph)
        
        # Get entities and relationships
        entities = graph.get("entities", []) if isinstance(graph, dict) else []
        relationships = graph.get("relationships", []) if isinstance(graph, dict) else []
        
        metrics = {
            "num_nodes": len(entities),
            "num_edges": len(relationships),
            **connectivity_metrics
        }
        
        return metrics
    
    def analyze_temporal_evolution(
        self,
        graph,
        start_time=None,
        end_time=None,
        metrics=["node_count", "edge_count", "density", "communities"],
        interval=None,
        **options
    ):
        """
        Analyze temporal evolution of graph.
        
        Args:
            graph: Temporal knowledge graph
            start_time: Start of analysis period
            end_time: End of analysis period
            metrics: Metrics to track over time
            interval: Time interval for analysis snapshots
            **options: Additional analysis options
            
        Returns:
            Evolution analysis results with time series data
        """
        self.logger.info("Analyzing temporal evolution")
        
        from .temporal_query import TemporalGraphQuery
        
        temporal_query = TemporalGraphQuery(**self.config)
        
        # Analyze evolution
        evolution = temporal_query.analyze_evolution(
            graph,
            start_time=start_time,
            end_time=end_time,
            metrics=metrics,
            **options
        )
        
        return {
            "evolution": evolution,
            "time_range": {"start": start_time, "end": end_time},
            "metrics_tracked": metrics
        }

