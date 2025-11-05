"""
Temporal Query Module for Knowledge Graphs

Provides time-aware querying capabilities for temporal knowledge graphs.
"""


class TemporalGraphQuery:
    """
    Temporal knowledge graph query engine.
    
    • Executes time-aware queries on knowledge graphs
    • Supports queries at specific time points
    • Handles temporal range queries
    • Manages temporal window queries
    • Processes temporal pattern queries
    • Supports temporal evolution analysis
    
    Attributes:
        • supported_query_languages: Supported query languages
        • temporal_reasoner: Temporal reasoning engine
        • pattern_detector: Temporal pattern detector
        • evolution_analyzer: Graph evolution analyzer
        
    Methods:
        • query_at_time(): Query graph at specific time
        • query_time_range(): Query within time range
        • query_temporal_pattern(): Query temporal patterns
        • analyze_evolution(): Analyze graph evolution
        • find_temporal_paths(): Find paths in temporal context
    """
    
    def __init__(
        self,
        enable_temporal_reasoning=True,
        temporal_granularity="day",
        max_temporal_depth=None,
        **kwargs
    ):
        """
        Initialize temporal query engine.
        
        Args:
            enable_temporal_reasoning: Enable temporal reasoning
            temporal_granularity: Time granularity for queries
            max_temporal_depth: Maximum depth for temporal queries
            **kwargs: Additional configuration
        """
        self.enable_temporal_reasoning = enable_temporal_reasoning
        self.temporal_granularity = temporal_granularity
        self.max_temporal_depth = max_temporal_depth
        
        # Initialize temporal query engine
        from ..utils.logging import get_logger
        self.logger = get_logger("temporal_query")
        
        # Initialize pattern detector
        self.pattern_detector = TemporalPatternDetector(**kwargs.get("pattern_detection", {}))
    
    def query_at_time(
        self,
        graph,
        query,
        at_time,
        include_history=False,
        temporal_precision=None,
        **options
    ):
        """
        Query graph at specific time point.
        
        Args:
            graph: Knowledge graph to query
            query: Query string (Cypher, SPARQL, or natural language)
            at_time: Time point (datetime, timestamp, or ISO string)
            include_history: Include historical context
            temporal_precision: Precision for time matching
            **options: Additional query options
            
        Returns:
            Query results valid at specified time
        """
        self.logger.info(f"Querying graph at time: {at_time}")
        
        # Parse time
        query_time = self._parse_time(at_time)
        
        # Filter relationships valid at query time
        relationships = []
        if "relationships" in graph:
            for rel in graph.get("relationships", []):
                valid_from = self._parse_time(rel.get("valid_from"))
                valid_until = self._parse_time(rel.get("valid_until"))
                
                # Check if relationship is valid at query time
                if valid_from and self._compare_times(query_time, valid_from) < 0:
                    continue
                if valid_until and self._compare_times(query_time, valid_until) > 0:
                    continue
                
                relationships.append(rel)
        
        # Get entities
        entities = graph.get("entities", [])
        
        # Include history if requested
        if include_history:
            # Add all relationships with temporal information
            relationships = graph.get("relationships", [])
        
        return {
            "query": query,
            "at_time": query_time,
            "entities": entities,
            "relationships": relationships,
            "num_entities": len(entities),
            "num_relationships": len(relationships)
        }
    
    def query_time_range(
        self,
        graph,
        query,
        start_time,
        end_time,
        temporal_aggregation="union",
        include_intervals=True,
        **options
    ):
        """
        Query graph within time range.
        
        Args:
            graph: Knowledge graph to query
            query: Query string
            start_time: Start of time range
            end_time: End of time range
            temporal_aggregation: How to aggregate results ("union", "intersection", "evolution")
            include_intervals: Include partial matches within range
            **options: Additional query options
            
        Returns:
            Query results within time range
        """
        self.logger.info(f"Querying graph in time range: {start_time} to {end_time}")
        
        # Parse times
        start = self._parse_time(start_time)
        end = self._parse_time(end_time)
        
        # Filter relationships valid in time range
        relationships = []
        if "relationships" in graph:
            for rel in graph.get("relationships", []):
                valid_from = self._parse_time(rel.get("valid_from"))
                valid_until = self._parse_time(rel.get("valid_until"))
                
                # Check if relationship overlaps with time range
                if valid_from and self._compare_times(end, valid_from) < 0:
                    continue
                if valid_until and self._compare_times(start, valid_until) > 0:
                    continue
                
                relationships.append(rel)
        
        # Aggregate based on strategy
        if temporal_aggregation == "intersection":
            # Only relationships valid throughout the entire range
            relationships = [
                rel for rel in relationships
                if self._parse_time(rel.get("valid_from")) <= start and
                (not rel.get("valid_until") or self._parse_time(rel.get("valid_until")) >= end)
            ]
        elif temporal_aggregation == "evolution":
            # Group by time periods
            relationships = self._group_by_time_periods(relationships, start, end)
        
        return {
            "query": query,
            "start_time": start,
            "end_time": end,
            "relationships": relationships,
            "num_relationships": len(relationships),
            "aggregation": temporal_aggregation
        }
    
    def query_temporal_pattern(
        self,
        graph,
        pattern,
        time_window=None,
        min_support=1,
        **options
    ):
        """
        Query for temporal patterns in graph.
        
        Args:
            graph: Knowledge graph to query
            pattern: Pattern to search for
            time_window: Time window for pattern matching
            min_support: Minimum support for pattern
            **options: Additional pattern query options
            
        Returns:
            Matching temporal patterns
        """
        self.logger.info(f"Querying temporal patterns: {pattern}")
        
        # Use pattern detector
        patterns = self.pattern_detector.detect_temporal_patterns(
            graph,
            pattern_type=pattern,
            min_frequency=min_support,
            time_window=time_window,
            **options
        )
        
        return {
            "pattern": pattern,
            "patterns": patterns,
            "num_patterns": len(patterns) if isinstance(patterns, list) else 0
        }
    
    def analyze_evolution(
        self,
        graph,
        entity=None,
        relationship=None,
        start_time=None,
        end_time=None,
        metrics=["count", "diversity", "stability"],
        **options
    ):
        """
        Analyze graph evolution over time.
        
        Args:
            graph: Knowledge graph to analyze
            entity: Specific entity to track (None for entire graph)
            relationship: Specific relationship type to track (None for all)
            start_time: Start of analysis period
            end_time: End of analysis period
            metrics: Metrics to calculate
            **options: Additional analysis options
            
        Returns:
            Evolution analysis results
        """
        self.logger.info("Analyzing graph evolution")
        
        # Filter relationships
        relationships = graph.get("relationships", [])
        
        if entity:
            relationships = [
                rel for rel in relationships
                if rel.get("source") == entity or rel.get("target") == entity
            ]
        
        if relationship:
            relationships = [
                rel for rel in relationships
                if rel.get("type") == relationship
            ]
        
        # Filter by time range
        if start_time or end_time:
            start = self._parse_time(start_time) if start_time else None
            end = self._parse_time(end_time) if end_time else None
            
            filtered = []
            for rel in relationships:
                valid_from = self._parse_time(rel.get("valid_from"))
                valid_until = self._parse_time(rel.get("valid_until"))
                
                if start and valid_until and self._compare_times(valid_until, start) < 0:
                    continue
                if end and valid_from and self._compare_times(valid_from, end) > 0:
                    continue
                
                filtered.append(rel)
            relationships = filtered
        
        # Calculate metrics
        result = {
            "entity": entity,
            "relationship": relationship,
            "time_range": {"start": start_time, "end": end_time},
            "num_relationships": len(relationships)
        }
        
        if "count" in metrics:
            result["count"] = len(relationships)
        
        if "diversity" in metrics:
            rel_types = set(rel.get("type") for rel in relationships)
            result["diversity"] = len(rel_types)
        
        if "stability" in metrics:
            # Calculate stability based on relationship duration
            durations = []
            for rel in relationships:
                valid_from = self._parse_time(rel.get("valid_from"))
                valid_until = self._parse_time(rel.get("valid_until"))
                if valid_from and valid_until:
                    # Simplified duration calculation
                    durations.append(1)  # Placeholder
            result["stability"] = sum(durations) / len(durations) if durations else 0
        
        return result
    
    def find_temporal_paths(
        self,
        graph,
        source,
        target,
        start_time=None,
        end_time=None,
        max_path_length=None,
        temporal_constraints=None,
        **options
    ):
        """
        Find paths between entities considering temporal validity.
        
        Args:
            graph: Knowledge graph to search
            source: Source entity
            target: Target entity
            start_time: Start time for path validity
            end_time: End time for path validity
            max_path_length: Maximum path length
            temporal_constraints: Additional temporal constraints
            **options: Additional path finding options
            
        Returns:
            Temporal paths between entities
        """
        self.logger.info(f"Finding temporal paths from {source} to {target}")
        
        # Build adjacency with temporal constraints
        adjacency = {}
        relationships = graph.get("relationships", [])
        
        for rel in relationships:
            s = rel.get("source")
            t = rel.get("target")
            
            # Check temporal validity
            if start_time or end_time:
                valid_from = self._parse_time(rel.get("valid_from"))
                valid_until = self._parse_time(rel.get("valid_until"))
                
                if start_time and valid_until and self._compare_times(valid_until, start_time) < 0:
                    continue
                if end_time and valid_from and self._compare_times(valid_from, end_time) > 0:
                    continue
            
            if s not in adjacency:
                adjacency[s] = []
            adjacency[s].append((t, rel))
        
        # BFS to find paths
        from collections import deque
        
        paths = []
        queue = deque([(source, [source], [])])
        visited = set()
        max_length = max_path_length or float('inf')
        
        while queue:
            node, path, edges = queue.popleft()
            
            if len(path) > max_length:
                continue
            
            if node == target:
                paths.append({"path": path, "edges": edges, "length": len(path) - 1})
                continue
            
            if node in visited:
                continue
            visited.add(node)
            
            for neighbor, rel in adjacency.get(node, []):
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor], edges + [rel]))
        
        return {
            "source": source,
            "target": target,
            "paths": paths,
            "num_paths": len(paths)
        }
    
    def _parse_time(self, time_value):
        """Parse time value."""
        from datetime import datetime
        
        if time_value is None:
            return None
        
        if isinstance(time_value, str):
            return time_value
        
        if isinstance(time_value, datetime):
            return time_value.isoformat()
        
        return str(time_value)
    
    def _compare_times(self, time1, time2):
        """Compare two time strings."""
        if time1 is None or time2 is None:
            return 0
        return (time1 > time2) - (time1 < time2)
    
    def _group_by_time_periods(self, relationships, start, end):
        """Group relationships by time periods."""
        # Simplified grouping
        return relationships


class TemporalPatternDetector:
    """
    Temporal pattern detection in knowledge graphs.
    
    • Detects recurring temporal patterns
    • Identifies temporal sequences
    • Finds temporal anomalies
    • Discovers temporal trends
    • Analyzes temporal correlations
    """
    
    def __init__(self, **config):
        """Initialize temporal pattern detector."""
        from ..utils.logging import get_logger
        self.logger = get_logger("temporal_pattern_detector")
        self.config = config
    
    def detect_temporal_patterns(
        self,
        graph,
        pattern_type="sequence",
        min_frequency=2,
        time_window=None,
        **options
    ):
        """
        Detect temporal patterns in graph.
        
        Args:
            graph: Knowledge graph to analyze
            pattern_type: Type of pattern ("sequence", "cycle", "trend", "anomaly")
            min_frequency: Minimum frequency for pattern
            time_window: Time window for pattern detection
            **options: Additional detection options
            
        Returns:
            Detected temporal patterns
        """
        self.logger.info(f"Detecting temporal patterns: {pattern_type}")
        
        relationships = graph.get("relationships", [])
        
        # Simple pattern detection
        patterns = []
        
        if pattern_type == "sequence":
            # Find sequential relationships
            sequences = self._find_sequences(relationships, min_frequency)
            patterns.extend(sequences)
        elif pattern_type == "cycle":
            # Find cyclic patterns
            cycles = self._find_cycles(relationships, min_frequency)
            patterns.extend(cycles)
        
        return patterns
    
    def _find_sequences(self, relationships, min_frequency):
        """Find sequential patterns."""
        # Simplified sequence detection
        return []
    
    def _find_cycles(self, relationships, min_frequency):
        """Find cyclic patterns."""
        # Simplified cycle detection
        return []


class TemporalVersionManager:
    """
    Temporal version management for knowledge graphs.
    
    • Creates temporal versions/snapshots
    • Manages version history
    • Handles version comparison
    • Supports version rollback
    • Tracks version metadata
    """
    
    def __init__(
        self,
        snapshot_interval=None,
        auto_snapshot=False,
        version_strategy="timestamp",
        **config
    ):
        """
        Initialize temporal version manager.
        
        Args:
            snapshot_interval: Interval for automatic snapshots
            auto_snapshot: Enable automatic snapshots
            version_strategy: Versioning strategy ("timestamp", "incremental", "semantic")
            **config: Additional configuration
        """
        self.snapshot_interval = snapshot_interval
        self.auto_snapshot = auto_snapshot
        self.version_strategy = version_strategy
    
    def create_version(
        self,
        graph,
        version_label=None,
        timestamp=None,
        metadata=None,
        **options
    ):
        """
        Create version snapshot of graph.
        
        Args:
            graph: Knowledge graph to version
            version_label: Optional version label
            timestamp: Timestamp for version (None for current)
            metadata: Additional version metadata
            **options: Additional version options
            
        Returns:
            Version snapshot object
        """
        from datetime import datetime
        
        version_time = timestamp or datetime.now().isoformat()
        
        version = {
            "label": version_label or f"version_{version_time}",
            "timestamp": version_time,
            "entities": graph.get("entities", []).copy(),
            "relationships": graph.get("relationships", []).copy(),
            "metadata": metadata or {}
        }
        
        return version
    
    def compare_versions(self, version1, version2, comparison_metrics=None, **options):
        """
        Compare two graph versions.
        
        Args:
            version1: First version
            version2: Second version
            comparison_metrics: Metrics for comparison
            **options: Additional comparison options
            
        Returns:
            Version comparison results
        """
        comparison = {
            "version1": version1.get("label", "unknown"),
            "version2": version2.get("label", "unknown"),
            "entities_added": len(version2.get("entities", [])) - len(version1.get("entities", [])),
            "relationships_added": len(version2.get("relationships", [])) - len(version1.get("relationships", []))
        }
        
        return comparison

