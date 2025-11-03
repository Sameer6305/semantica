"""
Main Orchestrator Module

The Semantica orchestrator coordinates all framework components and manages
the overall execution flow.

Key Responsibilities:
    - Initialize and coordinate all modules
    - Manage pipeline execution
    - Handle resource allocation
    - Coordinate plugin loading
    - Manage system lifecycle

Main Classes:
    - Semantica: Main framework class
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .config_manager import Config, ConfigManager
from .lifecycle import LifecycleManager, SystemState
from .plugin_registry import PluginRegistry
from ..utils.exceptions import ConfigurationError, ProcessingError
from ..utils.logging import get_logger, log_execution_time


class Semantica:
    """
    Main Semantica framework class.
    
    This is the primary entry point for using the framework. It coordinates
    all modules and provides a unified API for semantic processing.
    
    Attributes:
        config: Configuration object
        config_manager: Configuration management system
        plugin_registry: Plugin management system
        lifecycle_manager: Lifecycle management
        
    Methods:
        initialize(): Initialize all framework components
        build_knowledge_base(): Build knowledge base from sources
        run_pipeline(): Execute processing pipeline
        get_status(): Get system health and status
    """
    
    def __init__(self, config: Optional[Union[Config, Dict[str, Any]]] = None, **kwargs):
        """
        Initialize Semantica framework.
        
        Args:
            config: Configuration object or dict
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("semantica")
        self.config_manager = ConfigManager()
        
        # Load configuration
        if isinstance(config, Config):
            self.config = config
        elif isinstance(config, dict):
            self.config = self.config_manager.load_from_dict(config)
        else:
            # Load from kwargs or defaults
            self.config = self.config_manager.load_from_dict(kwargs or {})
        
        # Initialize core components
        self.lifecycle_manager = LifecycleManager()
        self.plugin_registry = PluginRegistry()
        
        # Register lifecycle manager with itself
        self.lifecycle_manager.register_component("lifecycle_manager", self.lifecycle_manager)
        self.lifecycle_manager.register_component("plugin_registry", self.plugin_registry)
        self.lifecycle_manager.register_component("config_manager", self.config_manager)
        
        # Module placeholders (to be initialized)
        self._modules: Dict[str, Any] = {}
        
        self.logger.info("Semantica framework initialized")
    
    @log_execution_time
    def initialize(self) -> None:
        """
        Initialize all framework components.
        
        This method sets up all modules, loads plugins, and prepares
        the system for processing.
        
        Raises:
            ConfigurationError: If configuration is invalid
            SemanticaError: If initialization fails
        """
        try:
            self.logger.info("Initializing Semantica framework")
            
            # Validate configuration
            self.config.validate()
            
            # Register startup hooks
            self._register_startup_hooks()
            
            # Execute startup sequence
            self.lifecycle_manager.startup()
            
            # Initialize modules
            self._initialize_modules()
            
            # Load plugins if configured
            if self.config.get("plugins", {}):
                self._load_plugins()
            
            # Run health checks
            health_summary = self.lifecycle_manager.get_health_summary()
            if not health_summary["is_healthy"]:
                self.logger.warning(
                    "Some components are unhealthy",
                    extra={"health_summary": health_summary}
                )
            
            self.logger.info("Semantica framework initialization completed")
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            raise
    
    @log_execution_time
    def build_knowledge_base(self, sources: List[Union[str, Path]], **kwargs) -> Dict[str, Any]:
        """
        Build knowledge base from data sources.
        
        This is a high-level method that orchestrates the complete
        knowledge base construction process.
        
        Args:
            sources: List of data sources (files, URLs, streams)
            **kwargs: Additional processing options:
                - pipeline: Custom pipeline configuration
                - embeddings: Whether to generate embeddings
                - graph: Whether to build knowledge graph
                - normalize: Whether to normalize data
                
        Returns:
            Dictionary containing:
                - knowledge_graph: Knowledge graph data
                - embeddings: Embedding vectors
                - metadata: Processing metadata
                - statistics: Processing statistics
                
        Raises:
            ProcessingError: If processing fails
        """
        try:
            self.logger.info(f"Building knowledge base from {len(sources)} sources")
            
            # Validate sources
            validated_sources = self._validate_sources(sources)
            
            # Create processing pipeline
            pipeline_config = kwargs.get("pipeline", {})
            pipeline = self._create_pipeline(pipeline_config)
            
            # Process sources
            results = []
            for source in validated_sources:
                try:
                    result = self.run_pipeline(pipeline, source)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to process source {source}: {e}")
                    if kwargs.get("fail_fast", False):
                        raise ProcessingError(f"Failed to process source {source}: {e}")
            
            # Build knowledge graph if requested
            knowledge_graph = None
            if kwargs.get("graph", True):
                knowledge_graph = self._build_knowledge_graph(results)
            
            # Generate embeddings if requested
            embeddings = None
            if kwargs.get("embeddings", True):
                embeddings = self._generate_embeddings(results)
            
            # Compile statistics
            statistics = {
                "sources_processed": len(results),
                "sources_total": len(sources),
                "success_rate": len([r for r in results if r.get("success")]) / len(results) if results else 0.0,
            }
            
            return {
                "knowledge_graph": knowledge_graph,
                "embeddings": embeddings,
                "results": results,
                "statistics": statistics,
                "metadata": {
                    "sources": validated_sources,
                    "pipeline": pipeline_config,
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to build knowledge base: {e}")
            raise ProcessingError(f"Failed to build knowledge base: {e}")
    
    @log_execution_time
    def run_pipeline(self, pipeline: Union[Dict[str, Any], Any], data: Any) -> Dict[str, Any]:
        """
        Execute a processing pipeline.
        
        Args:
            pipeline: Pipeline object or configuration dictionary
            data: Input data for pipeline
            
        Returns:
            Dictionary containing:
                - output: Pipeline output data
                - metadata: Processing metadata
                - metrics: Performance metrics
                
        Raises:
            ProcessingError: If pipeline execution fails
        """
        try:
            self.logger.info("Executing processing pipeline")
            
            # Validate pipeline
            if isinstance(pipeline, dict):
                pipeline = self._create_pipeline_from_dict(pipeline)
            
            # Validate pipeline object
            if not hasattr(pipeline, "execute"):
                raise ProcessingError("Pipeline must have execute() method")
            
            # Allocate resources
            resources = self._allocate_resources(pipeline)
            
            try:
                # Execute pipeline
                result = pipeline.execute(data)
                
                # Collect metrics
                metrics = self._collect_metrics(pipeline)
                
                return {
                    "success": True,
                    "output": result,
                    "metrics": metrics,
                    "metadata": {
                        "pipeline": str(pipeline),
                        "resources": resources,
                    }
                }
                
            finally:
                # Release resources
                self._release_resources(resources)
                
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise ProcessingError(f"Pipeline execution failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get system health and status.
        
        Returns:
            Dictionary containing:
                - state: System state
                - health: Health summary
                - modules: Module status
                - plugins: Plugin status
                - metrics: System metrics
        """
        health_summary = self.lifecycle_manager.get_health_summary()
        
        # Get module status
        module_status = {}
        for name, module in self._modules.items():
            module_status[name] = {
                "initialized": module is not None,
                "status": "ready" if module else "not_initialized"
            }
        
        # Get plugin status
        plugin_status = {}
        try:
            plugins = self.plugin_registry.list_plugins()
            for plugin_info in plugins:
                plugin_status[plugin_info.get("name", "unknown")] = {
                    "loaded": plugin_info.get("loaded", False),
                    "version": plugin_info.get("version", "unknown")
                }
        except Exception as e:
            self.logger.warning(f"Failed to get plugin status: {e}")
        
        return {
            "state": self.lifecycle_manager.get_state().value,
            "health": health_summary,
            "modules": module_status,
            "plugins": plugin_status,
            "config": {
                "loaded": self.config is not None,
                "validated": True  # Assuming validated if initialized
            }
        }
    
    def shutdown(self, graceful: bool = True) -> None:
        """
        Shutdown the framework.
        
        Args:
            graceful: Whether to shutdown gracefully (default: True)
        """
        self.logger.info("Shutting down Semantica framework")
        self.lifecycle_manager.shutdown(graceful=graceful)
        self.logger.info("Semantica framework shutdown completed")
    
    def _register_startup_hooks(self) -> None:
        """Register framework startup hooks."""
        # Register config validation hook
        def validate_config_hook():
            self.config.validate()
        
        self.lifecycle_manager.register_startup_hook(validate_config_hook, priority=10)
        
        # Register module initialization hook
        def initialize_modules_hook():
            self._initialize_modules()
        
        self.lifecycle_manager.register_startup_hook(initialize_modules_hook, priority=30)
    
    def _initialize_modules(self) -> None:
        """Initialize framework modules."""
        # Module initialization will be implemented when modules are available
        # For now, this is a placeholder
        pass
    
    def _load_plugins(self) -> None:
        """Load configured plugins."""
        plugins_config = self.config.get("plugins", {})
        
        for plugin_name, plugin_config in plugins_config.items():
            try:
                self.plugin_registry.load_plugin(plugin_name, **plugin_config)
                self.logger.info(f"Loaded plugin: {plugin_name}")
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
    
    def _validate_sources(self, sources: List[Union[str, Path]]) -> List[Union[str, Path]]:
        """Validate data sources."""
        validated = []
        
        for source in sources:
            source_path = Path(source) if isinstance(source, str) else source
            
            # Check if source exists (if it's a file path)
            if source_path.exists() or str(source).startswith(("http://", "https://")):
                validated.append(source)
            else:
                self.logger.warning(f"Source not found or invalid: {source}")
        
        if not validated:
            raise ProcessingError("No valid sources provided")
        
        return validated
    
    def _create_pipeline(self, pipeline_config: Dict[str, Any]) -> Any:
        """Create processing pipeline from configuration."""
        # Pipeline creation will be implemented in pipeline module
        # For now, return config as placeholder
        return pipeline_config
    
    def _create_pipeline_from_dict(self, pipeline_dict: Dict[str, Any]) -> Any:
        """Create pipeline object from dictionary."""
        # This will be implemented when pipeline module is available
        return pipeline_dict
    
    def _build_knowledge_graph(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build knowledge graph from processing results."""
        # Knowledge graph building will be implemented in kg module
        return {"status": "placeholder", "results": results}
    
    def _generate_embeddings(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate embeddings from processing results."""
        # Embedding generation will be implemented in embeddings module
        return {"status": "placeholder", "results": results}
    
    def _allocate_resources(self, pipeline: Any) -> Dict[str, Any]:
        """Allocate resources for pipeline execution."""
        # Resource allocation logic
        return {"allocated": True}
    
    def _release_resources(self, resources: Dict[str, Any]) -> None:
        """Release allocated resources."""
        # Resource release logic
        pass
    
    def _collect_metrics(self, pipeline: Any) -> Dict[str, Any]:
        """Collect performance metrics from pipeline execution."""
        # Metrics collection logic
        return {"execution_time": 0.0, "memory_usage": 0}