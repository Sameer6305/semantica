# Semantica Framework - Complete Modules Documentation

This document provides detailed documentation of all modules, submodules, classes, methods, and parameters in the Semantica framework.

---

## Table of Contents

1. [Core Modules](#core-modules)
2. [Ingestion Modules](#ingestion-modules)
3. [Parsing Modules](#parsing-modules)
4. [Normalization Modules](#normalization-modules)
5. [Semantic Extraction Modules](#semantic-extraction-modules)
6. [Knowledge Graph Modules](#knowledge-graph-modules)
7. [Embeddings Modules](#embeddings-modules)
8. [Pipeline Modules](#pipeline-modules)
9. [Reasoning Modules](#reasoning-modules)
10. [Vector Store Modules](#vector-store-modules)
11. [Triple Store Modules](#triple-store-modules)
12. [Export Modules](#export-modules)
13. [Visualization Modules](#visualization-modules)
14. [Quality Assurance Modules](#quality-assurance-modules)
15. [Context Modules](#context-modules)
16. [Deduplication Modules](#deduplication-modules)
17. [Conflict Modules](#conflict-modules)
18. [Split Modules](#split-modules)
19. [Ontology Modules](#ontology-modules)
20. [Seed Modules](#seed-modules)
21. [Utils Modules](#utils-modules)

---

## Core Modules

### `semantica.core.config_manager`

**What it does:**
This module provides comprehensive configuration management for the Semantica framework. It handles loading configuration from files (YAML/JSON), environment variables, validation, and dynamic updates. The module supports nested configuration access via dot notation, configuration inheritance and merging, and automatic type conversion from environment variables.

**Key Features:**
- Load configuration from YAML/JSON files
- Support for environment variables with `SEMANTICA_` prefix
- Configuration validation with detailed error messages
- Dynamic configuration updates at runtime
- Configuration inheritance and merging
- Nested configuration access via dot notation

#### Class: `Config`

Configuration data class that stores all framework configuration settings.

**Methods:**

##### `__init__(config_dict: Optional[Dict[str, Any]] = None, **kwargs)`
Initialize configuration.

**Parameters:**
- `config_dict` (Optional[Dict[str, Any]]): Dictionary of configuration values
- `**kwargs`: Additional configuration parameters merged into config

**Attributes:**
- `llm_provider`: LLM provider configuration
- `embedding_model`: Embedding model settings
- `vector_store`: Vector store configuration
- `graph_db`: Graph database settings
- `processing`: Processing pipeline settings
- `logging`: Logging configuration
- `quality`: Quality assurance settings
- `security`: Security settings
- `custom`: Custom configuration

##### `validate() -> None`
Validate configuration settings. Checks types, ranges, and required fields.

**Raises:**
- `ConfigurationError`: If configuration is invalid with detailed error messages

##### `to_dict() -> Dict[str, Any]`
Convert configuration to dictionary.

**Returns:**
- `dict`: Configuration as dictionary

##### `get(key_path: str, default: Any = None) -> Any`
Get nested configuration value by key path.

**Parameters:**
- `key_path` (str): Dot-separated key path (e.g., "processing.batch_size")
- `default` (Any): Default value if key not found

**Returns:**
- Configuration value or default

##### `set(key_path: str, value: Any) -> None`
Set nested configuration value by key path.

**Parameters:**
- `key_path` (str): Dot-separated key path
- `value` (Any): Value to set

##### `update(updates: Dict[str, Any], merge: bool = True) -> None`
Update configuration with new values.

**Parameters:**
- `updates` (Dict[str, Any]): Dictionary of updates
- `merge` (bool): Whether to merge nested dictionaries (default: True)

#### Class: `ConfigManager`

Configuration management system for loading, validating, and managing configuration.

**Methods:**

##### `__init__()`
Initialize configuration manager.

##### `load_from_file(file_path: Union[str, Path], validate: bool = True) -> Config`
Load configuration from file.

**Parameters:**
- `file_path` (Union[str, Path]): Path to configuration file (YAML or JSON)
- `validate` (bool): Whether to validate configuration after loading (default: True)

**Returns:**
- `Config`: Loaded configuration object

**Raises:**
- `ConfigurationError`: If file cannot be loaded or is invalid

##### `load_from_dict(config_dict: Dict[str, Any], validate: bool = True) -> Config`
Load configuration from dictionary.

**Parameters:**
- `config_dict` (Dict[str, Any]): Dictionary of configuration values
- `validate` (bool): Whether to validate configuration after loading (default: True)

**Returns:**
- `Config`: Configuration object

##### `merge_configs(*configs: Config, validate: bool = True) -> Config`
Merge multiple configurations. Later configurations take priority.

**Parameters:**
- `*configs` (Config): Configuration objects to merge
- `validate` (bool): Whether to validate merged configuration (default: True)

**Returns:**
- `Config`: Merged configuration

##### `get_config() -> Optional[Config]`
Get current configuration.

**Returns:**
- Current Config object or None if not loaded

##### `set_config(config: Config, validate: bool = True) -> None`
Set current configuration.

**Parameters:**
- `config` (Config): Configuration object to set
- `validate` (bool): Whether to validate configuration (default: True)

##### `reload(file_path: Optional[Union[str, Path]] = None) -> Config`
Reload configuration from file.

**Parameters:**
- `file_path` (Optional[Union[str, Path]]): Path to configuration file. If None, uses last loaded file.

**Returns:**
- `Config`: Reloaded configuration

**Different Approaches and Strategies:**

The `ConfigManager` supports multiple approaches for configuration management:

1. **File-based Configuration** - Load from YAML/JSON files (recommended for production)
2. **Dictionary-based Configuration** - Load from Python dictionaries (useful for programmatic setup)
3. **Environment Variable Configuration** - Use environment variables with `SEMANTICA_` prefix
4. **Merged Configuration** - Combine multiple configuration sources with priority
5. **Dynamic Configuration** - Update configuration at runtime without reloading

**When to Use Each Approach:**

- **File-based**: Production environments, version-controlled configurations, team collaboration
- **Dictionary-based**: Testing, programmatic configuration, dynamic setups
- **Environment Variables**: Docker containers, CI/CD pipelines, sensitive data
- **Merged Configuration**: Override defaults, environment-specific settings, feature flags
- **Dynamic Configuration**: Runtime adjustments, A/B testing, hot-reloading

**Code Examples for Different Approaches:**

**Approach 1: File-based Configuration (Production)**
```python
from semantica.core import ConfigManager

# Load from YAML file (recommended for production)
manager = ConfigManager()
config = manager.load_from_file("config.yaml")

# Access nested values
batch_size = config.get("processing.batch_size", default=32)
model_name = config.get("embedding_model.name", default="default-model")

# Example config.yaml structure:
# processing:
#   batch_size: 32
#   max_workers: 4
# embedding_model:
#   name: "sentence-transformers/all-MiniLM-L6-v2"
#   device: "cpu"
```

**Approach 2: Dictionary-based Configuration (Programmatic)**
```python
from semantica.core import ConfigManager

# Load from dictionary (useful for testing or dynamic setup)
manager = ConfigManager()
config_dict = {
    "processing": {
        "batch_size": 32,
        "max_workers": 4
    },
    "quality": {
        "min_confidence": 0.8
    }
}
config = manager.load_from_dict(config_dict)

# Update programmatically
config.set("processing.batch_size", 64)
config.update({"processing": {"max_workers": 8}})
```

**Approach 3: Environment Variable Configuration (Docker/CI/CD)**
```python
import os
from semantica.core import ConfigManager, Config

# Set environment variables (typically done in shell/Docker)
os.environ["SEMANTICA_PROCESSING_BATCH_SIZE"] = "64"
os.environ["SEMANTICA_EMBEDDING_MODEL_NAME"] = "custom-model"

# Load configuration (environment variables automatically loaded)
manager = ConfigManager()
config = manager.load_from_dict({})  # Base config, env vars override

# Environment variables with SEMANTICA_ prefix are automatically loaded
# Format: SEMANTICA_<SECTION>_<KEY> (uppercase, underscores)
```

**Approach 4: Merged Configuration (Override Defaults)**
```python
from semantica.core import ConfigManager, Config

# Create base configuration
base_config = Config(config_dict={
    "processing": {"batch_size": 32, "max_workers": 4},
    "embedding_model": {"name": "default-model"}
})

# Create override configuration
override_config = Config(config_dict={
    "processing": {"batch_size": 64},  # Override batch_size
    "embedding_model": {"device": "cuda"}  # Add new setting
})

# Merge configurations (later configs take priority)
manager = ConfigManager()
merged = manager.merge_configs(base_config, override_config)

# Result: batch_size=64, max_workers=4, name="default-model", device="cuda"
print(merged.get("processing.batch_size"))  # 64 (from override)
print(merged.get("processing.max_workers"))  # 4 (from base)
```

**Approach 5: Dynamic Runtime Configuration**
```python
from semantica.core import ConfigManager

# Load initial configuration
manager = ConfigManager()
config = manager.load_from_file("config.yaml")

# Dynamically update configuration at runtime
config.set("processing.batch_size", 128)  # Increase batch size
config.update({
    "embedding_model": {
        "device": "cuda",  # Switch to GPU
        "normalize": True
    }
})

# Changes take effect immediately without reloading
# Useful for A/B testing or performance tuning
```

**Approach 6: Configuration Validation and Error Handling**
```python
from semantica.core import ConfigManager, Config
from semantica.utils.exceptions import ConfigurationError

manager = ConfigManager()

try:
    # Load and validate configuration
    config = manager.load_from_file("config.yaml", validate=True)
    
    # Manual validation
    config.validate()  # Raises ConfigurationError if invalid
    
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle invalid configuration
    # Option 1: Use defaults
    config = manager.load_from_dict({"processing": {"batch_size": 32}})
    
    # Option 2: Load fallback configuration
    config = manager.load_from_file("config.default.yaml", validate=False)
```

**Best Practices:**

1. **Use file-based configuration for production** - Easier to version control and manage
2. **Use environment variables for secrets** - API keys, passwords, tokens
3. **Merge configurations for flexibility** - Base config + environment-specific overrides
4. **Validate early** - Always validate configuration on load
5. **Use dot notation** - Access nested values with `config.get("section.key")`
6. **Document defaults** - Provide sensible defaults for all configuration options

---

### `semantica.core.lifecycle`

**What it does:**
This module manages the complete lifecycle of the Semantica framework, including startup and shutdown sequences, component health monitoring, and resource management. It provides a hook-based system for executing code at specific lifecycle stages with priority ordering, allowing components to initialize and cleanup in the correct order.

**Key Features:**
- Priority-based startup/shutdown hooks
- Component registration and health monitoring
- State management and tracking (uninitialized, ready, running, stopped, error)
- Graceful error handling during lifecycle transitions
- Health check system for all registered components

#### Class: `LifecycleManager`

System lifecycle manager that coordinates startup, shutdown, and health monitoring.

**Methods:**

##### `__init__()`
Initialize lifecycle manager. Creates manager in UNINITIALIZED state.

##### `startup() -> None`
Execute startup sequence. Runs all registered startup hooks in priority order.

**Raises:**
- `SemanticaError`: If startup fails

##### `shutdown(graceful: bool = True) -> None`
Execute shutdown sequence.

**Parameters:**
- `graceful` (bool): Whether to shutdown gracefully (default: True)
  - True: Continue shutdown even if hooks fail
  - False: Stop shutdown on first hook failure

**Raises:**
- `SemanticaError`: If shutdown fails and graceful=False

##### `health_check() -> Dict[str, HealthStatus]`
Perform comprehensive system health check.

**Returns:**
- Dictionary mapping component names to HealthStatus objects

##### `register_component(name: str, component: Any) -> None`
Register a component for health monitoring.

**Parameters:**
- `name` (str): Component name
- `component` (Any): Component instance

##### `unregister_component(name: str) -> None`
Unregister a component.

**Parameters:**
- `name` (str): Component name

##### `register_startup_hook(hook_fn: Callable[[], None], priority: int = 50) -> None`
Register a startup hook.

**Parameters:**
- `hook_fn` (Callable[[], None]): Function to call during startup (no arguments)
- `priority` (int): Hook priority (lower = earlier execution, default: 50)

##### `register_shutdown_hook(hook_fn: Callable[[], None], priority: int = 50) -> None`
Register a shutdown hook.

**Parameters:**
- `hook_fn` (Callable[[], None]): Function to call during shutdown
- `priority` (int): Hook priority (lower = earlier execution, default: 50)

##### `get_state() -> SystemState`
Get current system state.

**Returns:**
- Current system state

##### `is_ready() -> bool`
Check if system is ready.

**Returns:**
- True if system is ready, False otherwise

##### `is_running() -> bool`
Check if system is running.

**Returns:**
- True if system is running, False otherwise

##### `get_health_summary() -> Dict[str, Any]`
Get summary of system health.

**Returns:**
- Dictionary with health summary information

**Different Approaches and Strategies:**

The lifecycle manager provides 6 different approaches for managing system lifecycle:

1. **Priority-Based Hook System** - Execute hooks in priority order (lower priority = earlier execution)
2. **Graceful vs Non-Graceful Shutdown** - Continue or stop on errors during shutdown
3. **Component Health Monitoring** - Automatic health checking with different health check methods
4. **State-Based Management** - Track system state transitions (uninitialized → initializing → ready → running → stopping → stopped)
5. **Error Handling Strategies** - Different error handling during lifecycle transitions
6. **Resource Cleanup Approaches** - Automatic cleanup using cleanup() or close() methods

**When to Use Each Approach:**

| Approach | Use Case | Example |
|----------|----------|---------|
| Priority-Based Hooks | Need ordered initialization | Config (priority=10) → Database (priority=20) → Cache (priority=30) |
| Graceful Shutdown | Production systems | Continue cleanup even if some components fail |
| Non-Graceful Shutdown | Development/Debugging | Stop immediately on first error for easier debugging |
| Component Health Monitoring | Production monitoring | Track health of database, cache, API connections |
| State-Based Management | Complex systems | Track system state for UI dashboards, monitoring |
| Resource Cleanup | Resource management | Automatically close connections, files, threads |

**Detailed Examples for Each Approach:**

**Approach 1: Priority-Based Hook System**
```python
from semantica.core import LifecycleManager

manager = LifecycleManager()

# Lower priority = earlier execution
manager.register_startup_hook(init_logging, priority=1)      # First
manager.register_startup_hook(init_config, priority=10)      # Second
manager.register_startup_hook(init_database, priority=20)  # Third
manager.register_startup_hook(init_cache, priority=30)      # Fourth

manager.startup()  # Executes in order: logging → config → database → cache
```

**Approach 2: Graceful vs Non-Graceful Shutdown**
```python
# Graceful shutdown (production) - continues even if hooks fail
manager.shutdown(graceful=True)  # Logs warnings but continues

# Non-graceful shutdown (debugging) - stops on first error
manager.shutdown(graceful=False)  # Raises error on first failure
```

**Approach 3: Component Health Monitoring**
```python
# Register components with automatic health checking
manager.register_component("database", db_connection)
manager.register_component("cache", cache_client)

# Components can implement health_check() method
class DatabaseConnection:
    def health_check(self):
        return {"healthy": self.is_connected(), "message": "Connected"}

# Or use simple boolean
class CacheClient:
    def health_check(self):
        return self.is_alive()  # Returns True/False

# Automatic health checking
health = manager.health_check()
for name, status in health.items():
    print(f"{name}: {status.healthy} - {status.message}")
```

**Approach 4: State-Based Management**
```python
# Track system state transitions
state = manager.get_state()  # Returns SystemState enum
print(f"Current state: {state}")  # uninitialized, initializing, ready, running, etc.

# Check if system is ready
if manager.is_ready():
    print("System ready for processing")

# Check if system is running
if manager.is_running():
    print("System is actively running")
```

**Approach 5: Error Handling Strategies**
```python
# Startup hooks with error handling
def init_database():
    try:
        db.connect()
    except Exception as e:
        # Error stops startup (raises SemanticaError)
        raise

# Shutdown hooks with graceful error handling
def cleanup_database():
    try:
        db.close()
    except Exception as e:
        # In graceful mode, error is logged but doesn't stop shutdown
        logger.warning(f"Cleanup failed: {e}")
```

**Approach 6: Resource Cleanup Approaches**
```python
# Components with cleanup() method
class Resource:
    def cleanup(self):
        self.close_connections()
        self.release_resources()

# Components with close() method
class Connection:
    def close(self):
        self.connection.close()

# Automatic cleanup on shutdown
manager.register_component("resource", Resource())
manager.register_component("connection", Connection())
manager.shutdown()  # Automatically calls cleanup() or close()
```

**Code Example:**
```python
from semantica.core import LifecycleManager

# Initialize lifecycle manager
manager = LifecycleManager()

# Register components for health monitoring
manager.register_component("database", db_connection)
manager.register_component("cache", cache_client)

# Register startup hooks with priorities (lower = earlier execution)
def init_config():
    print("Initializing configuration...")

def init_database():
    print("Connecting to database...")

manager.register_startup_hook(init_config, priority=10)  # Runs first
manager.register_startup_hook(init_database, priority=20)  # Runs second

# Register shutdown hooks
def cleanup_database():
    print("Closing database connections...")

manager.register_shutdown_hook(cleanup_database, priority=10)

# Execute startup sequence
manager.startup()  # Hooks execute in priority order

# Check system health
health = manager.health_check()
for component, status in health.items():
    print(f"{component}: {'Healthy' if status.healthy else 'Unhealthy'}")

# Get health summary
summary = manager.get_health_summary()
print(f"System state: {summary['state']}")
print(f"Healthy components: {summary['healthy_components']}/{summary['total_components']}")

# Check if system is ready
if manager.is_ready():
    print("System is ready for processing")

# Shutdown gracefully
manager.shutdown(graceful=True)  # Continues even if hooks fail
```

---

### `semantica.core.orchestrator`

**What it does:**
This is the main orchestrator module that coordinates all framework components and manages the overall execution flow. It provides the primary entry point for the Semantica framework, handling framework initialization, knowledge base construction from various data sources, pipeline execution, resource management, plugin system coordination, and system health monitoring.

**Key Features:**
- Framework initialization and lifecycle management
- Knowledge base construction from various data sources
- Pipeline execution and resource management
- Plugin system coordination
- System health monitoring
- Automatic component initialization

#### Class: `Semantica`

Main Semantica framework class - primary entry point.

**Methods:**

##### `__init__(config: Optional[Union[Config, Dict[str, Any]]] = None, **kwargs)`
Initialize Semantica framework.

**Parameters:**
- `config` (Optional[Union[Config, Dict[str, Any]]): Configuration object or dict
- `**kwargs`: Additional configuration parameters

##### `initialize() -> None`
Initialize all framework components.

**Raises:**
- `ConfigurationError`: If configuration is invalid
- `SemanticaError`: If initialization fails

##### `build_knowledge_base(sources: List[Union[str, Path]], **kwargs) -> Dict[str, Any]`
Build knowledge base from data sources.

**Parameters:**
- `sources` (List[Union[str, Path]]): List of data sources (files, URLs, streams)
- `**kwargs`: Additional processing options:
  - `pipeline`: Custom pipeline configuration
  - `embeddings`: Whether to generate embeddings (default: True)
  - `graph`: Whether to build knowledge graph (default: True)
  - `normalize`: Whether to normalize data (default: True)
  - `fail_fast`: Whether to fail on first error (default: False)

**Returns:**
- Dictionary containing:
  - `knowledge_graph`: Knowledge graph data
  - `embeddings`: Embedding vectors
  - `metadata`: Processing metadata
  - `statistics`: Processing statistics
  - `results`: Processing results

**Raises:**
- `ProcessingError`: If processing fails

##### `run_pipeline(pipeline: Union[Dict[str, Any], Any], data: Any) -> Dict[str, Any]`
Execute a processing pipeline.

**Parameters:**
- `pipeline` (Union[Dict[str, Any], Any]): Pipeline object or configuration dictionary
- `data` (Any): Input data for pipeline

**Returns:**
- Dictionary containing:
  - `output`: Pipeline output data
  - `metadata`: Processing metadata
  - `metrics`: Performance metrics

**Raises:**
- `ProcessingError`: If pipeline execution fails

##### `get_status() -> Dict[str, Any]`
Get system health and status.

**Returns:**
- Dictionary containing:
  - `state`: System state
  - `health`: Health summary
  - `modules`: Module status
  - `plugins`: Plugin status
  - `metrics`: System metrics

##### `shutdown(graceful: bool = True) -> None`
Shutdown the framework.

**Parameters:**
- `graceful` (bool): Whether to shutdown gracefully (default: True)

**Code Example:**
```python
from semantica import Semantica

# Initialize framework with configuration
framework = Semantica(config={
    "processing": {"batch_size": 32},
    "embedding_model": {"provider": "openai"}
})

# Initialize all components (auto-initializes if not done)
framework.initialize()

# Build knowledge base from multiple sources
result = framework.build_knowledge_base(
    sources=["doc1.pdf", "doc2.docx", "https://example.com/article"],
    embeddings=True,      # Generate embeddings
    graph=True,           # Build knowledge graph
    normalize=True,       # Normalize data
    fail_fast=False      # Continue on errors
)

# Access results
knowledge_graph = result["knowledge_graph"]
embeddings = result["embeddings"]
statistics = result["statistics"]

print(f"Processed {statistics['sources_processed']} sources")
print(f"Success rate: {statistics['success_rate']:.2%}")

# Get system status
status = framework.get_status()
print(f"System state: {status['state']}")
print(f"Healthy components: {status['health']['healthy_components']}")

# Shutdown gracefully
framework.shutdown(graceful=True)
```

---

### `semantica.core.plugin_registry`

**What it does:**
This module provides comprehensive plugin management for the Semantica framework, including dynamic plugin discovery from file system, loading, dependency resolution, and lifecycle management. It supports automatic plugin discovery from directories, version management, and plugin isolation with error handling.

**Key Features:**
- Dynamic plugin discovery from file system
- Plugin version management and compatibility checking
- Automatic dependency resolution and loading
- Plugin lifecycle management (load, unload, cleanup)
- Plugin isolation and error handling
- Plugin metadata and capability tracking

#### Class: `PluginRegistry`

Plugin registry and management system.

**Methods:**

##### `__init__(plugin_paths: Optional[List[Union[str, Path]]] = None)`
Initialize plugin registry.

**Parameters:**
- `plugin_paths` (Optional[List[Union[str, Path]]]): List of directory paths to search for plugins

##### `register_plugin(plugin_name: str, plugin_class: Type, version: str = "1.0.0", **metadata: Any) -> None`
Register a plugin.

**Parameters:**
- `plugin_name` (str): Name of the plugin
- `plugin_class` (Type): Plugin class to register
- `version` (str): Plugin version (default: "1.0.0")
- `**metadata`: Additional plugin metadata:
  - `description`: Plugin description
  - `author`: Plugin author
  - `dependencies`: List of dependency plugin names
  - `capabilities`: List of plugin capabilities

**Raises:**
- `ValidationError`: If plugin is invalid

##### `load_plugin(plugin_name: str, **config: Any) -> Any`
Load and initialize a plugin.

**Parameters:**
- `plugin_name` (str): Name of the plugin to load
- `**config`: Plugin configuration passed to plugin constructor

**Returns:**
- Loaded and initialized plugin instance

**Raises:**
- `ConfigurationError`: If plugin not found, dependencies missing, or initialization fails

##### `unload_plugin(plugin_name: str) -> None`
Unload a plugin.

**Parameters:**
- `plugin_name` (str): Name of plugin to unload

**Raises:**
- `ConfigurationError`: If plugin not loaded

##### `list_plugins() -> List[Dict[str, Any]]`
List all available plugins.

**Returns:**
- List of plugin information dictionaries

##### `get_plugin_info(plugin_name: str) -> Dict[str, Any]`
Get information about a plugin.

**Parameters:**
- `plugin_name` (str): Name of plugin

**Returns:**
- Dictionary with plugin information

**Raises:**
- `ConfigurationError`: If plugin not found

##### `is_plugin_loaded(plugin_name: str) -> bool`
Check if a plugin is loaded.

**Parameters:**
- `plugin_name` (str): Name of plugin

**Returns:**
- True if plugin is loaded, False otherwise

##### `get_loaded_plugin(plugin_name: str) -> Optional[Any]`
Get loaded plugin instance.

**Parameters:**
- `plugin_name` (str): Name of plugin

**Returns:**
- Plugin instance or None if not loaded

**Code Example:**
```python
from semantica.core import PluginRegistry

# Initialize registry with plugin paths
registry = PluginRegistry(plugin_paths=["./plugins", "./custom_plugins"])

# Register a plugin manually
class MyPlugin:
    def initialize(self):
        print("Plugin initialized")
    
    def execute(self, data):
        return f"Processed: {data}"

registry.register_plugin(
    "my_plugin",
    MyPlugin,
    version="1.0.0",
    description="My custom plugin",
    author="John Doe",
    dependencies=["base_plugin"],
    capabilities=["processing", "analysis"]
)

# Load a plugin (dependencies are automatically loaded first)
plugin = registry.load_plugin("my_plugin", config={"key": "value"})

# Use the plugin
result = plugin.execute("test data")

# List all available plugins
plugins = registry.list_plugins()
for plugin_info in plugins:
    print(f"{plugin_info['name']} v{plugin_info['version']}: {plugin_info['description']}")

# Get plugin information
info = registry.get_plugin_info("my_plugin")
print(f"Plugin loaded: {info['loaded']}")

# Check if plugin is loaded
if registry.is_plugin_loaded("my_plugin"):
    plugin_instance = registry.get_loaded_plugin("my_plugin")

# Unload plugin
registry.unload_plugin("my_plugin")
```

---

## Ingestion Modules

### `semantica.ingest.file_ingestor`

**What it does:**
This module provides comprehensive file ingestion capabilities from local filesystems and cloud storage providers (AWS S3, Google Cloud Storage, Azure Blob). It automatically detects file types using multiple methods (extension, MIME type, magic numbers), validates file sizes, and supports batch processing with progress tracking.

**Key Features:**
- Local file system scanning (recursive and filtered)
- Cloud storage integration (AWS S3, Google Cloud Storage, Azure Blob)
- Automatic file type detection (extension, MIME type, magic numbers)
- Batch processing with progress tracking
- File size validation and limits
- Support for all common document, image, audio, and video formats

#### Class: `FileIngestor`

File system and cloud storage ingestion handler.

**Methods:**

##### `__init__(config: Optional[Dict[str, Any]] = None, **kwargs)`
Initialize file ingestor.

**Parameters:**
- `config` (Optional[Dict[str, Any]]): Ingestion configuration dictionary
- `**kwargs`: Additional configuration parameters

##### `ingest_directory(directory_path: Union[str, Path], recursive: bool = True, **filters) -> List[FileObject]`
Ingest all files from a directory.

**Parameters:**
- `directory_path` (Union[str, Path]): Path to directory
- `recursive` (bool): Whether to scan subdirectories (default: True)
- `**filters`: File filtering criteria

**Returns:**
- List of ingested file objects

##### `ingest_file(file_path: Union[str, Path], **options) -> FileObject`
Ingest a single file from the filesystem.

**Parameters:**
- `file_path` (Union[str, Path]): Path to the file to ingest
- `**options`: Processing options:
  - `read_content` (bool): Whether to read file content (default: True)
  - Additional metadata to include in FileObject

**Returns:**
- `FileObject`: Ingested file object with metadata and optional content

**Raises:**
- `ValidationError`: If file doesn't exist, isn't a file, or exceeds size limits
- `ProcessingError`: If file cannot be read

##### `ingest_cloud(provider: str, bucket: str, prefix: str = "", **config) -> List[FileObject]`
Ingest files from cloud storage.

**Parameters:**
- `provider` (str): Cloud provider (s3, gcs, azure)
- `bucket` (str): Storage bucket name
- `prefix` (str): Object prefix filter (default: "")
- `**config`: Cloud provider configuration

**Returns:**
- List of ingested file objects

##### `scan_directory(directory_path: Union[str, Path], **filters) -> List[Dict[str, Any]]`
Scan directory and return file information without processing.

**Parameters:**
- `directory_path` (Union[str, Path]): Path to directory
- `**filters`: File filtering criteria:
  - `recursive` (bool): Whether to scan subdirectories (default: True)
  - `extensions` (List[str]): List of allowed extensions
  - `min_size` (int): Minimum file size
  - `max_size` (int): Maximum file size
  - `pattern` (str): Filename pattern (glob)

**Returns:**
- List of file metadata

##### `set_progress_callback(callback) -> None`
Set progress tracking callback.

**Parameters:**
- `callback`: Callback function for progress tracking

#### Class: `FileTypeDetector`

File type detection and validation.

**Methods:**

##### `__init__()`
Initialize file type detector.

##### `detect_type(file_path: Union[str, Path], content: Optional[bytes] = None) -> str`
Detect file type using multiple detection methods.

**Parameters:**
- `file_path` (Union[str, Path]): Path to file
- `content` (Optional[bytes]): Optional file content bytes for magic number detection

**Returns:**
- Detected file type (extension without dot, e.g., "pdf", "jpg")
  Returns "unknown" if type cannot be determined

##### `is_supported(file_type: str) -> bool`
Check if file type is supported.

**Parameters:**
- `file_type` (str): File type to check

**Returns:**
- Whether type is supported

**Code Example:**
```python
from semantica.ingest import FileIngestor
from pathlib import Path

# Initialize file ingestor
ingestor = FileIngestor()

# Ingest a single file
file_obj = ingestor.ingest_file(
    "document.pdf",
    read_content=True  # Read file content into memory
)

print(f"File: {file_obj.name}")
print(f"Type: {file_obj.file_type}")
print(f"Size: {file_obj.size:,} bytes")
print(f"MIME: {file_obj.mime_type}")

# Ingest entire directory (recursive)
file_objects = ingestor.ingest_directory(
    "./documents",
    recursive=True,  # Scan subdirectories
    extensions=[".pdf", ".docx", ".txt"],  # Filter by extension
    min_size=1024,  # Minimum file size (1KB)
    max_size=10485760  # Maximum file size (10MB)
)

print(f"Ingested {len(file_objects)} files")

# Scan directory without processing (faster)
file_info = ingestor.scan_directory(
    "./documents",
    recursive=True,
    extensions=[".pdf", ".docx"]
)

# Ingest from cloud storage (AWS S3)
cloud_files = ingestor.ingest_cloud(
    provider="s3",
    bucket="my-bucket",
    prefix="documents/",
    access_key_id="YOUR_KEY",
    secret_access_key="YOUR_SECRET",
    region="us-east-1"
)

# Set progress callback
def progress_callback(current, total, file_obj):
    print(f"Progress: {current}/{total} - {file_obj.name}")

ingestor.set_progress_callback(progress_callback)

# Detect file type
detector = FileTypeDetector()
file_type = detector.detect_type("document.pdf")
print(f"Detected type: {file_type}")
print(f"Supported: {detector.is_supported(file_type)}")
```

---

## Parsing Modules

### `semantica.parse.document_parser`

**What it does:**
This module handles parsing of various document formats including PDF, DOCX, HTML, and plain text files. It extracts text content, metadata, and document structure, handles embedded images and tables, supports batch document processing, and can handle password-protected documents.

**Key Features:**
- PDF text and metadata extraction
- DOCX content parsing
- HTML content cleaning
- Plain text processing
- Document structure analysis
- Batch document processing
- Password-protected document handling
- Embedded image and table extraction

#### Class: `DocumentParser`

Document format parsing handler.

**Methods:**

##### `__init__(config=None, **kwargs)`
Initialize document parser.

**Parameters:**
- `config`: Configuration dictionary
- `**kwargs`: Additional configuration options

##### `parse_document(file_path: Union[str, Path], file_type: Optional[str] = None, **options) -> Dict[str, Any]`
Parse document of any supported format.

**Parameters:**
- `file_path` (Union[str, Path]): Path to document file
- `file_type` (Optional[str]): Document type (auto-detected if None)
- `**options`: Parsing options

**Returns:**
- Parsed document data dictionary

##### `extract_text(file_path: Union[str, Path], **options) -> str`
Extract text content from document.

**Parameters:**
- `file_path` (Union[str, Path]): Path to document file
- `**options`: Parsing options

**Returns:**
- Extracted text content

##### `extract_metadata(file_path: Union[str, Path]) -> Dict[str, Any]`
Extract document metadata and properties.

**Parameters:**
- `file_path` (Union[str, Path]): Path to document file

**Returns:**
- Document metadata dictionary

##### `parse_batch(file_paths: List[Union[str, Path]], **options) -> Dict[str, Any]`
Parse multiple documents in batch.

**Parameters:**
- `file_paths` (List[Union[str, Path]]): List of document file paths
- `**options`: Parsing options:
  - `max_workers` (int): Maximum parallel workers
  - `continue_on_error` (bool): Continue on errors (default: True)

**Returns:**
- Batch processing results dictionary

**Code Example:**
```python
from semantica.parse import DocumentParser

# Initialize document parser
parser = DocumentParser()

# Parse a document (auto-detects format)
result = parser.parse_document("document.pdf")
print(f"Text length: {len(result['text'])} characters")
print(f"Metadata: {result['metadata']}")

# Extract just the text content
text = parser.extract_text("document.docx")
print(f"Extracted text: {text[:100]}...")

# Extract metadata only
metadata = parser.extract_metadata("document.pdf")
print(f"Title: {metadata.get('title')}")
print(f"Author: {metadata.get('author')}")
print(f"Pages: {metadata.get('page_count')}")

# Parse multiple documents in batch
results = parser.parse_batch(
    ["doc1.pdf", "doc2.docx", "doc3.html"],
    continue_on_error=True,  # Continue if one fails
    max_workers=4  # Parallel processing
)

print(f"Successful: {results['success_count']}")
print(f"Failed: {results['failure_count']}")

# Access parsed documents
for item in results["successful"]:
    print(f"File: {item['file_path']}")
    print(f"Text length: {len(item['result']['text'])}")
```

---

## Knowledge Graph Modules

### `semantica.kg.graph_builder`

**What it does:**
This module provides comprehensive knowledge graph construction capabilities from extracted entities and relationships. It supports temporal knowledge graphs with time-aware edges, entity resolution and deduplication, conflict detection and resolution, temporal snapshots and versioning, and Neo4j integration for graph storage.

**Key Features:**
- Build knowledge graphs from entities and relationships
- Temporal knowledge graph support with time-aware edges
- Entity resolution and deduplication
- Conflict detection and resolution
- Temporal snapshots and versioning
- Neo4j integration for graph storage

#### Class: `GraphBuilder`

Knowledge graph builder with temporal support.

**Methods:**

##### `__init__(merge_entities=True, entity_resolution_strategy="fuzzy", resolve_conflicts=True, enable_temporal=False, temporal_granularity="day", track_history=False, version_snapshots=False, **kwargs)`
Initialize graph builder.

**Parameters:**
- `merge_entities` (bool): Whether to merge duplicate entities (default: True)
- `entity_resolution_strategy` (str): Strategy for entity resolution ("fuzzy", "exact", "ml-based") (default: "fuzzy")
- `resolve_conflicts` (bool): Whether to resolve conflicts (default: True)
- `enable_temporal` (bool): Enable temporal knowledge graph features (default: False)
- `temporal_granularity` (str): Time granularity ("second", "minute", "hour", "day", "week", "month", "year") (default: "day")
- `track_history` (bool): Track historical changes (default: False)
- `version_snapshots` (bool): Create version snapshots at intervals (default: False)
- `**kwargs`: Additional configuration options

##### `build(sources: Union[List[Any], Any], entity_resolver: Optional[Any] = None, **options) -> Dict[str, Any]`
Build knowledge graph from sources.

**Parameters:**
- `sources` (Union[List[Any], Any]): List of sources in various formats:
  - Dict with "entities" and/or "relationships" keys
  - Dict with entity-like structure (has "id" or "entity_id")
  - Dict with relationship structure (has "source" and "target")
  - List of entity/relationship dicts
- `entity_resolver` (Optional[Any]): Optional custom entity resolver (overrides default)
- `**options`: Additional build options

**Returns:**
- Dictionary containing:
  - `entities`: List of resolved entities
  - `relationships`: List of relationships
  - `metadata`: Graph metadata including counts and timestamps

##### `add_temporal_edge(graph, source, target, relationship, valid_from=None, valid_until=None, temporal_metadata=None, **kwargs)`
Add edge with temporal validity information.

**Parameters:**
- `graph`: Knowledge graph to add edge to
- `source`: Source entity/node
- `target`: Target entity/node
- `relationship`: Relationship type
- `valid_from`: Start time for relationship validity (datetime, timestamp, or ISO string)
- `valid_until`: End time for relationship validity (None for ongoing)
- `temporal_metadata`: Additional temporal metadata (timezone, precision, etc.)
- `**kwargs`: Additional edge properties

**Returns:**
- Edge object with temporal annotations

##### `create_temporal_snapshot(graph, timestamp=None, snapshot_name=None, **options)`
Create temporal snapshot of graph at specific time point.

**Parameters:**
- `graph`: Knowledge graph to snapshot
- `timestamp`: Time point for snapshot (None for current time)
- `snapshot_name`: Optional name for snapshot
- `**options`: Additional snapshot options

**Returns:**
- Temporal snapshot object

##### `query_temporal(graph, query, at_time=None, time_range=None, temporal_window=None, **options)`
Query graph at specific time point or time range.

**Parameters:**
- `graph`: Knowledge graph to query
- `query`: Query (Cypher, SPARQL, or natural language)
- `at_time`: Query at specific time point
- `time_range`: Query within time range (start, end)
- `temporal_window`: Temporal window size
- `**options`: Additional query options

**Returns:**
- Query results with temporal context

##### `load_from_neo4j(uri="bolt://localhost:7687", username="neo4j", password="password", database="neo4j", enable_temporal=False, temporal_property="valid_time", **kwargs)`
Load graph from Neo4j database.

**Parameters:**
- `uri` (str): Neo4j connection URI (default: "bolt://localhost:7687")
- `username` (str): Neo4j username (default: "neo4j")
- `password` (str): Neo4j password (default: "password")
- `database` (str): Neo4j database name (default: "neo4j")
- `enable_temporal` (bool): Enable temporal features for loaded graph (default: False)
- `temporal_property` (str): Property name for temporal data (default: "valid_time")
- `**kwargs`: Additional connection options

**Returns:**
- Knowledge graph loaded from Neo4j

**Different Approaches and Strategies:**

The knowledge graph builder provides 8 different approaches for building knowledge graphs:

1. **Entity Resolution Strategies** - Three methods: fuzzy matching, exact matching, semantic similarity
2. **Temporal Graph Approaches** - Time-aware graphs with different granularities (second, minute, hour, day, week, month, year)
3. **Conflict Resolution Methods** - Automatic conflict detection and resolution (voting, credibility-weighted, recency-based)
4. **Graph Building Modes** - Incremental vs batch building, with or without entity merging
5. **Temporal Snapshot Strategies** - Version snapshots, history tracking, time-point queries
6. **Source Format Handling** - Multiple input formats (entities/relationships dicts, entity lists, relationship lists)
7. **Neo4j Integration Approaches** - Load from Neo4j, enable temporal features, custom temporal properties
8. **Graph Query Methods** - Temporal queries, time-range queries, Cypher/SPARQL queries

**When to Use Each Approach:**

| Approach | Use Case | Example |
|----------|----------|---------|
| Fuzzy Entity Resolution | Handling name variations | "Apple Inc." vs "Apple" vs "Apple Corporation" |
| Exact Entity Resolution | High precision requirements | Exact ID matching, no variations allowed |
| Semantic Entity Resolution | Context-aware matching | Using embeddings for similarity |
| Temporal Graphs | Time-sensitive data | Employee relationships, contract validity periods |
| Conflict Resolution | Multiple data sources | Resolving conflicting entity properties |
| Incremental Building | Large datasets | Add entities/relationships over time |
| Batch Building | Small datasets | Build entire graph at once |
| Temporal Snapshots | Version control | Track graph state at different times |

**Detailed Examples for Each Approach:**

**Approach 1: Entity Resolution Strategies**
```python
from semantica.kg import GraphBuilder

# Fuzzy matching (default) - handles name variations
builder_fuzzy = GraphBuilder(
    merge_entities=True,
    entity_resolution_strategy="fuzzy",  # Levenshtein, Jaro-Winkler
    similarity_threshold=0.8
)

# Exact matching - only exact string matches
builder_exact = GraphBuilder(
    merge_entities=True,
    entity_resolution_strategy="exact"  # Exact string comparison
)

# Semantic matching - uses embeddings
builder_semantic = GraphBuilder(
    merge_entities=True,
    entity_resolution_strategy="semantic",  # Embedding-based similarity
    similarity_threshold=0.85
)
```

**Approach 2: Temporal Graph Approaches**
```python
# Different temporal granularities
builder_second = GraphBuilder(
    enable_temporal=True,
    temporal_granularity="second"  # Second-level precision
)

builder_day = GraphBuilder(
    enable_temporal=True,
    temporal_granularity="day"  # Day-level precision (default)
)

builder_month = GraphBuilder(
    enable_temporal=True,
    temporal_granularity="month"  # Month-level precision
)

# Add temporal edge with validity period
builder.add_temporal_edge(
    graph,
    source="e1",
    target="e3",
    relationship="works_at",
    valid_from="2020-01-01",
    valid_until="2023-12-31"
)
```

**Approach 3: Conflict Resolution Methods**
```python
# Automatic conflict resolution
builder = GraphBuilder(
    resolve_conflicts=True,  # Automatically resolve conflicts
    conflict_resolution_strategy="voting"  # or "credibility", "recency", "confidence"
)

# Manual conflict resolution
builder = GraphBuilder(
    resolve_conflicts=False  # Detect but don't auto-resolve
)
```

**Approach 4: Graph Building Modes**
```python
# Incremental building (add entities over time)
builder = GraphBuilder(merge_entities=True)
graph1 = builder.build([{"entities": [e1, e2]}])
graph2 = builder.build([{"entities": [e3, e4]}], existing_graph=graph1)

# Batch building (build entire graph at once)
sources = [
    {"entities": [e1, e2, e3], "relationships": [r1, r2]},
    {"entities": [e4, e5], "relationships": [r3]}
]
graph = builder.build(sources)
```

**Approach 5: Temporal Snapshot Strategies**
```python
# Create snapshots at specific times
snapshot_2022 = builder.create_temporal_snapshot(
    graph,
    timestamp="2022-06-15",
    snapshot_name="mid_2022"
)

# Track history
builder = GraphBuilder(
    track_history=True,  # Track all changes
    version_snapshots=True  # Create version snapshots
)

# Query at specific time point
results = builder.query_temporal(
    graph,
    query="MATCH (p:Person)-[:works_at]->(o:Organization) RETURN p, o",
    at_time="2022-06-15"
)
```

**Approach 6: Source Format Handling**
```python
# Format 1: Entities and relationships dict
source1 = {
    "entities": [{"id": "e1", "name": "Alice"}],
    "relationships": [{"source": "e1", "target": "e2", "type": "knows"}]
}

# Format 2: Entity list
source2 = [{"id": "e1", "name": "Alice"}, {"id": "e2", "name": "Bob"}]

# Format 3: Relationship list
source3 = [{"source": "e1", "target": "e2", "type": "knows"}]

# All formats work
graph = builder.build([source1, source2, source3])
```

**Approach 7: Neo4j Integration**
```python
# Load from Neo4j
graph = builder.load_from_neo4j(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    enable_temporal=True,  # Enable temporal features
    temporal_property="valid_time"  # Custom temporal property name
)
```

**Approach 8: Graph Query Methods**
```python
# Temporal query at specific time
results = builder.query_temporal(
    graph,
    query="MATCH (p:Person) RETURN p",
    at_time="2022-06-15"
)

# Time range query
results = builder.query_temporal(
    graph,
    query="MATCH (p:Person)-[:works_at]->(o:Organization) RETURN p, o",
    time_range=("2020-01-01", "2023-12-31")
)

# Cypher query
results = builder.query_temporal(
    graph,
    query="MATCH (p:Person)-[:knows*2]->(f:Person) RETURN p, f",
    at_time="2022-06-15"
)
```

**Code Example:**
```python
from semantica.kg import GraphBuilder

# Initialize graph builder with entity resolution and conflict detection
builder = GraphBuilder(
    merge_entities=True,              # Merge duplicate entities
    entity_resolution_strategy="fuzzy",  # Use fuzzy matching
    resolve_conflicts=True,           # Automatically resolve conflicts
    enable_temporal=True,             # Enable temporal features
    temporal_granularity="day",       # Time granularity
    track_history=True                # Track changes over time
)

# Build knowledge graph from sources
sources = [
    {
        "entities": [
            {"id": "e1", "name": "Alice", "type": "Person"},
            {"id": "e2", "name": "Bob", "type": "Person"},
            {"id": "e3", "name": "Company X", "type": "Organization"}
        ],
        "relationships": [
            {"source": "e1", "target": "e2", "type": "knows"},
            {"source": "e1", "target": "e3", "type": "works_at"}
        ]
    }
]

graph = builder.build(sources)
print(f"Entities: {len(graph['entities'])}")
print(f"Relationships: {len(graph['relationships'])}")

# Add temporal edge (relationship valid for specific time period)
builder.add_temporal_edge(
    graph,
    source="e1",
    target="e3",
    relationship="works_at",
    valid_from="2020-01-01",
    valid_until="2023-12-31",
    temporal_metadata={"timezone": "UTC"}
)

# Create temporal snapshot (graph state at specific time)
snapshot = builder.create_temporal_snapshot(
    graph,
    timestamp="2022-06-15",
    snapshot_name="mid_2022"
)

# Query graph at specific time point
results = builder.query_temporal(
    graph,
    query="MATCH (p:Person)-[:works_at]->(o:Organization) RETURN p, o",
    at_time="2022-06-15"
)

# Load graph from Neo4j
neo4j_graph = builder.load_from_neo4j(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
    enable_temporal=True
)
```

---

## Embeddings Modules

### `semantica.embeddings.embedding_generator`

**What it does:**
This module provides comprehensive embedding generation capabilities for text, images, audio, and multi-modal content. It supports multiple embedding models (sentence-transformers, OpenAI, BGE, CLIP), batch processing for efficiency, embedding optimization and compression, and similarity comparison utilities.

**Key Features:**
- Text embedding generation (multiple models: sentence-transformers, OpenAI, BGE, etc.)
- Image embedding generation
- Audio embedding generation
- Multi-modal embedding support
- Batch processing for efficiency
- Embedding optimization and compression
- Similarity comparison utilities

#### Class: `EmbeddingGenerator`

Main embedding generation handler.

**Methods:**

##### `__init__(config: Optional[Dict[str, Any]] = None, **kwargs)`
Initialize embedding generator.

**Parameters:**
- `config` (Optional[Dict[str, Any]]): Configuration dictionary with keys:
  - `text`: Text embedder configuration
  - `image`: Image embedder configuration
  - `audio`: Audio embedder configuration
  - `multimodal`: Multi-modal embedder configuration
  - `optimizer`: Embedding optimizer configuration
- `**kwargs`: Additional configuration (merged into config)

##### `generate_embeddings(data: Union[str, Path, List[Union[str, Path]]], data_type: Optional[str] = None, **options) -> np.ndarray`
Generate embeddings for input data.

**Parameters:**
- `data` (Union[str, Path, List[Union[str, Path]]]): Input data to embed:
  - str: Text string or file path
  - Path: File path object
  - List: Batch of texts or file paths
- `data_type` (Optional[str]): Explicit data type ("text", "image", "audio"). If None, auto-detects from input
- `**options`: Additional generation options passed to embedder

**Returns:**
- `np.ndarray`: Generated embeddings
  - For single input: 1D array
  - For batch input: 2D array (batch_size, embedding_dim)

**Raises:**
- `ProcessingError`: If data type is unsupported or embedding fails

##### `optimize_embeddings(embeddings: np.ndarray, **options) -> np.ndarray`
Optimize embedding quality and performance.

**Parameters:**
- `embeddings` (np.ndarray): Input embeddings
- `**options`: Optimization options

**Returns:**
- `np.ndarray`: Optimized embeddings

##### `compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray, **options) -> float`
Compare embeddings for similarity.

**Parameters:**
- `embedding1` (np.ndarray): First embedding
- `embedding2` (np.ndarray): Second embedding
- `**options`: Comparison options:
  - `method` (str): Similarity method ("cosine", "euclidean") (default: "cosine")

**Returns:**
- `float`: Similarity score (0-1)

##### `process_batch(data_items: List[Union[str, Path]], **options) -> Dict[str, Any]`
Process multiple data items for embedding generation.

**Parameters:**
- `data_items` (List[Union[str, Path]]): List of data items
- `**options`: Processing options

**Returns:**
- Dictionary with batch processing results:
  - `embeddings`: List of generated embeddings
  - `successful`: List of successfully processed items
  - `failed`: List of failed items with error information
  - `total`: Total number of items
  - `success_count`: Number of successful items
  - `failure_count`: Number of failed items

**Different Approaches and Strategies:**

The embeddings module supports multiple embedding providers and models, each optimized for different use cases:

1. **Sentence-Transformers** - Open-source, high-quality sentence embeddings (recommended for most use cases)
2. **OpenAI Embeddings** - Cloud-based, high-quality embeddings via API
3. **BGE (BAAI General Embedding)** - State-of-the-art multilingual embeddings
4. **CLIP** - Multi-modal embeddings for images and text
5. **Custom Models** - Support for custom embedding models

**When to Use Each Provider:**

- **Sentence-Transformers**: Local processing, no API costs, good quality, open-source
- **OpenAI**: Highest quality, cloud-based, requires API key, paid service
- **BGE**: Multilingual support, best for non-English text, open-source
- **CLIP**: Image-text similarity, multi-modal applications
- **Custom Models**: Domain-specific embeddings, fine-tuned models

**Comparison of Embedding Providers:**

| Provider | Quality | Speed | Cost | Multilingual | Best For |
|----------|---------|-------|------|--------------|----------|
| Sentence-Transformers | High | Fast | Free | Limited | General purpose |
| OpenAI | Very High | Medium | Paid | Yes | Production, quality-critical |
| BGE | Very High | Fast | Free | Yes | Multilingual applications |
| CLIP | High | Medium | Free | Limited | Image-text tasks |

**Code Examples for Different Approaches:**

**Approach 1: Sentence-Transformers (Open-source, Recommended)**
```python
from semantica.embeddings import EmbeddingGenerator

# Initialize with sentence-transformers (default, no API key needed)
generator = EmbeddingGenerator(
    config={
        "text": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",  # Fast, 384-dim
            # Alternative models:
            # "all-mpnet-base-v2" - Higher quality, 768-dim
            # "paraphrase-multilingual-MiniLM-L12-v2" - Multilingual
            "device": "cpu",  # or "cuda" for GPU
            "normalize": True
        }
    }
)

# Generate single embedding
text = "The quick brown fox jumps over the lazy dog"
embedding = generator.generate_embeddings(text, data_type="text")
print(f"Embedding shape: {embedding.shape}")  # (384,)

# Batch processing (more efficient)
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = generator.generate_embeddings(texts, data_type="text")
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, 384)
```

**Approach 2: OpenAI Embeddings (Cloud-based, High Quality)**
```python
from semantica.embeddings import EmbeddingGenerator
import os

# Set OpenAI API key (or use environment variable)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize with OpenAI adapter
generator = EmbeddingGenerator(
    config={
        "text": {
            "provider": "openai",
            "model": "text-embedding-3-small",  # or "text-embedding-3-large"
            # text-embedding-3-small: 1536 dimensions, fast, cost-effective
            # text-embedding-3-large: 3072 dimensions, highest quality
        }
    }
)

# Generate embeddings (same API as sentence-transformers)
embedding = generator.generate_embeddings(
    "Your text here",
    data_type="text"
)
print(f"OpenAI embedding shape: {embedding.shape}")  # (1536,) or (3072,)

# Batch processing with OpenAI (handles rate limits automatically)
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = generator.generate_embeddings(texts, data_type="text")
```

**Approach 3: BGE Embeddings (Multilingual, High Quality)**
```python
from semantica.embeddings import EmbeddingGenerator

# Initialize with BGE model (excellent for multilingual)
generator = EmbeddingGenerator(
    config={
        "text": {
            "provider": "bge",
            "model_name": "BAAI/bge-small-en-v1.5",  # English
            # Alternative: "BAAI/bge-m3" - Multilingual, 1024-dim
            # Alternative: "BAAI/bge-large-en-v1.5" - Higher quality, 1024-dim
        }
    }
)

# Generate embeddings for English text
english_text = "Hello, world!"
embedding = generator.generate_embeddings(english_text, data_type="text")

# BGE models work well with multiple languages
multilingual_texts = [
    "Hello, world!",           # English
    "Bonjour le monde!",       # French
    "Hola, mundo!",            # Spanish
    "你好，世界！"              # Chinese
]
embeddings = generator.generate_embeddings(multilingual_texts, data_type="text")
```

**Approach 4: Using Provider Adapters Directly**
```python
from semantica.embeddings import ProviderAdapterFactory

# Create provider adapter directly (more control)
openai_adapter = ProviderAdapterFactory.create(
    "openai",
    api_key="your-key",
    model="text-embedding-3-small"
)

# Use adapter directly
embedding = openai_adapter.embed("Your text")
batch_embeddings = openai_adapter.embed_batch(["Text 1", "Text 2"])

# Switch providers easily
bge_adapter = ProviderAdapterFactory.create(
    "bge",
    model_name="BAAI/bge-small-en-v1.5"
)
bge_embedding = bge_adapter.embed("Your text")
```

**Approach 5: Multi-modal Embeddings (Text + Images)**
```python
from semantica.embeddings import EmbeddingGenerator

# Initialize with CLIP for multi-modal embeddings
generator = EmbeddingGenerator(
    config={
        "text": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        "image": {"model": "clip-vit-base-patch32"},  # CLIP model
        "multimodal": {
            "model": "clip-vit-base-patch32"  # For image-text similarity
        }
    }
)

# Generate text embedding
text_embedding = generator.generate_embeddings(
    "A photo of a cat",
    data_type="text"
)

# Generate image embedding
image_embedding = generator.generate_embeddings(
    "cat_photo.jpg",
    data_type="image"
)

# Compare text and image embeddings (CLIP enables cross-modal similarity)
similarity = generator.compare_embeddings(
    text_embedding,
    image_embedding,
    method="cosine"
)
print(f"Text-Image similarity: {similarity:.4f}")
```

**Approach 6: Embedding Optimization and Compression**
```python
from semantica.embeddings import EmbeddingGenerator
import numpy as np

generator = EmbeddingGenerator()

# Generate embeddings
texts = ["Document 1", "Document 2", "Document 3"]
embeddings = generator.generate_embeddings(texts, data_type="text")
print(f"Original shape: {embeddings.shape}")  # (3, 384)

# Optimize embeddings (reduce dimensionality, improve quality)
optimized = generator.optimize_embeddings(
    embeddings,
    method="pca",           # Principal Component Analysis
    target_dim=256         # Reduce from 384 to 256 dimensions
)
print(f"Optimized shape: {optimized.shape}")  # (3, 256)

# Compare original vs optimized (should maintain similarity structure)
original_sim = generator.compare_embeddings(embeddings[0], embeddings[1])
optimized_sim = generator.compare_embeddings(optimized[0], optimized[1])
print(f"Original similarity: {original_sim:.4f}")
print(f"Optimized similarity: {optimized_sim:.4f}")  # Should be similar
```

**Approach 7: Batch Processing with Error Handling**
```python
from semantica.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()

# Process large batch with error handling
file_paths = ["doc1.txt", "doc2.txt", "doc3.txt", "invalid_file.txt"]

results = generator.process_batch(
    file_paths,
    data_type="text",
    continue_on_error=True,  # Continue even if some fail
    batch_size=32             # Process in batches of 32
)

print(f"Total: {results['total']}")
print(f"Successful: {results['success_count']}")
print(f"Failed: {results['failure_count']}")

# Access successful embeddings
for i, embedding in enumerate(results['embeddings']):
    if embedding is not None:
        print(f"Document {i}: {embedding.shape}")

# Check failed items
for failed_item in results['failed']:
    print(f"Failed: {failed_item['item']} - {failed_item['error']}")
```

**Approach 8: Similarity Search and Comparison**
```python
from semantica.embeddings import EmbeddingGenerator
import numpy as np

generator = EmbeddingGenerator()

# Generate embeddings for query and documents
query = "machine learning algorithms"
documents = [
    "Deep learning neural networks",
    "Statistical analysis methods",
    "Computer vision applications",
    "Natural language processing"
]

query_embedding = generator.generate_embeddings(query, data_type="text")
doc_embeddings = generator.generate_embeddings(documents, data_type="text")

# Find most similar document
similarities = []
for doc_emb in doc_embeddings:
    sim = generator.compare_embeddings(
        query_embedding,
        doc_emb,
        method="cosine"  # or "euclidean"
    )
    similarities.append(sim)

# Get top-k most similar
top_k = 2
top_indices = np.argsort(similarities)[-top_k:][::-1]

print(f"Query: {query}")
for i, idx in enumerate(top_indices):
    print(f"{i+1}. {documents[idx]} (similarity: {similarities[idx]:.4f})")
```

**Best Practices:**

1. **Choose provider based on requirements**:
   - Local/offline → Sentence-Transformers or BGE
   - Highest quality → OpenAI
   - Multilingual → BGE or OpenAI
   - Image-text → CLIP

2. **Use batch processing**:
   - Always use batch processing for multiple items
   - Set appropriate batch_size based on memory
   - Process in batches for large datasets

3. **Normalize embeddings**:
   - Always normalize embeddings for cosine similarity
   - Use unit vectors for better similarity calculations

4. **Handle errors gracefully**:
   - Use `process_batch` with `continue_on_error=True`
   - Log failed items for debugging
   - Retry failed items with exponential backoff

5. **Optimize for your use case**:
   - Use smaller models for speed (all-MiniLM-L6-v2)
   - Use larger models for quality (all-mpnet-base-v2)
   - Consider dimensionality reduction for storage

---

## Pipeline Modules

### `semantica.pipeline.pipeline_builder`

**What it does:**
This module handles construction and configuration of processing pipelines, providing a fluent DSL for building workflows, step chaining, validation, and serialization. It supports complex pipeline topologies, dependency management, step status tracking, error handling and recovery, and pipeline versioning.

**Key Features:**
- Pipeline construction DSL
- Step configuration and chaining
- Pipeline validation and optimization
- Error handling and recovery
- Pipeline serialization and deserialization
- Dependency management
- Step status tracking

#### Class: `PipelineBuilder`

Pipeline construction and configuration handler.

**Methods:**

##### `__init__(config=None, **kwargs)`
Initialize pipeline builder.

**Parameters:**
- `config`: Configuration dictionary
- `**kwargs`: Additional configuration options

##### `add_step(step_name: str, step_type: str, **config) -> "PipelineBuilder"`
Add step to pipeline.

**Parameters:**
- `step_name` (str): Step name/identifier
- `step_type` (str): Step type/category
- `**config`: Step configuration

**Returns:**
- Self for method chaining

##### `connect_steps(from_step: str, to_step: str, **options) -> "PipelineBuilder"`
Connect pipeline steps.

**Parameters:**
- `from_step` (str): Source step name
- `to_step` (str): Target step name
- `**options`: Connection options

**Returns:**
- Self for method chaining

##### `set_parallelism(level: int) -> "PipelineBuilder"`
Set parallelism level.

**Parameters:**
- `level` (int): Parallelism level (number of parallel workers)

**Returns:**
- Self for method chaining

##### `build(name: str = "default_pipeline") -> Pipeline`
Build pipeline from configuration.

**Parameters:**
- `name` (str): Pipeline name (default: "default_pipeline")

**Returns:**
- Built pipeline

##### `build_pipeline(pipeline_config: Dict[str, Any], **options) -> Pipeline`
Build pipeline from configuration dictionary.

**Parameters:**
- `pipeline_config` (Dict[str, Any]): Pipeline configuration
- `**options`: Additional options

**Returns:**
- Built pipeline

##### `register_step_handler(step_type: str, handler: Callable) -> None`
Register step handler function.

**Parameters:**
- `step_type` (str): Step type
- `handler` (Callable): Handler function

##### `get_step(step_name: str) -> Optional[PipelineStep]`
Get step by name.

**Parameters:**
- `step_name` (str): Step name

**Returns:**
- PipelineStep or None if not found

##### `serialize(format: str = "json") -> Union[str, Dict[str, Any]]`
Serialize pipeline configuration.

**Parameters:**
- `format` (str): Serialization format (default: "json")

**Returns:**
- Serialized pipeline (string or dictionary)

##### `validate_pipeline() -> Dict[str, Any]`
Validate pipeline structure and configuration.

**Returns:**
- Validation results dictionary

#### Class: `PipelineSerializer`

Pipeline serialization handler.

**Methods:**

##### `__init__(**config)`
Initialize pipeline serializer.

**Parameters:**
- `**config`: Configuration options

##### `serialize_pipeline(pipeline: Pipeline, format: str = "json", **options) -> Union[str, Dict[str, Any]]`
Serialize pipeline to specified format.

**Parameters:**
- `pipeline` (Pipeline): Pipeline object
- `format` (str): Serialization format (default: "json")
- `**options`: Additional options

**Returns:**
- Serialized pipeline (string or dictionary)

##### `deserialize_pipeline(serialized_pipeline: Union[str, Dict[str, Any]], **options) -> Pipeline`
Deserialize pipeline from serialized format.

**Parameters:**
- `serialized_pipeline` (Union[str, Dict[str, Any]]): Serialized pipeline data
- `**options`: Additional options

**Returns:**
- Reconstructed pipeline

##### `version_pipeline(pipeline: Pipeline, version_info: Dict[str, Any]) -> Pipeline`
Add versioning information to pipeline.

**Parameters:**
- `pipeline` (Pipeline): Pipeline object
- `version_info` (Dict[str, Any]): Version information

**Returns:**
- Versioned pipeline

**Code Example:**
```python
from semantica.pipeline import PipelineSerializer, Pipeline

# Initialize serializer
serializer = PipelineSerializer()

# Serialize pipeline to JSON
json_str = serializer.serialize_pipeline(
    pipeline,
    format="json"
)

# Deserialize pipeline from JSON
restored_pipeline = serializer.deserialize_pipeline(json_str)

# Add versioning information
versioned = serializer.version_pipeline(
    pipeline,
    version_info={
        "version": "1.2.0",
        "author": "John Doe",
        "date": "2024-01-15"
    }
)
```

---

## Semantic Extraction Modules

### `semantica.semantic_extract.ner_extractor`

**What it does:**
This module provides core Named Entity Recognition (NER) capabilities using spaCy and transformers for entity identification and classification, with fallback pattern-based extraction. It supports multiple entity types, confidence scoring, batch processing, and entity filtering by confidence.

**Key Features:**
- spaCy-based entity extraction
- Pattern-based fallback extraction
- Multiple entity type support
- Confidence scoring
- Batch processing
- Entity filtering by confidence

#### Class: `NERExtractor`

Named Entity Recognition extractor using spaCy and pattern-based fallback.

**Methods:**

##### `__init__(**config)`
Initialize NER extractor.

**Parameters:**
- `**config`: Configuration options:
  - `model` (str): Model name (default: "en_core_web_sm")
  - `language` (str): Language code (default: "en")
  - `min_confidence` (float): Minimum confidence threshold (default: 0.5)

##### `extract_entities(text: str, **options) -> List[Entity]`
Extract named entities from text.

**Parameters:**
- `text` (str): Input text
- `**options`: Extraction options:
  - `entity_types` (List[str]): Filter by entity types
  - `min_confidence` (float): Minimum confidence threshold

**Returns:**
- List of extracted Entity objects

##### `extract_entities_batch(texts: List[str], **options) -> List[List[Entity]]`
Extract entities from multiple texts.

**Parameters:**
- `texts` (List[str]): List of input texts
- `**options`: Extraction options

**Returns:**
- List of entity lists for each text

##### `classify_entities(entities: List[Entity]) -> Dict[str, List[Entity]]`
Classify entities by type.

**Parameters:**
- `entities` (List[Entity]): List of entities

**Returns:**
- Dictionary with entities grouped by type

##### `filter_by_confidence(entities: List[Entity], min_confidence: float) -> List[Entity]`
Filter entities by confidence score.

**Parameters:**
- `entities` (List[Entity]): List of entities
- `min_confidence` (float): Minimum confidence threshold

**Returns:**
- Filtered entities

#### Class: `Entity`

Entity representation dataclass.

**Attributes:**
- `text` (str): Entity text
- `label` (str): Entity label/type
- `start_char` (int): Start character position
- `end_char` (int): End character position
- `confidence` (float): Confidence score (default: 1.0)
- `metadata` (Dict[str, Any]): Additional metadata

**Different Entity Recognition Methods and Strategies:**

The NER module supports multiple extraction methods, from exact keyword matching to advanced similarity-based approaches:

1. **spaCy-based NER** - Machine learning model for entity recognition (default, highest quality)
2. **Pattern-based Fallback** - Regex patterns for exact matching when spaCy unavailable
3. **Keyword Exact Matching** - Simple exact string matching
4. **Confidence-based Filtering** - Filter entities by confidence scores
5. **Type-based Filtering** - Filter by entity types (PERSON, ORG, GPE, etc.)
6. **Batch Processing** - Process multiple texts efficiently

**When to Use Each Method:**

- **spaCy NER**: Production use, high accuracy needed, multiple entity types
- **Pattern-based**: Fallback when spaCy unavailable, specific entity patterns known
- **Keyword Matching**: Simple use cases, known entity lists, fast processing
- **Confidence Filtering**: Quality control, reduce false positives
- **Type Filtering**: Focus on specific entity categories

**Comparison of NER Methods:**

| Method | Accuracy | Speed | Requires Model | Best For |
|--------|----------|-------|----------------|----------|
| spaCy NER | Very High | Medium | Yes | Production, general purpose |
| Pattern-based | Medium | Fast | No | Fallback, known patterns |
| Keyword Exact | Low | Very Fast | No | Simple lists, fast lookup |

**Code Examples for All NER Methods:**

**Method 1: spaCy-based NER (Machine Learning, Recommended)**
```python
from semantica.semantic_extract import NERExtractor

# Initialize with spaCy model (highest quality)
extractor = NERExtractor(
    model="en_core_web_sm",      # spaCy English model
    # Alternative models:
    # "en_core_web_md" - Medium model, better accuracy
    # "en_core_web_lg" - Large model, best accuracy
    # "en_core_web_trf" - Transformer model, highest accuracy
    language="en",
    min_confidence=0.7
)

# Extract entities (uses spaCy's ML model)
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
entities = extractor.extract_entities(text)

# spaCy automatically detects:
# - PERSON: "Steve Jobs"
# - ORG: "Apple Inc."
# - GPE: "Cupertino", "California"
# - DATE: "1976"

for entity in entities:
    print(f"{entity.text} ({entity.label}): {entity.confidence:.2f}")
```

**Method 2: Pattern-based Fallback (Regex Patterns, No Model Required)**
```python
from semantica.semantic_extract import NERExtractor

# Initialize without spaCy (uses pattern-based fallback)
extractor = NERExtractor(
    model=None,  # No spaCy model
    language="en"
)

# Pattern-based extraction uses regex patterns:
# - PERSON: Capitalized names (e.g., "Steve Jobs")
# - ORG: Company names with suffixes (e.g., "Apple Inc.")
# - GPE: Location names (e.g., "New York City")
# - DATE: Date patterns (e.g., "1976", "01/01/2024")

text = "Apple Inc. was founded by Steve Jobs in 1976."
entities = extractor.extract_entities(text)  # Uses _extract_fallback()

# Pattern-based has lower confidence (0.7 default)
for entity in entities:
    print(f"{entity.text} ({entity.label}): {entity.confidence:.2f}")
    print(f"  Method: {entity.metadata.get('extraction_method', 'unknown')}")
```

**Method 3: Keyword Exact Matching (Custom Entity Lists)**
```python
from semantica.semantic_extract import NERExtractor

# Create custom keyword list for exact matching
known_entities = {
    "PERSON": ["Steve Jobs", "Tim Cook", "Bill Gates"],
    "ORG": ["Apple Inc.", "Microsoft", "Google"],
    "GPE": ["Cupertino", "Redmond", "Mountain View"]
}

# Extract using exact matching
text = "Steve Jobs founded Apple Inc. in Cupertino."
extracted = []

for entity_type, keywords in known_entities.items():
    for keyword in keywords:
        if keyword.lower() in text.lower():
            # Find position
            start = text.lower().find(keyword.lower())
            if start >= 0:
                extracted.append({
                    "text": keyword,
                    "label": entity_type,
                    "start_char": start,
                    "end_char": start + len(keyword),
                    "confidence": 1.0,  # Exact match = 100% confidence
                    "method": "exact_keyword_match"
                })

print(f"Found {len(extracted)} entities via exact matching")
```

**Method 4: Confidence-based Filtering (Quality Control)**
```python
from semantica.semantic_extract import NERExtractor

extractor = NERExtractor(min_confidence=0.5)  # Low threshold initially

# Extract all entities
text = "Apple Inc. was founded by Steve Jobs in 1976."
all_entities = extractor.extract_entities(text, min_confidence=0.0)  # Get all

print(f"Total entities found: {len(all_entities)}")

# Filter by different confidence thresholds
high_confidence = extractor.filter_by_confidence(all_entities, min_confidence=0.9)
medium_confidence = extractor.filter_by_confidence(all_entities, min_confidence=0.7)
low_confidence = extractor.filter_by_confidence(all_entities, min_confidence=0.5)

print(f"High confidence (>=0.9): {len(high_confidence)}")
print(f"Medium confidence (>=0.7): {len(medium_confidence)}")
print(f"Low confidence (>=0.5): {len(low_confidence)}")

# Use case: Progressive filtering
# Start with high confidence, lower threshold if not enough entities
if len(high_confidence) < 3:
    entities_to_use = medium_confidence
else:
    entities_to_use = high_confidence
```

**Method 5: Type-based Filtering (Entity Category Selection)**
```python
from semantica.semantic_extract import NERExtractor

extractor = NERExtractor()

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."

# Extract only specific entity types
persons_only = extractor.extract_entities(
    text,
    entity_types=["PERSON"],  # Only persons
    min_confidence=0.8
)
print(f"Persons: {[e.text for e in persons_only]}")

# Extract organizations and locations
orgs_and_locations = extractor.extract_entities(
    text,
    entity_types=["ORG", "GPE"],  # Organizations and locations
    min_confidence=0.8
)
print(f"Orgs & Locations: {[e.text for e in orgs_and_locations]}")

# Extract all types, then classify
all_entities = extractor.extract_entities(text)
classified = extractor.classify_entities(all_entities)

for entity_type, entity_list in classified.items():
    print(f"{entity_type}: {[e.text for e in entity_list]}")
```

**Method 6: Batch Processing (Efficient Multi-text Processing)**
```python
from semantica.semantic_extract import NERExtractor

extractor = NERExtractor()

# Process multiple texts efficiently
texts = [
    "Microsoft is located in Redmond, Washington.",
    "Tim Cook is the CEO of Apple.",
    "Google was founded in Mountain View, California."
]

# Batch extraction (more efficient than individual calls)
batch_entities = extractor.extract_entities_batch(
    texts,
    entity_types=["PERSON", "ORG", "GPE"],
    min_confidence=0.8
)

# Returns list of entity lists (one per text)
for i, entities in enumerate(batch_entities):
    print(f"Text {i+1}: {len(entities)} entities")
    for entity in entities:
        print(f"  - {entity.text} ({entity.label})")
```

**Method 7: Hybrid Approach (Combine Multiple Methods)**
```python
from semantica.semantic_extract import NERExtractor

extractor = NERExtractor()

text = "Apple Inc. was founded by Steve Jobs in 1976."

# Step 1: Extract with spaCy (if available)
spacy_entities = extractor.extract_entities(text, min_confidence=0.8)

# Step 2: Add custom keyword matches
custom_keywords = {
    "PERSON": ["Steve Jobs", "Tim Cook"],
    "ORG": ["Apple Inc.", "Apple"]
}

all_entities = list(spacy_entities)

# Add exact keyword matches not found by spaCy
for entity_type, keywords in custom_keywords.items():
    for keyword in keywords:
        if keyword.lower() in text.lower():
            # Check if already extracted
            if not any(e.text == keyword for e in all_entities):
                start = text.lower().find(keyword.lower())
                all_entities.append({
                    "text": keyword,
                    "label": entity_type,
                    "start_char": start,
                    "end_char": start + len(keyword),
                    "confidence": 1.0,
                    "method": "custom_keyword"
                })

print(f"Total entities (spaCy + custom): {len(all_entities)}")
```

**Best Practices:**

1. **Use spaCy for production** - Highest accuracy, supports many entity types
2. **Set appropriate confidence thresholds** - Balance between recall and precision
3. **Filter by entity types** - Focus on relevant entity categories
4. **Use batch processing** - More efficient for multiple texts
5. **Combine methods** - Use spaCy + custom keywords for best coverage
6. **Validate extracted entities** - Check confidence scores and positions

---

## Normalization Modules

### `semantica.normalize.text_normalizer`

**What it does:**
This module provides comprehensive text normalization capabilities for the Semantica framework, enabling standardization of text content across various formats and encodings. It handles Unicode normalization, whitespace handling, special character processing, case normalization, and format standardization.

**Key Features:**
- Text cleaning and sanitization
- Unicode normalization (NFC, NFD, NFKC, NFKD)
- Case normalization (lower, upper, title, preserve)
- Whitespace handling (normalization, line breaks, indentation)
- Special character processing (punctuation, diacritics)
- Format standardization

#### Class: `TextNormalizer`

Text normalization and cleaning coordinator.

**Methods:**

##### `__init__(config: Optional[Dict[str, Any]] = None, **kwargs)`
Initialize text normalizer.

**Parameters:**
- `config` (Optional[Dict[str, Any]]): Configuration dictionary
- `**kwargs`: Additional configuration options

##### `normalize_text(text: str, unicode_form: str = "NFC", case: str = "preserve", normalize_diacritics: bool = False, line_break_type: str = "unix", **options) -> str`
Normalize text content.

**Parameters:**
- `text` (str): Input text to normalize
- `unicode_form` (str): Unicode normalization form (default: "NFC"):
  - "NFC": Canonical composition
  - "NFD": Canonical decomposition
  - "NFKC": Compatibility composition
  - "NFKD": Compatibility decomposition
- `case` (str): Case normalization type (default: "preserve"):
  - "preserve": Keep original case
  - "lower": Convert to lowercase
  - "upper": Convert to uppercase
  - "title": Convert to title case
- `normalize_diacritics` (bool): Whether to normalize diacritics (default: False)
- `line_break_type` (str): Line break type (default: "unix")
- `**options`: Additional normalization options

**Returns:**
- Normalized text string

##### `clean_text(text: str, **options) -> str`
Clean and sanitize text content.

**Parameters:**
- `text` (str): Input text to clean
- `**options`: Cleaning options (passed to TextCleaner.clean)

**Returns:**
- Cleaned text string

##### `standardize_format(text: str, format_type: str = "standard") -> str`
Standardize text format.

**Parameters:**
- `text` (str): Input text to standardize
- `format_type` (str): Format type (default: "standard"):
  - "standard": Apply standard formatting
  - "compact": Remove extra whitespace
  - "preserve": Preserve original formatting

**Returns:**
- Formatted text string

##### `process_batch(texts: List[str], **options) -> List[str]`
Process multiple texts in batch.

**Parameters:**
- `texts` (List[str]): List of texts to process
- `**options`: Processing options

**Returns:**
- List of normalized texts

#### Class: `UnicodeNormalizer`

Unicode normalization engine.

**Methods:**

##### `normalize_unicode(text: str, form: str = "NFC") -> str`
Normalize Unicode text.

**Parameters:**
- `text` (str): Input text to normalize
- `form` (str): Unicode normalization form (default: "NFC")

**Returns:**
- Unicode-normalized text

##### `handle_encoding(text: str, source_encoding: str, target_encoding: str = "utf-8") -> str`
Handle text encoding conversion.

**Parameters:**
- `text` (str): Input text (string or bytes)
- `source_encoding` (str): Source encoding name
- `target_encoding` (str): Target encoding name (default: "utf-8")

**Returns:**
- Converted text in target encoding

##### `process_special_chars(text: str) -> str`
Process special Unicode characters.

**Parameters:**
- `text` (str): Input text containing special Unicode characters

**Returns:**
- Text with special Unicode characters replaced with ASCII equivalents

#### Class: `WhitespaceNormalizer`

Whitespace normalization engine.

**Methods:**

##### `normalize_whitespace(text: str, line_break_type: str = "unix", **options) -> str`
Normalize whitespace in text.

**Parameters:**
- `text` (str): Input text with potentially irregular whitespace
- `line_break_type` (str): Line break type (default: "unix")
- `**options`: Additional normalization options

**Returns:**
- Text with normalized whitespace

##### `handle_line_breaks(text: str, line_break_type: str = "unix") -> str`
Normalize line breaks.

**Parameters:**
- `text` (str): Input text with potentially mixed line breaks
- `line_break_type` (str): Line break type (default: "unix")

**Returns:**
- Text with normalized line breaks

##### `process_indentation(text: str, indent_type: str = "spaces") -> str`
Normalize text indentation.

**Parameters:**
- `text` (str): Input text with potentially mixed indentation
- `indent_type` (str): Indentation type (default: "spaces")

**Returns:**
- Text with normalized indentation

#### Class: `SpecialCharacterProcessor`

Special character processing engine.

**Methods:**

##### `process_special_chars(text: str, normalize_diacritics: bool = False, **options) -> str`
Process special characters in text.

**Parameters:**
- `text` (str): Input text to process
- `normalize_diacritics` (bool): Whether to normalize diacritics (default: False)
- `**options`: Additional processing options

**Returns:**
- Text with special characters processed

##### `normalize_punctuation(text: str) -> str`
Normalize punctuation marks.

**Parameters:**
- `text` (str): Input text with potentially mixed punctuation

**Returns:**
- Text with normalized punctuation marks

##### `process_diacritics(text: str, remove_diacritics: bool = False, **options) -> str`
Process diacritical marks.

**Parameters:**
- `text` (str): Input text with diacritical marks
- `remove_diacritics` (bool): Whether to remove diacritics (default: False)
- `**options`: Additional processing options

**Returns:**
- Text with diacritics processed

**Code Example:**
```python
from semantica.normalize import TextNormalizer

# Initialize text normalizer
normalizer = TextNormalizer()

# Normalize text with various options
text = "Hello   World!!!  This is a test."
normalized = normalizer.normalize_text(
    text,
    unicode_form="NFC",        # Unicode normalization
    case="lower",              # Convert to lowercase
    normalize_diacritics=True, # Normalize diacritics
    line_break_type="unix"     # Unix line breaks
)
print(f"Normalized: {normalized}")

# Clean text (remove HTML, special chars, etc.)
dirty_text = "<p>Hello &amp; World</p>"
cleaned = normalizer.clean_text(
    dirty_text,
    remove_html=True,
    remove_special_chars=True
)
print(f"Cleaned: {cleaned}")

# Standardize format
text = "This   has    extra    spaces"
standardized = normalizer.standardize_format(
    text,
    format_type="compact"  # Remove extra whitespace
)
print(f"Standardized: {standardized}")

# Process batch of texts
texts = ["Text 1", "Text 2", "Text 3"]
normalized_batch = normalizer.process_batch(
    texts,
    case="lower",
    unicode_form="NFC"
)

# Use Unicode normalizer directly
from semantica.normalize.text_normalizer import UnicodeNormalizer
unicode_norm = UnicodeNormalizer()
normalized_unicode = unicode_norm.normalize_unicode("café", form="NFC")

# Use whitespace normalizer
from semantica.normalize.text_normalizer import WhitespaceNormalizer
ws_norm = WhitespaceNormalizer()
normalized_ws = ws_norm.normalize_whitespace(
    "Text   with\t\t\t tabs",
    line_break_type="unix"
)
```

---

## Export Modules

### `semantica.export.json_exporter`

**What it does:**
This module provides comprehensive JSON and JSON-LD export capabilities for the Semantica framework, enabling structured data export for knowledge graphs and semantic information. It supports both standard JSON and JSON-LD formats with configurable indentation, encoding, metadata, and provenance tracking.

**Key Features:**
- JSON and JSON-LD format export
- Knowledge graph serialization
- Entity and relationship export
- Metadata and provenance tracking
- Configurable indentation and encoding
- JSON-LD context management

#### Class: `JSONExporter`

JSON exporter for knowledge graphs and semantic data.

**Methods:**

##### `__init__(indent: int = 2, ensure_ascii: bool = False, format: str = "json", config: Optional[Dict[str, Any]] = None, **kwargs)`
Initialize JSON exporter.

**Parameters:**
- `indent` (int): JSON indentation level (default: 2)
- `ensure_ascii` (bool): Whether to escape non-ASCII characters (default: False)
- `format` (str): Export format - 'json' or 'json-ld' (default: 'json')
- `config` (Optional[Dict[str, Any]]): Optional configuration dictionary
- `**kwargs`: Additional configuration options

##### `export(data: Any, file_path: Union[str, Path], format: Optional[str] = None, include_metadata: bool = True, include_provenance: bool = True, **options) -> None`
Export data to JSON file.

**Parameters:**
- `data` (Any): Data to export (dict, list, or any JSON-serializable value)
- `file_path` (Union[str, Path]): Output JSON file path
- `format` (Optional[str]): Export format - 'json' or 'json-ld' (default: self.format)
- `include_metadata` (bool): Whether to include metadata (default: True)
- `include_provenance` (bool): Whether to include provenance information (default: True)
- `**options`: Additional options passed to conversion methods

##### `export_knowledge_graph(knowledge_graph: Dict[str, Any], file_path: Union[str, Path], format: Optional[str] = None, **options) -> None`
Export knowledge graph to JSON or JSON-LD format.

**Parameters:**
- `knowledge_graph` (Dict[str, Any]): Knowledge graph dictionary containing entities, relationships, nodes, edges, metadata, statistics
- `file_path` (Union[str, Path]): Output JSON file path
- `format` (Optional[str]): Export format (default: self.format)
- `**options`: Additional options

##### `export_entities(entities: List[Dict[str, Any]], file_path: Union[str, Path], **options) -> None`
Export entities to JSON file.

**Parameters:**
- `entities` (List[Dict[str, Any]]): List of entity dictionaries to export
- `file_path` (Union[str, Path]): Output JSON file path
- `**options`: Additional options:
  - `metadata`: Additional metadata to include in export

##### `export_relationships(relationships: List[Dict[str, Any]], file_path: Union[str, Path], **options) -> None`
Export relationships to JSON.

**Parameters:**
- `relationships` (List[Dict[str, Any]]): List of relationship dictionaries
- `file_path` (Union[str, Path]): Output file path
- `**options`: Additional options

**Code Example:**
```python
from semantica.export import JSONExporter

# Initialize JSON exporter
exporter = JSONExporter(
    indent=2,              # Pretty print with 2 spaces
    ensure_ascii=False,    # Allow Unicode characters
    format="json-ld"       # Use JSON-LD format
)

# Export knowledge graph
knowledge_graph = {
    "entities": [
        {"id": "e1", "name": "Alice", "type": "Person"},
        {"id": "e2", "name": "Bob", "type": "Person"}
    ],
    "relationships": [
        {"source": "e1", "target": "e2", "type": "knows"}
    ],
    "metadata": {"created": "2024-01-15"}
}

exporter.export_knowledge_graph(
    knowledge_graph,
    "output.json",
    format="json-ld",
    include_metadata=True,
    include_provenance=True
)

# Export entities only
entities = [
    {"id": "e1", "name": "Alice", "type": "Person"},
    {"id": "e2", "name": "Bob", "type": "Person"}
]

exporter.export_entities(
    entities,
    "entities.json",
    metadata={"source": "manual_entry"}
)

# Export relationships
relationships = [
    {"source": "e1", "target": "e2", "type": "knows", "confidence": 0.9}
]

exporter.export_relationships(
    relationships,
    "relationships.json"
)

# Export any data structure
data = {"key": "value", "nested": {"a": 1, "b": 2}}
exporter.export(
    data,
    "data.json",
    format="json",
    include_metadata=True
)
```

---

## Vector Store Modules

### `semantica.vector_store.vector_store`

**What it does:**
This module provides the core vector storage, indexing, and retrieval operations for the Semantica framework. It includes vector storage, similarity search, indexing management, metadata association with vectors, and vector store maintenance capabilities. It supports multiple backends through adapters (FAISS, Pinecone, Weaviate, Qdrant, Milvus).

**Key Features:**
- Vector storage and management
- Similarity search and retrieval
- Vector indexing and optimization
- Metadata association with vectors
- Vector update and deletion operations
- Multi-backend support through adapters

#### Class: `VectorStore`

Vector store interface and management.

**Methods:**

##### `__init__(backend="faiss", config=None, **kwargs)`
Initialize vector store.

**Parameters:**
- `backend` (str): Backend name (default: "faiss")
- `config`: Configuration dictionary
- `**kwargs`: Additional configuration options

##### `store_vectors(vectors: List[np.ndarray], metadata: Optional[List[Dict[str, Any]]] = None, **options) -> List[str]`
Store vectors in vector store.

**Parameters:**
- `vectors` (List[np.ndarray]): List of vector arrays
- `metadata` (Optional[List[Dict[str, Any]]]): List of metadata dictionaries
- `**options`: Storage options

**Returns:**
- List of vector IDs

##### `search_vectors(query_vector: np.ndarray, k: int = 10, **options) -> List[Dict[str, Any]]`
Search for similar vectors.

**Parameters:**
- `query_vector` (np.ndarray): Query vector
- `k` (int): Number of results to return (default: 10)
- `**options`: Search options

**Returns:**
- List of search results with scores

##### `update_vectors(vector_ids: List[str], new_vectors: List[np.ndarray], **options) -> bool`
Update existing vectors.

**Parameters:**
- `vector_ids` (List[str]): List of vector IDs to update
- `new_vectors` (List[np.ndarray]): List of new vector arrays
- `**options`: Update options

**Returns:**
- True if successful

##### `delete_vectors(vector_ids: List[str], **options) -> bool`
Delete vectors from store.

**Parameters:**
- `vector_ids` (List[str]): List of vector IDs to delete
- `**options`: Delete options

**Returns:**
- True if successful

##### `get_vector(vector_id: str) -> Optional[np.ndarray]`
Get vector by ID.

**Parameters:**
- `vector_id` (str): Vector ID

**Returns:**
- Vector array or None if not found

##### `get_metadata(vector_id: str) -> Optional[Dict[str, Any]]`
Get metadata for vector.

**Parameters:**
- `vector_id` (str): Vector ID

**Returns:**
- Metadata dictionary or None if not found

#### Class: `VectorIndexer`

Vector indexing engine.

**Methods:**

##### `__init__(backend: str = "faiss", dimension: int = 768, **config)`
Initialize vector indexer.

**Parameters:**
- `backend` (str): Backend name (default: "faiss")
- `dimension` (int): Vector dimension (default: 768)
- `**config`: Configuration options

##### `create_index(vectors: List[np.ndarray], ids: Optional[List[str]] = None, **options) -> Any`
Create vector index.

**Parameters:**
- `vectors` (List[np.ndarray]): List of vectors
- `ids` (Optional[List[str]]): Vector IDs
- `**options`: Indexing options

**Returns:**
- Index object

##### `update_index(index: Any, new_vectors: List[np.ndarray], **options) -> Any`
Update existing index.

**Parameters:**
- `index` (Any): Existing index
- `new_vectors` (List[np.ndarray]): New vectors to add
- `**options`: Update options

**Returns:**
- Updated index object

##### `optimize_index(index: Any, **options) -> Any`
Optimize index for better performance.

**Parameters:**
- `index` (Any): Index to optimize
- `**options`: Optimization options

**Returns:**
- Optimized index object

#### Class: `VectorRetriever`

Vector retrieval engine.

**Methods:**

##### `__init__(backend: str = "faiss", **config)`
Initialize vector retriever.

**Parameters:**
- `backend` (str): Backend name (default: "faiss")
- `**config`: Configuration options

##### `search_similar(query_vector: np.ndarray, vectors: List[np.ndarray], ids: List[str], k: int = 10, **options) -> List[Dict[str, Any]]`
Search for similar vectors.

**Parameters:**
- `query_vector` (np.ndarray): Query vector
- `vectors` (List[np.ndarray]): List of vectors to search
- `ids` (List[str]): Vector IDs
- `k` (int): Number of results (default: 10)
- `**options`: Search options

**Returns:**
- List of results with scores

##### `search_by_metadata(metadata_filters: Dict[str, Any], vectors: List[np.ndarray], metadata: List[Dict[str, Any]], **options) -> List[Dict[str, Any]]`
Search vectors by metadata.

**Parameters:**
- `metadata_filters` (Dict[str, Any]): Metadata filter criteria
- `vectors` (List[np.ndarray]): List of vectors
- `metadata` (List[Dict[str, Any]]): List of metadata dictionaries
- `**options`: Search options

**Returns:**
- List of matching vectors with metadata

##### `search_hybrid(query_vector: np.ndarray, metadata_filters: Dict[str, Any], vectors: List[np.ndarray], metadata: List[Dict[str, Any]], **options) -> List[Dict[str, Any]]`
Perform hybrid search (metadata + similarity).

**Parameters:**
- `query_vector` (np.ndarray): Query vector
- `metadata_filters` (Dict[str, Any]): Metadata filter criteria
- `vectors` (List[np.ndarray]): List of vectors
- `metadata` (List[Dict[str, Any]]): List of metadata dictionaries
- `**options`: Search options

**Returns:**
- List of hybrid search results

#### Class: `VectorManager`

Vector store management engine.

**Methods:**

##### `__init__(**config)`
Initialize vector manager.

**Parameters:**
- `**config`: Configuration options

##### `manage_store(store: VectorStore, **operations: Dict[str, Any]) -> Dict[str, Any]`
Manage vector store operations.

**Parameters:**
- `store` (VectorStore): Vector store instance
- `**operations`: Dictionary of operations to perform

**Returns:**
- Dictionary of operation results

##### `maintain_store(store: VectorStore, **options: Dict[str, Any]) -> Dict[str, Any]`
Maintain vector store health.

**Parameters:**
- `store` (VectorStore): Vector store instance
- `**options`: Maintenance options

**Returns:**
- Dictionary with health status

##### `collect_statistics(store: VectorStore) -> Dict[str, Any]`
Collect vector store statistics.

**Parameters:**
- `store` (VectorStore): Vector store instance

**Returns:**
- Dictionary with statistics

**Code Example:**
```python
from semantica.vector_store import VectorStore
import numpy as np

# Initialize vector store
store = VectorStore(
    backend="faiss",
    config={"dimension": 768}
)

# Store vectors with metadata
vectors = [
    np.random.rand(768),
    np.random.rand(768),
    np.random.rand(768)
]

metadata = [
    {"text": "Document 1", "category": "tech"},
    {"text": "Document 2", "category": "science"},
    {"text": "Document 3", "category": "tech"}
]

vector_ids = store.store_vectors(vectors, metadata=metadata)
print(f"Stored {len(vector_ids)} vectors")

# Search for similar vectors
query_vector = np.random.rand(768)
results = store.search_vectors(
    query_vector,
    k=5,  # Top 5 results
    method="cosine"
)

for result in results:
    print(f"ID: {result['id']}, Score: {result['score']:.4f}")
    print(f"Metadata: {store.get_metadata(result['id'])}")

# Update vectors
new_vectors = [np.random.rand(768) for _ in range(2)]
store.update_vectors(vector_ids[:2], new_vectors)

# Get specific vector and metadata
vector = store.get_vector(vector_ids[0])
meta = store.get_metadata(vector_ids[0])

# Delete vectors
store.delete_vectors(vector_ids[2:])

# Use vector indexer directly
from semantica.vector_store.vector_store import VectorIndexer
indexer = VectorIndexer(backend="faiss", dimension=768)
index = indexer.create_index(vectors, vector_ids)

# Use vector retriever for advanced search
from semantica.vector_store.vector_store import VectorRetriever
retriever = VectorRetriever(backend="faiss")

# Hybrid search (metadata + similarity)
results = retriever.search_hybrid(
    query_vector,
    metadata_filters={"category": "tech"},
    vectors=vectors,
    metadata=metadata,
    k=10
)
```

---

## Reasoning Modules

### `semantica.reasoning.inference_engine`

**What it does:**
This module provides rule-based inference capabilities for knowledge graph reasoning and analysis. It supports forward chaining (data-driven), backward chaining (goal-driven), and bidirectional inference strategies. The module includes rule management, performance optimization, error handling, and custom rule support.

**Key Features:**
- Rule-based inference and reasoning
- Forward and backward chaining
- Bidirectional inference
- Rule management and execution
- Performance optimization
- Error handling and recovery
- Custom rule support

#### Class: `InferenceEngine`

Rule-based inference engine.

**Methods:**

##### `__init__(config: Optional[Dict[str, Any]] = None, **kwargs)`
Initialize inference engine.

**Parameters:**
- `config` (Optional[Dict[str, Any]]): Configuration dictionary
- `**kwargs`: Additional configuration options:
  - `strategy` (str): Inference strategy (forward, backward, bidirectional)
  - `max_iterations` (int): Maximum inference iterations (default: 100)

##### `add_rule(rule_definition: str, **options) -> Rule`
Add inference rule to engine.

**Parameters:**
- `rule_definition` (str): Rule definition string or Rule object
- `**options`: Additional options

**Returns:**
- Created rule

##### `add_fact(fact: Any) -> None`
Add fact to knowledge base.

**Parameters:**
- `fact` (Any): Fact to add

##### `add_facts(facts: List[Any]) -> None`
Add multiple facts.

**Parameters:**
- `facts` (List[Any]): List of facts

##### `forward_chain(facts: Optional[List[Any]] = None, **options) -> List[InferenceResult]`
Perform forward chaining inference.

**Parameters:**
- `facts` (Optional[List[Any]]): Optional initial facts
- `**options`: Additional options

**Returns:**
- List of inference results

##### `backward_chain(goal: Any, **options) -> Optional[InferenceResult]`
Perform backward chaining inference.

**Parameters:**
- `goal` (Any): Goal to prove
- `**options`: Additional options

**Returns:**
- Inference result or None

##### `infer(query: Any, **options) -> List[InferenceResult]`
Perform inference based on strategy.

**Parameters:**
- `query` (Any): Query or goal
- `**options`: Additional options

**Returns:**
- List of inference results

##### `get_facts() -> Set[Any]`
Get all facts.

**Returns:**
- Set of all facts

##### `get_inferred_facts() -> List[InferenceResult]`
Get all inferred facts.

**Returns:**
- List of inference results

##### `clear_facts() -> None`
Clear all facts.

##### `reset() -> None`
Reset inference engine.

#### Class: `InferenceResult`

Inference result dataclass.

**Attributes:**
- `conclusion` (Any): Inferred conclusion
- `premises` (List[Any]): Premises used for inference
- `rule_used` (Optional[Rule]): Rule used for inference
- `confidence` (float): Confidence score (default: 1.0)
- `metadata` (Dict[str, Any]): Additional metadata

#### Enum: `InferenceStrategy`

Inference strategies.

**Values:**
- `FORWARD`: Forward chaining
- `BACKWARD`: Backward chaining
- `BIDIRECTIONAL`: Bidirectional inference

**Code Example:**
```python
from semantica.reasoning import InferenceEngine, InferenceStrategy

# Initialize inference engine
engine = InferenceEngine(
    strategy="forward",      # or "backward", "bidirectional"
    max_iterations=100
)

# Add facts to knowledge base
engine.add_fact("Alice is a Person")
engine.add_fact("Bob is a Person")
engine.add_fact("Alice knows Bob")

# Define and add rules
rule1 = engine.add_rule(
    "IF Person(X) AND Person(Y) AND knows(X, Y) THEN friends(X, Y)",
    name="friendship_rule",
    confidence=0.9
)

rule2 = engine.add_rule(
    "IF friends(X, Y) THEN can_trust(X, Y)",
    name="trust_rule"
)

# Perform forward chaining (derive new facts from existing facts)
results = engine.forward_chain()
for result in results:
    print(f"Inferred: {result.conclusion}")
    print(f"  Using rule: {result.rule_used.name}")
    print(f"  Confidence: {result.confidence}")

# Perform backward chaining (prove a specific goal)
goal_result = engine.backward_chain("can_trust(Alice, Bob)")
if goal_result:
    print(f"Goal proven: {goal_result.conclusion}")
    print(f"Premises: {goal_result.premises}")

# Perform inference based on strategy
query_results = engine.infer(
    query="can_trust(Alice, Bob)",
    strategy=InferenceStrategy.BIDIRECTIONAL
)

# Get all facts (original + inferred)
all_facts = engine.get_facts()
print(f"Total facts: {len(all_facts)}")

# Get only inferred facts
inferred = engine.get_inferred_facts()
print(f"Inferred facts: {len(inferred)}")

# Clear facts and reset
engine.clear_facts()
engine.reset()
```

---

## Split Modules

### `semantica.split`

**What it does:**
This module provides comprehensive document chunking and splitting capabilities for optimal processing and semantic analysis. It enables efficient handling of large documents through various chunking strategies, each optimized for different use cases and document types.

**Key Features:**
- Semantic-based chunking using NLP (spaCy)
- Structure-aware chunking (headings, paragraphs, lists, code blocks)
- Sliding window chunking with configurable overlap
- Table-specific chunking
- Chunk validation and quality assessment
- Provenance tracking for data lineage

**Different Approaches and Strategies:**

The split module provides four main chunking strategies, each optimized for different scenarios:

1. **Semantic Chunking** - Uses NLP to split at semantic boundaries (sentences, paragraphs)
2. **Structural Chunking** - Respects document structure (headings, sections, lists)
3. **Sliding Window Chunking** - Fixed-size chunks with overlap for context preservation
4. **Table Chunking** - Specialized chunking for tabular data

**When to Use Each Approach:**

- **Semantic Chunking**: Natural language documents, preserving meaning and context
- **Structural Chunking**: Markdown, HTML, technical documentation with clear structure
- **Sliding Window**: Fixed-size requirements, embedding generation, vector stores
- **Table Chunking**: Spreadsheets, CSV data, structured tabular content

**Comparison of Chunking Strategies:**

| Strategy | Best For | Preserves | Speed | Quality |
|----------|----------|-----------|-------|---------|
| Semantic | Natural language | Sentence/paragraph boundaries | Medium | High |
| Structural | Markdown/HTML docs | Document hierarchy | Fast | High |
| Sliding Window | Fixed-size needs | Context via overlap | Fast | Medium |
| Table | Tabular data | Table structure | Fast | High |

#### Class: `SemanticChunker`

Semantic chunker for meaning-based splitting using NLP.

**Methods:**

##### `__init__(**config)`
Initialize semantic chunker.

**Parameters:**
- `model` (str): spaCy model name (default: "en_core_web_sm")
- `chunk_size` (int): Target chunk size in characters (default: 1000)
- `chunk_overlap` (int): Overlap between chunks in characters (default: 200)
- `language` (str): Language code (default: "en")

##### `chunk(text: str, **options) -> List[Chunk]`
Split text into semantic chunks.

**Parameters:**
- `text` (str): Input text to chunk
- `preserve_sentences` (bool): Preserve sentence boundaries (default: True)
- `preserve_paragraphs` (bool): Preserve paragraph boundaries (default: True)

**Returns:**
- List of Chunk objects with text, start_index, end_index, and metadata

##### `chunk_by_sentences(text: str, max_sentences: int = 5) -> List[Chunk]`
Chunk text by sentence boundaries.

**Parameters:**
- `text` (str): Input text
- `max_sentences` (int): Maximum sentences per chunk (default: 5)

**Returns:**
- List of chunks, each containing up to max_sentences

**Code Examples for Different Approaches:**

**Approach 1: Semantic Chunking (NLP-based, Recommended for Natural Language)**
```python
from semantica.split import SemanticChunker

# Initialize semantic chunker with spaCy model
chunker = SemanticChunker(
    model="en_core_web_sm",      # spaCy model for English
    chunk_size=1000,              # Target 1000 characters per chunk
    chunk_overlap=200,            # 200 character overlap between chunks
    language="en"
)

# Chunk long document (preserves sentence and paragraph boundaries)
long_text = """
This is a long document with multiple paragraphs. 
Each paragraph contains several sentences.

The semantic chunker uses NLP to identify natural boundaries.
It won't split sentences in the middle, preserving meaning.

This ensures that each chunk is semantically coherent.
"""

chunks = chunker.chunk(long_text, preserve_sentences=True, preserve_paragraphs=True)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.text)} chars")
    print(f"  Sentences: {chunk.metadata.get('sentence_count', 'N/A')}")
    print(f"  Text preview: {chunk.text[:100]}...")
    print()

# Chunk by sentences (fixed number of sentences per chunk)
sentence_chunks = chunker.chunk_by_sentences(long_text, max_sentences=3)
print(f"Created {len(sentence_chunks)} sentence-based chunks")
```

**Approach 2: Structural Chunking (Document Structure-aware)**
```python
from semantica.split import StructuralChunker

# Initialize structural chunker
struct_chunker = StructuralChunker(
    respect_headers=True,         # Respect heading hierarchy
    respect_sections=True,        # Respect section boundaries
    max_chunk_size=2000           # Maximum chunk size
)

# Chunk structured document (Markdown, HTML, etc.)
markdown_doc = """
# Main Title

## Section 1
This is the first section with multiple paragraphs.

### Subsection 1.1
Details about subsection 1.1.

## Section 2
Another section with content.

- List item 1
- List item 2
- List item 3
"""

chunks = struct_chunker.chunk(markdown_doc)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Elements: {chunk.metadata.get('element_count', 0)}")
    print(f"  Types: {chunk.metadata.get('element_types', [])}")
    print(f"  Structure preserved: {chunk.metadata.get('structure_preserved', False)}")
    print(f"  Preview: {chunk.text[:100]}...")
    print()
```

**Approach 3: Sliding Window Chunking (Fixed-size with Overlap)**
```python
from semantica.split import SlidingWindowChunker

# Initialize sliding window chunker
window_chunker = SlidingWindowChunker(
    chunk_size=512,               # Fixed chunk size (characters)
    overlap=100,                  # 100 character overlap
    stride=412                    # Stride = chunk_size - overlap
)

# Chunk text with fixed-size windows
text = "Your long document text here..." * 100

chunks = window_chunker.chunk(
    text,
    preserve_boundaries=True      # Try to preserve word/sentence boundaries
)

print(f"Created {len(chunks)} fixed-size chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.text)} chars (start: {chunk.start_index}, end: {chunk.end_index})")

# Use case: For embedding generation where fixed-size chunks are required
# The overlap ensures context is preserved across chunk boundaries
```

**Approach 4: Table Chunking (Tabular Data)**
```python
from semantica.split import TableChunker

# Initialize table chunker
table_chunker = TableChunker()

# Chunk table data (CSV, Excel, HTML tables)
table_data = """
Name,Age,City
Alice,30,New York
Bob,25,San Francisco
Charlie,35,Chicago
"""

table_chunks = table_chunker.chunk(table_data)

for chunk in table_chunks:
    print(f"Table chunk: {chunk.metadata.get('row_count', 0)} rows")
    print(f"Columns: {chunk.metadata.get('column_count', 0)}")
```

**Approach 5: Hybrid Chunking (Combine Strategies)**
```python
from semantica.split import SemanticChunker, StructuralChunker

# Use semantic chunking for natural language sections
semantic_chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)

# Use structural chunking for code/documentation sections
struct_chunker = StructuralChunker(max_chunk_size=2000)

# Process different sections with appropriate chunkers
document = {
    "introduction": "Natural language introduction text...",
    "code_section": "```python\ndef function():\n    pass\n```",
    "conclusion": "Natural language conclusion..."
}

# Chunk each section with appropriate strategy
all_chunks = []
all_chunks.extend(semantic_chunker.chunk(document["introduction"]))
all_chunks.extend(struct_chunker.chunk(document["code_section"]))
all_chunks.extend(semantic_chunker.chunk(document["conclusion"]))

print(f"Total chunks: {len(all_chunks)}")
```

**Approach 6: Chunk Validation and Quality Assessment**
```python
from semantica.split import SemanticChunker, ChunkValidator

# Create chunks
chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(long_text)

# Validate chunk quality
validator = ChunkValidator(
    min_chunk_size=100,           # Minimum chunk size
    max_chunk_size=2000,           # Maximum chunk size
    min_sentence_count=1,          # Minimum sentences per chunk
    require_completeness=True      # Require complete sentences
)

validation_results = []
for chunk in chunks:
    result = validator.validate(chunk)
    validation_results.append(result)
    
    if not result.is_valid:
        print(f"Invalid chunk: {result.issues}")
    else:
        print(f"Valid chunk: Quality score = {result.metrics.get('quality_score', 0):.2f}")

# Filter valid chunks
valid_chunks = [c for c, r in zip(chunks, validation_results) if r.is_valid]
print(f"Valid chunks: {len(valid_chunks)}/{len(chunks)}")
```

**Approach 7: Chunking with Provenance Tracking**
```python
from semantica.split import SemanticChunker, ProvenanceTracker

# Initialize chunker and provenance tracker
chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
tracker = ProvenanceTracker()

# Chunk document with provenance tracking
document_id = "doc_123"
source_file = "document.pdf"
chunks = chunker.chunk(text)

# Track provenance for each chunk
for i, chunk in enumerate(chunks):
    tracker.track_chunk(
        chunk_id=f"{document_id}_chunk_{i}",
        chunk=chunk,
        source={
            "document_id": document_id,
            "file_path": source_file,
            "page_number": 1,
            "section": "main"
        }
    )

# Retrieve provenance information
provenance = tracker.get_provenance(f"{document_id}_chunk_0")
print(f"Chunk source: {provenance['source']}")
print(f"Document lineage: {provenance['lineage']}")
```

**Best Practices:**

1. **Choose chunking strategy based on document type**:
   - Natural language → Semantic chunking
   - Structured documents → Structural chunking
   - Fixed-size requirements → Sliding window
   - Tables → Table chunking

2. **Set appropriate chunk sizes**:
   - Too small: Loses context, poor embeddings
   - Too large: Exceeds model limits, inefficient
   - Recommended: 500-2000 characters for most use cases

3. **Use overlap for context preservation**:
   - 10-20% overlap recommended for sliding window
   - Semantic chunking automatically preserves context

4. **Validate chunk quality**:
   - Check chunk sizes are within acceptable range
   - Ensure chunks contain complete sentences/paragraphs
   - Verify semantic coherence

5. **Track provenance**:
   - Maintain source information for each chunk
   - Enable traceability and debugging
   - Support data lineage requirements

---

## Conflict Resolution Modules

### `semantica.conflicts.conflict_resolver`

**What it does:**
This module provides comprehensive conflict resolution capabilities for resolving detected conflicts in knowledge graphs. It offers multiple resolution strategies including voting mechanisms, credibility-based resolution, recency-based resolution, and expert review workflows.

**Key Features:**
- Multiple resolution strategies (voting, credibility-weighted, recency, confidence)
- Automatic conflict resolution
- Manual and expert review workflows
- Resolution rule configuration
- Conflict resolution history tracking
- Source credibility weighting

**Different Conflict Resolution Strategies:**

The conflict resolver supports 7 different resolution strategies:

1. **Voting** - Most common value wins (democratic approach)
2. **Credibility Weighted** - Weight values by source credibility
3. **Most Recent** - Use the most recent value (temporal priority)
4. **First Seen** - Use the first encountered value (original priority)
5. **Highest Confidence** - Use value with highest confidence score
6. **Manual Review** - Flag for human review
7. **Expert Review** - Flag for domain expert review

**When to Use Each Strategy:**

- **Voting**: Multiple sources, democratic resolution, equal source credibility
- **Credibility Weighted**: Sources have different reliability, quality matters
- **Most Recent**: Temporal data, recent information preferred
- **First Seen**: Original data preferred, historical accuracy
- **Highest Confidence**: Confidence scores available, quality-based
- **Manual Review**: Complex conflicts, human judgment needed
- **Expert Review**: Domain-specific conflicts, expert knowledge required

**Comparison of Resolution Strategies:**

| Strategy | Automation | Quality | Speed | Best For |
|----------|------------|---------|-------|----------|
| Voting | Full | Medium | Fast | Multiple equal sources |
| Credibility Weighted | Full | High | Fast | Varying source quality |
| Most Recent | Full | Medium | Fast | Temporal data |
| First Seen | Full | Medium | Fast | Historical data |
| Highest Confidence | Full | High | Fast | Confidence scores available |
| Manual Review | None | Very High | Slow | Complex conflicts |
| Expert Review | None | Very High | Very Slow | Domain-specific |

**Code Examples for All Resolution Strategies:**

**Strategy 1: Voting (Most Common Value Wins)**
```python
from semantica.conflicts import ConflictResolver, Conflict

resolver = ConflictResolver(default_strategy="voting")

# Conflict with multiple conflicting values
conflict = Conflict(
    conflict_id="conflict_1",
    entity_id="entity_1",
    property_name="name",
    conflicting_values=["Apple Inc.", "Apple Inc.", "Apple", "Apple Corp"],
    sources=[...],
    conflict_type=ConflictType.VALUE_CONFLICT
)

# Resolve by voting (most common value wins)
result = resolver.resolve_conflict(conflict, strategy="voting")

print(f"Resolved value: {result.resolved_value}")  # "Apple Inc." (2 votes)
print(f"Confidence: {result.confidence:.2f}")  # 0.5 (2/4 votes)
print(f"Resolution: {result.resolution_notes}")
```

**Strategy 2: Credibility Weighted (Source Quality Matters)**
```python
from semantica.conflicts import ConflictResolver

# Initialize with credibility tracking
resolver = ConflictResolver(
    default_strategy="credibility_weighted"
)

# Set source credibility scores
resolver.source_tracker.set_source_credibility("official_database", 0.9)
resolver.source_tracker.set_source_credibility("user_input", 0.5)
resolver.source_tracker.set_source_credibility("web_scraping", 0.3)

# Conflict with values from different sources
conflict = Conflict(
    conflict_id="conflict_2",
    entity_id="entity_1",
    property_name="founded_year",
    conflicting_values=[1976, 1977, 1976],
    sources=[
        {"document": "official_database", "confidence": 0.9},
        {"document": "user_input", "confidence": 0.6},
        {"document": "web_scraping", "confidence": 0.4}
    ]
)

# Resolve by credibility-weighted voting
result = resolver.resolve_conflict(conflict, strategy="credibility_weighted")

# Official database value wins due to high credibility
print(f"Resolved value: {result.resolved_value}")  # 1976
print(f"Confidence: {result.confidence:.2f}")
```

**Strategy 3: Most Recent (Temporal Priority)**
```python
from semantica.conflicts import ConflictResolver
from datetime import datetime

resolver = ConflictResolver()

# Conflict with timestamps
conflict = Conflict(
    conflict_id="conflict_3",
    entity_id="entity_1",
    property_name="ceo",
    conflicting_values=["Tim Cook", "Steve Jobs"],
    sources=[
        {
            "document": "source_1",
            "metadata": {"timestamp": datetime(2020, 1, 1)}
        },
        {
            "document": "source_2",
            "metadata": {"timestamp": datetime(2023, 1, 1)}
        }
    ]
)

# Resolve by most recent value
result = resolver.resolve_conflict(conflict, strategy="most_recent")

print(f"Resolved value: {result.resolved_value}")  # "Tim Cook" (most recent)
print(f"Confidence: {result.confidence:.2f}")  # 0.8
```

**Strategy 4: First Seen (Original Priority)**
```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver()

# Conflict where first value is preferred
conflict = Conflict(
    conflict_id="conflict_4",
    entity_id="entity_1",
    property_name="original_name",
    conflicting_values=["Apple Computer", "Apple Inc.", "Apple"],
    sources=[...]
)

# Resolve by first seen (original value)
result = resolver.resolve_conflict(conflict, strategy="first_seen")

print(f"Resolved value: {result.resolved_value}")  # "Apple Computer" (first)
print(f"Confidence: {result.confidence:.2f}")  # 0.7
```

**Strategy 5: Highest Confidence (Quality-based)**
```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver()

# Conflict with confidence scores
conflict = Conflict(
    conflict_id="conflict_5",
    entity_id="entity_1",
    property_name="revenue",
    conflicting_values=[1000000, 1200000, 1100000],
    sources=[
        {"document": "source_1", "confidence": 0.6},
        {"document": "source_2", "confidence": 0.9},  # Highest
        {"document": "source_3", "confidence": 0.7}
    ]
)

# Resolve by highest confidence
result = resolver.resolve_conflict(conflict, strategy="highest_confidence")

print(f"Resolved value: {result.resolved_value}")  # 1200000 (highest confidence)
print(f"Confidence: {result.confidence:.2f}")  # 0.9
```

**Strategy 6: Manual Review (Human Judgment)**
```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver()

# Complex conflict requiring human judgment
conflict = Conflict(
    conflict_id="conflict_6",
    entity_id="entity_1",
    property_name="industry",
    conflicting_values=["Technology", "Consumer Electronics", "Software"],
    sources=[...],
    severity="high"
)

# Flag for manual review
result = resolver.resolve_conflict(conflict, strategy="manual_review")

print(f"Resolved: {result.resolved}")  # False
print(f"Requires review: {result.metadata.get('requires_manual_review')}")  # True
print(f"Notes: {result.resolution_notes}")  # "Flagged for manual review"

# Later, manually resolve
result.resolved = True
result.resolved_value = "Technology"
result.resolution_notes = "Manually resolved by domain expert"
```

**Strategy 7: Expert Review (Domain Expertise)**
```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver()

# Domain-specific conflict
conflict = Conflict(
    conflict_id="conflict_7",
    entity_id="entity_1",
    property_name="legal_classification",
    conflicting_values=["Corporation", "LLC", "Partnership"],
    sources=[...],
    severity="critical"
)

# Flag for expert review
result = resolver.resolve_conflict(conflict, strategy="expert_review")

print(f"Requires expert review: {result.metadata.get('requires_expert_review')}")  # True
```

**Strategy 8: Custom Resolution Rules (Property-specific)**
```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver()

# Set custom resolution rule for specific property
resolver.set_resolution_rule(
    entity_id="entity_1",
    property_name="name",
    strategy="voting"
)

resolver.set_resolution_rule(
    entity_id="entity_1",
    property_name="founded_year",
    strategy="most_recent"  # Years should use most recent
)

# Conflicts will automatically use appropriate strategy
conflict = Conflict(
    conflict_id="conflict_8",
    entity_id="entity_1",
    property_name="name",  # Will use VOTING
    conflicting_values=["Apple Inc.", "Apple"],
    sources=[...]
)

result = resolver.resolve_conflict(conflict)  # Uses VOTING automatically
```

**Strategy 9: Batch Conflict Resolution**
```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver(default_strategy="voting")

# Resolve multiple conflicts at once
conflicts = [
    Conflict(conflict_id="c1", entity_id="e1", property_name="name", ...),
    Conflict(conflict_id="c2", entity_id="e2", property_name="type", ...),
    Conflict(conflict_id="c3", entity_id="e3", property_name="location", ...)
]

# Resolve all conflicts
results = resolver.resolve_conflicts(conflicts)

# Get statistics
stats = resolver.get_resolution_statistics()
print(f"Total resolved: {stats['resolved_count']}/{stats['total_resolutions']}")
print(f"Resolution rate: {stats['resolution_rate']:.2%}")
print(f"By strategy: {stats['by_strategy']}")
```

**Best Practices:**

1. **Choose strategy based on conflict type**:
   - Value conflicts → Voting or Credibility Weighted
   - Temporal conflicts → Most Recent
   - Quality conflicts → Highest Confidence
   - Complex conflicts → Manual/Expert Review

2. **Set property-specific rules** - Different properties may need different strategies
3. **Track resolution history** - Monitor resolution patterns and success rates
4. **Use credibility weighting** - When source quality varies significantly
5. **Flag complex conflicts** - Don't auto-resolve everything, use human judgment when needed

---

## Deduplication Modules

### `semantica.deduplication`

**What it does:**
This module provides comprehensive duplicate detection and entity merging capabilities for knowledge graphs. It uses multiple similarity calculation methods from exact matching to advanced semantic similarity to identify and merge duplicate entities.

**Key Features:**
- Multiple similarity calculation methods (exact, fuzzy, semantic)
- Duplicate detection with configurable thresholds
- Entity merging with multiple strategies
- Batch and incremental duplicate detection
- Confidence scoring for duplicate candidates
- Group-based duplicate clustering

**Different Duplication Detection Methods:**

The deduplication module supports multiple similarity calculation methods:

1. **Exact String Matching** - Perfect string equality
2. **Levenshtein Distance** - Edit distance-based similarity
3. **Jaro-Winkler Similarity** - String similarity with prefix bonus
4. **Cosine Similarity** - Character n-gram based similarity
5. **Property Similarity** - Property value comparison
6. **Relationship Similarity** - Jaccard similarity of relationships
7. **Embedding Similarity** - Semantic similarity using vector embeddings
8. **Multi-factor Similarity** - Weighted combination of all methods

**When to Use Each Method:**

- **Exact Matching**: Identical strings, fast lookup, high precision
- **Levenshtein**: Typos, spelling variations, edit distance
- **Jaro-Winkler**: Names, addresses, prefix importance
- **Cosine (n-grams)**: General text similarity, character-level
- **Property Similarity**: Entity properties comparison
- **Relationship Similarity**: Graph structure comparison
- **Embedding Similarity**: Semantic meaning, best quality
- **Multi-factor**: Production use, comprehensive comparison

**Comparison of Similarity Methods:**

| Method | Accuracy | Speed | Use Case | Best For |
|--------|----------|-------|----------|----------|
| Exact Match | Perfect | Very Fast | Identical strings | Fast lookup |
| Levenshtein | High | Fast | Typos, variations | Spelling errors |
| Jaro-Winkler | High | Fast | Names, addresses | Prefix matching |
| Cosine (n-gram) | Medium | Fast | General text | Character similarity |
| Property Similarity | Medium | Medium | Entity properties | Structured data |
| Relationship Similarity | Medium | Medium | Graph structure | Network analysis |
| Embedding Similarity | Very High | Slow | Semantic meaning | Best quality |
| Multi-factor | Very High | Medium | Production | Comprehensive |

**Code Examples for All Similarity Methods:**

**Method 1: Exact String Matching**
```python
from semantica.deduplication import SimilarityCalculator

calculator = SimilarityCalculator()

# Exact matching (built into string similarity)
entity1 = {"name": "Apple Inc."}
entity2 = {"name": "Apple Inc."}

# Exact match returns 1.0
similarity = calculator.calculate_string_similarity(
    entity1["name"],
    entity2["name"],
    method="levenshtein"  # Will detect exact match first
)
print(f"Similarity: {similarity}")  # 1.0 (exact match)
```

**Method 2: Levenshtein Distance (Edit Distance)**
```python
from semantica.deduplication import SimilarityCalculator

calculator = SimilarityCalculator()

# Levenshtein handles typos and variations
entity1 = {"name": "Apple Inc."}
entity2 = {"name": "Apple Inc"}  # Missing period

similarity = calculator.calculate_string_similarity(
    entity1["name"],
    entity2["name"],
    method="levenshtein"
)
print(f"Levenshtein similarity: {similarity:.4f}")  # ~0.91

# Works well for typos
entity3 = {"name": "Aple Inc."}  # Typo
similarity2 = calculator.calculate_string_similarity(
    entity1["name"],
    entity3["name"],
    method="levenshtein"
)
print(f"With typo: {similarity2:.4f}")  # ~0.82
```

**Method 3: Jaro-Winkler Similarity (Prefix Bonus)**
```python
from semantica.deduplication import SimilarityCalculator

calculator = SimilarityCalculator()

# Jaro-Winkler gives bonus for matching prefixes
entity1 = {"name": "Apple Inc."}
entity2 = {"name": "Apple Corporation"}

similarity = calculator.calculate_string_similarity(
    entity1["name"],
    entity2["name"],
    method="jaro_winkler"
)
print(f"Jaro-Winkler similarity: {similarity:.4f}")  # Higher than Levenshtein

# Great for names and addresses where prefix matters
name1 = "John Smith"
name2 = "John Smyth"  # Different suffix
similarity2 = calculator.calculate_string_similarity(
    name1, name2, method="jaro_winkler"
)
print(f"Name similarity: {similarity2:.4f}")  # High due to prefix match
```

**Method 4: Cosine Similarity (Character N-grams)**
```python
from semantica.deduplication import SimilarityCalculator

calculator = SimilarityCalculator()

# Cosine similarity using character bigrams
entity1 = {"name": "Apple Inc."}
entity2 = {"name": "Apple Corporation"}

similarity = calculator.calculate_string_similarity(
    entity1["name"],
    entity2["name"],
    method="cosine"
)
print(f"Cosine similarity: {similarity:.4f}")  # Based on character bigrams
```

**Method 5: Property Similarity (Entity Properties)**
```python
from semantica.deduplication import SimilarityCalculator

calculator = SimilarityCalculator()

# Compare entities based on properties
entity1 = {
    "name": "Apple Inc.",
    "properties": {
        "founded": 1976,
        "location": "Cupertino",
        "industry": "Technology"
    }
}

entity2 = {
    "name": "Apple",
    "properties": {
        "founded": 1976,
        "location": "Cupertino",
        "industry": "Tech"
    }
}

# Property similarity compares matching properties
property_sim = calculator.calculate_property_similarity(entity1, entity2)
print(f"Property similarity: {property_sim:.4f}")  # High (founded, location match)
```

**Method 6: Relationship Similarity (Jaccard Similarity)**
```python
from semantica.deduplication import SimilarityCalculator

calculator = SimilarityCalculator()

# Compare entities based on relationships
entity1 = {
    "name": "Apple Inc.",
    "relationships": [
        {"type": "founded_by", "target": "Steve Jobs"},
        {"type": "located_in", "target": "Cupertino"}
    ]
}

entity2 = {
    "name": "Apple",
    "relationships": [
        {"type": "founded_by", "target": "Steve Jobs"},
        {"type": "located_in", "target": "California"}
    ]
}

# Relationship similarity (Jaccard)
rel_sim = calculator.calculate_relationship_similarity(entity1, entity2)
print(f"Relationship similarity: {rel_sim:.4f}")  # 0.33 (1 common / 3 total)
```

**Method 7: Embedding Similarity (Semantic)**
```python
from semantica.deduplication import SimilarityCalculator
import numpy as np

calculator = SimilarityCalculator()

# Semantic similarity using embeddings
entity1 = {
    "name": "Apple Inc.",
    "embedding": np.random.rand(384)  # Embedding vector
}

entity2 = {
    "name": "Apple Corporation",
    "embedding": np.random.rand(384)  # Similar embedding
}

# Embedding similarity (cosine similarity of vectors)
embedding_sim = calculator.calculate_embedding_similarity(
    entity1["embedding"].tolist(),
    entity2["embedding"].tolist()
)
print(f"Embedding similarity: {embedding_sim:.4f}")  # Semantic similarity
```

**Method 8: Multi-factor Similarity (Weighted Combination)**
```python
from semantica.deduplication import SimilarityCalculator

# Configure weights for different factors
calculator = SimilarityCalculator(
    string_weight=0.3,          # 30% weight for string similarity
    property_weight=0.2,         # 20% weight for properties
    relationship_weight=0.1,    # 10% weight for relationships
    embedding_weight=0.4         # 40% weight for embeddings (highest)
)

entity1 = {
    "name": "Apple Inc.",
    "properties": {"founded": 1976},
    "relationships": [{"type": "founded_by", "target": "Steve Jobs"}],
    "embedding": [...]  # Embedding vector
}

entity2 = {
    "name": "Apple",
    "properties": {"founded": 1976},
    "relationships": [{"type": "founded_by", "target": "Steve Jobs"}],
    "embedding": [...]  # Similar embedding
}

# Multi-factor similarity (combines all methods)
result = calculator.calculate_similarity(entity1, entity2)

print(f"Overall similarity: {result.score:.4f}")
print(f"Components: {result.components}")
# {
#     "string": 0.85,
#     "property": 1.0,
#     "relationship": 1.0,
#     "embedding": 0.92
# }
print(f"Weights: {result.metadata['weights']}")
```

**Method 9: Duplicate Detection with Thresholds**
```python
from semantica.deduplication import DuplicateDetector

# Initialize with similarity threshold
detector = DuplicateDetector(
    similarity_threshold=0.8,    # Entities with >= 0.8 similarity are duplicates
    confidence_threshold=0.7     # Minimum confidence for duplicate candidates
)

entities = [
    {"id": "1", "name": "Apple Inc."},
    {"id": "2", "name": "Apple"},
    {"id": "3", "name": "Microsoft"},
    {"id": "4", "name": "Apple Corporation"}
]

# Detect duplicates
candidates = detector.detect_duplicates(entities, threshold=0.8)

for candidate in candidates:
    print(f"Duplicate pair:")
    print(f"  {candidate.entity1['name']} <-> {candidate.entity2['name']}")
    print(f"  Similarity: {candidate.similarity_score:.4f}")
    print(f"  Confidence: {candidate.confidence:.4f}")
    print(f"  Reasons: {candidate.reasons}")
```

**Method 10: Duplicate Group Detection (Clustering)**
```python
from semantica.deduplication import DuplicateDetector

detector = DuplicateDetector(similarity_threshold=0.7)

entities = [
    {"id": "1", "name": "Apple Inc."},
    {"id": "2", "name": "Apple"},
    {"id": "3", "name": "Apple Corp"},
    {"id": "4", "name": "Microsoft"}
]

# Detect duplicate groups (clusters)
groups = detector.detect_duplicate_groups(entities, threshold=0.7)

for group in groups:
    print(f"Duplicate group: {len(group.entities)} entities")
    print(f"  Confidence: {group.confidence:.4f}")
    print(f"  Representative: {group.representative['name']}")
    for entity in group.entities:
        print(f"    - {entity['name']}")
```

**Method 11: Incremental Duplicate Detection**
```python
from semantica.deduplication import DuplicateDetector

detector = DuplicateDetector(similarity_threshold=0.8)

# Existing entities in knowledge graph
existing_entities = [
    {"id": "1", "name": "Apple Inc."},
    {"id": "2", "name": "Microsoft"}
]

# New entities to add
new_entities = [
    {"id": "3", "name": "Apple"},
    {"id": "4", "name": "Google"}
]

# Incremental detection (only compare new vs existing)
candidates = detector.incremental_detect(
    new_entities,
    existing_entities,
    threshold=0.8
)

# More efficient than full duplicate detection
for candidate in candidates:
    print(f"New entity '{candidate.entity1['name']}' "
          f"duplicates existing '{candidate.entity2['name']}'")
```

**Best Practices:**

1. **Choose similarity method based on data type**:
   - Names/Addresses → Jaro-Winkler
   - General text → Levenshtein or Cosine
   - Semantic meaning → Embedding similarity
   - Production → Multi-factor

2. **Set appropriate thresholds**:
   - Too low: False positives (non-duplicates marked as duplicates)
   - Too high: False negatives (duplicates missed)
   - Recommended: 0.7-0.8 for most use cases

3. **Use multi-factor similarity** - Combines multiple signals for better accuracy
4. **Use incremental detection** - More efficient for streaming/updates
5. **Review duplicate groups** - Validate before merging
6. **Track confidence scores** - Use for quality control

---

## Additional Modules

Due to the extensive number of modules (234 Python files), the above covers the core and most commonly used modules. The remaining modules follow similar patterns:

- **Normalization Modules**: Data cleaning, text normalization, entity normalization, date/number normalization
- **Semantic Extraction Modules**: NER, relation extraction, triple extraction, event detection
- **Reasoning Modules**: Inference engines, rule managers, deductive/abductive reasoning
- **Vector Store Modules**: Adapters for FAISS, Pinecone, Weaviate, Qdrant, Milvus
- **Triple Store Modules**: Adapters for Jena, Virtuoso, Blazegraph, RDF4J
- **Export Modules**: Exporters for JSON, CSV, RDF, OWL, YAML formats
- **Visualization Modules**: Graph visualizers, ontology visualizers, quality visualizers
- **Quality Assurance Modules**: KG quality assessment, validation engines, automated fixes
- **Context Modules**: Context retrieval, entity linking, agent memory
- **Deduplication Modules**: Duplicate detection, entity merging, similarity calculation
- **Conflict Modules**: Conflict detection, resolution, analysis
- **Split Modules**: Text chunking (semantic, structural, sliding window)
- **Ontology Modules**: Ontology generation, validation, versioning
- **Utils Modules**: Helper functions, validators, constants, exceptions

Each module typically follows this structure:
- `__init__()`: Initialize with optional config
- Main processing methods: Process data with various options
- Utility methods: Helper functions for specific operations
- Configuration methods: Get/set configuration

For detailed documentation of specific modules not covered above, please refer to the individual module files or request specific module documentation.

---

## Summary

This framework provides a comprehensive semantic processing pipeline with:

- **26 main modules** covering all aspects of semantic data processing
- **234 Python files** with detailed implementations
- **Modular architecture** allowing flexible composition
- **Extensive configuration** options for customization
- **Multiple adapters** for various storage and processing backends
- **Quality assurance** and validation throughout
- **Temporal support** for time-aware knowledge graphs
- **Multi-modal processing** for text, images, and audio

All modules are designed with consistent interfaces, comprehensive error handling, and extensive logging capabilities.

