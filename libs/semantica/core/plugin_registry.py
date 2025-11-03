"""
Plugin Registry Module

Manages dynamic plugin loading, version compatibility, and dependency resolution.

Key Features:
    - Dynamic plugin discovery
    - Plugin version management
    - Dependency resolution
    - Plugin lifecycle management
    - Plugin isolation

Main Classes:
    - PluginRegistry: Plugin management system
"""

import importlib
import importlib.util
import inspect
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field

from ..utils.exceptions import ConfigurationError, ValidationError
from ..utils.logging import get_logger


@dataclass
class PluginInfo:
    """Plugin information."""
    
    name: str
    version: str
    plugin_class: Type
    description: str = ""
    author: str = ""
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedPlugin:
    """Loaded plugin instance."""
    
    info: PluginInfo
    instance: Any
    config: Dict[str, Any] = field(default_factory=dict)
    loaded_at: Optional[float] = None


class PluginRegistry:
    """
    Plugin registry and management system.
    
    Handles registration, loading, and management of plugins.
    
    Attributes:
        plugins: Dictionary of registered plugins
        loaded_plugins: Dictionary of loaded plugin instances
        plugin_paths: List of paths to search for plugins
    """
    
    def __init__(self, plugin_paths: Optional[List[Path]] = None):
        """
        Initialize plugin registry.
        
        Args:
            plugin_paths: List of paths to search for plugins
        """
        self.logger = get_logger("plugin_registry")
        self.plugins: Dict[str, PluginInfo] = {}
        self.loaded_plugins: Dict[str, LoadedPlugin] = {}
        self.plugin_paths = plugin_paths or []
        self._discovered_plugins: Dict[str, Path] = {}
        
        # Discover available plugins
        self._discover_plugins()
    
    def register_plugin(
        self,
        plugin_name: str,
        plugin_class: Type,
        version: str = "1.0.0",
        **metadata: Any
    ) -> None:
        """
        Register a plugin.
        
        Args:
            plugin_name: Name of the plugin
            plugin_class: Plugin class to register
            version: Plugin version
            **metadata: Additional plugin metadata:
                - description: Plugin description
                - author: Plugin author
                - dependencies: List of dependency plugin names
                - capabilities: List of plugin capabilities
                
        Raises:
            ValidationError: If plugin is invalid
        """
        try:
            # Validate plugin class
            if not inspect.isclass(plugin_class):
                raise ValidationError(f"Plugin {plugin_name} must be a class, got {type(plugin_class)}")
            
            # Check for required methods
            required_methods = ["initialize", "execute"]
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    raise ValidationError(
                        f"Plugin {plugin_name} must have {method}() method"
                    )
            
            # Create plugin info
            plugin_info = PluginInfo(
                name=plugin_name,
                version=version,
                plugin_class=plugin_class,
                description=metadata.get("description", ""),
                author=metadata.get("author", ""),
                dependencies=metadata.get("dependencies", []),
                capabilities=metadata.get("capabilities", []),
                metadata=metadata
            )
            
            # Validate dependencies
            self._validate_dependencies(plugin_info)
            
            # Register plugin
            self.plugins[plugin_name] = plugin_info
            self.logger.info(f"Registered plugin: {plugin_name} v{version}")
            
        except Exception as e:
            self.logger.error(f"Failed to register plugin {plugin_name}: {e}")
            raise
    
    def load_plugin(
        self,
        plugin_name: str,
        **config: Any
    ) -> Any:
        """
        Load and initialize a plugin.
        
        Args:
            plugin_name: Name of plugin to load
            **config: Plugin configuration
            
        Returns:
            Loaded and initialized plugin instance
            
        Raises:
            ConfigurationError: If plugin not found or load fails
        """
        try:
            # Check if already loaded
            if plugin_name in self.loaded_plugins:
                self.logger.debug(f"Plugin {plugin_name} already loaded")
                return self.loaded_plugins[plugin_name].instance
            
            # Check if plugin registered
            if plugin_name not in self.plugins:
                # Try to discover and register
                self._discover_plugin(plugin_name)
                
                if plugin_name not in self.plugins:
                    raise ConfigurationError(f"Plugin {plugin_name} not found")
            
            plugin_info = self.plugins[plugin_name]
            
            # Load dependencies first
            for dep_name in plugin_info.dependencies:
                if dep_name not in self.loaded_plugins:
                    self.logger.debug(f"Loading dependency: {dep_name}")
                    self.load_plugin(dep_name)
            
            # Create plugin instance
            plugin_class = plugin_info.plugin_class
            
            try:
                plugin_instance = plugin_class(**config)
            except TypeError as e:
                # If initialization fails, try without config
                self.logger.warning(f"Failed to initialize with config, trying without: {e}")
                plugin_instance = plugin_class()
            
            # Initialize plugin
            if hasattr(plugin_instance, "initialize"):
                plugin_instance.initialize()
            
            # Store loaded plugin
            loaded_plugin = LoadedPlugin(
                info=plugin_info,
                instance=plugin_instance,
                config=config,
                loaded_at=time.time()
            )
            
            self.loaded_plugins[plugin_name] = loaded_plugin
            
            self.logger.info(f"Loaded plugin: {plugin_name}")
            
            return plugin_instance
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin {plugin_name}: {e}")
            raise ConfigurationError(f"Failed to load plugin {plugin_name}: {e}")
    
    def unload_plugin(self, plugin_name: str) -> None:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Raises:
            ConfigurationError: If plugin not loaded
        """
        if plugin_name not in self.loaded_plugins:
            raise ConfigurationError(f"Plugin {plugin_name} is not loaded")
        
        try:
            plugin_instance = self.loaded_plugins[plugin_name].instance
            
            # Call plugin cleanup if available
            if hasattr(plugin_instance, "cleanup"):
                plugin_instance.cleanup()
            elif hasattr(plugin_instance, "close"):
                plugin_instance.close()
            
            # Remove from loaded plugins
            del self.loaded_plugins[plugin_name]
            
            self.logger.info(f"Unloaded plugin: {plugin_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            raise ConfigurationError(f"Failed to unload plugin {plugin_name}: {e}")
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """
        List all available plugins.
        
        Returns:
            List of plugin information dictionaries
        """
        plugins = []
        
        for name, plugin_info in self.plugins.items():
            plugin_data = {
                "name": plugin_info.name,
                "version": plugin_info.version,
                "description": plugin_info.description,
                "author": plugin_info.author,
                "dependencies": plugin_info.dependencies,
                "capabilities": plugin_info.capabilities,
                "loaded": name in self.loaded_plugins,
                "metadata": plugin_info.metadata
            }
            
            if name in self.loaded_plugins:
                loaded_plugin = self.loaded_plugins[name]
                plugin_data["config"] = loaded_plugin.config
                plugin_data["loaded_at"] = loaded_plugin.loaded_at
            
            plugins.append(plugin_data)
        
        return plugins
    
    def get_plugin_info(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get information about a plugin.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Dictionary with plugin information
            
        Raises:
            ConfigurationError: If plugin not found
        """
        if plugin_name not in self.plugins:
            raise ConfigurationError(f"Plugin {plugin_name} not found")
        
        plugin_info = self.plugins[plugin_name]
        
        info = {
            "name": plugin_info.name,
            "version": plugin_info.version,
            "description": plugin_info.description,
            "author": plugin_info.author,
            "dependencies": plugin_info.dependencies,
            "capabilities": plugin_info.capabilities,
            "metadata": plugin_info.metadata,
            "loaded": plugin_name in self.loaded_plugins
        }
        
        if plugin_name in self.loaded_plugins:
            loaded_plugin = self.loaded_plugins[plugin_name]
            info["config"] = loaded_plugin.config
            info["loaded_at"] = loaded_plugin.loaded_at
        
        return info
    
    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """
        Check if a plugin is loaded.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            True if plugin is loaded, False otherwise
        """
        return plugin_name in self.loaded_plugins
    
    def get_loaded_plugin(self, plugin_name: str) -> Optional[Any]:
        """
        Get loaded plugin instance.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            Plugin instance or None if not loaded
        """
        if plugin_name in self.loaded_plugins:
            return self.loaded_plugins[plugin_name].instance
        return None
    
    def _discover_plugins(self) -> None:
        """Discover available plugins from plugin paths."""
        for plugin_path in self.plugin_paths:
            if isinstance(plugin_path, str):
                plugin_path = Path(plugin_path)
            
            if plugin_path.exists() and plugin_path.is_dir():
                self._scan_directory(plugin_path)
    
    def _discover_plugin(self, plugin_name: str) -> None:
        """
        Discover a specific plugin.
        
        Args:
            plugin_name: Name of plugin to discover
        """
        for plugin_path in self.plugin_paths:
            if isinstance(plugin_path, str):
                plugin_path = Path(plugin_path)
            
            # Try to find plugin module
            plugin_module_path = plugin_path / f"{plugin_name}.py"
            if plugin_module_path.exists():
                try:
                    self._load_plugin_from_file(plugin_module_path, plugin_name)
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load plugin from {plugin_module_path}: {e}")
    
    def _scan_directory(self, directory: Path) -> None:
        """Scan directory for plugin files."""
        for file_path in directory.glob("*.py"):
            if file_path.name == "__init__.py":
                continue
            
            plugin_name = file_path.stem
            try:
                self._load_plugin_from_file(file_path, plugin_name)
            except Exception as e:
                self.logger.debug(f"Failed to load plugin from {file_path}: {e}")
    
    def _load_plugin_from_file(self, file_path: Path, plugin_name: str) -> None:
        """Load plugin from file."""
        try:
            # Import module
            spec = importlib.util.spec_from_file_location(plugin_name, file_path)
            if spec is None or spec.loader is None:
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class (typically named after plugin or has Plugin suffix)
            plugin_class = None
            
            # Try common naming patterns
            class_names = [
                plugin_name.capitalize(),
                plugin_name.capitalize() + "Plugin",
                plugin_name.replace("_", "").capitalize(),
            ]
            
            for class_name in class_names:
                if hasattr(module, class_name):
                    plugin_class = getattr(module, class_name)
                    if inspect.isclass(plugin_class):
                        break
            
            # If not found, look for any class with Plugin in name
            if plugin_class is None:
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and "Plugin" in name:
                        plugin_class = obj
                        break
            
            if plugin_class is None:
                self.logger.warning(f"No plugin class found in {file_path}")
                return
            
            # Get metadata from module if available
            metadata = {
                "description": getattr(module, "__doc__", "") or getattr(module, "description", ""),
                "author": getattr(module, "author", ""),
                "version": getattr(module, "version", "1.0.0"),
                "dependencies": getattr(module, "dependencies", []),
                "capabilities": getattr(module, "capabilities", []),
            }
            
            # Register plugin
            self.register_plugin(
                plugin_name=plugin_name,
                plugin_class=plugin_class,
                **metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {file_path}: {e}")
            raise
    
    def _validate_dependencies(self, plugin_info: PluginInfo) -> None:
        """
        Validate plugin dependencies.
        
        Args:
            plugin_info: Plugin information
            
        Raises:
            ValidationError: If dependencies are invalid
        """
        for dep_name in plugin_info.dependencies:
            if dep_name not in self.plugins:
                # Try to discover dependency
                self._discover_plugin(dep_name)
                
                if dep_name not in self.plugins:
                    raise ValidationError(
                        f"Plugin {plugin_info.name} depends on {dep_name}, "
                        f"but {dep_name} is not available"
                    )