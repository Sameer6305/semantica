"""
Configuration Management Module

Handles all configuration loading, validation, and management for the framework.

Key Features:
    - YAML/JSON configuration parsing
    - Environment variable support
    - Configuration validation
    - Dynamic configuration updates
    - Configuration inheritance

Main Classes:
    - Config: Configuration data class
    - ConfigManager: Configuration management system
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from ..utils.exceptions import ConfigurationError
from ..utils.helpers import (
    merge_dicts,
    read_json_file,
    get_nested_value,
    set_nested_value,
)
from ..utils.validators import validate_config
from ..utils.constants import DEFAULT_CONFIG


class Config:
    """
    Configuration data class.
    
    Stores all framework configuration settings with validation
    and type checking.
    
    Attributes:
        llm_provider: LLM provider configuration
        embedding_model: Embedding model settings
        vector_store: Vector store configuration
        graph_db: Graph database settings
        processing: Processing pipeline settings
        logging: Logging configuration
        quality: Quality assurance settings
        security: Security settings
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_dict: Dictionary of configuration values
            **kwargs: Additional configuration parameters
        """
        # Start with defaults
        default_dict = DEFAULT_CONFIG.copy()
        
        # Merge with provided config_dict
        if config_dict:
            default_dict = merge_dicts(default_dict, config_dict, deep=True)
        
        # Merge with kwargs
        if kwargs:
            default_dict = merge_dicts(default_dict, kwargs, deep=True)
        
        # Load from environment variables
        self._load_from_env(default_dict)
        
        # Initialize dataclass fields
        self.llm_provider = default_dict.get("llm_provider", {})
        self.embedding_model = default_dict.get("embedding_model", {})
        self.vector_store = default_dict.get("vector_store", {})
        self.graph_db = default_dict.get("graph_db", {})
        self.processing = default_dict.get("processing", DEFAULT_CONFIG.get("processing", {}))
        self.pipeline = default_dict.get("pipeline", {})
        self.logging = default_dict.get("logging", DEFAULT_CONFIG.get("logging", {}))
        self.quality = default_dict.get("quality", DEFAULT_CONFIG.get("quality", {}))
        self.security = default_dict.get("security", DEFAULT_CONFIG.get("security", {}))
        self.custom = default_dict.get("custom", {})
    
    def _load_from_env(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration values from environment variables.
        
        Environment variables with prefix SEMANTICA_ will override
        configuration values. Format: SEMANTICA_SECTION_KEY=value
        
        Args:
            config_dict: Configuration dictionary to update
        """
        prefix = "SEMANTICA_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                
                # Try to parse as JSON, otherwise use as string
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    # Convert boolean strings
                    if value.lower() in ("true", "false"):
                        parsed_value = value.lower() == "true"
                    # Convert numeric strings
                    elif value.isdigit():
                        parsed_value = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        parsed_value = float(value)
                    else:
                        parsed_value = value
                
                # Set nested value
                set_nested_value(config_dict, config_key, parsed_value)
    
    def validate(self) -> None:
        """
        Validate configuration settings.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors = []
        
        # Validate processing settings
        if "processing" in self.to_dict():
            processing = self.processing
            if "batch_size" in processing:
                batch_size = processing["batch_size"]
                if not isinstance(batch_size, int) or batch_size <= 0:
                    errors.append("processing.batch_size must be a positive integer")
            
            if "max_workers" in processing:
                max_workers = processing["max_workers"]
                if not isinstance(max_workers, int) or max_workers <= 0:
                    errors.append("processing.max_workers must be a positive integer")
        
        # Validate quality settings
        if "quality" in self.to_dict():
            quality = self.quality
            if "min_confidence" in quality:
                confidence = quality["min_confidence"]
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    errors.append("quality.min_confidence must be between 0.0 and 1.0")
        
        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {', '.join(errors)}",
                config_context=self.to_dict(),
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            dict: Configuration as dictionary
        """
        return {
            "llm_provider": self.llm_provider,
            "embedding_model": self.embedding_model,
            "vector_store": self.vector_store,
            "graph_db": self.graph_db,
            "processing": self.processing,
            "pipeline": self.pipeline,
            "logging": self.logging,
            "quality": self.quality,
            "security": self.security,
            "custom": self.custom,
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get nested configuration value by key path.
        
        Args:
            key_path: Dot-separated key path (e.g., "processing.batch_size")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config_dict = self.to_dict()
        return get_nested_value(config_dict, key_path, default=default)
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set nested configuration value by key path.
        
        Args:
            key_path: Dot-separated key path (e.g., "processing.batch_size")
            value: Value to set
        """
        config_dict = self.to_dict()
        set_nested_value(config_dict, key_path, value)
        
        # Reinitialize from updated dict
        updated = Config(config_dict=config_dict)
        self.__dict__.update(updated.__dict__)
    
    def update(self, updates: Dict[str, Any], merge: bool = True) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
            merge: Whether to merge nested dictionaries (default: True)
        """
        current_dict = self.to_dict()
        
        if merge:
            updated_dict = merge_dicts(current_dict, updates, deep=True)
        else:
            updated_dict = {**current_dict, **updates}
        
        # Reinitialize from updated dict
        updated = Config(config_dict=updated_dict)
        self.__dict__.update(updated.__dict__)


class ConfigManager:
    """
    Configuration management system.
    
    Handles loading, validation, and updating of configuration.
    """
    
    def __init__(self):
        """Initialize configuration manager."""
        self._config: Optional[Config] = None
    
    def load_from_file(
        self,
        file_path: Union[str, Path],
        validate: bool = True
    ) -> Config:
        """
        Load configuration from file.
        
        Supports YAML and JSON formats. Automatically detects format
        based on file extension.
        
        Args:
            file_path: Path to configuration file (YAML or JSON)
            validate: Whether to validate configuration after loading
            
        Returns:
            Config: Loaded configuration object
            
        Raises:
            ConfigurationError: If file cannot be loaded or is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                config_context={"file_path": str(file_path)}
            )
        
        # Detect format from extension
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in (".yaml", ".yml"):
                with open(file_path, "r", encoding="utf-8") as f:
                    config_dict = yaml.safe_load(f)
                    
            elif suffix == ".json":
                config_dict = read_json_file(file_path)
            else:
                raise ConfigurationError(
                    f"Unsupported configuration file format: {suffix}. "
                    "Supported formats: .yaml, .yml, .json"
                )
            
            # Create config object
            config = Config(config_dict=config_dict)
            
            # Validate if requested
            if validate:
                config.validate()
            
            self._config = config
            return config
            
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to load configuration file: {str(e)}",
                config_context={"file_path": str(file_path)}
            )
    
    def load_from_dict(
        self,
        config_dict: Dict[str, Any],
        validate: bool = True
    ) -> Config:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Dictionary of configuration values
            validate: Whether to validate configuration after loading
            
        Returns:
            Config: Configuration object
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        config = Config(config_dict=config_dict)
        
        if validate:
            config.validate()
        
        self._config = config
        return config
    
    def merge_configs(self, *configs: Config, validate: bool = True) -> Config:
        """
        Merge multiple configurations.
        
        Later configurations take priority over earlier ones.
        Nested dictionaries are merged deeply.
        
        Args:
            *configs: Configuration objects to merge
            validate: Whether to validate merged configuration
            
        Returns:
            Config: Merged configuration
        """
        if not configs:
            raise ConfigurationError("No configurations provided to merge")
        
        # Convert all configs to dicts
        config_dicts = [config.to_dict() for config in configs]
        
        # Merge all dicts
        merged_dict = {}
        for config_dict in config_dicts:
            merged_dict = merge_dicts(merged_dict, config_dict, deep=True)
        
        # Create merged config
        merged_config = Config(config_dict=merged_dict)
        
        if validate:
            merged_config.validate()
        
        self._config = merged_config
        return merged_config
    
    def get_config(self) -> Optional[Config]:
        """
        Get current configuration.
        
        Returns:
            Current Config object or None if not loaded
        """
        return self._config
    
    def set_config(self, config: Config, validate: bool = True) -> None:
        """
        Set current configuration.
        
        Args:
            config: Configuration object to set
            validate: Whether to validate configuration
        """
        if validate:
            config.validate()
        
        self._config = config
    
    def reload(self, file_path: Optional[Union[str, Path]] = None) -> Config:
        """
        Reload configuration from file.
        
        Args:
            file_path: Path to configuration file. If None, uses last loaded file.
            
        Returns:
            Config: Reloaded configuration
        """
        if file_path is None:
            if not hasattr(self, "_last_file_path"):
                raise ConfigurationError("No file path specified and no previous file loaded")
            file_path = self._last_file_path
        
        return self.load_from_file(file_path)