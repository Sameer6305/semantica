"""
Configuration Management Module

This module provides comprehensive configuration management for the Semantica framework,
including loading from files, environment variables, validation, and dynamic updates.
It supports multiple configuration sources and formats with automatic fallback chains
and validation.

Supported Configuration Sources:
    - Configuration files: YAML, JSON formats
    - Environment variables: SEMANTICA_ prefix for automatic loading
    - Programmatic: Python API for setting configuration values
    - Dictionary: Direct dictionary-based configuration

Algorithms Used:
    - YAML Parsing: YAML parser for configuration file loading
    - JSON Parsing: JSON parser for configuration file loading
    - Environment Variable Parsing: OS-level environment variable access with prefix matching
    - Fallback Chain: Priority-based configuration resolution (file -> env -> defaults)
    - Dictionary Merging: Deep merge algorithms for configuration updates
    - Validation: Type checking, range validation, required field checking
    - Nested Access: Dot notation parsing for nested configuration access

Key Features:
    - YAML/JSON configuration file parsing
    - Environment variable support with SEMANTICA_ prefix
    - Configuration validation with detailed error messages
    - Dynamic configuration updates at runtime
    - Configuration inheritance and merging
    - Nested configuration access via dot notation
    - Automatic type conversion for environment variables
    - Progress tracking for configuration loading operations

Main Classes:
    - Config: Configuration data class with validation
    - ConfigManager: Configuration loading, validation, and management

Example Usage:
    >>> from semantica.core import ConfigManager
    >>> manager = ConfigManager()
    >>> config = manager.load_from_file("config.yaml")
    >>> batch_size = config.get("processing.batch_size", default=32)
    >>> 
    >>> # Merge multiple configurations
    >>> merged = manager.merge_configs(config1, config2, config3)
    >>> 
    >>> # Load from dictionary
    >>> config = manager.load_from_dict({"processing": {"batch_size": 64}})

Author: Semantica Contributors
License: MIT
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
from ..utils.progress_tracker import get_progress_tracker


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
        
        Environment variables with prefix SEMANTICA_ will override configuration
        values. The format is: SEMANTICA_SECTION_KEY=value
        
        Examples:
            SEMANTICA_PROCESSING_BATCH_SIZE=64
            SEMANTICA_LLM_PROVIDER_MODEL=gpt-4
            SEMANTICA_QUALITY_MIN_CONFIDENCE=0.8
        
        Args:
            config_dict: Configuration dictionary to update with env values
        """
        prefix = "SEMANTICA_"
        prefix_length = len(prefix)
        
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue
            
            # Remove prefix and convert to lowercase for consistency
            config_key = env_key[prefix_length:].lower()
            
            # Parse the environment variable value
            # Try JSON first (for complex types like lists/dicts)
            try:
                parsed_value = json.loads(env_value)
            except (json.JSONDecodeError, ValueError):
                # Not valid JSON, try type conversion
                parsed_value = self._parse_env_value(env_value)
            
            # Set nested value using dot notation
            # e.g., "processing_batch_size" -> "processing.batch_size"
            normalized_key = config_key.replace("_", ".")
            set_nested_value(config_dict, normalized_key, parsed_value)
    
    def _parse_env_value(self, value: str) -> Union[str, int, float, bool]:
        """
        Parse environment variable value to appropriate Python type.
        
        Args:
            value: Raw environment variable value string
            
        Returns:
            Parsed value with appropriate type (bool, int, float, or str)
        """
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Integer values
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        
        # Float values
        try:
            # Check if it's a valid float (allows one decimal point)
            if value.replace(".", "", 1).replace("-", "", 1).isdigit():
                return float(value)
        except ValueError:
            pass
        
        # Default: return as string
        return value
    
    def validate(self) -> None:
        """
        Validate configuration settings.
        
        This method performs comprehensive validation of all configuration
        settings, checking types, ranges, and required fields.
        
        Raises:
            ConfigurationError: If configuration is invalid with detailed error messages
        """
        validation_errors = []
        
        # Validate processing settings
        processing_config = self.processing
        if processing_config:
            # Validate batch_size
            if "batch_size" in processing_config:
                batch_size = processing_config["batch_size"]
                if not isinstance(batch_size, int):
                    validation_errors.append(
                        f"processing.batch_size must be an integer, got {type(batch_size).__name__}"
                    )
                elif batch_size <= 0:
                    validation_errors.append(
                        f"processing.batch_size must be positive, got {batch_size}"
                    )
            
            # Validate max_workers
            if "max_workers" in processing_config:
                max_workers = processing_config["max_workers"]
                if not isinstance(max_workers, int):
                    validation_errors.append(
                        f"processing.max_workers must be an integer, got {type(max_workers).__name__}"
                    )
                elif max_workers <= 0:
                    validation_errors.append(
                        f"processing.max_workers must be positive, got {max_workers}"
                    )
        
        # Validate quality settings
        quality_config = self.quality
        if quality_config:
            # Validate min_confidence
            if "min_confidence" in quality_config:
                confidence = quality_config["min_confidence"]
                if not isinstance(confidence, (int, float)):
                    validation_errors.append(
                        f"quality.min_confidence must be a number, got {type(confidence).__name__}"
                    )
                elif not (0.0 <= confidence <= 1.0):
                    validation_errors.append(
                        f"quality.min_confidence must be between 0.0 and 1.0, got {confidence}"
                    )
        
        # Validate logging settings
        logging_config = self.logging
        if logging_config:
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if "level" in logging_config:
                level = logging_config["level"]
                if level not in valid_levels:
                    validation_errors.append(
                        f"logging.level must be one of {valid_levels}, got {level}"
                    )
        
        # Raise error if any validation failures
        if validation_errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in validation_errors
            )
            raise ConfigurationError(
                error_message,
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
    
    This class provides a centralized way to load, validate, merge, and manage
    configuration for the Semantica framework. It supports multiple configuration
    sources and formats.
    
    Features:
        - Load from YAML/JSON files
        - Load from dictionaries
        - Merge multiple configurations
        - Validate configuration
        - Reload configuration dynamically
    
    Example Usage:
        >>> manager = ConfigManager()
        >>> config = manager.load_from_file("config.yaml")
        >>> merged = manager.merge_configs(config1, config2, config3)
    """
    
    def __init__(self):
        """
        Initialize configuration manager.
        
        Creates a new ConfigManager instance with no loaded configuration.
        Use load_from_file() or load_from_dict() to load configuration.
        """
        self._config: Optional[Config] = None
        self._last_file_path: Optional[Path] = None
        self.progress_tracker = get_progress_tracker()
    
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
        # Track configuration loading
        tracking_id = self.progress_tracker.start_tracking(
            file=str(file_path),
            module="core",
            submodule="ConfigManager",
            message=f"Loading configuration from: {file_path}"
        )
        
        try:
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
                
                # Create config object from loaded dictionary
                config = Config(config_dict=config_dict)
                
                # Validate configuration if requested
                if validate:
                    config.validate()
                
                # Store config and file path for potential reload
                self._config = config
                self._last_file_path = file_path
                
                self.progress_tracker.stop_tracking(tracking_id, status="completed",
                                                   message="Configuration loaded successfully")
                return config
            except Exception as e:
                # Re-raise as ConfigurationError if inner try fails
                raise ConfigurationError(
                    f"Failed to parse configuration file: {str(e)}",
                    config_context={"file_path": str(file_path)}
                ) from e
            
        except Exception as e:
            self.progress_tracker.stop_tracking(tracking_id, status="failed", message=str(e))
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