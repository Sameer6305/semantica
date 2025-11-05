"""
Pipeline Construction Module

Handles construction and configuration of processing pipelines.

Key Features:
    - Pipeline construction DSL
    - Step configuration and chaining
    - Pipeline validation and optimization
    - Error handling and recovery
    - Pipeline serialization and deserialization
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from .pipeline_validator import PipelineValidator


class StepStatus(Enum):
    """Pipeline step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Pipeline step definition."""
    name: str
    step_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    handler: Optional[Callable] = None
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None


@dataclass
class Pipeline:
    """Pipeline definition."""
    name: str
    steps: List[PipelineStep] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineBuilder:
    """
    Pipeline construction and configuration handler.
    
    • Constructs processing pipelines using DSL
    • Configures pipeline steps and connections
    • Validates pipeline structure and dependencies
    • Optimizes pipeline performance
    • Handles pipeline serialization
    • Supports complex pipeline topologies
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize pipeline builder.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("pipeline_builder")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.validator = PipelineValidator(**self.config)
        self.steps: List[PipelineStep] = []
        self.step_registry: Dict[str, Callable] = {}
        self.pipeline_config: Dict[str, Any] = {}
    
    def add_step(
        self,
        step_name: str,
        step_type: str,
        **config
    ) -> "PipelineBuilder":
        """
        Add step to pipeline.
        
        Args:
            step_name: Step name/identifier
            step_type: Step type/category
            **config: Step configuration
        
        Returns:
            Self for method chaining
        """
        step = PipelineStep(
            name=step_name,
            step_type=step_type,
            config=config,
            dependencies=config.get("dependencies", []),
            handler=config.get("handler")
        )
        
        self.steps.append(step)
        self.logger.debug(f"Added step: {step_name} ({step_type})")
        
        return self
    
    def connect_steps(
        self,
        from_step: str,
        to_step: str,
        **options
    ) -> "PipelineBuilder":
        """
        Connect pipeline steps.
        
        Args:
            from_step: Source step name
            to_step: Target step name
            **options: Connection options
        
        Returns:
            Self for method chaining
        """
        # Find target step and add dependency
        target_step = next((s for s in self.steps if s.name == to_step), None)
        if target_step:
            if from_step not in target_step.dependencies:
                target_step.dependencies.append(from_step)
        else:
            raise ValidationError(f"Target step not found: {to_step}")
        
        return self
    
    def set_parallelism(self, level: int) -> "PipelineBuilder":
        """
        Set parallelism level.
        
        Args:
            level: Parallelism level (number of parallel workers)
        
        Returns:
            Self for method chaining
        """
        self.pipeline_config["parallelism"] = level
        return self
    
    def build(self, name: str = "default_pipeline") -> Pipeline:
        """
        Build pipeline from configuration.
        
        Args:
            name: Pipeline name
        
        Returns:
            Built pipeline
        """
        # Validate pipeline structure
        validation_result = self.validator.validate_pipeline(self)
        if not validation_result.get("valid", False):
            errors = validation_result.get("errors", [])
            raise ValidationError(f"Pipeline validation failed: {errors}")
        
        pipeline = Pipeline(
            name=name,
            steps=list(self.steps),
            config=self.pipeline_config,
            metadata={
                "step_count": len(self.steps),
                "parallelism": self.pipeline_config.get("parallelism", 1)
            }
        )
        
        self.logger.info(f"Built pipeline: {name} with {len(self.steps)} steps")
        
        return pipeline
    
    def build_pipeline(
        self,
        pipeline_config: Dict[str, Any],
        **options
    ) -> Pipeline:
        """
        Build pipeline from configuration dictionary.
        
        Args:
            pipeline_config: Pipeline configuration
            **options: Additional options
        
        Returns:
            Built pipeline
        """
        # Parse configuration
        pipeline_name = pipeline_config.get("name", "default_pipeline")
        steps_config = pipeline_config.get("steps", [])
        
        # Add steps from configuration
        for step_config in steps_config:
            step_name = step_config.get("name")
            step_type = step_config.get("type")
            if step_name and step_type:
                self.add_step(step_name, step_type, **step_config.get("config", {}))
        
        # Set parallelism if specified
        if "parallelism" in pipeline_config:
            self.set_parallelism(pipeline_config["parallelism"])
        
        return self.build(pipeline_name)
    
    def register_step_handler(
        self,
        step_type: str,
        handler: Callable
    ) -> None:
        """
        Register step handler function.
        
        Args:
            step_type: Step type
            handler: Handler function
        """
        self.step_registry[step_type] = handler
        self.logger.debug(f"Registered handler for step type: {step_type}")
    
    def get_step(self, step_name: str) -> Optional[PipelineStep]:
        """Get step by name."""
        return next((s for s in self.steps if s.name == step_name), None)
    
    def serialize(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        Serialize pipeline configuration.
        
        Args:
            format: Serialization format
        
        Returns:
            Serialized pipeline
        """
        pipeline_data = {
            "name": "pipeline",
            "steps": [
                {
                    "name": step.name,
                    "type": step.step_type,
                    "config": step.config,
                    "dependencies": step.dependencies
                }
                for step in self.steps
            ],
            "config": self.pipeline_config
        }
        
        if format == "json":
            import json
            return json.dumps(pipeline_data, indent=2)
        else:
            return pipeline_data
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """
        Validate pipeline structure and configuration.
        
        Returns:
            Validation results
        """
        return self.validator.validate_pipeline(self)


class PipelineSerializer:
    """
    Pipeline serialization handler.
    
    • Serializes pipelines to various formats
    • Handles pipeline deserialization
    • Manages pipeline versioning
    • Processes pipeline metadata
    """
    
    def __init__(self, **config):
        """Initialize pipeline serializer."""
        self.logger = get_logger("pipeline_serializer")
        self.config = config
    
    def serialize_pipeline(
        self,
        pipeline: Pipeline,
        format: str = "json",
        **options
    ) -> Union[str, Dict[str, Any]]:
        """
        Serialize pipeline to specified format.
        
        Args:
            pipeline: Pipeline object
            format: Serialization format
            **options: Additional options
        
        Returns:
            Serialized pipeline
        """
        pipeline_data = {
            "name": pipeline.name,
            "steps": [
                {
                    "name": step.name,
                    "type": step.step_type,
                    "config": step.config,
                    "dependencies": step.dependencies
                }
                for step in pipeline.steps
            ],
            "config": pipeline.config,
            "metadata": pipeline.metadata
        }
        
        if format == "json":
            import json
            return json.dumps(pipeline_data, indent=2, default=str)
        else:
            return pipeline_data
    
    def deserialize_pipeline(
        self,
        serialized_pipeline: Union[str, Dict[str, Any]],
        **options
    ) -> Pipeline:
        """
        Deserialize pipeline from serialized format.
        
        Args:
            serialized_pipeline: Serialized pipeline data
            **options: Additional options
        
        Returns:
            Reconstructed pipeline
        """
        # Parse if string
        if isinstance(serialized_pipeline, str):
            import json
            pipeline_data = json.loads(serialized_pipeline)
        else:
            pipeline_data = serialized_pipeline
        
        # Reconstruct pipeline
        builder = PipelineBuilder(**self.config)
        pipeline = builder.build_pipeline(pipeline_data, **options)
        
        return pipeline
    
    def version_pipeline(
        self,
        pipeline: Pipeline,
        version_info: Dict[str, Any]
    ) -> Pipeline:
        """
        Add versioning information to pipeline.
        
        Args:
            pipeline: Pipeline object
            version_info: Version information
        
        Returns:
            Versioned pipeline
        """
        pipeline.metadata["version"] = version_info.get("version", "1.0")
        pipeline.metadata["version_info"] = version_info
        
        return pipeline
