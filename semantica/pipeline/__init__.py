"""
Pipeline and Orchestration Module

This module provides comprehensive pipeline construction and orchestration capabilities.

Exports:
    - PipelineBuilder: Pipeline construction DSL
    - ExecutionEngine: Pipeline execution engine
    - FailureHandler: Error handling and retry mechanisms
    - ParallelismManager: Parallel execution management
    - ResourceScheduler: Resource allocation and scheduling
    - PipelineValidator: Pipeline validation and testing
    - PipelineTemplateManager: Pre-built pipeline templates
"""

from .pipeline_builder import (
    PipelineBuilder,
    Pipeline,
    PipelineStep,
    StepStatus,
    PipelineSerializer
)
from .execution_engine import (
    ExecutionEngine,
    ExecutionResult,
    PipelineStatus,
    ProgressTracker
)
from .failure_handler import (
    FailureHandler,
    RetryHandler,
    FallbackHandler,
    ErrorRecovery,
    RetryPolicy,
    RetryStrategy,
    ErrorSeverity,
    FailureRecovery
)
from .parallelism_manager import (
    ParallelismManager,
    ParallelExecutor,
    Task,
    ParallelExecutionResult
)
from .resource_scheduler import (
    ResourceScheduler,
    Resource,
    ResourceAllocation,
    ResourceType
)
from .pipeline_validator import (
    PipelineValidator,
    ValidationResult
)
from .pipeline_templates import (
    PipelineTemplateManager,
    PipelineTemplate
)

__all__ = [
    # Pipeline construction
    "PipelineBuilder",
    "Pipeline",
    "PipelineStep",
    "StepStatus",
    "PipelineSerializer",
    
    # Execution
    "ExecutionEngine",
    "ExecutionResult",
    "PipelineStatus",
    "ProgressTracker",
    
    # Failure handling
    "FailureHandler",
    "RetryHandler",
    "FallbackHandler",
    "ErrorRecovery",
    "RetryPolicy",
    "RetryStrategy",
    "ErrorSeverity",
    "FailureRecovery",
    
    # Parallelism
    "ParallelismManager",
    "ParallelExecutor",
    "Task",
    "ParallelExecutionResult",
    
    # Resource management
    "ResourceScheduler",
    "Resource",
    "ResourceAllocation",
    "ResourceType",
    
    # Validation
    "PipelineValidator",
    "ValidationResult",
    
    # Templates
    "PipelineTemplateManager",
    "PipelineTemplate",
]
