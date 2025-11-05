"""
Lifecycle Management Module

Manages system lifecycle including startup, shutdown, and health monitoring.

Key Features:
    - Startup/shutdown hooks
    - Health monitoring
    - Graceful degradation
    - Resource cleanup
    - State management

Main Classes:
    - LifecycleManager: Lifecycle coordination
"""

import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils.exceptions import SemanticaError
from ..utils.logging import get_logger


class SystemState(str, Enum):
    """System state enumeration."""
    
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class HealthStatus:
    """Health status information."""
    
    component: str
    healthy: bool
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


class LifecycleManager:
    """
    System lifecycle manager.
    
    Coordinates startup, shutdown, and health monitoring of all
    framework components.
    
    Attributes:
        state: Current system state
        health_status: Dictionary of component health statuses
        startup_hooks: List of startup hooks
        shutdown_hooks: List of shutdown hooks
    """
    
    def __init__(self):
        """Initialize lifecycle manager."""
        self.state: SystemState = SystemState.UNINITIALIZED
        self.health_status: Dict[str, HealthStatus] = {}
        self.startup_hooks: List[Tuple[Callable, int]] = []  # (hook, priority)
        self.shutdown_hooks: List[Tuple[Callable, int]] = []  # (hook, priority)
        self.logger = get_logger("lifecycle")
        self._component_registry: Dict[str, Any] = {}
        self._last_health_check: Optional[float] = None
    
    def startup(self) -> None:
        """
        Execute startup sequence.
        
        This method initializes all components in the correct order
        and verifies system readiness.
        
        Raises:
            SemanticaError: If startup fails
        """
        if self.state == SystemState.READY or self.state == SystemState.RUNNING:
            self.logger.warning("System already started")
            return
        
        try:
            self.state = SystemState.INITIALIZING
            self.logger.info("Starting system lifecycle")
            
            # Sort hooks by priority (lower = earlier)
            sorted_hooks = sorted(self.startup_hooks, key=lambda x: x[1])
            
            # Execute startup hooks
            for hook_fn, priority in sorted_hooks:
                try:
                    self.logger.debug(f"Executing startup hook (priority: {priority})")
                    hook_fn()
                except Exception as e:
                    self.logger.error(f"Startup hook failed: {e}")
                    self.state = SystemState.ERROR
                    raise SemanticaError(f"Startup hook failed: {e}")
            
            # Verify all components are initialized
            self._verify_components()
            
            # Run initial health checks
            health_results = self.health_check()
            unhealthy = [
                name for name, status in health_results.items()
                if not status.healthy
            ]
            
            if unhealthy:
                self.logger.warning(f"Some components are unhealthy: {unhealthy}")
            
            self.state = SystemState.READY
            self.logger.info("System startup completed successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"System startup failed: {e}")
            raise
    
    def shutdown(self, graceful: bool = True) -> None:
        """
        Execute shutdown sequence.
        
        Args:
            graceful: Whether to shutdown gracefully (default: True)
        """
        if self.state == SystemState.STOPPED:
            self.logger.warning("System already stopped")
            return
        
        try:
            self.state = SystemState.STOPPING
            self.logger.info(f"Shutting down system (graceful: {graceful})")
            
            # Sort hooks by priority (lower = earlier)
            sorted_hooks = sorted(self.shutdown_hooks, key=lambda x: x[1])
            
            # Execute shutdown hooks
            for hook_fn, priority in sorted_hooks:
                try:
                    self.logger.debug(f"Executing shutdown hook (priority: {priority})")
                    hook_fn()
                except Exception as e:
                    if graceful:
                        self.logger.warning(f"Shutdown hook failed: {e}")
                    else:
                        self.logger.error(f"Shutdown hook failed: {e}")
                        raise SemanticaError(f"Shutdown hook failed: {e}")
            
            # Cleanup resources
            self._cleanup_resources()
            
            self.state = SystemState.STOPPED
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.state = SystemState.ERROR
            self.logger.error(f"System shutdown failed: {e}")
            if not graceful:
                raise
    
    def health_check(self) -> Dict[str, HealthStatus]:
        """
        Perform system health check.
        
        Checks health of all registered components and returns
        status information.
        
        Returns:
            Dictionary mapping component names to HealthStatus objects
        """
        self._last_health_check = time.time()
        
        # Check registered components
        health_results = {}
        
        for component_name, component in self._component_registry.items():
            try:
                # Try to get health from component
                if hasattr(component, "health_check"):
                    component_health = component.health_check()
                    if isinstance(component_health, dict):
                        healthy = component_health.get("healthy", True)
                        message = component_health.get("message", "")
                        details = component_health.get("details", {})
                    else:
                        healthy = bool(component_health)
                        message = ""
                        details = {}
                else:
                    # Default: assume healthy if component exists
                    healthy = component is not None
                    message = ""
                    details = {}
                
                status = HealthStatus(
                    component=component_name,
                    healthy=healthy,
                    message=message,
                    details=details
                )
                
            except Exception as e:
                status = HealthStatus(
                    component=component_name,
                    healthy=False,
                    message=f"Health check failed: {e}",
                    details={"error": str(e)}
                )
            
            health_results[component_name] = status
            self.health_status[component_name] = status
        
        # Log unhealthy components
        unhealthy = [
            name for name, status in health_results.items()
            if not status.healthy
        ]
        
        if unhealthy:
            self.logger.warning(
                f"Unhealthy components detected: {unhealthy}",
                extra={"health_status": health_results}
            )
        
        return health_results
    
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component for health monitoring.
        
        Args:
            name: Component name
            component: Component instance
        """
        self._component_registry[name] = component
        self.logger.debug(f"Registered component: {name}")
    
    def unregister_component(self, name: str) -> None:
        """
        Unregister a component.
        
        Args:
            name: Component name
        """
        if name in self._component_registry:
            del self._component_registry[name]
            if name in self.health_status:
                del self.health_status[name]
            self.logger.debug(f"Unregistered component: {name}")
    
    def register_startup_hook(self, hook_fn: Callable[[], None], priority: int = 50) -> None:
        """
        Register a startup hook.
        
        Hooks are executed in order of priority (lower = earlier).
        Priority values are typically between 0-100.
        
        Args:
            hook_fn: Function to call during startup (no arguments)
            priority: Hook priority (lower = earlier execution, default: 50)
        """
        if not callable(hook_fn):
            raise ValueError("hook_fn must be callable")
        
        self.startup_hooks.append((hook_fn, priority))
        self.logger.debug(f"Registered startup hook with priority {priority}")
    
    def register_shutdown_hook(
        self,
        hook_fn: Callable[[], None],
        priority: int = 50
    ) -> None:
        """
        Register a shutdown hook.
        
        Hooks are executed in order of priority (lower = earlier).
        Priority values are typically between 0-100.
        
        Args:
            hook_fn: Function to call during shutdown (no arguments)
            priority: Hook priority (lower = earlier execution, default: 50)
        """
        if not callable(hook_fn):
            raise ValueError("hook_fn must be callable")
        
        self.shutdown_hooks.append((hook_fn, priority))
        self.logger.debug(f"Registered shutdown hook with priority {priority}")
    
    def get_state(self) -> SystemState:
        """
        Get current system state.
        
        Returns:
            Current system state
        """
        return self.state
    
    def is_ready(self) -> bool:
        """
        Check if system is ready.
        
        Returns:
            True if system is ready, False otherwise
        """
        return self.state == SystemState.READY or self.state == SystemState.RUNNING
    
    def is_running(self) -> bool:
        """
        Check if system is running.
        
        Returns:
            True if system is running, False otherwise
        """
        return self.state == SystemState.RUNNING
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get summary of system health.
        
        Returns:
            Dictionary with health summary information
        """
        health_results = self.health_check()
        
        total = len(health_results)
        healthy = sum(1 for status in health_results.values() if status.healthy)
        unhealthy = total - healthy
        
        return {
            "state": self.state.value,
            "total_components": total,
            "healthy_components": healthy,
            "unhealthy_components": unhealthy,
            "is_healthy": unhealthy == 0,
            "last_check": self._last_health_check,
            "components": {
                name: {
                    "healthy": status.healthy,
                    "message": status.message,
                    "timestamp": status.timestamp,
                }
                for name, status in health_results.items()
            }
        }
    
    def _verify_components(self) -> None:
        """
        Verify that all registered components are initialized.
        
        Raises:
            SemanticaError: If components are not properly initialized
        """
        uninitialized = []
        
        for name, component in self._component_registry.items():
            if component is None:
                uninitialized.append(name)
        
        if uninitialized:
            raise SemanticaError(
                f"Components not initialized: {', '.join(uninitialized)}"
            )
    
    def _cleanup_resources(self) -> None:
        """Cleanup system resources."""
        # Cleanup components if they have cleanup methods
        for name, component in self._component_registry.items():
            try:
                if hasattr(component, "cleanup"):
                    component.cleanup()
                elif hasattr(component, "close"):
                    component.close()
            except Exception as e:
                self.logger.warning(f"Failed to cleanup component {name}: {e}")
        
        # Clear registries
        self._component_registry.clear()
        self.health_status.clear()