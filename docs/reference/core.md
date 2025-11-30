# Core

> **Framework infrastructure, lifecycle management, and plugin system.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-cogs:{ .lg .middle } **Orchestrator**

    ---

    Central coordinator for all framework components and workflows

-   :material-lifecycle:{ .lg .middle } **Lifecycle Management**

    ---

    Manage initialization, startup, shutdown, and state transitions

-   :material-tune:{ .lg .middle } **Configuration**

    ---

    Unified configuration management via YAML and Environment variables

-   :material-puzzle:{ .lg .middle } **Plugin System**

    ---

    Extensible plugin registry for adding custom modules and capabilities

-   :material-console:{ .lg .middle } **Logging & Telemetry**

    ---

    Centralized logging and metrics collection

</div>

!!! tip "When to Use"
    - **Application Startup**: Initializing the Semantica framework in your app
    - **Configuration**: Tuning global settings
    - **Extension**: Developing custom plugins or modules
    - **Orchestration**: Coordinating complex workflows across multiple modules

---

## ⚙️ Algorithms Used

### Lifecycle Management
- **State Machine**: `CREATED` -> `INITIALIZED` -> `RUNNING` -> `STOPPED`
- **Dependency Injection**: Resolving and injecting dependencies between modules.
- **Graceful Shutdown**: Ensuring all resources (DB connections, thread pools) are closed properly.

### Configuration
- **Layered Loading**: Defaults -> Config File -> Environment Variables -> CLI Arguments (Priority order).
- **Schema Validation**: Validating config structure against defined schemas.

### Plugin System
- **Discovery**: Auto-discovery of plugins via entry points or directory scanning.
- **Registration**: Dynamic registration of classes and functions.
- **Hook Execution**: Running plugin hooks at specific lifecycle events.

---

## Main Classes

### Orchestrator

The brain of the framework.

**Methods:**

| Method | Description |
|--------|-------------|
| `start()` | Initialize and start all components |
| `stop()` | Graceful shutdown |
| `get_component(name)` | Access initialized module |

**Example:**

```python
from semantica.core import Orchestrator

app = Orchestrator()
app.start()

# Access modules
kg = app.get_component("knowledge_graph")
ingest = app.get_component("ingest")
```

### ConfigManager

Manages global configuration.

**Methods:**

| Method | Description |
|--------|-------------|
| `load(path)` | Load config from file |
| `get(key, default)` | Get config value |

### PluginRegistry

Manages extensions.

**Methods:**

| Method | Description |
|--------|-------------|
| `register(plugin)` | Register new plugin |
| `get_plugin(name)` | Retrieve plugin |

---

## Configuration

### Environment Variables

```bash
export SEMANTICA_ENV=production
export SEMANTICA_LOG_LEVEL=INFO
export SEMANTICA_CONFIG_PATH=./config.yaml
```

### YAML Configuration

```yaml
core:
  environment: production
  log_level: INFO
  plugins:
    enabled: true
    directory: ./plugins
```

---

## Integration Examples

### Custom Application

```python
from semantica.core import Orchestrator, ConfigManager

# 1. Load Config
config = ConfigManager()
config.load("config.yaml")

# 2. Initialize Orchestrator
app = Orchestrator(config=config)

# 3. Register Custom Plugin
class MyPlugin:
    name = "my_plugin"
    def initialize(self):
        print("My Plugin Started")

app.plugin_registry.register(MyPlugin())

# 4. Start
app.start()

# 5. Run Workload
try:
    app.run_pipeline("my_pipeline")
finally:
    app.stop()
```

---

## Best Practices

1.  **Use Orchestrator**: Avoid manually instantiating every module; let the Orchestrator handle dependencies.
2.  **Graceful Shutdown**: Always ensure `app.stop()` is called (e.g., in a `finally` block) to prevent resource leaks.
3.  **Config Layers**: Use `config.yaml` for defaults and Environment Variables for secrets/overrides.

---

## See Also

- [Pipeline Module](pipeline.md) - Executed by the Orchestrator
- [Utils Module](utils.md) - Shared utilities used by Core
