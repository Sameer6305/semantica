# Utils

> **Shared utilities for logging, validation, error handling, and common operations.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-console-line:{ .lg .middle } **Logging**

    ---

    Structured logging with performance tracking and error reporting

-   :material-alert-circle-outline:{ .lg .middle } **Error Handling**

    ---

    Custom exception hierarchy and error formatting

-   :material-check-all:{ .lg .middle } **Validation**

    ---

    Data validation for entities, relationships, and configuration

-   :material-progress-clock:{ .lg .middle } **Progress Tracking**

    ---

    Track long-running operations with console or file output

-   :material-tools:{ .lg .middle } **Helpers**

    ---

    Common functions for text cleaning, hashing, and file I/O

-   :material-code-json:{ .lg .middle } **Type Definitions**

    ---

    Shared TypedDicts and Enums for type safety

</div>

!!! tip "When to Use"
    - **Development**: Use `setup_logging` to configure output.
    - **Data Cleaning**: Use `clean_text` and `normalize_entities`.
    - **Validation**: Use `validate_data` before processing external input.
    - **Debugging**: Use `log_performance` to find bottlenecks.

---

## ⚙️ Key Components

### Logging
- **Structured Output**: JSON or formatted text logs.
- **Performance Metrics**: Decorators for timing functions.
- **Quality Logging**: Specialized loggers for data quality issues.

### Validation
- **Schema Validation**: Check dictionary structure against requirements.
- **Type Checking**: Runtime type validation.
- **Constraint Checking**: Numeric ranges, string lengths, regex patterns.

### Progress Tracking
- **Multi-Environment**: Supports Console (tqdm), Jupyter, and File logging.
- **Module Awareness**: Tracks progress per module.

---

## Main Classes

### Logger

Centralized logging configuration.

**Functions:**

| Function | Description |
|----------|-------------|
| `setup_logging(level)` | Configure global logging |
| `get_logger(name)` | Get named logger instance |
| `log_performance(func)` | Decorator for timing |

**Example:**

```python
from semantica.utils import setup_logging, get_logger, log_performance

setup_logging(level="INFO")
logger = get_logger(__name__)

@log_performance
def process_data(data):
    logger.info(f"Processing {len(data)} items")
```

### Validators

Data validation functions.

**Functions:**

| Function | Description |
|----------|-------------|
| `validate_entity(data)` | Check entity structure |
| `validate_config(cfg)` | Check configuration |

**Example:**

```python
from semantica.utils import validate_entity, ValidationError

try:
    validate_entity({"id": "1", "type": "PERSON"})
except ValidationError as e:
    print(f"Invalid entity: {e}")
```

### ProgressTracker

Tracks execution progress.

**Classes:**

| Class | Description |
|-------|-------------|
| `ProgressTracker` | Main tracker interface |
| `ConsoleProgressDisplay` | CLI output |

**Example:**

```python
from semantica.utils import track_progress

for item in track_progress(items, desc="Processing"):
    process(item)
```

---

## Convenience Functions

```python
from semantica.utils import clean_text, hash_data, safe_filename

# Text cleaning
clean = clean_text("  Hello   World  ")  # "Hello World"

# Hashing
id = hash_data({"key": "value"})

# File safety
fname = safe_filename("My File?.txt")  # "My_File_.txt"
```

---

## Configuration

### Environment Variables

```bash
export SEMANTICA_LOG_LEVEL=DEBUG
export SEMANTICA_LOG_FORMAT=json
export SEMANTICA_PROGRESS_BAR=true
```

---

## Best Practices

1.  **Use `get_logger`**: Always use `get_logger(__name__)` instead of `print` for production code.
2.  **Validate Early**: Validate input data at the boundary (Ingest/Parse) using `validate_data`.
3.  **Handle Exceptions**: Catch `SemanticaError` for framework-specific errors.
4.  **Clean Text**: Use `clean_text` before embedding or extraction to improve quality.

---

## See Also

- [Core Module](core.md) - Uses Utils for infrastructure
- [Pipeline Module](pipeline.md) - Uses ProgressTracker
