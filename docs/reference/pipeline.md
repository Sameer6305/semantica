# Pipeline

> **Robust orchestration engine for building, executing, and managing complex data processing workflows.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-pipe:{ .lg .middle } **Pipeline Builder**

    ---

    Fluent API for constructing complex DAG workflows

-   :material-play-circle:{ .lg .middle } **Execution Engine**

    ---

    Robust execution with status tracking and progress monitoring

-   :material-alert-circle-check:{ .lg .middle } **Error Handling**

    ---

    Configurable retry policies, fallbacks, and error recovery

-   :material-fast-forward:{ .lg .middle } **Parallel Execution**

    ---

    Execute independent steps in parallel for maximum performance

-   :material-cpu-64-bit:{ .lg .middle } **Resource Scheduling**

    ---

    Manage CPU/Memory allocation for resource-intensive tasks

-   :material-file-document-edit:{ .lg .middle } **Templates**

    ---

    Pre-built templates for common workflows (ETL, GraphRAG)

</div>

!!! tip "When to Use"
    - **ETL Workflows**: Ingest -> Parse -> Split -> Embed -> Store
    - **Graph Construction**: Extract Entities -> Extract Relations -> Build Graph
    - **Batch Processing**: Processing large volumes of documents reliably

---

## ⚙️ Algorithms Used

### Execution Management
- **DAG Topological Sort**: Determines execution order of steps
- **State Management**: Tracks `PENDING`, `RUNNING`, `COMPLETED`, `FAILED` states
- **Checkpointing**: Saves intermediate results to allow resuming failed pipelines

### Parallelism
- **ThreadPoolExecutor**: For I/O-bound tasks (network requests, DB writes)
- **ProcessPoolExecutor**: For CPU-bound tasks (parsing, embedding generation)
- **Dependency Resolution**: Identifies steps that can run concurrently

### Error Handling
- **Exponential Backoff**: `wait = base * (factor ^ attempt)`
- **Jitter**: Randomization to prevent thundering herd problem
- **Circuit Breaker**: Stops execution after threshold failures to prevent cascading issues

### Resource Scheduling
- **Token Bucket**: Rate limiting for API calls
- **Semaphore**: Concurrency limiting for resource constraints
- **Priority Queue**: Scheduling critical tasks first

---

## Main Classes

### PipelineBuilder

Fluent interface for constructing pipelines.

**Methods:**

| Method | Description |
|--------|-------------|
| `add_step(name, func)` | Add a processing step |
| `add_dependency(step, dep)` | Define execution order |
| `set_error_handler(handler)` | Configure error handling |
| `build()` | Create immutable Pipeline object |

**Example:**

```python
from semantica.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .add_step("ingest", ingest_func)
    .add_step("parse", parse_func)
    .add_step("embed", embed_func)
    .add_dependency("parse", "ingest")  # parse depends on ingest
    .add_dependency("embed", "parse")   # embed depends on parse
    .build()
)
```

### ExecutionEngine

Executes pipelines and manages lifecycle.

**Methods:**

| Method | Description |
|--------|-------------|
| `execute(pipeline, input)` | Run pipeline synchronously |
| `execute_async(pipeline)` | Run in background |
| `resume(execution_id)` | Resume failed execution |
| `get_status(execution_id)` | Check progress |

**Example:**

```python
from semantica.pipeline import ExecutionEngine

engine = ExecutionEngine()
result = engine.execute(pipeline, input_data={"files": ["doc.pdf"]})

if result.status == "COMPLETED":
    print("Success:", result.output)
else:
    print("Failed:", result.error)
```

### FailureHandler

Manages retries and error recovery.

**Methods:**

| Method | Description |
|--------|-------------|
| `handle_error(error, context)` | Process error |
| `should_retry(attempt)` | Check retry policy |

**Configuration:**

```python
from semantica.pipeline import RetryPolicy

policy = RetryPolicy(
    max_retries=3,
    backoff_factor=2.0,
    exceptions=[NetworkError, TimeoutError]
)
```

### ParallelismManager

Manages concurrent execution.

**Methods:**

| Method | Description |
|--------|-------------|
| `execute_parallel(tasks)` | Run tasks concurrently |
| `map(func, items)` | Parallel map operation |

---

## Convenience Functions

```python
from semantica.pipeline import build_linear_pipeline

# Quick linear pipeline
pipeline = build_linear_pipeline([
    step1_func,
    step2_func,
    step3_func
])
```

---

## Configuration

### Environment Variables

```bash
export PIPELINE_MAX_WORKERS=4
export PIPELINE_DEFAULT_TIMEOUT=300
export PIPELINE_CHECKPOINT_DIR=./checkpoints
```

### YAML Configuration

```yaml
pipeline:
  execution:
    max_workers: 4
    timeout_seconds: 300
    
  retry:
    default_retries: 3
    backoff_factor: 1.5
    
  resources:
    max_memory_mb: 4096
```

---

## Integration Examples

### Complete RAG Ingestion Pipeline

```python
from semantica.pipeline import PipelineBuilder, ExecutionEngine
from semantica.ingest import Ingestor
from semantica.split import TextSplitter
from semantica.embeddings import EmbeddingGenerator
from semantica.vector_store import VectorStore

# 1. Define Steps
def ingest(data):
    return Ingestor().ingest(data['path'])

def split(data):
    return TextSplitter().split(data['text'])

def embed(data):
    return EmbeddingGenerator().generate(data['chunks'])

def store(data):
    return VectorStore().store(data['embeddings'])

# 2. Build Pipeline
pipeline = (
    PipelineBuilder()
    .add_step("ingest", ingest)
    .add_step("split", split)
    .add_step("embed", embed)
    .add_step("store", store)
    .add_dependency("split", "ingest")
    .add_dependency("embed", "split")
    .add_dependency("store", "embed")
    .build()
)

# 3. Execute
engine = ExecutionEngine()
result = engine.execute(pipeline, {"path": "document.pdf"})
```

---

## Best Practices

1.  **Idempotency**: Ensure steps are idempotent (can be run multiple times without side effects) to support retries.
2.  **Granularity**: Keep steps focused on a single task. Smaller steps are easier to debug and retry.
3.  **Context Passing**: Use the execution context to pass metadata between steps, not just return values.
4.  **Error Handling**: Always define specific exceptions for retries; don't retry on `ValueError` or `TypeError`.

---

## Troubleshooting

**Issue**: Pipeline stuck in `RUNNING` state.
**Solution**: Check for deadlocks in dependency graph or infinite loops in steps. Use `timeout_seconds`.

**Issue**: `PickleError` during parallel execution.
**Solution**: Ensure all data passed between steps is serializable. Avoid passing open file handles or database connections.

---

## See Also

- [Ingest Module](ingest.md) - Common first step
- [Split Module](split.md) - Common processing step
- [Vector Store Module](vector_store.md) - Common sink step
