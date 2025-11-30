# Ingest

> **Universal data ingestion from files, web, feeds, streams, repos, emails, and databases.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-file-document-multiple:{ .lg .middle } **File Ingestion**

    ---

    Local and Cloud (S3/GCS/Azure) file processing with type detection

-   :material-web:{ .lg .middle } **Web Crawling**

    ---

    Scrape websites with rate limiting, robots.txt compliance, and JS rendering

-   :material-rss:{ .lg .middle } **Feed Monitoring**

    ---

    Consume RSS/Atom feeds with real-time update monitoring

-   :material-broadcast:{ .lg .middle } **Stream Processing**

    ---

    Ingest from Kafka, RabbitMQ, Kinesis, and Pulsar

-   :material-source-branch:{ .lg .middle } **Code Repos**

    ---

    Clone and analyze Git repositories (GitHub/GitLab)

-   :material-database:{ .lg .middle } **Databases**

    ---

    Ingest tables and query results from SQL databases

</div>

!!! tip "When to Use"
    - **Data Onboarding**: The entry point for all external data into Semantica
    - **Crawling**: Building a dataset from the web
    - **Monitoring**: Listening to real-time data streams
    - **Migration**: Importing legacy data from databases

---

## ⚙️ Algorithms Used

### File Processing
- **Magic Number Detection**: Identifying file types by binary signatures, not just extensions.
- **Recursive Traversal**: Efficiently walking directory trees.

### Web Ingestion
- **Sitemap Parsing**: Discovering URLs via XML sitemaps.
- **DOM Extraction**: Removing boilerplate (nav/ads) to extract main content.
- **Politeness**: Respecting `robots.txt` and enforcing crawl delays.

### Stream Processing
- **Consumer Groups**: Managing offsets and scaling consumption (Kafka).
- **Backpressure**: Handling high-velocity streams without crashing.

### Repository Analysis
- **AST Parsing**: Extracting code structure (classes/functions) without executing.
- **Dependency Parsing**: Reading `requirements.txt`, `package.json`, etc.

---

## Main Classes

### FileIngestor

Handles file systems and object storage.

**Methods:**

| Method | Description |
|--------|-------------|
| `ingest_file(path)` | Process single file |
| `ingest_directory(path)` | Process folder |

### WebIngestor

Handles web content.

**Methods:**

| Method | Description |
|--------|-------------|
| `ingest_url(url)` | Scrape page |
| `crawl(url, depth)` | Recursive crawl |

### StreamIngestor

Handles real-time data.

**Methods:**

| Method | Description |
|--------|-------------|
| `consume(topic)` | Start consumption |

### RepoIngestor

Handles code repositories.

**Methods:**

| Method | Description |
|--------|-------------|
| `ingest_repo(url)` | Clone and process |

---

## Convenience Functions

```python
from semantica.ingest import ingest

# Auto-detect source type
ingest("doc.pdf", source_type="file")
ingest("https://google.com", source_type="web")
ingest("kafka://topic", source_type="stream")
```

---

## Configuration

### Environment Variables

```bash
export INGEST_USER_AGENT="SemanticaBot/1.0"
export AWS_ACCESS_KEY_ID=...
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### YAML Configuration

```yaml
ingest:
  web:
    user_agent: "MyBot"
    rate_limit: 1.0 # seconds
    
  files:
    max_size: 100MB
    allowed_extensions: [.pdf, .txt, .md]
```

---

## Integration Examples

### Continuous Feed Ingestion

```python
from semantica.ingest import FeedIngestor
from semantica.pipeline import Pipeline

# 1. Setup Monitor
ingestor = FeedIngestor()
pipeline = Pipeline()

def on_new_item(item):
    # 2. Trigger Pipeline on new content
    pipeline.run(item.content)

# 3. Start Monitoring
ingestor.monitor(
    "https://news.ycombinator.com/rss",
    callback=on_new_item,
    interval=300
)
```

---

## Best Practices

1.  **Respect Rate Limits**: When crawling, always set a polite `rate_limit` to avoid getting banned.
2.  **Filter Content**: Use `allowed_extensions` or URL patterns to avoid ingesting junk data.
3.  **Handle Failures**: Stream ingestors should have error handlers for bad messages.
4.  **Use Credentials Securely**: Never hardcode API keys; use Environment Variables.

---

## See Also

- [Parse Module](parse.md) - Processes the raw data ingested here
- [Split Module](split.md) - Chunks the ingested content
- [Utils Module](utils.md) - Validation helpers
