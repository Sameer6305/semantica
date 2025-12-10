# Frequently Asked Questions

Common questions and answers about Semantica.

!!! tip "Can't find your question?"
    Browse existing questions or [ask a new question on GitHub Issues](https://github.com/Hawksight-AI/semantica/issues/new)

---

## General Questions

### What is Semantica?

Semantica is an open-source framework for building semantic layers and knowledge graphs from unstructured data. It transforms raw data into structured, queryable knowledge that powers AI applications.

### What can I use Semantica for?

- Building knowledge graphs from documents
- Creating semantic layers for AI applications
- Extracting entities and relationships
- Powering GraphRAG systems
- Integrating multi-source data
- Building AI agent memory

### Is Semantica free?

Yes! Semantica is 100% open source and free to use under the MIT License.

### What makes Semantica different?

- **Modular**: Use only what you need
- **Extensible**: Plug in custom models
- **Production-ready**: Built for scale
- **Open source**: Fully transparent

---

## Installation & Setup

### How do I install Semantica?

```bash
pip install semantica
```

See the [Installation Guide](installation.md) for details.

### What Python version do I need?

Python 3.8 or higher. Python 3.11+ is recommended for best performance.

### Do I need a GPU?

No, GPU is optional. Semantica works on CPU, but GPU acceleration is available for faster processing.

### How do I get started?

1. Install: `pip install semantica`
2. Follow the [Quick Start Guide](quickstart.md)
3. Try the [Examples](examples.md)

---

## Knowledge Graphs

### What is a knowledge graph?

A structured representation where entities (nodes) are connected by relationships (edges). It captures semantic meaning and relationships in data.

### How do I build a knowledge graph?

```python
from semantica import Semantica

semantica = Semantica()
result = semantica.build_knowledge_base(["document.pdf"])
kg = result["knowledge_graph"]
```

### Can I merge multiple knowledge graphs?

Yes! Use the `merge` method:

```python
merged = semantica.kg.merge([kg1, kg2, kg3])
```

### How do I visualize a knowledge graph?

```python
semantica.kg.visualize(kg, output_path="graph.html")
```

---

## Usage & Features

### Can I process PDF files?

Yes! Semantica supports PDF, DOCX, HTML, JSON, CSV, and many other formats.

### How do I extract entities from text?

```python
from semantica import Semantica

semantica = Semantica()
result = semantica.semantic_extract.extract_entities("Your text")
entities = result["entities"]
```

### Can I use my own models?

Yes, Semantica is extensible. You can plug in custom models for entity extraction, embeddings, and more.

### What export formats are supported?

- RDF/XML
- OWL (Ontology)
- JSON
- CSV
- YAML
- And more

---

## Conflict Resolution

### What is conflict resolution?

When the same entity appears in multiple sources with different information, conflict resolution determines which information to use.

### What strategies are available?

- **Voting**: Majority wins
- **Credibility Weighted**: Weight by source credibility
- **Most Recent**: Use latest information
- **Highest Confidence**: Use highest confidence score

### How do I set a resolution strategy?

```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver(default_strategy="voting")
```

---

## Integration

### Can I use Semantica with other tools?

Yes! Semantica exports to standard formats that work with:

- Neo4j
- Graph databases
- RDF stores
- Vector databases
- Any tool that accepts RDF/JSON/CSV

### Does it work with LangChain?

Yes, Semantica can be integrated with LangChain for RAG applications.

### Can I connect to databases?

Yes, Semantica supports connections to Neo4j, FalkorDB, and other graph databases.

---

## Performance

### How fast is Semantica?

Performance depends on:

- Document size
- Number of documents
- Hardware (CPU/GPU)
- Configuration options

For typical documents, processing takes seconds to minutes.

### Can I process large datasets?

Yes, but consider:

- Processing in batches
- Using GPU acceleration
- Incremental building
- Optimizing configuration

### How can I improve performance?

- Enable GPU if available
- Process in smaller batches
- Use faster models
- Optimize configuration
- Cache embeddings

---

## Troubleshooting

### Installation fails

- Upgrade pip: `pip install --upgrade pip`
- Use virtual environment
- Check Python version: `python --version`

### No entities extracted

- Verify document contains text (not just images)
- Check document format is supported
- Review extraction configuration

### Memory errors

- Process documents one at a time
- Reduce batch sizes
- Use smaller models
- Increase available RAM

### Slow processing

- Enable GPU if available
- Process in smaller batches
- Optimize configuration
- Use faster models

---

## Getting Help

### Where can I get help?

- **Documentation**: This site
- **GitHub Issues**: [Report bugs or ask questions](https://github.com/Hawksight-AI/semantica/issues)

### How do I report a bug?

Open an issue on [GitHub](https://github.com/Hawksight-AI/semantica/issues) with:

- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details

### Can I contribute?

Yes! We welcome contributions. See our [Contributing Guide](https://github.com/Hawksight-AI/semantica/blob/main/CONTRIBUTING.md).

### How do I request a feature?

Open a feature request on [GitHub Issues](https://github.com/Hawksight-AI/semantica/issues) with:

- Use case description
- Proposed solution
- Benefits to the community

---

!!! question "Still have questions?"
    Check the [API Reference](reference/core.md), browse the [Cookbook](cookbook.md), or [ask on GitHub Issues](https://github.com/Hawksight-AI/semantica/issues/new)
