# Modules & Architecture

Semantica is built with a modular architecture, designed to be flexible, extensible, and scalable. This guide provides a comprehensive overview of all modules, their responsibilities, key features, and components.

!!! info "About This Guide"
    This guide covers all 20+ core modules in Semantica, organized by their functional layer. Each module can be used independently or combined into powerful pipelines.

---

## Module Overview

Semantica's modules are organized into six logical layers:

| Layer | Modules | Description |
| :--- | :--- | :--- |
| **Input Layer** | [Ingest](#ingest-module), [Parse](#parse-module), [Split](#split-module), [Normalize](#normalize-module) | Data ingestion, parsing, chunking, and cleaning |
| **Core Processing** | [Semantic Extract](#semantic-extract-module), [Knowledge Graph](#knowledge-graph-kg-module), [Ontology](#ontology-module), [Reasoning](#reasoning-module) | Entity extraction, graph construction, inference |
| **Storage** | [Embeddings](#embeddings-module), [Vector Store](#vector-store-module), [Graph Store](#graph-store-module), [Triplet Store](#triplet-store-module) | Vector, graph, and triplet persistence |
| **Quality Assurance** | [Deduplication](#deduplication-module), [Conflicts](#conflicts-module) | Data quality and consistency |
| **Context & Memory** | [Context](#context-module), [Seed](#seed-module) | Agent memory and foundation data |
| **Output & Orchestration** | [Export](#export-module), [Visualization](#visualization-module), [Pipeline](#pipeline-module) | Export, visualization, and workflow management |

---

## Input Layer

These modules handle data ingestion, parsing, chunking, and preparation.

---

### Ingest Module

!!! abstract "Purpose"
    The entry point for data ingestion. Connects to various data sources including files, web, databases, and MCP servers.

**Key Features:**

- 50+ file format support (PDF, DOCX, HTML, JSON, CSV, etc.)
- Web scraping with JavaScript rendering
- Database integration (SQL, NoSQL)
- Real-time streaming support
- MCP (Model Context Protocol) server integration
- Batch processing capabilities
- Metadata extraction and preservation

**Components:**

- `FileIngestor` — Read files (PDF, DOCX, HTML, JSON, CSV, etc.)
- `WebIngestor` — Scrape and ingest web pages
- `FeedIngestor` — Process RSS/Atom feeds
- `StreamIngestor` — Real-time data streaming
- `DBIngestor` — Database queries and ingestion
- `EmailIngestor` — Process email messages
- `RepoIngestor` — Git repository analysis
- `MCPIngestor` — Connect to MCP servers for resource and tool-based ingestion

**Quick Example:**

```python
from semantica.ingest import FileIngestor, WebIngestor

# Ingest local files
file_ingestor = FileIngestor()
documents = file_ingestor.ingest("data/", recursive=True)

# Ingest web content
web_ingestor = WebIngestor()
web_docs = web_ingestor.ingest("https://example.com")
```

**API Reference**: [Ingest Module](reference/ingest.md)

---

### Parse Module

!!! abstract "Purpose"
    Extracts raw text and metadata from ingested documents. Supports OCR, table extraction, and structured data parsing.

**Key Features:**

- 50+ file format support
- OCR for images and scanned documents
- Table extraction from PDFs and spreadsheets
- Metadata preservation
- Automatic format detection
- Structured data parsing (JSON, CSV, XML)
- Code file parsing with syntax awareness

**Components:**

- `DocumentParser` — Main parser orchestrator
- `PDFParser` — Extract text, tables, images from PDFs
- `DOCXParser` — Parse Word documents
- `HTMLParser` — Extract content from HTML
- `JSONParser` — Parse structured JSON data
- `ExcelParser` — Process spreadsheets
- `ImageParser` — OCR and image analysis
- `CodeParser` — Parse source code files

**Quick Example:**

```python
from semantica.parse import DocumentParser

parser = DocumentParser(ocr_enabled=True)
parsed_docs = parser.parse(documents)

for doc in parsed_docs:
    print(f"Content: {doc.content[:100]}...")
    print(f"Tables found: {len(doc.tables)}")
```

**API Reference**: [Parse Module](reference/parse.md)

---

### Split Module

!!! abstract "Purpose"
    Comprehensive document chunking and splitting for optimal processing. Provides 15+ splitting methods including KG-aware chunking.

**Key Features:**

- Multiple standard splitting methods (recursive, token, sentence, paragraph)
- Semantic-based chunking using NLP and embeddings
- Entity-aware chunking for GraphRAG workflows
- Relation-aware chunking for KG preservation
- Graph-based and ontology-aware chunking
- Hierarchical multi-level chunking
- Community detection-based splitting
- Sliding window chunking with overlap
- Table-specific chunking
- Chunk validation and quality assessment
- Provenance tracking for data lineage

**Components:**

- `TextSplitter` — Unified text splitter with method parameter
- `SemanticChunker` — Semantic-based chunking coordinator
- `StructuralChunker` — Structure-aware chunking (headings, lists)
- `SlidingWindowChunker` — Fixed-size sliding window chunking
- `TableChunker` — Table-specific chunking
- `EntityAwareChunker` — Entity boundary-preserving chunker
- `RelationAwareChunker` — Triple-preserving chunker
- `GraphBasedChunker` — Graph structure-based chunker
- `OntologyAwareChunker` — Ontology concept-based chunker
- `HierarchicalChunker` — Multi-level hierarchical chunker
- `ChunkValidator` — Chunk quality validation
- `ProvenanceTracker` — Chunk provenance tracking

**Supported Methods:**

| Category | Methods |
| :--- | :--- |
| **Standard** | recursive, token, sentence, paragraph, character, word, semantic_transformer, llm |
| **KG/Ontology** | entity_aware, relation_aware, graph_based, ontology_aware, hierarchical, community_detection, centrality_based |

**Quick Example:**

```python
from semantica.split import TextSplitter

# Standard recursive splitting
splitter = TextSplitter(method="recursive", chunk_size=1000, chunk_overlap=200)
chunks = splitter.split(text)

# Entity-aware for GraphRAG
splitter = TextSplitter(method="entity_aware", ner_method="llm", chunk_size=1000)
chunks = splitter.split(text)
```

---

### Normalize Module

!!! abstract "Purpose"
    Cleans, standardizes, and prepares text for semantic extraction. Handles encoding, entity names, dates, and numbers.

**Key Features:**

- Text cleaning and noise removal
- Encoding normalization (Unicode handling)
- Entity name standardization
- Date and number formatting
- Language detection
- Whitespace normalization
- Special character handling

**Components:**

- `TextNormalizer` — Main normalization orchestrator
- `TextCleaner` — Remove noise, fix encoding
- `DataCleaner` — Clean structured data
- `EntityNormalizer` — Normalize entity names
- `DateNormalizer` — Standardize date formats
- `NumberNormalizer` — Normalize numeric values
- `LanguageDetector` — Detect document language
- `EncodingHandler` — Handle character encoding

**Quick Example:**

```python
from semantica.normalize import TextNormalizer

normalizer = TextNormalizer(
    normalize_entities=True,
    normalize_dates=True,
    detect_language=True
)
normalized = normalizer.normalize(parsed_docs)

for doc in normalized:
    print(f"Language: {doc.language}")
```

**API Reference**: [Normalize Module](reference/normalize.md)

---

## Core Processing Layer

These modules form the intelligence core—extracting meaning, building relationships, and inferring knowledge.

---

### Semantic Extract Module

!!! abstract "Purpose"
    The brain of Semantica. Uses LLMs and NLP to extract entities, relationships, and semantic meaning from text.

**Key Features:**

- Multiple NER methods (rule-based, ML, LLM)
- Relationship extraction with confidence scoring
- Event extraction
- Custom entity type support
- Multi-language support
- Semantic network extraction
- Coreference resolution

**Components:**

- `NERExtractor` — Named Entity Recognition
- `RelationExtractor` — Extract relationships between entities
- `SemanticAnalyzer` — Deep semantic analysis
- `SemanticNetworkExtractor` — Extract semantic networks
- `EventExtractor` — Extract events from text
- `CoreferenceResolver` — Resolve entity coreferences

**Quick Example:**

```python
from semantica.semantic_extract import NERExtractor, RelationExtractor

# Extract entities
extractor = NERExtractor(method="llm", model="gpt-4")
entities = extractor.extract(normalized_docs)

# Extract relationships
relation_extractor = RelationExtractor()
relationships = relation_extractor.extract(normalized_docs, entities=entities)

for rel in relationships[:5]:
    print(f"{rel.subject.text} --[{rel.predicate}]--> {rel.object.text}")
```

**API Reference**: [Semantic Extract Module](reference/semantic_extract.md)

---

### Knowledge Graph (KG) Module

!!! abstract "Purpose"
    Constructs and manages knowledge graphs from extracted entities and relationships. Supports multiple backends and advanced analytics.

**Key Features:**

- Graph construction from entities/relationships
- Multiple backend support (NetworkX, Neo4j)
- Temporal graph support
- Graph analytics and metrics
- Entity resolution and deduplication
- Community detection
- Centrality calculations
- Path finding algorithms
- Graph validation

**Components:**

- `GraphBuilder` — Construct knowledge graphs
- `GraphAnalyzer` — Analyze graph structure and properties
- `GraphValidator` — Validate graph quality and consistency
- `EntityResolver` — Resolve entity conflicts and duplicates
- `ConflictDetector` — Detect conflicting information
- `CentralityCalculator` — Calculate node importance metrics
- `CommunityDetector` — Detect communities in graphs
- `ConnectivityAnalyzer` — Analyze graph connectivity
- `TemporalQuery` — Query temporal knowledge graphs
- `Deduplicator` — Remove duplicate entities/relationships

**Quick Example:**

```python
from semantica.kg import GraphBuilder, GraphAnalyzer

# Build graph
builder = GraphBuilder(backend="networkx", temporal=True)
kg = builder.build(entities, relationships)

# Analyze graph
analyzer = GraphAnalyzer()
metrics = analyzer.analyze(kg)

print(f"Nodes: {metrics['nodes']}, Edges: {metrics['edges']}")
print(f"Density: {metrics['density']:.3f}")
```

**API Reference**: [Knowledge Graph Module](reference/kg.md)

---

### Ontology Module

!!! abstract "Purpose"
    Defines schema and structure for your knowledge domain. Generates and validates ontologies with OWL/RDF export.

**Key Features:**

- Automatic ontology generation (6-stage pipeline)
- OWL/RDF/Turtle export
- Class and property inference
- Ontology validation
- Symbolic reasoning (HermiT, Pellet)
- Version management
- SHACL constraint support
- Ontology merging and alignment

**Components:**

- `OntologyGenerator` — Generate ontologies from knowledge graphs
- `OntologyValidator` — Validate ontology structure
- `OWLGenerator` — Generate OWL format ontologies
- `PropertyGenerator` — Generate ontology properties
- `ClassInferrer` — Infer ontology classes
- `OntologyMerger` — Merge multiple ontologies
- `ReasonerInterface` — Interface with symbolic reasoners

**Quick Example:**

```python
from semantica.ontology import OntologyGenerator

generator = OntologyGenerator(base_uri="https://example.org/ontology/")
ontology = generator.generate_from_graph(kg)

# Export to OWL
owl_content = generator.export_owl(ontology, format="turtle")
print(f"Generated {len(owl_content)} lines of OWL")
```

**API Reference**: [Ontology Module](reference/ontology.md)

---

### Reasoning Module

!!! abstract "Purpose"
    Infers new facts and validates existing knowledge using logical rules. Supports forward/backward chaining and explanation generation.

**Key Features:**

- Forward and backward chaining
- Rule-based inference
- Deductive and abductive reasoning
- Explanation generation
- RETE algorithm support
- Custom rule definition
- Conflict detection in inferences
- Temporal reasoning

**Components:**

- `InferenceEngine` — Main inference orchestrator
- `RuleManager` — Manage inference rules
- `DeductiveReasoner` — Deductive reasoning
- `AbductiveReasoner` — Abductive reasoning
- `ExplanationGenerator` — Generate explanations for inferences
- `RETEEngine` — RETE algorithm for rule matching

**Quick Example:**

```python
from semantica.reasoning import InferenceEngine, RuleManager

inference_engine = InferenceEngine()
rule_manager = RuleManager()

rules = [
    "IF Person worksFor Company AND Company locatedIn City THEN Person livesIn City",
    "IF Person hasFriend Person2 AND Person2 hasFriend Person3 THEN Person knows Person3"
]
rule_manager.add_rules(rules)

new_facts = inference_engine.forward_chain(kg, rule_manager)
print(f"Inferred {len(new_facts)} new facts")
```

**API Reference**: [Reasoning Module](reference/reasoning.md)

---

## Storage Layer

These modules handle persistence and retrieval of vectors, graphs, and triples.

---

### Embeddings Module

!!! abstract "Purpose"
    Generates vector embeddings for text, images, and audio. Supports multiple providers with caching and batch processing.

**Key Features:**

- Multiple provider support (OpenAI, Cohere, HuggingFace, Sentence Transformers)
- Text, image, and audio embeddings
- Multimodal embeddings
- Batch processing
- Caching support
- Custom models
- Similarity calculations

**Components:**

- `EmbeddingGenerator` — Main embedding orchestrator
- `TextEmbedder` — Generate text embeddings
- `ImageEmbedder` — Generate image embeddings
- `AudioEmbedder` — Generate audio embeddings
- `MultimodalEmbedder` — Combine multiple modalities
- `EmbeddingOptimizer` — Optimize embedding quality
- `ProviderAdapters` — Support for OpenAI, Cohere, etc.

**Quick Example:**

```python
from semantica.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator(
    provider="openai",
    model="text-embedding-3-small"
)
embeddings = generator.generate(documents)

# Calculate similarity
similarity = generator.similarity(embeddings[0], embeddings[1])
print(f"Similarity: {similarity:.3f}")
```

**API Reference**: [Embeddings Module](reference/embeddings.md)

---

### Vector Store Module

!!! abstract "Purpose"
    Manages storage and retrieval of high-dimensional vectors. Supports hybrid search combining vector and keyword search.

**Key Features:**

- Multiple backend support (FAISS, Pinecone, Weaviate, Qdrant, Milvus)
- Hybrid search (vector + keyword)
- Metadata filtering
- Batch operations
- Similarity search with scoring
- Index management
- Namespace support

**Components:**

- `VectorStore` — Main vector store interface
- `FAISSAdapter` — FAISS integration
- `PineconeAdapter` — Pinecone integration
- `WeaviateAdapter` — Weaviate integration
- `HybridSearch` — Combine vector and keyword search
- `VectorRetriever` — Retrieve relevant vectors

**Quick Example:**

```python
from semantica.vector_store import VectorStore, HybridSearch

vector_store = VectorStore(backend="faiss")
vector_store.store(embeddings, documents, metadata)

# Hybrid search
hybrid_search = HybridSearch(vector_store)
results = hybrid_search.search(
    query="machine learning",
    top_k=10,
    filters={"category": "AI"}
)
```

**API Reference**: [Vector Store Module](reference/vector_store.md)

---

### Graph Store Module

!!! abstract "Purpose"
    Integration with property graph databases for storing and querying knowledge graphs.

**Key Features:**

- Multiple backend support (Neo4j, FalkorDB)
- Cypher query language
- Graph algorithms and analytics
- Transaction support
- Index management
- High-performance queries
- Batch operations

**Components:**

- `GraphStore` — Main graph store interface
- `Neo4jAdapter` — Neo4j database integration
- `FalkorDBAdapter` — FalkorDB (Redis-based) integration
- `NodeManager` — Node CRUD operations
- `RelationshipManager` — Relationship CRUD operations
- `QueryEngine` — Cypher query execution
- `GraphAnalytics` — Graph algorithms and analytics

**Quick Example:**

```python
from semantica.graph_store import GraphStore

store = GraphStore(backend="neo4j", uri="bolt://localhost:7687")
store.connect()

# Create nodes and relationships
alice = store.create_node(
    labels=["Person"],
    properties={"name": "Alice", "age": 30}
)
bob = store.create_node(
    labels=["Person"],
    properties={"name": "Bob", "age": 25}
)
store.create_relationship(
    start_node_id=alice["id"],
    end_node_id=bob["id"],
    rel_type="KNOWS",
    properties={"since": 2020}
)

# Query with Cypher
results = store.execute_query("MATCH (p:Person) RETURN p.name")
```

**API Reference**: [Graph Store Module](reference/graph_store.md)

---

### Triplet Store Module

!!! abstract "Purpose"
    RDF triplet store integration for semantic web applications. Supports SPARQL queries and multiple backends.

**Key Features:**

- Multi-backend support (Blazegraph, Jena, RDF4J, Virtuoso)
- CRUD operations for RDF triplets
- SPARQL query execution and optimization
- Bulk data loading with progress tracking
- Query caching and optimization
- Transaction support
- Store adapter pattern

**Components:**

- `TripletManager` — Main triplet store management coordinator
- `QueryEngine` — SPARQL query execution and optimization
- `BulkLoader` — High-volume data loading with progress tracking
- `BlazegraphAdapter` — Blazegraph integration
- `JenaAdapter` — Apache Jena integration
- `RDF4JAdapter` — Eclipse RDF4J integration
- `VirtuosoAdapter` — Virtuoso RDF store integration
- `QueryPlan` — Query execution plan dataclass
- `LoadProgress` — Bulk loading progress tracking

**Algorithms:**

| Category | Algorithms |
| :--- | :--- |
| **Query Optimization** | Cost estimation, query rewriting, LIMIT injection |
| **Caching** | MD5-based cache keys, LRU eviction |
| **Bulk Loading** | Batch processing, retry with exponential backoff |

**Quick Example:**

```python
from semantica.triplet_store import TripletManager, execute_query

manager = TripletManager()
store = manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")

# Add triple
result = manager.add_triple({
    "subject": "http://example.org/Alice",
    "predicate": "http://example.org/knows",
    "object": "http://example.org/Bob"
}, store_id="main")

# Execute SPARQL
query_result = execute_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10", store)
```

**API Reference**: [Triplet Store Module](reference/triplet_store.md)

---

## Quality Assurance Layer

These modules ensure data quality, handle duplicates, and resolve conflicts.

---

### Deduplication Module

!!! abstract "Purpose"
    Comprehensive entity deduplication and merging. Detects duplicates using multiple similarity methods and merges them intelligently.

**Key Features:**

- Multiple similarity methods (exact, Levenshtein, Jaro-Winkler, cosine, embedding)
- Duplicate detection with confidence scoring
- Entity merging with configurable strategies
- Cluster-based batch deduplication
- Provenance preservation during merges
- Relationship preservation
- Incremental processing support

**Components:**

- `DuplicateDetector` — Detects duplicate entities using similarity metrics
- `EntityMerger` — Merges duplicate entities using configurable strategies
- `SimilarityCalculator` — Multi-factor similarity between entities
- `MergeStrategyManager` — Manages merge strategies and conflict resolution
- `ClusterBuilder` — Builds clusters for batch deduplication

**Merge Strategies:**

| Strategy | Description |
| :--- | :--- |
| `keep_first` | Preserve first entity, merge others |
| `keep_last` | Preserve last entity, merge others |
| `keep_most_complete` | Preserve entity with most properties |
| `keep_highest_confidence` | Preserve entity with highest confidence |
| `merge_all` | Combine all properties and relationships |

**Quick Example:**

```python
from semantica.deduplication import DuplicateDetector, EntityMerger, MergeStrategy

# Detect duplicates
detector = DuplicateDetector(similarity_threshold=0.8)
duplicate_groups = detector.detect_duplicate_groups(entities)

# Merge duplicates
merger = EntityMerger(preserve_provenance=True)
merge_operations = merger.merge_duplicates(
    entities,
    strategy=MergeStrategy.KEEP_MOST_COMPLETE
)

merged_entities = [op.merged_entity for op in merge_operations]
print(f"Reduced from {len(entities)} to {len(merged_entities)} entities")
```

---

### Conflicts Module

!!! abstract "Purpose"
    Detects and resolves conflicts from multiple data sources. Provides investigation guides and source tracking.

**Key Features:**

- Multi-source conflict detection (value, type, relationship, temporal, logical)
- Source tracking and provenance management
- Conflict analysis and pattern identification
- Multiple resolution strategies (voting, credibility-weighted, recency)
- Investigation guide generation
- Source credibility scoring
- Conflict reporting and statistics

**Components:**

- `ConflictDetector` — Detects conflicts from multiple sources
- `ConflictResolver` — Resolves conflicts using various strategies
- `ConflictAnalyzer` — Analyzes conflict patterns and trends
- `SourceTracker` — Tracks source information and provenance
- `InvestigationGuideGenerator` — Generates investigation guides

**Resolution Strategies:**

| Strategy | Algorithm |
| :--- | :--- |
| **Voting** | Majority value selection using frequency counting |
| **Credibility Weighted** | Weighted average using source credibility scores |
| **Temporal Selection** | Newest/oldest value based on timestamps |
| **Confidence Selection** | Maximum confidence value selection |

**Quick Example:**

```python
from semantica.conflicts import detect_and_resolve, ConflictDetector

# Using convenience function
conflicts, results = detect_and_resolve(
    entities,
    property_name="name",
    resolution_strategy="voting"
)

# Using classes directly
detector = ConflictDetector()
conflicts = detector.detect_value_conflicts(entities, "name")
```

---

### KG Quality Assurance Module

!!! abstract "Purpose"
    Comprehensive quality assessment, validation, and automated fixes for knowledge graphs.

**Key Features:**

- Quality metrics calculation (overall, completeness, consistency)
- Consistency checking (logical, temporal, hierarchical)
- Completeness validation (entity, relationship, property)
- Automated fixes (duplicates, inconsistencies, missing properties)
- Quality reporting with issue tracking
- Validation engine with rules and constraints
- Improvement suggestions

**Components:**

- `KGQualityAssessor` — Overall quality assessment coordinator
- `ConsistencyChecker` — Consistency validation engine
- `CompletenessValidator` — Completeness validation engine
- `QualityMetrics` — Quality metrics calculator
- `ValidationEngine` — Rule and constraint validation
- `RuleValidator` — Rule-based validation
- `ConstraintValidator` — Constraint-based validation
- `QualityReporter` — Quality report generation
- `IssueTracker` — Issue tracking and management
- `ImprovementSuggestions` — Improvement suggestions generator
- `AutomatedFixer` — Automated issue fixing
- `AutoMerger` — Automatic merging of duplicates
- `AutoResolver` — Automatic conflict resolution

Note: The KG quality assessment module has been temporarily removed and will be reintroduced in a future release.

---

## Context & Memory Layer

These modules provide context engineering for agents and foundation data management.

---

### Context Module

!!! abstract "Purpose"
    Context engineering infrastructure for agents. Formalizes context as a graph of connections with RAG-enhanced memory.

**Key Features:**

- Context graph construction from entities, relationships, and conversations
- Agent memory management with RAG integration
- Entity linking across sources with URI assignment
- Hybrid context retrieval (vector + graph + memory)
- Conversation history management
- Context accumulation and synthesis
- Graph-based context traversal

**Components:**

- `ContextGraphBuilder` — Builds context graphs from various sources
- `ContextNode` — Context graph node data structure
- `ContextEdge` — Context graph edge data structure
- `AgentMemory` — Manages persistent agent memory with RAG
- `MemoryItem` — Memory item data structure
- `EntityLinker` — Links entities across sources with URIs
- `ContextRetriever` — Retrieves relevant context from multiple sources

**Algorithms:**

| Category | Algorithms |
| :--- | :--- |
| **Graph Construction** | BFS/DFS traversal, type-based indexing |
| **Memory Management** | Vector embedding, similarity search, retention policies |
| **Context Retrieval** | Vector similarity, multi-hop graph expansion, hybrid scoring |
| **Entity Linking** | Hash-based URI generation, text similarity matching |

**Quick Example:**

```python
from semantica.context import build_context, ContextGraphBuilder, AgentMemory

# Using convenience function
result = build_context(
    entities=entities,
    relationships=relationships,
    vector_store=vs,
    knowledge_graph=kg
)

# Using classes directly
builder = ContextGraphBuilder()
graph = builder.build_from_entities_and_relationships(entities, relationships)

memory = AgentMemory(vector_store=vs, knowledge_graph=kg)
memory_id = memory.store("User asked about Python", metadata={"type": "conversation"})
results = memory.retrieve("Python", max_results=5)
```

---

### Seed Module

!!! abstract "Purpose"
    Seed data management for initial knowledge graph construction. Builds on verified knowledge from multiple sources.

**Key Features:**

- Multi-source seed data loading (CSV, JSON, Database, API)
- Foundation graph creation from seed data
- Seed data quality validation
- Integration with extracted data using configurable merge strategies
- Version management for seed sources
- Export capabilities (JSON, CSV)
- Schema template validation

**Components:**

- `SeedDataManager` — Main coordinator for seed data operations
- `SeedDataSource` — Seed data source definition
- `SeedData` — Seed data container

**Merge Strategies:**

| Strategy | Description |
| :--- | :--- |
| `seed_first` | Seed data takes precedence, extracted fills gaps |
| `extracted_first` | Extracted data takes precedence, seed fills gaps |
| `merge` | Property merging, seed takes precedence for conflicts |

**Quick Example:**

```python
from semantica.seed import SeedDataManager

manager = SeedDataManager()
manager.register_source("entities", "json", "data/entities.json")
foundation = manager.create_foundation_graph()
validation = manager.validate_quality(foundation)
```

---

## Output & Orchestration Layer

These modules handle export, visualization, and workflow management.

---

### Export Module

!!! abstract "Purpose"
    Export knowledge graphs and data to various formats for use in external tools.

**Key Features:**

- Multiple export formats (JSON, RDF, CSV, OWL, GraphML, GEXF)
- Custom export formats
- Batch export
- Metadata preservation
- Streaming export for large graphs
- Vector export support

**Components:**

- `JSONExporter` — Export to JSON
- `RDFExporter` — Export to RDF/XML
- `CSVExporter` — Export to CSV
- `GraphExporter` — Export to graph formats (GraphML, GEXF)
- `OWLExporter` — Export to OWL
- `VectorExporter` — Export vectors

**Quick Example:**

```python
from semantica.export import JSONExporter, RDFExporter, CSVExporter

# Export to multiple formats
JSONExporter().export(kg, "output.json")
RDFExporter().export(kg, "output.rdf")
CSVExporter().export(kg, "output.csv")
```

**API Reference**: [Export Module](reference/export.md)

---

### Visualization Module

!!! abstract "Purpose"
    Visual exploration of knowledge graphs, embeddings, and analytics data.

**Key Features:**

- Interactive graph visualization
- Embedding visualization (t-SNE, PCA, UMAP)
- Quality metrics visualization
- Temporal data visualization
- Ontology visualization
- Multiple output formats (HTML, PNG, SVG)
- Custom styling

**Components:**

- `KGVisualizer` — Visualize knowledge graphs
- `EmbeddingVisualizer` — Visualize embeddings (t-SNE, PCA, UMAP)
- `QualityVisualizer` — Visualize quality metrics
- `AnalyticsVisualizer` — Visualize graph analytics
- `TemporalVisualizer` — Visualize temporal data
- `OntologyVisualizer` — Visualize ontology structure
- `SemanticNetworkVisualizer` — Visualize semantic networks

**Quick Example:**

```python
from semantica.visualization import KGVisualizer, EmbeddingVisualizer

# Visualize knowledge graph
KGVisualizer().visualize(kg, output_format="html", output_path="graph.html")

# Visualize embeddings
EmbeddingVisualizer().visualize(embeddings, method="tsne", output_path="embeddings.png")
```

**API Reference**: [Visualization Module](reference/visualization.md)

---

### Pipeline Module

!!! abstract "Purpose"
    Orchestrates workflows, connecting modules into robust, executable pipelines.

**Key Features:**

- Pipeline construction DSL
- Parallel execution
- Error handling and recovery
- Resource scheduling
- Pipeline validation
- Monitoring and logging
- Checkpoint support

**Components:**

- `PipelineBuilder` — Build complex pipelines
- `ExecutionEngine` — Execute pipelines
- `FailureHandler` — Handle pipeline failures
- `ParallelismManager` — Enable parallel processing
- `ResourceScheduler` — Schedule resources
- `PipelineValidator` — Validate pipeline configuration

**Quick Example:**

```python
from semantica.pipeline import PipelineBuilder
from semantica.ingest import FileIngestor
from semantica.parse import DocumentParser
from semantica.semantic_extract import NERExtractor

builder = PipelineBuilder()
pipeline = builder \
    .add_step("ingest", FileIngestor()) \
    .add_step("parse", DocumentParser()) \
    .add_step("extract", NERExtractor()) \
    .build()

result = pipeline.execute(sources=["data/"], parallel=True)
```

**API Reference**: [Pipeline Module](reference/pipeline.md)

---

## Integration Patterns

### Pattern 1: Complete Knowledge Graph Pipeline

```python
from semantica import Semantica

semantica = Semantica()
result = semantica.build_knowledge_base(
    sources=["documents/"],
    embeddings=True,
    graph=True,
    normalize=True
)
```

### Pattern 2: Custom Pipeline with Module Selection

```python
from semantica.ingest import FileIngestor
from semantica.parse import DocumentParser
from semantica.split import TextSplitter
from semantica.normalize import TextNormalizer
from semantica.semantic_extract import NERExtractor, RelationExtractor
from semantica.kg import GraphBuilder
from semantica.deduplication import DuplicateDetector, EntityMerger, MergeStrategy

# Ingest and parse
documents = FileIngestor().ingest("data/")
parsed = DocumentParser().parse(documents)

# Split and normalize
chunks = TextSplitter(method="entity_aware").split(parsed)
normalized = TextNormalizer().normalize(chunks)

# Extract and build
entities = NERExtractor().extract(normalized)
relationships = RelationExtractor().extract(normalized, entities)
kg = GraphBuilder().build(entities, relationships)

# Quality assurance - deduplicate entities
detector = DuplicateDetector(similarity_threshold=0.8)
duplicate_groups = detector.detect_duplicate_groups(entities)
merger = EntityMerger()
merge_operations = merger.merge_duplicates(entities, strategy=MergeStrategy.KEEP_MOST_COMPLETE)
deduplicated = [op.merged_entity for op in merge_operations]
```

### Pattern 3: GraphRAG with Hybrid Search

```python
from semantica import Semantica
from semantica.vector_store import VectorStore, HybridSearch
from semantica.context import AgentMemory

semantica = Semantica()
result = semantica.build_knowledge_base(["documents/"])

vector_store = VectorStore()
vector_store.store(result["embeddings"], result["documents"])

# Agent memory with RAG
memory = AgentMemory(vector_store=vector_store, knowledge_graph=result["knowledge_graph"])
memory.store("User query about AI", metadata={"type": "query"})

# Hybrid search
hybrid_search = HybridSearch(vector_store)
results = hybrid_search.search(
    query="What is the relationship between X and Y?",
    graph=result["knowledge_graph"],
    top_k=10
)
```

### Pattern 4: Temporal Graph with Reasoning

```python
from semantica.kg import GraphBuilder
from semantica.reasoning import InferenceEngine, RuleManager

# Build temporal graph
builder = GraphBuilder(temporal=True)
kg = builder.build(entities, relationships)

# Add reasoning
inference_engine = InferenceEngine()
rule_manager = RuleManager()
rule_manager.add_rules(["IF A THEN B"])

new_facts = inference_engine.forward_chain(kg, rule_manager)
```

---

## Quick Reference: All Modules

| Module | Import | Main Class | Purpose |
| :--- | :--- | :--- | :--- |
| **Ingest** | `semantica.ingest` | `FileIngestor` | Data ingestion |
| **Parse** | `semantica.parse` | `DocumentParser` | Document parsing |
| **Split** | `semantica.split` | `TextSplitter` | Text chunking |
| **Normalize** | `semantica.normalize` | `TextNormalizer` | Data cleaning |
| **Semantic Extract** | `semantica.semantic_extract` | `NERExtractor` | Entity extraction |
| **KG** | `semantica.kg` | `GraphBuilder` | Graph construction |
| **Ontology** | `semantica.ontology` | `OntologyGenerator` | Ontology generation |
| **Reasoning** | `semantica.reasoning` | `InferenceEngine` | Logical inference |
| **Embeddings** | `semantica.embeddings` | `EmbeddingGenerator` | Vector generation |
| **Vector Store** | `semantica.vector_store` | `VectorStore` | Vector storage |
| **Graph Store** | `semantica.graph_store` | `GraphStore` | Graph database |
| **Triplet Store** | `semantica.triplet_store` | `TripletManager` | RDF storage |
| **Deduplication** | `semantica.deduplication` | `DuplicateDetector` | Duplicate removal |
| **Conflicts** | `semantica.conflicts` | `ConflictDetector` | Conflict resolution |
| **Context** | `semantica.context` | `AgentMemory` | Agent context |
| **Seed** | `semantica.seed` | `SeedDataManager` | Foundation data |
| **Export** | `semantica.export` | `JSONExporter` | Data export |
| **Visualization** | `semantica.visualization` | `KGVisualizer` | Visualization |
| **Pipeline** | `semantica.pipeline` | `PipelineBuilder` | Workflow orchestration |

---

## Next Steps

- **[Core Concepts](concepts.md)** — Understand the fundamental concepts
- **[Use Cases](use-cases.md)** — See real-world applications
- **[Examples](examples.md)** — Practical code examples
- **[API Reference](reference/core.md)** — Detailed API documentation

---

!!! info "Contribute"
    Found an issue or want to improve this guide? [Contribute on GitHub](https://github.com/Hawksight-AI/semantica)

**Last Updated**: 2024
