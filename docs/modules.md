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
| **Context & Memory** | [Context](#context-module), [Seed](#seed-module), [LLM Providers](#llm-providers-module) | Agent memory, foundation data, and LLM integration |
| **Output & Orchestration** | [Export](#export-module), [Visualization](#visualization-module), [Pipeline](#pipeline-module) | Export, visualization, and workflow management |

---

## Input Layer

These modules handle data ingestion, parsing, chunking, and preparation.

---

### Ingest Module

!!! abstract "Purpose"
    The entry point for data ingestion. Connects to various data sources including files, web, databases, and MCP servers.

**Key Features:**

- Multiple file format support (PDF, DOCX, HTML, JSON, CSV, etc.)
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

- Multiple file format support
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
- `RelationAwareChunker` — Triplet-preserving chunker
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
- `EntityResolver` — Resolve entity conflicts and duplicates
- `ConflictDetector` — Detect conflicting information
- `CentralityCalculator` — Calculate node importance metrics
- `CommunityDetector` — Detect community structure
- `ConnectivityAnalyzer` — Analyze graph connectivity
- `SeedManager` — Manage seed data for KG initialization
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
- `OntologyValidator` — Validate ontologies
- `OWLGenerator` — Generate OWL format ontologies
- `PropertyGenerator` — Generate ontology properties
- `ClassInferrer` — Infer ontology classes
- `OntologyMerger` — Merge multiple ontologies
- `ReasonerInterface` — Interface with symbolic reasoners

**Quick Example:**

```python
from semantica.ontology import OntologyEngine

# Initialize engine
engine = OntologyEngine(base_uri="https://example.org/ontology/")

# Generate ontology from data
ontology = engine.from_data({
    "entities": [...],
    "relationships": [...]
})

# Validate ontology
result = engine.validate(ontology)
if result.valid:
    print("Ontology is valid!")

# Export to OWL
owl_content = engine.to_owl(ontology, format="turtle")
print(f"Generated {len(owl_content)} lines of OWL")
```

**API Reference**: [Ontology Module](reference/ontology.md)

---

### Reasoning Module

!!! abstract "Purpose"
    Infers new facts and validates existing knowledge using logical rules. Supports forward-chaining, high-performance pattern matching, and explanation generation.

**Key Features:**

- Forward-chaining inference engine
- IF-THEN rule support with variable substitution
- High-performance Rete algorithm for large-scale rule matching
- Natural language explanation generation for inferred facts
- SPARQL query expansion for RDF graphs
- Conflict detection in inferences
- Priority-based rule execution

**Components:**

- `Reasoner` — High-level facade for all reasoning tasks
- `ReteEngine` — High-performance pattern matching (Rete algorithm)
- `ExplanationGenerator` — Generate justifications for inferred facts
- `SPARQLReasoner` — Query expansion for triplet stores

**Quick Example:**

```python
from semantica.reasoning import Reasoner

reasoner = Reasoner()

# Add rules and facts
reasoner.add_rule("IF Person(?x) AND Parent(?x, ?y) THEN ParentOfPerson(?x, ?y)")
reasoner.add_fact("Person(Alice)")
reasoner.add_fact("Parent(Alice, Bob)")

# Perform inference
inferred = reasoner.infer_facts(["Person(Alice)", "Parent(Alice, Bob)"])
# Inferred: ["ParentOfPerson(Alice, Bob)"]
```

**API Reference**: [Reasoning Module](reference/reasoning.md)

---

## Storage Layer

These modules handle persistence and retrieval of vectors, graphs, and triplets.

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
- `ProviderStores` — Support for OpenAI, Cohere, etc.

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

- Multiple backend support (FAISS, Weaviate, Qdrant, Milvus)
- Hybrid search (vector + keyword)
- Metadata filtering
- Batch operations
- Similarity search with scoring
- Index management
- Namespace support

**Components:**

- `VectorStore` — Main vector store interface
- `FAISSStore` — FAISS integration
- `WeaviateStore` — Weaviate integration
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
- `Neo4jStore` — Neo4j database integration
- `FalkorDBStore` — FalkorDB (Redis-based) integration
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

- Multi-backend support (Blazegraph, Jena, RDF4J)
- CRUD operations for RDF triplets
- SPARQL query execution and optimization
- Bulk data loading with progress tracking
- Query caching and optimization
- Transaction support
- Store backend pattern

**Components:**

- `TripletStore` — Main triplet store interface
- `QueryEngine` — SPARQL query execution and optimization
- `BulkLoader` — High-volume data loading with progress tracking
- `BlazegraphStore` — Blazegraph integration
- `JenaStore` — Apache Jena integration
- `RDF4JStore` — Eclipse RDF4J integration
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
from semantica.triplet_store import TripletStore

store = TripletStore(backend="blazegraph", endpoint="http://localhost:9999/blazegraph")

# Add triplet
result = store.add_triplet({
    "subject": "http://example.org/Alice",
    "predicate": "http://example.org/knows",
    "object": "http://example.org/Bob"
})

# Execute SPARQL
query_result = store.execute_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10")
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
- **Advanced String Matching**: Jaro-Winkler by default for better company/person name resolution
- **Smart Property Handling**: Neutral scoring for disjoint properties to prevent false negatives
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
| `"keep_first"` | Preserve first entity, merge others |
| `"keep_last"` | Preserve last entity, merge others |
| `"keep_most_complete"` | Preserve entity with most properties |
| `"keep_highest_confidence"` | Preserve entity with highest confidence |
| `"merge_all"` | Combine all properties and relationships |

**Quick Example:**

```python
from semantica.deduplication import DuplicateDetector, EntityMerger

# Detect duplicates
detector = DuplicateDetector(similarity_threshold=0.8)
duplicate_groups = detector.detect_duplicate_groups(entities)

# Merge duplicates
merger = EntityMerger(preserve_provenance=True)
merge_operations = merger.merge_duplicates(
    entities,
    strategy="keep_most_complete"
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
from semantica.conflicts import ConflictDetector, ConflictResolver

detector = ConflictDetector()
conflicts = detector.detect_value_conflicts(entities, "name")

resolver = ConflictResolver()
results = resolver.resolve_conflicts(conflicts, strategy="voting")
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
    Context engineering infrastructure for agents. Formalizes context as a graph of connections with RAG-enhanced memory. Features GraphRAG with multi-hop reasoning and LLM-generated responses.

**Key Features:**

- Context graph construction from entities, relationships, and conversations
- Agent memory management with RAG integration
- Entity linking across sources with URI assignment
- Hybrid context retrieval (vector + graph + memory)
- **Multi-hop reasoning** through knowledge graphs
- **LLM-generated responses** grounded in graph context
- **Reasoning trace** showing entity relationship paths
- Conversation history management
- Context accumulation and synthesis
- Graph-based context traversal

**Components:**

- `ContextGraph` — In-memory context graph store and builder methods
- `ContextNode` — Context graph node data structure
- `ContextEdge` — Context graph edge data structure
- `AgentMemory` — Manages persistent agent memory with RAG
- `AgentContext` — High-level context interface with GraphRAG capabilities
- `ContextRetriever` — Retrieves relevant context with multi-hop reasoning
- `MemoryItem` — Memory item data structure
- `EntityLinker` — Links entities across sources with URI assignment

**Algorithms:**

| Category | Algorithms |
| :--- | :--- |
| **Graph Construction** | BFS/DFS traversal, type-based indexing |
| **Memory Management** | Vector embedding, similarity search, retention policies |
| **Context Retrieval** | Vector similarity, multi-hop graph expansion, hybrid scoring |
| **Multi-Hop Reasoning** | BFS traversal up to N hops, reasoning path construction |
| **LLM Integration** | Prompt engineering with context and reasoning paths |
| **Entity Linking** | Hash-based URI generation, text similarity matching |

**Quick Example:**

```python
from semantica.context import AgentContext, ContextGraph, AgentMemory
from semantica.llms import Groq
from semantica.vector_store import VectorStore
import os

# Using AgentContext with GraphRAG reasoning
context = AgentContext(
    vector_store=VectorStore(backend="faiss"),
    knowledge_graph=kg
)

# Configure LLM provider
llm_provider = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# Query with multi-hop reasoning and LLM-generated response
result = context.query_with_reasoning(
    query="What IPs are associated with security alerts?",
    llm_provider=llm_provider,
    max_results=10,
    max_hops=2
)

print(f"Response: {result['response']}")
print(f"Reasoning Path: {result['reasoning_path']}")
print(f"Confidence: {result['confidence']:.3f}")

# Traditional context graph and memory
graph = ContextGraph()
graph_data = graph.build_from_entities_and_relationships(entities, relationships)

memory = AgentMemory(vector_store=vs, knowledge_graph=kg)
memory_id = memory.store("User asked about Python", metadata={"type": "conversation"})
results = memory.retrieve("Python", max_results=5)
```

**API Reference**: [Context Module](reference/context.md)

---

### LLM Providers Module

!!! abstract "Purpose"
    Unified interface for LLM providers. Supports Groq, OpenAI, HuggingFace, and LiteLLM (100+ LLMs) with clean imports and consistent API.

**Key Features:**

- **Unified Interface**: Same `generate()` and `generate_structured()` methods across all providers
- **Multiple Providers**: Groq, OpenAI, HuggingFace, and LiteLLM (100+ LLMs)
- **Clean Imports**: Simple `from semantica.llms import Groq, OpenAI, HuggingFaceLLM, LiteLLM`
- **Structured Output**: JSON generation support
- **API Key Management**: Environment variable and direct key support
- **Error Handling**: Graceful fallback when providers unavailable

**Components:**

- `Groq` — Groq API provider for fast inference
- `OpenAI` — OpenAI API provider (GPT-3.5, GPT-4, etc.)
- `HuggingFaceLLM` — HuggingFace Transformers for local LLM inference
- `LiteLLM` — Unified interface to 100+ LLM providers (OpenAI, Anthropic, Azure, Bedrock, Vertex AI, etc.)

**Supported Providers via LiteLLM:**

- OpenAI, Anthropic, Groq, Azure, Bedrock, Vertex AI, Cohere, Mistral, and 90+ more

**Quick Example:**

```python
from semantica.llms import Groq, OpenAI, HuggingFaceLLM, LiteLLM
import os

# Groq - Fast inference
groq = Groq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)
response = groq.generate("What is AI?")

# OpenAI
openai = OpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)
response = openai.generate("What is AI?")

# HuggingFace - Local models
hf = HuggingFaceLLM(model_name="gpt2")  # or model="gpt2" for consistency
response = hf.generate("What is AI?")

# LiteLLM - Unified interface to 100+ LLMs
litellm = LiteLLM(
    model="openai/gpt-4o",  # or "anthropic/claude-sonnet-4-20250514", etc.
    api_key=os.getenv("OPENAI_API_KEY")
)
response = litellm.generate("What is AI?")

# Structured output
structured = groq.generate_structured("Extract entities from: Apple Inc. was founded by Steve Jobs.")
```

**API Reference**: [LLM Providers Module](reference/llms.md)

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
- Temporal data visualization
- Ontology visualization
- Multiple output formats (HTML, PNG, SVG)
- Custom styling

**Components:**

- `KGVisualizer` — Visualize knowledge graphs
- `EmbeddingVisualizer` — Visualize embeddings (t-SNE, PCA, UMAP)
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
from semantica.core import Semantica

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
from semantica.core import Semantica
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
from semantica.reasoning import Reasoner

# Build temporal graph
builder = GraphBuilder(temporal=True)
kg = builder.build(entities, relationships)

# Add reasoning
reasoner = Reasoner()
reasoner.add_rule("IF A THEN B")

new_facts = reasoner.infer_facts(kg)
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
| **Reasoning** | `semantica.reasoning` | `Reasoner` | Logical inference |
| **Embeddings** | `semantica.embeddings` | `EmbeddingGenerator` | Vector generation |
| **Vector Store** | `semantica.vector_store` | `VectorStore` | Vector storage |
| **Graph Store** | `semantica.graph_store` | `GraphStore` | Graph database |
| **Triplet Store** | `semantica.triplet_store` | `TripletStore` | RDF storage |
| **Deduplication** | `semantica.deduplication` | `DuplicateDetector` | Duplicate removal |
| **Conflicts** | `semantica.conflicts` | `ConflictDetector` | Conflict resolution |
| **Context** | `semantica.context` | `AgentContext` | Agent context & GraphRAG |
| **LLM Providers** | `semantica.llms` | `Groq`, `OpenAI`, `HuggingFaceLLM`, `LiteLLM` | LLM integration |
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
