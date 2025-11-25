# Add Intelligence Cookbook Notebooks with MCP, Agents, and Orchestrator-Worker Pattern

## Overview
Add comprehensive intelligence-focused notebooks to `cookbook/use_cases/intelligence/` with complete end-to-end pipelines. The **Intelligence Analysis** notebook will use the **Orchestrator-Worker Pattern** with detailed graph analytics, hybrid RAG, and ontology building. Update documentation in `docs/cookbook.md` and `docs/use-cases.md`.

## New Notebooks to Create

### 1. Criminal Network Analysis (`Criminal_Network_Analysis.ipynb`)
Complete pipeline from data sources to GraphRAG with agent-based workflows:
- **Data Sources**: Ingest from police reports, court records, surveillance data, communication logs
- **MCP Integration**: Utilize MCP for accessing public records databases, court records APIs, and real-time data streams
- **Semantica Agents**: 
  - Data Gathering Agent (autonomous data collection with AgentMemory)
  - Network Analysis Agent (graph analytics and community detection)
  - Pattern Detection Agent (identifying suspicious patterns)
  - Report Generation Agent (compiling intelligence reports)
- **Agent Coordination**: Use Pipeline module for parallel agent workflows
- **Agent Memory**: AgentMemory for persistent context across interactions
- **Complete Pipeline**: Data sources → MCP → Parsing → Extraction → KG → Graph Analytics → GraphRAG → Agent Analysis → Visualization → Reporting

### 2. Law Enforcement and Forensics (`Law_Enforcement_Forensics.ipynb`)
Complete forensic analysis pipeline with agent-based workflows:
- **Data Sources**: Case files, evidence logs, witness statements, forensic reports, crime scene data
- **Semantica Agents**:
  - Evidence Collection Agent (autonomous evidence gathering)
  - Timeline Analysis Agent (temporal case timelines)
  - Cross-Case Correlation Agent (connections across cases)
  - Forensic Report Agent (comprehensive report generation)
- **Agent Coordination**: Multi-agent pipeline for parallel evidence processing
- **Agent Memory**: Persistent memory for case context and evidence chains
- **Complete Pipeline**: Case files → Parsing → Evidence Extraction → Temporal KG → Graph Analytics → GraphRAG → Agent Analysis → Visualization → Reporting

### 3. Intelligence Analysis (`Intelligence_Analysis.ipynb`) - **ORCHESTRATOR-WORKER PATTERN**
Comprehensive intelligence analysis using **Orchestrator-Worker Pattern** with detailed implementation:

#### Orchestrator-Worker Architecture:
- **Orchestrator**: ExecutionEngine coordinates all workers using PipelineBuilder and ParallelismManager
- **Worker 1 - Data Ingestion Worker**: Handles multi-source data ingestion (FileIngestor, WebIngestor, StreamIngestor, FeedIngestor, DBIngestor)
- **Worker 2 - Ontology Building Worker**: Complete 6-stage ontology generation pipeline
  - Stage 1: Semantic Network Parsing (extract domain concepts)
  - Stage 2: YAML-to-Definition (transform concepts to class definitions)
  - Stage 3: Definition-to-Types (map to OWL types)
  - Stage 4: Hierarchy Generation (build taxonomic structures)
  - Stage 5: TTL Generation (generate OWL/Turtle syntax)
  - Stage 6: Symbolic Validation (HermiT/Pellet reasoning)
- **Worker 3 - Graph Construction Worker**: Builds knowledge graphs (GraphBuilder, TemporalGraphQuery)
- **Worker 4 - Graph Analytics Worker**: Comprehensive graph analytics including:
  - Centrality Measures: PageRank, Betweenness, Closeness, Eigenvector
  - Community Detection: Louvain algorithm
  - Connectivity Analysis: Path finding, shortest paths, connectivity metrics
  - Graph Metrics: Density, clustering coefficient, diameter, radius
- **Worker 5 - Hybrid RAG Worker**: Complete hybrid RAG implementation:
  - Vector Store setup with embeddings
  - Knowledge Graph queries
  - Hybrid Search (combining vector similarity + graph traversal)
  - Context Retrieval (ContextRetriever)
  - Query Orchestration across KG and vector store
- **Worker 6 - Intelligence Analysis Worker**: Threat assessment, geospatial analysis, pattern detection
- **Worker 7 - Report Generation Worker**: Compiles comprehensive intelligence reports

#### Complete Features:
- **Data Sources**: OSINT feeds, threat intelligence, social media, news, public records, geospatial data
- **MCP Integration**: Real-time data fetching, web scraping, API integration, browser automation for OSINT
- **Agent Memory**: Persistent memory for threat context and intelligence history
- **Complete Pipeline**: OSINT sources → MCP → Orchestrator → Parallel Workers → Ontology → KG → Graph Analytics → Hybrid RAG → Intelligence Analysis → Visualization → Reporting

## Files to Create/Modify

### New Notebooks (in `cookbook/use_cases/intelligence/`)
- `Criminal_Network_Analysis.ipynb`
- `Law_Enforcement_Forensics.ipynb`
- `Intelligence_Analysis.ipynb` (with Orchestrator-Worker Pattern)

### Documentation Updates
- `docs/cookbook.md` - Add new notebooks to Intelligence section
- `docs/use-cases.md` - Add use case cards for Criminal Network Analysis and Law Enforcement & Forensics

## Implementation Details

### Intelligence Analysis - Orchestrator-Worker Pipeline Structure:

1. **Orchestrator Setup** - Initialize ExecutionEngine, PipelineBuilder, ParallelismManager
2. **Data Sources** - Multiple ingestion (FileIngestor, DBIngestor, WebIngestor, StreamIngestor, FeedIngestor)
3. **MCP Integration** - External data access, web scraping, browser automation
4. **Worker 1 - Data Ingestion Worker** - Parallel data gathering from multiple sources
5. **Data Parsing** - Parse structured/unstructured data (JSONParser, XMLParser, CSVParser, DocumentParser, StructuredDataParser)
6. **Data Normalization** - Clean and standardize (TextNormalizer, DataNormalizer)
7. **Entity & Relation Extraction** - Extract entities, relationships, events (NERExtractor, RelationExtractor, TripleExtractor, EventDetector)
8. **Worker 2 - Ontology Building Worker** - Complete 6-stage ontology generation:
   - Use OntologyGenerator, ClassInferrer, PropertyGenerator
   - Generate OWL/Turtle with OWLGenerator
   - Validate with OntologyValidator (HermiT/Pellet)
9. **Worker 3 - Graph Construction Worker** - Build knowledge graphs:
   - GraphBuilder for entity/relationship graphs
   - TemporalGraphQuery for time-aware graphs
10. **Worker 4 - Graph Analytics Worker** - All graph analytics:
    - GraphAnalyzer: PageRank, Betweenness, Closeness, Eigenvector centrality
    - CommunityDetector: Louvain community detection
    - ConnectivityAnalyzer: Path finding, shortest paths, connectivity
    - CentralityCalculator: All centrality measures
    - Graph metrics: density, clustering, diameter, radius
11. **Worker 5 - Hybrid RAG Worker** - Complete hybrid RAG:
    - EmbeddingGenerator: Generate embeddings for entities and text
    - VectorStore: Store and index embeddings
    - HybridSearch: Combine vector similarity + graph queries
    - ContextRetriever: Retrieve relevant context from KG and vectors
    - Query orchestration: Coordinate queries across KG and vector store
12. **Worker 6 - Intelligence Analysis Worker** - Threat assessment, geospatial analysis, pattern detection
13. **Agent Memory Integration** - Store and retrieve agent context using AgentMemory
14. **Orchestrator Coordination** - Coordinate all workers with parallel execution
15. **Visualization** - Network graphs, analytics dashboards, maps (KGVisualizer, AnalyticsVisualizer, TemporalVisualizer)
16. **Worker 7 - Report Generation Worker** - Compile comprehensive intelligence reports
17. **Report Generation** - Professional HTML reports (ReportGenerator, HTMLExporter)

### Other Notebooks - Standard Pipeline Structure:

1. **Data Sources** - Multiple ingestion
2. **MCP Integration** - (Criminal Network Analysis only)
3. **Semantica Agent Setup** - Initialize AgentMemory, create specialized agents
4. **Agent-Based Data Gathering** - Autonomous agents gather data
5. **Data Parsing** - Parse structured/unstructured data
6. **Data Normalization** - Clean and standardize
7. **Entity & Relation Extraction** - Extract entities, relationships, events
8. **Knowledge Graph Construction** - Build graphs
9. **Agent-Based Analysis** - Specialized agents perform parallel analysis
10. **Graph Analytics** - Community detection, centrality, connectivity
11. **GraphRAG Implementation** - Embeddings, vector store, hybrid search
12. **Agent Memory Integration** - Store and retrieve agent context
13. **Detailed Analysis** - Reasoning, inference, pattern detection
14. **Agent Coordination** - Pipeline module for multi-agent workflow orchestration
15. **Visualization** - Network graphs, analytics dashboards, maps
16. **Agent-Based Report Generation** - Agents compile comprehensive reports
17. **Report Generation** - Professional HTML reports

### Semantica Agent Implementation:

- **AgentMemory**: Persistent context storage, memory retrieval, conversation history
- **Pipeline Coordination**: PipelineBuilder, ExecutionEngine, ParallelismManager for multi-agent workflows
- **Specialized Agents**: Each agent has specific role (data gathering, analysis, reporting)
- **Agent Examples**: Code demonstrations of agent workflows with memory integration

### MCP Integration:

- **Intelligence Analysis**: MCP browser tools for OSINT, resources for external feeds
- **Criminal Network Analysis**: MCP for public records, court databases, API integration
- **Agent-MCP Coordination**: Agents use MCP for autonomous data gathering

### Notebook Structure:

#### Intelligence Analysis (Orchestrator-Worker Pattern):
- Overview with Orchestrator-Worker pattern explanation
- Semantica modules used (30+ modules including Orchestrator, Workers, Ontology, Graph Analytics, Hybrid RAG)
- **Orchestrator Architecture**: Detailed explanation of orchestrator and worker roles
- **Worker Implementation**: Detailed code for each worker (7 workers)
- **Ontology Building**: Complete 6-stage ontology generation pipeline demonstration
- **Graph Analytics**: All analytics methods (PageRank, Betweenness, Closeness, Eigenvector, Louvain, connectivity, paths)
- **Hybrid RAG**: Complete implementation with KG queries + vector search, query orchestration
- MCP integration demonstration
- Step-by-step implementation with orchestrator coordinating workers
- Parallel worker execution examples
- Agent memory integration
- Best practices for orchestrator-worker pattern
- Best practices for agents and MCP
- Conclusion with key takeaways

#### Other Notebooks:
- Overview with complete pipeline description
- Semantica modules used (20+ modules including AgentMemory, Pipeline)
- Agent Architecture explanation
- MCP integration demonstration (Criminal Network Analysis)
- Step-by-step implementation with agent workflows
- Agent memory integration examples
- Multi-agent pipeline orchestration
- Best practices for agents and MCP
- Conclusion with key takeaways

## Key Implementation Details for Orchestrator-Worker Pattern:

### Orchestrator Code Example:
```python
from semantica.pipeline import PipelineBuilder, ExecutionEngine, ParallelismManager
from semantica.ontology import OntologyGenerator
from semantica.kg import GraphBuilder, GraphAnalyzer
from semantica.vector_store import VectorStore, HybridSearch
from semantica.context import AgentMemory

# Initialize orchestrator
orchestrator = ExecutionEngine()
parallelism_manager = ParallelismManager(max_workers=7)

# Define workers
def data_ingestion_worker(sources):
    # Worker 1: Multi-source data ingestion
    pass

def ontology_building_worker(entities, relationships):
    # Worker 2: Complete 6-stage ontology generation
    ontology_gen = OntologyGenerator()
    ontology = ontology_gen.generate_ontology({"entities": entities, "relationships": relationships})
    return ontology

def graph_construction_worker(entities, relationships):
    # Worker 3: Build knowledge graph
    graph_builder = GraphBuilder()
    kg = graph_builder.build(entities, relationships)
    return kg

def graph_analytics_worker(kg):
    # Worker 4: All graph analytics
    analyzer = GraphAnalyzer()
    pagerank = analyzer.compute_centrality(kg, method="pagerank")
    betweenness = analyzer.compute_centrality(kg, method="betweenness")
    communities = analyzer.detect_communities(kg, method="louvain")
    # ... all analytics
    return {"pagerank": pagerank, "betweenness": betweenness, "communities": communities}

def hybrid_rag_worker(kg, vector_store):
    # Worker 5: Hybrid RAG with KG and vector store
    hybrid_search = HybridSearch(vector_store=vector_store, knowledge_graph=kg)
    # Query orchestration
    pass

# Build pipeline with workers
pipeline = PipelineBuilder() \
    .add_step("data_ingestion", "custom", func=data_ingestion_worker) \
    .add_step("ontology_building", "custom", func=ontology_building_worker) \
    .add_step("graph_construction", "custom", func=graph_construction_worker) \
    .add_step("graph_analytics", "custom", func=graph_analytics_worker) \
    .add_step("hybrid_rag", "custom", func=hybrid_rag_worker) \
    .build()

# Execute with parallel workers
result = orchestrator.execute_pipeline(pipeline, parallel=True, max_workers=7)
```

Each notebook demonstrates the full journey from raw data sources through autonomous agent workflows (or orchestrator-worker pattern) and GraphRAG to actionable intelligence.

