# Add Intelligence Cookbook Notebooks with MCP and Semantica Agents

## Overview
Add comprehensive intelligence-focused notebooks to `cookbook/use_cases/intelligence/` with complete end-to-end pipelines covering data ingestion (including MCP integration), knowledge graph construction, GraphRAG implementation, **Semantica agent-based workflows**, and detailed analysis. Update documentation in `docs/cookbook.md` and `docs/use-cases.md`.

## New Notebooks to Create

### 1. Criminal Network Analysis (`Criminal_Network_Analysis.ipynb`)
Complete pipeline from data sources to GraphRAG with **agent-based workflows**:
- **Data Sources**: Ingest from police reports, court records, surveillance data, communication logs
- **MCP Integration**: Utilize MCP for accessing public records databases, court records APIs, and real-time data streams
- **Semantica Agents**: 
  - **Data Gathering Agent**: Autonomous agent using AgentMemory to gather and track data from multiple sources
  - **Network Analysis Agent**: Specialized agent for graph analytics and community detection
  - **Pattern Detection Agent**: Agent for identifying suspicious patterns and relationships
  - **Report Generation Agent**: Agent for compiling intelligence reports
- **Agent Coordination**: Use Pipeline module (PipelineBuilder, ExecutionEngine, ParallelismManager) to coordinate parallel agent workflows
- **Agent Memory**: Use AgentMemory for persistent context across agent interactions
- **Parsing**: Parse structured/unstructured documents, JSON, CSV, PDFs
- **Extraction**: Extract suspects, organizations, locations, events, relationships
- **Knowledge Graph**: Build criminal network graph with temporal relationships
- **Graph Analytics**: Community detection, centrality measures, key player identification
- **GraphRAG**: Vector store, hybrid search, context retrieval for intelligence queries
- **Detailed Analysis**: Pattern detection, network structure analysis, threat assessment
- **Visualization**: Network graphs, community visualization, centrality rankings
- **Reporting**: Generate intelligence reports on criminal structures

### 2. Law Enforcement and Forensics (`Law_Enforcement_Forensics.ipynb`)
Complete forensic analysis pipeline with **agent-based workflows**:
- **Data Sources**: Case files, evidence logs, witness statements, forensic reports, crime scene data
- **Semantica Agents**:
  - **Evidence Collection Agent**: Autonomous agent for gathering and organizing evidence
  - **Timeline Analysis Agent**: Agent for building temporal case timelines
  - **Cross-Case Correlation Agent**: Agent for finding connections across multiple cases
  - **Forensic Report Agent**: Agent for generating comprehensive forensic reports
- **Agent Coordination**: Multi-agent pipeline for parallel evidence processing
- **Agent Memory**: Persistent memory for case context and evidence chains
- **Parsing**: Parse PDFs, structured reports, evidence databases, temporal logs
- **Extraction**: Extract entities (persons, locations, evidence, events), relationships, timelines
- **Knowledge Graph**: Build temporal knowledge graph for case timelines and evidence correlation
- **Graph Analytics**: Timeline analysis, evidence correlation, pattern detection across cases
- **GraphRAG**: Semantic search across case files, evidence retrieval, context-aware queries
- **Detailed Analysis**: Cross-case correlation, evidence chain analysis, suspect identification
- **Visualization**: Timeline visualization, evidence networks, case correlation graphs
- **Reporting**: Generate forensic analysis reports with evidence chains

### 3. Intelligence Analysis (`Intelligence_Analysis.ipynb`)
Comprehensive intelligence analysis with **agent-based workflows**:
- **Data Sources**: OSINT feeds, threat intelligence, social media, news, public records, geospatial data
- **MCP Integration**: Utilize MCP for real-time data fetching, web scraping, API integration, external database access, and browser automation for OSINT gathering
- **Semantica Agents**:
  - **OSINT Gathering Agent**: Autonomous agent using MCP browser tools for web scraping and OSINT collection
  - **Threat Assessment Agent**: Specialized agent for threat analysis and risk scoring
  - **Geospatial Intelligence Agent**: Agent for location-based tracking and geographic analysis
  - **Multi-Source Fusion Agent**: Agent for correlating intelligence from multiple sources
  - **Intelligence Report Agent**: Agent for generating comprehensive threat intelligence reports
- **Agent Coordination**: Complex multi-agent pipeline with parallel execution for intelligence gathering
- **Agent Memory**: Persistent memory for threat context, entity tracking, and intelligence history
- **Parsing**: Multi-format parsing (RSS feeds, JSON, XML, web scraping, geospatial formats)
- **Extraction**: Extract threat actors, locations, events, relationships, temporal patterns
- **Knowledge Graph**: Build multi-source intelligence graph with geospatial and temporal dimensions
- **Graph Analytics**: Threat assessment, risk scoring, entity relationship mapping, pattern detection
- **GraphRAG**: Multi-source intelligence fusion, hybrid search, contextual threat queries
- **Detailed Analysis**: 
  - Multi-source intelligence fusion and correlation
  - Threat assessment and risk analysis
  - Geospatial intelligence with location tracking
  - Temporal threat evolution analysis
- **Visualization**: Geographic network maps, threat timelines, relationship networks
- **Reporting**: Generate comprehensive threat intelligence reports

## Files to Create/Modify

### New Notebooks (in `cookbook/use_cases/intelligence/`)
- `Criminal_Network_Analysis.ipynb`
- `Law_Enforcement_Forensics.ipynb`
- `Intelligence_Analysis.ipynb`

### Documentation Updates
- `docs/cookbook.md` - Add new notebooks to Intelligence section
- `docs/use-cases.md` - Add new use case cards for criminal networks and law enforcement

## Implementation Details

### Complete Pipeline Structure (All Notebooks):
1. **Data Sources** - Multiple ingestion sources (FileIngestor, DBIngestor, WebIngestor, StreamIngestor, FeedIngestor)
2. **MCP Integration** - Utilize MCP servers for external data access, real-time feeds, API integration, web scraping, and browser automation (in Intelligence Analysis and Criminal Network Analysis notebooks)
3. **Semantica Agent Setup** - Initialize AgentMemory, create specialized agents, set up agent coordination
4. **Agent-Based Data Gathering** - Autonomous agents gather data using MCP and Semantica ingestors
5. **Data Parsing** - Parse structured/unstructured data (JSONParser, XMLParser, CSVParser, DocumentParser, StructuredDataParser)
6. **Data Normalization** - Clean and standardize (TextNormalizer, DataNormalizer)
7. **Entity & Relation Extraction** - Extract entities, relationships, events (NERExtractor, RelationExtractor, TripleExtractor, EventDetector)
8. **Knowledge Graph Construction** - Build graphs (GraphBuilder, TemporalGraphQuery)
9. **Agent-Based Analysis** - Specialized agents perform parallel analysis tasks
10. **Graph Analytics** - Community detection, centrality, connectivity (GraphAnalyzer, ConnectivityAnalyzer, CentralityCalculator)
11. **GraphRAG Implementation** - Embeddings, vector store, hybrid search, context retrieval (EmbeddingGenerator, VectorStore, HybridSearch, ContextRetriever)
12. **Agent Memory Integration** - Store and retrieve agent context using AgentMemory
13. **Detailed Analysis** - Reasoning, inference, pattern detection (InferenceEngine, RuleManager, ExplanationGenerator)
14. **Agent Coordination** - Use Pipeline module for multi-agent workflow orchestration
15. **Visualization** - Network graphs, analytics dashboards, geographic maps (KGVisualizer, AnalyticsVisualizer, TemporalVisualizer)
16. **Agent-Based Report Generation** - Agents compile and generate professional reports
17. **Report Generation** - Professional HTML reports (ReportGenerator, HTMLExporter)

### Semantica Agent Implementation Details:

#### AgentMemory Usage:
- **Persistent Context**: Store agent interactions, decisions, and findings
- **Memory Retrieval**: Retrieve relevant context for agent decision-making
- **Conversation History**: Track agent conversations and analysis sessions
- **Context Accumulation**: Build up intelligence context over time

#### Pipeline Agent Coordination:
- **PipelineBuilder**: Define multi-agent workflows
- **ExecutionEngine**: Execute agent pipelines with error handling
- **ParallelismManager**: Run agents in parallel for efficiency
- **Specialized Agents**: Each agent has a specific role (data gathering, analysis, reporting)

#### Agent Workflow Examples:
```python
# Example: Multi-agent intelligence gathering
from semantica.context import AgentMemory
from semantica.pipeline import PipelineBuilder, ExecutionEngine, ParallelismManager

# Initialize agent memory
agent_memory = AgentMemory(vector_store=vs, knowledge_graph=kg)

# Define specialized agents
def osint_gathering_agent(query, memory):
    """Autonomous OSINT gathering agent"""
    # Use MCP for web scraping
    # Store findings in agent memory
    findings = gather_osint(query)
    memory.store(f"OSINT findings: {findings}", metadata={"agent": "osint", "query": query})
    return findings

def threat_assessment_agent(intel_data, memory):
    """Threat assessment agent"""
    # Retrieve relevant context from memory
    context = memory.retrieve("threat patterns", max_results=10)
    # Perform threat analysis
    assessment = analyze_threats(intel_data, context)
    memory.store(f"Threat assessment: {assessment}", metadata={"agent": "threat"})
    return assessment

# Build multi-agent pipeline
pipeline = PipelineBuilder() \
    .add_step("osint_gathering", "custom", func=osint_gathering_agent, args=(query, agent_memory)) \
    .add_step("threat_assessment", "custom", func=threat_assessment_agent, args=(intel_data, agent_memory)) \
    .build()

# Execute with parallel agents
engine = ExecutionEngine()
result = engine.execute_pipeline(pipeline, parallel=True)
```

### MCP Integration Details:
- **Intelligence Analysis Notebook**: 
  - Use MCP browser tools for web scraping and OSINT gathering
  - Use MCP resources for accessing external intelligence feeds
  - Demonstrate real-time data fetching via MCP
  - Agents use MCP for autonomous data gathering
- **Criminal Network Analysis Notebook**:
  - Use MCP for accessing public records and court databases
  - Demonstrate API integration via MCP
  - Show real-time data stream processing
  - Agents coordinate MCP-based data gathering

### Notebook Structure:
- Overview with complete pipeline description
- Semantica modules used (20+ modules including AgentMemory, Pipeline)
- **Agent Architecture**: Explanation of agent roles and coordination
- MCP integration demonstration (for Intelligence Analysis and Criminal Network Analysis)
- Step-by-step implementation:
  - **Agent Setup**: Initialize AgentMemory and create specialized agents
  - Data ingestion from multiple sources (including MCP resources)
  - **Agent-Based Data Gathering**: Autonomous agents gather data
  - MCP-based external data fetching and API integration
  - Parsing and normalization
  - Entity and relation extraction
  - Knowledge graph construction
  - **Agent-Based Analysis**: Parallel agent workflows for analysis
  - Graph analytics and pattern detection
  - **Agent Memory Integration**: Store and retrieve agent context
  - GraphRAG setup and query examples
  - **Agent Coordination**: Multi-agent pipeline orchestration
  - Detailed analysis with insights
  - Visualization examples
  - **Agent-Based Report Generation**: Agents compile reports
  - Report generation
- Best practices and deployment recommendations
- **Agent Best Practices**: Agent memory management, coordination patterns
- MCP integration best practices
- Conclusion with key takeaways

Each notebook will be comprehensive, demonstrating the full journey from raw data sources (including MCP-enabled external sources) through **autonomous agent workflows** and GraphRAG to actionable intelligence and detailed analysis.

## Key Agent Features to Highlight:

1. **Autonomous Data Gathering**: Agents independently gather data from multiple sources
2. **Persistent Memory**: AgentMemory maintains context across sessions
3. **Parallel Coordination**: Multiple agents work simultaneously on different tasks
4. **Specialized Roles**: Each agent has a specific expertise area
5. **Context-Aware Analysis**: Agents use memory to make informed decisions
6. **Coordinated Workflows**: Pipeline module orchestrates complex multi-agent systems
7. **Intelligent Reporting**: Agents compile findings into comprehensive reports

