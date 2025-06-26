import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Brain, 
  Database, 
  Network, 
  Zap, 
  Shield, 
  Activity,
  Code2,
  GitFork,
  Star,
  Download,
  ArrowRight,
  CheckCircle,
  Globe,
  Cpu,
  BarChart3,
  Users,
  BookOpen,
  MessageSquare,
  Github,
  ExternalLink,
  Copy,
  Terminal,
  Layers,
  Target,
  Workflow,
  FileText,
  Search,
  Bot,
  Settings,
  Sparkles,
  Lock,
  PlayCircle,
  Eye,
  TrendingUp,
  Building,
  Lightbulb,
  Rocket,
  Package,
  Wrench,
  Heart,
  Award
} from "lucide-react";
import { useState } from "react";

const Index = () => {
  const [copiedCode, setCopiedCode] = useState("");

  const copyToClipboard = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(""), 2000);
  };

  const installCode = `pip install semanticore`;
  
  const quickStartCode = `from semanticore import SemantiCore

# Initialize with your preferred providers
core = SemantiCore(
    llm_provider="openai",
    embedding_model="text-embedding-3-large",
    vector_store="pinecone",
    graph_db="neo4j"
)

# Transform unstructured text into semantic knowledge
text = """
Tesla reported Q4 2024 earnings with $25.2B revenue, a 15% increase YoY.
CEO Elon Musk highlighted the success of Model Y and Chinese market expansion.
The company plans to launch three new models in 2025, including Cybertruck.
"""

# Extract semantic information
result = core.extract_semantics(text)

print("Entities:", result.entities)
# [Entity(name="Tesla", type="ORGANIZATION"), Entity(name="Elon Musk", type="PERSON")]

print("Relationships:", result.relationships) 
# [Relation(subject="Tesla", predicate="reported", object="Q4 2024 earnings")]

print("Events:", result.events)
# [Event(type="EARNINGS_REPORT", date="Q4 2024", amount="$25.2B")]

# Generate knowledge graph
knowledge_graph = core.build_knowledge_graph(text)
print(f"Graph nodes: {len(knowledge_graph.nodes)}")
print(f"Graph edges: {len(knowledge_graph.edges)}")`;

  const semanticExtractionCode = `from semanticore.extract import EntityExtractor, RelationExtractor, TopicClassifier

# Named Entity Recognition with custom models
extractor = EntityExtractor(
    model="en_core_web_trf",  # spaCy model
    custom_labels=["MALWARE", "THREAT_ACTOR", "VULNERABILITY"]
)

entities = extractor.extract("APT29 used FrostBite malware against critical infrastructure")

# Relation and Triple Extraction
rel_extractor = RelationExtractor(llm_provider="openai")
relations = rel_extractor.extract_relations(text, entities)

# Topic Classification and Categorization
classifier = TopicClassifier()
topics = classifier.classify(text, categories=["cybersecurity", "technology", "politics"])`;

  const schemaGenerationCode = `from semanticore.schema import SchemaGenerator, validate_data

# Generate Pydantic models from extracted entities
generator = SchemaGenerator()
schema = generator.from_entities(entities)

# Export to various formats
schema.to_pydantic()    # Python Pydantic model
schema.to_json_schema() # JSON Schema
schema.to_yaml()        # YAML Schema
schema.to_typescript()  # TypeScript interfaces

# Validate new data against generated schema
is_valid = validate_data(new_data, schema)`;

  const vectorEmbeddingCode = `from semanticore.vectorizer import SemanticChunker, EmbeddingEngine

# Semantic-aware chunking
chunker = SemanticChunker(
    chunk_size=512,
    overlap=50,
    respect_boundaries=True,  # Don't split entities/relations
    add_metadata=True
)

chunks = chunker.chunk_document(document, semantic_info=result)

# Multi-modal embedding support
embedder = EmbeddingEngine(
    provider="sentence-transformers",
    model="all-MiniLM-L6-v2"
)

embedded_chunks = embedder.embed_chunks(chunks)

# Direct vector database integration
from semanticore.vector_stores import FAISSStore, PineconeStore
store = FAISSStore()
store.add_embeddings(embedded_chunks)`;

  const knowledgeGraphCode = `from semanticore.kg import Neo4jExporter, RDFExporter

# Neo4j export with Cypher generation
neo4j_exporter = Neo4jExporter(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Create nodes and relationships
neo4j_exporter.export_entities(entities)
neo4j_exporter.export_relations(relations)

# Query the generated knowledge graph
from semanticore.kg.query import GraphQuerier
querier = GraphQuerier(neo4j_exporter)
results = querier.cypher("MATCH (n:ORGANIZATION)-[r:RELEASED]->(m:PRODUCT) RETURN n, r, m")`;

  const advancedFeaturesCode = `from semanticore.streaming import SemanticStreamProcessor
from semanticore.context import ContextEngineer
from semanticore.domains import CybersecurityProcessor

# Real-time semantic processing
stream_processor = SemanticStreamProcessor(
    input_streams=["kafka://events", "websocket://feeds"],
    processing_pipeline=[
        "entity_extraction",
        "relationship_detection", 
        "ontology_mapping",
        "knowledge_graph_update"
    ],
    batch_size=100,
    processing_interval="5s"
)

# Advanced context engineering for RAG
context_engineer = ContextEngineer(
    max_context_length=128000,
    compression_strategy="semantic_preservation",
    relevance_scoring=True
)

# Domain-specific processing
cyber_processor = CybersecurityProcessor(
    threat_intelligence_feeds=["misp", "stix"],
    ontology="cybersecurity.owl",
    enable_threat_hunting=True
)`;

  const integrationCode = `# LangChain Integration
from semanticore.integrations.langchain import SemanticChain
from langchain.chains import ConversationalRetrievalChain

semantic_chain = SemanticChain(
    semanticore_instance=core,
    retriever_type="semantic",
    context_engineering=True
)

# LlamaIndex Integration
from semanticore.integrations.llamaindex import SemanticIndex
llama_index = VectorStoreIndex.from_vector_store(
    semantic_index.get_vector_store()
)

# CrewAI Integration
from semanticore.integrations.crewai import SemanticCrew
from crewai import Agent, Task, Crew

researcher = Agent(
    role='Research Analyst',
    goal='Analyze semantic patterns in data',
    backstory='Expert in semantic data analysis',
    semantic_memory=core.get_semantic_memory()
)`;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50/30">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-200/50 shadow-sm">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <img 
              src="/lovable-uploads/94bf75c5-552c-44a9-a37f-6cf436463006.png" 
              alt="Hawksight AI" 
              className="h-10 w-10 rounded-lg shadow-sm"
            />
            <div className="flex flex-col">
              <div className="flex items-center space-x-2">
                <Brain className="h-7 w-7 text-blue-600" />
                <span className="text-2xl font-bold bg-gradient-to-r from-slate-900 to-blue-800 bg-clip-text text-transparent">SemantiCore</span>
              </div>
              <span className="text-xs text-slate-500 font-medium">by Hawksight AI</span>
            </div>
            <Badge variant="secondary" className="bg-emerald-100 text-emerald-800 ml-2">
              v1.0
            </Badge>
          </div>
          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-slate-600 hover:text-blue-600 transition-colors font-medium">Features</a>
            <a href="#quickstart" className="text-slate-600 hover:text-blue-600 transition-colors font-medium">Quick Start</a>
            <a href="#examples" className="text-slate-600 hover:text-blue-600 transition-colors font-medium">Examples</a>
            <a href="#integrations" className="text-slate-600 hover:text-blue-600 transition-colors font-medium">Integrations</a>
            <a href="#docs" className="text-slate-600 hover:text-blue-600 transition-colors font-medium">Docs</a>
            <Button variant="outline" size="sm" className="border-slate-300">
              <Github className="w-4 h-4 mr-2" />
              GitHub
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-20 text-center">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-center space-x-2 mb-6">
            <Badge variant="secondary" className="bg-blue-100 text-blue-800 border-blue-200">
              <Star className="w-3 h-3 mr-1" />
              Open Source
            </Badge>
            <Badge variant="secondary" className="bg-emerald-100 text-emerald-800 border-emerald-200">
              <Download className="w-3 h-3 mr-1" />
              PyPI Available
            </Badge>
            <Badge variant="secondary" className="bg-purple-100 text-purple-800 border-purple-200">
              Python 3.8+
            </Badge>
            <Badge variant="secondary" className="bg-orange-100 text-orange-800 border-orange-200">
              MIT License
            </Badge>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold text-slate-900 mb-6 leading-tight">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-purple-600">Unified Toolkit</span><br />
            for Transforming <span className="text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600">Unstructured Data</span><br />
            into <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-600 to-orange-600">Semantic Layer</span>
          </h1>
          
          <p className="text-xl text-slate-600 mb-8 max-w-4xl mx-auto leading-relaxed">
            SemantiCore is the comprehensive open-source toolkit that transforms raw, unstructured data into intelligent semantic knowledge representations. Build <strong>AI agents</strong>, <strong>RAG systems</strong>, and <strong>knowledge graphs</strong> that truly understand meaning, not just text.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <Button size="lg" className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white shadow-lg">
              <PlayCircle className="w-5 h-5 mr-2" />
              Get Started
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
            <Button size="lg" variant="outline" className="border-slate-300 hover:bg-slate-50">
              <BookOpen className="w-5 h-5 mr-2" />
              Documentation
            </Button>
            <Button size="lg" variant="outline" className="border-slate-300 hover:bg-slate-50">
              <Eye className="w-5 h-5 mr-2" />
              Live Demo
            </Button>
          </div>

          {/* Quick Install */}
          <div className="bg-gradient-to-r from-slate-900 to-blue-900 rounded-xl p-6 max-w-2xl mx-auto shadow-xl">
            <div className="flex items-center justify-between mb-2">
              <span className="text-slate-300 text-sm font-medium">Install via pip</span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => copyToClipboard(installCode, "install")}
                className="text-slate-300 hover:text-white hover:bg-white/10"
              >
                {copiedCode === "install" ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </Button>
            </div>
            <code className="text-emerald-400 text-lg font-mono">{installCode}</code>
          </div>

          {/* Organization Info */}
          <div className="mt-16 p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl border border-blue-100">
            <div className="flex items-center justify-center space-x-4 mb-4">
              <img 
                src="/lovable-uploads/94bf75c5-552c-44a9-a37f-6cf436463006.png" 
                alt="Hawksight AI" 
                className="h-12 w-12 rounded-lg shadow-md"
              />
              <div>
                <h3 className="text-2xl font-bold text-slate-900">Hawksight AI</h3>
                <p className="text-slate-600">Open Source Organization Building Tools for Society</p>
              </div>
            </div>
            <p className="text-slate-700 max-w-2xl mx-auto">
              Empowering developers and researchers with cutting-edge AI tools that democratize access to advanced semantic processing capabilities.
            </p>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mt-16">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">15K+</div>
              <div className="text-slate-600">Downloads</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-emerald-600 mb-2">800+</div>
              <div className="text-slate-600">GitHub Stars</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">75+</div>
              <div className="text-slate-600">Contributors</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">8</div>
              <div className="text-slate-600">Core Modules</div>
            </div>
          </div>
        </div>
      </section>

      {/* Why SemantiCore */}
      <section className="py-20 bg-gradient-to-r from-blue-50 to-purple-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4">Why SemantiCore?</h2>
            <p className="text-xl text-slate-600 max-w-3xl mx-auto">
              Modern AI systems require structured, semantically rich data to perform effectively. SemantiCore solves the fundamental challenge of converting messy, unstructured information into clean, schema-compliant semantic layers.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <Bot className="w-12 h-12 text-blue-600 mb-4" />
                <CardTitle>🤖 Intelligent Agents</CardTitle>
                <CardDescription>
                  Type-safe, validated input/output schemas for reliable AI agent operations
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <Search className="w-12 h-12 text-green-600 mb-4" />
                <CardTitle>🔍 RAG Systems</CardTitle>
                <CardDescription>
                  Enhanced with semantic chunking, enriched metadata, and context preservation
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <Network className="w-12 h-12 text-purple-600 mb-4" />
                <CardTitle>🕸️ Knowledge Graphs</CardTitle>
                <CardDescription>
                  Automatically extracted entities, relations, and semantic triples
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <Wrench className="w-12 h-12 text-orange-600 mb-4" />
                <CardTitle>🛠️ LLM Tools</CardTitle>
                <CardDescription>
                  Wrapped with semantic contracts for reliable, consistent operation
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <BarChart3 className="w-12 h-12 text-red-600 mb-4" />
                <CardTitle>📊 Data Pipelines</CardTitle>
                <CardDescription>
                  Consistent, validated data flows across your entire AI stack
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow bg-white/70 backdrop-blur-sm">
              <CardHeader>
                <Sparkles className="w-12 h-12 text-pink-600 mb-4" />
                <CardTitle>✨ Context Engineering</CardTitle>
                <CardDescription>
                  Intelligent context compression and enhancement for optimal LLM performance
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Core Features */}
      <section id="features" className="py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4">Comprehensive Semantic Processing</h2>
            <p className="text-xl text-slate-600 max-w-3xl mx-auto">
              Everything you need to transform unstructured data into intelligent knowledge representations
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <Brain className="w-12 h-12 text-blue-600 mb-4" />
                <CardTitle>🧠 Semantic Processing</CardTitle>
                <CardDescription>
                  Multi-layer understanding: lexical, syntactic, semantic, and pragmatic analysis with entity & relationship extraction, context preservation, and domain adaptation
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <Target className="w-12 h-12 text-purple-600 mb-4" />
                <CardTitle>🎯 LLM Optimization</CardTitle>
                <CardDescription>
                  Context engineering, prompt optimization, memory management, and multi-model support for OpenAI, Anthropic, Google Gemini, Hugging Face, and local models
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <Network className="w-12 h-12 text-green-600 mb-4" />
                <CardTitle>🕸️ Knowledge Graphs</CardTitle>
                <CardDescription>
                  Automated construction with Neo4j, KuzuDB, ArangoDB, Amazon Neptune integration, semantic reasoning, and temporal modeling
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <Database className="w-12 h-12 text-orange-600 mb-4" />
                <CardTitle>📊 Vector & Embeddings</CardTitle>
                <CardDescription>
                  Contextual embeddings with Pinecone, Milvus, Weaviate, Chroma, FAISS integration, hybrid search, and multiple embedding models
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <Layers className="w-12 h-12 text-red-600 mb-4" />
                <CardTitle>🔗 Ontology Generation</CardTitle>
                <CardDescription>
                  Automated OWL/RDF ontology creation, schema evolution, standard compliance, and multi-format export (OWL, RDF, JSON-LD, Turtle)
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-0 shadow-lg hover:shadow-xl transition-shadow">
              <CardHeader>
                <Bot className="w-12 h-12 text-pink-600 mb-4" />
                <CardTitle>🤖 Agent Integration</CardTitle>
                <CardDescription>
                  Semantic routing, agent orchestration, LangChain/LlamaIndex/CrewAI compatibility, and real-time processing
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* Quick Start */}
      <section id="quickstart" className="py-20 bg-gradient-to-br from-slate-50 to-blue-50/50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4">30-Second Semantic Transformation</h2>
            <p className="text-xl text-slate-600">See how SemantiCore transforms unstructured text into rich semantic knowledge</p>
          </div>

          <div className="max-w-6xl mx-auto">
            <div className="bg-gradient-to-br from-slate-900 via-blue-900 to-purple-900 rounded-xl overflow-hidden shadow-2xl">
              <div className="bg-gradient-to-r from-slate-800 to-blue-800 px-6 py-3 flex items-center justify-between">
                <span className="text-slate-200 font-medium">Complete Semantic Extraction Pipeline</span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => copyToClipboard(quickStartCode, "quickstart")}
                  className="text-slate-300 hover:text-white hover:bg-white/10"
                >
                  {copiedCode === "quickstart" ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </Button>
              </div>
              <div className="p-6">
                <pre className="text-sm text-slate-200 overflow-x-auto leading-relaxed">
                  <code className="language-python">
                    <span className="text-purple-400">from</span> <span className="text-blue-300">semanticore</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">SemantiCore</span>
                    {"\n\n"}
                    <span className="text-gray-400"># Initialize with your preferred providers</span>
                    {"\n"}
                    <span className="text-blue-300">core</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">SemantiCore</span><span className="text-slate-300">(</span>
                    {"\n"}    <span className="text-green-300">llm_provider</span><span className="text-purple-400">=</span><span className="text-orange-300">"openai"</span><span className="text-slate-300">,</span>
                    {"\n"}    <span className="text-green-300">embedding_model</span><span className="text-purple-400">=</span><span className="text-orange-300">"text-embedding-3-large"</span><span className="text-slate-300">,</span>
                    {"\n"}    <span className="text-green-300">vector_store</span><span className="text-purple-400">=</span><span className="text-orange-300">"pinecone"</span><span className="text-slate-300">,</span>
                    {"\n"}    <span className="text-green-300">graph_db</span><span className="text-purple-400">=</span><span className="text-orange-300">"neo4j"</span>
                    {"\n"}<span className="text-slate-300">)</span>
                    {"\n\n"}
                    <span className="text-gray-400"># Transform unstructured text into semantic knowledge</span>
                    {"\n"}
                    <span className="text-blue-300">text</span> <span className="text-purple-400">=</span> <span className="text-orange-300">"""</span>
                    {"\n"}<span className="text-orange-300">Tesla reported Q4 2024 earnings with $25.2B revenue...</span>
                    {"\n"}<span className="text-orange-300">"""</span>
                    {"\n\n"}
                    <span className="text-gray-400"># Extract semantic information</span>
                    {"\n"}
                    <span className="text-blue-300">result</span> <span className="text-purple-400">=</span> <span className="text-blue-300">core</span><span className="text-slate-300">.</span><span className="text-yellow-300">extract_semantics</span><span className="text-slate-300">(</span><span className="text-blue-300">text</span><span className="text-slate-300">)</span>
                    {"\n\n"}
                    <span className="text-purple-400">print</span><span className="text-slate-300">(</span><span className="text-orange-300">"Entities:"</span><span className="text-slate-300">,</span> <span className="text-blue-300">result</span><span className="text-slate-300">.</span><span className="text-emerald-300">entities</span><span className="text-slate-300">)</span>
                    {"\n"}
                    <span className="text-gray-400"># [Entity(name="Tesla", type="ORGANIZATION"), ...]</span>
                  </code>
                </pre>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-8 mt-12">
              <Card className="border-0 shadow-lg hover:shadow-xl transition-all bg-gradient-to-br from-white to-blue-50/50">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Zap className="w-5 h-5 mr-2 text-yellow-500" />
                    What You Get
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    <li className="flex items-center">
                      <ArrowRight className="w-4 h-4 text-blue-500 mr-2" />
                      <span className="text-slate-700">Named entities & relationships</span>
                    </li>
                    <li className="flex items-center">
                      <ArrowRight className="w-4 h-4 text-purple-500 mr-2" />
                      <span className="text-slate-700">Knowledge graph triples</span>
                    </li>
                    <li className="flex items-center">
                      <ArrowRight className="w-4 h-4 text-emerald-500 mr-2" />
                      <span className="text-slate-700">Type-safe schemas</span>
                    </li>
                    <li className="flex items-center">
                      <ArrowRight className="w-4 h-4 text-orange-500 mr-2" />
                      <span className="text-slate-700">Rich metadata context</span>
                    </li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="border-0 shadow-lg hover:shadow-xl transition-all bg-gradient-to-br from-white to-emerald-50/50">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Activity className="w-5 h-5 mr-2 text-emerald-500" />
                    Key Benefits
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-emerald-500 mr-2" />
                      <span className="text-slate-700">One-line semantic extraction</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-emerald-500 mr-2" />
                      <span className="text-slate-700">Auto-generated schemas</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-emerald-500 mr-2" />
                      <span className="text-slate-700">Multi-format export</span>
                    </li>
                    <li className="flex items-center">
                      <CheckCircle className="w-4 h-4 text-emerald-500 mr-2" />
                      <span className="text-slate-700">Enterprise-ready validation</span>
                    </li>
                  </ul>
                </CardContent>
              </Card>

              <Card className="border-0 shadow-lg hover:shadow-xl transition-all bg-gradient-to-br from-white to-purple-50/50">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <TrendingUp className="w-5 h-5 mr-2 text-purple-500" />
                    Performance
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    <li className="flex items-center">
                      <ArrowRight className="w-4 h-4 text-purple-500 mr-2" />
                      <span className="text-slate-700">~2ms per extraction</span>
                    </li>
                    <li className="flex items-center">
                      <ArrowRight className="w-4 h-4 text-purple-500 mr-2" />
                      <span className="text-slate-700">Batch processing support</span>
                    </li>
                    <li className="flex items-center">
                      <ArrowRight className="w-4 h-4 text-purple-500 mr-2" />
                      <span className="text-slate-700">Streaming capabilities</span>
                    </li>
                    <li className="flex items-center">
                      <ArrowRight className="w-4 h-4 text-purple-500 mr-2" />
                      <span className="text-slate-700">Memory efficient</span>
                    </li>
                  </ul>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* Core Features Deep Dive */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4">Colorful Code Examples</h2>
            <p className="text-xl text-slate-600">Explore powerful capabilities with beautiful, syntax-highlighted examples</p>
          </div>

          <Tabs defaultValue="extraction" className="max-w-6xl mx-auto">
            <TabsList className="grid w-full grid-cols-3 md:grid-cols-6 bg-slate-100">
              <TabsTrigger value="extraction" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white">Extraction</TabsTrigger>
              <TabsTrigger value="schema" className="data-[state=active]:bg-purple-500 data-[state=active]:text-white">Schema</TabsTrigger>
              <TabsTrigger value="vector" className="data-[state=active]:bg-emerald-500 data-[state=active]:text-white">Vector</TabsTrigger>
              <TabsTrigger value="graph" className="data-[state=active]:bg-orange-500 data-[state=active]:text-white">Graph</TabsTrigger>
              <TabsTrigger value="advanced" className="data-[state=active]:bg-pink-500 data-[state=active]:text-white">Advanced</TabsTrigger>
              <TabsTrigger value="integrations" className="data-[state=active]:bg-indigo-500 data-[state=active]:text-white">Integrations</TabsTrigger>
            </TabsList>

            <TabsContent value="extraction" className="mt-8">
              <Card className="border-0 shadow-xl">
                <CardHeader className="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-t-lg">
                  <CardTitle className="flex items-center">
                    <Brain className="w-6 h-6 mr-2" />
                    Advanced Semantic Extraction Engine
                  </CardTitle>
                  <CardDescription className="text-blue-100">
                    Multi-layer NLP pipeline that extracts meaningful structure from unstructured data
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="bg-gradient-to-br from-slate-900 via-blue-900 to-purple-900 rounded-b-lg overflow-hidden">
                    <div className="flex items-center justify-between p-4 border-b border-slate-700">
                      <span className="text-slate-200 text-sm font-medium">Entity & Relation Extraction</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyToClipboard(semanticExtractionCode, "extraction")}
                        className="text-slate-300 hover:text-white hover:bg-white/10"
                      >
                        {copiedCode === "extraction" ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      </Button>
                    </div>
                    <pre className="p-6 text-sm text-slate-200 overflow-x-auto leading-relaxed">
                      <code className="language-python">
                        <span className="text-purple-400">from</span> <span className="text-blue-300">semanticore.extract</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">EntityExtractor</span><span className="text-slate-300">,</span> <span className="text-yellow-300">RelationExtractor</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Named Entity Recognition with custom models</span>
                        {"\n"}
                        <span className="text-blue-300">extractor</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">EntityExtractor</span><span className="text-slate-300">(</span>
                        {"\n"}    <span className="text-green-300">model</span><span className="text-purple-400">=</span><span className="text-orange-300">"en_core_web_trf"</span><span className="text-slate-300">,</span>
                        {"\n"}    <span className="text-green-300">custom_labels</span><span className="text-purple-400">=</span><span className="text-slate-300">[</span><span className="text-orange-300">"MALWARE"</span><span className="text-slate-300">,</span> <span className="text-orange-300">"THREAT_ACTOR"</span><span className="text-slate-300">]</span>
                        {"\n"}<span className="text-slate-300">)</span>
                        {"\n\n"}
                        <span className="text-blue-300">entities</span> <span className="text-purple-400">=</span> <span className="text-blue-300">extractor</span><span className="text-slate-300">.</span><span className="text-yellow-300">extract</span><span className="text-slate-300">(</span><span className="text-orange-300">"APT29 used FrostBite malware"</span><span className="text-slate-300">)</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Relation and Triple Extraction</span>
                        {"\n"}
                        <span className="text-blue-300">rel_extractor</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">RelationExtractor</span><span className="text-slate-300">(</span><span className="text-green-300">llm_provider</span><span className="text-purple-400">=</span><span className="text-orange-300">"openai"</span><span className="text-slate-300">)</span>
                        {"\n"}
                        <span className="text-blue-300">relations</span> <span className="text-purple-400">=</span> <span className="text-blue-300">rel_extractor</span><span className="text-slate-300">.</span><span className="text-yellow-300">extract_relations</span><span className="text-slate-300">(</span><span className="text-blue-300">text</span><span className="text-slate-300">,</span> <span className="text-blue-300">entities</span><span className="text-slate-300">)</span>
                      </code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="schema" className="mt-8">
              <Card className="border-0 shadow-xl">
                <CardHeader className="bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-t-lg">
                  <CardTitle className="flex items-center">
                    <Code2 className="w-6 h-6 mr-2" />
                    Dynamic Schema Generation
                  </CardTitle>
                  <CardDescription className="text-purple-100">
                    Automatically generate type-safe schemas from extracted semantic data
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="bg-gradient-to-br from-slate-900 via-purple-900 to-pink-900 rounded-b-lg overflow-hidden">
                    <div className="flex items-center justify-between p-4 border-b border-slate-700">
                      <span className="text-slate-200 text-sm font-medium">Multi-format Schema Export</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyToClipboard(schemaGenerationCode, "schema")}
                        className="text-slate-300 hover:text-white hover:bg-white/10"
                      >
                        {copiedCode === "schema" ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      </Button>
                    </div>
                    <pre className="p-6 text-sm text-slate-200 overflow-x-auto leading-relaxed">
                      <code className="language-python">
                        <span className="text-purple-400">from</span> <span className="text-blue-300">semanticore.schema</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">SchemaGenerator</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Generate Pydantic models from extracted entities</span>
                        {"\n"}
                        <span className="text-blue-300">generator</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">SchemaGenerator</span><span className="text-slate-300">()</span>
                        {"\n"}
                        <span className="text-blue-300">schema</span> <span className="text-purple-400">=</span> <span className="text-blue-300">generator</span><span className="text-slate-300">.</span><span className="text-yellow-300">from_entities</span><span className="text-slate-300">(</span><span className="text-blue-300">entities</span><span className="text-slate-300">)</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Export to various formats</span>
                        {"\n"}
                        <span className="text-blue-300">schema</span><span className="text-slate-300">.</span><span className="text-emerald-300">to_pydantic</span><span className="text-slate-300">()</span>    <span className="text-gray-400"># Python Pydantic model</span>
                        {"\n"}
                        <span className="text-blue-300">schema</span><span className="text-slate-300">.</span><span className="text-emerald-300">to_json_schema</span><span className="text-slate-300">()</span> <span className="text-gray-400"># JSON Schema</span>
                        {"\n"}
                        <span className="text-blue-300">schema</span><span className="text-slate-300">.</span><span className="text-emerald-300">to_typescript</span><span className="text-slate-300">()</span>  <span className="text-gray-400"># TypeScript interfaces</span>
                      </code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="vector" className="mt-8">
              <Card className="border-0 shadow-xl">
                <CardHeader className="bg-gradient-to-r from-emerald-500 to-emerald-600 text-white rounded-t-lg">
                  <CardTitle className="flex items-center">
                    <Database className="w-6 h-6 mr-2" />
                    Intelligent Chunking & Embedding
                  </CardTitle>
                  <CardDescription className="text-emerald-100">
                    RAG-optimized document processing with semantic awareness
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="bg-gradient-to-br from-slate-900 via-emerald-900 to-teal-900 rounded-b-lg overflow-hidden">
                    <div className="flex items-center justify-between p-4 border-b border-slate-700">
                      <span className="text-slate-200 text-sm font-medium">Semantic Chunking & Embedding</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyToClipboard(vectorEmbeddingCode, "vector")}
                        className="text-slate-300 hover:text-white hover:bg-white/10"
                      >
                        {copiedCode === "vector" ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      </Button>
                    </div>
                    <pre className="p-6 text-sm text-slate-200 overflow-x-auto leading-relaxed">
                      <code className="language-python">
                        <span className="text-purple-400">from</span> <span className="text-blue-300">semanticore.vectorizer</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">SemanticChunker</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Semantic-aware chunking</span>
                        {"\n"}
                        <span className="text-blue-300">chunker</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">SemanticChunker</span><span className="text-slate-300">(</span>
                        {"\n"}    <span className="text-green-300">chunk_size</span><span className="text-purple-400">=</span><span className="text-orange-300">512</span><span className="text-slate-300">,</span>
                        {"\n"}    <span className="text-green-300">respect_boundaries</span><span className="text-purple-400">=</span><span className="text-cyan-300">True</span><span className="text-slate-300">,</span>  <span className="text-gray-400"># Don't split entities</span>
                        {"\n"}    <span className="text-green-300">add_metadata</span><span className="text-purple-400">=</span><span className="text-cyan-300">True</span>
                        {"\n"}<span className="text-slate-300">)</span>
                        {"\n\n"}
                        <span className="text-blue-300">chunks</span> <span className="text-purple-400">=</span> <span className="text-blue-300">chunker</span><span className="text-slate-300">.</span><span className="text-yellow-300">chunk_document</span><span className="text-slate-300">(</span><span className="text-blue-300">document</span><span className="text-slate-300">)</span>
                      </code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="graph" className="mt-8">
              <Card className="border-0 shadow-xl">
                <CardHeader className="bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-t-lg">
                  <CardTitle className="flex items-center">
                    <Network className="w-6 h-6 mr-2" />
                    Knowledge Graph Export
                  </CardTitle>
                  <CardDescription className="text-orange-100">
                    Transform extracted semantics into queryable graph databases
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="bg-gradient-to-br from-slate-900 via-orange-900 to-red-900 rounded-b-lg overflow-hidden">
                    <div className="flex items-center justify-between p-4 border-b border-slate-700">
                      <span className="text-slate-200 text-sm font-medium">Graph Database Integration</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyToClipboard(knowledgeGraphCode, "graph")}
                        className="text-slate-300 hover:text-white hover:bg-white/10"
                      >
                        {copiedCode === "graph" ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      </Button>
                    </div>
                    <pre className="p-6 text-sm text-slate-200 overflow-x-auto leading-relaxed">
                      <code className="language-python">
                        <span className="text-purple-400">from</span> <span className="text-blue-300">semanticore.kg</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">Neo4jExporter</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Neo4j export with Cypher generation</span>
                        {"\n"}
                        <span className="text-blue-300">neo4j_exporter</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">Neo4jExporter</span><span className="text-slate-300">(</span>
                        {"\n"}    <span className="text-green-300">uri</span><span className="text-purple-400">=</span><span className="text-orange-300">"bolt://localhost:7687"</span><span className="text-slate-300">,</span>
                        {"\n"}    <span className="text-green-300">user</span><span className="text-purple-400">=</span><span className="text-orange-300">"neo4j"</span>
                        {"\n"}<span className="text-slate-300">)</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Create nodes and relationships</span>
                        {"\n"}
                        <span className="text-blue-300">neo4j_exporter</span><span className="text-slate-300">.</span><span className="text-yellow-300">export_entities</span><span className="text-slate-300">(</span><span className="text-blue-300">entities</span><span className="text-slate-300">)</span>
                        {"\n"}
                        <span className="text-blue-300">neo4j_exporter</span><span className="text-slate-300">.</span><span className="text-yellow-300">export_relations</span><span className="text-slate-300">(</span><span className="text-blue-300">relations</span><span className="text-slate-300">)</span>
                      </code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="advanced" className="mt-8">
              <Card className="border-0 shadow-xl">
                <CardHeader className="bg-gradient-to-r from-pink-500 to-pink-600 text-white rounded-t-lg">
                  <CardTitle className="flex items-center">
                    <Sparkles className="w-6 h-6 mr-2" />
                    Advanced Features
                  </CardTitle>
                  <CardDescription className="text-pink-100">
                    Real-time processing, context engineering, and domain-specific capabilities
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="bg-gradient-to-br from-slate-900 via-pink-900 to-purple-900 rounded-b-lg overflow-hidden">
                    <div className="flex items-center justify-between p-4 border-b border-slate-700">
                      <span className="text-slate-200 text-sm font-medium">Advanced Processing Pipeline</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyToClipboard(advancedFeaturesCode, "advanced")}
                        className="text-slate-300 hover:text-white hover:bg-white/10"
                      >
                        {copiedCode === "advanced" ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      </Button>
                    </div>
                    <pre className="p-6 text-sm text-slate-200 overflow-x-auto leading-relaxed">
                      <code className="language-python">
                        <span className="text-purple-400">from</span> <span className="text-blue-300">semanticore.streaming</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">SemanticStreamProcessor</span>
                        {"\n"}
                        <span className="text-purple-400">from</span> <span className="text-blue-300">semanticore.context</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">ContextEngineer</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Real-time semantic processing</span>
                        {"\n"}
                        <span className="text-blue-300">stream_processor</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">SemanticStreamProcessor</span><span className="text-slate-300">(</span>
                        {"\n"}    <span className="text-green-300">input_streams</span><span className="text-purple-400">=</span><span className="text-slate-300">[</span><span className="text-orange-300">"kafka://events"</span><span className="text-slate-300">],</span>
                        {"\n"}    <span className="text-green-300">processing_pipeline</span><span className="text-purple-400">=</span><span className="text-slate-300">[</span>
                        {"\n"}        <span className="text-orange-300">"entity_extraction"</span><span className="text-slate-300">,</span>
                        {"\n"}        <span className="text-orange-300">"relationship_detection"</span><span className="text-slate-300">,</span>
                        {"\n"}        <span className="text-orange-300">"knowledge_graph_update"</span>
                        {"\n"}    <span className="text-slate-300">]</span>
                        {"\n"}<span className="text-slate-300">)</span>
                        {"\n\n"}
                        <span className="text-gray-400"># Advanced context engineering</span>
                        {"\n"}
                        <span className="text-blue-300">context_engineer</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">ContextEngineer</span><span className="text-slate-300">(</span>
                        {"\n"}    <span className="text-green-300">compression_strategy</span><span className="text-purple-400">=</span><span className="text-orange-300">"semantic_preservation"</span>
                        {"\n"}<span className="text-slate-300">)</span>
                      </code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="integrations" className="mt-8">
              <Card className="border-0 shadow-xl">
                <CardHeader className="bg-gradient-to-r from-indigo-500 to-indigo-600 text-white rounded-t-lg">
                  <CardTitle className="flex items-center">
                    <Workflow className="w-6 h-6 mr-2" />
                    Framework Integrations
                  </CardTitle>
                  <CardDescription className="text-indigo-100">
                    Seamless integration with popular AI frameworks and tools
                  </CardDescription>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="bg-gradient-to-br from-slate-900 via-indigo-900 to-blue-900 rounded-b-lg overflow-hidden">
                    <div className="flex items-center justify-between p-4 border-b border-slate-700">
                      <span className="text-slate-200 text-sm font-medium">LangChain, LlamaIndex & CrewAI</span>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => copyToClipboard(integrationCode, "integrations")}
                        className="text-slate-300 hover:text-white hover:bg-white/10"
                      >
                        {copiedCode === "integrations" ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                      </Button>
                    </div>
                    <pre className="p-6 text-sm text-slate-200 overflow-x-auto leading-relaxed">
                      <code className="language-python">
                        <span className="text-gray-400"># LangChain Integration</span>
                        {"\n"}
                        <span className="text-purple-400">from</span> <span className="text-blue-300">semanticore.integrations.langchain</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">SemanticChain</span>
                        {"\n\n"}
                        <span className="text-blue-300">semantic_chain</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">SemanticChain</span><span className="text-slate-300">(</span>
                        {"\n"}    <span className="text-green-300">semanticore_instance</span><span className="text-purple-400">=</span><span className="text-blue-300">core</span><span className="text-slate-300">,</span>
                        {"\n"}    <span className="text-green-300">context_engineering</span><span className="text-purple-400">=</span><span className="text-cyan-300">True</span>
                        {"\n"}<span className="text-slate-300">)</span>
                        {"\n\n"}
                        <span className="text-gray-400"># CrewAI Integration</span>
                        {"\n"}
                        <span className="text-purple-400">from</span> <span className="text-blue-300">crewai</span> <span className="text-purple-400">import</span> <span className="text-yellow-300">Agent</span>
                        {"\n\n"}
                        <span className="text-blue-300">researcher</span> <span className="text-purple-400">=</span> <span className="text-yellow-300">Agent</span><span className="text-slate-300">(</span>
                        {"\n"}    <span className="text-green-300">role</span><span className="text-purple-400">=</span><span className="text-orange-300">'Research Analyst'</span><span className="text-slate-300">,</span>
                        {"\n"}    <span className="text-green-300">semantic_memory</span><span className="text-purple-400">=</span><span className="text-blue-300">core</span><span className="text-slate-300">.</span><span className="text-yellow-300">get_semantic_memory</span><span className="text-slate-300">()</span>
                        {"\n"}<span className="text-slate-300">)</span>
                      </code>
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </section>

      {/* Use Cases */}
      <section id="examples" className="py-20 bg-slate-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4">Domain-Specific Applications</h2>
            <p className="text-xl text-slate-600">Specialized processing for different industries and use cases</p>
          </div>

          <Tabs defaultValue="cyber" className="max-w-6xl mx-auto">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="cyber" className="flex items-center">
                <Shield className="w-4 h-4 mr-2" />
                Cybersecurity
              </TabsTrigger>
              <TabsTrigger value="finance" className="flex items-center">
                <BarChart3 className="w-4 h-4 mr-2" />
                Finance
              </TabsTrigger>
              <TabsTrigger value="health" className="flex items-center">
                <Activity className="w-4 h-4 mr-2" />
                Healthcare
              </TabsTrigger>
            </TabsList>

            <TabsContent value="cyber" className="mt-8">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Shield className="w-6 h-6 mr-2 text-red-500" />
                    Cybersecurity Threat Intelligence
                  </CardTitle>
                  <CardDescription>
                    Extract threat actors, malware, vulnerabilities, and attack patterns from security reports
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-slate-900 rounded-lg p-4">
                      <pre className="text-sm text-slate-300">
{`from semanticore.domains.cyber import ThreatIntelExtractor

threat_extractor = ThreatIntelExtractor()
threat_report = """
APT29 (Cozy Bear) launched a sophisticated 
spear-phishing campaign targeting US government 
agencies using FrostBite malware variant.
The attack exploited CVE-2024-1234 in 
Microsoft Exchange servers.
"""

intel = threat_extractor.extract(threat_report)
print(intel.threat_actors)    # ["APT29", "Cozy Bear"]
print(intel.malware)          # ["FrostBite"]
print(intel.vulnerabilities)  # ["CVE-2024-1234"]
print(intel.attack_patterns)  # ["spear-phishing"]

# Export to STIX format
stix_bundle = intel.to_stix()`}
                      </pre>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>STIX format export</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>MITRE ATT&CK mapping</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>IoC extraction</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>CVE tracking</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Threat hunting integration</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="finance" className="mt-8">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <BarChart3 className="w-6 h-6 mr-2 text-green-500" />
                    Financial Document Analysis
                  </CardTitle>
                  <CardDescription>
                    Extract financial metrics, events, and insights from earnings reports and financial documents
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-slate-900 rounded-lg p-4">
                      <pre className="text-sm text-slate-300">
{`from semanticore.domains.finance import FinancialExtractor

fin_extractor = FinancialExtractor()
earnings_report = """
Q4 2024 revenue increased 15% YoY to $2.3B, 
driven by strong cloud computing performance. 
Operating margin improved to 23.5% from 21.2%.
The company announced a $1B share buyback program.
"""

financial_data = fin_extractor.extract(earnings_report)
print(financial_data.metrics)     # {"revenue": "$2.3B"}
print(financial_data.periods)     # ["Q4 2024"]
print(financial_data.events)      # ["$1B share buyback"]

# Export to analysis tools
json_data = financial_data.to_standardized_json()`}
                      </pre>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Revenue & margin extraction</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Sentiment analysis</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Event detection</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Market data integration</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Risk factor analysis</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="health" className="mt-8">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Activity className="w-6 h-6 mr-2 text-blue-500" />
                    Biomedical Research Assistant
                  </CardTitle>
                  <CardDescription>
                    Extract drugs, conditions, outcomes, and clinical data from research papers and medical documents
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-slate-900 rounded-lg p-4">
                      <pre className="text-sm text-slate-300">
{`from semanticore.domains.biomedical import BiomedicalExtractor

bio_extractor = BiomedicalExtractor()
research_text = """
The study investigated remdesivir efficacy in 
COVID-19 patients. Results showed 31% reduction 
in recovery time (p<0.001). Side effects 
included nausea in 12% of patients.
"""

bio_data = bio_extractor.extract(research_text)
print(bio_data.drugs)         # ["remdesivir"]
print(bio_data.conditions)    # ["COVID-19"]
print(bio_data.outcomes)      # ["31% reduction"]
print(bio_data.side_effects)  # ["nausea"]

# Generate clinical schema
schema = bio_data.to_clinical_schema()`}
                      </pre>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Drug interaction detection</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>SNOMED/ICD-10 mapping</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Clinical schema generation</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Adverse event tracking</span>
                      </div>
                      <div className="flex items-center">
                        <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                        <span>Research data integration</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </section>

      {/* Integrations */}
      <section id="integrations" className="py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4">Seamless Integrations</h2>
            <p className="text-xl text-slate-600">Works with your existing AI stack and infrastructure</p>
          </div>

          <div className="grid md:grid-cols-4 gap-8 mb-12">
            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <Cpu className="w-12 h-12 text-blue-600 mx-auto mb-2" />
                <CardTitle className="text-lg">LLM Providers</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm text-slate-600">
                  <div>OpenAI GPT-4</div>
                  <div>Anthropic Claude</div>
                  <div>Google Gemini</div>
                  <div>Hugging Face</div>
                  <div>Local Models</div>
                </div>
              </CardContent>
            </Card>

            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <Database className="w-12 h-12 text-green-600 mx-auto mb-2" />
                <CardTitle className="text-lg">Vector Stores</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm text-slate-600">
                  <div>Pinecone</div>
                  <div>Milvus</div>
                  <div>Weaviate</div>
                  <div>Chroma</div>
                  <div>FAISS</div>
                </div>
              </CardContent>
            </Card>

            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <Network className="w-12 h-12 text-purple-600 mx-auto mb-2" />
                <CardTitle className="text-lg">Graph Databases</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm text-slate-600">
                  <div>Neo4j</div>
                  <div>KuzuDB</div>
                  <div>ArangoDB</div>
                  <div>Amazon Neptune</div>
                  <div>RDF Triple Stores</div>
                </div>
              </CardContent>
            </Card>

            <Card className="text-center hover:shadow-lg transition-shadow">
              <CardHeader>
                <Workflow className="w-12 h-12 text-orange-600 mx-auto mb-2" />
                <CardTitle className="text-lg">AI Frameworks</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm text-slate-600">
                  <div>LangChain</div>
                  <div>LlamaIndex</div>
                  <div>CrewAI</div>
                  <div>AutoGen</div>
                  <div>Custom Agents</div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-2xl p-8">
            <div className="text-center">
              <h3 className="text-2xl font-bold text-slate-900 mb-4">Universal Compatibility</h3>
              <p className="text-slate-600 mb-6 max-w-2xl mx-auto">
                SemantiCore is designed to work with your existing infrastructure. No need to rip and replace - just plug in and enhance your current AI pipeline.
              </p>
              <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                View All Integrations
                <ExternalLink className="w-4 h-4 ml-2" />
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section className="py-20 bg-slate-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4">Modular Architecture</h2>
            <p className="text-xl text-slate-600">Extensible design that grows with your needs</p>
          </div>

          <div className="max-w-4xl mx-auto">
            <Card className="p-8">
              <div className="text-center mb-8">
                <h3 className="text-2xl font-bold text-slate-900 mb-2">SemantiCore Engine</h3>
                <p className="text-slate-600">Modular, extensible architecture for enterprise-scale deployments</p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div className="text-center">
                  <div className="bg-blue-100 p-4 rounded-lg mb-3">
                    <FileText className="w-8 h-8 text-blue-600 mx-auto" />
                  </div>
                  <h4 className="font-semibold text-slate-900 mb-2">Connectors</h4>
                  <div className="text-sm text-slate-600 space-y-1">
                    <div>File</div>
                    <div>Web</div>
                    <div>API</div>
                    <div>Database</div>
                  </div>
                </div>

                <div className="text-center">
                  <div className="bg-purple-100 p-4 rounded-lg mb-3">
                    <Brain className="w-8 h-8 text-purple-600 mx-auto" />
                  </div>
                  <h4 className="font-semibold text-slate-900 mb-2">Extractors</h4>
                  <div className="text-sm text-slate-600 space-y-1">
                    <div>NER</div>
                    <div>Relations</div>
                    <div>Topics</div>
                    <div>LLM</div>
                  </div>
                </div>

                <div className="text-center">
                  <div className="bg-green-100 p-4 rounded-lg mb-3">
                    <Code2 className="w-8 h-8 text-green-600 mx-auto" />
                  </div>
                  <h4 className="font-semibold text-slate-900 mb-2">Schema</h4>
                  <div className="text-sm text-slate-600 space-y-1">
                    <div>Pydantic</div>
                    <div>JSON</div>
                    <div>YAML</div>
                    <div>TypeScript</div>
                  </div>
                </div>

                <div className="text-center">
                  <div className="bg-orange-100 p-4 rounded-lg mb-3">
                    <Database className="w-8 h-8 text-orange-600 mx-auto" />
                  </div>
                  <h4 className="font-semibold text-slate-900 mb-2">Exporters</h4>
                  <div className="text-sm text-slate-600 space-y-1">
                    <div>Neo4j</div>
                    <div>RDF</div>
                    <div>Vector DB</div>
                    <div>Custom</div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-red-50 p-4 rounded-lg text-center">
                  <Shield className="w-8 h-8 text-red-600 mx-auto mb-2" />
                  <h4 className="font-semibold text-slate-900">Validation & QA</h4>
                  <p className="text-sm text-slate-600 mt-1">Schema validation, consistency checking, quality metrics</p>
                </div>

                <div className="bg-pink-50 p-4 rounded-lg text-center">
                  <Workflow className="w-8 h-8 text-pink-600 mx-auto mb-2" />
                  <h4 className="font-semibold text-slate-900">Semantic Routing</h4>
                  <p className="text-sm text-slate-600 mt-1">Intelligent query routing and agent orchestration</p>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </section>

      {/* Open Source Community */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-slate-900 mb-4">Thriving Open Source Community</h2>
            <p className="text-xl text-slate-600">Join thousands of developers building the future of semantic AI</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 mb-12">
            <Card className="text-center p-6">
              <CardHeader>
                <Users className="w-12 h-12 text-blue-600 mx-auto mb-4" />
                <CardTitle>Active Contributors</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-blue-600 mb-2">50+</div>
                <p className="text-slate-600">Developers from around the world contributing code, documentation, and ideas</p>
              </CardContent>
            </Card>

            <Card className="text-center p-6">
              <CardHeader>
                <MessageSquare className="w-12 h-12 text-green-600 mx-auto mb-4" />
                <CardTitle>Community Support</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-600 mb-2">24/7</div>
                <p className="text-slate-600">Discord community providing help, sharing knowledge, and collaborative development</p>
              </CardContent>
            </Card>

            <Card className="text-center p-6">
              <CardHeader>
                <Award className="w-12 h-12 text-purple-600 mx-auto mb-4" />
                <CardTitle>MIT Licensed</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-purple-600 mb-2">100%</div>
                <p className="text-slate-600">Free and open source forever. Use commercially, modify, and distribute freely</p>
              </CardContent>
            </Card>
          </div>

          <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-2xl p-8 text-white">
            <div className="grid md:grid-cols-2 gap-8 items-center">
              <div>
                <h3 className="text-2xl font-bold mb-4">Join the Community</h3>
                <p className="text-slate-300 mb-6">
                  Connect with developers, share your projects, get help, and contribute to the future of semantic AI processing.
                </p>
                <div className="flex flex-wrap gap-4">
                  <Button variant="outline" className="text-white border-white hover:bg-white hover:text-slate-900">
                    <Github className="w-4 h-4 mr-2" />
                    GitHub
                  </Button>
                  <Button variant="outline" className="text-white border-white hover:bg-white hover:text-slate-900">
                    <MessageSquare className="w-4 h-4 mr-2" />
                    Discord
                  </Button>
                </div>
              </div>
              <div className="text-center">
                <div className="bg-white/10 rounded-lg p-6">
                  <Heart className="w-16 h-16 text-red-400 mx-auto mb-4" />
                  <p className="text-slate-300">
                    "SemantiCore has revolutionized how we process unstructured data. The community support is incredible!"
                  </p>
                  <p className="text-sm text-slate-400 mt-2">- AI Developer, Fortune 500</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 text-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold mb-4">Ready to Transform Your Data?</h2>
          <p className="text-xl text-blue-100 mb-8 max-w-2xl mx-auto">
            Join thousands of developers building intelligent AI systems with SemantiCore's unified semantic processing toolkit
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
            <Button size="lg" variant="secondary" className="bg-white text-blue-600 hover:bg-blue-50 shadow-lg">
              <Terminal className="w-5 h-5 mr-2" />
              pip install semanticore
            </Button>
            <Button size="lg" variant="outline" className="text-white border-white hover:bg-white hover:text-blue-600">
              <BookOpen className="w-5 h-5 mr-2" />
              Read the Docs
            </Button>
            <Button size="lg" variant="outline" className="text-white border-white hover:bg-white hover:text-blue-600">
              <PlayCircle className="w-5 h-5 mr-2" />
              Try Demo
            </Button>
          </div>

          <div className="flex justify-center space-x-6 text-blue-100">
            <a href="#" className="hover:text-white transition-colors flex items-center">
              <Github className="w-5 h-5 mr-2" />
              GitHub
            </a>
            <a href="#" className="hover:text-white transition-colors flex items-center">
              <MessageSquare className="w-5 h-5 mr-2" />
              Discord
            </a>
            <a href="#" className="hover:text-white transition-colors flex items-center">
              <BookOpen className="w-5 h-5 mr-2" />
              Documentation
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-900 text-slate-300 py-12">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center space-x-3 mb-4">
                <img 
                  src="/lovable-uploads/94bf75c5-552c-44a9-a37f-6cf436463006.png" 
                  alt="Hawksight AI" 
                  className="h-8 w-8 rounded-lg"
                />
                <div>
                  <div className="flex items-center space-x-2">
                    <Brain className="h-6 w-6 text-blue-400" />
                    <span className="text-xl font-bold text-white">SemantiCore</span>
                  </div>
                  <span className="text-xs text-slate-400">by Hawksight AI</span>
                </div>
              </div>
              <p className="text-slate-400 mb-4">
                Unified toolkit for transforming unstructured data into semantic layers. Built for society by Hawksight AI.
              </p>
              <div className="flex space-x-4">
                <Github className="w-5 h-5 hover:text-white cursor-pointer" />
                <MessageSquare className="w-5 h-5 hover:text-white cursor-pointer" />
              </div>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Product</h4>
              <ul className="space-y-2">
                <li><a href="#" className="hover:text-white">Features</a></li>
                <li><a href="#" className="hover:text-white">Integrations</a></li>
                <li><a href="#" className="hover:text-white">Examples</a></li>
                <li><a href="#" className="hover:text-white">Roadmap</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Resources</h4>
              <ul className="space-y-2">
                <li><a href="#" className="hover:text-white">Documentation</a></li>
                <li><a href="#" className="hover:text-white">API Reference</a></li>
                <li><a href="#" className="hover:text-white">Tutorials</a></li>
                <li><a href="#" className="hover:text-white">Best Practices</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="text-white font-semibold mb-4">Community</h4>
              <ul className="space-y-2">
                <li><a href="#" className="hover:text-white">Discord</a></li>
                <li><a href="#" className="hover:text-white">GitHub</a></li>
                <li><a href="#" className="hover:text-white">Contributing</a></li>
                <li><a href="#" className="hover:text-white">Code of Conduct</a></li>
              </ul>
            </div>
          </div>
          
          <div className="border-t border-slate-800 mt-8 pt-8 text-center text-slate-400">
            <p>&copy; 2025 Hawksight AI. SemantiCore is released under the MIT License.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Index;
