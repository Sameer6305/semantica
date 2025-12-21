# 🍳 Semantica Cookbook

Welcome to the **Semantica Cookbook**!

This collection of Jupyter notebooks is designed to take you from a beginner to an expert in building semantic AI applications. Whether you're looking for quick recipes or deep-dive tutorials, you'll find it here.

!!! tip "How to use this Cookbook"
    - **Beginners**: Start with the [Core Tutorials](#core-tutorials) to learn the basics.
    - **Developers**: Check out [Advanced Concepts](#advanced-concepts) for deep dives into specific features.
    - **Architects**: Explore [Industry Use Cases](#industry-use-cases) for end-to-end solutions.

!!! note "Prerequisites"
    Before running these notebooks, ensure you have:
    - Python 3.8+ installed
    - A basic understanding of Python and Jupyter
    - An OpenAI API key (for most examples)

!!! success "Installation"
    Install Semantica from PyPI (recommended):
    
    ```bash
    pip install semantica
    # Or with all optional dependencies:
    pip install semantica[all]
    ```
    
    For more installation options, see the [Installation Guide](installation.md).

---

## � Featured Recipes

Hand-picked tutorials to show you the power of Semantica.

<div class="grid cards" markdown>

-   :material-robot: **GraphRAG Complete**
    ---
    Build a production-ready Graph Retrieval Augmented Generation system.
    
    **Topics**: RAG, LLMs, Vector Search, Graph Traversal
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/advanced_rag/01_GraphRAG_Complete.ipynb)

-   :material-scale-balance: **RAG vs. GraphRAG Comparison**
    ---
    Side-by-side comparison of Standard RAG vs. GraphRAG using real-world data.
    
    **Topics**: RAG, GraphRAG, Benchmarking, Visualization
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/advanced_rag/02_RAG_vs_GraphRAG_Comparison.ipynb)

-   :material-graph: **Your First Knowledge Graph**
    ---
    Go from raw text to a queryable knowledge graph in 20 minutes.
    
    **Topics**: Extraction, Graph Construction, Visualization
    
    **Difficulty**: Beginner
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/08_Your_First_Knowledge_Graph.ipynb)

-   :material-shield-alert: **Real-Time Anomaly Detection**
    ---
    Detect anomalies in streaming data using dynamic graphs.
    
    **Topics**: Streaming, Security, Dynamic Graphs
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/cybersecurity/01_Anomaly_Detection_Real_Time.ipynb)

</div>

---

## 🏁 Core Tutorials {#core-tutorials}

Essential guides to master the Semantica framework.

<div class="grid cards" markdown>

-   :material-hand-wave: **Welcome to Semantica**
    ---
    An interactive introduction to the framework's core philosophy and all modules including ingestion, parsing, extraction, knowledge graphs, embeddings, and more.
    
    **Topics**: Framework Overview, Architecture, All Modules
    
    **Difficulty**: Beginner
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/01_Welcome_to_Semantica.ipynb)

-   :material-database-import: **Data Ingestion**
    ---
    Techniques for loading data from multiple sources using FileIngestor, WebIngestor, FeedIngestor, StreamIngestor, RepoIngestor, EmailIngestor, DBIngestor, and MCPIngestor.
    
    **Topics**: File Ingestion, Web Scraping, Database Integration, Streams, Feeds, Repositories, Email, MCP
    
    **Difficulty**: Beginner
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/02_Data_Ingestion.ipynb)

-   :material-file-document-outline: **Document Parsing**
    ---
    Extracting clean text from complex formats like PDF, DOCX, and HTML.
    
    **Topics**: OCR, PDF Parsing, Text Extraction
    
    **Difficulty**: Beginner
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/03_Document_Parsing.ipynb)

-   :material-broom: **Data Normalization**
    ---
    Pipelines for cleaning, normalizing, and preparing text.
    
    **Topics**: Text Cleaning, Unicode, Formatting
    
    **Difficulty**: Beginner
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/04_Data_Normalization.ipynb)

-   :material-account-search: **Entity Extraction**
    ---
    Using NER to identify people, organizations, and custom entities.
    
    **Topics**: NER, Spacy, LLM Extraction
    
    **Difficulty**: Beginner
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/05_Entity_Extraction.ipynb)

-   :material-relation-many-to-many: **Relation Extraction**
    ---
    Discovering and classifying relationships between entities.
    
    **Topics**: Relation Classification, Dependency Parsing
    
    **Difficulty**: Beginner
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/06_Relation_Extraction.ipynb)

-   :material-vector-square: **Embedding Generation**
    ---
    Creating and managing vector embeddings for semantic search.
    
    **Topics**: Embeddings, OpenAI, HuggingFace
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/12_Embedding_Generation.ipynb)

-   :material-database-search: **Vector Store**
    ---
    Setting up vector stores for similarity search and retrieval.
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/13_Vector_Store.ipynb)

-   :material-database-settings: **Graph Store**
    ---
    Persisting knowledge graphs in Neo4j or FalkorDB.
    
    **Topics**: Neo4j, Cypher, Persistence
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/09_Graph_Store.ipynb)

-   :material-sitemap: **Ontology**
    ---
    Defining domain schemas and ontologies to structure your data.
    
    **Topics**: OWL, RDF, Schema Design
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/14_Ontology.ipynb)

</div>

---

## 🧠 Advanced Concepts

Deep dive into advanced features, customization, and complex workflows.

<div class="grid cards" markdown>

-   :material-flask: **Advanced Extraction**
    ---
    Custom extractors, LLM-based extraction, and complex pattern matching.
    
    **Topics**: Custom Models, Regex, LLMs
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/01_Advanced_Extraction.ipynb)

-   :material-chart-network: **Advanced Graph Analytics**
    ---
    Centrality, community detection, and pathfinding algorithms.
    
    **Topics**: PageRank, Louvain, Shortest Path
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/02_Advanced_Graph_Analytics.ipynb)

-   :material-monitor-dashboard: **Complete Visualization Suite**
    ---
    Creating interactive, publication-ready visualizations of your graphs.
    
    **Topics**: PyVis, NetworkX, D3.js
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/03_Complete_Visualization_Suite.ipynb)

-   :material-scale-balance: **Conflict Resolution**
    ---
    Strategies for handling contradictory information from multiple sources.
    
    **Topics**: Truth Discovery, Voting, Confidence
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/17_Conflict_Detection_and_Resolution.ipynb)

-   :material-export: **Multi-Format Export**
    ---
    Exporting to RDF, OWL, JSON-LD, and NetworkX formats.
    
    **Topics**: Serialization, Interoperability
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/05_Multi_Format_Export.ipynb)

-   :material-source-merge: **Multi-Source Integration**
    ---
    Merging data from disparate sources into a unified graph.
    
    **Topics**: Entity Resolution, Merging, Fusion
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/06_Multi_Source_Data_Integration.ipynb)


-   :material-brain: **Reasoning and Inference**
    ---
    Using logical reasoning to infer new knowledge from existing facts.
    
    **Topics**: Logic Rules, Inference Engines
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/08_Reasoning_and_Inference.ipynb)

-   :material-layers: **Semantic Layer Construction**
    ---
    Building a semantic layer over your data warehouse or lake.
    
    **Topics**: Semantic Layer, Data Warehouse
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/09_Semantic_Layer_Construction.ipynb)

-   :material-clock-outline: **Temporal Knowledge Graphs**
    ---
    Modeling and querying data that changes over time.
    
    **Topics**: Time Series, Temporal Logic
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/10_Temporal_Knowledge_Graphs.ipynb)

</div>

---

## 🏭 Industry Use Cases {#industry-use-cases}

Real-world examples and end-to-end applications across various industries.

### Biomedical

<div class="grid cards" markdown>

-   :material-pill: **Drug Discovery Pipeline**
    ---
    Accelerating drug discovery by connecting genes, proteins, and drugs.
    
    **Topics**: Bioinformatics, KG Construction
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/biomedical/01_Drug_Discovery_Pipeline.ipynb)

-   :material-dna: **Genomic Variant Analysis**
    ---
    Analyzing genomic variants and their implications for disease.
    
    **Topics**: Genomics, Variant Calling
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/biomedical/02_Genomic_Variant_Analysis.ipynb)

</div>

### Healthcare

<div class="grid cards" markdown>

-   :material-hospital-box: **Clinical Reports Processing**
    ---
    Processing and structuring unstructured clinical reports.
    
    **Topics**: NLP, Medical Records
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/healthcare/01_Clinical_Reports_Processing.ipynb)

-   :material-virus: **Disease Network Analysis**
    ---
    Analyzing disease networks and comorbidities for population health.
    
    **Topics**: Disease Modeling, Comorbidity Networks
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/healthcare/02_Disease_Network_Analysis.ipynb)

-   :material-pill-multiple: **Drug Interactions Analysis**
    ---
    Identifying potential drug interactions and contraindications.
    
    **Topics**: Pharmacology, Drug Safety
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/healthcare/03_Drug_Interactions_Analysis.ipynb)

-   :material-robot-love: **Healthcare GraphRAG Hybrid**
    ---
    Hybrid RAG system for healthcare knowledge retrieval.
    
    **Topics**: RAG, Medical Knowledge, LLMs
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/healthcare/04_Healthcare_GraphRAG_Hybrid.ipynb)

-   :material-database-plus: **Medical Database Integration**
    ---
    Integrating multiple medical databases into unified knowledge graphs.
    
    **Topics**: Data Integration, Medical Databases
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/healthcare/05_Medical_Database_Integration.ipynb)

-   :material-account-heart: **Patient Records Temporal**
    ---
    Analyzing patient records over time to track health progression.
    
    **Topics**: Temporal Analysis, Patient Journeys
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/healthcare/06_Patient_Records_Temporal.ipynb)

</div>

### Finance

<div class="grid cards" markdown>

-   :material-finance: **Financial Data Integration**
    ---
    Merging financial data from reports, news, and market feeds.
    
    **Topics**: Finance, Data Fusion
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/finance/01_Financial_Data_Integration.ipynb)

-   :material-file-chart: **Financial Reports Analysis**
    ---
    Extracting insights from financial reports and earnings calls.
    
    **Topics**: Financial Analysis, NLP
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/finance/02_Financial_Reports_Analysis.ipynb)

-   :material-incognito: **Fraud Detection**
    ---
    Identifying fraudulent activities and patterns in transaction networks.
    
    **Topics**: Anomaly Detection, Graph Mining
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/finance/03_Fraud_Detection.ipynb)

-   :material-chart-box: **Investment Analysis Hybrid RAG**
    ---
    AI-powered investment analysis using hybrid RAG approach.
    
    **Topics**: Investment Research, RAG, Financial Analysis
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/finance/04_Investment_Analysis_Hybrid_RAG.ipynb)

-   :material-gavel: **Regulatory Compliance**
    ---
    Ensuring compliance with financial regulations using knowledge graphs.
    
    **Topics**: Compliance, Regulatory Analysis
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/finance/05_Regulatory_Compliance.ipynb)

</div>

### Blockchain

<div class="grid cards" markdown>

-   :material-bitcoin: **DeFi Protocol Intelligence**
    ---
    Analyzing decentralized finance protocols and transaction flows.
    
    **Topics**: Blockchain, DeFi, Smart Contracts
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/blockchain/01_DeFi_Protocol_Intelligence.ipynb)

-   :material-network: **Transaction Network Analysis**
    ---
    Mapping and analyzing blockchain transaction networks.
    
    **Topics**: Blockchain Analytics, Network Analysis
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/blockchain/02_Transaction_Network_Analysis.ipynb)

</div>

### Cybersecurity

<div class="grid cards" markdown>

-   :material-shield-alert: **Anomaly Detection Real-Time**
    ---
    Detecting anomalies in real-time network traffic streams.
    
    **Topics**: Network Security, Streaming
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/cybersecurity/01_Anomaly_Detection_Real_Time.ipynb)

-   :material-shield-search: **Incident Analysis**
    ---
    Analyzing security incidents and breaches using graph forensics.
    
    **Topics**: Incident Response, Forensics
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/cybersecurity/02_Incident_Analysis.ipynb)

-   :material-shield-link-variant: **Threat Correlation**
    ---
    Correlating threats across different vectors to identify campaigns.
    
    **Topics**: Threat Intel, Correlation
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/cybersecurity/03_Threat_Correlation.ipynb)

-   :material-robot-angry: **Threat Intelligence Hybrid RAG**
    ---
    Combining RAG with threat intelligence for enhanced security insights.
    
    **Topics**: Threat Intelligence, RAG, Security
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/cybersecurity/04_Threat_Intelligence_Hybrid_RAG.ipynb)

-   :material-shield-plus: **Threat Intelligence Integration**
    ---
    Integrating threat feeds into a unified knowledge graph.
    
    **Topics**: STIX/TAXII, Threat Feeds, Integration
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/cybersecurity/05_Threat_Intelligence_Integration.ipynb)

-   :material-bug: **Vulnerability Tracking**
    ---
    Tracking and managing system vulnerabilities using knowledge graphs.
    
    **Topics**: CVE, Vulnerability Management
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/cybersecurity/06_Vulnerability_Tracking.ipynb)

</div>

### Intelligence

<div class="grid cards" markdown>

-   :material-account-network: **Criminal Network Analysis**
    ---
    Analyze criminal networks with graph analytics and key player detection.
    
    **Topics**: Forensics, Social Network Analysis
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/intelligence/01_Criminal_Network_Analysis.ipynb)

-   :material-file-search: **Intelligence Analysis**
    ---
    Comprehensive intelligence analysis using orchestrator-worker pattern with graph analytics and hybrid RAG.
    
    **Topics**: Intelligence Analysis, Orchestrator-Worker, Graph Analytics
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/intelligence/02_Intelligence_Analysis.ipynb)

-   :material-gavel: **Law Enforcement Forensics**
    ---
    Forensic analysis pipeline for processing case files and evidence.
    
    **Topics**: Forensics, Evidence Analysis, Case Correlation
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/intelligence/03_Law_Enforcement_Forensics.ipynb)

</div>

### Trading

<div class="grid cards" markdown>

-   :material-chart-areaspline: **Market Data Analysis**
    ---
    Analyzing trading market data for patterns and opportunities.
    
    **Topics**: Trading, Market Analysis
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/trading/01_Market_Data_Analysis.ipynb)

-   :material-newspaper-variant: **News Sentiment Analysis**
    ---
    Analyzing news sentiment for trading signals and market predictions.
    
    **Topics**: Sentiment Analysis, Trading Signals
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/trading/02_News_Sentiment_Analysis.ipynb)

-   :material-monitor-dashboard: **Real-Time Monitoring**
    ---
    Monitoring trading systems and positions in real-time.
    
    **Topics**: Monitoring, Real-Time Systems
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/trading/03_Real_Time_Monitoring.ipynb)

-   :material-shield-check: **Risk Assessment**
    ---
    Assessing trading risks using knowledge graphs and analytics.
    
    **Topics**: Risk Management, Portfolio Analysis
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/trading/04_Risk_Assessment.ipynb)

-   :material-history: **Strategy Backtesting**
    ---
    Backtesting trading strategies using historical data and graphs.
    
    **Topics**: Backtesting, Strategy Optimization
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/trading/05_Strategy_Backtesting.ipynb)

</div>

### Renewable Energy

<div class="grid cards" markdown>

-   :material-wind-turbine: **Energy Market Analysis**
    ---
    Analyzing trends and pricing in the renewable energy market.
    
    **Topics**: Energy, Time Series
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/renewable_energy/01_Energy_Market_Analysis.ipynb)

-   :material-leaf: **Environmental Impact**
    ---
    Assessing environmental impact of energy projects and policies.
    
    **Topics**: Environmental Science, Impact Analysis
    
    **Difficulty**: Intermediate
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/renewable_energy/02_Environmental_Impact.ipynb)

-   :material-transmission-tower: **Grid Management**
    ---
    Optimizing power grid management and distribution.
    
    **Topics**: Grid Optimization, Energy Distribution
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/renewable_energy/03_Grid_Management.ipynb)

-   :material-solar-power: **Resource Optimization**
    ---
    Optimizing renewable energy resources and generation.
    
    **Topics**: Resource Management, Optimization
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/renewable_energy/04_Resource_Optimization.ipynb)

</div>

### Supply Chain

<div class="grid cards" markdown>

-   :material-truck-delivery: **Supply Chain Data Integration**
    ---
    Integrating supply chain data to optimize logistics and reduce risk.
    
    **Topics**: Logistics, Risk Management
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/supply_chain/01_Supply_Chain_Data_Integration.ipynb)

-   :material-alert-octagon: **Supply Chain Risk Management**
    ---
    Managing and mitigating supply chain risks using knowledge graphs.
    
    **Topics**: Risk Management, Supply Chain Resilience
    
    **Difficulty**: Advanced
    
    [Open Notebook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/use_cases/supply_chain/02_Supply_Chain_Risk_Management.ipynb)

</div>

---

## 🛠️ How to Run

To run these notebooks locally:

1.  **Install Semantica from PyPI** (recommended):
    ```bash
    pip install semantica[all]
    pip install jupyter
    ```

2.  **Or install from source** (for development):
    ```bash
    git clone https://github.com/Hawksight-AI/semantica.git
    cd semantica
    pip install -e .[all]
    pip install jupyter
    ```

3.  **Launch Jupyter**:
    ```bash
    jupyter notebook
    ```

!!! tip "Using Docker"
    You can also run the cookbook using Docker:
    ```bash
    docker run -p 8888:8888 hawksight/semantica-cookbook
    ```
