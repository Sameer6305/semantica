# Visualization

> **Comprehensive visualization suite for Knowledge Graphs, Ontologies, Embeddings, and Temporal data.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-graph:{ .lg .middle } **KG Visualization**

    ---

    Interactive network graphs with Force-directed, Hierarchical, and Circular layouts

-   :material-file-tree:{ .lg .middle } **Ontology View**

    ---

    Visualize class hierarchies, property domains/ranges, and taxonomy trees

-   :material-chart-scatter-plot:{ .lg .middle } **Embedding Projector**

    ---

    2D/3D visualization of vector embeddings using UMAP, t-SNE, and PCA

-   :material-clock-time-four-outline:{ .lg .middle } **Temporal Analysis**

    ---

    Timeline views and graph evolution visualization

-   :material-chart-bar:{ .lg .middle } **Analytics Dashboards**

    ---

    Visual dashboards for centrality, community structure, and connectivity

-   :material-export:{ .lg .middle } **Multi-Format Export**

    ---

    Export to HTML (interactive), PNG, SVG, PDF, and JSON

</div>

!!! tip "When to Use"
    - **Exploration**: Interactively explore graph connections and clusters
    - **Reporting**: Generate static charts for reports and presentations
    - **Debugging**: Visually inspect graph structure and disconnected components
    - **Analysis**: Identify patterns, outliers, and trends in data

---

## ⚙️ Algorithms Used

### Layout Algorithms
- **Force-Directed**: Simulates physical forces (repulsion between nodes, springs for edges) to find equilibrium.
- **Hierarchical**: Tree-based layout for taxonomies and directed acyclic graphs (DAGs).
- **Circular**: Arranges nodes in a circle, useful for analyzing interconnectivity.
- **Community-Based**: Groups nodes by community (Louvain/Leiden) and separates clusters.

### Dimensionality Reduction
- **UMAP**: Uniform Manifold Approximation and Projection. Preserves global structure better than t-SNE.
- **t-SNE**: t-Distributed Stochastic Neighbor Embedding. Good for local clustering.
- **PCA**: Principal Component Analysis. Linear projection for variance maximization.

### Analytics Visualization
- **Centrality Sizing**: Node size proportional to Degree/Betweenness/PageRank.
- **Heatmaps**: Matrix visualization for adjacency or similarity.
- **Sankey Diagrams**: Flow visualization for lineage or process steps.

---

## Main Classes

### KGVisualizer

Visualizes Knowledge Graph structure and communities.

**Methods:**

| Method | Description |
|--------|-------------|
| `visualize_network(graph)` | Standard network plot |
| `visualize_communities(graph)` | Color by community |
| `visualize_path(path)` | Highlight specific path |

**Example:**

```python
from semantica.visualization import KGVisualizer

viz = KGVisualizer(layout="force", height=800)
fig = viz.visualize_network(kg, output="interactive")
fig.write_html("graph.html")
```

### OntologyVisualizer

Visualizes schema and taxonomy.

**Methods:**

| Method | Description |
|--------|-------------|
| `visualize_hierarchy(ontology)` | Tree view of classes |
| `visualize_properties(ontology)` | Property domain/range graph |

### EmbeddingVisualizer

Project high-dimensional vectors to 2D/3D.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `visualize_2d(embeddings)` | 2D Scatter plot | UMAP/t-SNE |
| `visualize_3d(embeddings)` | 3D Scatter plot | UMAP/t-SNE |
| `visualize_clusters(embeddings)` | Colored by cluster | K-Means/DBSCAN |

**Example:**

```python
from semantica.visualization import EmbeddingVisualizer

viz = EmbeddingVisualizer()
viz.visualize_2d_projection(
    embeddings, 
    labels=labels, 
    method="umap",
    output="embeddings.html"
)
```

### TemporalVisualizer

Visualizes time-series and graph evolution.

**Methods:**

| Method | Description |
|--------|-------------|
| `visualize_timeline(events)` | Event timeline |
| `visualize_evolution(snapshots)` | Graph changes over time |

---

## Convenience Functions

```python
from semantica.visualization import visualize_kg, visualize_embeddings

# One-line visualization
visualize_kg(kg, output="graph.html")
visualize_embeddings(embeddings, method="umap")
```

---

## Configuration

### Environment Variables

```bash
export VIZ_DEFAULT_LAYOUT=force
export VIZ_COLOR_SCHEME=vibrant
export VIZ_RENDERER=plotly
```

### YAML Configuration

```yaml
visualization:
  layout:
    algorithm: force
    iterations: 50
    
  style:
    node_size: 10
    edge_width: 1
    color_scheme: "vibrant" # vibrant, pastel, dark
    
  export:
    width: 1200
    height: 800
    scale: 2.0
```

---

## Integration Examples

### Exploratory Data Analysis (EDA)

```python
from semantica.ingest import Ingestor
from semantica.kg import KnowledgeGraph
from semantica.visualization import KGVisualizer, AnalyticsVisualizer

# 1. Load Data
kg = KnowledgeGraph.load("my_graph")

# 2. Visualize Structure
kg_viz = KGVisualizer()
kg_viz.visualize_network(kg, output="structure.html")

# 3. Visualize Analytics
analytics_viz = AnalyticsVisualizer()
analytics_viz.visualize_centrality(kg, output="centrality.png")
analytics_viz.visualize_degree_distribution(kg, output="degree_dist.png")
```

---

## Best Practices

1.  **Filter First**: Don't try to visualize 1M nodes. Filter to a subgraph of <5000 nodes for readability.
2.  **Use Interactive**: Interactive HTML plots (Plotly) allow zooming and hovering, which is essential for dense graphs.
3.  **Color Meaningfully**: Use color to represent node types or communities, not just random assignment.
4.  **Size by Importance**: Map node size to centrality (e.g., PageRank) to highlight important entities.

---

## See Also

- [Knowledge Graph Module](kg.md) - The data source
- [Embeddings Module](embeddings.md) - Source for vector visualizations
- [Ontology Module](ontology.md) - Source for hierarchy visualizations
