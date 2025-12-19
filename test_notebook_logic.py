
import logging
import json
from datetime import datetime

# Set up logging to see what's happening under the hood
# Using WARNING to keep the output clean for the notebook demonstration
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Import all the powerful tools from Semantica
from semantica.kg import (
    GraphBuilder,
    GraphAnalyzer,
    GraphValidator,
    ConnectivityAnalyzer,
    CentralityCalculator,
    CommunityDetector,
    TemporalGraphQuery,
    ProvenanceTracker
)
from semantica.deduplication import DuplicateDetector
from semantica.conflicts import ConflictDetector, ConflictResolver

# Our "Raw" Messy Data
raw_entities = [
    {"id": "startup_1", "type": "Startup", "name": "TechFlow AI", "revenue": 1000000, "founded": "2021-01-01"},
    {"id": "startup_2", "type": "Startup", "name": "GreenEnergy Co", "revenue": 500000, "founded": "2020-05-15"},
    {"id": "startup_1_dup", "type": "Startup", "name": "TechFlow Inc.", "revenue": 1200000, "founded": "2021-01-01"}, # Duplicate!
    {"id": "investor_1", "type": "Investor", "name": "Venture Capital X"},
    {"id": "founder_1", "type": "Person", "name": "Alice Chen"},
    {"id": "founder_2", "type": "Person", "name": "Bob Smith"}
]

raw_relationships = [
    # Valid Relationships
    {"source": "founder_1", "target": "startup_1", "type": "FOUNDED", "valid_from": "2021-01-01"},
    {"source": "investor_1", "target": "startup_1", "type": "INVESTED_IN", "amount": 5000000, "valid_from": "2023-06-01"},
    
    # Dangling Edge (Error!)
    {"source": "founder_2", "target": "startup_999", "type": "FOUNDED", "valid_from": "2020-05-15"}, 
    
    # Temporal Data (History)
    {"source": "founder_1", "target": "startup_2", "type": "ADVISED", "valid_from": "2020-01-01", "valid_until": "2021-01-01"}
]

print(f"Loaded {len(raw_entities)} raw entities and {len(raw_relationships)} raw relationships.")

# Initialize Validator
validator = GraphValidator()

# Create a temporary graph object for validation
temp_graph = {"entities": raw_entities, "relationships": raw_relationships}

# Run Validation
print("Running Validation Check...")
validation_result = validator.validate(temp_graph)

if not validation_result.is_valid:
    print("Validation Failed! Issues found:")
    for issue in validation_result.issues:
        print(f"   - [{issue.severity.name}] {issue.message} (Code: {issue.code})")
        
        # AUTOMATIC FIX: If it's a dangling edge, remove it
        if issue.code == "DANGLING_EDGE":
            print("     Auto-Fixing: Removing invalid relationship...")
            raw_relationships = [r for r in raw_relationships 
                               if r['target'] != issue.details.get('target_id')]
else:
    print("Graph is valid!")

# Re-validate to confirm fix
print("\nRe-validating after fixes...")
temp_graph = {"entities": raw_entities, "relationships": raw_relationships}
if validator.validate(temp_graph).is_valid:
    print("Graph is now clean and valid!")

# 1. Detect Duplicates
print("Scanning for duplicates...")
deduper = DuplicateDetector(similarity_threshold=0.7) # 70% similarity threshold
duplicates = deduper.detect_duplicates(raw_entities)

for candidate in duplicates:
    print(f"Found potential duplicate pair (Score: {candidate.similarity_score:.2f}):")
    print(f"   - {candidate.entity1['name']} (ID: {candidate.entity1['id']})")
    print(f"   - {candidate.entity2['name']} (ID: {candidate.entity2['id']})")
    
    # MERGE STRATEGY: Keep entity1, merge data from entity2
    print("   Merging entities...")
    # (In a real app, you'd use EntityMerger, but here's the logic:)
    # We keep startup_1 and discard startup_1_dup, but we note the conflict
    
# 2. Detect Conflicts
print("\nChecking for data conflicts...")
conflict_detector = ConflictDetector()

# Simulating a conflict check between the two versions of TechFlow
# To check conflicts, we treat them as the same entity (same ID)
entity_a = raw_entities[0].copy()
entity_b = raw_entities[2].copy()
entity_b['id'] = entity_a['id'] # Force same ID for conflict detection

conflicts = conflict_detector.detect_conflicts([entity_a, entity_b])

for conflict in conflicts:
    print(f"   Conflict detected in field '{conflict.property_name}':")
    print(f"      Values: {conflict.conflicting_values}")
    
    # RESOLUTION: Trust the higher number (optimistic!)
    if conflict.property_name == "revenue":
        # values are strings or ints, need to handle types
        vals = [float(v) for v in conflict.conflicting_values if v is not None]
        resolved_val = max(vals)
        print(f"      Resolved to: {resolved_val}")
        raw_entities[0]['revenue'] = resolved_val

# Final Cleanup: Remove the duplicate entity from our list
clean_entities = [e for e in raw_entities if e['id'] != 'startup_1_dup']
clean_relationships = raw_relationships # (We'd normally re-link relationships too)

print(f"\nCleaned Data: {len(clean_entities)} entities remaining.")


# Manual Graph Construction (since we already cleaned it)
kg = {
    "entities": clean_entities,
    "relationships": clean_relationships,
    "metadata": {
        "created_at": datetime.now().isoformat(),
        "source": "Manual Advanced Pipeline"
    }
}
print("Knowledge Graph Assembled Successfully!")

# Initialize the Master Analyzer
analyzer = GraphAnalyzer(enable_temporal=True)

# 1. Structural Analysis (Connectivity)
print("\n--- Connectivity Analysis ---")
connectivity = analyzer.analyze_connectivity(kg)
print(f"   • Graph Connected? {'Yes' if connectivity['is_connected'] else 'No'}")
print(f"   • Connected Components: {connectivity['num_components']}")

# 2. Centrality (Who is important?)
print("\n--- Centrality Analysis ---")
centrality_result = analyzer.calculate_centrality(kg, centrality_type="degree")
degree_data = centrality_result["centrality_measures"]["degree"]

# Get pre-calculated rankings
top_nodes = degree_data["rankings"][:3]

print("   • Top Influencers (Degree Centrality):")
for item in top_nodes:
    print(f"     - {item['node']}: {item['score']:.2f}")

# 3. Community Detection (Clustering)
print("\n--- Community Detection ---")
communities = analyzer.detect_communities(kg, algorithm="louvain")
community_result = communities
communities = community_result["communities"]

print(f"   • Detected {len(communities)} communities.")
for i, comm in enumerate(communities):
    # comm is a set of node IDs
    members = list(comm)
    print(f"     Community {i+1}: {', '.join(members)}")

temporal_engine = TemporalGraphQuery(temporal_granularity="year")

# 1. Time Travel Query: What did the world look like in 2020?
print("\n--- Time Travel: 2020 ---")
snapshot_2020 = temporal_engine.query_at_time(kg, query="*", at_time="2020-06-01")
print(f"   Active Relationships in 2020: {len(snapshot_2020['relationships'])}")
for rel in snapshot_2020['relationships']:
    print(f"   - {rel['source']} --[{rel['type']}]--> {rel['target']}")

# 2. Time Travel Query: What about 2023?
print("\n--- Time Travel: 2023 ---")
snapshot_2023 = temporal_engine.query_at_time(kg, query="*", at_time="2023-07-01")
print(f"   Active Relationships in 2023: {len(snapshot_2023['relationships'])}")
for rel in snapshot_2023['relationships']:
    print(f"   - {rel['source']} --[{rel['type']}]--> {rel['target']}")
    
# Notice how 'ADVISED' might disappear if it ended, and 'INVESTED_IN' appears!

tracker = ProvenanceTracker()

# Let's pretend we're tracking the source of our data
tracker.track_entity("startup_1", source="Crunchbase_API_v2", metadata={"confidence": 0.95})
tracker.track_entity("startup_1", source="Manual_Entry_User_Bob", metadata={"confidence": 1.0})

print("\n--- Provenance Report: TechFlow AI ---")
lineage = tracker.get_lineage("startup_1")
print(f"   Entity: startup_1")
print(f"   First Seen: {lineage['first_seen']}")
print(f"   Sources:")
for src in lineage['sources']:
    print(f"     - {src['source']} (at {src['timestamp']})")
