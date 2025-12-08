import json

# Read the notebook
with open(r'c:\Users\Mohd Kaif\semantica\cookbook\use_cases\advanced_rag\01_GraphRAG_Complete.ipynb', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find and update the cell that uses build function (cell 14, lines 522-546)
# Replace the build function usage with class-based approach
data['cells'][14]['source'] = [
    "from semantica.semantic_extract import NamedEntityRecognizer, RelationExtractor, TripleExtractor\n",
    "\n",
    "print(\"Extracting entities, relationships, and triples...\")\n",
    "\n",
    "ner = NamedEntityRecognizer()\n",
    "rel_extractor = RelationExtractor()\n",
    "triple_extractor = TripleExtractor()\n",
    "\n",
    "flat_entities = []\n",
    "flat_relationships = []\n",
    "flat_triples = []\n",
    "\n",
    "for doc in normalized_documents:\n",
    "    text = str(doc.content) if hasattr(doc, 'content') else str(doc)\n",
    "    \n",
    "    entities = ner.extract_entities(text)\n",
    "    flat_entities.extend(entities if isinstance(entities, list) else [entities])\n",
    "    \n",
    "    relations = rel_extractor.extract_relations(text, entities=entities)\n",
    "    flat_relationships.extend(relations if isinstance(relations, list) else [relations])\n",
    "    \n",
    "    triples = triple_extractor.extract_triples(text, entities=entities, relationships=relations)\n",
    "    flat_triples.extend(triples if isinstance(triples, list) else [triples])\n",
    "\n",
    "print(f\"Extracted {len(flat_entities)} entities\")\n",
    "print(f\"Extracted {len(flat_relationships)} relationships\")\n",
    "print(f\"Extracted {len(flat_triples)} triples\")\n"
]

# Write back the notebook
with open(r'c:\Users\Mohd Kaif\semantica\cookbook\use_cases\advanced_rag\01_GraphRAG_Complete.ipynb', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)

print("Updated notebook successfully!")
