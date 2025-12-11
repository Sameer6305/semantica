"""
Ontology Generation Module

This module handles automatic generation of ontologies from data and text using
a 6-stage pipeline that transforms raw data into structured OWL ontologies.

Key Features:
    - Automatic ontology generation (6-stage pipeline)
    - Class and property inference
    - Ontology structure optimization
    - Domain-specific ontology creation
    - Ontology quality assessment
    - Semantic network parsing
    - Hierarchy generation

Main Classes:
    - OntologyGenerator: Main ontology generation class (6-stage pipeline)
    - ClassInferencer: Class inference engine (legacy alias)
    - PropertyInferencer: Property inference engine (legacy alias)
    - OntologyOptimizer: Ontology optimization engine

Example Usage:
    >>> from semantica.ontology import OntologyGenerator
    >>> generator = OntologyGenerator(base_uri="https://example.org/ontology/")
    >>> ontology = generator.generate_ontology({"entities": [...], "relationships": [...]})
    >>> classes = generator.infer_classes(data)
    >>> properties = generator.infer_properties(data, classes)

Author: Semantica Contributors
License: MIT
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .class_inferrer import ClassInferrer
from .namespace_manager import NamespaceManager
from .naming_conventions import NamingConventions
from .property_generator import PropertyGenerator


class OntologyGenerator:
    """
    Ontology generation handler with 6-stage pipeline.

    6-Stage Pipeline:
    1. Semantic Network Parsing → Extract domain concepts
    2. YAML-to-Definition → Transform into class definitions
    3. Definition-to-Types → Map to OWL types
    4. Hierarchy Generation → Build taxonomic structures
    5. TTL Generation → Generate OWL/Turtle syntax
    6. Symbolic Validation → HermiT/Pellet reasoning

    • Generates ontologies from data and text
    • Infers classes and properties automatically
    • Creates domain-specific ontologies
    • Optimizes ontology structure
    • Validates ontology quality
    • Supports various ontology formats
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize ontology generator.

        Sets up the ontology generator with namespace management, naming conventions,
        and inference engines for classes and properties.

        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - base_uri: Base URI for ontology (default: "https://semantica.dev/ontology/")
                - namespace_manager: Optional namespace manager instance
                - min_occurrences: Minimum occurrences for class inference (default: 2)

        Example:
            ```python
            generator = OntologyGenerator(
                base_uri="https://example.org/ontology/",
                min_occurrences=3
            )
            ```
        """
        self.logger = get_logger("ontology_generator")
        self.config = config or {}
        self.config.update(kwargs)

        # Initialize progress tracker
        self.progress_tracker = get_progress_tracker()

        # Initialize components
        self.namespace_manager = self.config.get(
            "namespace_manager"
        ) or NamespaceManager(**self.config)
        self.naming_conventions = NamingConventions(**self.config)
        self.class_inferrer = ClassInferrer(
            namespace_manager=self.namespace_manager, **self.config
        )
        self.property_generator = PropertyGenerator(
            namespace_manager=self.namespace_manager, **self.config
        )

        self.supported_formats = ["owl", "ttl", "rdf", "json-ld"]

    def generate_ontology(self, data: Dict[str, Any], **options) -> Dict[str, Any]:
        """
        Generate ontology from data using 6-stage pipeline.

        Executes the complete 6-stage ontology generation pipeline:
        1. Semantic Network Parsing: Extract domain concepts from entities/relationships
        2. YAML-to-Definition: Transform concepts into class definitions
        3. Definition-to-Types: Map definitions to OWL types
        4. Hierarchy Generation: Build taxonomic structures
        5. TTL Generation: (Handled by OWLGenerator)
        6. Symbolic Validation: (Handled by OntologyValidator)

        Args:
            data: Input data dictionary containing:
                - entities: List of entity dictionaries
                - relationships: List of relationship dictionaries
                - semantic_network: Optional pre-parsed semantic network
            **options: Generation options:
                - name: Ontology name (default: "GeneratedOntology")
                - build_hierarchy: Whether to build class hierarchy (default: True)
                - namespace_manager: Optional namespace manager instance

        Returns:
            Generated ontology dictionary containing:
                - uri: Ontology URI
                - name: Ontology name
                - version: Version string
                - classes: List of class definitions
                - properties: List of property definitions
                - metadata: Additional metadata

        Example:
            ```python
            generator = OntologyGenerator(base_uri="https://example.org/ontology/")
            ontology = generator.generate_ontology({
                "entities": [{"type": "Person", "name": "John"}],
                "relationships": [{"type": "worksFor", "source": "John", "target": "Acme"}]
            })
            ```
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="ontology",
            submodule="OntologyGenerator",
            message="Generating ontology using 6-stage pipeline",
        )

        try:
            self.logger.info("Starting ontology generation pipeline")

            # Stage 1: Semantic Network Parsing
            self.progress_tracker.update_tracking(
                tracking_id, message="Stage 1: Parsing semantic network..."
            )
            semantic_network = self._stage1_parse_semantic_network(data, **options)

            # Stage 2: YAML-to-Definition
            self.progress_tracker.update_tracking(
                tracking_id, message="Stage 2: Converting to class definitions..."
            )
            definitions = self._stage2_yaml_to_definition(semantic_network, **options)

            # Stage 3: Definition-to-Types
            self.progress_tracker.update_tracking(
                tracking_id, message="Stage 3: Mapping to OWL types..."
            )
            
            # Ensure entities and relationships are available for property inference
            stage3_options = options.copy()
            stage3_options["entities"] = data.get("entities", [])
            stage3_options["relationships"] = data.get("relationships", [])
            
            typed_definitions = self._stage3_definition_to_types(definitions, **stage3_options)

            # Stage 4: Hierarchy Generation
            self.progress_tracker.update_tracking(
                tracking_id, message="Stage 4: Building class hierarchy..."
            )
            ontology = self._stage4_hierarchy_generation(typed_definitions, **options)

            # Stage 5: TTL Generation (handled by OWLGenerator)
            # Stage 6: Symbolic Validation (handled by OntologyValidator)

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Generated ontology with {len(ontology.get('classes', []))} classes, {len(ontology.get('properties', []))} properties",
            )
            return ontology

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def _stage1_parse_semantic_network(
        self, data: Dict[str, Any], **options
    ) -> Dict[str, Any]:
        """
        Stage 1: Parse semantic network from data.

        Extract domain concepts from entities and relationships.
        """
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        # Extract concepts
        concepts = {}
        for entity in entities:
            entity_type = entity.get("type") or entity.get("entity_type", "Entity")
            if entity_type not in concepts:
                concepts[entity_type] = {"instances": [], "relationships": []}
            concepts[entity_type]["instances"].append(entity)

        # Extract relationships
        for rel in relationships:
            rel_type = rel.get("type") or rel.get("relationship_type", "relatedTo")
            source_type = rel.get("source_type")
            target_type = rel.get("target_type")

            if source_type and source_type in concepts:
                concepts[source_type]["relationships"].append(rel)

        return {
            "concepts": concepts,
            "entities": entities,
            "relationships": relationships,
        }

    def _stage2_yaml_to_definition(
        self, semantic_network: Dict[str, Any], **options
    ) -> Dict[str, Any]:
        """
        Stage 2: Transform semantic network to class definitions.

        Convert concepts to structured class definitions.
        """
        concepts = semantic_network.get("concepts", {})
        entities = semantic_network.get("entities", [])

        # Infer classes from entities
        classes = self.class_inferrer.infer_classes(entities, **options)

        # Create definitions
        definitions = {
            "classes": classes,
            "properties": [],
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "concept_count": len(concepts),
            },
        }

        return definitions

    def _stage3_definition_to_types(
        self, definitions: Dict[str, Any], **options
    ) -> Dict[str, Any]:
        """
        Stage 3: Map definitions to OWL types.

        Convert class definitions to OWL-compliant types.
        """
        classes = definitions.get("classes", [])
        relationships = options.get("relationships", [])
        entities = options.get("entities", [])

        # Clean options for infer_properties to avoid multiple values for arguments
        prop_options = options.copy()
        prop_options.pop("entities", None)
        prop_options.pop("relationships", None)

        # Infer properties
        properties = self.property_generator.infer_properties(
            entities=entities, relationships=relationships, classes=classes, **prop_options
        )

        # Add types to classes
        for cls in classes:
            cls["@type"] = "owl:Class"
            if "uri" not in cls:
                cls["uri"] = self.namespace_manager.generate_class_iri(cls["name"])

        # Add types to properties
        for prop in properties:
            if prop["type"] == "object":
                prop["@type"] = "owl:ObjectProperty"
            else:
                prop["@type"] = "owl:DatatypeProperty"

            if "uri" not in prop:
                prop["uri"] = self.namespace_manager.generate_property_iri(prop["name"])

        return {
            "classes": classes,
            "properties": properties,
            "metadata": definitions.get("metadata", {}),
        }

    def _stage4_hierarchy_generation(
        self, typed_definitions: Dict[str, Any], **options
    ) -> Dict[str, Any]:
        """
        Stage 4: Build taxonomic structures.

        Generate class hierarchies and property relationships.
        """
        classes = typed_definitions.get("classes", [])
        properties = typed_definitions.get("properties", [])

        # Build class hierarchy
        classes = self.class_inferrer.build_class_hierarchy(classes, **options)

        # Build ontology structure
        ontology = {
            "uri": self.namespace_manager.get_base_uri(),
            "name": options.get("name", "GeneratedOntology"),
            "version": self.namespace_manager.version,
            "classes": classes,
            "properties": properties,
            "metadata": {
                **typed_definitions.get("metadata", {}),
                "class_count": len(classes),
                "property_count": len(properties),
            },
        }

        return ontology

    def infer_classes(self, data: Dict[str, Any], **options) -> List[Dict[str, Any]]:
        """
        Infer ontology classes from data.

        Args:
            data: Input data
            **options: Additional options

        Returns:
            List of inferred classes
        """
        entities = data.get("entities", [])
        return self.class_inferrer.infer_classes(entities, **options)

    def infer_properties(
        self, data: Dict[str, Any], classes: List[Dict[str, Any]], **options
    ) -> List[Dict[str, Any]]:
        """
        Infer ontology properties from data.

        Args:
            data: Input data
            classes: List of class definitions
            **options: Additional options

        Returns:
            List of inferred properties
        """
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        return self.property_generator.infer_properties(
            entities=entities, relationships=relationships, classes=classes, **options
        )

    def optimize_ontology(self, ontology: Dict[str, Any], **options) -> Dict[str, Any]:
        """
        Optimize ontology structure and quality.

        Args:
            ontology: Ontology dictionary
            **options: Additional options

        Returns:
            Optimized ontology
        """
        optimizer = OntologyOptimizer(**self.config)
        return optimizer.optimize_ontology(ontology, **options)


class ClassInferencer:
    """
    Class inference engine (legacy compatibility).

    • Infers ontology classes from data
    • Handles class hierarchies
    • Manages class relationships
    • Processes class constraints
    """

    def __init__(self, **config):
        """Initialize class inferencer."""
        self.class_inferrer = ClassInferrer(**config)

    def infer_classes(self, data, **options):
        """Infer classes from data."""
        entities = data.get("entities", []) if isinstance(data, dict) else data
        return self.class_inferrer.infer_classes(entities, **options)

    def build_class_hierarchy(self, classes, **options):
        """Build class hierarchy."""
        return self.class_inferrer.build_class_hierarchy(classes, **options)

    def validate_classes(self, classes, **criteria):
        """Validate classes."""
        return self.class_inferrer.validate_classes(classes, **criteria)


class PropertyInferencer:
    """
    Property inference engine (legacy compatibility).

    • Infers ontology properties from data
    • Handles property domains and ranges
    • Manages property relationships
    • Processes property constraints
    """

    def __init__(self, **config):
        """Initialize property inferencer."""
        self.property_generator = PropertyGenerator(**config)

    def infer_properties(self, data, classes, **options):
        """Infer properties from data and classes."""
        entities = data.get("entities", []) if isinstance(data, dict) else []
        relationships = data.get("relationships", []) if isinstance(data, dict) else []

        return self.property_generator.infer_properties(
            entities=entities, relationships=relationships, classes=classes, **options
        )

    def infer_domains_and_ranges(self, properties, classes):
        """Infer domains and ranges."""
        return self.property_generator.infer_domains_and_ranges(properties, classes)

    def validate_properties(self, properties, **criteria):
        """Validate properties."""
        return self.property_generator.validate_properties(properties, **criteria)


class OntologyOptimizer:
    """
    Ontology optimization engine.

    • Optimizes ontology structure
    • Removes redundant elements
    • Improves ontology coherence
    • Manages optimization metrics
    """

    def __init__(self, **config):
        """
        Initialize ontology optimizer.

        Args:
            **config: Configuration options
        """
        self.logger = get_logger("ontology_optimizer")
        self.config = config

    def optimize_ontology(self, ontology: Dict[str, Any], **options) -> Dict[str, Any]:
        """
        Optimize ontology structure.

        Args:
            ontology: Ontology dictionary
            **options: Additional options

        Returns:
            Optimized ontology
        """
        optimized = dict(ontology)

        # Remove redundancy
        if options.get("remove_redundancy", True):
            optimized = self.remove_redundancy(optimized)

        # Improve coherence
        if options.get("improve_coherence", True):
            optimized = self.improve_coherence(optimized)

        return optimized

    def remove_redundancy(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove redundant elements from ontology.

        Args:
            ontology: Ontology dictionary

        Returns:
            Cleaned ontology
        """
        # Remove duplicate classes
        classes = ontology.get("classes", [])
        seen_names = set()
        unique_classes = []

        for cls in classes:
            class_name = cls.get("name")
            if class_name and class_name not in seen_names:
                seen_names.add(class_name)
                unique_classes.append(cls)

        ontology["classes"] = unique_classes

        # Remove duplicate properties
        properties = ontology.get("properties", [])
        seen_prop_names = set()
        unique_properties = []

        for prop in properties:
            prop_name = prop.get("name")
            if prop_name and prop_name not in seen_prop_names:
                seen_prop_names.add(prop_name)
                unique_properties.append(prop)

        ontology["properties"] = unique_properties

        return ontology

    def improve_coherence(self, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Improve ontology coherence.

        Args:
            ontology: Ontology dictionary

        Returns:
            Improved ontology
        """
        # Ensure all classes have required fields
        classes = ontology.get("classes", [])
        for cls in classes:
            if "uri" not in cls:
                cls["uri"] = cls.get("name", "Entity")
            if "label" not in cls:
                cls["label"] = cls.get("name", "Entity")

        # Ensure all properties have domains and ranges
        properties = ontology.get("properties", [])
        for prop in properties:
            if prop.get("type") == "object":
                if "domain" not in prop or not prop["domain"]:
                    prop["domain"] = ["owl:Thing"]
                if "range" not in prop or not prop["range"]:
                    prop["range"] = ["owl:Thing"]

        return ontology
