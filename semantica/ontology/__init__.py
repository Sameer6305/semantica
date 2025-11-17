"""
Ontology Management Module

This module provides comprehensive ontology management and generation capabilities,
following semantic modeling best practices and guidelines for building effective ontologies.

Algorithms Used:

Ontology Generation (6-Stage Pipeline):
    - Stage 1 - Semantic Network Parsing: Extract domain concepts from entities/relationships, entity type analysis (Counter), relationship pattern extraction, concept grouping
    - Stage 2 - YAML-to-Definition: Transform concepts into class definitions, YAML parsing, definition structure creation, class inference (ClassInferrer.infer_classes)
    - Stage 3 - Definition-to-Types: Map definitions to OWL types, type inference, OWL class/property mapping (@type assignment: owl:Class, owl:ObjectProperty, owl:DatatypeProperty), property inference (PropertyGenerator.infer_properties)
    - Stage 4 - Hierarchy Generation: Build taxonomic structures, parent-child relationship inference, hierarchy validation, circular dependency detection (DFS), transitive closure calculation
    - Stage 5 - TTL Generation: Generate OWL/Turtle syntax using rdflib, namespace prefix handling, RDF serialization (rdflib.serialize format="turtle"), triple generation (subject-predicate-object)
    - Stage 6 - Symbolic Validation: HermiT/Pellet reasoning (owlready2.sync_reasoner), consistency checking, satisfiability checking, constraint validation

Class Inference:
    - Pattern-Based Inference: Entity type frequency analysis (Counter), minimum occurrence threshold filtering, similarity-based class merging (threshold matching), entity grouping by type
    - Hierarchy Building: Parent-child relationship inference, transitive closure calculation, hierarchy depth analysis, circular dependency detection (DFS), parent class assignment
    - Class Validation: Naming convention enforcement (PascalCase), IRI generation (namespace_manager.generate_class_iri), namespace validation, class name normalization

Property Inference:
    - Object Property Inference: Relationship type analysis, domain/range inference from entity types, property cardinality detection, relationship-to-property mapping
    - Data Property Inference: Entity attribute analysis, XSD type detection (string, integer, float, boolean, date), property domain inference, attribute frequency analysis
    - Property Validation: Domain/range validation, property hierarchy management, naming convention enforcement (camelCase), property IRI generation

Ontology Validation:
    - Symbolic Reasoning: HermiT reasoner integration (owlready2.sync_reasoner), Pellet reasoner integration, consistency checking, satisfiability checking, reasoner selection (auto/hermit/pellet)
    - Constraint Validation: Domain/range constraint checking, cardinality constraint validation, logical constraint validation, structure validation
    - Hallucination Detection: LLM-generated ontology validation, fact verification, relationship validation, error/warning collection

OWL/RDF Generation:
    - RDF Graph Construction: rdflib.Graph creation, namespace binding, triple generation (subject-predicate-object), namespace prefix declaration
    - Serialization: Turtle format (rdflib.serialize format="turtle"), RDF/XML format, JSON-LD format, N3 format, format selection and conversion
    - Namespace Management: Prefix declaration, IRI resolution, namespace prefix mapping, standard namespace registration (RDF, RDFS, OWL, XSD, SKOS, DC)

Ontology Evaluation:
    - Competency Question Validation: Question parsing, ontology query generation, answer coverage analysis, question-to-ontology element tracing
    - Coverage Metrics: Class coverage calculation, property coverage calculation, relationship coverage calculation, answerable question ratio
    - Completeness Metrics: Required class detection, missing property identification, gap analysis, completeness score calculation
    - Granularity Evaluation: Class granularity assessment, generalization/specialization analysis, granularity score calculation

Requirements Specification:
    - Competency Question Management: Question storage, categorization (general, organizational, temporal), validation, priority assignment (1=high, 2=medium, 3=low)
    - Scope Definition: Domain boundary definition, entity type scoping, relationship scoping, purpose documentation
    - Traceability: Requirements-to-ontology mapping, coverage tracking, trace-to-elements assignment

Ontology Reuse:
    - Ontology Research: Known ontology catalog lookup (FOAF, Dublin Core, Schema.org), URI resolution, metadata extraction, ontology information retrieval
    - Alignment Evaluation: Concept alignment scoring, compatibility assessment, interoperability analysis, alignment score calculation
    - Import Management: External ontology import, namespace merging, conflict resolution, import decision tracking (reuse/partial/reject)

Version Management:
    - Version-Aware IRI Generation: Version in ontology IRI (not element IRIs), version-less element IRIs, logical version-less IRIs, versioned IRI construction (urljoin)
    - Version Comparison: Diff generation, change detection, migration path identification, version diff analysis
    - Multi-Version Coexistence: Version isolation, import closure resolution, version record management

Namespace Management:
    - IRI Generation: Base URI + local name construction (urljoin), namespace prefix mapping, IRI validation, speaking IRI support (human-readable)
    - Prefix Handling: Prefix declaration, namespace binding, prefix resolution, standard namespace registration

Associative Class Creation:
    - Complex Relationship Modeling: N-ary relationship handling, relationship properties, intermediate class creation, multi-entity connection
    - Pattern Detection: Relationship pattern analysis, associative class inference, temporal association detection

Key Features:
    - Automatic ontology generation from data (6-stage pipeline)
    - Class and property inference from entities and relationships
    - Ontology validation with symbolic reasoners (HermiT, Pellet)
    - OWL/RDF generation and serialization
    - Ontology quality evaluation and assessment
    - Requirements specification and competency questions
    - Ontology reuse and integration management
    - Version management with best practices
    - Namespace and IRI management
    - Naming convention enforcement
    - Modular ontology development
    - Pre-built domain ontologies
    - Comprehensive documentation management
    - Associative class creation for complex relationships
    - Method registry for extensibility
    - Configuration management with environment variables and config files

Main Classes:
    - OntologyGenerator: Main ontology generation class (6-stage pipeline)
    - ClassInferrer: Class discovery and hierarchy building
    - PropertyGenerator: Property inference and data types
    - OntologyValidator: Validation with symbolic reasoners
    - OWLGenerator: OWL/RDF generation using rdflib
    - OntologyEvaluator: Ontology quality evaluation
    - RequirementsSpecManager: Requirements specification and competency questions
    - CompetencyQuestionsManager: Competency question management
    - ReuseManager: Ontology reuse management
    - VersionManager: Ontology versioning
    - NamespaceManager: Namespace and IRI management
    - NamingConventions: Naming convention enforcement
    - ModuleManager: Ontology module management
    - DomainOntologies: Pre-built domain ontologies
    - OntologyDocumentationManager: Documentation management
    - AssociativeClassBuilder: Associative class creation
    - MethodRegistry: Registry for custom ontology methods
    - OntologyConfig: Configuration manager for ontology module

Convenience Functions:
    - generate_ontology: Ontology generation wrapper (6-stage pipeline)
    - infer_classes: Class inference wrapper
    - infer_properties: Property inference wrapper
    - validate_ontology: Ontology validation wrapper
    - generate_owl: OWL/RDF generation wrapper
    - evaluate_ontology: Ontology evaluation wrapper
    - create_requirements_spec: Requirements specification wrapper
    - add_competency_question: Competency question management wrapper
    - research_ontology: Ontology research wrapper
    - import_external_ontology: External ontology import wrapper
    - create_version: Version creation wrapper
    - manage_namespace: Namespace management wrapper
    - create_associative_class: Associative class creation wrapper
    - get_ontology_method: Get ontology method by name
    - list_available_methods: List registered methods

Example Usage:
    >>> from semantica.ontology import generate_ontology, infer_classes, validate_ontology, OntologyGenerator
    >>> # Using convenience functions
    >>> ontology = generate_ontology({"entities": [...], "relationships": [...]}, method="default")
    >>> classes = infer_classes(entities, method="default")
    >>> result = validate_ontology(ontology, method="default")
    >>> # Using classes directly
    >>> from semantica.ontology import OntologyGenerator, ClassInferrer, PropertyGenerator
    >>> generator = OntologyGenerator(base_uri="https://example.org/ontology/")
    >>> ontology = generator.generate_ontology({"entities": [...], "relationships": [...]})
    >>> inferrer = ClassInferrer()
    >>> classes = inferrer.infer_classes(entities)
    >>> prop_gen = PropertyGenerator()
    >>> properties = prop_gen.infer_properties(entities, relationships, classes)

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union

from .ontology_generator import OntologyGenerator, ClassInferencer, PropertyInferencer, OntologyOptimizer
from .class_inferrer import ClassInferrer
from .property_generator import PropertyGenerator
from .ontology_validator import OntologyValidator, ValidationResult
from .owl_generator import OWLGenerator
from .ontology_evaluator import OntologyEvaluator, EvaluationResult
from .requirements_spec import RequirementsSpecManager, RequirementsSpec
from .competency_questions import CompetencyQuestionsManager, CompetencyQuestion
from .reuse_manager import ReuseManager, ReuseDecision
from .version_manager import VersionManager, OntologyVersion
from .namespace_manager import NamespaceManager
from .naming_conventions import NamingConventions
from .module_manager import ModuleManager, OntologyModule
from .domain_ontologies import DomainOntologies
from .ontology_documentation import OntologyDocumentationManager, OntologyDocumentation
from .associative_class import AssociativeClassBuilder, AssociativeClass
from .registry import MethodRegistry, method_registry
from .methods import (
    generate_ontology,
    infer_classes,
    infer_properties,
    validate_ontology,
    generate_owl,
    evaluate_ontology,
    create_requirements_spec,
    add_competency_question,
    research_ontology,
    import_external_ontology,
    create_version,
    manage_namespace,
    create_associative_class,
    get_ontology_method,
    list_available_methods,
)
from .config import OntologyConfig, ontology_config

__all__ = [
    # Main generators
    "OntologyGenerator",
    "ClassInferrer",
    "ClassInferencer",  # Legacy alias
    "PropertyGenerator",
    "PropertyInferencer",  # Legacy alias
    "OntologyOptimizer",
    
    # Validation and evaluation
    "OntologyValidator",
    "ValidationResult",
    "OntologyEvaluator",
    "EvaluationResult",
    
    # OWL/RDF generation
    "OWLGenerator",
    
    # Requirements and competency questions
    "RequirementsSpecManager",
    "RequirementsSpec",
    "CompetencyQuestionsManager",
    "CompetencyQuestion",
    
    # Management
    "ReuseManager",
    "ReuseDecision",
    "VersionManager",
    "OntologyVersion",
    "NamespaceManager",
    "NamingConventions",
    "ModuleManager",
    "OntologyModule",
    "DomainOntologies",
    "OntologyDocumentationManager",
    "OntologyDocumentation",
    
    # Special classes
    "AssociativeClassBuilder",
    "AssociativeClass",
    
    # Registry and Methods
    "MethodRegistry",
    "method_registry",
    "generate_ontology",
    "infer_classes",
    "infer_properties",
    "validate_ontology",
    "generate_owl",
    "evaluate_ontology",
    "create_requirements_spec",
    "add_competency_question",
    "research_ontology",
    "import_external_ontology",
    "create_version",
    "manage_namespace",
    "create_associative_class",
    "get_ontology_method",
    "list_available_methods",
    
    # Configuration
    "OntologyConfig",
    "ontology_config",
]
