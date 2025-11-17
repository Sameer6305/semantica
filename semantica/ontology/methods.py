"""
Ontology Methods Module

This module provides all ontology operations as simple, reusable functions for
ontology generation, class/property inference, validation, evaluation, OWL generation,
requirements specification, reuse management, versioning, namespace management, and
associative class creation. It supports multiple approaches and integrates with the
method registry for extensibility.

Supported Methods:

Ontology Generation:
    - "default": Default ontology generation using 6-stage pipeline
    - "from_data": Generate from entity/relationship data
    - "from_text": Generate from text (future support)

Class/Property Inference:
    - "default": Default inference using ClassInferrer/PropertyGenerator
    - "pattern": Pattern-based inference
    - "hierarchical": Hierarchy-focused inference

Validation:
    - "default": Default validation using OntologyValidator
    - "hermit": HermiT reasoner validation
    - "pellet": Pellet reasoner validation
    - "basic": Basic structure validation only

Evaluation:
    - "default": Default evaluation using OntologyEvaluator
    - "coverage": Coverage-focused evaluation
    - "completeness": Completeness-focused evaluation

OWL Generation:
    - "default": Default OWL generation using OWLGenerator
    - "turtle": Turtle format generation
    - "rdfxml": RDF/XML format generation
    - "jsonld": JSON-LD format generation

Algorithms Used:

Ontology Generation (6-Stage Pipeline):
    - Stage 1 - Semantic Network Parsing: Extract domain concepts from entities/relationships, entity type analysis (Counter), relationship pattern extraction
    - Stage 2 - YAML-to-Definition: Transform concepts into class definitions, YAML parsing, definition structure creation
    - Stage 3 - Definition-to-Types: Map definitions to OWL types, type inference, OWL class/property mapping (@type assignment)
    - Stage 4 - Hierarchy Generation: Build taxonomic structures, parent-child relationship inference, hierarchy validation, circular dependency detection (DFS)
    - Stage 5 - TTL Generation: Generate OWL/Turtle syntax using rdflib, namespace prefix handling, RDF serialization (rdflib.serialize)
    - Stage 6 - Symbolic Validation: HermiT/Pellet reasoning (owlready2.sync_reasoner), consistency checking, satisfiability checking, constraint validation

Class Inference:
    - Pattern-Based Inference: Entity type frequency analysis (Counter), minimum occurrence threshold filtering, similarity-based class merging (threshold matching)
    - Hierarchy Building: Parent-child relationship inference, transitive closure calculation, hierarchy depth analysis, circular dependency detection (DFS)
    - Class Validation: Naming convention enforcement (PascalCase), IRI generation (namespace_manager.generate_class_iri), namespace validation

Property Inference:
    - Object Property Inference: Relationship type analysis, domain/range inference from entity types, property cardinality detection
    - Data Property Inference: Entity attribute analysis, XSD type detection (string, integer, float, boolean, date), property domain inference
    - Property Validation: Domain/range validation, property hierarchy management, naming convention enforcement (camelCase)

Ontology Validation:
    - Symbolic Reasoning: HermiT reasoner integration (owlready2.sync_reasoner), Pellet reasoner integration, consistency checking, satisfiability checking
    - Constraint Validation: Domain/range constraint checking, cardinality constraint validation, logical constraint validation
    - Hallucination Detection: LLM-generated ontology validation, fact verification, relationship validation

OWL/RDF Generation:
    - RDF Graph Construction: rdflib.Graph creation, namespace binding, triple generation (subject-predicate-object)
    - Serialization: Turtle format (rdflib.serialize format="turtle"), RDF/XML format, JSON-LD format, N3 format
    - Namespace Management: Prefix declaration, IRI resolution, namespace prefix mapping

Ontology Evaluation:
    - Competency Question Validation: Question parsing, ontology query generation, answer coverage analysis
    - Coverage Metrics: Class coverage calculation, property coverage calculation, relationship coverage calculation
    - Completeness Metrics: Required class detection, missing property identification, gap analysis
    - Granularity Evaluation: Class granularity assessment, generalization/specialization analysis

Requirements Specification:
    - Competency Question Management: Question storage, categorization, validation
    - Scope Definition: Domain boundary definition, entity type scoping, relationship scoping
    - Traceability: Requirements-to-ontology mapping, coverage tracking

Ontology Reuse:
    - Ontology Research: Known ontology catalog lookup (FOAF, Dublin Core, Schema.org), URI resolution, metadata extraction
    - Alignment Evaluation: Concept alignment scoring, compatibility assessment, interoperability analysis
    - Import Management: External ontology import, namespace merging, conflict resolution

Version Management:
    - Version-Aware IRI Generation: Version in ontology IRI (not element IRIs), version-less element IRIs, logical version-less IRIs
    - Version Comparison: Diff generation, change detection, migration path identification
    - Multi-Version Coexistence: Version isolation, import closure resolution

Namespace Management:
    - IRI Generation: Base URI + local name construction (urljoin), namespace prefix mapping, IRI validation
    - Prefix Handling: Prefix declaration, namespace binding, prefix resolution

Associative Class Creation:
    - Complex Relationship Modeling: N-ary relationship handling, relationship properties, intermediate class creation
    - Pattern Detection: Relationship pattern analysis, associative class inference

Key Features:
    - Multiple ontology operation methods
    - Ontology operations with method dispatch
    - Method dispatchers with registry support
    - Custom method registration capability
    - Consistent interface across all methods

Main Functions:
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
    >>> from semantica.ontology.methods import generate_ontology, infer_classes, validate_ontology
    >>> ontology = generate_ontology({"entities": [...], "relationships": [...]}, method="default")
    >>> classes = infer_classes(entities, method="default")
    >>> result = validate_ontology(ontology, method="default")
"""

from typing import Any, Dict, List, Optional, Callable, Union

from ..utils.logging import get_logger
from ..utils.exceptions import ProcessingError, ConfigurationError
from .ontology_generator import OntologyGenerator
from .class_inferrer import ClassInferrer
from .property_generator import PropertyGenerator
from .ontology_validator import OntologyValidator
from .owl_generator import OWLGenerator
from .ontology_evaluator import OntologyEvaluator
from .requirements_spec import RequirementsSpecManager
from .competency_questions import CompetencyQuestionsManager
from .reuse_manager import ReuseManager
from .version_manager import VersionManager
from .namespace_manager import NamespaceManager
from .associative_class import AssociativeClassBuilder
from .registry import method_registry
from .config import ontology_config

logger = get_logger("ontology_methods")


def generate_ontology(
    data: Dict[str, Any],
    method: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """
    Generate ontology from data using 6-stage pipeline (convenience function).
    
    This is a user-friendly wrapper that generates ontologies using the specified method.
    
    Args:
        data: Input data dictionary containing:
            - entities: List of entity dictionaries
            - relationships: List of relationship dictionaries
            - semantic_network: Optional pre-parsed semantic network
        method: Generation method (default: "default")
            - "default": Use OntologyGenerator with 6-stage pipeline
            - "from_data": Generate from entity/relationship data
        **kwargs: Additional options passed to OntologyGenerator
            - name: Ontology name (default: "GeneratedOntology")
            - base_uri: Base URI for ontology
            - build_hierarchy: Whether to build class hierarchy (default: True)
        
    Returns:
        dict: Generated ontology dictionary containing:
            - uri: Ontology URI
            - name: Ontology name
            - version: Version string
            - classes: List of class definitions
            - properties: List of property definitions
            - metadata: Additional metadata
        
    Examples:
        >>> from semantica.ontology.methods import generate_ontology
        >>> ontology = generate_ontology({
        ...     "entities": [{"type": "Person", "name": "John"}],
        ...     "relationships": [{"type": "worksFor", "source": "John", "target": "Acme"}]
        ... }, method="default")
    """
    custom_method = method_registry.get("generate", method)
    if custom_method:
        try:
            return custom_method(data, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("generate")
        config.update(kwargs)
        
        generator = OntologyGenerator(**config)
        return generator.generate_ontology(data, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to generate ontology: {e}")
        raise


def infer_classes(
    entities: List[Dict[str, Any]],
    method: str = "default",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Infer classes from entities (convenience function).
    
    This is a user-friendly wrapper that infers ontology classes using the specified method.
    
    Args:
        entities: List of entity dictionaries
        method: Inference method (default: "default")
            - "default": Use ClassInferrer with default settings
            - "pattern": Pattern-based inference
            - "hierarchical": Hierarchy-focused inference
        **kwargs: Additional options passed to ClassInferrer
            - build_hierarchy: Whether to build class hierarchy (default: True)
            - min_occurrences: Minimum occurrences for class inference (default: 2)
            - similarity_threshold: Similarity threshold for class merging (default: 0.8)
        
    Returns:
        list: List of inferred class definition dictionaries
        
    Examples:
        >>> from semantica.ontology.methods import infer_classes
        >>> classes = infer_classes(entities, method="default", build_hierarchy=True)
    """
    custom_method = method_registry.get("infer", method)
    if custom_method:
        try:
            return custom_method(entities, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("infer")
        config.update(kwargs)
        
        inferrer = ClassInferrer(**config)
        return inferrer.infer_classes(entities, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to infer classes: {e}")
        raise


def infer_properties(
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    classes: List[Dict[str, Any]],
    method: str = "default",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Infer properties from entities and relationships (convenience function).
    
    This is a user-friendly wrapper that infers ontology properties using the specified method.
    
    Args:
        entities: List of entity dictionaries
        relationships: List of relationship dictionaries
        classes: List of class definitions
        method: Inference method (default: "default")
        **kwargs: Additional options passed to PropertyGenerator
        
    Returns:
        list: List of inferred property definitions
        
    Examples:
        >>> from semantica.ontology.methods import infer_properties
        >>> properties = infer_properties(entities, relationships, classes, method="default")
    """
    custom_method = method_registry.get("infer", method)
    if custom_method:
        try:
            return custom_method(entities, relationships, classes, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("infer")
        config.update(kwargs)
        
        generator = PropertyGenerator(**config)
        return generator.infer_properties(entities, relationships, classes, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to infer properties: {e}")
        raise


def validate_ontology(
    ontology: Dict[str, Any],
    method: str = "default",
    **kwargs
) -> Any:
    """
    Validate ontology structure and consistency (convenience function).
    
    This is a user-friendly wrapper that validates ontologies using the specified method.
    
    Args:
        ontology: Ontology dictionary
        method: Validation method (default: "default")
            - "default": Use OntologyValidator with default settings
            - "hermit": HermiT reasoner validation
            - "pellet": Pellet reasoner validation
            - "basic": Basic structure validation only
        **kwargs: Additional options passed to OntologyValidator
            - reasoner: Reasoner to use ('hermit', 'pellet', 'auto')
            - check_consistency: Check consistency (default: True)
            - check_satisfiability: Check satisfiability (default: True)
        
    Returns:
        ValidationResult: Validation result containing:
            - valid: True if valid, False otherwise
            - consistent: True if consistent, False otherwise
            - errors: List of error messages
            - warnings: List of warning messages
            - metrics: Additional validation metrics
        
    Examples:
        >>> from semantica.ontology.methods import validate_ontology
        >>> result = validate_ontology(ontology, method="default")
        >>> if result.valid: print("Ontology is valid")
    """
    custom_method = method_registry.get("validate", method)
    if custom_method:
        try:
            return custom_method(ontology, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("validate")
        config.update(kwargs)
        
        validator = OntologyValidator(**config)
        return validator.validate_ontology(ontology, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to validate ontology: {e}")
        raise


def generate_owl(
    ontology: Dict[str, Any],
    format: Optional[str] = None,
    method: str = "default",
    **kwargs
) -> Union[str, Any]:
    """
    Generate OWL/RDF from ontology dictionary (convenience function).
    
    This is a user-friendly wrapper that generates OWL using the specified method.
    
    Args:
        ontology: Ontology dictionary
        format: Output format ('turtle', 'rdfxml', 'json-ld', 'n3', default: 'turtle')
        method: Generation method (default: "default")
            - "default": Use OWLGenerator with default settings
            - "turtle": Turtle format generation
            - "rdfxml": RDF/XML format generation
            - "jsonld": JSON-LD format generation
        **kwargs: Additional options passed to OWLGenerator
        
    Returns:
        str or Graph: OWL serialization string or rdflib Graph object
        
    Examples:
        >>> from semantica.ontology.methods import generate_owl
        >>> turtle = generate_owl(ontology, format="turtle", method="default")
    """
    custom_method = method_registry.get("owl", method)
    if custom_method:
        try:
            return custom_method(ontology, format, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("owl")
        config.update(kwargs)
        
        generator = OWLGenerator(**config)
        return generator.generate_owl(ontology, format=format, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to generate OWL: {e}")
        raise


def evaluate_ontology(
    ontology: Dict[str, Any],
    competency_questions: Optional[List[str]] = None,
    method: str = "default",
    **kwargs
) -> Any:
    """
    Evaluate ontology against competency questions (convenience function).
    
    This is a user-friendly wrapper that evaluates ontologies using the specified method.
    
    Args:
        ontology: Ontology dictionary
        competency_questions: List of competency questions (optional)
        method: Evaluation method (default: "default")
            - "default": Use OntologyEvaluator with default settings
            - "coverage": Coverage-focused evaluation
            - "completeness": Completeness-focused evaluation
        **kwargs: Additional options passed to OntologyEvaluator
        
    Returns:
        EvaluationResult: Evaluation result containing:
            - coverage_score: Coverage score (0.0 to 1.0)
            - completeness_score: Completeness score (0.0 to 1.0)
            - gaps: List of identified gaps
            - suggestions: List of improvement suggestions
            - metrics: Additional evaluation metrics
        
    Examples:
        >>> from semantica.ontology.methods import evaluate_ontology
        >>> result = evaluate_ontology(ontology, competency_questions=["Who are the employees?"], method="default")
    """
    custom_method = method_registry.get("evaluate", method)
    if custom_method:
        try:
            return custom_method(ontology, competency_questions, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("evaluate")
        config.update(kwargs)
        
        evaluator = OntologyEvaluator(**config)
        return evaluator.evaluate_ontology(ontology, competency_questions=competency_questions, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to evaluate ontology: {e}")
        raise


def create_requirements_spec(
    name: str,
    purpose: str,
    scope: str,
    method: str = "default",
    **kwargs
) -> Any:
    """
    Create requirements specification (convenience function).
    
    This is a user-friendly wrapper that creates requirements specifications using the specified method.
    
    Args:
        name: Specification name
        purpose: Purpose description
        scope: Scope description
        method: Method (default: "default")
        **kwargs: Additional options passed to RequirementsSpecManager
            - domain: Domain name
            - stakeholders: List of stakeholders
            - use_cases: List of use cases
        
    Returns:
        RequirementsSpec: Created requirements specification
        
    Examples:
        >>> from semantica.ontology.methods import create_requirements_spec
        >>> spec = create_requirements_spec("PersonOntology", "Model person concepts", "Person entities", method="default")
    """
    custom_method = method_registry.get("requirements", method)
    if custom_method:
        try:
            return custom_method(name, purpose, scope, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("requirements")
        config.update(kwargs)
        
        manager = RequirementsSpecManager(**config)
        return manager.create_spec(name, purpose, scope, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to create requirements spec: {e}")
        raise


def add_competency_question(
    spec_name: str,
    question: str,
    category: str = "general",
    method: str = "default",
    **kwargs
) -> Any:
    """
    Add competency question to requirements specification (convenience function).
    
    This is a user-friendly wrapper that adds competency questions using the specified method.
    
    Args:
        spec_name: Specification name
        question: Question text
        category: Question category (default: "general")
        method: Method (default: "default")
        **kwargs: Additional options passed to CompetencyQuestionsManager
        
    Returns:
        CompetencyQuestion: Added competency question
        
    Examples:
        >>> from semantica.ontology.methods import add_competency_question
        >>> cq = add_competency_question("PersonOntology", "Who are the employees?", category="organizational", method="default")
    """
    custom_method = method_registry.get("requirements", method)
    if custom_method:
        try:
            return custom_method(spec_name, question, category, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("requirements")
        config.update(kwargs)
        
        manager = RequirementsSpecManager(**config)
        return manager.add_competency_question(spec_name, question, category=category, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to add competency question: {e}")
        raise


def research_ontology(
    uri: str,
    method: str = "default",
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Research existing ontology (convenience function).
    
    This is a user-friendly wrapper that researches ontologies using the specified method.
    
    Args:
        uri: Ontology URI
        method: Method (default: "default")
        **kwargs: Additional options passed to ReuseManager
        
    Returns:
        Optional[dict]: Ontology information or None if not found
        
    Examples:
        >>> from semantica.ontology.methods import research_ontology
        >>> info = research_ontology("http://xmlns.com/foaf/0.1/", method="default")
    """
    custom_method = method_registry.get("reuse", method)
    if custom_method:
        try:
            return custom_method(uri, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("reuse")
        config.update(kwargs)
        
        manager = ReuseManager(**config)
        return manager.research_ontology(uri, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to research ontology: {e}")
        raise


def import_external_ontology(
    source_uri: str,
    target_ontology: Dict[str, Any],
    method: str = "default",
    **kwargs
) -> Dict[str, Any]:
    """
    Import external ontology (convenience function).
    
    This is a user-friendly wrapper that imports external ontologies using the specified method.
    
    Args:
        source_uri: Source ontology URI
        target_ontology: Target ontology dictionary
        method: Method (default: "default")
        **kwargs: Additional options passed to ReuseManager
        
    Returns:
        dict: Updated target ontology with imported elements
        
    Examples:
        >>> from semantica.ontology.methods import import_external_ontology
        >>> updated = import_external_ontology("http://xmlns.com/foaf/0.1/", ontology, method="default")
    """
    custom_method = method_registry.get("reuse", method)
    if custom_method:
        try:
            return custom_method(source_uri, target_ontology, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("reuse")
        config.update(kwargs)
        
        manager = ReuseManager(**config)
        return manager.import_external_ontology(source_uri, target_ontology, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to import external ontology: {e}")
        raise


def create_version(
    version: str,
    ontology: Dict[str, Any],
    method: str = "default",
    **kwargs
) -> Any:
    """
    Create ontology version (convenience function).
    
    This is a user-friendly wrapper that creates ontology versions using the specified method.
    
    Args:
        version: Version string (e.g., "1.0", "2.1")
        ontology: Ontology dictionary
        method: Method (default: "default")
        **kwargs: Additional options passed to VersionManager
            - changes: List of changes
            - metadata: Additional metadata
        
    Returns:
        OntologyVersion: Created version record
        
    Examples:
        >>> from semantica.ontology.methods import create_version
        >>> version = create_version("1.0", ontology, changes=["Added Person class"], method="default")
    """
    custom_method = method_registry.get("version", method)
    if custom_method:
        try:
            return custom_method(version, ontology, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("version")
        config.update(kwargs)
        
        manager = VersionManager(**config)
        return manager.create_version(version, ontology, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to create version: {e}")
        raise


def manage_namespace(
    operation: str,
    **kwargs
) -> Any:
    """
    Manage namespace operations (convenience function).
    
    This is a user-friendly wrapper that manages namespaces using the specified method.
    
    Args:
        operation: Operation to perform:
            - "generate_class_iri": Generate class IRI
            - "generate_property_iri": Generate property IRI
            - "register_namespace": Register namespace
            - "get_base_uri": Get base URI
        **kwargs: Additional options passed to NamespaceManager
            - class_name: Class name (for generate_class_iri)
            - property_name: Property name (for generate_property_iri)
            - prefix: Namespace prefix (for register_namespace)
            - uri: Namespace URI (for register_namespace)
        
    Returns:
        Result depends on operation:
            - generate_class_iri: Class IRI string
            - generate_property_iri: Property IRI string
            - register_namespace: None
            - get_base_uri: Base URI string
        
    Examples:
        >>> from semantica.ontology.methods import manage_namespace
        >>> class_iri = manage_namespace("generate_class_iri", class_name="Person", base_uri="https://example.org/")
    """
    custom_method = method_registry.get("namespace", "default")
    if custom_method:
        try:
            return custom_method(operation, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("namespace")
        config.update(kwargs)
        
        manager = NamespaceManager(**config)
        
        if operation == "generate_class_iri":
            return manager.generate_class_iri(kwargs.get("class_name", ""), **kwargs)
        elif operation == "generate_property_iri":
            return manager.generate_property_iri(kwargs.get("property_name", ""), **kwargs)
        elif operation == "register_namespace":
            return manager.register_namespace(kwargs.get("prefix", ""), kwargs.get("uri", ""), **kwargs)
        elif operation == "get_base_uri":
            return manager.get_base_uri()
        else:
            raise ValueError(f"Unknown namespace operation: {operation}")
        
    except Exception as e:
        logger.error(f"Failed to manage namespace: {e}")
        raise


def create_associative_class(
    name: str,
    connects: List[str],
    method: str = "default",
    **kwargs
) -> Any:
    """
    Create associative class (convenience function).
    
    This is a user-friendly wrapper that creates associative classes using the specified method.
    
    Args:
        name: Associative class name
        connects: List of class names this associative class connects
        method: Method (default: "default")
        **kwargs: Additional options passed to AssociativeClassBuilder
            - properties: Dictionary of properties
            - temporal: Whether this is a temporal association (default: False)
            - metadata: Additional metadata
        
    Returns:
        AssociativeClass: Created associative class
        
    Examples:
        >>> from semantica.ontology.methods import create_associative_class
        >>> position = create_associative_class("Position", ["Person", "Organization", "Role"], method="default")
    """
    custom_method = method_registry.get("associative", method)
    if custom_method:
        try:
            return custom_method(name, connects, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        config = ontology_config.get_method_config("associative")
        config.update(kwargs)
        
        builder = AssociativeClassBuilder(**config)
        return builder.create_associative_class(name, connects, **kwargs)
        
    except Exception as e:
        logger.error(f"Failed to create associative class: {e}")
        raise


def get_ontology_method(task: str, name: str) -> Optional[Callable]:
    """Get ontology method by task and name."""
    return method_registry.get(task, name)


def list_available_methods(task: Optional[str] = None) -> Dict[str, List[str]]:
    """List all registered ontology methods."""
    return method_registry.list_all(task)


# Register default methods
method_registry.register("generate", "default", generate_ontology)
method_registry.register("infer", "default", infer_classes)
method_registry.register("validate", "default", validate_ontology)
method_registry.register("evaluate", "default", evaluate_ontology)
method_registry.register("owl", "default", generate_owl)
method_registry.register("requirements", "default", create_requirements_spec)
method_registry.register("reuse", "default", research_ontology)
method_registry.register("version", "default", create_version)
method_registry.register("namespace", "default", manage_namespace)
method_registry.register("associative", "default", create_associative_class)

