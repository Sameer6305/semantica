"""
Ontology Management Module

This module provides comprehensive ontology management and generation capabilities,
following semantic modeling best practices and guidelines for building effective ontologies.

Exports:
    - OntologyGenerator: Ontology generation from data (6-stage pipeline)
    - ClassInferrer: Class discovery and hierarchy building
    - PropertyGenerator: Property inference and data types
    - OntologyValidator: Validation with symbolic reasoners
    - OWLGenerator: OWL/RDF generation
    - OntologyEvaluator: Ontology quality evaluation
    - RequirementsSpecManager: Requirements specification and competency questions
    - CompetencyQuestionsManager: Competency question management
    - ReuseManager: Ontology reuse management
    - VersionManager: Ontology versioning
    - NamespaceManager: Namespace management
    - NamingConventions: Naming convention enforcement
    - ModuleManager: Ontology module management
    - DomainOntologies: Pre-built domain ontologies
    - OntologyDocumentationManager: Documentation management
    - AssociativeClassBuilder: Associative class creation
"""

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
]
