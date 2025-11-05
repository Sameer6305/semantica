"""
Ontology Requirements Specification Module

This module supports the ontology requirements specification phase,
including competency questions, scope definition, and purpose documentation.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from .competency_questions import CompetencyQuestionsManager, CompetencyQuestion


@dataclass
class RequirementsSpec:
    """Requirements specification for ontology."""
    name: str
    purpose: str
    scope: str
    competency_questions: List[CompetencyQuestion] = field(default_factory=list)
    stakeholders: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequirementsSpecManager:
    """
    Requirements specification manager for ontologies.
    
    • Competency question management
    • Scope definition and validation
    • Purpose and use case documentation
    • Stakeholder collaboration tracking
    • Domain expert input integration
    • Requirements traceability
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize requirements spec manager.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("requirements_spec_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.competency_questions_manager = CompetencyQuestionsManager(**self.config)
        self.specs: Dict[str, RequirementsSpec] = {}
    
    def create_spec(
        self,
        name: str,
        purpose: str,
        scope: str,
        **options
    ) -> RequirementsSpec:
        """
        Create requirements specification.
        
        Args:
            name: Specification name
            purpose: Purpose description
            scope: Scope description
            **options: Additional options:
                - domain: Domain name
                - stakeholders: List of stakeholders
                - use_cases: List of use cases
        
        Returns:
            Created requirements specification
        """
        spec = RequirementsSpec(
            name=name,
            purpose=purpose,
            scope=scope,
            domain=options.get("domain", ""),
            stakeholders=options.get("stakeholders", []),
            use_cases=options.get("use_cases", []),
            metadata={
                "created_at": datetime.now().isoformat(),
                **options.get("metadata", {})
            }
        )
        
        self.specs[name] = spec
        self.logger.info(f"Created requirements specification: {name}")
        
        return spec
    
    def add_competency_question(
        self,
        spec_name: str,
        question: str,
        category: str = "general",
        priority: int = 1,
        **metadata
    ) -> CompetencyQuestion:
        """
        Add competency question to specification.
        
        Args:
            spec_name: Specification name
            question: Question text
            category: Question category
            priority: Priority
            **metadata: Additional metadata
        
        Returns:
            Created competency question
        """
        if spec_name not in self.specs:
            raise ValidationError(f"Specification not found: {spec_name}")
        
        cq = self.competency_questions_manager.add_question(
            question=question,
            category=category,
            priority=priority,
            **metadata
        )
        
        self.specs[spec_name].competency_questions.append(cq)
        
        return cq
    
    def validate_spec(self, spec_name: str) -> Dict[str, Any]:
        """
        Validate requirements specification.
        
        Args:
            spec_name: Specification name
        
        Returns:
            Validation results
        """
        if spec_name not in self.specs:
            raise ValidationError(f"Specification not found: {spec_name}")
        
        spec = self.specs[spec_name]
        errors = []
        warnings = []
        
        # Check required fields
        if not spec.purpose:
            errors.append("Specification missing purpose")
        if not spec.scope:
            errors.append("Specification missing scope")
        if not spec.competency_questions:
            warnings.append("Specification has no competency questions")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def trace_requirements(
        self,
        spec_name: str,
        ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Trace requirements to ontology elements.
        
        Args:
            spec_name: Specification name
            ontology: Ontology dictionary
        
        Returns:
            Traceability mapping
        """
        if spec_name not in self.specs:
            raise ValidationError(f"Specification not found: {spec_name}")
        
        spec = self.specs[spec_name]
        
        # Validate ontology against competency questions
        validation = self.competency_questions_manager.validate_ontology(ontology)
        
        # Trace each question
        traces = {}
        for question in spec.competency_questions:
            elements = self.competency_questions_manager.trace_to_elements(question, ontology)
            traces[question.question] = {
                "elements": elements,
                "answerable": question.answerable
            }
        
        return {
            "specification": spec_name,
            "validation": validation,
            "traces": traces,
            "coverage": validation["answerable"] / validation["total_questions"] if validation["total_questions"] > 0 else 0.0
        }
    
    def get_spec(self, spec_name: str) -> Optional[RequirementsSpec]:
        """Get requirements specification by name."""
        return self.specs.get(spec_name)
    
    def list_specs(self) -> List[str]:
        """List all specification names."""
        return list(self.specs.keys())
