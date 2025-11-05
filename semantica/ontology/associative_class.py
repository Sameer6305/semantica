"""
Associative Class Builder

Supports creating associative classes (like Position) that connect
multiple classes together when simple relations are insufficient.
Useful for modeling complex relationships like Person-Role-Organization.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


@dataclass
class AssociativeClass:
    """Associative class definition."""
    name: str
    connects: List[str]  # List of class names this connects
    properties: Dict[str, Any] = field(default_factory=dict)
    temporal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AssociativeClassBuilder:
    """
    Associative class builder for complex relationships.
    
    • Create associative classes connecting multiple entities
    • Model complex multi-entity relationships
    • Support temporal associations (time-based connections)
    • Enable position/role modeling
    • Handle association cardinality
    • Support association property management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize associative class builder.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("associative_class_builder")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.associative_classes: Dict[str, AssociativeClass] = {}
    
    def create_associative_class(
        self,
        name: str,
        connects: List[str],
        **options
    ) -> AssociativeClass:
        """
        Create an associative class.
        
        Args:
            name: Class name
            connects: List of class names this connects
            **options: Additional options:
                - temporal: Whether this is a temporal association
                - properties: Properties for the association
        
        Returns:
            Created associative class
        """
        if not name:
            raise ValidationError("Associative class name is required")
        
        if not connects or len(connects) < 2:
            raise ValidationError("Associative class must connect at least 2 classes")
        
        assoc_class = AssociativeClass(
            name=name,
            connects=connects,
            properties=options.get("properties", {}),
            temporal=options.get("temporal", False),
            metadata=options.get("metadata", {})
        )
        
        self.associative_classes[name] = assoc_class
        
        self.logger.info(f"Created associative class: {name} connecting {connects}")
        
        return assoc_class
    
    def create_position_class(
        self,
        person_class: str,
        organization_class: str,
        role_class: Optional[str] = None,
        **options
    ) -> AssociativeClass:
        """
        Create a position/role associative class.
        
        Args:
            person_class: Person class name
            organization_class: Organization class name
            role_class: Optional role class name
            **options: Additional options
        
        Returns:
            Created associative class
        """
        connects = [person_class, organization_class]
        if role_class:
            connects.append(role_class)
        
        return self.create_associative_class(
            name=options.get("name", "Position"),
            connects=connects,
            temporal=options.get("temporal", True),
            properties={
                "startDate": "xsd:date",
                "endDate": "xsd:date",
                **options.get("properties", {})
            },
            **options
        )
    
    def create_temporal_association(
        self,
        name: str,
        connects: List[str],
        **options
    ) -> AssociativeClass:
        """
        Create a temporal associative class.
        
        Args:
            name: Class name
            connects: List of class names
            **options: Additional options
        
        Returns:
            Created associative class
        """
        return self.create_associative_class(
            name=name,
            connects=connects,
            temporal=True,
            properties={
                "startDate": "xsd:dateTime",
                "endDate": "xsd:dateTime",
                **options.get("properties", {})
            },
            **options
        )
    
    def get_associative_class(self, name: str) -> Optional[AssociativeClass]:
        """
        Get associative class by name.
        
        Args:
            name: Class name
        
        Returns:
            Associative class or None
        """
        return self.associative_classes.get(name)
    
    def list_associative_classes(self) -> List[AssociativeClass]:
        """
        List all associative classes.
        
        Returns:
            List of associative classes
        """
        return list(self.associative_classes.values())
    
    def validate_associative_class(self, assoc_class: AssociativeClass) -> Dict[str, Any]:
        """
        Validate associative class.
        
        Args:
            assoc_class: Associative class to validate
        
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check name
        if not assoc_class.name:
            errors.append("Associative class must have a name")
        
        # Check connects
        if len(assoc_class.connects) < 2:
            errors.append("Associative class must connect at least 2 classes")
        
        # Check for duplicate connections
        if len(assoc_class.connects) != len(set(assoc_class.connects)):
            warnings.append("Associative class has duplicate connections")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
