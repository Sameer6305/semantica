"""
Namespace and IRI Manager

Manages ontology namespaces and generates consistent, human-readable
IRIs following best practices (speaking IRIs, version management).
"""

from typing import Any, Dict, List, Optional
import re
from urllib.parse import urljoin

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


class NamespaceManager:
    """
    Namespace and IRI management for ontologies.
    
    • Define stable host namespaces
    • Generate consistent IRI conventions
    • PascalCase for classes
    • camelCase for properties and relations
    • Version-aware IRI management
    • Resolvable IRI generation
    • Speaking IRI support (human-readable)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize namespace manager.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - base_uri: Base URI for the ontology
                - version: Ontology version
                - use_speaking_iris: Use human-readable IRIs (default: True)
        """
        self.logger = get_logger("namespace_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.base_uri = self.config.get("base_uri", "https://semantica.dev/ontology/")
        self.version = self.config.get("version", "1.0")
        self.use_speaking_iris = self.config.get("use_speaking_iris", True)
        
        # Standard namespaces
        self.namespaces = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dc": "http://purl.org/dc/elements/1.1/",
            "dcterms": "http://purl.org/dc/terms/",
        }
    
    def get_base_uri(self) -> str:
        """
        Get base URI for ontology.
        
        Returns:
            Base URI string
        """
        if self.version and self.version != "1.0":
            return urljoin(self.base_uri.rstrip('/') + '/', f"v{self.version}/")
        return self.base_uri
    
    def generate_class_iri(self, class_name: str, **options) -> str:
        """
        Generate IRI for class (PascalCase).
        
        Args:
            class_name: Class name
            **options: Additional options
        
        Returns:
            Class IRI
        """
        # Convert to PascalCase
        class_name = self._to_pascal_case(class_name)
        
        # Generate speaking IRI if enabled
        if self.use_speaking_iris:
            iri = urljoin(self.get_base_uri(), class_name)
        else:
            # Use hash-based IRI
            import hashlib
            hash_id = hashlib.md5(class_name.encode()).hexdigest()[:8]
            iri = urljoin(self.get_base_uri(), f"class/{hash_id}")
        
        return iri
    
    def generate_property_iri(self, property_name: str, **options) -> str:
        """
        Generate IRI for property (camelCase).
        
        Args:
            property_name: Property name
            **options: Additional options
        
        Returns:
            Property IRI
        """
        # Convert to camelCase
        property_name = self._to_camel_case(property_name)
        
        # Generate speaking IRI if enabled
        if self.use_speaking_iris:
            iri = urljoin(self.get_base_uri(), property_name)
        else:
            # Use hash-based IRI
            import hashlib
            hash_id = hashlib.md5(property_name.encode()).hexdigest()[:8]
            iri = urljoin(self.get_base_uri(), f"property/{hash_id}")
        
        return iri
    
    def generate_individual_iri(self, individual_name: str, **options) -> str:
        """
        Generate IRI for individual instance.
        
        Args:
            individual_name: Individual name
            **options: Additional options
        
        Returns:
            Individual IRI
        """
        # Clean and normalize name
        individual_name = re.sub(r'[^a-zA-Z0-9]', '', individual_name)
        
        if self.use_speaking_iris:
            iri = urljoin(self.get_base_uri(), f"individual/{individual_name}")
        else:
            import hashlib
            hash_id = hashlib.md5(individual_name.encode()).hexdigest()[:8]
            iri = urljoin(self.get_base_uri(), f"individual/{hash_id}")
        
        return iri
    
    def register_namespace(self, prefix: str, uri: str) -> None:
        """
        Register a namespace.
        
        Args:
            prefix: Namespace prefix
            uri: Namespace URI
        """
        self.namespaces[prefix] = uri
        self.logger.debug(f"Registered namespace: {prefix} -> {uri}")
    
    def get_namespace(self, prefix: str) -> Optional[str]:
        """
        Get namespace URI by prefix.
        
        Args:
            prefix: Namespace prefix
        
        Returns:
            Namespace URI or None
        """
        return self.namespaces.get(prefix)
    
    def get_all_namespaces(self) -> Dict[str, str]:
        """
        Get all registered namespaces.
        
        Returns:
            Dictionary of prefix -> URI mappings
        """
        return dict(self.namespaces)
    
    def _to_pascal_case(self, name: str) -> str:
        """Convert name to PascalCase."""
        # Remove special characters and split
        words = re.findall(r'[a-zA-Z0-9]+', name)
        if not words:
            return "Entity"
        
        # Capitalize first letter of each word
        return ''.join(word.capitalize() for word in words)
    
    def _to_camel_case(self, name: str) -> str:
        """Convert name to camelCase."""
        # Remove special characters and split
        words = re.findall(r'[a-zA-Z0-9]+', name)
        if not words:
            return "hasProperty"
        
        # First word lowercase, rest capitalized
        return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    
    def validate_iri(self, iri: str) -> bool:
        """
        Validate IRI format.
        
        Args:
            iri: IRI to validate
        
        Returns:
            True if valid
        """
        # Basic IRI validation
        try:
            from urllib.parse import urlparse
            parsed = urlparse(iri)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False
