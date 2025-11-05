"""
Ontology Reuse Manager

Manages ontology reuse including research, evaluation, and integration
of existing ontologies (internal and external) and non-ontological resources.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


@dataclass
class ReuseDecision:
    """Ontology reuse decision record."""
    source_uri: str
    decision: str  # 'reuse', 'partial', 'reject'
    reason: str
    imported_elements: List[str] = field(default_factory=list)
    compatibility_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReuseManager:
    """
    Ontology reuse management system.
    
    • Research existing ontologies
    • Evaluate ontology alignment and compatibility
    • Assess interoperability benefits
    • Manage external ontology imports
    • Handle internal ontology reuse
    • Convert non-ontological resources to ontologies
    • Track reuse decisions and dependencies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize reuse manager.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("reuse_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.reuse_decisions: List[ReuseDecision] = []
        self.known_ontologies: Dict[str, Dict[str, Any]] = {}
        
        self._load_known_ontologies()
    
    def _load_known_ontologies(self) -> None:
        """Load known ontology catalog."""
        # Common ontology URIs
        self.known_ontologies = {
            "foaf": {
                "uri": "http://xmlns.com/foaf/0.1/",
                "name": "FOAF",
                "description": "Friend of a Friend vocabulary"
            },
            "dublin-core": {
                "uri": "http://purl.org/dc/elements/1.1/",
                "name": "Dublin Core",
                "description": "Dublin Core metadata vocabulary"
            },
            "schema.org": {
                "uri": "https://schema.org/",
                "name": "Schema.org",
                "description": "Schema.org vocabulary"
            }
        }
    
    def research_ontology(self, uri: str) -> Optional[Dict[str, Any]]:
        """
        Research existing ontology.
        
        Args:
            uri: Ontology URI
        
        Returns:
            Ontology information or None
        """
        # Check known ontologies
        for key, info in self.known_ontologies.items():
            if info["uri"] == uri:
                return info
        
        # Try to load from URI (placeholder)
        self.logger.warning(f"Ontology not found in catalog: {uri}")
        return None
    
    def evaluate_alignment(
        self,
        source_uri: str,
        target_ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate ontology alignment and compatibility.
        
        Args:
            source_uri: Source ontology URI
            target_ontology: Target ontology dictionary
        
        Returns:
            Alignment evaluation results
        """
        # Basic alignment check
        source_info = self.research_ontology(source_uri)
        
        if not source_info:
            return {
                "compatible": False,
                "score": 0.0,
                "issues": ["Source ontology not found"]
            }
        
        # Check namespace compatibility
        target_namespace = target_ontology.get("uri", "")
        namespace_compatible = not target_namespace.startswith(source_uri)
        
        # Calculate compatibility score
        score = 0.5 if namespace_compatible else 0.0
        
        return {
            "compatible": score > 0.3,
            "score": score,
            "namespace_compatible": namespace_compatible,
            "issues": [] if namespace_compatible else ["Namespace conflict"]
        }
    
    def assess_interoperability(
        self,
        source_uri: str,
        target_ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess interoperability benefits.
        
        Args:
            source_uri: Source ontology URI
            target_ontology: Target ontology dictionary
        
        Returns:
            Interoperability assessment
        """
        alignment = self.evaluate_alignment(source_uri, target_ontology)
        
        benefits = []
        if alignment["compatible"]:
            benefits.append("Can reuse existing classes and properties")
            benefits.append("Improved interoperability with other systems")
        
        return {
            "benefits": benefits,
            "compatibility_score": alignment["score"],
            "recommendation": "reuse" if alignment["compatible"] else "reject"
        }
    
    def import_external_ontology(
        self,
        source_uri: str,
        target_ontology: Dict[str, Any],
        **options
    ) -> Dict[str, Any]:
        """
        Import external ontology elements.
        
        Args:
            source_uri: Source ontology URI
            target_ontology: Target ontology dictionary
            **options: Additional options
        
        Returns:
            Updated ontology with imports
        """
        # Add import to ontology
        if "imports" not in target_ontology:
            target_ontology["imports"] = []
        
        if source_uri not in target_ontology["imports"]:
            target_ontology["imports"].append(source_uri)
        
        # Record reuse decision
        decision = ReuseDecision(
            source_uri=source_uri,
            decision="reuse",
            reason="External ontology import",
            metadata={"imported_at": datetime.now().isoformat()}
        )
        self.reuse_decisions.append(decision)
        
        self.logger.info(f"Imported external ontology: {source_uri}")
        
        return target_ontology
    
    def convert_non_ontological_resource(
        self,
        resource: Dict[str, Any],
        **options
    ) -> Dict[str, Any]:
        """
        Convert non-ontological resource to ontology.
        
        Args:
            resource: Resource dictionary
            **options: Additional options
        
        Returns:
            Converted ontology dictionary
        """
        # Basic conversion
        ontology = {
            "name": resource.get("name", "ConvertedOntology"),
            "uri": resource.get("uri", ""),
            "version": "1.0",
            "classes": resource.get("classes", []),
            "properties": resource.get("properties", []),
            "metadata": {
                "converted_from": resource.get("type", "unknown"),
                "converted_at": datetime.now().isoformat()
            }
        }
        
        return ontology
    
    def track_reuse_decision(
        self,
        source_uri: str,
        decision: str,
        reason: str,
        **metadata
    ) -> ReuseDecision:
        """
        Track reuse decision.
        
        Args:
            source_uri: Source ontology URI
            decision: Decision ('reuse', 'partial', 'reject')
            reason: Reason for decision
            **metadata: Additional metadata
        
        Returns:
            Reuse decision record
        """
        reuse_decision = ReuseDecision(
            source_uri=source_uri,
            decision=decision,
            reason=reason,
            metadata={
                "decided_at": datetime.now().isoformat(),
                **metadata
            }
        )
        
        self.reuse_decisions.append(reuse_decision)
        self.logger.info(f"Tracked reuse decision: {decision} for {source_uri}")
        
        return reuse_decision
    
    def get_reuse_history(self) -> List[ReuseDecision]:
        """Get reuse decision history."""
        return list(self.reuse_decisions)
    
    def list_known_ontologies(self) -> List[str]:
        """List known ontology URIs."""
        return list(self.known_ontologies.keys())
