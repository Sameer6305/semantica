"""
Source Tracker for Conflict Detection

Tracks source information for entities and properties to enable
conflict investigation and source disagreement analysis.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from ..utils.exceptions import ValidationError
from ..utils.logging import get_logger


@dataclass
class SourceReference:
    """Source reference for a property value."""
    document: str
    page: Optional[int] = None
    section: Optional[str] = None
    line: Optional[int] = None
    timestamp: Optional[datetime] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PropertySource:
    """Source information for a property value."""
    property_name: str
    value: Any
    sources: List[SourceReference] = field(default_factory=list)
    entity_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SourceTracker:
    """
    Source tracker for conflict detection.
    
    • Tracks source documents for each property value
    • Maintains source provenance for entities
    • Tracks source disagreements and conflicts
    • Generates source analysis reports
    • Supports source credibility scoring
    • Enables traceability for conflict resolution
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize source tracker.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("source_tracker")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Entity -> Property -> PropertySource
        self.entity_sources: Dict[str, Dict[str, PropertySource]] = defaultdict(dict)
        
        # Relationship sources
        self.relationship_sources: Dict[str, List[SourceReference]] = defaultdict(list)
        
        # Source credibility scores
        self.source_credibility: Dict[str, float] = {}
        
    def track_entity_source(
        self,
        entity_id: str,
        source: SourceReference,
        **metadata
    ) -> bool:
        """
        Track source for an entity.
        
        Args:
            entity_id: Entity identifier
            source: Source reference
            **metadata: Additional metadata
            
        Returns:
            True if tracking successful
        """
        if entity_id not in self.entity_sources:
            self.entity_sources[entity_id] = {}
        
        # Store entity-level source
        if "_entity_sources" not in self.entity_sources[entity_id]:
            self.entity_sources[entity_id]["_entity_sources"] = PropertySource(
                property_name="_entity_sources",
                value=entity_id,
                entity_id=entity_id
            )
        
        self.entity_sources[entity_id]["_entity_sources"].sources.append(source)
        self.logger.debug(f"Tracked entity source: {entity_id} from {source.document}")
        return True
    
    def track_property_source(
        self,
        entity_id: str,
        property_name: str,
        value: Any,
        source: SourceReference,
        **metadata
    ) -> bool:
        """
        Track source for a property value.
        
        Args:
            entity_id: Entity identifier
            property_name: Property name
            value: Property value
            source: Source reference
            **metadata: Additional metadata
            
        Returns:
            True if tracking successful
        """
        if entity_id not in self.entity_sources:
            self.entity_sources[entity_id] = {}
        
        if property_name not in self.entity_sources[entity_id]:
            self.entity_sources[entity_id][property_name] = PropertySource(
                property_name=property_name,
                value=value,
                entity_id=entity_id,
                metadata=metadata
            )
        
        property_source = self.entity_sources[entity_id][property_name]
        
        # Add source if not already present
        source_exists = any(
            s.document == source.document and
            s.page == source.page and
            s.section == source.section
            for s in property_source.sources
        )
        
        if not source_exists:
            property_source.sources.append(source)
        
        # Update value if different (will be used for conflict detection)
        if property_source.value != value:
            property_source.value = value  # Store latest value
        
        self.logger.debug(
            f"Tracked property source: {entity_id}.{property_name} = {value} "
            f"from {source.document}"
        )
        return True
    
    def track_relationship_source(
        self,
        relationship_id: str,
        source: SourceReference,
        **metadata
    ) -> bool:
        """
        Track source for a relationship.
        
        Args:
            relationship_id: Relationship identifier
            source: Source reference
            **metadata: Additional metadata
            
        Returns:
            True if tracking successful
        """
        source_exists = any(
            s.document == source.document and
            s.page == source.page
            for s in self.relationship_sources[relationship_id]
        )
        
        if not source_exists:
            self.relationship_sources[relationship_id].append(source)
        
        self.logger.debug(
            f"Tracked relationship source: {relationship_id} from {source.document}"
        )
        return True
    
    def get_entity_sources(self, entity_id: str) -> List[SourceReference]:
        """
        Get all sources for an entity.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            List of source references
        """
        sources = []
        
        if entity_id in self.entity_sources:
            # Get entity-level sources
            if "_entity_sources" in self.entity_sources[entity_id]:
                sources.extend(
                    self.entity_sources[entity_id]["_entity_sources"].sources
                )
            
            # Get all property sources
            for prop_source in self.entity_sources[entity_id].values():
                sources.extend(prop_source.sources)
        
        # Deduplicate
        seen = set()
        unique_sources = []
        for source in sources:
            key = (source.document, source.page, source.section)
            if key not in seen:
                seen.add(key)
                unique_sources.append(source)
        
        return unique_sources
    
    def get_property_sources(
        self,
        entity_id: str,
        property_name: str
    ) -> Optional[PropertySource]:
        """
        Get sources for a specific property.
        
        Args:
            entity_id: Entity identifier
            property_name: Property name
            
        Returns:
            PropertySource or None if not found
        """
        if entity_id in self.entity_sources:
            return self.entity_sources[entity_id].get(property_name)
        return None
    
    def get_relationship_sources(self, relationship_id: str) -> List[SourceReference]:
        """
        Get sources for a relationship.
        
        Args:
            relationship_id: Relationship identifier
            
        Returns:
            List of source references
        """
        return self.relationship_sources.get(relationship_id, [])
    
    def find_source_disagreements(
        self,
        entity_id: str,
        property_name: str
    ) -> List[Dict[str, Any]]:
        """
        Find source disagreements for a property.
        
        Args:
            entity_id: Entity identifier
            property_name: Property name
            
        Returns:
            List of disagreement records
        """
        property_source = self.get_property_sources(entity_id, property_name)
        if not property_source or len(property_source.sources) < 2:
            return []
        
        # Group sources by value (if we have multiple values)
        # For now, we'll check if sources have different confidence or metadata
        disagreements = []
        
        sources = property_source.sources
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                # Check for disagreements in confidence, metadata, or document
                if (source1.document != source2.document or
                    source1.confidence != source2.confidence):
                    disagreements.append({
                        "entity_id": entity_id,
                        "property_name": property_name,
                        "source1": {
                            "document": source1.document,
                            "page": source1.page,
                            "confidence": source1.confidence
                        },
                        "source2": {
                            "document": source2.document,
                            "page": source2.page,
                            "confidence": source2.confidence
                        }
                    })
        
        return disagreements
    
    def set_source_credibility(
        self,
        document: str,
        credibility: float
    ) -> bool:
        """
        Set credibility score for a source document.
        
        Args:
            document: Document identifier
            credibility: Credibility score (0.0 to 1.0)
            
        Returns:
            True if set successfully
        """
        if not 0.0 <= credibility <= 1.0:
            raise ValidationError("Credibility must be between 0.0 and 1.0")
        
        self.source_credibility[document] = credibility
        self.logger.info(f"Set credibility for {document}: {credibility}")
        return True
    
    def get_source_credibility(self, document: str) -> float:
        """
        Get credibility score for a source document.
        
        Args:
            document: Document identifier
            
        Returns:
            Credibility score (default: 0.5)
        """
        return self.source_credibility.get(document, 0.5)
    
    def generate_source_report(
        self,
        entity_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate source analysis report.
        
        Args:
            entity_id: Optional entity ID to filter by
            
        Returns:
            Source report dictionary
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "entities_tracked": 0,
            "properties_tracked": 0,
            "relationships_tracked": len(self.relationship_sources),
            "total_sources": 0,
            "source_credibility": dict(self.source_credibility),
            "entities": []
        }
        
        entities_to_report = [entity_id] if entity_id else list(self.entity_sources.keys())
        
        for eid in entities_to_report:
            if eid not in self.entity_sources:
                continue
            
            entity_info = {
                "entity_id": eid,
                "properties": [],
                "total_sources": 0
            }
            
            for prop_name, prop_source in self.entity_sources[eid].items():
                if prop_name == "_entity_sources":
                    continue
                
                prop_info = {
                    "property_name": prop_name,
                    "value": prop_source.value,
                    "sources": [
                        {
                            "document": s.document,
                            "page": s.page,
                            "section": s.section,
                            "confidence": s.confidence,
                            "credibility": self.get_source_credibility(s.document)
                        }
                        for s in prop_source.sources
                    ],
                    "source_count": len(prop_source.sources)
                }
                
                entity_info["properties"].append(prop_info)
                entity_info["total_sources"] += len(prop_source.sources)
                report["total_sources"] += len(prop_source.sources)
            
            if entity_info["properties"]:
                report["entities"].append(entity_info)
                report["entities_tracked"] += 1
                report["properties_tracked"] += len(entity_info["properties"])
        
        return report
    
    def get_traceability_chain(
        self,
        entity_id: str,
        property_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get traceability chain for an entity or property.
        
        Args:
            entity_id: Entity identifier
            property_name: Optional property name
            
        Returns:
            List of traceability records
        """
        chain = []
        
        if property_name:
            prop_source = self.get_property_sources(entity_id, property_name)
            if prop_source:
                for source in prop_source.sources:
                    chain.append({
                        "type": "property",
                        "entity_id": entity_id,
                        "property_name": property_name,
                        "value": prop_source.value,
                        "source": {
                            "document": source.document,
                            "page": source.page,
                            "section": source.section,
                            "confidence": source.confidence,
                            "timestamp": source.timestamp.isoformat() if source.timestamp else None
                        }
                    })
        else:
            # Get all entity sources
            sources = self.get_entity_sources(entity_id)
            for source in sources:
                chain.append({
                    "type": "entity",
                    "entity_id": entity_id,
                    "source": {
                        "document": source.document,
                        "page": source.page,
                        "section": source.section,
                        "confidence": source.confidence,
                        "timestamp": source.timestamp.isoformat() if source.timestamp else None
                    }
                })
        
        return chain
