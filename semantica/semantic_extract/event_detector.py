"""
Event Detection Module

Handles detection and extraction of events from text.

Key Features:
    - Event detection and classification
    - Temporal event processing
    - Event relationship extraction
    - Event confidence scoring
    - Custom event pattern detection

Main Classes:
    - EventDetector: Main event detection class
    - EventClassifier: Event type classification
    - TemporalEventProcessor: Temporal event handling
    - EventRelationshipExtractor: Event relationship extraction
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .ner_extractor import Entity


@dataclass
class Event:
    """Event representation."""
    
    text: str
    event_type: str
    start_char: int
    end_char: int
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    time: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventDetector:
    """Event detection and extraction handler."""
    
    def __init__(self, config=None, **kwargs):
        """Initialize event detector."""
        self.logger = get_logger("event_detector")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.event_classifier = EventClassifier(**self.config.get("classifier", {}))
        self.temporal_processor = TemporalEventProcessor(**self.config.get("temporal", {}))
        self.relationship_extractor = EventRelationshipExtractor(**self.config.get("relationship", {}))
        
        # Event patterns
        self.event_patterns = {
            "founded": r"founded|created|established",
            "acquired": r"acquired|bought|purchased",
            "launched": r"launched|released|introduced",
            "announced": r"announced|declared|stated",
            "meeting": r"met|meeting|conference|summit"
        }
    
    def detect_events(self, text: str, **options) -> List[Event]:
        """
        Detect events in text content.
        
        Args:
            text: Input text
            **options: Detection options
            
        Returns:
            list: List of detected events
        """
        events = []
        
        # Detect events using patterns
        for event_type, pattern in self.event_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Extract surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Extract participants (simplified)
                participants = self._extract_participants(context)
                
                event = Event(
                    text=match.group(0),
                    event_type=event_type,
                    start_char=match.start(),
                    end_char=match.end(),
                    participants=participants,
                    confidence=0.7,
                    metadata={"context": context}
                )
                events.append(event)
        
        return events
    
    def _extract_participants(self, context: str) -> List[str]:
        """Extract event participants from context."""
        # Simple extraction - look for capitalized words
        participants = []
        words = context.split()
        
        # Look for patterns like "X and Y", "X, Y, and Z"
        capitalized = [w for w in words if w[0].isupper() and len(w) > 2]
        participants.extend(capitalized[:3])  # Limit to first few
        
        return participants
    
    def classify_events(self, events: List[Event], **context) -> Dict[str, List[Event]]:
        """
        Classify events by type and category.
        
        Args:
            events: List of events
            **context: Context information
            
        Returns:
            dict: Events grouped by type
        """
        return self.event_classifier.classify_events(events, **context)
    
    def extract_event_properties(self, events: List[Event], **options) -> List[Event]:
        """
        Extract properties and attributes from events.
        
        Args:
            events: List of events
            **options: Extraction options
            
        Returns:
            list: Events with extracted properties
        """
        for event in events:
            # Extract location
            event.location = self._extract_location(event.metadata.get("context", ""))
            
            # Extract time
            event.time = self._extract_time(event.metadata.get("context", ""))
        
        return events
    
    def _extract_location(self, context: str) -> Optional[str]:
        """Extract location from context."""
        # Simple pattern matching for locations
        location_patterns = [
            r"in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"at\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_time(self, context: str) -> Optional[str]:
        """Extract time from context."""
        # Simple pattern matching for dates/times
        time_patterns = [
            r"on\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
            r"in\s+(\d{4})",
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1)
        
        return None
    
    def process_temporal_events(self, events: List[Event], **options) -> List[Event]:
        """
        Process temporal information in events.
        
        Args:
            events: List of events
            **options: Processing options
            
        Returns:
            list: Events with temporal information
        """
        return self.temporal_processor.process_temporal(events, **options)


class EventClassifier:
    """Event type classification."""
    
    def __init__(self, **config):
        """Initialize event classifier."""
        self.logger = get_logger("event_classifier")
        self.config = config
    
    def classify_events(self, events: List[Event], **context) -> Dict[str, List[Event]]:
        """
        Classify events by type.
        
        Args:
            events: List of events
            **context: Context information
            
        Returns:
            dict: Events grouped by type
        """
        classified = {}
        for event in events:
            if event.event_type not in classified:
                classified[event.event_type] = []
            classified[event.event_type].append(event)
        
        return classified


class TemporalEventProcessor:
    """Temporal event handling."""
    
    def __init__(self, **config):
        """Initialize temporal event processor."""
        self.logger = get_logger("temporal_event_processor")
        self.config = config
    
    def process_temporal(self, events: List[Event], **options) -> List[Event]:
        """
        Process temporal information in events.
        
        Args:
            events: List of events
            **options: Processing options
            
        Returns:
            list: Events with processed temporal info
        """
        # Extract and normalize temporal information
        for event in events:
            if not event.time:
                # Try to extract from context
                context = event.metadata.get("context", "")
                event.time = self._extract_temporal(context)
        
        # Sort events by time if available
        if options.get("sort_by_time", False):
            events.sort(key=lambda e: self._parse_time(e.time) if e.time else 0)
        
        return events
    
    def _extract_temporal(self, context: str) -> Optional[str]:
        """Extract temporal expression from context."""
        # Simple temporal extraction
        patterns = [
            r"(\d{4})",
            r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1)
        
        return None
    
    def _parse_time(self, time_str: Optional[str]) -> int:
        """Parse time string to integer for sorting."""
        if not time_str:
            return 0
        
        # Extract year if available
        year_match = re.search(r"(\d{4})", time_str)
        if year_match:
            return int(year_match.group(1))
        
        return 0


class EventRelationshipExtractor:
    """Event relationship extraction."""
    
    def __init__(self, **config):
        """Initialize event relationship extractor."""
        self.logger = get_logger("event_relationship_extractor")
        self.config = config
    
    def extract_relationships(self, events: List[Event], **options) -> List[Dict[str, Any]]:
        """
        Extract relationships between events.
        
        Args:
            events: List of events
            **options: Extraction options
            
        Returns:
            list: List of event relationships
        """
        relationships = []
        
        # Find related events (same participants, same location, sequential time)
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                # Check for shared participants
                shared_participants = set(event1.participants) & set(event2.participants)
                if shared_participants:
                    relationships.append({
                        "event1": event1.event_type,
                        "event2": event2.event_type,
                        "relationship": "related",
                        "reason": "shared_participants",
                        "participants": list(shared_participants)
                    })
        
        return relationships
