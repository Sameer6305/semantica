"""
Context manager for Semantica framework.

This module provides context window management for
embedding generation and processing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger


@dataclass
class ContextWindow:
    """Context window representation."""
    
    text: str
    start_index: int
    end_index: int
    context_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """Context window manager for embeddings."""
    
    def __init__(self, **config):
        """
        Initialize context manager.
        
        Args:
            **config: Configuration options:
                - window_size: Context window size in tokens/characters
                - overlap: Overlap between windows
                - max_contexts: Maximum number of contexts to keep
        """
        self.logger = get_logger("context_manager")
        self.config = config
        
        self.window_size = config.get("window_size", 512)
        self.overlap = config.get("overlap", 50)
        self.max_contexts = config.get("max_contexts", 100)
        
        # Store contexts
        self.contexts: Dict[str, ContextWindow] = {}
    
    def create_context_window(self, text: str, start_index: int = 0, **metadata) -> ContextWindow:
        """
        Create a context window from text.
        
        Args:
            text: Input text
            start_index: Start index in original text
            **metadata: Context metadata
            
        Returns:
            ContextWindow: Created context window
        """
        import uuid
        
        context_id = str(uuid.uuid4())
        end_index = start_index + len(text)
        
        window = ContextWindow(
            text=text,
            start_index=start_index,
            end_index=end_index,
            context_id=context_id,
            metadata=metadata
        )
        
        # Store context
        if len(self.contexts) >= self.max_contexts:
            # Remove oldest context
            oldest_id = min(self.contexts.keys(), key=lambda k: self.contexts[k].start_index)
            del self.contexts[oldest_id]
        
        self.contexts[context_id] = window
        
        return window
    
    def split_into_windows(self, text: str, **options) -> List[ContextWindow]:
        """
        Split text into context windows.
        
        Args:
            text: Input text
            **options: Splitting options:
                - preserve_sentences: Preserve sentence boundaries
                
        Returns:
            list: List of context windows
        """
        windows = []
        text_length = len(text)
        start = 0
        
        preserve_sentences = options.get("preserve_sentences", True)
        
        while start < text_length:
            end = min(start + self.window_size, text_length)
            
            if preserve_sentences and end < text_length:
                # Try to find sentence boundary
                for boundary in ['.', '!', '?', '\n']:
                    last_boundary = text.rfind(boundary, start, end)
                    if last_boundary > start + self.window_size * 0.5:  # At least 50% of window
                        end = last_boundary + 1
                        break
            
            window_text = text[start:end]
            window = self.create_context_window(window_text, start, **options)
            windows.append(window)
            
            # Move to next window with overlap
            start += (self.window_size - self.overlap)
        
        return windows
    
    def get_context(self, context_id: str) -> Optional[ContextWindow]:
        """
        Get context by ID.
        
        Args:
            context_id: Context ID
            
        Returns:
            ContextWindow: Context window or None
        """
        return self.contexts.get(context_id)
    
    def merge_contexts(self, context_ids: List[str], **options) -> Optional[ContextWindow]:
        """
        Merge multiple contexts.
        
        Args:
            context_ids: List of context IDs
            **options: Merge options
            
        Returns:
            ContextWindow: Merged context window
        """
        contexts = [self.contexts.get(cid) for cid in context_ids if cid in self.contexts]
        contexts = [c for c in contexts if c is not None]
        
        if not contexts:
            return None
        
        # Sort by start index
        contexts.sort(key=lambda c: c.start_index)
        
        # Merge texts
        merged_text = ' '.join(c.text for c in contexts)
        start_index = contexts[0].start_index
        end_index = contexts[-1].end_index
        
        # Combine metadata
        merged_metadata = {}
        for ctx in contexts:
            merged_metadata.update(ctx.metadata)
        
        return self.create_context_window(
            merged_text,
            start_index,
            merged=True,
            source_contexts=context_ids,
            **merged_metadata
        )
    
    def clear_contexts(self):
        """Clear all stored contexts."""
        self.contexts.clear()
    
    def get_context_count(self) -> int:
        """Get number of stored contexts."""
        return len(self.contexts)
