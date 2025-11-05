"""
Text cleaning utilities for Semantica framework.

This module provides text cleaning and preprocessing functions
for HTML removal, whitespace normalization, and text sanitization.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Union

from bs4 import BeautifulSoup

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger


class TextCleaner:
    """Text cleaning utilities."""
    
    def __init__(self, **config):
        """
        Initialize text cleaner.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("text_cleaner")
        self.config = config
    
    def clean(
        self,
        text: str,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True,
        remove_special_chars: bool = False,
        **options
    ) -> str:
        """
        Clean text with various options.
        
        Args:
            text: Input text to clean
            remove_html: Whether to remove HTML tags
            normalize_whitespace: Whether to normalize whitespace
            normalize_unicode: Whether to normalize unicode
            remove_special_chars: Whether to remove special characters
            **options: Additional cleaning options
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove HTML tags
        if remove_html:
            cleaned = self.remove_html(cleaned)
        
        # Normalize unicode
        if normalize_unicode:
            cleaned = self.normalize_unicode(cleaned, form=options.get("unicode_form", "NFC"))
        
        # Normalize whitespace
        if normalize_whitespace:
            cleaned = self.normalize_whitespace(cleaned)
        
        # Remove special characters
        if remove_special_chars:
            cleaned = self.remove_special_chars(cleaned, allow_spaces=options.get("allow_spaces", True))
        
        return cleaned.strip()
    
    def remove_html(self, text: str, preserve_structure: bool = False) -> str:
        """
        Remove HTML tags from text.
        
        Args:
            text: Input text with HTML
            preserve_structure: Whether to preserve structure (use BeautifulSoup)
            
        Returns:
            str: Text without HTML tags
        """
        if not text:
            return ""
        
        if preserve_structure:
            try:
                soup = BeautifulSoup(text, 'html.parser')
                return soup.get_text(separator='\n', strip=True)
            except Exception:
                # Fallback to regex if BeautifulSoup fails
                pass
        
        # Remove HTML tags using regex
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        # Remove remaining HTML entities
        text = re.sub(r'&#?\w+;', '', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            str: Text with normalized whitespace
        """
        if not text:
            return ""
        
        # Replace tabs and newlines with spaces
        text = re.sub(r'[\t\n\r]+', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def normalize_unicode(self, text: str, form: str = "NFC") -> str:
        """
        Normalize unicode characters.
        
        Args:
            text: Input text
            form: Normalization form (NFC, NFD, NFKC, NFKD)
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        try:
            return unicodedata.normalize(form, text)
        except Exception as e:
            self.logger.warning(f"Failed to normalize unicode: {e}")
            return text
    
    def remove_special_chars(self, text: str, allow_spaces: bool = True, allow_newlines: bool = False) -> str:
        """
        Remove special characters from text.
        
        Args:
            text: Input text
            allow_spaces: Whether to allow spaces
            allow_newlines: Whether to allow newlines
            
        Returns:
            str: Text without special characters
        """
        if not text:
            return ""
        
        if allow_spaces and allow_newlines:
            # Keep alphanumeric, spaces, and newlines
            pattern = r'[^a-zA-Z0-9\s]'
        elif allow_spaces:
            # Keep alphanumeric and spaces
            pattern = r'[^a-zA-Z0-9 ]'
        else:
            # Keep only alphanumeric
            pattern = r'[^a-zA-Z0-9]'
        
        return re.sub(pattern, '', text)
    
    def sanitize(self, text: str, **options) -> str:
        """
        Sanitize text for security.
        
        Args:
            text: Input text
            **options: Sanitization options
            
        Returns:
            str: Sanitized text
        """
        if not text:
            return ""
        
        # Remove potential script tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'<iframe[^>]*>.*?</iframe>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: URLs
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        
        # Remove data: URLs if requested
        if options.get("remove_data_urls", False):
            text = re.sub(r'data:[^;]*;base64,', '', text, flags=re.IGNORECASE)
        
        return text
    
    def trim(self, text: str, **options) -> str:
        """
        Trim text.
        
        Args:
            text: Input text
            **options: Trimming options
            
        Returns:
            str: Trimmed text
        """
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove empty lines if requested
        if options.get("remove_empty_lines", False):
            lines = [line for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
        
        return text
    
    def clean_batch(self, texts: List[str], **options) -> List[str]:
        """
        Clean multiple texts in batch.
        
        Args:
            texts: List of texts to clean
            **options: Cleaning options
            
        Returns:
            list: List of cleaned texts
        """
        return [self.clean(text, **options) for text in texts]
