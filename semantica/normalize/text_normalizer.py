"""
Text Normalization Module

Handles text cleaning, normalization, and standardization.

Key Features:
    - Text cleaning and sanitization
    - Unicode normalization
    - Case normalization
    - Whitespace handling
    - Special character processing

Main Classes:
    - TextNormalizer: Main text normalization class
    - UnicodeNormalizer: Unicode processing
    - WhitespaceNormalizer: Whitespace handling
    - SpecialCharacterProcessor: Special character handling
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from .text_cleaner import TextCleaner


class TextNormalizer:
    """
    Text normalization and cleaning handler.
    
    • Cleans and normalizes text content
    • Handles various text encodings
    • Processes special characters and symbols
    • Standardizes text formatting
    • Removes unwanted content and noise
    • Supports multiple languages and scripts
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize text normalizer.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("text_normalizer")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.text_cleaner = TextCleaner(**self.config)
        self.unicode_normalizer = UnicodeNormalizer(**self.config)
        self.whitespace_normalizer = WhitespaceNormalizer(**self.config)
        self.special_char_processor = SpecialCharacterProcessor(**self.config)
    
    def normalize_text(self, text: str, **options) -> str:
        """
        Normalize text content.
        
        Args:
            text: Input text
            **options: Normalization options
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        normalized = text
        
        # Unicode normalization
        unicode_form = options.get("unicode_form", "NFC")
        normalized = self.unicode_normalizer.normalize_unicode(normalized, form=unicode_form)
        
        # Whitespace normalization
        normalized = self.whitespace_normalizer.normalize_whitespace(normalized, **options)
        
        # Special character processing
        normalized = self.special_char_processor.process_special_chars(normalized, **options)
        
        # Case normalization
        case_type = options.get("case", "preserve")
        if case_type == "lower":
            normalized = normalized.lower()
        elif case_type == "upper":
            normalized = normalized.upper()
        elif case_type == "title":
            normalized = normalized.title()
        
        return normalized.strip()
    
    def clean_text(self, text: str, **options) -> str:
        """
        Clean and sanitize text content.
        
        Args:
            text: Input text
            **options: Cleaning options
        
        Returns:
            Cleaned text
        """
        return self.text_cleaner.clean(text, **options)
    
    def standardize_format(self, text: str, format_type: str = "standard") -> str:
        """
        Standardize text format.
        
        Args:
            text: Input text
            format_type: Format type ('standard', 'compact', 'preserve')
        
        Returns:
            Formatted text
        """
        if format_type == "compact":
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
        elif format_type == "preserve":
            # Preserve original formatting
            pass
        
        return text.strip()
    
    def process_batch(self, texts: List[str], **options) -> List[str]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts
            **options: Processing options
        
        Returns:
            List of processed texts
        """
        return [self.normalize_text(text, **options) for text in texts]


class UnicodeNormalizer:
    """
    Unicode normalization engine.
    
    • Handles Unicode normalization
    • Processes different Unicode forms
    • Manages character encoding
    • Handles special Unicode characters
    • Supports various scripts and languages
    """
    
    def __init__(self, **config):
        """
        Initialize Unicode normalizer.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("unicode_normalizer")
        self.config = config
    
    def normalize_unicode(self, text: str, form: str = "NFC") -> str:
        """
        Normalize Unicode text.
        
        Args:
            text: Input text
            form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        try:
            return unicodedata.normalize(form, text)
        except Exception as e:
            self.logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def handle_encoding(self, text: str, source_encoding: str, target_encoding: str = "utf-8") -> str:
        """
        Handle text encoding conversion.
        
        Args:
            text: Input text
            source_encoding: Source encoding
            target_encoding: Target encoding
        
        Returns:
            Converted text
        """
        if isinstance(text, bytes):
            try:
                return text.decode(source_encoding).encode(target_encoding).decode(target_encoding)
            except Exception:
                return text.decode('utf-8', errors='replace')
        else:
            return text
    
    def process_special_chars(self, text: str) -> str:
        """
        Process special Unicode characters.
        
        Args:
            text: Input text
        
        Returns:
            Processed text
        """
        # Replace common special characters
        replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201C': '"',  # Left double quotation mark
            '\u201D': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--',  # Em dash
            '\u2026': '...',  # Horizontal ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text


class WhitespaceNormalizer:
    """
    Whitespace normalization engine.
    
    • Normalizes whitespace characters
    • Handles different whitespace types
    • Manages line breaks and spacing
    • Processes indentation and formatting
    """
    
    def __init__(self, **config):
        """
        Initialize whitespace normalizer.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("whitespace_normalizer")
        self.config = config
    
    def normalize_whitespace(self, text: str, **options) -> str:
        """
        Normalize whitespace in text.
        
        Args:
            text: Input text
            **options: Normalization options
        
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        # Normalize line breaks
        line_break_type = options.get("line_break_type", "unix")
        text = self.handle_line_breaks(text, line_break_type)
        
        # Remove excessive whitespace
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize multiple newlines
        
        return text.strip()
    
    def handle_line_breaks(self, text: str, line_break_type: str = "unix") -> str:
        """
        Normalize line breaks.
        
        Args:
            text: Input text
            line_break_type: Line break type ('unix', 'windows', 'mac')
        
        Returns:
            Normalized text
        """
        if line_break_type == "unix":
            text = text.replace('\r\n', '\n')
            text = text.replace('\r', '\n')
        elif line_break_type == "windows":
            text = text.replace('\r\n', '\r\n')
            text = text.replace('\r', '\r\n')
            text = text.replace('\n', '\r\n')
        
        return text
    
    def process_indentation(self, text: str, indent_type: str = "spaces") -> str:
        """
        Normalize text indentation.
        
        Args:
            text: Input text
            indent_type: Indentation type ('spaces', 'tabs')
        
        Returns:
            Normalized text
        """
        if indent_type == "spaces":
            text = text.replace('\t', '    ')  # Convert tabs to 4 spaces
        elif indent_type == "tabs":
            text = re.sub(r'    ', '\t', text)  # Convert 4 spaces to tabs
        
        return text


class SpecialCharacterProcessor:
    """
    Special character processing engine.
    
    • Processes special characters and symbols
    • Handles punctuation and diacritics
    • Manages mathematical symbols
    • Processes currency and unit symbols
    """
    
    def __init__(self, **config):
        """
        Initialize special character processor.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("special_char_processor")
        self.config = config
    
    def process_special_chars(self, text: str, **options) -> str:
        """
        Process special characters in text.
        
        Args:
            text: Input text
            **options: Processing options
        
        Returns:
            Processed text
        """
        # Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Process diacritics if requested
        if options.get("normalize_diacritics", False):
            text = self.process_diacritics(text)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation marks.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        
        # Normalize ellipsis
        text = text.replace('…', '...')
        
        return text
    
    def process_diacritics(self, text: str, **options) -> str:
        """
        Process diacritical marks.
        
        Args:
            text: Input text
            **options: Processing options
        
        Returns:
            Processed text
        """
        if options.get("remove_diacritics", False):
            # Remove diacritics
            nfd = unicodedata.normalize('NFD', text)
            return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        else:
            # Normalize diacritics
            return unicodedata.normalize('NFC', text)
