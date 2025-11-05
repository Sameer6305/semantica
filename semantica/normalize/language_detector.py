"""
Language detection utilities for Semantica framework.

This module provides multi-language detection capabilities
using langdetect and other language identification libraries.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger


class LanguageDetector:
    """Language detection utilities."""
    
    def __init__(self, **config):
        """
        Initialize language detector.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("language_detector")
        self.config = config
        self.default_language = config.get("default_language", "en")
        self.min_confidence = config.get("min_confidence", 0.5)
    
    def detect(self, text: str, **options) -> str:
        """
        Detect language of text.
        
        Args:
            text: Input text
            **options: Detection options
            
        Returns:
            str: Detected language code
        """
        if not text or len(text.strip()) < 10:
            return self.default_language
        
        try:
            language = detect(text)
            return language
        except LangDetectException:
            self.logger.warning(f"Failed to detect language, using default: {self.default_language}")
            return self.default_language
        except Exception as e:
            self.logger.error(f"Language detection error: {e}")
            return self.default_language
    
    def detect_with_confidence(self, text: str, **options) -> Tuple[str, float]:
        """
        Detect language with confidence score.
        
        Args:
            text: Input text
            **options: Detection options
            
        Returns:
            tuple: (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 10:
            return (self.default_language, 0.0)
        
        try:
            languages = detect_langs(text)
            if languages:
                top_language = languages[0]
                if top_language.prob >= self.min_confidence:
                    return (top_language.lang, top_language.prob)
                else:
                    return (self.default_language, top_language.prob)
            else:
                return (self.default_language, 0.0)
        except LangDetectException:
            self.logger.warning(f"Failed to detect language, using default: {self.default_language}")
            return (self.default_language, 0.0)
        except Exception as e:
            self.logger.error(f"Language detection error: {e}")
            return (self.default_language, 0.0)
    
    def detect_multiple(self, text: str, top_n: int = 3, **options) -> List[Tuple[str, float]]:
        """
        Detect top N languages with confidence scores.
        
        Args:
            text: Input text
            top_n: Number of top languages to return
            **options: Detection options
            
        Returns:
            list: List of (language_code, confidence_score) tuples
        """
        if not text or len(text.strip()) < 10:
            return [(self.default_language, 0.0)]
        
        try:
            languages = detect_langs(text)
            if languages:
                return [(lang.lang, lang.prob) for lang in languages[:top_n]]
            else:
                return [(self.default_language, 0.0)]
        except LangDetectException:
            self.logger.warning(f"Failed to detect languages, using default: {self.default_language}")
            return [(self.default_language, 0.0)]
        except Exception as e:
            self.logger.error(f"Language detection error: {e}")
            return [(self.default_language, 0.0)]
    
    def detect_batch(self, texts: List[str], **options) -> List[str]:
        """
        Detect languages for multiple texts in batch.
        
        Args:
            texts: List of texts to analyze
            **options: Detection options
            
        Returns:
            list: List of detected language codes
        """
        return [self.detect(text, **options) for text in texts]
    
    def detect_batch_with_confidence(self, texts: List[str], **options) -> List[Tuple[str, float]]:
        """
        Detect languages with confidence for multiple texts.
        
        Args:
            texts: List of texts to analyze
            **options: Detection options
            
        Returns:
            list: List of (language_code, confidence_score) tuples
        """
        return [self.detect_with_confidence(text, **options) for text in texts]
    
    def is_language(self, text: str, target_language: str, **options) -> bool:
        """
        Check if text is in target language.
        
        Args:
            text: Input text
            target_language: Target language code
            **options: Detection options
            
        Returns:
            bool: True if text is in target language
        """
        detected, confidence = self.detect_with_confidence(text, **options)
        min_confidence = options.get("min_confidence", self.min_confidence)
        return detected == target_language and confidence >= min_confidence
    
    def get_language_name(self, language_code: str) -> str:
        """
        Get language name from code.
        
        Args:
            language_code: Language code (e.g., 'en', 'fr')
            
        Returns:
            str: Language name
        """
        language_names = {
            'en': 'English',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'cs': 'Czech',
            'hu': 'Hungarian',
            'ro': 'Romanian',
            'th': 'Thai',
            'vi': 'Vietnamese'
        }
        return language_names.get(language_code, language_code.upper())
