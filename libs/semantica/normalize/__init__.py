"""
Data Normalization Module

This module provides comprehensive data normalization and cleaning capabilities.

Exports:
    - TextNormalizer: Text cleaning and normalization
    - EntityNormalizer: Entity name normalization
    - DateNormalizer: Date and time normalization
    - NumberNormalizer: Number and quantity normalization
    - DataCleaner: General data cleaning utilities
    - TextCleaner: Text cleaning utilities
    - LanguageDetector: Language detection
    - EncodingHandler: Encoding detection and conversion
"""

from .text_normalizer import (
    TextNormalizer,
    UnicodeNormalizer,
    WhitespaceNormalizer,
    SpecialCharacterProcessor,
)
from .entity_normalizer import (
    EntityNormalizer,
    AliasResolver,
    EntityDisambiguator,
    NameVariantHandler,
)
from .date_normalizer import (
    DateNormalizer,
    TimeZoneNormalizer,
    RelativeDateProcessor,
    TemporalExpressionParser,
)
from .number_normalizer import (
    NumberNormalizer,
    UnitConverter,
    CurrencyNormalizer,
    ScientificNotationHandler,
)
from .data_cleaner import (
    DataCleaner,
    DuplicateDetector,
    DataValidator,
    MissingValueHandler,
    DuplicateGroup,
    ValidationResult,
)
from .text_cleaner import TextCleaner
from .language_detector import LanguageDetector
from .encoding_handler import EncodingHandler

__all__ = [
    "TextNormalizer",
    "UnicodeNormalizer",
    "WhitespaceNormalizer",
    "SpecialCharacterProcessor",
    "EntityNormalizer",
    "AliasResolver",
    "EntityDisambiguator",
    "NameVariantHandler",
    "DateNormalizer",
    "TimeZoneNormalizer",
    "RelativeDateProcessor",
    "TemporalExpressionParser",
    "NumberNormalizer",
    "UnitConverter",
    "CurrencyNormalizer",
    "ScientificNotationHandler",
    "DataCleaner",
    "DuplicateDetector",
    "DataValidator",
    "MissingValueHandler",
    "DuplicateGroup",
    "ValidationResult",
    "TextCleaner",
    "LanguageDetector",
    "EncodingHandler",
]
