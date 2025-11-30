# Normalize

> **Clean, standardize, and prepare text and data for semantic processing with comprehensive normalization capabilities.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-text-box-remove:{ .lg .middle } **Text Cleaning**

    ---

    Remove noise, fix encoding issues, and standardize whitespace for clean text

-   :material-format-text:{ .lg .middle } **Entity Normalization**

    ---

    Standardize entity names, abbreviations, and formats across documents

-   :material-calendar-clock:{ .lg .middle } **Date & Time**

    ---

    Parse and standardize date/time formats to ISO 8601

-   :material-numeric:{ .lg .middle } **Number Normalization**

    ---

    Standardize numeric values, units, and measurements

-   :material-translate:{ .lg .middle } **Language Detection**

    ---

    Automatically detect document language with confidence scoring

-   :material-file-code:{ .lg .middle } **Encoding Handling**

    ---

    Fix character encoding issues and ensure UTF-8 compliance

</div>

!!! tip "Why Normalize?"
    Normalization is crucial for:
    
    - **Consistency**: Ensure uniform data representation
    - **Accuracy**: Improve entity extraction and matching
    - **Quality**: Reduce noise and errors in downstream processing
    - **Performance**: Enable better deduplication and search

---

## ⚙️ Algorithms Used

### Text Normalization
- **Unicode Normalization**: NFC, NFD, NFKC, NFKD forms using Unicode standard
- **Whitespace Normalization**: Regex-based cleanup (`\s+` → single space)
- **Case Folding**: Locale-aware case normalization (Unicode case folding)
- **Diacritic Removal**: Unicode decomposition and combining character removal
- **Punctuation Handling**: Smart punctuation normalization preserving sentence structure

### Entity Normalization
- **Fuzzy Matching**: Levenshtein distance with configurable threshold (default: 0.85)
- **Phonetic Matching**: Soundex and Metaphone algorithms for name variants
- **Abbreviation Expansion**: Dictionary-based expansion with context awareness
- **Canonical Form Selection**: Frequency-based or confidence-based selection
- **Entity Linking**: Hash-based entity ID generation for cross-document linking

### Date/Time Normalization
- **Parsing**: dateutil parser with 100+ format support
- **Timezone Handling**: pytz for timezone conversion and DST handling
- **Standardization**: ISO 8601 format output (YYYY-MM-DDTHH:MM:SSZ)
- **Relative Date Resolution**: Convert "yesterday", "last week" to absolute dates
- **Fuzzy Date Parsing**: Handle incomplete dates (e.g., "March 2024")

### Number Normalization
- **Numeric Parsing**: Handle various formats (1,000.00, 1.000,00, 1 000.00)
- **Unit Conversion**: Standardize units (km → meters, lbs → kg)
- **Scientific Notation**: Parse and normalize scientific notation
- **Percentage Handling**: Normalize percentage representations
- **Currency Normalization**: Standardize currency symbols and amounts

### Language Detection
- **N-gram Analysis**: Character and word n-gram frequency analysis
- **Statistical Models**: Language-specific statistical models
- **Confidence Scoring**: Probability-based confidence scores
- **Multi-language Support**: 100+ languages supported

---

## Main Classes

### TextNormalizer

Main text normalization orchestrator with comprehensive cleaning capabilities.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `normalize(text)` | Normalize single text | Full normalization pipeline |
| `normalize_documents(docs)` | Batch normalize documents | Parallel processing |
| `clean(text)` | Clean text only | Noise removal |
| `fix_encoding(text)` | Fix encoding issues | Character encoding detection |
| `normalize_whitespace(text)` | Standardize whitespace | Regex-based cleanup |
| `remove_diacritics(text)` | Remove accents | Unicode decomposition |

**Configuration Options:**

```python
TextNormalizer(
    lowercase=False,              # Convert to lowercase
    remove_punctuation=False,     # Remove all punctuation
    remove_numbers=False,         # Remove numeric values
    remove_whitespace=True,       # Normalize whitespace
    fix_encoding=True,            # Fix encoding issues
    remove_diacritics=False,      # Remove accents/diacritics
    normalize_unicode=True,       # Unicode normalization (NFC)
    remove_urls=False,            # Remove URLs
    remove_emails=False,          # Remove email addresses
    remove_phone_numbers=False,   # Remove phone numbers
    expand_contractions=False,    # Expand contractions (don't → do not)
    remove_html=True,             # Remove HTML tags
    remove_extra_spaces=True,     # Remove extra spaces
    strip=True                    # Strip leading/trailing whitespace
)
```

**Example:**

```python
from semantica.normalize import TextNormalizer

# Basic normalization
normalizer = TextNormalizer(
    lowercase=False,
    remove_punctuation=False,
    fix_encoding=True,
    normalize_whitespace=True
)

text = "  Apple Inc.  was founded in 1976.  "
normalized = normalizer.normalize(text)
print(normalized)
# Output: "Apple Inc. was founded in 1976."

# Aggressive cleaning
aggressive_normalizer = TextNormalizer(
    lowercase=True,
    remove_punctuation=True,
    remove_numbers=True,
    remove_urls=True,
    expand_contractions=True
)

text = "Check out https://example.com! It's amazing (founded in 2020)."
cleaned = aggressive_normalizer.normalize(text)
print(cleaned)
# Output: "check out it is amazing founded in"
```

---

### EntityNormalizer

Standardize entity names and resolve variations to canonical forms.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `normalize(entities)` | Normalize entity list | Fuzzy + phonetic matching |
| `normalize_single(entity)` | Normalize one entity | Canonical form lookup |
| `add_canonical(entity, canonical)` | Add canonical mapping | Dictionary update |
| `get_canonical(entity)` | Get canonical form | Fuzzy search |

**Configuration Options:**

```python
EntityNormalizer(
    fuzzy_matching=True,          # Enable fuzzy matching
    similarity_threshold=0.85,    # Similarity threshold (0-1)
    phonetic_matching=False,      # Enable phonetic matching
    case_sensitive=False,         # Case-sensitive matching
    preserve_case=True,           # Preserve original case in output
    expand_abbreviations=True,    # Expand common abbreviations
    canonical_dict=None           # Custom canonical mappings
)
```

**Example:**

```python
from semantica.normalize import EntityNormalizer

normalizer = EntityNormalizer(
    fuzzy_matching=True,
    similarity_threshold=0.85,
    expand_abbreviations=True
)

# Normalize entity variations
entities = [
    "Apple Inc.",
    "Apple",
    "AAPL",
    "Apple Computer",
    "Apple, Inc."
]

normalized = normalizer.normalize(entities)
print(normalized)
# All mapped to canonical form: "Apple Inc."

# Add custom canonical mappings
normalizer.add_canonical("AAPL", "Apple Inc.")
normalizer.add_canonical("Microsoft Corp", "Microsoft Corporation")

# Get canonical form
canonical = normalizer.get_canonical("MSFT")
print(canonical)  # "Microsoft Corporation"
```

---

### DateNormalizer

Parse and standardize date/time formats to ISO 8601.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `normalize(date_string)` | Parse and normalize date | dateutil parser |
| `normalize_batch(dates)` | Batch normalize dates | Parallel processing |
| `parse_relative(relative_date)` | Parse relative dates | Date arithmetic |
| `detect_format(date_string)` | Detect date format | Pattern matching |

**Configuration Options:**

```python
DateNormalizer(
    output_format="ISO8601",      # ISO8601, UNIX, custom format
    timezone="UTC",               # Target timezone
    handle_relative=True,         # Parse "yesterday", "last week"
    fuzzy=True,                   # Fuzzy parsing
    default_day=1,                # Default day for incomplete dates
    default_month=1               # Default month for incomplete dates
)
```

**Example:**

```python
from semantica.normalize import DateNormalizer

normalizer = DateNormalizer(
    output_format="ISO8601",
    timezone="UTC",
    handle_relative=True
)

# Normalize various date formats
dates = [
    "Jan 1, 2024",
    "01/01/2024",
    "2024-01-01",
    "1st January 2024",
    "yesterday",
    "March 2024"
]

normalized = [normalizer.normalize(d) for d in dates]
for original, norm in zip(dates, normalized):
    print(f"{original:20} → {norm}")

# Output:
# Jan 1, 2024          → 2024-01-01T00:00:00Z
# 01/01/2024           → 2024-01-01T00:00:00Z
# 2024-01-01           → 2024-01-01T00:00:00Z
# 1st January 2024     → 2024-01-01T00:00:00Z
# yesterday            → 2024-11-29T00:00:00Z
# March 2024           → 2024-03-01T00:00:00Z

# Custom format
custom_normalizer = DateNormalizer(output_format="%Y/%m/%d")
result = custom_normalizer.normalize("Jan 1, 2024")
print(result)  # "2024/01/01"
```

---

### NumberNormalizer

Standardize numeric values, units, and measurements.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `normalize(number_string)` | Parse and normalize number | Regex parsing |
| `convert_units(value, from_unit, to_unit)` | Convert units | Unit conversion |
| `parse_currency(currency_string)` | Parse currency | Currency parsing |
| `normalize_percentage(percent_string)` | Normalize percentage | Percentage parsing |

**Example:**

```python
from semantica.normalize import NumberNormalizer

normalizer = NumberNormalizer(
    decimal_separator=".",
    thousands_separator=",",
    normalize_units=True
)

# Normalize various number formats
numbers = [
    "1,000.50",
    "1.000,50",
    "1 000.50",
    "$1,234.56",
    "50%",
    "1.5e3"
]

normalized = [normalizer.normalize(n) for n in numbers]
for original, norm in zip(numbers, normalized):
    print(f"{original:15} → {norm}")

# Unit conversion
distance_km = normalizer.convert_units(5, "km", "m")
print(f"5 km = {distance_km} m")  # 5000 m
```

---

### LanguageDetector

Detect document language with confidence scoring.

**Methods:**

| Method | Description | Algorithm |
|--------|-------------|-----------|
| `detect(text)` | Detect language | N-gram analysis |
| `detect_batch(texts)` | Batch detect languages | Parallel processing |
| `get_confidence(text, language)` | Get confidence score | Statistical model |

**Example:**

```python
from semantica.normalize import LanguageDetector

detector = LanguageDetector()

# Detect language
texts = [
    "Hello, how are you?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?",
    "Hallo, wie geht es dir?",
    "こんにちは、お元気ですか？"
]

for text in texts:
    result = detector.detect(text)
    print(f"{text[:30]:30} → {result['language']} ({result['confidence']:.2f})")

# Output:
# Hello, how are you?           → en (0.99)
# Bonjour, comment allez-vous?  → fr (0.98)
# Hola, ¿cómo estás?            → es (0.97)
# Hallo, wie geht es dir?       → de (0.96)
# こんにちは、お元気ですか？         → ja (0.99)
```

---

## Convenience Functions

Quick access to normalization operations:

```python
from semantica.normalize import (
    normalize_text,
    normalize_entities,
    normalize_dates,
    normalize_numbers,
    detect_language,
    fix_encoding
)

# Normalize text
clean_text = normalize_text("  Messy   text  ", remove_extra_spaces=True)

# Normalize entities
canonical_entities = normalize_entities(["Apple Inc.", "Apple", "AAPL"])

# Normalize dates
iso_dates = normalize_dates(["Jan 1, 2024", "01/01/2024"])

# Normalize numbers
standard_numbers = normalize_numbers(["1,000.50", "$1,234.56"])

# Detect language
language = detect_language("Hello, world!")

# Fix encoding
fixed_text = fix_encoding("Caf\u00e9")  # Café
```

---

## Configuration

### Environment Variables

```bash
# Normalization settings
export NORMALIZE_DEFAULT_LOWERCASE=false
export NORMALIZE_DEFAULT_ENCODING=utf-8
export NORMALIZE_DEFAULT_TIMEZONE=UTC

# Entity normalization
export NORMALIZE_ENTITY_SIMILARITY_THRESHOLD=0.85
export NORMALIZE_ENTITY_FUZZY_MATCHING=true

# Language detection
export NORMALIZE_LANGUAGE_DETECTOR=langdetect
export NORMALIZE_LANGUAGE_CONFIDENCE_THRESHOLD=0.8
```

### YAML Configuration

```yaml
# config.yaml - Normalize Module Configuration

normalize:
  text:
    lowercase: false
    remove_punctuation: false
    fix_encoding: true
    normalize_whitespace: true
    remove_urls: false
    expand_contractions: false
    
  entity:
    fuzzy_matching: true
    similarity_threshold: 0.85
    phonetic_matching: false
    expand_abbreviations: true
    
  date:
    output_format: "ISO8601"
    timezone: "UTC"
    handle_relative: true
    fuzzy: true
    
  number:
    decimal_separator: "."
    thousands_separator: ","
    normalize_units: true
    
  language:
    detector: "langdetect"  # langdetect, fasttext
    confidence_threshold: 0.8
    fallback_language: "en"
```

---

## Integration Examples

### Complete Document Normalization Pipeline

```python
from semantica.normalize import TextNormalizer, EntityNormalizer, DateNormalizer, LanguageDetector
from semantica.parse import DocumentParser

# Parse documents
parser = DocumentParser()
documents = parser.parse(["document1.pdf", "document2.docx"])

# Detect language
detector = LanguageDetector()
for doc in documents:
    lang_result = detector.detect(doc.content)
    doc.metadata["language"] = lang_result["language"]
    doc.metadata["language_confidence"] = lang_result["confidence"]

# Normalize text
text_normalizer = TextNormalizer(
    fix_encoding=True,
    normalize_whitespace=True,
    remove_html=True
)

for doc in documents:
    doc.content = text_normalizer.normalize(doc.content)

# Normalize dates in metadata
date_normalizer = DateNormalizer(output_format="ISO8601")
for doc in documents:
    if "date" in doc.metadata:
        doc.metadata["date"] = date_normalizer.normalize(doc.metadata["date"])

# Normalize entities
entity_normalizer = EntityNormalizer(similarity_threshold=0.85)
# ... entity normalization logic
```

### Multi-Language Document Processing

```python
from semantica.normalize import LanguageDetector, TextNormalizer

detector = LanguageDetector()
normalizers = {
    "en": TextNormalizer(expand_contractions=True),
    "fr": TextNormalizer(remove_diacritics=False),
    "de": TextNormalizer(lowercase=False)
}

def process_multilingual_document(text):
    # Detect language
    lang_result = detector.detect(text)
    language = lang_result["language"]
    
    # Use language-specific normalizer
    normalizer = normalizers.get(language, TextNormalizer())
    normalized = normalizer.normalize(text)
    
    return {
        "text": normalized,
        "language": language,
        "confidence": lang_result["confidence"]
    }

# Process documents
documents = ["Hello world", "Bonjour le monde", "Hallo Welt"]
results = [process_multilingual_document(doc) for doc in documents]
```

---

## Best Practices

### 1. Choose Appropriate Normalization Level

```python
# Minimal normalization for entity extraction
minimal = TextNormalizer(
    fix_encoding=True,
    normalize_whitespace=True
)

# Moderate normalization for search
moderate = TextNormalizer(
    fix_encoding=True,
    normalize_whitespace=True,
    lowercase=True,
    remove_urls=True
)

# Aggressive normalization for topic modeling
aggressive = TextNormalizer(
    lowercase=True,
    remove_punctuation=True,
    remove_numbers=True,
    remove_urls=True,
    expand_contractions=True
)
```

### 2. Preserve Original Data

```python
# Always keep original text
doc.original_content = doc.content
doc.content = normalizer.normalize(doc.content)

# Store normalization metadata
doc.metadata["normalized"] = True
doc.metadata["normalization_config"] = normalizer.config
```

### 3. Batch Processing for Performance

```python
# Batch normalize for better performance
texts = [doc.content for doc in documents]
normalized_texts = normalizer.normalize_documents(texts)

for doc, normalized in zip(documents, normalized_texts):
    doc.content = normalized
```

---

## Troubleshooting

### Common Issues

**Issue**: Encoding errors with special characters

```python
# Solution: Enable encoding fix
normalizer = TextNormalizer(fix_encoding=True)

# Or manually fix encoding
from semantica.normalize import fix_encoding
fixed_text = fix_encoding(problematic_text)
```

**Issue**: Over-normalization losing important information

```python
# Solution: Use conservative settings
normalizer = TextNormalizer(
    lowercase=False,           # Keep case
    remove_punctuation=False,  # Keep punctuation
    remove_numbers=False       # Keep numbers
)
```

**Issue**: Slow processing for large documents

```python
# Solution: Use batch processing
normalizer = TextNormalizer()
normalized = normalizer.normalize_documents(
    documents,
    batch_size=100,
    n_jobs=4  # Parallel processing
)
```

---

## Performance Tips

### Memory Optimization

```python
# Process documents in chunks
def normalize_large_corpus(documents, chunk_size=1000):
    normalizer = TextNormalizer()
    
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        normalized_chunk = normalizer.normalize_documents(chunk)
        yield from normalized_chunk
```

### Speed Optimization

```python
# Disable unnecessary features
fast_normalizer = TextNormalizer(
    fix_encoding=False,        # Skip if encoding is known good
    normalize_unicode=False,   # Skip if not needed
    remove_diacritics=False    # Skip if not needed
)

# Use parallel processing
normalizer.normalize_documents(
    documents,
    n_jobs=-1  # Use all CPU cores
)
```

---

## See Also

- [Parse Module](parse.md) - Document parsing and extraction
- [Semantic Extract Module](semantic_extract.md) - Entity and relation extraction
- [Split Module](split.md) - Text chunking and splitting
- [Ingest Module](ingest.md) - Data ingestion
