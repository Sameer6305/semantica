# Parse

> **Universal data parser supporting documents, web content, structured data, emails, code, and media.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-file-document:{ .lg .middle } **Document Parsing**

    ---

    Extract text, tables, and metadata from PDF, DOCX, PPTX, Excel, and TXT

-   :material-web:{ .lg .middle } **Web Content**

    ---

    Parse HTML, XML, and JavaScript-rendered pages with Selenium/Playwright

-   :material-code-json:{ .lg .middle } **Structured Data**

    ---

    Handle JSON, CSV, XML, and YAML with nested structure preservation

-   :material-email:{ .lg .middle } **Email Parsing**

    ---

    Extract headers, bodies, attachments, and thread structure from MIME messages

-   :material-code-braces:{ .lg .middle } **Code Analysis**

    ---

    Parse source code (Python, JS, etc.) into ASTs, extracting functions and dependencies

-   :material-image:{ .lg .middle } **Media Processing**

    ---

    OCR for images and metadata extraction for audio/video files

</div>

!!! tip "When to Use"
    - **Ingestion**: The first step after loading raw files to convert them into usable text/data
    - **Data Extraction**: Pulling specific fields from structured files (JSON/CSV)
    - **Content Analysis**: Analyzing codebases or email archives
    - **OCR**: Extracting text from scanned documents or images

---

## ⚙️ Algorithms Used

### Document Parsing
- **PDF**: `pdfplumber` for precise layout preservation, table extraction, and image handling. Fallback to `PyPDF2`.
- **Office (DOCX/PPTX/XLSX)**: XML-based parsing of OpenXML formats to extract text, styles, and properties.
- **OCR**: Tesseract-based optical character recognition for image-based PDFs and image files.

### Web Parsing
- **DOM Traversal**: BeautifulSoup for static HTML parsing and element extraction.
- **Headless Browser**: Selenium/Playwright for rendering dynamic JavaScript content before extraction.
- **Content Cleaning**: Heuristic removal of boilerplates (navbars, footers, ads).

### Code Parsing
- **AST Traversal**: Abstract Syntax Tree parsing to identify classes, functions, and imports.
- **Dependency Graphing**: Static analysis of import statements to build dependency networks.
- **Comment Extraction**: Regex and parser-based extraction of docstrings and inline comments.

---

## Main Classes

### DocumentParser

Unified interface for document formats.

**Methods:**

| Method | Description | Supported Formats |
|--------|-------------|-------------------|
| `parse_document(path)` | Auto-detect and parse | PDF, DOCX, PPTX, TXT |
| `parse_pdf(path)` | PDF specific parsing | PDF |
| `parse_docx(path)` | Word specific parsing | DOCX |

**Example:**

```python
from semantica.parse import DocumentParser

parser = DocumentParser()
doc = parser.parse_document("report.pdf")
print(f"Title: {doc.metadata.title}")
print(f"Text: {doc.text[:100]}...")
```

### WebParser

Parses web content.

**Methods:**

| Method | Description |
|--------|-------------|
| `parse_html(url)` | Static HTML parsing |
| `parse_dynamic(url)` | JS-rendered parsing |

### StructuredDataParser

Parses data files.

**Methods:**

| Method | Description |
|--------|-------------|
| `parse_json(path)` | JSON with nesting |
| `parse_csv(path)` | CSV with type inference |

### CodeParser

Parses source code.

**Methods:**

| Method | Description |
|--------|-------------|
| `parse_code(path)` | Extract AST & symbols |
| `get_dependencies(path)` | Find imports |

---

## Convenience Functions

```python
from semantica.parse import parse_document, parse_json, parse_web_content

# Auto-detect format
doc = parse_document("file.pdf")

# Parse specific types
data = parse_json("data.json")
web = parse_web_content("https://google.com")
```

---

## Configuration

### Environment Variables

```bash
export PARSE_OCR_ENABLED=true
export PARSE_OCR_LANG=eng
export PARSE_USER_AGENT="SemanticaBot/1.0"
```

### YAML Configuration

```yaml
parse:
  ocr:
    enabled: true
    language: eng
    
  web:
    user_agent: "MyBot/1.0"
    timeout: 30
    
  pdf:
    extract_tables: true
    extract_images: false
```

---

## Integration Examples

### Ingest & Parse Pipeline

```python
from semantica.ingest import Ingestor
from semantica.parse import DocumentParser, ImageParser

# 1. Ingest Raw File
ingestor = Ingestor()
file_path = ingestor.ingest("scan.png")

# 2. Parse (with OCR)
if file_path.endswith(".png"):
    parser = ImageParser(ocr_enabled=True)
    content = parser.parse_image(file_path)
else:
    parser = DocumentParser()
    content = parser.parse_document(file_path)

print(content.text)
```

---

## Best Practices

1.  **Disable OCR if not needed**: OCR is slow. Only enable it (`ocr_enabled=True`) if you expect scanned documents.
2.  **Use Specific Parsers**: If you know the format, use `parse_json` or `parse_pdf` directly for better type hinting.
3.  **Handle Encodings**: The parser tries to auto-detect encoding, but for CSV/TXT, explicitly specifying it is safer.
4.  **Clean Web Content**: Use `parse_web_content` which includes boilerplate removal, rather than raw HTML parsing.

---

## Troubleshooting

**Issue**: `TesseractNotFoundError`
**Solution**: Install Tesseract OCR on your system (`apt-get install tesseract-ocr` or brew).

**Issue**: PDF tables are messy.
**Solution**: Try `pdfplumber` settings in config or use specialized table extraction tools if layout is complex.

---

## See Also

- [Ingest Module](ingest.md) - Handles file downloading/loading
- [Split Module](split.md) - Chunks the parsed text
- [Semantic Extract Module](semantic_extract.md) - Extracts entities from text
