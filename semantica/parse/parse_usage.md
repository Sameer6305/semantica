# Data Parsing Module Usage Guide

This comprehensive guide demonstrates how to use the data parsing module for document parsing, web content parsing, structured data parsing, email parsing, code parsing, and media parsing. The module supports parsing various file formats including PDF, DOCX, PPTX, HTML, JSON, CSV, XML, images, and more.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Document Parsing](#document-parsing)
3. [Web Content Parsing](#web-content-parsing)
4. [Structured Data Parsing](#structured-data-parsing)
5. [Email Parsing](#email-parsing)
6. [Code Parsing](#code-parsing)
7. [Media Parsing](#media-parsing)
8. [Format-Specific Parsers](#format-specific-parsers)
9. [Using Methods](#using-methods)
10. [Using Registry](#using-registry)
11. [Configuration](#configuration)
12. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using the Convenience Functions

```python
from semantica.parse import parse_document, parse_web_content, parse_json

# Parse a PDF document
doc = parse_document("document.pdf", method="default")
print(f"Text: {doc.get('full_text', '')}")
print(f"Pages: {doc.get('total_pages', 0)}")

# Parse web content
web = parse_web_content("https://example.com", method="default")
print(f"Content: {web.get('text', '')}")

# Parse JSON data
data = parse_json("data.json", method="default")
print(f"Data: {data.data}")
```

### Using Main Classes

```python
from semantica.parse import DocumentParser, WebParser, StructuredDataParser

# Create parsers
doc_parser = DocumentParser()
web_parser = WebParser()
data_parser = StructuredDataParser()

# Parse document
doc = doc_parser.parse_document("document.pdf")
print(f"Text: {doc.get('full_text', '')}")

# Parse web content
web = web_parser.parse_web_content("https://example.com", content_type="html")
print(f"Content: {web.get('text', '')}")

# Parse structured data
data = data_parser.parse_data("data.json", data_format="json")
print(f"Data: {data}")
```

## Document Parsing

### PDF Parsing

```python
from semantica.parse import parse_pdf, PDFParser

# Using convenience function
pdf = parse_pdf("document.pdf", method="default", extract_tables=True)
print(f"Text: {pdf.get('full_text', '')}")
print(f"Pages: {pdf.get('total_pages', 0)}")
print(f"Tables: {pdf.get('pages', [{}])[0].get('tables', [])}")

# Using PDFParser directly
pdf_parser = PDFParser()
pdf_data = pdf_parser.parse("document.pdf", extract_text=True, extract_tables=True)

# Extract specific pages
pdf_pages = pdf_parser.parse("document.pdf", pages=[1, 2, 3])

# Extract only text
text = pdf_parser.extract_text("document.pdf")

# Extract only tables
tables = pdf_parser.extract_tables("document.pdf")
```

### DOCX Parsing

```python
from semantica.parse import parse_docx, DOCXParser

# Using convenience function
docx = parse_docx("document.docx", method="default")
print(f"Text: {docx.get('text', '')}")
print(f"Sections: {docx.get('sections', [])}")

# Using DOCXParser directly
docx_parser = DOCXParser()
docx_data = docx_parser.parse("document.docx", extract_tables=True)

# Extract sections
sections = docx_parser.extract_sections("document.docx")

# Extract metadata
metadata = docx_parser.extract_metadata("document.docx")
print(f"Author: {metadata.author}")
print(f"Title: {metadata.title}")
```

### PPTX Parsing

```python
from semantica.parse import PPTXParser

pptx_parser = PPTXParser()
pptx_data = pptx_parser.parse("presentation.pptx")

print(f"Title: {pptx_data.title}")
print(f"Slides: {len(pptx_data.slides)}")

for slide in pptx_data.slides:
    print(f"Slide {slide.slide_number}: {slide.title}")
    print(f"  Text: {slide.text}")
    print(f"  Notes: {slide.notes}")
```

### Excel Parsing

```python
from semantica.parse import ExcelParser

excel_parser = ExcelParser()
excel_data = excel_parser.parse("spreadsheet.xlsx")

print(f"Sheets: {excel_data.sheet_names}")

for sheet_name, sheet in excel_data.sheets.items():
    print(f"Sheet: {sheet_name}")
    print(f"  Rows: {sheet.row_count}")
    print(f"  Columns: {sheet.column_count}")
    print(f"  Headers: {sheet.headers}")
```

### HTML Document Parsing

```python
from semantica.parse import parse_document, HTMLParser

# Using convenience function
html_doc = parse_document("page.html", method="default")
print(f"Text: {html_doc.get('text', '')}")

# Using HTMLParser directly
html_parser = HTMLParser()
html_data = html_parser.parse("page.html", extract_links=True, extract_images=True)

# Extract metadata
metadata = html_parser.extract_metadata("page.html")
print(f"Title: {metadata.title}")
print(f"Description: {metadata.description}")

# Extract links
links = html_parser.extract_links("page.html")
for link in links:
    print(f"Link: {link.get('href', '')} - {link.get('text', '')}")
```

### Text File Parsing

```python
from semantica.parse import parse_document

# Parse plain text file
text_doc = parse_document("document.txt", method="default")
print(f"Text: {text_doc.get('text', '')}")
```

## Web Content Parsing

### HTML Content Parsing

```python
from semantica.parse import parse_web_content, WebParser

# Using convenience function
web = parse_web_content("https://example.com", content_type="html", method="default")
print(f"Text: {web.get('text', '')}")

# Using WebParser directly
web_parser = WebParser()
html_data = web_parser.parse_web_content("https://example.com", content_type="html")

# Extract text
text = web_parser.extract_text("https://example.com")

# Extract links
links = web_parser.extract_links("https://example.com")
for link in links:
    print(f"Link: {link.get('href', '')}")

# Render JavaScript content
rendered = web_parser.render_javascript("https://example.com", wait_time=5)
```

### XML Content Parsing

```python
from semantica.parse import parse_web_content, XMLParser

# Using convenience function
xml_data = parse_web_content("data.xml", content_type="xml", method="default")

# Using XMLParser directly
xml_parser = XMLParser()
xml_data = xml_parser.parse("data.xml")

# Find elements using XPath
elements = xml_parser.find_elements("data.xml", "//item")
for element in elements:
    print(f"Element: {element.tag} - {element.text}")
```

### JavaScript-Rendered Content

```python
from semantica.parse import WebParser, JavaScriptRenderer

# Using JavaScript renderer
js_renderer = JavaScriptRenderer()
rendered_html = js_renderer.render_page("https://example.com", wait_time=5)

# Using WebParser with JavaScript rendering
web_parser = WebParser()
rendered_content = web_parser.parse_web_content(
    "https://example.com",
    content_type="html",
    render_javascript=True,
    wait_time=5
)
```

## Structured Data Parsing

### JSON Parsing

```python
from semantica.parse import parse_json, JSONParser

# Using convenience function
json_data = parse_json("data.json", method="default")
print(f"Data: {json_data.data}")
print(f"Type: {json_data.type}")

# Using JSONParser directly
json_parser = JSONParser()
json_data = json_parser.parse("data.json", flatten=True)

# Extract JSON paths
paths = json_parser.extract_paths("data.json")
for path in paths:
    print(f"Path: {path}")
```

### CSV Parsing

```python
from semantica.parse import parse_csv, CSVParser

# Using convenience function
csv_data = parse_csv("data.csv", delimiter=",", method="default")
print(f"Headers: {csv_data.headers}")
print(f"Rows: {csv_data.row_count}")

# Using CSVParser directly
csv_parser = CSVParser()
csv_data = csv_parser.parse("data.csv", delimiter=",", has_header=True)

# Parse to dictionary list
rows = csv_parser.parse_to_dict("data.csv")
for row in rows:
    print(f"Row: {row}")
```

### XML Parsing

```python
from semantica.parse import parse_xml, XMLParser

# Using convenience function
xml_data = parse_xml("data.xml", method="default")
print(f"Root: {xml_data.root.tag}")

# Using XMLParser directly
xml_parser = XMLParser()
xml_data = xml_parser.parse("data.xml", engine="lxml")

# Find elements
elements = xml_parser.find_elements("data.xml", "//item")
for element in elements:
    print(f"Element: {element.tag} - {element.text}")
```

### YAML Parsing

```python
from semantica.parse import StructuredDataParser

data_parser = StructuredDataParser()
yaml_data = data_parser.parse_data("config.yaml", data_format="yaml")
print(f"Data: {yaml_data}")
```

## Email Parsing

### Basic Email Parsing

```python
from semantica.parse import parse_email, EmailParser

# Using convenience function
email = parse_email("email.eml", method="default")
print(f"Subject: {email.headers.subject}")
print(f"From: {email.headers.from_address}")
print(f"To: {email.headers.to_addresses}")
print(f"Body: {email.body.text}")

# Using EmailParser directly
email_parser = EmailParser()
email_data = email_parser.parse_email("email.eml", extract_attachments=True)

# Extract headers only
headers = email_parser.parse_headers("email.eml")
print(f"Subject: {headers.subject}")
print(f"Date: {headers.date}")

# Extract body only
body = email_parser.extract_body("email.eml")
print(f"Text: {body.text}")
print(f"HTML: {body.html}")
```

### Email Thread Analysis

```python
from semantica.parse import EmailParser

email_parser = EmailParser()

# Analyze email thread
thread = email_parser.analyze_thread("email.eml")
print(f"Thread ID: {thread.thread_id}")
print(f"Messages: {len(thread.messages)}")
print(f"Subject: {thread.subject}")

# Parse multiple emails in a thread
for email_file in ["email1.eml", "email2.eml", "email3.eml"]:
    email = email_parser.parse_email(email_file)
    print(f"Email: {email.headers.subject}")
```

### Email Attachment Extraction

```python
from semantica.parse import EmailParser

email_parser = EmailParser()
email_data = email_parser.parse_email("email.eml", extract_attachments=True)

# Access attachments
for attachment in email_data.body.attachments:
    print(f"Filename: {attachment.get('filename', '')}")
    print(f"Content Type: {attachment.get('content_type', '')}")
    print(f"Size: {attachment.get('size', 0)}")
```

## Code Parsing

### Basic Code Parsing

```python
from semantica.parse import parse_code, CodeParser

# Using convenience function
code = parse_code("script.py", method="default")
print(f"Functions: {code.get('structure', {}).get('functions', [])}")
print(f"Classes: {code.get('structure', {}).get('classes', [])}")
print(f"Imports: {code.get('structure', {}).get('imports', [])}")

# Using CodeParser directly
code_parser = CodeParser()
code_data = code_parser.parse_code("script.py", language="python")

# Extract structure
structure = code_parser.extract_structure("script.py", language="python")
print(f"Functions: {structure.functions}")
print(f"Classes: {structure.classes}")

# Extract comments
comments = code_parser.extract_comments("script.py", language="python")
for comment in comments:
    print(f"Comment: {comment.text} (Line {comment.line_number})")

# Analyze dependencies
dependencies = code_parser.analyze_dependencies("script.py", language="python")
print(f"Dependencies: {dependencies}")
```

### AST Parsing

```python
from semantica.parse import CodeParser, SyntaxTreeParser

code_parser = CodeParser()
syntax_parser = SyntaxTreeParser()

# Parse syntax tree
tree = syntax_parser.parse_syntax_tree("script.py", language="python")

# Extract functions
functions = syntax_parser.extract_functions("script.py", language="python")
for func in functions:
    print(f"Function: {func.get('name', '')}")

# Extract classes
classes = syntax_parser.extract_classes("script.py", language="python")
for cls in classes:
    print(f"Class: {cls.get('name', '')}")

# Extract imports
imports = syntax_parser.extract_imports("script.py", language="python")
for imp in imports:
    print(f"Import: {imp}")
```

### Multi-Language Code Parsing

```python
from semantica.parse import CodeParser

code_parser = CodeParser()

# Parse Python code
python_code = code_parser.parse_code("script.py", language="python")

# Parse JavaScript code
js_code = code_parser.parse_code("script.js", language="javascript")

# Parse Java code
java_code = code_parser.parse_code("Main.java", language="java")
```

## Media Parsing

### Image Parsing with OCR

```python
from semantica.parse import parse_image, ImageParser

# Using convenience function
image = parse_image("image.jpg", method="default", extract_text=True)
print(f"Format: {image.get('metadata', {}).get('format', '')}")
print(f"Size: {image.get('metadata', {}).get('size', (0, 0))}")

# Extract OCR text
if "ocr_result" in image:
    print(f"OCR Text: {image['ocr_result'].text}")
    print(f"Confidence: {image['ocr_result'].confidence}")

# Using ImageParser directly
image_parser = ImageParser()
image_data = image_parser.parse("image.jpg", extract_text=True, ocr_language="eng")

# Extract metadata only
metadata = image_parser.extract_metadata("image.jpg")
print(f"Format: {metadata.format}")
print(f"Size: {metadata.size}")
print(f"EXIF: {metadata.exif}")

# Extract text with OCR
ocr_result = image_parser.extract_text("image.jpg", language="eng")
print(f"Text: {ocr_result.text}")
print(f"Confidence: {ocr_result.confidence}")
```

### Media File Parsing

```python
from semantica.parse import parse_media, MediaParser

# Using convenience function
media = parse_media("video.mp4", method="default")
print(f"Type: {media.get('media_type', '')}")
print(f"Metadata: {media.get('metadata', {})}")

# Using MediaParser directly
media_parser = MediaParser()
media_data = media_parser.parse_media("image.jpg", media_type="image")

# Get supported formats
formats = media_parser.get_supported_formats()
print(f"Supported formats: {formats}")
```

## Format-Specific Parsers

### PDF Parser

```python
from semantica.parse import PDFParser, PDFPage, PDFMetadata

pdf_parser = PDFParser()

# Parse PDF
pdf_data = pdf_parser.parse("document.pdf", extract_text=True, extract_tables=True)

# Access pages
for page_dict in pdf_data.get("pages", []):
    page = PDFPage(**page_dict)
    print(f"Page {page.page_number}: {len(page.text)} characters")
    print(f"  Tables: {len(page.tables)}")
    print(f"  Images: {len(page.images)}")

# Access metadata
metadata = PDFMetadata(**pdf_data.get("metadata", {}))
print(f"Title: {metadata.title}")
print(f"Author: {metadata.author}")
print(f"Page Count: {metadata.page_count}")
```

### DOCX Parser

```python
from semantica.parse import DOCXParser, DocxSection, DocxMetadata

docx_parser = DOCXParser()

# Parse DOCX
docx_data = docx_parser.parse("document.docx", extract_tables=True)

# Access sections
for section_dict in docx_data.get("sections", []):
    section = DocxSection(**section_dict)
    print(f"Section: {section.heading} (Level {section.level})")
    print(f"  Content: {section.content[:100]}...")

# Access metadata
metadata = DocxMetadata(**docx_data.get("metadata", {}))
print(f"Title: {metadata.title}")
print(f"Author: {metadata.author}")
```

### JSON Parser

```python
from semantica.parse import JSONParser, JSONData

json_parser = JSONParser()

# Parse JSON
json_data = json_parser.parse("data.json", flatten=True)

# Access data
print(f"Type: {json_data.type}")
print(f"Data: {json_data.data}")
print(f"Metadata: {json_data.metadata}")

# Extract paths
paths = json_parser.extract_paths("data.json")
for path in paths:
    print(f"Path: {path}")
```

### CSV Parser

```python
from semantica.parse import CSVParser, CSVData

csv_parser = CSVParser()

# Parse CSV
csv_data = csv_parser.parse("data.csv", delimiter=",", has_header=True)

# Access data
print(f"Headers: {csv_data.headers}")
print(f"Row Count: {csv_data.row_count}")

# Access rows
for row in csv_data.rows:
    print(f"Row: {row}")

# Parse to dictionary list
rows = csv_parser.parse_to_dict("data.csv")
for row in rows:
    print(f"Row: {row}")
```

### XML Parser

```python
from semantica.parse import XMLParser, XMLData, XMLElement

xml_parser = XMLParser()

# Parse XML
xml_data = xml_parser.parse("data.xml", engine="lxml")

# Access root element
root = xml_data.root
print(f"Root Tag: {root.tag}")
print(f"Root Text: {root.text}")

# Access children
for child in root.children:
    print(f"Child: {child.tag} - {child.text}")

# Find elements
elements = xml_parser.find_elements("data.xml", "//item")
for element in elements:
    print(f"Element: {element.tag} - {element.text}")
```

### Image Parser

```python
from semantica.parse import ImageParser, ImageMetadata, OCRResult

image_parser = ImageParser()

# Parse image
image_data = image_parser.parse("image.jpg", extract_text=True, ocr_language="eng")

# Access metadata
metadata = ImageMetadata(**image_data.get("metadata", {}))
print(f"Format: {metadata.format}")
print(f"Size: {metadata.size}")
print(f"EXIF: {metadata.exif}")

# Access OCR result
if "ocr_result" in image_data:
    ocr = OCRResult(**image_data["ocr_result"])
    print(f"OCR Text: {ocr.text}")
    print(f"Confidence: {ocr.confidence}")
    print(f"Language: {ocr.language}")
```

## Using Methods

### Method Selection

```python
from semantica.parse import parse_document, parse_web_content, parse_json

# Use default method
doc = parse_document("document.pdf", method="default")

# Use specific method (if registered)
doc = parse_document("document.pdf", method="custom_pdf_parser")

# List available methods
from semantica.parse import list_available_methods

methods = list_available_methods("document")
print(f"Available document methods: {methods}")
```

### Custom Method Registration

```python
from semantica.parse import method_registry, parse_document

# Define custom parsing method
def custom_document_parser(file_path, file_type=None, **kwargs):
    # Custom parsing logic
    return {"text": "Custom parsed text", "metadata": {}}

# Register custom method
method_registry.register("document", "custom", custom_document_parser)

# Use custom method
doc = parse_document("document.pdf", method="custom")
```

## Using Registry

### Registering Custom Methods

```python
from semantica.parse import method_registry

# Register document parsing method
def my_document_parser(file_path, file_type=None, **kwargs):
    # Custom implementation
    return {"text": "Parsed", "metadata": {}}

method_registry.register("document", "my_parser", my_document_parser)

# Register web parsing method
def my_web_parser(content, content_type="html", base_url=None, **kwargs):
    # Custom implementation
    return {"text": "Parsed", "links": []}

method_registry.register("web", "my_parser", my_web_parser)
```

### Listing Registered Methods

```python
from semantica.parse import method_registry

# List all methods
all_methods = method_registry.list_all()
print(f"All methods: {all_methods}")

# List methods for specific task
document_methods = method_registry.list_all("document")
print(f"Document methods: {document_methods}")

web_methods = method_registry.list_all("web")
print(f"Web methods: {web_methods}")
```

### Getting Methods

```python
from semantica.parse import method_registry, get_parse_method

# Get method directly
method = method_registry.get("document", "default")
if method:
    result = method("document.pdf")

# Using convenience function
method = get_parse_method("document", "default")
if method:
    result = method("document.pdf")
```

### Unregistering Methods

```python
from semantica.parse import method_registry

# Unregister specific method
method_registry.unregister("document", "custom")

# Clear all methods for a task
method_registry.clear("document")

# Clear all methods
method_registry.clear()
```

## Configuration

### Environment Variables

```python
import os

# Set environment variables
os.environ["PARSE_DEFAULT_ENCODING"] = "utf-8"
os.environ["PARSE_OCR_LANGUAGE"] = "eng"
os.environ["PARSE_EXTRACT_TABLES"] = "true"
os.environ["PARSE_EXTRACT_IMAGES"] = "false"
os.environ["PARSE_EXTRACT_METADATA"] = "true"
os.environ["PARSE_JS_RENDERING"] = "false"
```

### Config File

Create a `config.yaml` file:

```yaml
parse:
  default_encoding: "utf-8"
  ocr_language: "eng"
  extract_tables: true
  extract_images: false
  extract_metadata: true
  js_rendering: false

parse_methods:
  document:
    extract_tables: true
    extract_images: false
  web:
    render_javascript: false
  structured:
    encoding: "utf-8"
```

Load config:

```python
from semantica.parse.config import ParseConfig

# Load from config file
parse_config = ParseConfig(config_file="config.yaml")

# Access configuration
encoding = parse_config.get("default_encoding", default="utf-8")
ocr_lang = parse_config.get("ocr_language", default="eng")
```

### Programmatic Configuration

```python
from semantica.parse import parse_config

# Set configuration
parse_config.set("default_encoding", "utf-8")
parse_config.set("ocr_language", "eng")
parse_config.set("extract_tables", True)

# Get configuration
encoding = parse_config.get("default_encoding", default="utf-8")
ocr_lang = parse_config.get("ocr_language", default="eng")

# Method-specific configuration
parse_config.set_method_config("document", extract_tables=True, extract_images=False)
parse_config.set_method_config("web", render_javascript=False)

# Get method configuration
doc_config = parse_config.get_method_config("document")
web_config = parse_config.get_method_config("web")

# Get all configuration
all_config = parse_config.get_all()
print(f"All config: {all_config}")
```

## Advanced Examples

### Batch Document Processing

```python
from semantica.parse import parse_document
from pathlib import Path

# Process multiple documents
documents = ["doc1.pdf", "doc2.docx", "doc3.html"]
results = []

for doc_path in documents:
    try:
        result = parse_document(doc_path, method="default")
        results.append({
            "file": doc_path,
            "text": result.get("full_text", ""),
            "metadata": result.get("metadata", {})
        })
    except Exception as e:
        print(f"Error parsing {doc_path}: {e}")

# Process all PDFs in a directory
pdf_dir = Path("documents")
pdf_files = list(pdf_dir.glob("*.pdf"))

for pdf_file in pdf_files:
    result = parse_document(pdf_file, method="default")
    print(f"Processed: {pdf_file.name} ({result.get('total_pages', 0)} pages)")
```

### Multi-Format Data Extraction

```python
from semantica.parse import (
    parse_document, parse_json, parse_csv, parse_xml
)

# Extract data from multiple formats
files = {
    "document.pdf": parse_document,
    "data.json": parse_json,
    "data.csv": parse_csv,
    "data.xml": parse_xml,
}

extracted_data = {}

for file_path, parser_func in files.items():
    try:
        data = parser_func(file_path, method="default")
        extracted_data[file_path] = data
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

# Process extracted data
for file_path, data in extracted_data.items():
    print(f"File: {file_path}")
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
```

### Email Thread Reconstruction

```python
from semantica.parse import EmailParser
from pathlib import Path

email_parser = EmailParser()

# Process email thread
email_files = ["email1.eml", "email2.eml", "email3.eml"]
thread_messages = []

for email_file in email_files:
    email = email_parser.parse_email(email_file)
    thread_messages.append({
        "subject": email.headers.subject,
        "from": email.headers.from_address,
        "date": email.headers.date,
        "body": email.body.text,
        "in_reply_to": email.headers.in_reply_to,
        "references": email.headers.references
    })

# Sort by date
thread_messages.sort(key=lambda x: x["date"] or "")

# Display thread
for i, msg in enumerate(thread_messages, 1):
    print(f"Message {i}:")
    print(f"  From: {msg['from']}")
    print(f"  Subject: {msg['subject']}")
    print(f"  Date: {msg['date']}")
    print(f"  Body: {msg['body'][:100]}...")
```

### Code Repository Analysis

```python
from semantica.parse import CodeParser
from pathlib import Path

code_parser = CodeParser()

# Analyze code repository
repo_path = Path("repository")
code_files = list(repo_path.rglob("*.py"))

all_functions = []
all_classes = []
all_imports = []

for code_file in code_files:
    try:
        code_data = code_parser.parse_code(code_file, language="python")
        structure = code_data.get("structure", {})
        
        all_functions.extend(structure.get("functions", []))
        all_classes.extend(structure.get("classes", []))
        all_imports.extend(structure.get("imports", []))
    except Exception as e:
        print(f"Error parsing {code_file}: {e}")

print(f"Total functions: {len(all_functions)}")
print(f"Total classes: {len(all_classes)}")
print(f"Total imports: {len(set(all_imports))}")
```

### OCR Batch Processing

```python
from semantica.parse import parse_image
from pathlib import Path

# Process all images in a directory
image_dir = Path("images")
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

ocr_results = []

for image_file in image_files:
    try:
        image_data = parse_image(image_file, method="default", extract_text=True)
        
        if "ocr_result" in image_data:
            ocr_results.append({
                "file": image_file.name,
                "text": image_data["ocr_result"].text,
                "confidence": image_data["ocr_result"].confidence
            })
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# Display results
for result in ocr_results:
    print(f"File: {result['file']}")
    print(f"  Text: {result['text'][:100]}...")
    print(f"  Confidence: {result['confidence']:.2f}")
```

### Custom Parser Pipeline

```python
from semantica.parse import (
    parse_document, parse_json, parse_csv,
    method_registry
)

# Define custom pipeline
def pipeline_parser(file_path, file_type=None, **kwargs):
    # Step 1: Parse document
    doc = parse_document(file_path, file_type=file_type, **kwargs)
    
    # Step 2: Extract structured data if available
    structured_data = {}
    if "json" in str(file_path):
        structured_data = parse_json(file_path, **kwargs)
    elif "csv" in str(file_path):
        structured_data = parse_csv(file_path, **kwargs)
    
    # Step 3: Combine results
    return {
        "document": doc,
        "structured": structured_data,
        "combined_text": doc.get("full_text", "") + str(structured_data)
    }

# Register pipeline
method_registry.register("document", "pipeline", pipeline_parser)

# Use pipeline
result = parse_document("document.pdf", method="pipeline")
print(f"Combined text: {result['combined_text']}")
```

This comprehensive guide covers all major features of the data parsing module. For more specific use cases or advanced scenarios, refer to the individual parser class documentation or explore the source code.

