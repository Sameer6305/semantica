"""
Data Parsing Module

This module provides comprehensive data parsing capabilities for various file formats,
enabling extraction of text, metadata, and structured data from documents, web content,
emails, code files, and media files.

Key Features:
    - Document format parsing (PDF, DOCX, PPTX, HTML, TXT)
    - Web content parsing (HTML, XML, JavaScript-rendered content)
    - Structured data parsing (JSON, CSV, XML, YAML)
    - Email content parsing (headers, body, attachments, threads)
    - Source code parsing (multi-language, syntax trees, dependencies)
    - Media content parsing (images, audio, video metadata)
    - Batch processing support
    - Metadata extraction
    - Content structure analysis

Main Classes:
    - DocumentParser: Document format parsing (PDF, DOCX, etc.)
    - WebParser: Web content parsing (HTML, XML, etc.)
    - StructuredDataParser: Structured data parsing (JSON, CSV, etc.)
    - EmailParser: Email content parsing
    - CodeParser: Source code parsing
    - MediaParser: Media content parsing
    - PDFParser: PDF document parser
    - DOCXParser: Word document parser
    - PPTXParser: PowerPoint parser
    - ExcelParser: Excel spreadsheet parser
    - HTMLParser: HTML document parser
    - JSONParser: JSON data parser
    - CSVParser: CSV data parser
    - XMLParser: XML document parser
    - ImageParser: Image file parser with OCR

Example Usage:
    >>> from semantica.parse import DocumentParser, WebParser, StructuredDataParser
    >>> doc_parser = DocumentParser()
    >>> text = doc_parser.parse_document("document.pdf")
    >>> web_parser = WebParser()
    >>> content = web_parser.parse_html("https://example.com")
    >>> data_parser = StructuredDataParser()
    >>> data = data_parser.parse_json("data.json")

Author: Semantica Contributors
License: MIT
"""

from .document_parser import DocumentParser
from .web_parser import WebParser, HTMLContentParser, JavaScriptRenderer
from .structured_data_parser import StructuredDataParser
from .email_parser import EmailParser, EmailHeaders, EmailBody, EmailData, MIMEParser, EmailThreadAnalyzer
from .code_parser import CodeParser, CodeStructure, CodeComment, SyntaxTreeParser, CommentExtractor, DependencyAnalyzer
from .media_parser import MediaParser
from .pdf_parser import PDFParser, PDFPage, PDFMetadata
from .docx_parser import DOCXParser, DocxSection, DocxMetadata
from .pptx_parser import PPTXParser, SlideContent, PPTXData
from .excel_parser import ExcelParser, ExcelSheet, ExcelData
from .html_parser import HTMLParser, HTMLMetadata, HTMLElement
from .json_parser import JSONParser, JSONData
from .csv_parser import CSVParser, CSVData
from .xml_parser import XMLParser, XMLElement, XMLData
from .image_parser import ImageParser, ImageMetadata, OCRResult

__all__ = [
    # Main parsers
    "DocumentParser",
    "WebParser",
    "HTMLContentParser",
    "JavaScriptRenderer",
    "StructuredDataParser",
    "EmailParser",
    "EmailHeaders",
    "EmailBody",
    "EmailData",
    "MIMEParser",
    "EmailThreadAnalyzer",
    "CodeParser",
    "CodeStructure",
    "CodeComment",
    "SyntaxTreeParser",
    "CommentExtractor",
    "DependencyAnalyzer",
    "MediaParser",
    # Format-specific parsers
    "PDFParser",
    "PDFPage",
    "PDFMetadata",
    "DOCXParser",
    "DocxSection",
    "DocxMetadata",
    "PPTXParser",
    "SlideContent",
    "PPTXData",
    "ExcelParser",
    "ExcelSheet",
    "ExcelData",
    "HTMLParser",
    "HTMLMetadata",
    "HTMLElement",
    "JSONParser",
    "JSONData",
    "CSVParser",
    "CSVData",
    "XMLParser",
    "XMLElement",
    "XMLData",
    "ImageParser",
    "ImageMetadata",
    "OCRResult",
]
