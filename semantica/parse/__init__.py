"""
Data Parsing Module

This module provides comprehensive data parsing capabilities for various file formats.

Exports:
    - DocumentParser: Document format parsing (PDF, DOCX, etc.)
    - WebParser: Web content parsing (HTML, XML, etc.)
    - StructuredDataParser: Structured data parsing (JSON, CSV, etc.)
    - EmailParser: Email content parsing
    - CodeParser: Source code parsing
    - MediaParser: Media content parsing
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
