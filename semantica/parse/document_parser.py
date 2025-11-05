"""
Document Parsing Module

Handles parsing of various document formats.

Key Features:
    - PDF text and metadata extraction
    - DOCX content parsing
    - HTML content cleaning
    - Plain text processing
    - Document structure analysis

Main Classes:
    - DocumentParser: Main document parsing class
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .html_parser import HTMLParser


class DocumentParser:
    """
    Document format parsing handler.
    
    • Parses various document formats (PDF, DOCX, HTML, TXT)
    • Extracts text content and metadata
    • Handles document structure and formatting
    • Processes embedded images and tables
    • Supports batch document processing
    • Handles password-protected documents
    
    Attributes:
        • pdf_parser: PDF parsing engine
        • docx_parser: DOCX parsing engine
        • html_parser: HTML parsing engine
        • text_parser: Plain text processor
        • supported_formats: List of supported formats
        
    Methods:
        • parse_document(): Parse any document format
        • extract_text(): Extract text content
        • extract_metadata(): Extract document metadata
        • parse_batch(): Process multiple documents
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize document parser.
        
        • Setup format-specific parsers
        • Configure parsing options
        • Initialize metadata extractors
        • Setup error handling
        • Configure batch processing
        """
        self.logger = get_logger("document_parser")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Initialize format-specific parsers
        self.pdf_parser = PDFParser(**self.config.get("pdf", {}))
        self.docx_parser = DOCXParser(**self.config.get("docx", {}))
        self.html_parser = HTMLParser(**self.config.get("html", {}))
        
        # Supported formats
        self.supported_formats = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',
            '.html': 'html',
            '.htm': 'html',
            '.txt': 'text',
            '.text': 'text'
        }
    
    def parse_document(self, file_path: Union[str, Path], file_type: Optional[str] = None, **options) -> Dict[str, Any]:
        """
        Parse document of any supported format.
        
        • Detect document format if not specified
        • Route to appropriate format parser
        • Extract text content and structure
        • Extract metadata and properties
        • Handle parsing errors gracefully
        • Return parsed document object
        
        Args:
            file_path: Path to document file
            file_type: Document type (auto-detected if None)
            **options: Parsing options
            
        Returns:
            dict: Parsed document data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"Document file not found: {file_path}")
        
        # Detect file type if not specified
        if file_type is None:
            file_type = self._detect_file_type(file_path)
        
        # Route to appropriate parser
        try:
            if file_type == 'pdf':
                return self.pdf_parser.parse(file_path, **options)
            elif file_type == 'docx':
                return self.docx_parser.parse(file_path, **options)
            elif file_type == 'html':
                return self.html_parser.parse(file_path, **options)
            elif file_type == 'text':
                return self._parse_text(file_path, **options)
            else:
                raise ValidationError(f"Unsupported document format: {file_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to parse document {file_path}: {e}")
            raise ProcessingError(f"Failed to parse document: {e}")
    
    def extract_text(self, file_path: Union[str, Path], **options) -> str:
        """
        Extract text content from document.
        
        • Parse document content
        • Extract all text elements
        • Preserve text structure and formatting
        • Handle special characters and encoding
        • Clean and normalize text
        • Return extracted text content
        
        Args:
            file_path: Path to document file
            **options: Parsing options
            
        Returns:
            str: Extracted text content
        """
        file_path = Path(file_path)
        file_type = self._detect_file_type(file_path)
        
        if file_type == 'pdf':
            return self.pdf_parser.extract_text(file_path, **options)
        elif file_type == 'docx':
            return self.docx_parser.extract_text(file_path)
        elif file_type == 'html':
            return self.html_parser.extract_text(file_path, clean=options.get("clean", True))
        elif file_type == 'text':
            return self._parse_text(file_path, **options).get("text", "")
        else:
            raise ValidationError(f"Unsupported document format: {file_type}")
    
    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract document metadata and properties.
        
        • Parse document properties
        • Extract creation and modification dates
        • Get author and title information
        • Extract document statistics
        • Return metadata dictionary
        
        Args:
            file_path: Path to document file
            
        Returns:
            dict: Document metadata
        """
        file_path = Path(file_path)
        file_type = self._detect_file_type(file_path)
        
        result = self.parse_document(file_path, file_type=file_type, extract_tables=False, extract_images=False)
        return result.get("metadata", {})
    
    def parse_batch(self, file_paths: List[Union[str, Path]], **options) -> Dict[str, Any]:
        """
        Parse multiple documents in batch.
        
        • Process multiple files concurrently
        • Track parsing progress
        • Handle individual file errors
        • Collect parsing results
        • Return batch processing results
        
        Args:
            file_paths: List of document file paths
            **options: Parsing options:
                - max_workers: Maximum parallel workers
                - continue_on_error: Continue on errors (default: True)
                
        Returns:
            dict: Batch processing results
        """
        results = {
            "successful": [],
            "failed": [],
            "total": len(file_paths)
        }
        
        continue_on_error = options.get("continue_on_error", True)
        
        for file_path in file_paths:
            try:
                result = self.parse_document(file_path, **options)
                results["successful"].append({
                    "file_path": str(file_path),
                    "result": result
                })
            except Exception as e:
                error_info = {
                    "file_path": str(file_path),
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                results["failed"].append(error_info)
                
                if not continue_on_error:
                    raise ProcessingError(f"Batch processing failed at {file_path}: {e}")
        
        results["success_count"] = len(results["successful"])
        results["failure_count"] = len(results["failed"])
        
        return results
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect document file type from extension."""
        suffix = file_path.suffix.lower()
        return self.supported_formats.get(suffix, 'unknown')
    
    def _parse_text(self, file_path: Path, **options) -> Dict[str, Any]:
        """Parse plain text file."""
        encoding = options.get("encoding", "utf-8")
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                text = f.read()
            
            return {
                "text": text,
                "metadata": {
                    "file_path": str(file_path),
                    "encoding": encoding,
                    "size": len(text)
                },
                "full_text": text
            }
        except Exception as e:
            self.logger.error(f"Failed to parse text file {file_path}: {e}")
            raise ProcessingError(f"Failed to parse text file: {e}")
