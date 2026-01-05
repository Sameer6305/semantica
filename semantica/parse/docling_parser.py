"""
Docling Document Parser Module

This module provides a standalone document parser that uses Docling as its core dependency.
DoclingParser is completely independent from DocumentParser and uses only:
    - docling: Core document parsing library (DocumentConverter)
    - semantica utilities: Logging, progress tracking, exceptions

Key Features:
    - Multi-format document parsing (PDF, DOCX, PPTX, XLSX, HTML, images)
    - Superior table extraction accuracy
    - Enhanced document structure understanding
    - Markdown, HTML, and JSON export formats
    - Local execution support
    - OCR support for scanned documents
    - Standalone parser - no dependency on DocumentParser

Core Dependency:
    - docling: Required for all parsing functionality

Main Classes:
    - DoclingParser: Standalone Docling-based document parser

Example Usage:
    >>> from semantica.parse import DoclingParser
    >>> parser = DoclingParser()
    >>> result = parser.parse("document.pdf")
    >>> text = parser.extract_text("document.pdf")
    >>> tables = parser.extract_tables("document.pdf")

Author: Semantica Contributors
License: MIT
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.helpers import safe_import
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker

# Try to import docling, handle gracefully if not available
DOCLING_AVAILABLE = False
DOCLING_IMPORT_ERROR = None
DocumentConverter = None
InputFormat = None
PdfPipelineOptions = None

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
    DOCLING_IMPORT_ERROR = None
except (ImportError, OSError) as e:
    DOCLING_AVAILABLE = False
    DOCLING_IMPORT_ERROR = str(e)


@dataclass
class DoclingMetadata:
    """Document metadata representation from Docling."""

    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    format: Optional[str] = None


class DoclingParser:
    """Docling-based document parser for enhanced table extraction."""

    def __init__(self, **config):
        """
        Initialize Docling parser.

        Args:
            **config: Parser configuration:
                - export_format: Export format ("markdown", "html", "json") (default: "markdown")
                - enable_ocr: Enable OCR for scanned documents (default: False)
                - table_extraction_mode: Table extraction mode (default: "auto")
        """
        self.logger = get_logger("docling_parser")
        self.config = config
        self.progress_tracker = get_progress_tracker()
        # Ensure progress tracker is enabled
        if not self.progress_tracker.enabled:
            self.progress_tracker.enabled = True

        # Store config for lazy initialization
        self.export_format = config.get("export_format", "markdown")
        self.enable_ocr = config.get("enable_ocr", False)
        self.table_extraction_mode = config.get("table_extraction_mode", "auto")
        self._converter = None

    def parse(self, file_path: Union[str, Path], **options) -> Dict[str, Any]:
        """
        Parse document using Docling.

        Args:
            file_path: Path to document file (PDF, DOCX, PPTX, XLSX, HTML, images)
            **options: Parsing options:
                - extract_text: Whether to extract text (default: True)
                - extract_tables: Whether to extract tables (default: True)
                - extract_images: Whether to extract images (default: False)
                - export_format: Export format ("markdown", "html", "json") (default: from config)
                - pages: Specific page numbers to parse (None = all pages) - PDF only

        Returns:
            dict: Parsed document data matching Semantica format
        """
        file_path = Path(file_path)

        # Get pipeline_id from options if provided
        pipeline_id = options.get("pipeline_id", None)
        
        # Track document parsing
        tracking_id = self.progress_tracker.start_tracking(
            file=str(file_path),
            module="parse",
            submodule="DoclingParser",
            message=f"Docling: {file_path.name}",
            pipeline_id=pipeline_id,
        )

        try:
            if not file_path.exists():
                raise ValidationError(f"Document file not found: {file_path}")

            # Check if docling is available and initialize converter lazily
            if not DOCLING_AVAILABLE:
                if DOCLING_IMPORT_ERROR:
                    raise ImportError(DOCLING_IMPORT_ERROR)
                else:
                    raise ImportError("Docling is not installed")

            # Stage 1: Initialization (0-10%)
            self.progress_tracker.update_progress(
                tracking_id,
                processed=1,
                total=10,
                message="Initializing Docling converter..."
            )

            # Lazy initialization of converter
            if self._converter is None:
                self._converter = DocumentConverter(
                    format_options={
                        "markdown": {"table_format": self.table_extraction_mode},
                    },
                )

            # Determine export format
            export_format = options.get("export_format", self.export_format)

            # Stage 2: Document conversion (10-70%) - This is the longest step
            # Note: This is a blocking operation, but we'll update progress after it completes
            # Emphasize Docling as core dependency in message
            self.progress_tracker.update_progress(
                tracking_id,
                processed=2,
                total=10,
                message=f"Converting document with Docling (this may take a while for large PDFs)..."
            )

            # Convert document using Docling (blocking operation)
            # Store start time for ETA calculation
            conversion_start = time.time()
            result = self._converter.convert(str(file_path))
            conversion_elapsed = time.time() - conversion_start
            
            # Stage 3: Document conversion complete (70%)
            # Update progress immediately after conversion completes with timing info
            self.progress_tracker.update_progress(
                tracking_id,
                processed=7,
                total=10,
                message=f"Document conversion complete ({conversion_elapsed:.1f}s), extracting content..."
            )

            # Extract content based on export format
            extract_text = options.get("extract_text", True)
            extract_tables = options.get("extract_tables", True)
            extract_images = options.get("extract_images", False)

            # Stage 4: Text extraction (70-80%)
            if extract_text:
                self.progress_tracker.update_progress(
                    tracking_id,
                    processed=8,
                    total=10,
                    message=f"Extracting text content ({export_format} format)..."
                )

            # Get document content
            if export_format == "markdown":
                full_text = result.document.export_to_markdown()
            elif export_format == "html":
                full_text = result.document.export_to_html()
            elif export_format == "json":
                # JSON export returns structured data
                doc_dict = result.document.export_to_dict()
                full_text = self._extract_text_from_dict(doc_dict)
            else:
                full_text = result.document.export_to_markdown()

            # Stage 5: Metadata extraction (80-85%)
            self.progress_tracker.update_progress(
                tracking_id,
                processed=8,
                total=10,
                message="Extracting document metadata..."
            )
            metadata = self._extract_metadata(result, file_path)

            # Stage 6: Table extraction (85-95%)
            tables = []
            if extract_tables:
                self.progress_tracker.update_progress(
                    tracking_id,
                    processed=9,
                    total=10,
                    message="Extracting tables from document..."
                )
                tables = self._extract_tables(result, export_format)

            # Stage 7: Page extraction (95-98%)
            self.progress_tracker.update_progress(
                tracking_id,
                processed=9,
                total=10,
                message="Extracting page structure..."
            )
            pages = self._extract_pages(result, options)

            # Stage 8: Image extraction (98-100%)
            images = []
            if extract_images:
                self.progress_tracker.update_progress(
                    tracking_id,
                    processed=9,
                    total=10,
                    message="Extracting images from document..."
                )
                images = self._extract_images(result)

            # Prepare extraction counts and metadata
            extraction_counts = {
                "tables": len(tables),
                "images": len(images),
                "pages": len(pages) or metadata.page_count or 0,
            }
            
            # Build completion message with Docling emphasis
            count_parts = []
            if extraction_counts["tables"] > 0:
                count_parts.append(f"{extraction_counts['tables']} tables")
            if extraction_counts["images"] > 0:
                count_parts.append(f"{extraction_counts['images']} images")
            if extraction_counts["pages"] > 0:
                count_parts.append(f"{extraction_counts['pages']} pages")
            
            if count_parts:
                completion_message = f"Parsed document (Docling): {', '.join(count_parts)} extracted"
            else:
                completion_message = f"Parsed document (Docling): 0 tables, 0 images, {extraction_counts['pages']} pages extracted"
            
            # Store metadata with extraction counts and core dependency
            metadata_dict = {
                "extraction_counts": extraction_counts,
                "core_dependency": "docling",
            }
            
            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=completion_message,
                metadata=metadata_dict,
            )

            return {
                "metadata": metadata.__dict__,
                "pages": pages,
                "full_text": full_text if extract_text else "",
                "tables": tables,
                "images": images,
                "total_pages": metadata.page_count,
                "export_format": export_format,
            }

        except (ImportError, OSError):
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message="Docling not installed"
            )
            raise
        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            self.logger.error(f"Failed to parse document with Docling {file_path}: {e}")
            raise ProcessingError(f"Failed to parse document with Docling: {e}")

    def extract_text(
        self, file_path: Union[str, Path], export_format: str = "markdown"
    ) -> str:
        """
        Extract text from document.

        Args:
            file_path: Path to document file
            export_format: Export format ("markdown", "html", "json")

        Returns:
            str: Extracted text
        """
        result = self.parse(
            file_path,
            extract_tables=False,
            extract_images=False,
            export_format=export_format,
        )
        return result["full_text"]

    def extract_tables(
        self, file_path: Union[str, Path], export_format: str = "markdown"
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from document.

        Args:
            file_path: Path to document file
            export_format: Export format for table extraction

        Returns:
            list: Extracted tables
        """
        result = self.parse(
            file_path,
            extract_text=False,
            extract_images=False,
            export_format=export_format,
        )
        return result["tables"]

    def _extract_text_from_dict(self, doc_dict: Dict[str, Any]) -> str:
        """Extract text content from Docling dict structure."""
        text_parts = []

        def extract_from_item(item: Dict[str, Any]):
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "paragraph":
                    if "content" in item:
                        for content_item in item["content"]:
                            extract_from_item(content_item)
                elif "content" in item:
                    for content_item in item["content"]:
                        extract_from_item(content_item)

        if isinstance(doc_dict, dict) and "content" in doc_dict:
            for item in doc_dict["content"]:
                extract_from_item(item)

        return "\n\n".join(text_parts)

    def _extract_tables(
        self, result: Any, export_format: str
    ) -> List[Dict[str, Any]]:
        """Extract tables from Docling result."""
        tables = []

        try:
            # Try to get tables from document structure
            doc_dict = result.document.export_to_dict()

            def find_tables(item: Dict[str, Any], page_num: int = 1):
                if isinstance(item, dict):
                    if item.get("type") == "table":
                        table_data = self._convert_table_to_dict(item)
                        table_data["page_number"] = page_num
                        tables.append(table_data)
                    elif "content" in item:
                        # Check if this is a page
                        if item.get("type") == "page":
                            page_num = item.get("page", page_num)
                        for content_item in item.get("content", []):
                            find_tables(content_item, page_num)

            if isinstance(doc_dict, dict) and "content" in doc_dict:
                for item in doc_dict["content"]:
                    find_tables(item)

        except Exception as e:
            self.logger.warning(f"Could not extract tables from Docling result: {e}")

        return tables

    def _convert_table_to_dict(self, table_item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Docling table structure to Semantica format."""
        table_data = {
            "rows": [],
            "row_count": 0,
            "col_count": 0,
            "data": [],
        }

        try:
            # Extract table rows
            if "content" in table_item:
                for row_item in table_item["content"]:
                    if row_item.get("type") == "table-row":
                        row_data = []
                        if "content" in row_item:
                            for cell_item in row_item["content"]:
                                if cell_item.get("type") == "table-cell":
                                    cell_text = ""
                                    if "content" in cell_item:
                                        for cell_content in cell_item["content"]:
                                            if isinstance(cell_content, dict):
                                                cell_text += cell_content.get("text", "")
                                            elif isinstance(cell_content, str):
                                                cell_text += cell_content
                                    row_data.append(cell_text.strip())
                        if row_data:
                            table_data["rows"].append(row_data)
                            table_data["data"].append(row_data)

            if table_data["rows"]:
                table_data["row_count"] = len(table_data["rows"])
                table_data["col_count"] = (
                    max(len(row) for row in table_data["rows"]) if table_data["rows"] else 0
                )

        except Exception as e:
            self.logger.warning(f"Error converting table: {e}")

        return table_data

    def _extract_pages(self, result: Any, options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pages from Docling result."""
        pages = []

        try:
            doc_dict = result.document.export_to_dict()

            def extract_page(page_item: Dict[str, Any], page_num: int):
                page_text = ""
                page_tables = []

                if "content" in page_item:
                    for content_item in page_item["content"]:
                        if content_item.get("type") == "text":
                            page_text += content_item.get("text", "") + "\n"
                        elif content_item.get("type") == "table":
                            table_data = self._convert_table_to_dict(content_item)
                            table_data["page_number"] = page_num
                            page_tables.append(table_data)

                pages.append({
                    "page_number": page_num,
                    "text": page_text.strip(),
                    "width": page_item.get("width", 0),
                    "height": page_item.get("height", 0),
                    "tables": [t for t in page_tables],
                    "images": [],
                })

            # Find pages in document structure
            if isinstance(doc_dict, dict) and "content" in doc_dict:
                page_num = 1
                for item in doc_dict["content"]:
                    if item.get("type") == "page":
                        extract_page(item, page_num)
                        page_num += 1
                    elif "pages" in item:
                        # Handle paginated content
                        for page_item in item.get("pages", []):
                            extract_page(page_item, page_num)
                            page_num += 1

            # If no pages found, create a single page from full content
            if not pages:
                full_text = result.document.export_to_markdown()
                pages.append({
                    "page_number": 1,
                    "text": full_text,
                    "width": 0,
                    "height": 0,
                    "tables": [],
                    "images": [],
                })

        except Exception as e:
            self.logger.warning(f"Could not extract pages from Docling result: {e}")
            # Fallback: create single page
            try:
                full_text = result.document.export_to_markdown()
                pages.append({
                    "page_number": 1,
                    "text": full_text,
                    "width": 0,
                    "height": 0,
                    "tables": [],
                    "images": [],
                })
            except:
                pass

        return pages

    def _extract_images(self, result: Any) -> List[Dict[str, Any]]:
        """Extract images from Docling result."""
        images = []

        try:
            doc_dict = result.document.export_to_dict()

            def find_images(item: Dict[str, Any], page_num: int = 1):
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        img_data = {
                            "page_number": page_num,
                            "x0": item.get("bbox", {}).get("x0", 0) if "bbox" in item else 0,
                            "y0": item.get("bbox", {}).get("y0", 0) if "bbox" in item else 0,
                            "x1": item.get("bbox", {}).get("x1", 0) if "bbox" in item else 0,
                            "y1": item.get("bbox", {}).get("y1", 0) if "bbox" in item else 0,
                            "width": item.get("width", 0),
                            "height": item.get("height", 0),
                        }
                        images.append(img_data)
                    elif "content" in item:
                        if item.get("type") == "page":
                            page_num = item.get("page", page_num)
                        for content_item in item.get("content", []):
                            find_images(content_item, page_num)

            if isinstance(doc_dict, dict) and "content" in doc_dict:
                for item in doc_dict["content"]:
                    find_images(item)

        except Exception as e:
            self.logger.warning(f"Could not extract images from Docling result: {e}")

        return images

    def _extract_metadata(self, result: Any, file_path: Path) -> DoclingMetadata:
        """Extract metadata from Docling result."""
        metadata = DoclingMetadata()

        try:
            # Try to get metadata from document
            doc_dict = result.document.export_to_dict()

            # Extract format
            metadata.format = file_path.suffix.lower()

            # Try to extract page count
            if isinstance(doc_dict, dict) and "content" in doc_dict:
                page_count = 0
                for item in doc_dict["content"]:
                    if item.get("type") == "page":
                        page_count += 1
                metadata.page_count = page_count if page_count > 0 else 1

            # Docling may not provide all metadata fields directly
            # These would need to be extracted from the original document if available

        except Exception as e:
            self.logger.warning(f"Could not extract metadata from Docling result: {e}")
            metadata.page_count = 1
            metadata.format = file_path.suffix.lower()

        return metadata

