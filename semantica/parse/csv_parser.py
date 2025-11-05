"""
CSV document parser for Semantica framework.

This module handles CSV file parsing for structured data extraction.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


@dataclass
class CSVData:
    """CSV data representation."""
    
    headers: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class CSVParser:
    """CSV document parser."""
    
    def __init__(self, **config):
        """
        Initialize CSV parser.
        
        Args:
            **config: Parser configuration
        """
        self.logger = get_logger("csv_parser")
        self.config = config
    
    def parse(
        self,
        file_path: Union[str, Path],
        delimiter: str = ',',
        **options
    ) -> CSVData:
        """
        Parse CSV file.
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter (default: ',')
            **options: Parsing options:
                - has_header: Whether CSV has header row (default: True)
                - encoding: File encoding (default: 'utf-8')
                - skip_rows: Number of rows to skip
                - max_rows: Maximum number of rows to read
                
        Returns:
            CSVData: Parsed CSV data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"CSV file not found: {file_path}")
        
        has_header = options.get("has_header", True)
        encoding = options.get("encoding", "utf-8")
        skip_rows = options.get("skip_rows", 0)
        max_rows = options.get("max_rows")
        
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore', newline='') as f:
                # Detect delimiter if not specified
                sample = f.read(1024)
                f.seek(0)
                
                if delimiter is None:
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(f, delimiter=delimiter) if has_header else csv.reader(f, delimiter=delimiter)
                
                headers = []
                rows = []
                
                # Get headers
                if has_header:
                    headers = reader.fieldnames or []
                else:
                    # Read first row as headers
                    try:
                        first_row = next(reader)
                        headers = [f"Column_{i+1}" for i in range(len(first_row))]
                        rows.append(dict(zip(headers, first_row)))
                    except StopIteration:
                        pass
                
                # Skip rows if requested
                for _ in range(skip_rows):
                    try:
                        next(reader)
                    except StopIteration:
                        break
                
                # Read rows
                for idx, row in enumerate(reader):
                    if max_rows and idx >= max_rows:
                        break
                    
                    if has_header:
                        rows.append(row)
                    else:
                        rows.append(dict(zip(headers, row if isinstance(row, list) else [row])))
            
            metadata = {
                "file_path": str(file_path),
                "delimiter": delimiter,
                "encoding": encoding,
                "has_header": has_header
            }
            
            return CSVData(
                headers=headers,
                rows=rows,
                row_count=len(rows),
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse CSV {file_path}: {e}")
            raise ProcessingError(f"Failed to parse CSV: {e}")
    
    def parse_to_dict(self, file_path: Union[str, Path], **options) -> List[Dict[str, Any]]:
        """
        Parse CSV to list of dictionaries.
        
        Args:
            file_path: Path to CSV file
            **options: Parsing options
            
        Returns:
            list: List of row dictionaries
        """
        csv_data = self.parse(file_path, **options)
        return csv_data.rows
    
    def parse_to_list(self, file_path: Union[str, Path], **options) -> List[List[Any]]:
        """
        Parse CSV to list of lists.
        
        Args:
            file_path: Path to CSV file
            **options: Parsing options
            
        Returns:
            list: List of row lists
        """
        csv_data = self.parse(file_path, **options)
        return [[row.get(header, '') for header in csv_data.headers] for row in csv_data.rows]
