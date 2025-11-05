"""
Encoding handling utilities for Semantica framework.

This module provides encoding detection, conversion, and handling
for UTF-8 conversion and BOM handling.
"""

import chardet
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


class EncodingHandler:
    """Encoding handling utilities."""
    
    def __init__(self, **config):
        """
        Initialize encoding handler.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("encoding_handler")
        self.config = config
        self.default_encoding = config.get("default_encoding", "utf-8")
        self.fallback_encodings = config.get("fallback_encodings", ["latin-1", "cp1252", "iso-8859-1"])
    
    def detect(self, data: Union[str, bytes], **options) -> Tuple[str, float]:
        """
        Detect encoding of data.
        
        Args:
            data: Input data (string or bytes)
            **options: Detection options
            
        Returns:
            tuple: (encoding, confidence)
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if not data:
            return (self.default_encoding, 0.0)
        
        try:
            result = chardet.detect(data)
            if result:
                encoding = result.get('encoding', self.default_encoding)
                confidence = result.get('confidence', 0.0)
                return (encoding, confidence)
            else:
                return (self.default_encoding, 0.0)
        except Exception as e:
            self.logger.warning(f"Failed to detect encoding: {e}")
            return (self.default_encoding, 0.0)
    
    def detect_file(self, file_path: Union[str, Path], **options) -> Tuple[str, float]:
        """
        Detect encoding of file.
        
        Args:
            file_path: Path to file
            **options: Detection options
            
        Returns:
            tuple: (encoding, confidence)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Read sample bytes for detection
        sample_size = options.get("sample_size", 10000)
        
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(sample_size)
            
            return self.detect(sample, **options)
        except Exception as e:
            self.logger.error(f"Failed to detect file encoding: {e}")
            raise ProcessingError(f"Failed to detect file encoding: {e}")
    
    def convert_to_utf8(self, data: Union[str, bytes], source_encoding: Optional[str] = None, **options) -> str:
        """
        Convert data to UTF-8.
        
        Args:
            data: Input data
            source_encoding: Source encoding (auto-detected if None)
            **options: Conversion options
            
        Returns:
            str: UTF-8 encoded string
        """
        if isinstance(data, str):
            # Already a string, just ensure it's valid UTF-8
            return data.encode('utf-8', errors='replace').decode('utf-8')
        
        if source_encoding is None:
            source_encoding, _ = self.detect(data, **options)
        
        # Try to decode with detected encoding
        try:
            return data.decode(source_encoding, errors='replace')
        except (UnicodeDecodeError, LookupError):
            # Try fallback encodings
            for encoding in self.fallback_encodings:
                try:
                    return data.decode(encoding, errors='replace')
                except (UnicodeDecodeError, LookupError):
                    continue
            
            # Final fallback: decode with errors replaced
            self.logger.warning(f"Failed to decode with detected encoding, using errors='replace'")
            return data.decode(self.default_encoding, errors='replace')
    
    def convert_file_to_utf8(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **options
    ) -> str:
        """
        Convert file to UTF-8.
        
        Args:
            file_path: Path to input file
            output_path: Path to output file (overwrites if None)
            **options: Conversion options
            
        Returns:
            str: Converted content as string
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Detect encoding
        source_encoding, _ = self.detect_file(file_path, **options)
        
        # Read file with detected encoding
        try:
            with open(file_path, 'r', encoding=source_encoding, errors='replace') as f:
                content = f.read()
        except Exception as e:
            # Try fallback encodings
            content = None
            for encoding in self.fallback_encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()
                    break
                except Exception:
                    continue
            
            if content is None:
                self.logger.error(f"Failed to read file with any encoding: {e}")
                raise ProcessingError(f"Failed to read file: {e}")
        
        # Ensure content is UTF-8
        utf8_content = content.encode('utf-8', errors='replace').decode('utf-8')
        
        # Write to output file if specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(utf8_content)
        
        return utf8_content
    
    def remove_bom(self, data: Union[str, bytes]) -> Union[str, bytes]:
        """
        Remove BOM (Byte Order Mark) from data.
        
        Args:
            data: Input data
            
        Returns:
            Data without BOM
        """
        if isinstance(data, bytes):
            # Remove UTF-8 BOM
            if data.startswith(b'\xef\xbb\xbf'):
                return data[3:]
            # Remove UTF-16 BOMs
            elif data.startswith(b'\xff\xfe'):
                return data[2:]
            elif data.startswith(b'\xfe\xff'):
                return data[2:]
            else:
                return data
        else:
            # String data - remove if present
            if data.startswith('\ufeff'):
                return data[1:]
            else:
                return data
    
    def handle_encoding_error(self, data: bytes, **options) -> str:
        """
        Handle encoding errors gracefully.
        
        Args:
            data: Input bytes data
            **options: Error handling options
            
        Returns:
            str: Decoded string with errors handled
        """
        error_strategy = options.get("error_strategy", "replace")
        
        # Try detected encoding first
        encoding, _ = self.detect(data, **options)
        
        try:
            return data.decode(encoding, errors=error_strategy)
        except Exception:
            # Try fallback encodings
            for fallback_encoding in self.fallback_encodings:
                try:
                    return data.decode(fallback_encoding, errors=error_strategy)
                except Exception:
                    continue
            
            # Final fallback
            return data.decode(self.default_encoding, errors=error_strategy)
    
    def validate_encoding(self, data: Union[str, bytes], encoding: str, **options) -> bool:
        """
        Validate that data can be decoded with given encoding.
        
        Args:
            data: Input data
            encoding: Encoding to validate
            **options: Validation options
            
        Returns:
            bool: True if encoding is valid for data
        """
        if isinstance(data, str):
            try:
                data.encode(encoding)
                return True
            except Exception:
                return False
        else:
            try:
                data.decode(encoding, errors='strict')
                return True
            except Exception:
                return False
