"""
File Ingestion Module

Handles ingestion from local filesystem and cloud storage providers.

Key Features:
    - Local file system scanning
    - Cloud storage integration (S3, GCS, Azure)
    - File type detection
    - Batch processing
    - Progress tracking

Main Classes:
    - FileIngestor: Main file ingestion class
    - CloudStorageIngestor: Cloud storage specific handling
    - FileTypeDetector: File type identification
"""

import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.constants import (
    SUPPORTED_DOCUMENT_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    FILE_SIZE_LIMITS
)


@dataclass
class FileObject:
    """File object representation."""
    
    path: str
    name: str
    size: int
    file_type: str
    mime_type: Optional[str] = None
    content: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=datetime.now)


class FileTypeDetector:
    """
    File type detection and validation.
    
    Identifies file types using multiple methods including
    extension, MIME type, and content analysis.
    """
    
    def __init__(self):
        """Initialize file type detector."""
        self.logger = get_logger("file_type_detector")
        self.supported_formats = (
            SUPPORTED_DOCUMENT_FORMATS +
            SUPPORTED_IMAGE_FORMATS +
            SUPPORTED_AUDIO_FORMATS +
            SUPPORTED_VIDEO_FORMATS
        )
        # Initialize MIME types database
        mimetypes.init()
    
    def detect_type(self, file_path: Union[str, Path], content: Optional[bytes] = None) -> str:
        """
        Detect file type using multiple methods.
        
        Args:
            file_path: Path to file
            content: Optional file content
            
        Returns:
            str: Detected file type
        """
        file_path = Path(file_path)
        
        # Method 1: Check file extension
        extension = file_path.suffix.lstrip('.').lower()
        if extension:
            return extension
        
        # Method 2: Check MIME type
        if file_path.exists():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                # Map MIME type to extension
                ext = mimetypes.guess_extension(mime_type)
                if ext:
                    return ext.lstrip('.').lower()
        
        # Method 3: Check magic numbers if content provided
        if content:
            file_type = self._detect_by_magic_numbers(content)
            if file_type:
                return file_type
        
        return "unknown"
    
    def _detect_by_magic_numbers(self, content: bytes) -> Optional[str]:
        """Detect file type by magic numbers (file signatures)."""
        if len(content) < 4:
            return None
        
        # Common magic numbers
        magic_numbers = {
            b'\x25\x50\x44\x46': 'pdf',  # PDF
            b'\x50\x4B\x03\x04': 'zip',  # ZIP/DOCX/XLSX/PPTX
            b'\x89\x50\x4E\x47': 'png',  # PNG
            b'\xFF\xD8\xFF': 'jpg',      # JPEG
            b'\x47\x49\x46\x38': 'gif',  # GIF
            b'%PDF': 'pdf',               # PDF (text)
        }
        
        for magic, file_type in magic_numbers.items():
            if content.startswith(magic):
                return file_type
        
        return None
    
    def is_supported(self, file_type: str) -> bool:
        """
        Check if file type is supported.
        
        Args:
            file_type: File type to check
            
        Returns:
            bool: Whether type is supported
        """
        return file_type.lower() in self.supported_formats


class CloudStorageIngestor:
    """
    Cloud storage specific ingestion handler.
    
    Handles authentication, object listing, and downloading
    from various cloud storage providers.
    """
    
    def __init__(self, provider: str, **config):
        """
        Initialize cloud storage ingestor.
        
        Args:
            provider: Cloud provider name (s3, gcs, azure)
            **config: Provider-specific configuration
        """
        self.logger = get_logger("cloud_storage_ingestor")
        self.provider = provider.lower()
        self.config = config
        self._client = None
        
        # Initialize provider-specific client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize cloud storage client based on provider."""
        if self.provider == "s3":
            import boto3
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.config.get('access_key_id'),
                aws_secret_access_key=self.config.get('secret_access_key'),
                region_name=self.config.get('region', 'us-east-1')
            )
        elif self.provider == "gcs":
            from google.cloud import storage
            self._client = storage.Client()
        elif self.provider == "azure":
            from azure.storage.blob import BlobServiceClient
            self._client = BlobServiceClient.from_connection_string(
                self.config.get('connection_string')
            )
        else:
            raise ValueError(f"Unsupported cloud provider: {self.provider}")
    
    def list_objects(self, bucket: str, prefix: str = "", **filters) -> List[Dict[str, Any]]:
        """
        List objects in cloud storage.
        
        Args:
            bucket: Storage bucket name
            prefix: Object prefix filter
            **filters: Additional filters
            
        Returns:
            list: List of object metadata
        """
        try:
            objects = []
            
            if self.provider == "s3":
                paginator = self._client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
                
                for page in pages:
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            objects.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'last_modified': obj['LastModified'],
                                'etag': obj['ETag']
                            })
            elif self.provider == "gcs":
                bucket_obj = self._client.bucket(bucket)
                blobs = bucket_obj.list_blobs(prefix=prefix)
                
                for blob in blobs:
                    objects.append({
                        'key': blob.name,
                        'size': blob.size,
                        'last_modified': blob.updated,
                        'etag': blob.etag
                    })
            elif self.provider == "azure":
                container_client = self._client.get_container_client(bucket)
                blobs = container_client.list_blobs(name_starts_with=prefix)
                
                for blob in blobs:
                    objects.append({
                        'key': blob.name,
                        'size': blob.size,
                        'last_modified': blob.last_modified,
                        'etag': blob.etag
                    })
            
            # Apply additional filters
            if filters:
                objects = self._apply_filters(objects, filters)
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to list objects from {self.provider}: {e}")
            raise ProcessingError(f"Failed to list objects: {e}")
    
    def _apply_filters(self, objects: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to object list."""
        filtered = objects
        
        if 'min_size' in filters:
            filtered = [obj for obj in filtered if obj['size'] >= filters['min_size']]
        
        if 'max_size' in filters:
            filtered = [obj for obj in filtered if obj['size'] <= filters['max_size']]
        
        if 'extensions' in filters:
            exts = filters['extensions']
            filtered = [obj for obj in filtered if Path(obj['key']).suffix.lstrip('.') in exts]
        
        return filtered
    
    def download_object(self, bucket: str, key: str, **options) -> bytes:
        """
        Download object from cloud storage.
        
        Args:
            bucket: Storage bucket name
            key: Object key
            **options: Download options
            
        Returns:
            bytes: Object content
        """
        try:
            if self.provider == "s3":
                response = self._client.get_object(Bucket=bucket, Key=key)
                content = response['Body'].read()
            elif self.provider == "gcs":
                bucket_obj = self._client.bucket(bucket)
                blob = bucket_obj.blob(key)
                content = blob.download_as_bytes()
            elif self.provider == "azure":
                container_client = self._client.get_container_client(bucket)
                blob_client = container_client.get_blob_client(key)
                content = blob_client.download_blob().readall()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            return content
            
        except Exception as e:
            self.logger.error(f"Failed to download object {key} from {self.provider}: {e}")
            raise ProcessingError(f"Failed to download object: {e}")


class FileIngestor:
    """
    File system and cloud storage ingestion handler.
    
    Supports local files, cloud storage (S3, GCS, Azure), and
    various file formats with automatic type detection.
    
    Attributes:
        config: Ingestion configuration
        supported_formats: List of supported file formats
        cloud_providers: Available cloud storage providers
        
    Methods:
        ingest_directory(): Ingest entire directory
        ingest_file(): Ingest single file
        ingest_cloud(): Ingest from cloud storage
        scan_directory(): Scan directory for files
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize file ingestor.
        
        Args:
            config: Ingestion configuration
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("file_ingestor")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Initialize file type detector
        self.type_detector = FileTypeDetector()
        self.supported_formats = self.type_detector.supported_formats
        
        # Cloud providers (lazy initialization)
        self._cloud_providers: Dict[str, CloudStorageIngestor] = {}
        
        # Progress tracking
        self._progress_callback = None
    
    def ingest_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        **filters
    ) -> List[FileObject]:
        """
        Ingest all files from a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to scan subdirectories
            **filters: File filtering criteria
            
        Returns:
            list: List of ingested file objects
        """
        directory_path = Path(directory_path)
        
        # Validate directory path
        if not directory_path.exists():
            raise ValidationError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValidationError(f"Path is not a directory: {directory_path}")
        
        # Scan directory for files
        files = self.scan_directory(directory_path, recursive=recursive, **filters)
        
        # Process each file
        file_objects = []
        total_files = len(files)
        
        for idx, file_info in enumerate(files, 1):
            try:
                file_obj = self.ingest_file(file_info['path'], **file_info)
                file_objects.append(file_obj)
                
                # Track progress
                if self._progress_callback:
                    self._progress_callback(idx, total_files, file_obj)
                
                self.logger.debug(f"Ingested file {idx}/{total_files}: {file_info['path']}")
                
            except Exception as e:
                self.logger.error(f"Failed to ingest file {file_info['path']}: {e}")
                if self.config.get('fail_fast', False):
                    raise ProcessingError(f"Failed to ingest file: {e}")
        
        return file_objects
    
    def ingest_file(self, file_path: Union[str, Path], **options) -> FileObject:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to file
            **options: Processing options
            
        Returns:
            FileObject: Ingested file object
        """
        file_path = Path(file_path)
        
        # Validate file path
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Check file size limits
        file_size = file_path.stat().st_size
        max_size = FILE_SIZE_LIMITS.get('MAX_DOCUMENT_SIZE', 104857600)
        if file_size > max_size:
            raise ValidationError(f"File size {file_size} exceeds maximum {max_size}")
        
        # Detect file type
        file_type = self.type_detector.detect_type(file_path)
        
        if not self.type_detector.is_supported(file_type):
            self.logger.warning(f"Unsupported file type: {file_type} for {file_path}")
        
        # Read file content if requested
        content = None
        if options.get('read_content', True):
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
            except Exception as e:
                self.logger.error(f"Failed to read file content: {e}")
                raise ProcessingError(f"Failed to read file: {e}")
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Create file object
        file_obj = FileObject(
            path=str(file_path),
            name=file_path.name,
            size=file_size,
            file_type=file_type,
            mime_type=mime_type,
            content=content,
            metadata={
                'extension': file_path.suffix,
                'parent': str(file_path.parent),
                'is_supported': self.type_detector.is_supported(file_type),
                **options
            }
        )
        
        return file_obj
    
    def ingest_cloud(
        self,
        provider: str,
        bucket: str,
        prefix: str = "",
        **config
    ) -> List[FileObject]:
        """
        Ingest files from cloud storage.
        
        Args:
            provider: Cloud provider (s3, gcs, azure)
            bucket: Storage bucket name
            prefix: Object prefix filter
            **config: Cloud provider configuration
            
        Returns:
            list: List of ingested file objects
        """
        # Initialize cloud client if not already done
        if provider not in self._cloud_providers:
            self._cloud_providers[provider] = CloudStorageIngestor(provider, **config)
        
        cloud_ingestor = self._cloud_providers[provider]
        
        # List objects in bucket
        objects = cloud_ingestor.list_objects(bucket, prefix=prefix)
        
        # Process each object
        file_objects = []
        
        for obj_info in objects:
            try:
                # Download object
                content = cloud_ingestor.download_object(bucket, obj_info['key'])
                
                # Detect file type
                file_type = self.type_detector._detect_by_magic_numbers(content) or \
                          self.type_detector.detect_type(obj_info['key'])
                
                # Create file object
                file_obj = FileObject(
                    path=obj_info['key'],
                    name=Path(obj_info['key']).name,
                    size=obj_info['size'],
                    file_type=file_type,
                    content=content,
                    metadata={
                        'provider': provider,
                        'bucket': bucket,
                        'etag': obj_info.get('etag'),
                        'last_modified': str(obj_info.get('last_modified')),
                    }
                )
                
                file_objects.append(file_obj)
                
            except Exception as e:
                self.logger.error(f"Failed to ingest object {obj_info['key']}: {e}")
                if self.config.get('fail_fast', False):
                    raise ProcessingError(f"Failed to ingest object: {e}")
        
        return file_objects
    
    def scan_directory(
        self,
        directory_path: Union[str, Path],
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Scan directory and return file information without processing.
        
        Args:
            directory_path: Path to directory
            **filters: File filtering criteria:
                - recursive: Whether to scan subdirectories (default: True)
                - extensions: List of allowed extensions
                - min_size: Minimum file size
                - max_size: Maximum file size
                - pattern: Filename pattern (glob)
                
        Returns:
            list: List of file metadata
        """
        directory_path = Path(directory_path)
        recursive = filters.pop('recursive', True)
        
        # Walk directory tree
        files = []
        
        if recursive:
            for file_path in directory_path.rglob('*'):
                if file_path.is_file():
                    files.append(self._get_file_info(file_path))
        else:
            for file_path in directory_path.iterdir():
                if file_path.is_file():
                    files.append(self._get_file_info(file_path))
        
        # Apply filters
        if filters:
            files = self._apply_file_filters(files, filters)
        
        return files
    
    def _get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get file information."""
        stat = file_path.stat()
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size': stat.st_size,
            'extension': file_path.suffix.lstrip('.'),
            'modified': stat.st_mtime,
        }
    
    def _apply_file_filters(self, files: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to file list."""
        filtered = files
        
        if 'extensions' in filters:
            exts = [ext.lstrip('.') for ext in filters['extensions']]
            filtered = [f for f in filtered if f['extension'] in exts]
        
        if 'min_size' in filters:
            filtered = [f for f in filtered if f['size'] >= filters['min_size']]
        
        if 'max_size' in filters:
            filtered = [f for f in filtered if f['size'] <= filters['max_size']]
        
        if 'pattern' in filters:
            import fnmatch
            pattern = filters['pattern']
            filtered = [f for f in filtered if fnmatch.fnmatch(f['name'], pattern)]
        
        return filtered
    
    def set_progress_callback(self, callback):
        """Set progress tracking callback."""
        self._progress_callback = callback
