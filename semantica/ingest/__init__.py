"""
Data Ingestion Module

This module provides comprehensive data ingestion capabilities from various sources.

Exports:
    - FileIngestor: Local and cloud file processing
    - WebIngestor: Web scraping and crawling
    - FeedIngestor: RSS/Atom feed processing
    - StreamIngestor: Real-time stream processing
    - RepoIngestor: Git repository processing
    - EmailIngestor: Email protocol handling
    - DBIngestor: Database export handling
"""

from .file_ingestor import FileIngestor, FileObject, FileTypeDetector, CloudStorageIngestor
from .web_ingestor import WebIngestor, WebContent, RateLimiter, RobotsChecker, ContentExtractor, SitemapCrawler
from .feed_ingestor import FeedIngestor, FeedItem, FeedData, FeedParser, FeedMonitor
from .stream_ingestor import (
    StreamIngestor,
    StreamMessage,
    StreamProcessor,
    KafkaProcessor,
    RabbitMQProcessor,
    KinesisProcessor,
    PulsarProcessor,
    StreamMonitor,
)
from .repo_ingestor import RepoIngestor, CodeFile, CommitInfo, CodeExtractor, GitAnalyzer
from .email_ingestor import EmailIngestor, EmailData, AttachmentProcessor, EmailParser as EmailIngestorParser
from .db_ingestor import DBIngestor, TableData, DatabaseConnector, DataExporter

__all__ = [
    # File ingestion
    "FileIngestor",
    "FileObject",
    "FileTypeDetector",
    "CloudStorageIngestor",
    # Web ingestion
    "WebIngestor",
    "WebContent",
    "RateLimiter",
    "RobotsChecker",
    "ContentExtractor",
    "SitemapCrawler",
    # Feed ingestion
    "FeedIngestor",
    "FeedItem",
    "FeedData",
    "FeedParser",
    "FeedMonitor",
    # Stream ingestion
    "StreamIngestor",
    "StreamMessage",
    "StreamProcessor",
    "KafkaProcessor",
    "RabbitMQProcessor",
    "KinesisProcessor",
    "PulsarProcessor",
    "StreamMonitor",
    # Repository ingestion
    "RepoIngestor",
    "CodeFile",
    "CommitInfo",
    "CodeExtractor",
    "GitAnalyzer",
    # Email ingestion
    "EmailIngestor",
    "EmailData",
    "AttachmentProcessor",
    "EmailIngestorParser",
    # Database ingestion
    "DBIngestor",
    "TableData",
    "DatabaseConnector",
    "DataExporter",
]
