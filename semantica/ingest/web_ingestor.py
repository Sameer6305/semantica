"""
Web Ingestion Module

Handles web scraping, crawling, and content extraction from websites.

Key Features:
    - HTTP/HTTPS content fetching
    - Sitemap crawling
    - RSS/Atom feed processing
    - JavaScript rendering
    - Rate limiting and politeness
    - Content extraction

Main Classes:
    - WebIngestor: Main web ingestion class
    - SitemapCrawler: Sitemap-based crawling
    - ContentExtractor: Web content extraction
"""

import time
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


@dataclass
class WebContent:
    """Web content representation."""
    
    url: str
    title: str = ""
    text: str = ""
    html: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.now)
    status_code: Optional[int] = None


class RateLimiter:
    """Simple rate limiter for web requests."""
    
    def __init__(self, delay: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            delay: Delay between requests in seconds
        """
        self.delay = delay
        self.last_request_time: float = 0.0
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self.last_request_time = time.time()


class RobotsChecker:
    """Robots.txt compliance checker."""
    
    def __init__(self, user_agent: str = "SemanticaBot/1.0"):
        """
        Initialize robots checker.
        
        Args:
            user_agent: User agent string
        """
        self.user_agent = user_agent
        self.parsers: Dict[str, RobotFileParser] = {}
    
    def can_fetch(self, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt.
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if can fetch, False otherwise
        """
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        if domain not in self.parsers:
            # Create robots.txt URL
            robots_url = urljoin(domain, "/robots.txt")
            
            # Create parser
            parser = RobotFileParser()
            parser.set_url(robots_url)
            
            try:
                parser.read()
            except Exception:
                # If robots.txt not available, assume allowed
                pass
            
            self.parsers[domain] = parser
        
        parser = self.parsers[domain]
        return parser.can_fetch(self.user_agent, url)


class ContentExtractor:
    """
    Web content extraction and cleaning.
    
    Extracts text, metadata, and structured data from
    web pages with various extraction strategies.
    """
    
    def __init__(self, **config):
        """
        Initialize content extractor.
        
        Args:
            **config: Extraction configuration
        """
        self.logger = get_logger("content_extractor")
        self.config = config
    
    def extract_text(self, html_content: str) -> str:
        """
        Extract clean text from HTML.
        
        Args:
            html_content: HTML content
            
        Returns:
            str: Extracted text content
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style']):
            element.decompose()
        
        # Extract text
        text = soup.get_text()
        
        # Clean whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_metadata(self, html_content: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from HTML.
        
        Args:
            html_content: HTML content
            url: Source URL
            
        Returns:
            dict: Extracted metadata
        """
        metadata = {'url': url} if url else {}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Extract meta tags
        for meta_tag in soup.find_all('meta'):
            name = meta_tag.get('name') or meta_tag.get('property')
            content = meta_tag.get('content')
            if name and content:
                metadata[name.lower()] = content
        
        # Extract Open Graph data
        og_data = {}
        for meta_tag in soup.find_all('meta', property=re.compile(r'^og:')):
            property_name = meta_tag.get('property', '').replace('og:', '')
            content = meta_tag.get('content')
            if property_name and content:
                og_data[property_name] = content
        
        if og_data:
            metadata['og'] = og_data
        
        return metadata
    
    def extract_links(self, html_content: str, base_url: Optional[str] = None) -> List[str]:
        """
        Extract links from HTML.
        
        Args:
            html_content: HTML content
            base_url: Base URL for relative links
            
        Returns:
            list: List of extracted links
        """
        links = []
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']
            
            # Resolve relative URLs
            if base_url:
                link = urljoin(base_url, href)
            else:
                link = href
            
            # Filter valid links
            parsed = urlparse(link)
            if parsed.scheme in ('http', 'https'):
                links.append(link)
        
        return links


class SitemapCrawler:
    """
    Sitemap-based web crawling.
    
    Specialized crawler for processing XML sitemaps and
    extracting URLs for processing.
    """
    
    def __init__(self, **config):
        """
        Initialize sitemap crawler.
        
        Args:
            **config: Crawler configuration
        """
        self.logger = get_logger("sitemap_crawler")
        self.config = config
    
    def parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse sitemap and extract URLs.
        
        Args:
            sitemap_url: URL of sitemap
            
        Returns:
            list: List of URLs from sitemap
        """
        try:
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Namespace handling
            namespaces = {
                'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'
            }
            
            urls = []
            
            # Extract URLs from urlset
            for url_elem in root.findall('.//sitemap:url', namespaces):
                loc_elem = url_elem.find('sitemap:loc', namespaces)
                if loc_elem is not None and loc_elem.text:
                    urls.append(loc_elem.text.strip())
            
            # Also handle non-namespaced sitemaps
            if not urls:
                for url_elem in root.findall('.//url'):
                    loc_elem = url_elem.find('loc')
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text.strip())
            
            return urls
            
        except Exception as e:
            self.logger.error(f"Failed to parse sitemap {sitemap_url}: {e}")
            raise ProcessingError(f"Failed to parse sitemap: {e}")
    
    def crawl_sitemap_index(self, index_url: str) -> List[str]:
        """
        Crawl sitemap index and all referenced sitemaps.
        
        Args:
            index_url: URL of sitemap index
            
        Returns:
            list: List of all URLs from all sitemaps
        """
        try:
            response = requests.get(index_url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Namespace handling
            namespaces = {
                'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'
            }
            
            all_urls = []
            
            # Find sitemap references
            for sitemap_elem in root.findall('.//sitemap:sitemap', namespaces):
                loc_elem = sitemap_elem.find('sitemap:loc', namespaces)
                if loc_elem is not None and loc_elem.text:
                    sitemap_url = loc_elem.text.strip()
                    # Parse the referenced sitemap
                    urls = self.parse_sitemap(sitemap_url)
                    all_urls.extend(urls)
            
            # Also handle non-namespaced sitemaps
            if not all_urls:
                for sitemap_elem in root.findall('.//sitemap'):
                    loc_elem = sitemap_elem.find('loc')
                    if loc_elem is not None and loc_elem.text:
                        sitemap_url = loc_elem.text.strip()
                        urls = self.parse_sitemap(sitemap_url)
                        all_urls.extend(urls)
            
            return all_urls
            
        except Exception as e:
            self.logger.error(f"Failed to crawl sitemap index {index_url}: {e}")
            raise ProcessingError(f"Failed to crawl sitemap index: {e}")


class WebIngestor:
    """
    Web content ingestion handler.
    
    Fetches and processes content from websites with support
    for various protocols and content types.
    
    Attributes:
        session: HTTP session for requests
        rate_limiter: Rate limiting controller
        content_extractor: Content extraction engine
        robots_checker: Robots.txt compliance checker
        
    Methods:
        ingest_url(): Ingest single URL
        crawl_sitemap(): Crawl sitemap
        crawl_domain(): Crawl entire domain
        extract_content(): Extract content from page
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize web ingestor.
        
        Args:
            config: Web ingestion configuration
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("web_ingestor")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Initialize HTTP session
        self.session = requests.Session()
        
        # Configure user agent
        user_agent = self.config.get('user_agent', 'SemanticaBot/1.0')
        self.session.headers.update({'User-Agent': user_agent})
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=self.config.get('max_retries', 3),
            backoff_factor=self.config.get('backoff_factor', 1),
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Setup rate limiting
        delay = self.config.get('delay', 1.0)
        self.rate_limiter = RateLimiter(delay=delay)
        
        # Initialize content extractor
        self.content_extractor = ContentExtractor(**self.config)
        
        # Setup robots checker
        respect_robots = self.config.get('respect_robots', True)
        if respect_robots:
            self.robots_checker = RobotsChecker(user_agent=user_agent)
        else:
            self.robots_checker = None
    
    def ingest_url(self, url: str, **options) -> WebContent:
        """
        Ingest content from a single URL.
        
        Args:
            url: URL to ingest
            **options: Processing options
            
        Returns:
            WebContent: Extracted web content
        """
        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError(f"Invalid URL: {url}")
        except Exception as e:
            raise ValidationError(f"Invalid URL: {url}") from e
        
        # Check robots.txt
        if self.robots_checker and not self.robots_checker.can_fetch(url):
            self.logger.warning(f"URL {url} blocked by robots.txt")
            raise ProcessingError(f"URL blocked by robots.txt: {url}")
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Fetch content
        try:
            timeout = options.get('timeout', self.config.get('timeout', 30))
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch URL {url}: {e}")
            raise ProcessingError(f"Failed to fetch URL: {e}")
        
        # Extract content
        html_content = response.text
        web_content = self.extract_content(html_content, url=url)
        web_content.status_code = response.status_code
        
        return web_content
    
    def crawl_sitemap(self, sitemap_url: str, **filters) -> List[WebContent]:
        """
        Crawl URLs from sitemap.
        
        Args:
            sitemap_url: URL of sitemap
            **filters: URL filtering criteria
            
        Returns:
            list: List of web content objects
        """
        crawler = SitemapCrawler(**self.config)
        
        # Parse sitemap
        try:
            urls = crawler.parse_sitemap(sitemap_url)
        except Exception as e:
            # Try as sitemap index
            try:
                urls = crawler.crawl_sitemap_index(sitemap_url)
            except Exception:
                raise ProcessingError(f"Failed to parse sitemap: {e}")
        
        # Apply filters
        if filters:
            urls = self._apply_url_filters(urls, filters)
        
        # Process each URL
        web_contents = []
        max_urls = filters.get('max_urls', self.config.get('max_urls'))
        
        for idx, url in enumerate(urls):
            if max_urls and idx >= max_urls:
                break
            
            try:
                web_content = self.ingest_url(url)
                web_contents.append(web_content)
                self.logger.debug(f"Crawled URL {idx+1}/{len(urls)}: {url}")
            except Exception as e:
                self.logger.error(f"Failed to crawl URL {url}: {e}")
                if self.config.get('fail_fast', False):
                    raise ProcessingError(f"Failed to crawl URL: {e}")
        
        return web_contents
    
    def crawl_domain(self, domain: str, max_pages: int = 100, **options) -> List[WebContent]:
        """
        Crawl entire domain starting from root.
        
        Args:
            domain: Domain to crawl
            max_pages: Maximum pages to crawl
            **options: Crawling options
            
        Returns:
            list: List of web content objects
        """
        # Normalize domain
        if not domain.startswith(('http://', 'https://')):
            domain = f"https://{domain}"
        
        # Start from domain root
        visited: Set[str] = set()
        to_visit: List[str] = [domain]
        web_contents = []
        
        while to_visit and len(web_contents) < max_pages:
            url = to_visit.pop(0)
            
            if url in visited:
                continue
            
            visited.add(url)
            
            try:
                web_content = self.ingest_url(url)
                web_contents.append(web_content)
                
                # Discover links
                for link in web_content.links:
                    parsed_link = urlparse(link)
                    parsed_domain = urlparse(domain)
                    
                    # Only follow links from same domain
                    if parsed_link.netloc == parsed_domain.netloc:
                        if link not in visited:
                            to_visit.append(link)
                
                self.logger.debug(f"Crawled page {len(web_contents)}/{max_pages}: {url}")
                
            except Exception as e:
                self.logger.error(f"Failed to crawl URL {url}: {e}")
                if self.config.get('fail_fast', False):
                    raise ProcessingError(f"Failed to crawl URL: {e}")
        
        return web_contents
    
    def extract_content(self, html_content: str, url: Optional[str] = None) -> WebContent:
        """
        Extract content from HTML.
        
        Args:
            html_content: HTML content to extract from
            url: Source URL for context
            
        Returns:
            WebContent: Extracted content
        """
        # Extract text content
        text = self.content_extractor.extract_text(html_content)
        
        # Extract metadata
        metadata = self.content_extractor.extract_metadata(html_content, url=url)
        
        # Extract links
        links = self.content_extractor.extract_links(html_content, base_url=url)
        
        # Get title
        title = metadata.get('title', '')
        
        # Create web content object
        web_content = WebContent(
            url=url or '',
            title=title,
            text=text,
            html=html_content,
            metadata=metadata,
            links=links
        )
        
        return web_content
    
    def _apply_url_filters(self, urls: List[str], filters: Dict[str, Any]) -> List[str]:
        """Apply filters to URL list."""
        filtered = urls
        
        if 'pattern' in filters:
            pattern = filters['pattern']
            import re
            regex = re.compile(pattern)
            filtered = [url for url in filtered if regex.search(url)]
        
        if 'domains' in filters:
            allowed_domains = filters['domains']
            filtered = [
                url for url in filtered
                if any(domain in url for domain in allowed_domains)
            ]
        
        if 'exclude_pattern' in filters:
            pattern = filters['exclude_pattern']
            import re
            regex = re.compile(pattern)
            filtered = [url for url in filtered if not regex.search(url)]
        
        return filtered
