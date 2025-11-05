"""
Email Ingestion Module

Handles email protocol processing and content extraction.

Key Features:
    - IMAP/POP3 email retrieval
    - Email content parsing
    - Attachment processing
    - Email metadata extraction
    - Thread analysis

Main Classes:
    - EmailIngestor: Main email ingestion class
    - EmailParser: Email content parser
    - AttachmentProcessor: Email attachment handler
"""

import email
import imaplib
import poplib
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from email.header import decode_header
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


@dataclass
class EmailData:
    """Email data representation."""
    
    message_id: str
    subject: str
    from_address: str
    to_addresses: List[str]
    cc_addresses: List[str] = field(default_factory=list)
    date: Optional[datetime] = None
    body_text: str = ""
    body_html: str = ""
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    thread_id: Optional[str] = None


class AttachmentProcessor:
    """
    Email attachment processing and extraction.
    
    Processes various attachment types,
    extracts text content from documents,
    and handles image and media attachments.
    """
    
    def __init__(self, **config):
        """
        Initialize attachment processor.
        
        Args:
            **config: Processor configuration
        """
        self.logger = get_logger("attachment_processor")
        self.config = config
        self.temp_dir = tempfile.mkdtemp(prefix="semantica_attachments_")
    
    def process_attachment(self, attachment_data: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        """
        Process individual email attachment.
        
        Args:
            attachment_data: Attachment content bytes
            filename: Attachment filename
            content_type: MIME content type
            
        Returns:
            dict: Attachment information
        """
        attachment_info = {
            "filename": filename,
            "content_type": content_type,
            "size": len(attachment_data),
            "saved_path": None,
            "text_content": None
        }
        
        # Save attachment to temporary file
        safe_filename = "".join(c for c in filename if c.isalnum() or c in ".-_")
        file_path = Path(self.temp_dir) / safe_filename
        
        try:
            with open(file_path, 'wb') as f:
                f.write(attachment_data)
            attachment_info["saved_path"] = str(file_path)
        except Exception as e:
            self.logger.error(f"Failed to save attachment {filename}: {e}")
            return attachment_info
        
        # Extract text content if applicable
        if content_type.startswith('text/'):
            try:
                text_content = attachment_data.decode('utf-8', errors='ignore')
                attachment_info["text_content"] = text_content
            except Exception:
                pass
        elif content_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            # Extract text from documents
            text_content = self.extract_text_content(file_path, content_type)
            attachment_info["text_content"] = text_content
        
        return attachment_info
    
    def extract_text_content(self, attachment_path: Path, file_type: str) -> Optional[str]:
        """
        Extract text content from attachment.
        
        Args:
            attachment_path: Path to attachment file
            file_type: File MIME type
            
        Returns:
            str: Extracted text content or None
        """
        try:
            if file_type == 'application/pdf':
                # PDF text extraction would require PyPDF2 or pdfplumber
                return None  # Placeholder - would need PDF library
            elif file_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                # Word document extraction would require python-docx
                return None  # Placeholder - would need docx library
            elif file_type.startswith('text/'):
                with open(attachment_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            self.logger.error(f"Failed to extract text from {attachment_path}: {e}")
        
        return None
    
    def cleanup_attachments(self, attachment_paths: List[str]):
        """
        Clean up temporary attachment files.
        
        Args:
            attachment_paths: List of attachment file paths
        """
        import shutil
        try:
            if Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Failed to cleanup attachments: {e}")


class EmailParser:
    """
    Email content parsing and extraction.
    
    Parses email headers and metadata,
    extracts email body content,
    handles MIME multipart messages,
    and processes HTML and plain text content.
    """
    
    def __init__(self, **config):
        """
        Initialize email parser.
        
        Args:
            **config: Parser configuration
        """
        self.logger = get_logger("email_parser")
        self.config = config
    
    def parse_headers(self, email_message: email.message.Message) -> Dict[str, str]:
        """
        Parse email headers and metadata.
        
        Args:
            email_message: Email message object
            
        Returns:
            dict: Header dictionary
        """
        headers = {}
        
        for header_name in email_message.keys():
            header_value = email_message[header_name]
            if header_value:
                # Decode header if needed
                decoded_value = decode_header(header_value)
                decoded_str = ''.join(
                    part[0].decode(part[1] or 'utf-8') if isinstance(part[0], bytes) else part[0]
                    for part in decoded_value
                )
                headers[header_name.lower()] = decoded_str
        
        return headers
    
    def parse_body(self, email_message: email.message.Message) -> Dict[str, str]:
        """
        Parse email body content.
        
        Args:
            email_message: Email message object
            
        Returns:
            dict: Body content dictionary with 'text' and 'html' keys
        """
        body = {"text": "", "html": ""}
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition", ""))
                
                # Skip attachments
                if "attachment" in content_disposition:
                    continue
                
                # Extract text content
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            body["text"] = payload.decode(charset, errors='ignore')
                    except Exception as e:
                        self.logger.warning(f"Failed to decode text part: {e}")
                
                # Extract HTML content
                elif content_type == "text/html":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            charset = part.get_content_charset() or 'utf-8'
                            body["html"] = payload.decode(charset, errors='ignore')
                    except Exception as e:
                        self.logger.warning(f"Failed to decode HTML part: {e}")
        else:
            # Single part message
            content_type = email_message.get_content_type()
            try:
                payload = email_message.get_payload(decode=True)
                if payload:
                    charset = email_message.get_content_charset() or 'utf-8'
                    decoded = payload.decode(charset, errors='ignore')
                    
                    if content_type == "text/html":
                        body["html"] = decoded
                    else:
                        body["text"] = decoded
            except Exception as e:
                self.logger.warning(f"Failed to decode body: {e}")
        
        return body
    
    def extract_links(self, email_content: str) -> List[str]:
        """
        Extract links from email content.
        
        Args:
            email_content: Email content (HTML or text)
            
        Returns:
            list: List of extracted links
        """
        links = []
        
        # Extract from HTML if available
        if email_content.strip().startswith('<'):
            try:
                soup = BeautifulSoup(email_content, 'html.parser')
                for link_tag in soup.find_all('a', href=True):
                    links.append(link_tag['href'])
            except Exception:
                pass
        
        # Extract URLs from text using regex
        import re
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text_links = re.findall(url_pattern, email_content)
        links.extend(text_links)
        
        # Remove duplicates
        return list(set(links))


class EmailIngestor:
    """
    Email protocol ingestion handler.
    
    Connects to email servers via IMAP/POP3,
    retrieves emails from mailboxes,
    and processes email content and attachments.
    
    Attributes:
        imap_client: IMAP client for email access
        pop3_client: POP3 client for email access
        parser: Email content parser
        attachment_processor: Attachment handling system
        
    Methods:
        ingest_mailbox(): Ingest emails from mailbox
        process_email(): Process individual email
        extract_attachments(): Extract email attachments
        analyze_threads(): Analyze email threads
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize email ingestor.
        
        Args:
            config: Email ingestion configuration
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("email_ingestor")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Initialize parser
        self.parser = EmailParser(**self.config)
        
        # Initialize attachment processor
        self.attachment_processor = AttachmentProcessor(**self.config)
        
        # Connection objects (initialized on connect)
        self.imap_client: Optional[imaplib.IMAP4_SSL] = None
        self.pop3_client: Optional[poplib.POP3_SSL] = None
    
    def connect_imap(self, server: str, port: int = 993, username: str = None, password: str = None):
        """
        Connect to IMAP server.
        
        Args:
            server: IMAP server address
            port: IMAP server port (default: 993)
            username: Email username
            password: Email password
        """
        try:
            self.imap_client = imaplib.IMAP4_SSL(server, port)
            if username and password:
                self.imap_client.login(username, password)
            self.logger.info(f"Connected to IMAP server: {server}")
        except Exception as e:
            self.logger.error(f"Failed to connect to IMAP: {e}")
            raise ProcessingError(f"Failed to connect to IMAP server: {e}")
    
    def connect_pop3(self, server: str, port: int = 995, username: str = None, password: str = None):
        """
        Connect to POP3 server.
        
        Args:
            server: POP3 server address
            port: POP3 server port (default: 995)
            username: Email username
            password: Email password
        """
        try:
            self.pop3_client = poplib.POP3_SSL(server, port)
            if username and password:
                self.pop3_client.user(username)
                self.pop3_client.pass_(password)
            self.logger.info(f"Connected to POP3 server: {server}")
        except Exception as e:
            self.logger.error(f"Failed to connect to POP3: {e}")
            raise ProcessingError(f"Failed to connect to POP3 server: {e}")
    
    def ingest_mailbox(self, mailbox_name: str = "INBOX", protocol: str = "imap", **filters) -> List[EmailData]:
        """
        Ingest emails from specified mailbox.
        
        Args:
            mailbox_name: Mailbox name (default: "INBOX")
            protocol: Protocol to use ("imap" or "pop3")
            **filters: Email filtering criteria:
                - since: Date to start from
                - max_emails: Maximum number of emails
                - unread_only: Only fetch unread emails
                
        Returns:
            list: Processed email collection
        """
        if protocol.lower() == "imap":
            if not self.imap_client:
                raise ProcessingError("IMAP client not connected. Call connect_imap() first.")
            return self._ingest_imap(mailbox_name, **filters)
        elif protocol.lower() == "pop3":
            if not self.pop3_client:
                raise ProcessingError("POP3 client not connected. Call connect_pop3() first.")
            return self._ingest_pop3(**filters)
        else:
            raise ValidationError(f"Unsupported protocol: {protocol}")
    
    def _ingest_imap(self, mailbox_name: str, **filters) -> List[EmailData]:
        """Ingest emails using IMAP."""
        try:
            # Select mailbox
            self.imap_client.select(mailbox_name)
            
            # Build search criteria
            search_criteria = "ALL"
            if filters.get("unread_only"):
                search_criteria = "UNSEEN"
            if filters.get("since"):
                search_criteria = f"({search_criteria} SINCE {filters['since']})"
            
            # Search for emails
            status, message_ids = self.imap_client.search(None, search_criteria)
            
            if status != "OK":
                raise ProcessingError("Failed to search emails")
            
            email_ids = message_ids[0].split()
            
            # Limit number of emails
            max_emails = filters.get("max_emails")
            if max_emails:
                email_ids = email_ids[-max_emails:]  # Get most recent
            
            emails = []
            for email_id in email_ids:
                try:
                    status, msg_data = self.imap_client.fetch(email_id, '(RFC822)')
                    if status == "OK":
                        email_message = email.message_from_bytes(msg_data[0][1])
                        email_data = self.process_email(email_message)
                        emails.append(email_data)
                except Exception as e:
                    self.logger.error(f"Failed to process email {email_id}: {e}")
            
            return emails
            
        except Exception as e:
            self.logger.error(f"Error ingesting IMAP mailbox: {e}")
            raise ProcessingError(f"Failed to ingest mailbox: {e}")
    
    def _ingest_pop3(self, **filters) -> List[EmailData]:
        """Ingest emails using POP3."""
        try:
            # Get email list
            num_messages = len(self.pop3_client.list()[1])
            
            max_emails = filters.get("max_emails", num_messages)
            emails = []
            
            # Fetch emails (most recent first)
            start = max(1, num_messages - max_emails + 1)
            for i in range(start, num_messages + 1):
                try:
                    response = self.pop3_client.retr(i)
                    email_content = b'\n'.join(response[1])
                    email_message = email.message_from_bytes(email_content)
                    email_data = self.process_email(email_message)
                    emails.append(email_data)
                except Exception as e:
                    self.logger.error(f"Failed to process email {i}: {e}")
            
            return emails
            
        except Exception as e:
            self.logger.error(f"Error ingesting POP3: {e}")
            raise ProcessingError(f"Failed to ingest POP3: {e}")
    
    def process_email(self, email_message: email.message.Message) -> EmailData:
        """
        Process individual email message.
        
        Args:
            email_message: Email message object
            
        Returns:
            EmailData: Structured email data
        """
        # Parse headers
        headers = self.parser.parse_headers(email_message)
        
        # Extract basic information
        message_id = headers.get('message-id', '')
        subject = headers.get('subject', '')
        from_address = headers.get('from', '')
        to_addresses = self._parse_address_list(headers.get('to', ''))
        cc_addresses = self._parse_address_list(headers.get('cc', ''))
        
        # Parse date
        date_str = headers.get('date', '')
        email_date = None
        if date_str:
            try:
                from email.utils import parsedate_to_datetime
                email_date = parsedate_to_datetime(date_str)
            except Exception:
                pass
        
        # Parse body
        body = self.parser.parse_body(email_message)
        
        # Extract thread ID
        thread_id = headers.get('in-reply-to') or headers.get('references', '').split()[0] if headers.get('references') else None
        
        # Extract attachments
        attachments = self.extract_attachments(email_message)
        
        return EmailData(
            message_id=message_id,
            subject=subject,
            from_address=from_address,
            to_addresses=to_addresses,
            cc_addresses=cc_addresses,
            date=email_date,
            body_text=body["text"],
            body_html=body["html"],
            attachments=attachments,
            headers=headers,
            thread_id=thread_id
        )
    
    def extract_attachments(self, email_message: email.message.Message) -> List[Dict[str, Any]]:
        """
        Extract and process email attachments.
        
        Args:
            email_message: Email message object
            
        Returns:
            list: Attachment information
        """
        attachments = []
        
        if email_message.is_multipart():
            for part in email_message.walk():
                content_disposition = str(part.get("Content-Disposition", ""))
                
                if "attachment" in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        # Decode filename
                        decoded_filename = decode_header(filename)[0][0]
                        if isinstance(decoded_filename, bytes):
                            decoded_filename = decoded_filename.decode('utf-8', errors='ignore')
                        
                        # Get attachment data
                        attachment_data = part.get_payload(decode=True)
                        content_type = part.get_content_type()
                        
                        # Process attachment
                        attachment_info = self.attachment_processor.process_attachment(
                            attachment_data,
                            decoded_filename,
                            content_type
                        )
                        attachments.append(attachment_info)
        
        return attachments
    
    def analyze_threads(self, emails: List[EmailData]) -> Dict[str, Any]:
        """
        Analyze email threads and conversations.
        
        Args:
            emails: List of email data objects
            
        Returns:
            dict: Thread analysis results
        """
        threads = {}
        
        for email_data in emails:
            thread_id = email_data.thread_id or email_data.message_id
            
            if thread_id not in threads:
                threads[thread_id] = {
                    "thread_id": thread_id,
                    "emails": [],
                    "participants": set(),
                    "subject": email_data.subject,
                    "message_count": 0
                }
            
            threads[thread_id]["emails"].append(email_data)
            threads[thread_id]["participants"].add(email_data.from_address)
            threads[thread_id]["participants"].update(email_data.to_addresses)
            threads[thread_id]["message_count"] += 1
        
        # Convert sets to lists for JSON serialization
        for thread in threads.values():
            thread["participants"] = list(thread["participants"])
        
        return {
            "total_threads": len(threads),
            "threads": list(threads.values())
        }
    
    def _parse_address_list(self, address_string: str) -> List[str]:
        """Parse email address list string into list of addresses."""
        if not address_string:
            return []
        
        addresses = []
        try:
            from email.utils import parseaddr, getaddresses
            parsed_addresses = getaddresses([address_string])
            addresses = [addr[1] for addr in parsed_addresses if addr[1]]
        except Exception:
            # Fallback: simple extraction
            import re
            email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
            addresses = re.findall(email_pattern, address_string)
        
        return addresses
    
    def disconnect(self):
        """Disconnect from email servers."""
        if self.imap_client:
            try:
                self.imap_client.close()
                self.imap_client.logout()
            except Exception:
                pass
            self.imap_client = None
        
        if self.pop3_client:
            try:
                self.pop3_client.quit()
            except Exception:
                pass
            self.pop3_client = None
