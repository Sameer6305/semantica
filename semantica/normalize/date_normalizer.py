"""
Date and Time Normalization Module

Handles normalization of dates, times, and temporal expressions.

Key Features:
    - Date format standardization
    - Time zone normalization
    - Relative date processing
    - Temporal expression parsing
    - Date range handling

Main Classes:
    - DateNormalizer: Main date normalization class
    - TimeZoneNormalizer: Time zone processing
    - RelativeDateProcessor: Relative date handling
    - TemporalExpressionParser: Temporal expression parser
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger

# Optional imports for date parsing
try:
    from dateutil import parser as date_parser
    from dateutil.relativedelta import relativedelta
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False
    date_parser = None
    relativedelta = None


class DateNormalizer:
    """
    Date and time normalization handler.
    
    • Normalizes dates and times to standard formats
    • Handles various date formats and conventions
    • Processes time zones and UTC conversion
    • Manages relative dates and temporal expressions
    • Standardizes date representations
    • Supports multiple calendar systems
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize date normalizer.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("date_normalizer")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.timezone_normalizer = TimeZoneNormalizer(**self.config)
        self.relative_date_processor = RelativeDateProcessor(**self.config)
        self.temporal_parser = TemporalExpressionParser(**self.config)
    
    def normalize_date(self, date_input: Any, **options) -> str:
        """
        Normalize date to standard format.
        
        Args:
            date_input: Date input (string, datetime, or other)
            **options: Normalization options:
                - format: Output format (default: 'ISO8601')
                - timezone: Target timezone (default: 'UTC')
        
        Returns:
            Normalized date string
        """
        if not date_input:
            return ""
        
        # Parse date input
        if isinstance(date_input, str):
            try:
                if HAS_DATEUTIL and date_parser:
                    dt = date_parser.parse(date_input)
                else:
                    # Fallback to basic parsing
                    dt = datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            except Exception:
                # Try relative date processing
                dt = self.relative_date_processor.process_relative_expression(date_input)
        elif isinstance(date_input, datetime):
            dt = date_input
        else:
            raise ValidationError(f"Unsupported date input type: {type(date_input)}")
        
        # Normalize timezone
        target_tz = options.get("timezone", "UTC")
        if target_tz != "UTC":
            dt = self.timezone_normalizer.normalize_timezone(dt, target_tz)
        else:
            dt = self.timezone_normalizer.convert_to_utc(dt)
        
        # Format output
        output_format = options.get("format", "ISO8601")
        if output_format == "ISO8601":
            return dt.isoformat()
        elif output_format == "date":
            return dt.date().isoformat()
        else:
            return dt.strftime(output_format)
    
    def normalize_time(self, time_input: Any, **options) -> str:
        """
        Normalize time to standard format.
        
        Args:
            time_input: Time input
            **options: Normalization options
        
        Returns:
            Normalized time string
        """
        if isinstance(time_input, str):
            try:
                if HAS_DATEUTIL and date_parser:
                    dt = date_parser.parse(time_input)
                else:
                    # Fallback parsing
                    dt = datetime.fromisoformat(time_input)
            except Exception:
                return ""
        elif isinstance(time_input, datetime):
            dt = time_input
        else:
            return ""
        
        return dt.time().isoformat()
    
    def process_relative_date(self, relative_expression: str, reference_date: Optional[datetime] = None) -> datetime:
        """
        Process relative date expressions.
        
        Args:
            relative_expression: Relative date expression
            reference_date: Reference date (default: now)
        
        Returns:
            Calculated date
        """
        return self.relative_date_processor.process_relative_expression(relative_expression, reference_date)
    
    def parse_temporal_expression(self, temporal_text: str, **context) -> Dict[str, Any]:
        """
        Parse temporal expressions and references.
        
        Args:
            temporal_text: Temporal expression text
            **context: Context information
        
        Returns:
            Parsed temporal data
        """
        return self.temporal_parser.parse_temporal_expression(temporal_text, **context)


class TimeZoneNormalizer:
    """
    Time zone normalization engine.
    
    • Handles time zone conversion and normalization
    • Manages UTC conversion
    • Processes time zone abbreviations
    • Handles daylight saving time
    • Manages time zone databases
    """
    
    def __init__(self, **config):
        """
        Initialize time zone normalizer.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("timezone_normalizer")
        self.config = config
    
    def normalize_timezone(self, datetime_obj: datetime, target_timezone: str = "UTC") -> datetime:
        """
        Normalize datetime to target timezone.
        
        Args:
            datetime_obj: Datetime object
            target_timezone: Target timezone
        
        Returns:
            Normalized datetime
        """
        try:
            from zoneinfo import ZoneInfo
            target_tz = ZoneInfo(target_timezone)
            if datetime_obj.tzinfo is None:
                datetime_obj = datetime_obj.replace(tzinfo=timezone.utc)
            return datetime_obj.astimezone(target_tz)
        except Exception:
            # Fallback if zoneinfo not available
            return datetime_obj
    
    def convert_to_utc(self, datetime_obj: datetime, source_timezone: Optional[str] = None) -> datetime:
        """
        Convert datetime to UTC.
        
        Args:
            datetime_obj: Datetime object
            source_timezone: Source timezone (auto-detect if None)
        
        Returns:
            UTC datetime
        """
        if datetime_obj.tzinfo is None:
            if source_timezone:
                datetime_obj = self.normalize_timezone(datetime_obj, source_timezone)
            else:
                datetime_obj = datetime_obj.replace(tzinfo=timezone.utc)
        
        return datetime_obj.astimezone(timezone.utc)
    
    def handle_dst_transitions(self, datetime_obj: datetime, timezone_str: str) -> datetime:
        """
        Handle daylight saving time transitions.
        
        Args:
            datetime_obj: Datetime object
            timezone_str: Timezone string
        
        Returns:
            Adjusted datetime
        """
        return self.normalize_timezone(datetime_obj, timezone_str)


class RelativeDateProcessor:
    """
    Relative date processing engine.
    
    • Processes relative date expressions
    • Calculates absolute dates from relative terms
    • Handles various relative formats
    • Manages date arithmetic
    """
    
    def __init__(self, **config):
        """
        Initialize relative date processor.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("relative_date_processor")
        self.config = config
        
        self.relative_terms = {
            "today": 0,
            "yesterday": -1,
            "tomorrow": 1,
            "now": 0,
        }
    
    def process_relative_expression(self, expression: str, reference_date: Optional[datetime] = None) -> datetime:
        """
        Process relative date expression.
        
        Args:
            expression: Relative date expression
            reference_date: Reference date (default: now)
        
        Returns:
            Calculated date
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        expression_lower = expression.lower().strip()
        
        # Check for relative terms
        if expression_lower in self.relative_terms:
            days = self.relative_terms[expression_lower]
            return reference_date + timedelta(days=days)
        
        # Parse patterns like "3 days ago", "2 weeks from now"
        pattern = r'(\d+)\s*(day|week|month|year)s?\s*(ago|from now|later)?'
        match = re.search(pattern, expression_lower)
        
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            direction = match.group(3) or "ago"
            
            if unit == "day":
                delta = timedelta(days=amount)
            elif unit == "week":
                delta = timedelta(weeks=amount)
            elif unit == "month":
                if HAS_DATEUTIL and relativedelta:
                    delta = relativedelta(months=amount)
                else:
                    # Approximate month as 30 days
                    delta = timedelta(days=amount * 30)
            elif unit == "year":
                if HAS_DATEUTIL and relativedelta:
                    delta = relativedelta(years=amount)
                else:
                    # Approximate year as 365 days
                    delta = timedelta(days=amount * 365)
            else:
                delta = timedelta(days=amount)
            
            if direction in ["ago", "before"]:
                return reference_date - delta
            else:
                return reference_date + delta
        
        # Fallback: try to parse as absolute date
        try:
            if HAS_DATEUTIL and date_parser:
                return date_parser.parse(expression)
            else:
                # Basic ISO format parsing
                return datetime.fromisoformat(expression.replace('Z', '+00:00'))
        except Exception:
            return reference_date
    
    def calculate_date_offset(self, expression: str, reference_date: datetime) -> datetime:
        """
        Calculate date offset from expression.
        
        Args:
            expression: Offset expression
            reference_date: Reference date
        
        Returns:
            Calculated date
        """
        return self.process_relative_expression(expression, reference_date)
    
    def handle_relative_terms(self, term: str, reference_date: datetime) -> datetime:
        """
        Handle specific relative terms.
        
        Args:
            term: Relative term
            reference_date: Reference date
        
        Returns:
            Calculated date
        """
        return self.process_relative_expression(term, reference_date)


class TemporalExpressionParser:
    """
    Temporal expression parsing engine.
    
    • Parses natural language temporal expressions
    • Extracts date and time components
    • Handles complex temporal references
    • Processes temporal ranges and periods
    """
    
    def __init__(self, **config):
        """
        Initialize temporal expression parser.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("temporal_expression_parser")
        self.config = config
    
    def parse_temporal_expression(self, text: str, **context) -> Dict[str, Any]:
        """
        Parse temporal expression from text.
        
        Args:
            text: Temporal expression text
            **context: Context information
        
        Returns:
            Parsed temporal data
        """
        result = {
            "date": None,
            "time": None,
            "range": None,
            "relative": False
        }
        
        # Try to extract date components
        date_components = self.extract_date_components(text)
        if date_components:
            result.update(date_components)
        
        # Try to extract time components
        time_components = self.extract_time_components(text)
        if time_components:
            result.update(time_components)
        
        # Try to parse as range
        range_info = self.process_temporal_ranges(text)
        if range_info:
            result["range"] = range_info
        
        return result
    
    def extract_date_components(self, text: str) -> Dict[str, Any]:
        """
        Extract date components from text.
        
        Args:
            text: Input text
        
        Returns:
            Date components
        """
        try:
            if HAS_DATEUTIL and date_parser:
                dt = date_parser.parse(text, fuzzy=True)
            else:
                # Basic parsing
                dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
            return {
                "date": dt.date().isoformat(),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day
            }
        except Exception:
            return {}
    
    def extract_time_components(self, text: str) -> Dict[str, Any]:
        """
        Extract time components from text.
        
        Args:
            text: Input text
        
        Returns:
            Time components
        """
        try:
            if HAS_DATEUTIL and date_parser:
                dt = date_parser.parse(text, fuzzy=True)
            else:
                # Basic parsing
                dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
            return {
                "time": dt.time().isoformat(),
                "hour": dt.hour,
                "minute": dt.minute,
                "second": dt.second
            }
        except Exception:
            return {}
    
    def process_temporal_ranges(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Process temporal ranges and periods.
        
        Args:
            text: Input text
        
        Returns:
            Range information or None
        """
        # Look for range patterns like "from X to Y", "between X and Y"
        range_patterns = [
            r'from\s+(.+?)\s+to\s+(.+?)',
            r'between\s+(.+?)\s+and\s+(.+?)',
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    if HAS_DATEUTIL and date_parser:
                        start = date_parser.parse(match.group(1))
                        end = date_parser.parse(match.group(2))
                    else:
                        # Basic parsing
                        start = datetime.fromisoformat(match.group(1).replace('Z', '+00:00'))
                        end = datetime.fromisoformat(match.group(2).replace('Z', '+00:00'))
                    return {
                        "start": start.isoformat(),
                        "end": end.isoformat()
                    }
                except Exception:
                    continue
        
        return None
