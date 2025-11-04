"""
Data Cleaning Module

Handles general data cleaning and quality improvement.

Key Features:
    - Data quality assessment
    - Duplicate detection and removal
    - Data validation and correction
    - Missing value handling
    - Data consistency checking

Main Classes:
    - DataCleaner: Main data cleaning class
    - DuplicateDetector: Duplicate detection engine
    - DataValidator: Data validation engine
    - MissingValueHandler: Missing value processor
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger


@dataclass
class DuplicateGroup:
    """Duplicate record group."""
    records: List[Dict[str, Any]]
    similarity_score: float
    canonical_record: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Data validation result."""
    valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


class DataCleaner:
    """
    General data cleaning and quality improvement handler.
    
    • Cleans and improves data quality
    • Detects and removes duplicates
    • Validates data integrity
    • Handles missing values
    • Ensures data consistency
    • Supports various data types
    """
    
    def __init__(self, config=None, **kwargs):
        """
        Initialize data cleaner.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("data_cleaner")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.duplicate_detector = DuplicateDetector(**self.config)
        self.data_validator = DataValidator(**self.config)
        self.missing_value_handler = MissingValueHandler(**self.config)
    
    def clean_data(self, dataset: List[Dict[str, Any]], **options) -> List[Dict[str, Any]]:
        """
        Clean dataset with various cleaning operations.
        
        Args:
            dataset: List of data records
            **options: Cleaning options:
                - remove_duplicates: Remove duplicates (default: True)
                - validate: Validate data (default: True)
                - handle_missing: Handle missing values (default: True)
        
        Returns:
            Cleaned dataset
        """
        cleaned = list(dataset)
        
        # Handle missing values
        if options.get("handle_missing", True):
            strategy = options.get("missing_strategy", "remove")
            cleaned = self.missing_value_handler.handle_missing_values(cleaned, strategy=strategy)
        
        # Validate data
        if options.get("validate", True):
            schema = options.get("schema")
            validation = self.data_validator.validate_dataset(cleaned, schema)
            if not validation.valid:
                self.logger.warning(f"Validation found {len(validation.errors)} errors")
        
        # Remove duplicates
        if options.get("remove_duplicates", True):
            criteria = options.get("duplicate_criteria", {})
            duplicates = self.detect_duplicates(cleaned, **criteria)
            
            # Remove duplicates (keep first occurrence)
            duplicate_indices = set()
            for group in duplicates:
                for record in group.records[1:]:  # Skip first (canonical)
                    if record in cleaned:
                        idx = cleaned.index(record)
                        duplicate_indices.add(idx)
            
            cleaned = [r for i, r in enumerate(cleaned) if i not in duplicate_indices]
        
        return cleaned
    
    def detect_duplicates(self, dataset: List[Dict[str, Any]], **criteria) -> List[DuplicateGroup]:
        """
        Detect duplicate records in dataset.
        
        Args:
            dataset: List of data records
            **criteria: Duplicate detection criteria
        
        Returns:
            List of duplicate groups
        """
        return self.duplicate_detector.detect_duplicates(dataset, **criteria)
    
    def validate_data(self, dataset: List[Dict[str, Any]], schema=None) -> ValidationResult:
        """
        Validate data against schema or rules.
        
        Args:
            dataset: List of data records
            schema: Validation schema
        
        Returns:
            Validation result
        """
        return self.data_validator.validate_dataset(dataset, schema)
    
    def handle_missing_values(self, dataset: List[Dict[str, Any]], **strategy) -> List[Dict[str, Any]]:
        """
        Handle missing values in dataset.
        
        Args:
            dataset: List of data records
            **strategy: Missing value handling strategy
        
        Returns:
            Processed dataset
        """
        return self.missing_value_handler.handle_missing_values(dataset, **strategy)


class DuplicateDetector:
    """
    Duplicate detection engine.
    
    • Detects duplicate records
    • Calculates similarity scores
    • Handles fuzzy matching
    • Manages duplicate resolution
    """
    
    def __init__(self, **config):
        """
        Initialize duplicate detector.
        
        Args:
            **config: Configuration options:
                - similarity_threshold: Minimum similarity for duplicates (default: 0.8)
                - key_fields: Fields to use for comparison
        """
        self.logger = get_logger("duplicate_detector")
        self.config = config
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        self.key_fields = config.get("key_fields", [])
    
    def detect_duplicates(self, dataset: List[Dict[str, Any]], **criteria) -> List[DuplicateGroup]:
        """
        Detect duplicates in dataset.
        
        Args:
            dataset: List of records
            **criteria: Detection criteria
        
        Returns:
            List of duplicate groups
        """
        threshold = criteria.get("threshold", self.similarity_threshold)
        key_fields = criteria.get("key_fields", self.key_fields)
        
        duplicate_groups = []
        processed = set()
        
        for i, record1 in enumerate(dataset):
            if i in processed:
                continue
            
            group = [record1]
            
            for j, record2 in enumerate(dataset[i+1:], start=i+1):
                if j in processed:
                    continue
                
                similarity = self.calculate_similarity(record1, record2, key_fields=key_fields)
                
                if similarity >= threshold:
                    group.append(record2)
                    processed.add(j)
            
            if len(group) > 1:
                # Calculate average similarity
                avg_similarity = sum(
                    self.calculate_similarity(group[0], r, key_fields=key_fields)
                    for r in group[1:]
                ) / (len(group) - 1)
                
                duplicate_groups.append(
                    DuplicateGroup(
                        records=group,
                        similarity_score=avg_similarity,
                        canonical_record=group[0]
                    )
                )
                processed.add(i)
        
        return duplicate_groups
    
    def calculate_similarity(self, record1: Dict[str, Any], record2: Dict[str, Any], **options) -> float:
        """
        Calculate similarity between records.
        
        Args:
            record1: First record
            record2: Second record
            **options: Similarity calculation options
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        key_fields = options.get("key_fields", self.key_fields)
        
        if not key_fields:
            # Use all common fields
            key_fields = list(set(record1.keys()) & set(record2.keys()))
        
        if not key_fields:
            return 0.0
        
        similarities = []
        
        for field in key_fields:
            val1 = record1.get(field)
            val2 = record2.get(field)
            
            if val1 is None or val2 is None:
                continue
            
            # Exact match
            if val1 == val2:
                similarities.append(1.0)
            else:
                # String similarity
                if isinstance(val1, str) and isinstance(val2, str):
                    sim = self._string_similarity(val1, val2)
                    similarities.append(sim)
                else:
                    similarities.append(0.0)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using simple ratio."""
        if not s1 or not s2:
            return 0.0
        
        if s1 == s2:
            return 1.0
        
        # Simple character overlap
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        if s1_lower == s2_lower:
            return 0.95
        
        # Character-level similarity
        set1 = set(s1_lower)
        set2 = set(s2_lower)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def resolve_duplicates(self, duplicate_groups: List[DuplicateGroup], **strategy) -> List[Dict[str, Any]]:
        """
        Resolve duplicate groups.
        
        Args:
            duplicate_groups: List of duplicate groups
            **strategy: Resolution strategy
        
        Returns:
            List of resolved records
        """
        resolved = []
        strategy_type = strategy.get("strategy", "keep_first")
        
        for group in duplicate_groups:
            if strategy_type == "keep_first":
                resolved.append(group.canonical_record or group.records[0])
            elif strategy_type == "merge":
                merged = self._merge_records(group.records)
                resolved.append(merged)
            else:
                resolved.append(group.records[0])
        
        return resolved
    
    def _merge_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple records into one."""
        merged = {}
        
        for record in records:
            for key, value in record.items():
                if key not in merged or merged[key] is None:
                    merged[key] = value
                elif value is not None and merged[key] != value:
                    # Keep first non-null value
                    pass
        
        return merged


class DataValidator:
    """
    Data validation engine.
    
    • Validates data integrity
    • Checks data types and formats
    • Validates constraints
    • Handles validation errors
    """
    
    def __init__(self, **config):
        """
        Initialize data validator.
        
        Args:
            **config: Configuration options
        """
        self.logger = get_logger("data_validator")
        self.config = config
    
    def validate_dataset(self, dataset: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate entire dataset.
        
        Args:
            dataset: List of records
            schema: Validation schema
        
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        if not dataset:
            return ValidationResult(valid=True)
        
        for i, record in enumerate(dataset):
            record_validation = self.validate_record(record, schema)
            
            for error in record_validation.errors:
                errors.append({"record_index": i, **error})
            
            for warning in record_validation.warnings:
                warnings.append({"record_index": i, **warning})
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_record(self, record: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate individual record.
        
        Args:
            record: Record to validate
            schema: Validation schema
        
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        if schema:
            # Validate against schema
            expected_fields = schema.get("fields", {})
            
            for field, field_schema in expected_fields.items():
                value = record.get(field)
                
                # Check required fields
                if field_schema.get("required", False) and value is None:
                    errors.append({"field": field, "message": f"Required field '{field}' is missing"})
                
                # Check type
                expected_type = field_schema.get("type")
                if expected_type and value is not None:
                    type_check = self.check_data_types(value, expected_type)
                    if not type_check:
                        errors.append({
                            "field": field,
                            "message": f"Field '{field}' has incorrect type, expected {expected_type}"
                        })
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def check_data_types(self, data: Any, expected_types: Union[type, List[type]]) -> bool:
        """
        Check data types against expected types.
        
        Args:
            data: Data to check
            expected_types: Expected type(s)
        
        Returns:
            True if type matches
        """
        if isinstance(expected_types, type):
            expected_types = [expected_types]
        
        actual_type = type(data)
        
        # Map string type names to actual types
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        
        for expected_type in expected_types:
            if isinstance(expected_type, str):
                expected_type = type_map.get(expected_type, str)
            
            if isinstance(data, expected_type):
                return True
        
        return False


class MissingValueHandler:
    """
    Missing value processing engine.
    
    • Identifies missing values
    • Applies handling strategies
    • Fills missing data
    • Removes incomplete records
    """
    
    def __init__(self, **config):
        """
        Initialize missing value handler.
        
        Args:
            **config: Configuration options:
                - missing_values: List of values considered missing (default: [None, "", "N/A", "null"])
        """
        self.logger = get_logger("missing_value_handler")
        self.config = config
        self.missing_values = config.get("missing_values", [None, "", "N/A", "null", "NULL"])
    
    def identify_missing_values(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify missing values in dataset.
        
        Args:
            dataset: List of records
        
        Returns:
            Missing value information
        """
        missing_info = defaultdict(int)
        total_records = len(dataset)
        
        if not dataset:
            return {"total_records": 0, "missing_counts": {}}
        
        all_fields = set()
        for record in dataset:
            all_fields.update(record.keys())
        
        for field in all_fields:
            missing_count = 0
            for record in dataset:
                value = record.get(field)
                if value in self.missing_values or value is None:
                    missing_count += 1
            missing_info[field] = missing_count
        
        return {
            "total_records": total_records,
            "missing_counts": dict(missing_info),
            "missing_percentages": {
                field: (count / total_records * 100) if total_records > 0 else 0
                for field, count in missing_info.items()
            }
        }
    
    def handle_missing_values(self, dataset: List[Dict[str, Any]], strategy: str = "remove") -> List[Dict[str, Any]]:
        """
        Handle missing values using specified strategy.
        
        Args:
            dataset: List of records
            strategy: Handling strategy ('remove', 'fill', 'impute')
        
        Returns:
            Processed dataset
        """
        if strategy == "remove":
            return self._remove_missing(dataset)
        elif strategy == "fill":
            return self._fill_missing(dataset)
        elif strategy == "impute":
            return self.impute_values(dataset)
        else:
            return dataset
    
    def _remove_missing(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove records with missing values."""
        return [
            record for record in dataset
            if not any(
                value in self.missing_values or value is None
                for value in record.values()
            )
        ]
    
    def _fill_missing(self, dataset: List[Dict[str, Any]], fill_value: Any = "") -> List[Dict[str, Any]]:
        """Fill missing values with default value."""
        filled = []
        for record in dataset:
            filled_record = {}
            for key, value in record.items():
                if value in self.missing_values or value is None:
                    filled_record[key] = fill_value
                else:
                    filled_record[key] = value
            filled.append(filled_record)
        return filled
    
    def impute_values(self, dataset: List[Dict[str, Any]], method: str = "mean") -> List[Dict[str, Any]]:
        """
        Impute missing values using specified method.
        
        Args:
            dataset: List of records
            method: Imputation method ('mean', 'median', 'mode', 'zero')
        
        Returns:
            Imputed dataset
        """
        if not dataset:
            return dataset
        
        # Collect numeric values by field
        numeric_fields = {}
        for record in dataset:
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(value)
        
        # Calculate imputation values
        imputation_values = {}
        for field, values in numeric_fields.items():
            if method == "mean":
                imputation_values[field] = sum(values) / len(values) if values else 0
            elif method == "median":
                sorted_values = sorted(values)
                n = len(sorted_values)
                imputation_values[field] = sorted_values[n // 2] if n > 0 else 0
            elif method == "zero":
                imputation_values[field] = 0
            else:
                imputation_values[field] = 0
        
        # Impute missing values
        imputed = []
        for record in dataset:
            imputed_record = {}
            for key, value in record.items():
                if value in self.missing_values or value is None:
                    imputed_record[key] = imputation_values.get(key, "")
                else:
                    imputed_record[key] = value
            imputed.append(imputed_record)
        
        return imputed
