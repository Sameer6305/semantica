"""
Chunk validator for Semantica framework.

This module provides quality validation for document chunks
to ensure proper splitting and data integrity.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .semantic_chunker import Chunk


@dataclass
class ValidationResult:
    """Validation result representation."""
    
    valid: bool
    score: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class ChunkValidator:
    """Chunk validator for quality assessment."""
    
    def __init__(self, **config):
        """
        Initialize chunk validator.
        
        Args:
            **config: Configuration options:
                - min_size: Minimum chunk size in characters (default: 10)
                - max_size: Maximum chunk size in characters (default: 10000)
                - min_score: Minimum quality score (default: 0.5)
        """
        self.logger = get_logger("chunk_validator")
        self.config = config
        
        self.min_size = config.get("min_size", 10)
        self.max_size = config.get("max_size", 10000)
        self.min_score = config.get("min_score", 0.5)
    
    def validate(self, chunk: Chunk, **options) -> ValidationResult:
        """
        Validate a single chunk.
        
        Args:
            chunk: Chunk to validate
            **options: Validation options
            
        Returns:
            ValidationResult: Validation result
        """
        errors = []
        warnings = []
        metrics = {}
        
        # Size validation
        chunk_size = len(chunk.text)
        metrics["size"] = chunk_size
        
        if chunk_size < self.min_size:
            errors.append(f"Chunk too small: {chunk_size} < {self.min_size}")
        
        if chunk_size > self.max_size:
            errors.append(f"Chunk too large: {chunk_size} > {self.max_size}")
        
        # Content validation
        if not chunk.text.strip():
            errors.append("Chunk is empty or whitespace only")
        
        # Coherence validation
        coherence_score = self._check_coherence(chunk.text)
        metrics["coherence"] = coherence_score
        
        if coherence_score < 0.3:
            warnings.append(f"Low semantic coherence: {coherence_score:.2f}")
        
        # Structure validation
        structure_score = self._check_structure(chunk.text)
        metrics["structure"] = structure_score
        
        # Calculate overall score
        score = self._calculate_score(chunk_size, coherence_score, structure_score, errors)
        metrics["overall_score"] = score
        
        valid = len(errors) == 0 and score >= self.min_score
        
        return ValidationResult(
            valid=valid,
            score=score,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
    
    def validate_batch(self, chunks: List[Chunk], **options) -> Dict[str, Any]:
        """
        Validate multiple chunks.
        
        Args:
            chunks: List of chunks to validate
            **options: Validation options
            
        Returns:
            dict: Batch validation results
        """
        results = []
        valid_count = 0
        total_score = 0.0
        
        for chunk in chunks:
            result = self.validate(chunk, **options)
            results.append(result)
            
            if result.valid:
                valid_count += 1
            total_score += result.score
        
        avg_score = total_score / len(chunks) if chunks else 0.0
        
        return {
            "total_chunks": len(chunks),
            "valid_chunks": valid_count,
            "invalid_chunks": len(chunks) - valid_count,
            "average_score": avg_score,
            "results": results,
            "summary": {
                "validity_rate": valid_count / len(chunks) if chunks else 0.0,
                "average_coherence": sum(r.metrics.get("coherence", 0) for r in results) / len(results) if results else 0.0,
                "average_structure": sum(r.metrics.get("structure", 0) for r in results) / len(results) if results else 0.0
            }
        }
    
    def _check_coherence(self, text: str) -> float:
        """
        Check semantic coherence of text.
        
        Args:
            text: Text to check
            
        Returns:
            float: Coherence score (0-1)
        """
        if not text:
            return 0.0
        
        score = 1.0
        
        # Check sentence completeness
        sentences = text.split('.')
        incomplete_sentences = sum(1 for s in sentences if s.strip() and not s.strip()[-1] in '.!?')
        if sentences:
            completeness = 1.0 - (incomplete_sentences / len(sentences))
            score *= completeness
        
        # Check for very short fragments
        words = text.split()
        if len(words) < 3:
            score *= 0.5
        
        # Check word repetition (high repetition = lower coherence)
        if len(words) > 10:
            unique_words = len(set(word.lower() for word in words))
            diversity = unique_words / len(words)
            score *= (0.5 + diversity * 0.5)
        
        return max(0.0, min(1.0, score))
    
    def _check_structure(self, text: str) -> float:
        """
        Check text structure quality.
        
        Args:
            text: Text to check
            
        Returns:
            float: Structure score (0-1)
        """
        if not text:
            return 0.0
        
        score = 1.0
        
        # Check for proper capitalization
        first_char = text.strip()[0] if text.strip() else ''
        if first_char and first_char.islower():
            score *= 0.9
        
        # Check for balanced punctuation
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 1:
            # More sentences = better structure
            structure_factor = min(1.0, len(sentences) / 5.0)
            score *= (0.7 + structure_factor * 0.3)
        
        # Check for whitespace issues
        if '  ' in text or '\t\t' in text:
            score *= 0.9
        
        return max(0.0, min(1.0, score))
    
    def _calculate_score(
        self,
        size: int,
        coherence: float,
        structure: float,
        errors: List[str]
    ) -> float:
        """
        Calculate overall validation score.
        
        Args:
            size: Chunk size
            coherence: Coherence score
            structure: Structure score
            errors: List of errors
            
        Returns:
            float: Overall score (0-1)
        """
        # Start with base score
        score = 1.0
        
        # Apply size penalty
        if size < self.min_size or size > self.max_size:
            score *= 0.5
        
        # Weighted combination of coherence and structure
        score = score * (0.6 * coherence + 0.4 * structure)
        
        # Penalize for errors
        score *= max(0.0, 1.0 - len(errors) * 0.2)
        
        return max(0.0, min(1.0, score))
    
    def filter_valid_chunks(self, chunks: List[Chunk], **options) -> List[Chunk]:
        """
        Filter chunks to only include valid ones.
        
        Args:
            chunks: List of chunks to filter
            **options: Validation options
            
        Returns:
            list: List of valid chunks
        """
        valid_chunks = []
        
        for chunk in chunks:
            result = self.validate(chunk, **options)
            if result.valid:
                valid_chunks.append(chunk)
        
        return valid_chunks
