"""
Similarity Calculator Module

This module provides comprehensive similarity calculation capabilities for the
Semantica framework, computing semantic similarity between entities using multiple
metrics including string similarity, property similarity, relationship similarity,
and embedding similarity.

Algorithms Used:
    - Levenshtein Distance: Dynamic programming algorithm for edit distance calculation
    - Jaro Similarity: Character-based similarity with match window algorithm
    - Jaro-Winkler Similarity: Jaro with prefix bonus (up to 4 characters, 0.1 weight)
    - Cosine Similarity: Vector dot product divided by magnitudes for embeddings
    - Jaccard Similarity: Intersection over union for relationship sets
    - Property Matching: Weighted comparison of property values with type-aware matching
    - Multi-factor Aggregation: Weighted sum of similarity components with normalization

Key Features:
    - Multi-factor similarity calculation (string, property, relationship, embedding)
    - Multiple string similarity algorithms (Levenshtein, Jaro-Winkler, cosine)
    - Weighted aggregation of similarity components with automatic normalization
    - Batch similarity calculation for efficiency (O(n²) optimized)
    - Configurable similarity thresholds and component weights
    - Support for exact matching, fuzzy matching, and semantic matching

Main Classes:
    - SimilarityCalculator: Main similarity calculation engine
    - SimilarityResult: Similarity calculation result with component scores

Example Usage:
    >>> from semantica.deduplication import SimilarityCalculator
    >>> calculator = SimilarityCalculator(
    ...     string_weight=0.4,
    ...     property_weight=0.3,
    ...     embedding_weight=0.3
    ... )
    >>> similarity = calculator.calculate_similarity(entity1, entity2)
    >>> batch_results = calculator.batch_calculate_similarity(entities, threshold=0.7)
    >>> 
    >>> # String similarity methods
    >>> lev_score = calculator.calculate_string_similarity("Apple", "Apple Inc.", method="levenshtein")
    >>> jaro_score = calculator.calculate_string_similarity("Apple", "Apple Inc.", method="jaro_winkler")

Author: Semantica Contributors
License: MIT
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker


@dataclass
class SimilarityResult:
    """Similarity calculation result."""

    score: float
    method: str
    components: Dict[str, float] = None
    metadata: Dict[str, Any] = None


class SimilarityCalculator:
    """
    Similarity calculation engine for entity comparison.

    This class provides comprehensive similarity calculation using multiple factors:
    string similarity, property similarity, relationship similarity, and embedding
    similarity. Results are aggregated using configurable weights.

    Similarity Components:
        - String similarity: Name/identifier comparison using various algorithms
        - Property similarity: Comparison of entity properties
        - Relationship similarity: Comparison of entity relationships
        - Embedding similarity: Semantic similarity using vector embeddings

    Example Usage:
        >>> calculator = SimilarityCalculator(
        ...     string_weight=0.4,
        ...     property_weight=0.3,
        ...     embedding_weight=0.3
        ... )
        >>> result = calculator.calculate_similarity(entity1, entity2)
        >>> print(f"Similarity: {result.score:.2f}")
    """

    def __init__(
        self,
        embedding_weight: float = 0.0,
        string_weight: float = 0.6,
        property_weight: float = 0.2,
        relationship_weight: float = 0.2,
        similarity_threshold: float = 0.7,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize similarity calculator.

        Sets up the calculator with configurable weights for different similarity
        components. Weights are normalized automatically if they don't sum to 1.0.

        Args:
            embedding_weight: Weight for embedding similarity (default: 0.4)
            string_weight: Weight for string similarity (default: 0.3)
            property_weight: Weight for property similarity (default: 0.2)
            relationship_weight: Weight for relationship similarity (default: 0.1)
            similarity_threshold: Default similarity threshold for filtering (default: 0.7)
            config: Configuration dictionary (merged with kwargs)
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("similarity_calculator")

        # Merge configuration
        self.config = config or {}
        self.config.update(kwargs)

        # Component weights (used for weighted aggregation)
        self.embedding_weight = embedding_weight
        self.string_weight = string_weight
        self.property_weight = property_weight
        self.relationship_weight = relationship_weight
        self.similarity_threshold = similarity_threshold

        # Validate weights sum to approximately 1.0
        total_weight = (
            self.embedding_weight
            + self.string_weight
            + self.property_weight
            + self.relationship_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            self.logger.debug(
                f"Weights sum to {total_weight:.2f}, will be normalized during calculation"
            )

        # Initialize progress tracker
        self.progress_tracker = get_progress_tracker()

        self.logger.debug("Similarity calculator initialized")

    def calculate_similarity(
        self, entity1: Dict[str, Any], entity2: Dict[str, Any], **options
    ) -> SimilarityResult:
        """
        Calculate overall similarity between two entities.

        This method computes a comprehensive similarity score by combining multiple
        similarity factors: string similarity, property similarity, relationship
        similarity, and embedding similarity (if available). Results are aggregated
        using configurable weights.

        Similarity Components:
            - String: Name/identifier similarity (Levenshtein, Jaro-Winkler, etc.)
            - Property: Property value similarity
            - Relationship: Relationship overlap (Jaccard similarity)
            - Embedding: Cosine similarity of embeddings (if available)

        Args:
            entity1: First entity dictionary. Should have "name" field and optionally
                    "properties", "relationships", and "embedding" fields.
            entity2: Second entity dictionary (same structure as entity1)
            **options: Additional calculation options (currently unused)

        Returns:
            SimilarityResult object containing:
                - score: Overall similarity score (0.0 to 1.0)
                - method: Calculation method used ("multi_factor")
                - components: Dict of individual component scores
                - metadata: Weights used for aggregation

        Example:
            >>> entity1 = {"name": "Apple Inc.", "type": "Company"}
            >>> entity2 = {"name": "Apple", "type": "Company"}
            >>> result = calculator.calculate_similarity(entity1, entity2)
            >>> print(f"Similarity: {result.score:.2f}")
            >>> print(f"Components: {result.components}")
        """
        # Track similarity calculation
        tracking_id = self.progress_tracker.start_tracking(
            file=None,
            module="deduplication",
            submodule="SimilarityCalculator",
            message="Calculating similarity between entities",
        )

        try:
            components = {}

            # String similarity
            self.progress_tracker.update_tracking(
                tracking_id, message="Calculating string similarity..."
            )
            string_score = self.calculate_string_similarity(
                entity1.get("name", ""), entity2.get("name", "")
            )
            components["string"] = string_score

            # Property similarity
            self.progress_tracker.update_tracking(
                tracking_id, message="Calculating property similarity..."
            )
            property_score = self.calculate_property_similarity(entity1, entity2)
            components["property"] = property_score

            # Relationship similarity
            self.progress_tracker.update_tracking(
                tracking_id, message="Calculating relationship similarity..."
            )
            relationship_score = self.calculate_relationship_similarity(
                entity1, entity2
            )
            components["relationship"] = relationship_score

            # Embedding similarity (if available)
            embedding_score = 0.0
            if "embedding" in entity1 and "embedding" in entity2:
                self.progress_tracker.update_tracking(
                    tracking_id, message="Calculating embedding similarity..."
                )
                embedding_score = self.calculate_embedding_similarity(
                    entity1["embedding"], entity2["embedding"]
                )
                components["embedding"] = embedding_score

            # Weighted aggregation
            self.progress_tracker.update_tracking(
                tracking_id, message="Aggregating similarity scores..."
            )
            weights = {
                "string": self.string_weight,
                "property": self.property_weight,
                "relationship": self.relationship_weight,
                "embedding": self.embedding_weight if embedding_score > 0 else 0.0,
            }

            # Normalize weights
            total_weight = sum(w for k, w in weights.items() if k in components)
            if total_weight > 0:
                weights = {
                    k: w / total_weight for k, w in weights.items() if k in components
                }

            overall_score = sum(
                components.get(key, 0.0) * weight for key, weight in weights.items()
            )

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Similarity score: {overall_score:.2f}",
            )
            return SimilarityResult(
                score=overall_score,
                method="multi_factor",
                components=components,
                metadata={"weights": weights},
            )

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def calculate_string_similarity(
        self, str1: str, str2: str, method: str = "jaro_winkler"
    ) -> float:
        """
        Calculate string similarity between two strings.

        Args:
            str1: First string
            str2: Second string
            method: Similarity method ("levenshtein", "jaro_winkler", "cosine")

        Returns:
            Similarity score (0-1)
        """
        if not str1 or not str2:
            return 0.0

        str1_lower = str1.lower().strip()
        str2_lower = str2.lower().strip()

        if str1_lower == str2_lower:
            return 1.0

        if method == "levenshtein":
            return self._levenshtein_similarity(str1_lower, str2_lower)
        elif method == "jaro_winkler":
            return self._jaro_winkler_similarity(str1_lower, str2_lower)
        elif method == "cosine":
            return self._cosine_similarity(str1_lower, str2_lower)
        else:
            return self._levenshtein_similarity(str1_lower, str2_lower)

    def calculate_property_similarity(
        self, entity1: Dict[str, Any], entity2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity based on entity properties.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Property similarity score (0-1)
        """
        props1 = entity1.get("properties", {})
        props2 = entity2.get("properties", {})

        if not props1 and not props2:
            return 1.0

        all_keys = set(props1.keys()) | set(props2.keys())
        if not all_keys:
            return 0.0

        matches = 0
        total = 0

        for key in all_keys:
            val1 = props1.get(key)
            val2 = props2.get(key)

            if val1 is None or val2 is None:
                # Missing value in one entity is not a mismatch, but lack of evidence
                # Assign neutral score (0.5) instead of 0.0
                matches += 0.5
                total += 1
                continue

            if isinstance(val1, str) and isinstance(val2, str):
                sim = self.calculate_string_similarity(str(val1), str(val2))
                matches += sim
            elif val1 == val2:
                matches += 1.0
            else:
                matches += 0.5  # Partial match

            total += 1

        return matches / total if total > 0 else 0.0

    def calculate_relationship_similarity(
        self, entity1: Dict[str, Any], entity2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity based on relationships.

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Relationship similarity score (0-1)
        """
        def _make_hashable(item):
            if isinstance(item, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in item.items()))
            if isinstance(item, list):
                return tuple(_make_hashable(x) for x in item)
            return item

        rels1 = set(_make_hashable(r) for r in entity1.get("relationships", []))
        rels2 = set(_make_hashable(r) for r in entity2.get("relationships", []))

        if not rels1 and not rels2:
            return 0.5

        if not rels1 or not rels2:
            return 0.0

        intersection = rels1 & rels2
        union = rels1 | rels2

        return len(intersection) / len(union) if union else 0.0

    def calculate_embedding_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity (0-1)
        """
        if len(embedding1) != len(embedding2):
            return 0.0

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(b * b for b in embedding2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        cosine_sim = dot_product / (magnitude1 * magnitude2)

        # Normalize to 0-1 range
        return (cosine_sim + 1) / 2

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein distance-based similarity."""
        if not s1 or not s2:
            return 0.0

        if s1 == s2:
            return 1.0

        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))

        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _jaro_winkler_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro-Winkler similarity."""
        if s1 == s2:
            return 1.0

        jaro = self._jaro_similarity(s1, s2)

        # Winkler prefix bonus
        prefix_len = 0
        min_len = min(len(s1), len(s2))
        for i in range(min_len):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break

        prefix_bonus = min(prefix_len, 4) * 0.1
        return jaro + prefix_bonus * (1 - jaro)

    def _jaro_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaro similarity."""
        if s1 == s2:
            return 1.0

        match_window = max(len(s1), len(s2)) // 2 - 1
        if match_window < 0:
            match_window = 0

        s1_matches = [False] * len(s1)
        s2_matches = [False] * len(s2)

        matches = 0
        transpositions = 0

        # Find matches
        for i in range(len(s1)):
            start = max(0, i - match_window)
            end = min(i + match_window + 1, len(s2))

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len(s1)):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (
            matches / len(s1)
            + matches / len(s2)
            + (matches - transpositions / 2) / matches
        ) / 3.0
        return jaro

    def _cosine_similarity(self, s1: str, s2: str) -> float:
        """Calculate cosine similarity based on character n-grams."""

        # Simple character bigram approach
        def get_bigrams(text):
            return set(text[i : i + 2] for i in range(len(text) - 1))

        bigrams1 = get_bigrams(s1)
        bigrams2 = get_bigrams(s2)

        if not bigrams1 and not bigrams2:
            return 1.0

        intersection = bigrams1 & bigrams2
        union = bigrams1 | bigrams2

        return len(intersection) / len(union) if union else 0.0

    def batch_calculate_similarity(
        self, entities: List[Dict[str, Any]], threshold: Optional[float] = None
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Calculate similarity for all pairs of entities.

        Args:
            entities: List of entities
            threshold: Minimum similarity threshold

        Returns:
            List of (entity1, entity2, similarity) tuples
        """
        threshold = threshold or self.similarity_threshold
        results = []

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                similarity = self.calculate_similarity(entities[i], entities[j])

                if similarity.score >= threshold:
                    results.append((entities[i], entities[j], similarity.score))

        return results
