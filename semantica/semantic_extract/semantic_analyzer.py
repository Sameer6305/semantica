"""
Semantic Analysis Module

Handles comprehensive semantic analysis of text and data.

Key Features:
    - Semantic similarity analysis
    - Semantic role labeling
    - Semantic clustering and grouping
    - Semantic relationship analysis
    - Semantic quality assessment

Main Classes:
    - SemanticAnalyzer: Main semantic analysis class
    - SimilarityAnalyzer: Semantic similarity analysis
    - RoleLabeler: Semantic role labeling
    - SemanticClusterer: Semantic clustering engine
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger


@dataclass
class SemanticRole:
    """Semantic role representation."""
    
    word: str
    role: str  # agent, patient, theme, location, etc.
    start_char: int
    end_char: int
    confidence: float = 1.0


@dataclass
class SemanticCluster:
    """Semantic cluster representation."""
    
    texts: List[str]
    cluster_id: int
    centroid: Optional[str] = None
    similarity_score: float = 0.0


class SemanticAnalyzer:
    """Comprehensive semantic analysis handler."""
    
    def __init__(self, config=None, **kwargs):
        """Initialize semantic analyzer."""
        self.logger = get_logger("semantic_analyzer")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.similarity_analyzer = SimilarityAnalyzer(**self.config.get("similarity", {}))
        self.role_labeler = RoleLabeler(**self.config.get("role", {}))
        self.semantic_clusterer = SemanticClusterer(**self.config.get("clustering", {}))
    
    def analyze_semantics(self, text: str, **options) -> Dict[str, Any]:
        """
        Perform comprehensive semantic analysis.
        
        Args:
            text: Input text
            **options: Analysis options
            
        Returns:
            dict: Semantic analysis results
        """
        results = {
            "text": text,
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(text.split('.'))
        }
        
        # Semantic role labeling
        if options.get("label_roles", False):
            roles = self.label_semantic_roles(text, **options)
            results["semantic_roles"] = [r.__dict__ for r in roles]
        
        # Semantic features
        results["semantic_features"] = self._extract_features(text)
        
        return results
    
    def calculate_similarity(self, text1: str, text2: str, **options) -> float:
        """
        Calculate semantic similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            **options: Similarity options
            
        Returns:
            float: Similarity score (0-1)
        """
        return self.similarity_analyzer.calculate_similarity(text1, text2, **options)
    
    def label_semantic_roles(self, text: str, **options) -> List[SemanticRole]:
        """
        Label semantic roles in text.
        
        Args:
            text: Input text
            **options: Labeling options
            
        Returns:
            list: List of semantic roles
        """
        return self.role_labeler.label_roles(text, **options)
    
    def cluster_semantically(self, texts: List[str], **options) -> List[SemanticCluster]:
        """
        Perform semantic clustering of texts.
        
        Args:
            texts: List of texts to cluster
            **options: Clustering options
            
        Returns:
            list: List of semantic clusters
        """
        return self.semantic_clusterer.cluster(texts, **options)
    
    def _extract_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text."""
        words = text.lower().split()
        
        return {
            "unique_words": len(set(words)),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "verb_count": sum(1 for w in words if self._is_verb(w)),
            "noun_count": sum(1 for w in words if self._is_noun(w))
        }
    
    def _is_verb(self, word: str) -> bool:
        """Simple verb detection."""
        verb_endings = ["ed", "ing", "es", "s"]
        return any(word.endswith(ending) for ending in verb_endings)
    
    def _is_noun(self, word: str) -> bool:
        """Simple noun detection."""
        noun_endings = ["tion", "sion", "ment", "ness", "ity"]
        return any(word.endswith(ending) for ending in noun_endings)


class SimilarityAnalyzer:
    """Semantic similarity analysis."""
    
    def __init__(self, **config):
        """Initialize similarity analyzer."""
        self.logger = get_logger("similarity_analyzer")
        self.config = config
    
    def calculate_similarity(self, text1: str, text2: str, **options) -> float:
        """
        Calculate semantic similarity between texts.
        
        Args:
            text1: First text
            text2: Second text
            **options: Similarity options
            
        Returns:
            float: Similarity score (0-1)
        """
        method = options.get("method", "jaccard")
        
        if method == "jaccard":
            return self._jaccard_similarity(text1, text2)
        elif method == "cosine":
            return self._cosine_similarity(text1, text2)
        else:
            return self._jaccard_similarity(text1, text2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity (simplified)."""
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Simple word frequency vectors
        all_words = set(words1 + words2)
        vec1 = [words1.count(w) for w in all_words]
        vec2 = [words2.count(w) for w in all_words]
        
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


class RoleLabeler:
    """Semantic role labeling."""
    
    def __init__(self, **config):
        """Initialize role labeler."""
        self.logger = get_logger("role_labeler")
        self.config = config
    
    def label_roles(self, text: str, **options) -> List[SemanticRole]:
        """
        Label semantic roles in text.
        
        Args:
            text: Input text
            **options: Labeling options
            
        Returns:
            list: List of semantic roles
        """
        roles = []
        words = text.split()
        
        # Simple role labeling based on position and patterns
        for i, word in enumerate(words):
            role = self._assign_role(word, i, words, text)
            
            if role:
                # Find position in original text
                start = text.find(word)
                roles.append(SemanticRole(
                    word=word,
                    role=role,
                    start_char=start if start >= 0 else i * 10,
                    end_char=start + len(word) if start >= 0 else (i + 1) * 10,
                    confidence=0.7
                ))
        
        return roles
    
    def _assign_role(self, word: str, position: int, all_words: List[str], text: str) -> Optional[str]:
        """Assign semantic role to word."""
        word_lower = word.lower()
        
        # Agent indicators
        if word_lower in ["i", "we", "he", "she", "they"]:
            return "agent"
        
        # Patient/theme indicators (objects)
        if position > 2 and word[0].isupper():
            return "theme"
        
        # Location indicators
        if word_lower in ["in", "at", "on", "near", "from"]:
            return "location"
        
        return None


class SemanticClusterer:
    """Semantic clustering engine."""
    
    def __init__(self, **config):
        """Initialize semantic clusterer."""
        self.logger = get_logger("semantic_clusterer")
        self.config = config
    
    def cluster(self, texts: List[str], **options) -> List[SemanticCluster]:
        """
        Perform semantic clustering of texts.
        
        Args:
            texts: List of texts to cluster
            **options: Clustering options:
                - num_clusters: Number of clusters (default: auto)
                - similarity_threshold: Minimum similarity for clustering
                
        Returns:
            list: List of semantic clusters
        """
        if not texts:
            return []
        
        similarity_threshold = options.get("similarity_threshold", 0.5)
        similarity_analyzer = SimilarityAnalyzer()
        
        clusters = []
        assigned = set()
        
        cluster_id = 0
        for i, text1 in enumerate(texts):
            if i in assigned:
                continue
            
            cluster_texts = [text1]
            assigned.add(i)
            
            # Find similar texts
            for j, text2 in enumerate(texts[i+1:], start=i+1):
                if j in assigned:
                    continue
                
                similarity = similarity_analyzer.calculate_similarity(text1, text2)
                if similarity >= similarity_threshold:
                    cluster_texts.append(text2)
                    assigned.add(j)
            
            # Create cluster
            cluster = SemanticCluster(
                texts=cluster_texts,
                cluster_id=cluster_id,
                centroid=cluster_texts[0],  # Use first as centroid
                similarity_score=similarity_threshold
            )
            clusters.append(cluster)
            cluster_id += 1
        
        return clusters
