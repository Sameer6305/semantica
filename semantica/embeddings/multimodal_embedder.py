"""
Multimodal embedder for Semantica framework.

This module provides cross-modal embedding generation
for text, image, and audio content.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder
from .audio_embedder import AudioEmbedder


class MultimodalEmbedder:
    """Multimodal embedding generator."""
    
    def __init__(self, **config):
        """
        Initialize multimodal embedder.
        
        Args:
            **config: Configuration options:
                - align_modalities: Align embeddings across modalities (default: True)
                - normalize: Normalize embeddings (default: True)
        """
        self.logger = get_logger("multimodal_embedder")
        self.config = config
        
        self.align_modalities = config.get("align_modalities", True)
        self.normalize = config.get("normalize", True)
        
        # Initialize modality-specific embedders
        self.text_embedder = TextEmbedder(**config.get("text", {}))
        self.image_embedder = ImageEmbedder(**config.get("image", {}))
        self.audio_embedder = AudioEmbedder(**config.get("audio", {}))
    
    def embed_text(self, text: str, **options) -> np.ndarray:
        """
        Generate embedding for text.
        
        Args:
            text: Input text
            **options: Embedding options
            
        Returns:
            np.ndarray: Text embedding
        """
        return self.text_embedder.embed_text(text, **options)
    
    def embed_image(self, image_path: Union[str, Path], **options) -> np.ndarray:
        """
        Generate embedding for image.
        
        Args:
            image_path: Path to image file
            **options: Embedding options
            
        Returns:
            np.ndarray: Image embedding
        """
        return self.image_embedder.embed_image(image_path, **options)
    
    def embed_audio(self, audio_path: Union[str, Path], **options) -> np.ndarray:
        """
        Generate embedding for audio.
        
        Args:
            audio_path: Path to audio file
            **options: Embedding options
            
        Returns:
            np.ndarray: Audio embedding
        """
        return self.audio_embedder.embed_audio(audio_path, **options)
    
    def embed_multimodal(
        self,
        text: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        audio_path: Optional[Union[str, Path]] = None,
        **options
    ) -> np.ndarray:
        """
        Generate multimodal embedding from multiple inputs.
        
        Args:
            text: Input text (optional)
            image_path: Path to image (optional)
            audio_path: Path to audio (optional)
            **options: Embedding options
            
        Returns:
            np.ndarray: Multimodal embedding
        """
        embeddings = []
        
        # Embed each modality
        if text:
            text_emb = self.embed_text(text, **options)
            embeddings.append(text_emb)
        
        if image_path:
            img_emb = self.embed_image(image_path, **options)
            embeddings.append(img_emb)
        
        if audio_path:
            audio_emb = self.embed_audio(audio_path, **options)
            embeddings.append(audio_emb)
        
        if not embeddings:
            raise ProcessingError("At least one input (text, image, or audio) is required")
        
        # Combine embeddings
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Align dimensions if needed
        if self.align_modalities:
            embeddings = self._align_dimensions(embeddings)
        
        # Concatenate or average
        if options.get("combine_method", "concat") == "concat":
            combined = np.concatenate(embeddings)
        else:
            # Average
            combined = np.mean(embeddings, axis=0)
        
        # Normalize if requested
        if self.normalize:
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
        
        return combined
    
    def _align_dimensions(self, embeddings: List[np.ndarray]) -> List[np.ndarray]:
        """Align embedding dimensions."""
        # Find target dimension (smallest common dimension or average)
        dims = [emb.shape[0] for emb in embeddings]
        target_dim = min(dims)
        
        aligned = []
        for emb in embeddings:
            if emb.shape[0] == target_dim:
                aligned.append(emb)
            elif emb.shape[0] > target_dim:
                # Truncate
                aligned.append(emb[:target_dim])
            else:
                # Pad
                padding = target_dim - emb.shape[0]
                aligned.append(np.pad(emb, (0, padding), 'constant'))
        
        return aligned
    
    def compute_cross_modal_similarity(
        self,
        text: Optional[str] = None,
        image_path: Optional[Union[str, Path]] = None,
        audio_path: Optional[Union[str, Path]] = None,
        **options
    ) -> float:
        """
        Compute similarity between different modalities.
        
        Args:
            text: Input text
            image_path: Path to image
            audio_path: Path to audio
            **options: Similarity options
            
        Returns:
            float: Similarity score (0-1)
        """
        embeddings = {}
        
        if text:
            embeddings['text'] = self.embed_text(text, **options)
        
        if image_path:
            embeddings['image'] = self.embed_image(image_path, **options)
        
        if audio_path:
            embeddings['audio'] = self.embed_audio(audio_path, **options)
        
        if len(embeddings) < 2:
            raise ProcessingError("At least two modalities required for similarity")
        
        # Compute pairwise similarities
        emb_list = list(embeddings.values())
        
        # Align dimensions
        aligned = self._align_dimensions(emb_list)
        
        # Compute cosine similarity
        similarities = []
        for i in range(len(aligned)):
            for j in range(i + 1, len(aligned)):
                similarity = self._cosine_similarity(aligned[i], aligned[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
