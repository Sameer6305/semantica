"""
Image embedder for Semantica framework.

This module provides image embedding generation using
CLIP and other vision models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class ImageEmbedder:
    """Image embedding generator."""
    
    def __init__(self, **config):
        """
        Initialize image embedder.
        
        Args:
            **config: Configuration options:
                - model_name: CLIP model name (default: "ViT-B/32")
                - device: Device ("cpu", "cuda")
                - normalize: Normalize embeddings (default: True)
        """
        self.logger = get_logger("image_embedder")
        self.config = config
        
        self.model_name = config.get("model_name", "ViT-B/32")
        self.device = config.get("device", "cpu")
        self.normalize = config.get("normalize", True)
        
        # Initialize model
        self.model = None
        self.preprocess = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP model."""
        if CLIP_AVAILABLE:
            try:
                self.model, self.preprocess = clip.load(self.model_name, device=self.device)
                self.logger.info(f"Loaded CLIP model: {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load CLIP model: {e}")
        else:
            self.logger.warning("CLIP not available. Image embedding limited.")
    
    def embed_image(self, image_path: Union[str, Path], **options) -> np.ndarray:
        """
        Generate embedding for image.
        
        Args:
            image_path: Path to image file
            **options: Embedding options
            
        Returns:
            np.ndarray: Image embedding vector
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise ValidationError(f"Image file not found: {image_path}")
        
        if self.model and self.preprocess:
            return self._embed_with_clip(image_path, **options)
        else:
            return self._embed_fallback(image_path, **options)
    
    def _embed_with_clip(self, image_path: Path, **options) -> np.ndarray:
        """Embed image using CLIP."""
        try:
            image = Image.open(image_path)
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
            
            embedding = image_features[0].cpu().numpy().astype(np.float32)
            
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to embed image with CLIP: {e}")
            raise ProcessingError(f"Failed to embed image: {e}")
    
    def _embed_fallback(self, image_path: Path, **options) -> np.ndarray:
        """Fallback embedding using image features."""
        try:
            from PIL import Image
            
            image = Image.open(image_path)
            
            # Simple feature extraction
            # Resize to fixed size
            image = image.resize((224, 224))
            
            # Convert to numpy array
            img_array = np.array(image).astype(np.float32)
            
            # Flatten and take sample (for simple embedding)
            flat = img_array.flatten()
            
            # Downsample to fixed dimension (512)
            if len(flat) > 512:
                step = len(flat) // 512
                embedding = flat[::step][:512]
            else:
                embedding = np.pad(flat, (0, 512 - len(flat)), 'constant')[:512]
            
            # Normalize
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            return embedding.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Failed to embed image: {e}")
            raise ProcessingError(f"Failed to embed image: {e}")
    
    def embed_batch(self, image_paths: List[Union[str, Path]], **options) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            image_paths: List of image paths
            **options: Embedding options
            
        Returns:
            np.ndarray: Array of embeddings (n_images, embedding_dim)
        """
        embeddings = []
        
        for image_path in image_paths:
            embedding = self.embed_image(image_path, **options)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            int: Embedding dimension
        """
        if self.model:
            return self.model.visual.output_dim
        else:
            return 512  # Fallback dimension
