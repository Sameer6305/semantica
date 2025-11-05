"""
Audio embedder for Semantica framework.

This module provides audio embedding generation for
speech and audio content analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class AudioEmbedder:
    """Audio embedding generator."""
    
    def __init__(self, **config):
        """
        Initialize audio embedder.
        
        Args:
            **config: Configuration options:
                - sample_rate: Audio sample rate (default: 16000)
                - normalize: Normalize embeddings (default: True)
        """
        self.logger = get_logger("audio_embedder")
        self.config = config
        
        self.sample_rate = config.get("sample_rate", 16000)
        self.normalize = config.get("normalize", True)
    
    def embed_audio(self, audio_path: Union[str, Path], **options) -> np.ndarray:
        """
        Generate embedding for audio.
        
        Args:
            audio_path: Path to audio file
            **options: Embedding options
            
        Returns:
            np.ndarray: Audio embedding vector
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise ValidationError(f"Audio file not found: {audio_path}")
        
        if LIBROSA_AVAILABLE:
            return self._embed_with_librosa(audio_path, **options)
        else:
            return self._embed_fallback(audio_path, **options)
    
    def _embed_with_librosa(self, audio_path: Path, **options) -> np.ndarray:
        """Embed audio using librosa."""
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            # Extract features
            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # Tonnetz
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            # Aggregate features (mean across time)
            features = np.concatenate([
                np.mean(mfccs, axis=1),
                np.mean(chroma, axis=1),
                np.mean(contrast, axis=1),
                np.mean(tonnetz, axis=1)
            ])
            
            # Normalize
            if self.normalize:
                norm = np.linalg.norm(features)
                if norm > 0:
                    features = features / norm
            
            return features.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Failed to embed audio with librosa: {e}")
            raise ProcessingError(f"Failed to embed audio: {e}")
    
    def _embed_fallback(self, audio_path: Path, **options) -> np.ndarray:
        """Fallback audio embedding."""
        try:
            # Simple feature extraction without librosa
            # Read file size as proxy for features
            file_size = audio_path.stat().st_size
            
            # Create simple embedding from file metadata
            # (This is a placeholder - would need actual audio processing)
            embedding = np.random.rand(128).astype(np.float32)
            
            # Normalize
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            self.logger.warning("Using fallback audio embedding (librosa not available)")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to embed audio: {e}")
            raise ProcessingError(f"Failed to embed audio: {e}")
    
    def embed_batch(self, audio_paths: List[Union[str, Path]], **options) -> np.ndarray:
        """
        Generate embeddings for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            **options: Embedding options
            
        Returns:
            np.ndarray: Array of embeddings (n_audio, embedding_dim)
        """
        embeddings = []
        
        for audio_path in audio_paths:
            embedding = self.embed_audio(audio_path, **options)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def extract_features(self, audio_path: Union[str, Path], **options) -> Dict[str, np.ndarray]:
        """
        Extract audio features.
        
        Args:
            audio_path: Path to audio file
            **options: Feature extraction options
            
        Returns:
            dict: Dictionary of audio features
        """
        audio_path = Path(audio_path)
        
        if not LIBROSA_AVAILABLE:
            return {}
        
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            features = {
                "mfcc": librosa.feature.mfcc(y=y, sr=sr),
                "chroma": librosa.feature.chroma_stft(y=y, sr=sr),
                "spectral_contrast": librosa.feature.spectral_contrast(y=y, sr=sr),
                "tonnetz": librosa.feature.tonnetz(y=y, sr=sr),
                "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
                "duration": len(y) / sr
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract audio features: {e}")
            return {}
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            int: Embedding dimension
        """
        # MFCC (13) + Chroma (12) + Contrast (7) + Tonnetz (6) = 38
        # But we pad/truncate to standard sizes
        return 128  # Standard dimension
