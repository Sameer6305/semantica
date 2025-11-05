"""
Media Content Parsing Module

Handles parsing of media files and content.

Key Features:
    - Image metadata extraction
    - Audio content processing
    - Video metadata analysis
    - Media file information extraction
    - Content type detection

Main Classes:
    - MediaParser: Main media parsing class
    - ImageParser: Image file parser
    - AudioParser: Audio file parser
    - VideoParser: Video file parser
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from .image_parser import ImageParser


class MediaParser:
    """
    Media content parsing handler.
    
    • Parses various media file formats
    • Extracts metadata and properties
    • Processes media content information
    • Handles different media types
    • Supports batch media processing
    • Analyzes media characteristics
    """
    
    def __init__(self, config=None, **kwargs):
        """Initialize media parser."""
        self.logger = get_logger("media_parser")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Initialize parsers
        self.image_parser = ImageParser(**self.config.get("image", {}))
        
        # Supported formats
        self.supported_formats = {
            # Image formats
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.gif': 'image',
            '.bmp': 'image',
            '.tiff': 'image',
            '.webp': 'image',
            # Audio formats (would need additional libraries)
            '.mp3': 'audio',
            '.wav': 'audio',
            '.flac': 'audio',
            '.aac': 'audio',
            # Video formats (would need additional libraries)
            '.mp4': 'video',
            '.avi': 'video',
            '.mov': 'video',
            '.mkv': 'video',
            '.webm': 'video'
        }
    
    def parse_media(self, file_path: Union[str, Path], media_type: Optional[str] = None, **options) -> Dict[str, Any]:
        """
        Parse media file of any supported type.
        
        Args:
            file_path: Path to media file
            media_type: Media type (auto-detected if None)
            **options: Parsing options
            
        Returns:
            dict: Parsed media data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"Media file not found: {file_path}")
        
        # Detect media type if not specified
        if media_type is None:
            media_type = self._detect_media_type(file_path)
        
        if media_type == "image":
            return self.image_parser.parse(file_path, **options)
        elif media_type == "audio":
            return self._parse_audio(file_path, **options)
        elif media_type == "video":
            return self._parse_video(file_path, **options)
        else:
            raise ValidationError(f"Unsupported media type: {media_type}")
    
    def extract_metadata(self, file_path: Union[str, Path], media_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract metadata from media file.
        
        Args:
            file_path: Path to media file
            media_type: Media type (auto-detected if None)
            
        Returns:
            dict: Media metadata
        """
        file_path = Path(file_path)
        
        if media_type is None:
            media_type = self._detect_media_type(file_path)
        
        if media_type == "image":
            return self.image_parser.extract_metadata(file_path).__dict__
        elif media_type == "audio":
            return self._extract_audio_metadata(file_path)
        elif media_type == "video":
            return self._extract_video_metadata(file_path)
        else:
            return {}
    
    def _detect_media_type(self, file_path: Path) -> str:
        """Detect media type from file extension."""
        suffix = file_path.suffix.lower()
        return self.supported_formats.get(suffix, "unknown")
    
    def _parse_audio(self, file_path: Path, **options) -> Dict[str, Any]:
        """Parse audio file."""
        metadata = self._extract_audio_metadata(file_path)
        
        return {
            "metadata": metadata,
            "type": "audio"
        }
    
    def _parse_video(self, file_path: Path, **options) -> Dict[str, Any]:
        """Parse video file."""
        metadata = self._extract_video_metadata(file_path)
        
        return {
            "metadata": metadata,
            "type": "video"
        }
    
    def _extract_audio_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract audio metadata."""
        metadata = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "format": file_path.suffix.lower()
        }
        
        # Try to extract metadata using mutagen if available
        try:
            from mutagen import File
            audio_file = File(str(file_path))
            if audio_file:
                metadata.update({
                    "title": audio_file.get("TIT2", [None])[0] or audio_file.get("TITLE", [None])[0],
                    "artist": audio_file.get("TPE1", [None])[0] or audio_file.get("ARTIST", [None])[0],
                    "album": audio_file.get("TALB", [None])[0] or audio_file.get("ALBUM", [None])[0],
                    "duration": audio_file.info.length if hasattr(audio_file.info, 'length') else None,
                    "bitrate": audio_file.info.bitrate if hasattr(audio_file.info, 'bitrate') else None,
                    "sample_rate": audio_file.info.sample_rate if hasattr(audio_file.info, 'sample_rate') else None
                })
        except ImportError:
            self.logger.warning("mutagen not available for audio metadata extraction")
        except Exception as e:
            self.logger.warning(f"Failed to extract audio metadata: {e}")
        
        return metadata
    
    def _extract_video_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract video metadata."""
        metadata = {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "format": file_path.suffix.lower()
        }
        
        # Try to extract metadata using ffmpeg if available
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', str(file_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                import json
                video_data = json.loads(result.stdout)
                
                if 'streams' in video_data:
                    for stream in video_data['streams']:
                        if stream.get('codec_type') == 'video':
                            metadata.update({
                                "width": stream.get("width"),
                                "height": stream.get("height"),
                                "duration": float(video_data.get('format', {}).get('duration', 0)),
                                "codec": stream.get("codec_name"),
                                "fps": eval(stream.get("r_frame_rate", "0/1")) if stream.get("r_frame_rate") else None
                            })
                            break
        except FileNotFoundError:
            self.logger.warning("ffprobe not available for video metadata extraction")
        except Exception as e:
            self.logger.warning(f"Failed to extract video metadata: {e}")
        
        return metadata
