"""
Vector Exporter for Vector Embeddings Use Cases

Exports vector embeddings and vector data to various formats
for use with vector stores and embedding systems.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json
import numpy as np

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory, write_json_file


class VectorExporter:
    """
    Vector exporter for embeddings and vector data.
    
    • Exports vector embeddings to various formats
    • Supports JSON, NumPy, and binary formats
    • Handles metadata and document associations
    • Generates vector store import formats
    • Supports batch vector export
    • Handles multi-dimensional vectors
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize vector exporter.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - format: Export format ('json', 'numpy', 'binary', 'faiss')
                - include_metadata: Include metadata (default: True)
                - include_text: Include original text (default: True)
        """
        self.logger = get_logger("vector_exporter")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.format = self.config.get("format", "json")
        self.include_metadata = self.config.get("include_metadata", True)
        self.include_text = self.config.get("include_text", True)
    
    def export(
        self,
        vectors: Union[List[Dict[str, Any]], Dict[str, Any]],
        file_path: Union[str, Path],
        format: Optional[str] = None,
        **options
    ) -> None:
        """
        Export vectors to file.
        
        Args:
            vectors: List of vector dictionaries or dict with 'vectors' key
            file_path: Output file path
            format: Export format ('json', 'numpy', 'binary', 'faiss')
            **options: Additional options
        """
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        export_format = format or self.format
        
        # Normalize input
        if isinstance(vectors, dict):
            vector_list = vectors.get("vectors", [])
            metadata = vectors.get("metadata", {})
        else:
            vector_list = vectors
            metadata = {}
        
        if export_format == "json":
            self._export_json(vector_list, file_path, metadata, **options)
        elif export_format == "numpy":
            self._export_numpy(vector_list, file_path, **options)
        elif export_format == "binary":
            self._export_binary(vector_list, file_path, **options)
        elif export_format == "faiss":
            self._export_faiss(vector_list, file_path, **options)
        else:
            raise ValidationError(f"Unsupported vector format: {export_format}")
        
        self.logger.info(f"Exported vectors ({export_format}) to: {file_path}")
    
    def export_embeddings(
        self,
        embeddings: List[Dict[str, Any]],
        file_path: Union[str, Path],
        format: Optional[str] = None,
        **options
    ) -> None:
        """
        Export embeddings to file.
        
        Args:
            embeddings: List of embedding dictionaries with 'id', 'vector', 'text', etc.
            file_path: Output file path
            format: Export format
            **options: Additional options
        """
        self.export(embeddings, file_path, format=format, **options)
    
    def _export_json(
        self,
        vectors: List[Dict[str, Any]],
        file_path: Path,
        metadata: Dict[str, Any],
        **options
    ) -> None:
        """Export to JSON format."""
        export_data = {
            "vectors": [],
            "metadata": {
                "vector_count": len(vectors),
                "dimension": None,
                **metadata
            }
        }
        
        for vec_data in vectors:
            vector_entry = {
                "id": vec_data.get("id") or vec_data.get("vector_id", ""),
                "vector": vec_data.get("vector") or vec_data.get("embedding", [])
            }
            
            if self.include_text and "text" in vec_data:
                vector_entry["text"] = vec_data["text"]
            
            if self.include_metadata and "metadata" in vec_data:
                vector_entry["metadata"] = vec_data["metadata"]
            
            # Determine dimension
            vector = vector_entry["vector"]
            if isinstance(vector, list) and len(vector) > 0:
                if export_data["metadata"]["dimension"] is None:
                    export_data["metadata"]["dimension"] = len(vector)
            
            export_data["vectors"].append(vector_entry)
        
        write_json_file(export_data, file_path, indent=2)
    
    def _export_numpy(
        self,
        vectors: List[Dict[str, Any]],
        file_path: Path,
        **options
    ) -> None:
        """Export to NumPy format."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy not installed. Install with: pip install numpy")
        
        # Extract vectors
        vector_list = []
        ids = []
        texts = []
        metadata_list = []
        
        for vec_data in vectors:
            vector = vec_data.get("vector") or vec_data.get("embedding", [])
            if vector:
                vector_list.append(vector)
                ids.append(vec_data.get("id") or vec_data.get("vector_id", ""))
                
                if self.include_text and "text" in vec_data:
                    texts.append(vec_data["text"])
                else:
                    texts.append(None)
                
                if self.include_metadata and "metadata" in vec_data:
                    metadata_list.append(vec_data["metadata"])
                else:
                    metadata_list.append(None)
        
        if not vector_list:
            raise ValidationError("No vectors to export")
        
        # Convert to numpy array
        vector_array = np.array(vector_list)
        
        # Save as .npz (compressed numpy format)
        save_dict = {"vectors": vector_array, "ids": np.array(ids)}
        
        if any(t for t in texts if t):
            save_dict["texts"] = np.array(texts, dtype=object)
        
        if any(m for m in metadata_list if m):
            save_dict["metadata"] = np.array(metadata_list, dtype=object)
        
        np.savez_compressed(file_path, **save_dict)
        
        # Also save metadata separately as JSON
        metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
        metadata = {
            "vector_count": len(vector_list),
            "dimension": vector_array.shape[1] if len(vector_array.shape) > 1 else vector_array.shape[0],
            "shape": list(vector_array.shape)
        }
        write_json_file(metadata, metadata_file)
    
    def _export_binary(
        self,
        vectors: List[Dict[str, Any]],
        file_path: Path,
        **options
    ) -> None:
        """Export to binary format."""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy not installed. Install with: pip install numpy")
        
        # Extract vectors
        vector_list = []
        for vec_data in vectors:
            vector = vec_data.get("vector") or vec_data.get("embedding", [])
            if vector:
                vector_list.append(vector)
        
        if not vector_list:
            raise ValidationError("No vectors to export")
        
        # Convert to numpy array
        vector_array = np.array(vector_list, dtype=np.float32)
        
        # Save as binary
        vector_array.tofile(file_path)
        
        # Save metadata
        metadata_file = file_path.parent / f"{file_path.stem}_metadata.json"
        metadata = {
            "vector_count": len(vector_list),
            "dimension": vector_array.shape[1] if len(vector_array.shape) > 1 else vector_array.shape[0],
            "shape": list(vector_array.shape),
            "dtype": "float32"
        }
        write_json_file(metadata, metadata_file)
    
    def _export_faiss(
        self,
        vectors: List[Dict[str, Any]],
        file_path: Path,
        **options
    ) -> None:
        """Export to FAISS index format."""
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu or faiss-gpu")
        
        # Extract vectors
        vector_list = []
        ids = []
        
        for vec_data in vectors:
            vector = vec_data.get("vector") or vec_data.get("embedding", [])
            if vector:
                vector_list.append(vector)
                ids.append(vec_data.get("id") or vec_data.get("vector_id", ""))
        
        if not vector_list:
            raise ValidationError("No vectors to export")
        
        # Convert to numpy array
        vector_array = np.array(vector_list, dtype=np.float32)
        
        # Create FAISS index
        dimension = vector_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Normalize vectors if requested
        if options.get("normalize", False):
            faiss.normalize_L2(vector_array)
        
        # Add vectors to index
        index.add(vector_array)
        
        # Save index
        faiss.write_index(index, str(file_path))
        
        # Save ID mapping
        id_file = file_path.parent / f"{file_path.stem}_ids.json"
        id_mapping = {
            "ids": ids,
            "vector_count": len(ids),
            "dimension": dimension
        }
        write_json_file(id_mapping, id_file)
    
    def export_for_vector_store(
        self,
        vectors: List[Dict[str, Any]],
        file_path: Union[str, Path],
        vector_store_type: str = "pinecone",
        **options
    ) -> None:
        """
        Export in format suitable for specific vector store.
        
        Args:
            vectors: List of vector dictionaries
            file_path: Output file path
            vector_store_type: Vector store type ('pinecone', 'weaviate', 'qdrant', 'faiss')
            **options: Additional options
        """
        if vector_store_type == "pinecone":
            self._export_pinecone_format(vectors, file_path, **options)
        elif vector_store_type == "weaviate":
            self._export_weaviate_format(vectors, file_path, **options)
        elif vector_store_type == "qdrant":
            self._export_qdrant_format(vectors, file_path, **options)
        elif vector_store_type == "faiss":
            self._export_faiss(vectors, file_path, **options)
        else:
            # Default to JSON
            self._export_json(vectors, Path(file_path), {}, **options)
    
    def _export_pinecone_format(
        self,
        vectors: List[Dict[str, Any]],
        file_path: Path,
        **options
    ) -> None:
        """Export in Pinecone format."""
        pinecone_data = []
        
        for vec_data in vectors:
            vector_id = vec_data.get("id") or vec_data.get("vector_id", "")
            vector = vec_data.get("vector") or vec_data.get("embedding", [])
            metadata = vec_data.get("metadata", {})
            
            if "text" in vec_data and self.include_text:
                metadata["text"] = vec_data["text"]
            
            pinecone_data.append({
                "id": vector_id,
                "values": vector,
                "metadata": metadata
            })
        
        export_data = {"vectors": pinecone_data}
        write_json_file(export_data, file_path, indent=2)
    
    def _export_weaviate_format(
        self,
        vectors: List[Dict[str, Any]],
        file_path: Path,
        **options
    ) -> None:
        """Export in Weaviate format."""
        weaviate_data = []
        
        for vec_data in vectors:
            vector_id = vec_data.get("id") or vec_data.get("vector_id", "")
            vector = vec_data.get("vector") or vec_data.get("embedding", [])
            
            weaviate_obj = {
                "id": vector_id,
                "vector": vector
            }
            
            if "text" in vec_data and self.include_text:
                weaviate_obj["text"] = vec_data["text"]
            
            if "metadata" in vec_data and self.include_metadata:
                weaviate_obj.update(vec_data["metadata"])
            
            weaviate_data.append(weaviate_obj)
        
        export_data = {"objects": weaviate_data}
        write_json_file(export_data, file_path, indent=2)
    
    def _export_qdrant_format(
        self,
        vectors: List[Dict[str, Any]],
        file_path: Path,
        **options
    ) -> None:
        """Export in Qdrant format."""
        qdrant_data = []
        
        for vec_data in vectors:
            vector_id = vec_data.get("id") or vec_data.get("vector_id", "")
            vector = vec_data.get("vector") or vec_data.get("embedding", [])
            payload = vec_data.get("metadata", {})
            
            if "text" in vec_data and self.include_text:
                payload["text"] = vec_data["text"]
            
            qdrant_data.append({
                "id": vector_id,
                "vector": vector,
                "payload": payload
            })
        
        export_data = {"points": qdrant_data}
        write_json_file(export_data, file_path, indent=2)

