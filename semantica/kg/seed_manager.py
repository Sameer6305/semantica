"""
Seed manager for Semantica framework.

This module provides initial data loading and seeding
for knowledge graph construction.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from ..utils.logging import get_logger


class SeedManager:
    """
    Seed manager.
    
    Manages initial data loading and seeding for knowledge graph construction.
    """
    
    def __init__(self, **config):
        """Initialize seed manager."""
        self.logger = get_logger("seed_manager")
        self.config = config
        self.seed_data: List[Dict[str, Any]] = []
    
    def load_seed_data(self, source: str, data: Any) -> None:
        """
        Load seed data.
        
        Args:
            source: Source identifier
            data: Seed data to load
        """
        self.logger.info(f"Loading seed data from source: {source}")
        
        # Normalize data format
        if isinstance(data, list):
            entities = data
        elif isinstance(data, dict):
            entities = data.get("entities", [data])
        else:
            entities = [data]
        
        # Validate and process entities
        processed_entities = []
        for entity in entities:
            if not isinstance(entity, dict):
                self.logger.warning(f"Skipping invalid entity format: {type(entity)}")
                continue
            
            # Ensure entity has required fields
            if "id" not in entity and "entity_id" not in entity:
                # Generate ID if missing
                entity["id"] = f"{source}_{len(processed_entities)}"
            
            # Add source metadata
            entity["source"] = source
            entity["seed_data"] = True
            
            processed_entities.append(entity)
        
        self.seed_data.append({
            "source": source,
            "entities": processed_entities,
            "count": len(processed_entities),
            "timestamp": datetime.now().isoformat()
        })
        
        self.logger.info(f"Loaded {len(processed_entities)} entities from {source}")
    
    def load_from_file(self, file_path: str, source: Optional[str] = None) -> None:
        """
        Load seed data from file.
        
        Args:
            file_path: Path to seed data file
            source: Optional source identifier
        """
        import json
        from pathlib import Path
        
        path = Path(file_path)
        source = source or path.stem
        
        if not path.exists():
            raise FileNotFoundError(f"Seed data file not found: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
            
            self.load_seed_data(source, data)
            
        except Exception as e:
            self.logger.error(f"Error loading seed data from file: {e}")
            raise
    
    def get_seed_data(self) -> List[Dict[str, Any]]:
        """
        Get loaded seed data.
        
        Returns:
            List of seed data entries
        """
        return self.seed_data
    
    def clear_seed_data(self) -> None:
        """Clear all seed data."""
        self.seed_data = []
