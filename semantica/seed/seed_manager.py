"""
Seed Data Manager for Semantica framework.

Manages seed data for initial knowledge graph construction,
enabling the AI to build on existing verified knowledge.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import csv
from datetime import datetime

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import read_json_file, write_json_file
from ..utils.types import EntityDict, RelationshipDict


@dataclass
class SeedDataSource:
    """Seed data source definition."""
    name: str
    format: str  # csv, json, database, api
    location: Union[str, Path]
    entity_type: Optional[str] = None
    relationship_type: Optional[str] = None
    verified: bool = True
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SeedData:
    """Seed data container."""
    entities: List[EntityDict] = field(default_factory=list)
    relationships: List[RelationshipDict] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SeedDataManager:
    """
    Seed data manager for initial knowledge graph construction.
    
    • Loads verified seed data from multiple sources
    • Creates foundation graph from seed data
    • Validates seed data quality
    • Integrates seed data with extraction results
    • Manages seed data versions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize seed data manager.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options
        """
        self.logger = get_logger("seed_manager")
        self.config = config or {}
        self.config.update(kwargs)
        
        self.sources: Dict[str, SeedDataSource] = {}
        self.seed_data: SeedData = SeedData()
        self.versions: Dict[str, List[str]] = {}  # source_name -> versions
        
    def register_source(
        self,
        name: str,
        format: str,
        location: Union[str, Path],
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        verified: bool = True,
        **metadata
    ) -> bool:
        """
        Register a seed data source.
        
        Args:
            name: Source name
            format: Data format ('csv', 'json', 'database', 'api')
            location: Source location (file path, DB connection, API URL)
            entity_type: Entity type for entities in this source
            relationship_type: Relationship type for relationships in this source
            verified: Whether data is verified
            **metadata: Additional metadata
            
        Returns:
            True if registration successful
        """
        if name in self.sources:
            self.logger.warning(f"Source '{name}' already registered, updating")
        
        source = SeedDataSource(
            name=name,
            format=format,
            location=location,
            entity_type=entity_type,
            relationship_type=relationship_type,
            verified=verified,
            metadata=metadata
        )
        
        self.sources[name] = source
        
        # Track versions
        if name not in self.versions:
            self.versions[name] = []
        
        self.logger.info(f"Registered seed data source: {name} ({format})")
        return True
    
    def load_from_csv(
        self,
        file_path: Union[str, Path],
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        source_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load seed data from CSV file.
        
        Args:
            file_path: Path to CSV file
            entity_type: Entity type (if applicable)
            relationship_type: Relationship type (if applicable)
            source_name: Source name for tracking
            
        Returns:
            List of loaded data records
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ProcessingError(f"CSV file not found: {file_path}")
        
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean up row data
                    record = {k: v for k, v in row.items() if v}
                    
                    if entity_type:
                        record['entity_type'] = entity_type
                    if relationship_type:
                        record['relationship_type'] = relationship_type
                    if source_name:
                        record['source'] = source_name
                    
                    records.append(record)
            
            self.logger.info(f"Loaded {len(records)} records from CSV: {file_path}")
            return records
            
        except Exception as e:
            raise ProcessingError(f"Failed to load CSV: {e}") from e
    
    def load_from_json(
        self,
        file_path: Union[str, Path],
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        source_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load seed data from JSON file.
        
        Args:
            file_path: Path to JSON file
            entity_type: Entity type (if applicable)
            relationship_type: Relationship type (if applicable)
            source_name: Source name for tracking
            
        Returns:
            List of loaded data records
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ProcessingError(f"JSON file not found: {file_path}")
        
        try:
            data = read_json_file(file_path)
            
            # Handle different JSON structures
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Try common keys
                if 'entities' in data:
                    records = data['entities']
                elif 'data' in data:
                    records = data['data']
                elif 'records' in data:
                    records = data['records']
                else:
                    records = [data]
            else:
                records = []
            
            # Add metadata
            for record in records:
                if entity_type and 'entity_type' not in record:
                    record['entity_type'] = entity_type
                if relationship_type and 'relationship_type' not in record:
                    record['relationship_type'] = relationship_type
                if source_name and 'source' not in record:
                    record['source'] = source_name
            
            self.logger.info(f"Loaded {len(records)} records from JSON: {file_path}")
            return records
            
        except Exception as e:
            raise ProcessingError(f"Failed to load JSON: {e}") from e
    
    def load_from_database(
        self,
        connection_string: str,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        source_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load seed data from database.
        
        Args:
            connection_string: Database connection string
            query: Optional SQL query
            table_name: Optional table name (if no query provided)
            entity_type: Entity type (if applicable)
            relationship_type: Relationship type (if applicable)
            source_name: Source name for tracking
            
        Returns:
            List of loaded data records
        """
        try:
            from ..ingest.db_ingestor import DBIngestor
            
            # Initialize DB ingestor
            db_ingestor = DBIngestor(config={"connection_string": connection_string})
            
            # Execute query or export table
            if query:
                # Execute custom query
                result = db_ingestor.execute_query(query)
                records = result if isinstance(result, list) else [result]
            elif table_name:
                # Export table
                table_data = db_ingestor.export_table(table_name)
                records = table_data.rows if hasattr(table_data, 'rows') else []
            else:
                raise ProcessingError("Either 'query' or 'table_name' must be provided")
            
            # Add metadata
            for record in records:
                if entity_type and 'entity_type' not in record:
                    record['entity_type'] = entity_type
                if relationship_type and 'relationship_type' not in record:
                    record['relationship_type'] = relationship_type
                if source_name and 'source' not in record:
                    record['source'] = source_name
            
            self.logger.info(f"Loaded {len(records)} records from database")
            return records
            
        except ImportError:
            raise ProcessingError("Database ingestion module not available. Install required dependencies.")
        except Exception as e:
            raise ProcessingError(f"Failed to load from database: {e}") from e
    
    def load_from_api(
        self,
        api_url: str,
        endpoint: Optional[str] = None,
        entity_type: Optional[str] = None,
        relationship_type: Optional[str] = None,
        source_name: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load seed data from API.
        
        Args:
            api_url: Base API URL
            endpoint: API endpoint path
            entity_type: Entity type (if applicable)
            relationship_type: Relationship type (if applicable)
            source_name: Source name for tracking
            api_key: Optional API key
            headers: Optional request headers
            
        Returns:
            List of loaded data records
        """
        try:
            import requests
            
            # Build full URL
            if endpoint:
                full_url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
            else:
                full_url = api_url
            
            # Prepare headers
            request_headers = headers or {}
            if api_key:
                request_headers["Authorization"] = f"Bearer {api_key}"
            
            # Make API request
            response = requests.get(full_url, headers=request_headers, timeout=30)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Handle different response structures
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                # Try common keys
                if 'entities' in data:
                    records = data['entities']
                elif 'data' in data:
                    records = data['data']
                elif 'results' in data:
                    records = data['results']
                elif 'items' in data:
                    records = data['items']
                else:
                    records = [data]
            else:
                records = []
            
            # Add metadata
            for record in records:
                if not isinstance(record, dict):
                    continue
                if entity_type and 'entity_type' not in record:
                    record['entity_type'] = entity_type
                if relationship_type and 'relationship_type' not in record:
                    record['relationship_type'] = relationship_type
                if source_name and 'source' not in record:
                    record['source'] = source_name
            
            self.logger.info(f"Loaded {len(records)} records from API: {full_url}")
            return records
            
        except ImportError:
            raise ProcessingError("requests library not available. Install with: pip install requests")
        except Exception as e:
            raise ProcessingError(f"Failed to load from API: {e}") from e
    
    def load_source(self, source_name: str) -> List[Dict[str, Any]]:
        """
        Load data from registered source.
        
        Args:
            source_name: Source name
            
        Returns:
            List of loaded data records
        """
        if source_name not in self.sources:
            raise ProcessingError(f"Source '{source_name}' not registered")
        
        source = self.sources[source_name]
        
        if source.format == "csv":
            return self.load_from_csv(
                source.location,
                entity_type=source.entity_type,
                relationship_type=source.relationship_type,
                source_name=source_name
            )
        elif source.format == "json":
            return self.load_from_json(
                source.location,
                entity_type=source.entity_type,
                relationship_type=source.relationship_type,
                source_name=source_name
            )
        elif source.format == "database":
            return self.load_from_database(
                source.location,
                entity_type=source.entity_type,
                relationship_type=source.relationship_type,
                source_name=source_name
            )
        elif source.format == "api":
            return self.load_from_api(
                source.location,
                entity_type=source.entity_type,
                relationship_type=source.relationship_type,
                source_name=source_name
            )
        else:
            raise ProcessingError(f"Unsupported source format: {source.format}")
    
    def create_foundation_graph(
        self,
        schema_template: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Create foundation graph from seed data.
        
        Args:
            schema_template: Optional schema template for validation
            
        Returns:
            Foundation graph dictionary with entities and relationships
        """
        foundation = {
            "entities": [],
            "relationships": [],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "source_count": len(self.sources),
                "verified": True
            }
        }
        
        # Load data from all sources
        for source_name in self.sources:
            try:
                records = self.load_source(source_name)
                
                for record in records:
                    # Extract entities
                    if 'entity_type' in record or 'id' in record:
                        entity = self._record_to_entity(record)
                        if entity:
                            foundation["entities"].append(entity)
                    
                    # Extract relationships
                    if 'relationship_type' in record or ('source_id' in record and 'target_id' in record):
                        relationship = self._record_to_relationship(record)
                        if relationship:
                            foundation["relationships"].append(relationship)
            
            except Exception as e:
                self.logger.warning(f"Failed to load source '{source_name}': {e}")
        
        # Validate against schema template if provided
        if schema_template:
            foundation = self._validate_against_template(foundation, schema_template)
        
        self.logger.info(
            f"Created foundation graph: {len(foundation['entities'])} entities, "
            f"{len(foundation['relationships'])} relationships"
        )
        
        return foundation
    
    def integrate_with_extracted(
        self,
        seed_data: Dict[str, Any],
        extracted_data: Dict[str, Any],
        merge_strategy: str = "seed_first"
    ) -> Dict[str, Any]:
        """
        Integrate seed data with extracted data.
        
        Args:
            seed_data: Seed data dictionary
            extracted_data: Extracted data dictionary
            merge_strategy: Merge strategy ('seed_first', 'extracted_first', 'merge')
            
        Returns:
            Integrated data dictionary
        """
        integrated = {
            "entities": [],
            "relationships": [],
            "metadata": {
                "merged_at": datetime.now().isoformat(),
                "merge_strategy": merge_strategy,
                "seed_count": len(seed_data.get("entities", [])),
                "extracted_count": len(extracted_data.get("entities", []))
            }
        }
        
        seed_entities = {e.get("id"): e for e in seed_data.get("entities", [])}
        extracted_entities = {e.get("id"): e for e in extracted_data.get("entities", [])}
        
        # Merge entities based on strategy
        if merge_strategy == "seed_first":
            # Seed data takes precedence
            integrated["entities"] = list(seed_entities.values())
            for eid, entity in extracted_entities.items():
                if eid not in seed_entities:
                    integrated["entities"].append(entity)
        
        elif merge_strategy == "extracted_first":
            # Extracted data takes precedence
            integrated["entities"] = list(extracted_entities.values())
            for eid, entity in seed_entities.items():
                if eid not in extracted_entities:
                    integrated["entities"].append(entity)
        
        elif merge_strategy == "merge":
            # Merge properties, seed takes precedence for conflicts
            all_entity_ids = set(seed_entities.keys()) | set(extracted_entities.keys())
            for eid in all_entity_ids:
                seed_entity = seed_entities.get(eid, {})
                extracted_entity = extracted_entities.get(eid, {})
                
                merged = {**extracted_entity, **seed_entity}
                integrated["entities"].append(merged)
        
        # Merge relationships
        seed_rels = {(r.get("source_id"), r.get("target_id"), r.get("type")): r 
                    for r in seed_data.get("relationships", [])}
        extracted_rels = {(r.get("source_id"), r.get("target_id"), r.get("type")): r 
                         for r in extracted_data.get("relationships", [])}
        
        if merge_strategy == "seed_first":
            integrated["relationships"] = list(seed_rels.values())
            for key, rel in extracted_rels.items():
                if key not in seed_rels:
                    integrated["relationships"].append(rel)
        else:
            integrated["relationships"] = list(seed_rels.values())
            for key, rel in extracted_rels.items():
                if key not in seed_rels:
                    integrated["relationships"].append(rel)
        
        self.logger.info(
            f"Integrated data: {len(integrated['entities'])} entities, "
            f"{len(integrated['relationships'])} relationships"
        )
        
        return integrated
    
    def validate_quality(
        self,
        seed_data: Dict[str, Any],
        **options
    ) -> Dict[str, Any]:
        """
        Validate seed data quality.
        
        Args:
            seed_data: Seed data dictionary
            **options: Validation options:
                - check_required_fields: Check required fields (default: True)
                - check_types: Validate data types (default: True)
                - check_consistency: Check consistency (default: True)
                
        Returns:
            Validation result dictionary
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        entities = seed_data.get("entities", [])
        relationships = seed_data.get("relationships", [])
        
        # Check entities
        entity_ids = []
        for entity in entities:
            if "id" not in entity:
                results["errors"].append("Entity missing 'id' field")
                results["valid"] = False
            else:
                if entity["id"] in entity_ids:
                    results["warnings"].append(f"Duplicate entity ID: {entity['id']}")
                entity_ids.append(entity["id"])
            
            if "type" not in entity:
                results["warnings"].append(f"Entity {entity.get('id')} missing 'type' field")
        
        # Check relationships
        for rel in relationships:
            if "source_id" not in rel or "target_id" not in rel:
                results["errors"].append("Relationship missing source_id or target_id")
                results["valid"] = False
            
            if "type" not in rel:
                results["warnings"].append("Relationship missing 'type' field")
        
        # Calculate metrics
        results["metrics"] = {
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "unique_entity_ids": len(set(entity_ids)),
            "duplicate_entities": len(entities) - len(set(entity_ids))
        }
        
        return results
    
    def _record_to_entity(self, record: Dict[str, Any]) -> Optional[EntityDict]:
        """Convert record to entity dictionary."""
        if "id" not in record:
            return None
        
        entity = {
            "id": record["id"],
            "text": record.get("text") or record.get("name") or record.get("label", ""),
            "type": record.get("entity_type") or record.get("type", "UNKNOWN"),
            "confidence": record.get("confidence", 1.0),
            "metadata": {k: v for k, v in record.items() 
                        if k not in ["id", "text", "name", "label", "type", "entity_type", "confidence"]}
        }
        
        return entity
    
    def _record_to_relationship(self, record: Dict[str, Any]) -> Optional[RelationshipDict]:
        """Convert record to relationship dictionary."""
        if "source_id" not in record or "target_id" not in record:
            return None
        
        relationship = {
            "id": record.get("id") or f"{record['source_id']}_{record['target_id']}",
            "source_id": record["source_id"],
            "target_id": record["target_id"],
            "type": record.get("relationship_type") or record.get("type", "RELATED_TO"),
            "confidence": record.get("confidence", 1.0),
            "metadata": {k: v for k, v in record.items() 
                        if k not in ["id", "source_id", "target_id", "type", "relationship_type", "confidence"]}
        }
        
        return relationship
    
    def _validate_against_template(
        self,
        foundation: Dict[str, Any],
        schema_template: Any
    ) -> Dict[str, Any]:
        """Validate foundation against schema template."""
        # This would use schema template validation if available
        # For now, just return the foundation
        return foundation
    
    def export_seed_data(
        self,
        file_path: Union[str, Path],
        format: str = "json"
    ) -> None:
        """
        Export seed data to file.
        
        Args:
            file_path: Output file path
            format: Export format ('json', 'csv')
        """
        file_path = Path(file_path)
        
        if format == "json":
            export_data = {
                "entities": self.seed_data.entities,
                "relationships": self.seed_data.relationships,
                "metadata": {
                    **self.seed_data.metadata,
                    "exported_at": datetime.now().isoformat()
                }
            }
            write_json_file(export_data, file_path)
        
        elif format == "csv":
            # Export entities to CSV
            entities_file = file_path.parent / f"{file_path.stem}_entities.csv"
            if self.seed_data.entities:
                with open(entities_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.seed_data.entities[0].keys())
                    writer.writeheader()
                    writer.writerows(self.seed_data.entities)
            
            # Export relationships to CSV
            relationships_file = file_path.parent / f"{file_path.stem}_relationships.csv"
            if self.seed_data.relationships:
                with open(relationships_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.seed_data.relationships[0].keys())
                    writer.writeheader()
                    writer.writerows(self.seed_data.relationships)
        
        self.logger.info(f"Exported seed data to: {file_path}")
