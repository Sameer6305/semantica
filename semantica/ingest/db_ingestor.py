"""
Database Ingestion Module

Handles database export and data extraction.

Key Features:
    - Database connection management
    - SQL query execution
    - Data export and transformation
    - Schema analysis
    - Data type handling

Main Classes:
    - DBIngestor: Main database ingestion class
    - DatabaseConnector: Database connection handler
    - DataExporter: Data export processor
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import sqlalchemy
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger


@dataclass
class TableData:
    """Table data representation."""
    
    table_name: str
    columns: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]
    row_count: int
    schema: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatabaseConnector:
    """
    Database connection management.
    
    Manages connections to various databases,
    handles connection pooling and reuse,
    and provides unified interface for different DBs.
    """
    
    SUPPORTED_DATABASES = {
        'postgresql': 'postgresql',
        'postgres': 'postgresql',
        'mysql': 'mysql+pymysql',
        'mariadb': 'mysql+pymysql',
        'sqlite': 'sqlite',
        'oracle': 'oracle',
        'mssql': 'mssql+pyodbc',
        'sqlserver': 'mssql+pyodbc',
    }
    
    def __init__(self, db_type: str, **config):
        """
        Initialize database connector.
        
        Args:
            db_type: Database type (postgresql, mysql, sqlite, etc.)
            **config: Connection configuration
        """
        self.logger = get_logger("database_connector")
        self.db_type = db_type.lower()
        self.config = config
        self.engine: Optional[Engine] = None
    
    def connect(self, connection_string: str) -> Engine:
        """
        Establish database connection.
        
        Args:
            connection_string: Database connection string
            
        Returns:
            Engine: SQLAlchemy engine object
        """
        try:
            # Parse connection string to detect database type
            parsed = urlparse(connection_string)
            
            # Determine database type from connection string if not provided
            if parsed.scheme:
                db_type = parsed.scheme.split('+')[0]
                if db_type in self.SUPPORTED_DATABASES:
                    self.db_type = db_type
            
            # Get SQLAlchemy dialect
            dialect = self.SUPPORTED_DATABASES.get(self.db_type, self.db_type)
            
            # Create engine
            self.engine = create_engine(connection_string, echo=False)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info(f"Connected to {self.db_type} database")
            return self.engine
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise ProcessingError(f"Failed to connect to database: {e}")
    
    def disconnect(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.logger.info("Disconnected from database")
    
    def test_connection(self, connection_string: str) -> bool:
        """
        Test database connection.
        
        Args:
            connection_string: Database connection string
            
        Returns:
            bool: True if connection successful
        """
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            engine.dispose()
            return True
        except Exception:
            return False


class DataExporter:
    """
    Database data export and transformation.
    
    Exports data from database tables,
    transforms data to standard formats,
    handles different data types,
    and manages large dataset exports.
    """
    
    def __init__(self, **config):
        """
        Initialize data exporter.
        
        Args:
            **config: Exporter configuration
        """
        self.logger = get_logger("data_exporter")
        self.config = config
    
    def export_table_data(
        self,
        connection: Engine,
        table_name: str,
        schema: Optional[str] = None,
        **options
    ) -> TableData:
        """
        Export data from database table.
        
        Args:
            connection: Database connection engine
            table_name: Table name to export
            schema: Schema name (optional)
            **options: Export options:
                - limit: Maximum number of rows
                - offset: Row offset
                - where: WHERE clause
                - order_by: ORDER BY clause
                
        Returns:
            TableData: Exported table data
        """
        try:
            inspector = inspect(connection)
            
            # Get column information
            columns = inspector.get_columns(table_name, schema=schema)
            column_info = [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col.get("nullable", True),
                    "default": str(col.get("default")) if col.get("default") else None
                }
                for col in columns
            ]
            
            # Build query
            query = f'SELECT * FROM {f"{schema}." if schema else ""}"{table_name}"'
            
            if options.get("where"):
                query += f" WHERE {options['where']}"
            
            if options.get("order_by"):
                query += f" ORDER BY {options['order_by']}"
            
            if options.get("limit"):
                query += f" LIMIT {options['limit']}"
            
            if options.get("offset"):
                query += f" OFFSET {options['offset']}"
            
            # Execute query
            with connection.connect() as conn:
                result = conn.execute(text(query))
                rows = []
                for row in result:
                    row_dict = {}
                    for col_name, value in row._mapping.items():
                        # Convert datetime and other types to JSON-serializable format
                        if isinstance(value, datetime):
                            row_dict[col_name] = value.isoformat()
                        elif hasattr(value, '__dict__'):
                            row_dict[col_name] = str(value)
                        else:
                            row_dict[col_name] = value
                    rows.append(row_dict)
                
                # Get row count if limit not applied
                row_count = len(rows)
                if not options.get("limit"):
                    count_query = f'SELECT COUNT(*) FROM {f"{schema}." if schema else ""}"{table_name}"'
                    if options.get("where"):
                        count_query += f" WHERE {options['where']}"
                    count_result = conn.execute(text(count_query))
                    row_count = count_result.scalar()
            
            return TableData(
                table_name=table_name,
                columns=column_info,
                rows=rows,
                row_count=row_count,
                schema=schema
            )
            
        except Exception as e:
            self.logger.error(f"Failed to export table {table_name}: {e}")
            raise ProcessingError(f"Failed to export table: {e}")
    
    def transform_data(self, raw_data: List[Dict[str, Any]], schema_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transform raw database data.
        
        Args:
            raw_data: Raw database rows
            schema_info: Schema information
            
        Returns:
            list: Transformed data
        """
        transformed = []
        
        for row in raw_data:
            transformed_row = {}
            for col_name, value in row.items():
                # Apply transformations
                if value is None:
                    transformed_row[col_name] = None
                elif isinstance(value, str):
                    # Clean and normalize strings
                    transformed_row[col_name] = value.strip() if isinstance(value, str) else value
                else:
                    transformed_row[col_name] = value
            
            transformed.append(transformed_row)
        
        return transformed
    
    def export_schema(self, connection: Engine, schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Export database schema information.
        
        Args:
            connection: Database connection engine
            schema: Schema name (optional)
            
        Returns:
            dict: Schema information
        """
        try:
            inspector = inspect(connection)
            
            schema_info = {
                "tables": [],
                "views": [],
                "foreign_keys": []
            }
            
            # Get tables
            tables = inspector.get_table_names(schema=schema)
            for table_name in tables:
                table_info = {
                    "name": table_name,
                    "columns": [],
                    "primary_keys": [],
                    "indexes": []
                }
                
                # Get columns
                columns = inspector.get_columns(table_name, schema=schema)
                for col in columns:
                    table_info["columns"].append({
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True),
                        "primary_key": col.get("primary_key", False)
                    })
                
                # Get primary keys
                pk = inspector.get_pk_constraint(table_name, schema=schema)
                if pk:
                    table_info["primary_keys"] = pk.get("constrained_columns", [])
                
                # Get indexes
                indexes = inspector.get_indexes(table_name, schema=schema)
                table_info["indexes"] = [idx["name"] for idx in indexes]
                
                schema_info["tables"].append(table_info)
            
            # Get views
            try:
                views = inspector.get_view_names(schema=schema)
                schema_info["views"] = views
            except Exception:
                pass
            
            # Get foreign keys
            for table_name in tables:
                try:
                    foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
                    schema_info["foreign_keys"].extend(foreign_keys)
                except Exception:
                    pass
            
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Failed to export schema: {e}")
            raise ProcessingError(f"Failed to export schema: {e}")


class DBIngestor:
    """
    Database ingestion handler.
    
    Connects to various database systems,
    executes SQL queries and exports data,
    handles different database schemas,
    and processes large datasets efficiently.
    
    Attributes:
        connectors: Dictionary of database connectors
        exporter: Data export processor
        schema_analyzer: Database schema analyzer
        supported_databases: List of supported database types
        
    Methods:
        ingest_database(): Ingest entire database
        execute_query(): Execute SQL query
        export_table(): Export specific table
        analyze_schema(): Analyze database schema
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize database ingestor.
        
        Args:
            config: Database ingestion configuration
            **kwargs: Additional configuration parameters
        """
        self.logger = get_logger("db_ingestor")
        self.config = config or {}
        self.config.update(kwargs)
        
        # Initialize connectors dictionary
        self.connectors: Dict[str, DatabaseConnector] = {}
        
        # Initialize exporter
        self.exporter = DataExporter(**self.config)
        
        # Supported databases
        self.supported_databases = list(DatabaseConnector.SUPPORTED_DATABASES.keys())
    
    def ingest_database(self, connection_string: str, **options) -> Dict[str, Any]:
        """
        Ingest data from entire database.
        
        Args:
            connection_string: Database connection string
            **options: Processing options:
                - include_tables: List of specific tables to include
                - exclude_tables: List of tables to exclude
                - max_rows_per_table: Maximum rows per table
                
        Returns:
            dict: Database content collection
        """
        # Connect to database
        connector = DatabaseConnector("", **self.config)
        engine = connector.connect(connection_string)
        
        try:
            # Analyze schema
            schema = self.analyze_schema(connection_string)
            
            # Get table names
            inspector = inspect(engine)
            all_tables = inspector.get_table_names()
            
            # Apply filters
            include_tables = options.get("include_tables")
            exclude_tables = options.get("exclude_tables", [])
            
            if include_tables:
                tables = [t for t in all_tables if t in include_tables]
            else:
                tables = [t for t in all_tables if t not in exclude_tables]
            
            # Export data from each table
            table_data = {}
            max_rows = options.get("max_rows_per_table")
            
            for table_name in tables:
                try:
                    table_options = {"limit": max_rows} if max_rows else {}
                    table_info = self.exporter.export_table_data(
                        engine,
                        table_name,
                        **table_options
                    )
                    table_data[table_name] = {
                        "columns": table_info.columns,
                        "row_count": table_info.row_count,
                        "rows": table_info.rows[:max_rows] if max_rows else table_info.rows
                    }
                    self.logger.debug(f"Exported table {table_name}: {table_info.row_count} rows")
                except Exception as e:
                    self.logger.error(f"Failed to export table {table_name}: {e}")
            
            return {
                "schema": schema,
                "tables": table_data,
                "total_tables": len(tables),
                "connection_string": connection_string
            }
            
        finally:
            connector.disconnect()
    
    def execute_query(self, connection_string: str, query: str, **params) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            **params: Query parameters
            
        Returns:
            list: Query results
        """
        connector = DatabaseConnector("", **self.config)
        engine = connector.connect(connection_string)
        
        try:
            with engine.connect() as conn:
                # Execute query with parameters
                result = conn.execute(text(query), params)
                
                # Convert results to list of dictionaries
                rows = []
                for row in result:
                    row_dict = {}
                    for col_name, value in row._mapping.items():
                        if isinstance(value, datetime):
                            row_dict[col_name] = value.isoformat()
                        elif hasattr(value, '__dict__'):
                            row_dict[col_name] = str(value)
                        else:
                            row_dict[col_name] = value
                    rows.append(row_dict)
                
                return rows
                
        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise ProcessingError(f"Failed to execute query: {e}")
        finally:
            connector.disconnect()
    
    def export_table(
        self,
        connection_string: str,
        table_name: str,
        schema: Optional[str] = None,
        **filters
    ) -> TableData:
        """
        Export data from specific table.
        
        Args:
            connection_string: Database connection string
            table_name: Table name to export
            schema: Schema name (optional)
            **filters: Filtering criteria:
                - where: WHERE clause
                - limit: Maximum number of rows
                - offset: Row offset
                - order_by: ORDER BY clause
                
        Returns:
            TableData: Table data collection
        """
        connector = DatabaseConnector("", **self.config)
        engine = connector.connect(connection_string)
        
        try:
            table_data = self.exporter.export_table_data(
                engine,
                table_name,
                schema=schema,
                **filters
            )
            
            # Transform data if requested
            if filters.get("transform", False):
                schema_info = {"columns": table_data.columns}
                table_data.rows = self.exporter.transform_data(table_data.rows, schema_info)
            
            return table_data
            
        finally:
            connector.disconnect()
    
    def analyze_schema(self, connection_string: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze database schema and structure.
        
        Args:
            connection_string: Database connection string
            schema: Schema name (optional)
            
        Returns:
            dict: Schema analysis results
        """
        connector = DatabaseConnector("", **self.config)
        engine = connector.connect(connection_string)
        
        try:
            schema_info = self.exporter.export_schema(engine, schema=schema)
            
            # Additional analysis
            schema_info["analysis"] = {
                "total_tables": len(schema_info["tables"]),
                "total_views": len(schema_info["views"]),
                "total_foreign_keys": len(schema_info["foreign_keys"]),
                "tables_with_foreign_keys": len(set(
                    fk["referred_table"] for fk in schema_info["foreign_keys"]
                ))
            }
            
            return schema_info
            
        finally:
            connector.disconnect()
