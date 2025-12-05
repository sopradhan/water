"""Base model class for RAG database models."""

import sqlite3
from typing import Optional, Dict, List, Any


class BaseModel:
    """
    Base class for all RAG database models.
    Provides common database operations and connection management.
    """
    
    # Override in subclasses
    table: str = None
    fields: List[str] = []
    
    def __init__(self, conn: sqlite3.Connection):
        """Initialize model with database connection."""
        if conn is None:
            raise ValueError("Database connection required")
        self.conn = conn
    
    def _row_to_dict(self, row) -> Optional[Dict]:
        """Convert sqlite3.Row or tuple to dictionary based on self.fields."""
        if row is None:
            return None
        
        # Check if row is a dictionary-like object (from row_factory=sqlite3.Row)
        if hasattr(row, 'keys'):
            return dict(row)
        
        # Fallback: convert tuple to dict using fields
        if isinstance(row, (tuple, list)):
            return dict(zip(self.fields, row))
        
        return row
    
    def execute(self, sql: str, params: tuple = None) -> sqlite3.Cursor:
        """Execute SQL statement."""
        if params:
            return self.conn.execute(sql, params)
        return self.conn.execute(sql)
    
    def executemany(self, sql: str, params_list: list) -> sqlite3.Cursor:
        """Execute multiple SQL statements."""
        return self.conn.executemany(sql, params_list)
    
    def commit(self):
        """Commit transaction."""
        self.conn.commit()
    
    def rollback(self):
        """Rollback transaction."""
        self.conn.rollback()
    
    def fetchone(self, sql: str, params: tuple = None) -> Optional[Dict]:
        """Fetch single row."""
        row = self.execute(sql, params).fetchone()
        return self._row_to_dict(row)
    
    def fetchall(self, sql: str, params: tuple = None) -> List[Dict]:
        """Fetch all rows."""
        rows = self.execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]
