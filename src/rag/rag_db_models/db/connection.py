"""Database connection management for RAG metadata storage."""

import sqlite3
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGDatabaseConnection:
    """Manages SQLite connection for RAG metadata."""
    
    _instance: Optional['RAGDatabaseConnection'] = None
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database connection."""
        if db_path is None:
            db_path = self._get_default_db_path()
        
        self.db_path = db_path
        self._ensure_directory()
        self.conn: Optional[sqlite3.Connection] = None
    
    @staticmethod
    def _get_default_db_path() -> str:
        """Get default database path from environment or use default."""
        default_path = "src/data/RAG/rag_metadata.db"
        db_path = os.getenv("RAG_DB_PATH", default_path)
        
        # If relative path, make it absolute from project root
        if not os.path.isabs(db_path):
            # From db/connection.py -> rag_db_models/ -> rag/ -> src/ -> project_root/
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent.parent.parent
            db_path = str(project_root / db_path)
        
        return db_path
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.close()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get RAG database connection (singleton pattern)."""
    if RAGDatabaseConnection._instance is None or db_path is not None:
        RAGDatabaseConnection._instance = RAGDatabaseConnection(db_path)
    return RAGDatabaseConnection._instance.connect()


def close_connection():
    """Close RAG database connection."""
    if RAGDatabaseConnection._instance:
        RAGDatabaseConnection._instance.close()
