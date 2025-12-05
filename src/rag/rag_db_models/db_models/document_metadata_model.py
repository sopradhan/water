from .base_model import BaseModel
import json
import sqlite3
from datetime import datetime
from ..db.connection import get_connection

class DocumentMetadataModel(BaseModel):
    """
    Model for document_metadata table in optimized schema.
    Stores document-level metadata with chunking strategy and RBAC namespace.
    """
    
    # --- Class-level Configuration (Following BaseModel Example) ---
    table = 'document_metadata'
    fields = [
        'doc_id', 'title', 'author', 'source', 'summary', 
        'company_id', 'dept_id',
        'rbac_namespace', 'chunk_strategy', 'chunk_size_char', 
        'overlap_char', 'metadata_json', 'rbac_tags', 'meta_tags', 'last_ingested'
    ]
    
    def __init__(self, conn=None):
        """Initialize with optional connection. If none provided, create default connection."""
        if conn is None:
            conn = get_connection()
        super().__init__(conn)

    def create(self, doc_id: str, title: str, author: str = None, 
                 source: str = None, summary: str = None, 
                 company_id: int = None, dept_id: int = None,
                 rbac_namespace: str = "general",
                 chunk_strategy: str = "recursive_splitter",
                 chunk_size_char: int = 512, overlap_char: int = 50,
                 metadata_json: str = None, rbac_tags: str = None, 
                 meta_tags: str = None) -> bool:
        """
        Create or update a document metadata record (using INSERT OR REPLACE).
        
        Args:
            rbac_tags: JSON string of RBAC access control tags
            meta_tags: JSON string of semantic metadata tags
        """
        try:
            # Prepare data, using defaults and current timestamp
            now_iso = datetime.now().isoformat()
            data = (
                doc_id,
                title,
                author or "unknown",
                source or "ingestion",
                summary or "",
                company_id,
                dept_id,
                rbac_namespace,
                chunk_strategy,
                chunk_size_char,
                overlap_char,
                metadata_json or json.dumps({}),
                rbac_tags or json.dumps([]),  # Store RBAC tags as JSON
                meta_tags or json.dumps([]),  # Store semantic tags as JSON
                now_iso
            )
            
            # Construct the column names for the query
            columns = ", ".join(self.fields)
            # Construct placeholders for the values
            placeholders = ", ".join(["?"] * len(self.fields))
            
            self.conn.execute(f"""
                INSERT OR REPLACE INTO {self.table}
                ({columns})
                VALUES ({placeholders})
            """, data)
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error creating document metadata: {e}")
            return False

    def _row_to_dict(self, row) -> dict | None:
        """Helper to convert sqlite3.Row or tuple to dictionary."""
        if row is None:
            return None
        
        # If BaseModel uses row_factory=sqlite3.Row, row is a dictionary-like object
        if hasattr(row, 'keys'):
            return dict(row)
        
        # Fallback if row is a tuple and row_factory is not set correctly on the cursor
        return dict(zip(self.fields, row))

    def get_by_id(self, doc_id: str) -> dict | None:
        """Get document metadata by doc_id."""
        try:
            cur = self.conn.execute(f"""
                SELECT {', '.join(self.fields)}
                FROM {self.table}
                WHERE doc_id = ?
            """, (doc_id,))
            
            row = cur.fetchone()
            return self._row_to_dict(row)
            
        except Exception as e:
            print(f"Error getting document metadata: {e}")
            return None

    def get_all(self) -> list[dict]:
        """Get all document metadata records, ordered by last_ingested."""
        try:
            cur = self.conn.execute(f"""
                SELECT {', '.join(self.fields)}
                FROM {self.table}
                ORDER BY last_ingested DESC
            """)
            
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error getting all document metadata: {e}")
            return []

    def get_by_namespace(self, rbac_namespace: str) -> list[dict]:
        """Get document metadata by RBAC namespace."""
        try:
            cur = self.conn.execute(f"""
                SELECT {', '.join(self.fields)}
                FROM {self.table}
                WHERE rbac_namespace = ?
                ORDER BY last_ingested DESC
            """, (rbac_namespace,))
            
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error getting documents by namespace: {e}")
            return []

    def update_summary(self, doc_id: str, summary: str) -> bool:
        """Update the summary for a document."""
        try:
            self.conn.execute(f"""
                UPDATE {self.table}
                SET summary = ?
                WHERE doc_id = ?
            """, (summary, doc_id))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error updating summary: {e}")
            return False

    def update_ingestion_time(self, doc_id: str) -> bool:
        """Update the last_ingested timestamp for a document."""
        try:
            now_iso = datetime.now().isoformat()
            self.conn.execute(f"""
                UPDATE {self.table}
                SET last_ingested = ?
                WHERE doc_id = ?
            """, (now_iso, doc_id))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error updating ingestion time: {e}")
            return False

    def get_total_count(self) -> int:
        """Get the total count of documents in the database."""
        try:
            cur = self.conn.execute(f"SELECT COUNT(*) FROM {self.table}")
            row = cur.fetchone()
            return row[0] if row else 0
            
        except Exception as e:
            print(f"Error getting total count: {e}")
            return 0
