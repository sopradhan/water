from .base_model import BaseModel 
import json
import sqlite3
from datetime import datetime
from ..db.connection import get_connection

class ChunkEmbeddingDataModel(BaseModel):
    """
    Model for chunk_embedding_data table in optimized schema.
    Tracks per-chunk embedding health, versioning, and quality for RL healing agent.
    """
    
    # --- Class-level Configuration (Adopting BaseModel Structure) ---
    table = 'chunk_embedding_data'
    fields = [
        'chunk_id', 'doc_id', 'embedding_model', 'embedding_version', 
        'quality_score', 'reindex_count', 'healing_suggestions', 
        'rbac_tags', 'meta_tags', 'created_at', 'last_healed'
    ]
    
    def __init__(self, conn=None):
        """Initialize with optional connection. If none provided, create default connection."""
        if conn is None:
            conn = get_connection()
        super().__init__(conn)

    def _row_to_dict(self, row) -> dict | None:
        """Helper to convert sqlite3.Row or tuple to a dictionary based on self.fields."""
        if row is None:
            return None
        
        # Check if row is a dictionary-like object (from row_factory=sqlite3.Row)
        if hasattr(row, 'keys'):
            return dict(row)
        
        # Fallback in case row_factory is not used (assuming row is a tuple)
        return dict(zip(self.fields, row))

    def create(self, chunk_id: str, doc_id: str, embedding_model: str,
                 embedding_version: str = "1.0", quality_score: float = 0.8,
                 reindex_count: int = 0, healing_suggestions: str = None,
                 rbac_tags: str = None, meta_tags: str = None) -> bool:
        """
        Create or update a chunk embedding record using INSERT OR REPLACE.
        
        Args:
            rbac_tags: JSON string of RBAC access control tags
            meta_tags: JSON string of semantic metadata tags
        """
        try:
            now_iso = datetime.now().isoformat()
            
            # Prepare data tuple based on the table columns, matching the order of placeholders
            data = (
                chunk_id, doc_id, embedding_model, embedding_version, 
                quality_score, reindex_count, 
                healing_suggestions or json.dumps({}),
                rbac_tags or json.dumps([]),  # Store RBAC tags as JSON
                meta_tags or json.dumps([]),  # Store semantic tags as JSON
                now_iso, 
                None # last_healed is NULL initially
            )
            
            # Dynamically generate SQL query string
            columns = ", ".join(self.fields)
            placeholders = ", ".join(["?"] * len(self.fields))
            
            self.conn.execute(f"""
                INSERT OR REPLACE INTO {self.table}
                ({columns})
                VALUES ({placeholders})
            """, data)
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error creating chunk embedding data: {e}")
            return False
    
    # --- Data Retrieval Methods ---

    def get_by_id(self, chunk_id: str) -> dict | None:
        """Get chunk embedding data by chunk_id."""
        try:
            fields_str = ', '.join(self.fields)
            cur = self.conn.execute(f"""
                SELECT {fields_str}
                FROM {self.table}
                WHERE chunk_id = ?
            """, (chunk_id,))
            
            row = cur.fetchone()
            return self._row_to_dict(row)
            
        except Exception as e:
            print(f"Error getting chunk embedding data: {e}")
            return None

    def get_by_doc_id(self, doc_id: str) -> list[dict]:
        """Get all chunks for a specific document."""
        try:
            fields_str = ', '.join(self.fields)
            cur = self.conn.execute(f"""
                SELECT {fields_str}
                FROM {self.table}
                WHERE doc_id = ?
                ORDER BY created_at ASC
            """, (doc_id,))
            
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error getting chunks by doc_id: {e}")
            return []
    
    def get_low_quality_chunks(self, threshold: float = 0.6) -> list[dict]:
        """Get chunks with quality score below the given threshold."""
        try:
            fields_str = ', '.join(self.fields)
            cur = self.conn.execute(f"""
                SELECT {fields_str}
                FROM {self.table}
                WHERE quality_score < ?
                ORDER BY quality_score ASC
            """, (threshold,))
            
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error getting low quality chunks: {e}")
            return []

    # --- Update Methods ---

    def update_quality_score(self, chunk_id: str, quality_score: float) -> bool:
        """Update quality score for a chunk."""
        try:
            self.conn.execute(f"""
                UPDATE {self.table}
                SET quality_score = ?
                WHERE chunk_id = ?
            """, (quality_score, chunk_id))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error updating quality score: {e}")
            return False
    
    def increment_reindex_count(self, chunk_id: str) -> bool:
        """Increment reindex count and update last_healed timestamp for a chunk."""
        try:
            now_iso = datetime.now().isoformat()
            self.conn.execute(f"""
                UPDATE {self.table}
                SET reindex_count = reindex_count + 1,
                    last_healed = ?
                WHERE chunk_id = ?
            """, (now_iso, chunk_id))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error incrementing reindex count: {e}")
            return False

    # --- Statistics Method ---
    
    def get_statistics(self, doc_id: str = None) -> dict:
        """Get statistics (count, avg/min/max quality) for chunks, optionally filtered by doc_id."""
        try:
            query = f"""
                SELECT COUNT(*), AVG(quality_score), MIN(quality_score), MAX(quality_score)
                FROM {self.table}
            """
            params = ()
            
            if doc_id:
                query += " WHERE doc_id = ?"
                params = (doc_id,)
                
            cur = self.conn.execute(query, params)
            row = cur.fetchone()
            
            # Using row[index] for aggregate results
            return {
                "total_chunks": row[0] or 0,
                "avg_quality": row[1] or 0.0,
                "min_quality": row[2] or 0.0,
                "max_quality": row[3] or 0.0
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
