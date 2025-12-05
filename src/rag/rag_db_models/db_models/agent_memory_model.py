from .base_model import BaseModel
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import hashlib


class AgentMemoryModel(BaseModel):
    """
    Model for agent_memory table in SQLite.
    Stores agent memories for self-reflection, learning, and debugging.
    
    MEMORY TYPES:
    - "context": Contextual information (query patterns, user profiles, document summaries)
    - "log": Execution logs (ingestion status, retrieval attempts, healing actions)
    - "decision": Strategic decisions (which healing strategy worked, parameter adjustments)
    - "performance": Performance metrics (quality scores, cost analysis, latency trends)
    
    WORKFLOW:
    1. Agent records memory during execution
    2. Memory cached in-memory for fast access (LRU cache with TTL)
    3. Memory persisted to SQLite for long-term learning
    4. On startup, agent can query historical memory for self-improvement
    """
    
    # --- Class-level Configuration ---
    table = 'agent_memory'
    fields = [
        'id', 'agent_id', 'memory_type', 'memory_key', 'content', 
        'importance_score', 'access_count', 'created_at', 'updated_at', 'expires_at'
    ]
    
    def __init__(self, conn=None):
        """Initialize Agent Memory Model with database connection"""
        if conn is None:
            from ..db.connection import get_connection
            conn = get_connection()
        super().__init__(conn)
        self.db_path = getattr(conn, 'db_path', 'unknown')
        self._create_table_if_not_exists()
    
    def _create_table_if_not_exists(self):
        """Create agent_memory table if it doesn't exist"""
        try:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL CHECK(memory_type IN ('context', 'log', 'decision', 'performance')),
                    memory_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    UNIQUE(agent_id, memory_type, memory_key)
                )
            """)
            self.conn.commit()
            
            # Create index for faster queries
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_id 
                ON {self.table}(agent_id)
            """)
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_agent_memory_type 
                ON {self.table}(agent_id, memory_type)
            """)
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_agent_memory_expires 
                ON {self.table}(expires_at)
            """)
            self.conn.commit()
            
        except Exception as e:
            print(f"[WARNING] Failed to create agent_memory table: {e}")
    
    def record_memory(self, agent_id: str, memory_type: str, memory_key: str, 
                     content: str, importance_score: float = 0.5, 
                     ttl_hours: Optional[int] = None) -> int:
        """
        Record a new memory or update existing one.
        
        Args:
            agent_id: ID of the agent (e.g., 'langgraph_agent', 'ingestion_agent')
            memory_type: Type of memory ('context', 'log', 'decision', 'performance')
            memory_key: Unique key for this memory (e.g., 'healing_strategy_re_embed')
            content: Memory content (JSON string recommended for complex data)
            importance_score: Score from 0-1 (higher = more important to keep)
            ttl_hours: Time-to-live in hours (None = keep forever)
        
        Returns:
            memory_id (int): ID of the recorded memory
        """
        try:
            now = datetime.now().isoformat()
            expires_at = None
            
            if ttl_hours:
                expires_at = (datetime.now() + timedelta(hours=ttl_hours)).isoformat()
            
            # Try to update if already exists
            existing = self.conn.execute(f"""
                SELECT id FROM {self.table} 
                WHERE agent_id = ? AND memory_type = ? AND memory_key = ?
            """, (agent_id, memory_type, memory_key)).fetchone()
            
            if existing:
                # Update existing memory
                memory_id = existing[0]
                self.conn.execute(f"""
                    UPDATE {self.table} 
                    SET content = ?, importance_score = ?, access_count = access_count + 1, 
                        updated_at = ?, expires_at = ?
                    WHERE id = ?
                """, (content, importance_score, now, expires_at, memory_id))
            else:
                # Insert new memory
                cur = self.conn.execute(f"""
                    INSERT INTO {self.table} 
                    (agent_id, memory_type, memory_key, content, importance_score, 
                     created_at, updated_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (agent_id, memory_type, memory_key, content, importance_score, now, now, expires_at))
                memory_id = cur.lastrowid
            
            self.conn.commit()
            return memory_id
        
        except Exception as e:
            print(f"[ERROR] Failed to record memory: {e}")
            return None
    
    def retrieve_memory(self, agent_id: str, memory_type: Optional[str] = None, 
                       memory_key: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve memories for an agent.
        
        Args:
            agent_id: ID of the agent to retrieve memories for
            memory_type: Filter by type (None = all types)
            memory_key: Filter by specific key (None = all keys)
            limit: Maximum number of memories to return
        
        Returns:
            List of memory dictionaries sorted by importance and recency
        """
        try:
            query = f"SELECT * FROM {self.table} WHERE agent_id = ?"
            params = [agent_id]
            
            if memory_type:
                query += " AND memory_type = ?"
                params.append(memory_type)
            
            if memory_key:
                query += " AND memory_key = ?"
                params.append(memory_key)
            
            # Filter out expired memories
            query += " AND (expires_at IS NULL OR expires_at > ?)"
            params.append(datetime.now().isoformat())
            
            # Sort by importance and recency
            query += " ORDER BY importance_score DESC, updated_at DESC LIMIT ?"
            params.append(limit)
            
            cur = self.conn.execute(query, params)
            rows = cur.fetchall()
            
            # Update access count
            for row in rows:
                self.conn.execute(f"""
                    UPDATE {self.table} SET access_count = access_count + 1 
                    WHERE id = ?
                """, (row[0],))
            self.conn.commit()
            
            # Convert to dictionaries
            return [dict(zip(self.fields, row)) for row in rows]
        
        except Exception as e:
            print(f"[ERROR] Failed to retrieve memory: {e}")
            return []
    
    def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get statistics about an agent's memories"""
        try:
            stats = {}
            
            # Total memories
            cur = self.conn.execute(
                f"SELECT COUNT(*) FROM {self.table} WHERE agent_id = ?", 
                (agent_id,)
            )
            stats['total_memories'] = cur.fetchone()[0]
            
            # By type
            cur = self.conn.execute(
                f"SELECT memory_type, COUNT(*) FROM {self.table} WHERE agent_id = ? GROUP BY memory_type", 
                (agent_id,)
            )
            stats['by_type'] = {row[0]: row[1] for row in cur.fetchall()}
            
            # Average importance
            cur = self.conn.execute(
                f"SELECT AVG(importance_score) FROM {self.table} WHERE agent_id = ?", 
                (agent_id,)
            )
            stats['avg_importance'] = cur.fetchone()[0] or 0.0
            
            # Most accessed
            cur = self.conn.execute(
                f"SELECT memory_key, access_count FROM {self.table} WHERE agent_id = ? ORDER BY access_count DESC LIMIT 5", 
                (agent_id,)
            )
            stats['most_accessed'] = {row[0]: row[1] for row in cur.fetchall()}
            
            return stats
        
        except Exception as e:
            print(f"[ERROR] Failed to get memory stats: {e}")
            return {}
    
    def cleanup_expired_memories(self) -> int:
        """Delete expired memories and return count deleted"""
        try:
            now = datetime.now().isoformat()
            cur = self.conn.execute(f"""
                DELETE FROM {self.table} 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """, (now,))
            self.conn.commit()
            return cur.rowcount
        
        except Exception as e:
            print(f"[ERROR] Failed to cleanup expired memories: {e}")
            return 0
    
    def delete_memory(self, agent_id: str, memory_type: str, memory_key: str) -> bool:
        """Delete a specific memory"""
        try:
            self.conn.execute(f"""
                DELETE FROM {self.table} 
                WHERE agent_id = ? AND memory_type = ? AND memory_key = ?
            """, (agent_id, memory_type, memory_key))
            self.conn.commit()
            return True
        
        except Exception as e:
            print(f"[ERROR] Failed to delete memory: {e}")
            return False
    
    def clear_agent_memories(self, agent_id: str) -> int:
        """Clear all memories for an agent and return count deleted"""
        try:
            cur = self.conn.execute(f"""
                DELETE FROM {self.table} WHERE agent_id = ?
            """, (agent_id,))
            self.conn.commit()
            return cur.rowcount
        
        except Exception as e:
            print(f"[ERROR] Failed to clear agent memories: {e}")
            return 0
