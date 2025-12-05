from .base_model import BaseModel 
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any
import threading

class RAGHistoryModel(BaseModel):
    """
    Model for rag_history_and_optimization table in optimized schema.
    Unified historical log for queries, healing operations, and synthetic tests.
    """
    
    # --- Class-level Configuration (Adopting BaseModel Structure) ---
    table = 'rag_history_and_optimization'
    fields = [
        'history_id', 'event_type', 'timestamp', 'query_text', 'target_doc_id',
        'target_chunk_id', 'metrics_json', 'context_json', 'reward_signal',
        'action_taken', 'state_before', 'state_after', 'agent_id', 'user_id', 
        'session_id'
    ]
    
    def __init__(self, conn=None):
        """Initialize RAG History Model with database connection"""
        if conn is None:
            # Get connection using the project's standard connection method
            from ..db.connection import get_connection
            conn = get_connection()
        super().__init__(conn)
        self.db_path = getattr(conn, 'db_path', 'unknown')  # For debugging
        self._local = threading.local()  # Thread-local storage for connections
    
    def _get_connection(self):
        """Get thread-safe connection (creates new connection per thread)"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # Create a new connection for this thread (not shared across threads)
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _row_to_dict(self, row) -> Dict | None:
        """Helper to convert sqlite3.Row or tuple to a dictionary based on self.fields."""
        if row is None:
            return None
        
        # Check if row is a dictionary-like object (from row_factory=sqlite3.Row)
        if hasattr(row, 'keys'):
            return dict(row)
        
        # Fallback in case row_factory is not used (assuming row is a tuple)
        return dict(zip(self.fields, row))

    def log_query(self, query_text: str, target_doc_id: str, metrics_json: str,
                  context_json: str = None, agent_id: str = "langgraph_agent",
                  user_id: str = None, session_id: str = None) -> int:
        """
        Log a query event with metadata.
        """
        try:
            now_iso = datetime.now().isoformat()
            data = (
                query_text, target_doc_id, None,  # target_chunk_id is None for queries
                metrics_json, context_json or json.dumps({}), 0.0,  # reward_signal=0.0 for queries
                "QUERY", None, None, agent_id, user_id, session_id,
                "QUERY", now_iso
            )
            
            # Reorder data to match table column order (excluding history_id which is auto-increment)
            data_ordered = (
                "QUERY", now_iso, query_text, target_doc_id, None,
                metrics_json, context_json or json.dumps({}), 0.0,
                "QUERY", None, None, agent_id, user_id, session_id
            )
            
            conn = self._get_connection()
            cur = conn.execute(f"""
                INSERT INTO {self.table} 
                (event_type, timestamp, query_text, target_doc_id, target_chunk_id,
                 metrics_json, context_json, reward_signal, action_taken, state_before,
                 state_after, agent_id, user_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_ordered)
            
            conn.commit()
            return cur.lastrowid
            
        except Exception as e:
            print(f"Error logging query: {e}")
            return None

    def log_healing(self, target_doc_id: str, target_chunk_id: str, metrics_json: str,
                    context_json: str, action_taken: str = "RE_EMBED",
                    reward_signal: float = 0.0, agent_id: str = "rl_healing_agent",
                    session_id: str = None) -> int:
        """
        Log a healing/optimization event with metrics and reward signal.
        """
        try:
            now_iso = datetime.now().isoformat()
            
            data_ordered = (
                "HEAL", now_iso, None, target_doc_id, target_chunk_id,
                metrics_json, context_json, reward_signal, action_taken,
                None, None, agent_id, None, session_id
            )
            
            conn = self._get_connection()
            cur = conn.execute(f"""
                INSERT INTO {self.table} 
                (event_type, timestamp, query_text, target_doc_id, target_chunk_id,
                 metrics_json, context_json, reward_signal, action_taken, state_before,
                 state_after, agent_id, user_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_ordered)
            
            conn.commit()
            return cur.lastrowid
            
        except Exception as e:
            print(f"Error logging healing: {e}")
            return None

    def log_synthetic_test(self, query_text: str, target_doc_id: str, 
                          metrics_json: str, context_json: str,
                          reward_signal: float = 0.0, agent_id: str = "synthetic_test_agent",
                          session_id: str = None) -> int:
        """
        Log a synthetic test event (for evaluating RAG pipeline quality).
        """
        try:
            now_iso = datetime.now().isoformat()
            
            data_ordered = (
                "SYNTHETIC_TEST", now_iso, query_text, target_doc_id, None,
                metrics_json, context_json, reward_signal, "EVALUATE",
                None, None, agent_id, None, session_id
            )
            
            conn = self._get_connection()
            cur = conn.execute(f"""
                INSERT INTO {self.table} 
                (event_type, timestamp, query_text, target_doc_id, target_chunk_id,
                 metrics_json, context_json, reward_signal, action_taken, state_before,
                 state_after, agent_id, user_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_ordered)
            
            conn.commit()
            return cur.lastrowid
            
        except Exception as e:
            print(f"Error logging synthetic test: {e}")
            return None

    def get_recent_queries(self, limit: int = 10) -> List[Dict]:
        """Get recent query events."""
        try:
            cur = self.conn.execute(f"""
                SELECT {', '.join(self.fields)}
                FROM {self.table}
                WHERE event_type = 'QUERY'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error getting recent queries: {e}")
            return []

    def get_recent_healings(self, limit: int = 10) -> List[Dict]:
        """Get recent healing events."""
        try:
            cur = self.conn.execute(f"""
                SELECT {', '.join(self.fields)}
                FROM {self.table}
                WHERE event_type = 'HEAL'
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error getting recent healings: {e}")
            return []

    def get_doc_performance_history(self, target_doc_id: str) -> List[Dict]:
        """Get performance history for a specific document."""
        try:
            cur = self.conn.execute(f"""
                SELECT {', '.join(self.fields)}
                FROM {self.table}
                WHERE target_doc_id = ?
                ORDER BY timestamp ASC
            """, (target_doc_id,))
            
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error getting doc performance history: {e}")
            return []

    def get_agent_performance(self, agent_id: str) -> List[Dict]:
        """Get performance metrics for a specific agent."""
        try:
            cur = self.conn.execute(f"""
                SELECT {', '.join(self.fields)}
                FROM {self.table}
                WHERE agent_id = ?
                ORDER BY timestamp DESC
            """, (agent_id,))
            
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error getting agent performance: {e}")
            return []

    def search_by_filters(self, event_type: str = None, target_doc_id: str = None,
                         agent_id: str = None, start_time: str = None, 
                         end_time: str = None, limit: int = 100) -> List[Dict]:
        """Search history with multiple filters."""
        try:
            query = f"SELECT {', '.join(self.fields)} FROM {self.table} WHERE 1=1"
            params = []
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
                
            if target_doc_id:
                query += " AND target_doc_id = ?"
                params.append(target_doc_id)
                
            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)
                
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
                
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cur = self.conn.execute(query, params)
            return [self._row_to_dict(row) for row in cur.fetchall()]
            
        except Exception as e:
            print(f"Error searching history: {e}")
            return []

    def log_guardrail_check(self, target_doc_id: str, checks_json: str, is_safe: bool = True,
                           agent_id: str = "langgraph_agent", session_id: str = None) -> int:
        """
        Log a guardrails validation check event.
        
        Args:
            target_doc_id: Document ID that was checked
            checks_json: JSON with guardrail check results
            is_safe: Whether response passed all guardrails
            agent_id: Agent that performed the check
            session_id: Session identifier
            
        Returns:
            History ID of the logged event
        """
        try:
            now_iso = datetime.now().isoformat()
            
            # Create action_taken based on safety result
            action_taken = "PASS" if is_safe else "FLAG"
            
            data_ordered = (
                "GUARDRAIL_CHECK", now_iso, None, target_doc_id, None,
                checks_json, json.dumps({"is_safe": is_safe}), 1.0 if is_safe else 0.0, 
                action_taken, None, None, agent_id, None, session_id
            )
            
            conn = self._get_connection()
            cur = conn.execute(f"""
                INSERT INTO {self.table} 
                (event_type, timestamp, query_text, target_doc_id, target_chunk_id,
                 metrics_json, context_json, reward_signal, action_taken, state_before,
                 state_after, agent_id, user_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data_ordered)
            
            conn.commit()
            return cur.lastrowid
            
        except Exception as e:
            print(f"Error logging guardrail check: {e}")
            return None

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()
