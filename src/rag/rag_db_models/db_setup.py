"""
RAG Database Setup Script
================================================================================
Initializes SQLite database for RAG metadata management.

Two-Phase Setup:
1. MIGRATIONS: Create tables (idempotent - safe to run multiple times)
2. SEEDERS: Populate initial data (one-time setup)

Database Location: src/data/RAG/rag_metadata.db

Tables Created:
- document_metadata: Document-level tracking with RBAC and chunking strategy
- rag_history_and_optimization: Query logs, healing actions, synthetic tests
- chunk_embedding_data: Per-chunk embedding health and quality metrics
- agent_memory: Agent self-reflection, learning, and decision history

Usage:
    python -m src.rag.rag_db_models.db_setup  # Full setup
    python -m src.rag.rag_db_models.db_setup --migrations-only  # Only create tables
    python -m src.rag.rag_db_models.db_setup --seed-only  # Only populate data
    python -m src.rag.rag_db_models.db_setup --reset  # Drop and recreate
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Callable, Tuple, Optional


# ============================================================================
# Database Connection Management
# ============================================================================

class DatabaseConnection:
    """Manages SQLite database connection and transactions."""
    
    def __init__(self, db_path: str):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
    
    def __enter__(self):
        """Context manager entry - open connection."""
        self._ensure_directory()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Database directory ready: {db_dir}")
    
    def execute(self, sql: str, params: tuple = None):
        """Execute SQL statement."""
        if self.conn is None:
            raise RuntimeError("Database connection not open")
        if params:
            self.conn.execute(sql, params)
        else:
            self.conn.execute(sql)
    
    def executemany(self, sql: str, params_list: list):
        """Execute multiple SQL statements."""
        if self.conn is None:
            raise RuntimeError("Database connection not open")
        self.conn.executemany(sql, params_list)
    
    def commit(self):
        """Commit transaction."""
        if self.conn:
            self.conn.commit()


# ============================================================================
# Migration System
# ============================================================================

class MigrationRunner:
    """Runs database migrations in sequence."""
    
    def __init__(self, conn: sqlite3.Connection):
        """Initialize migration runner with connection."""
        self.conn = conn
        self.migrations = self._get_migrations()
    
    @staticmethod
    def _get_migrations() -> list[Tuple[str, Callable]]:
        """Get list of migrations to run."""
        return [
            ("008_document_metadata", lambda c: create_document_metadata_table(c)),
            ("009_rag_history", lambda c: create_rag_history_table(c)),
            ("010_chunk_embedding", lambda c: create_chunk_embedding_table(c)),
            ("012_agent_memory", lambda c: create_agent_memory_table(c)),
        ]
    
    def run_all(self) -> bool:
        """Run all migrations and return success status."""
        print("\n" + "=" * 80)
        print("PHASE 1: RUNNING MIGRATIONS (Creating Tables)")
        print("=" * 80)
        
        all_success = True
        for migration_name, migration_func in self.migrations:
            try:
                print(f"\n[RUN] {migration_name}...")
                migration_func(self.conn)
                print(f"[OK] {migration_name} completed successfully")
            except Exception as e:
                print(f"[FAIL] {migration_name} failed: {e}")
                all_success = False
        
        return all_success
    
    def rollback_all(self) -> bool:
        """Rollback all migrations (drops all tables)."""
        print("\n" + "=" * 80)
        print("ROLLBACK: Dropping All Tables")
        print("=" * 80)
        
        tables = [
            'document_metadata',
            'rag_history_and_optimization',
            'chunk_embedding_data',
            'agent_memory',
        ]
        
        all_success = True
        for table in tables:
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
                print(f"[OK] Dropped table: {table}")
            except Exception as e:
                print(f"[FAIL] Failed to drop {table}: {e}")
                all_success = False
        
        self.conn.commit()
        return all_success


# ============================================================================
# Migration Functions (Create Tables)
# ============================================================================

def create_document_metadata_table(conn: sqlite3.Connection):
    """Migration 008: Create document_metadata table."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS document_metadata (
            doc_id TEXT PRIMARY KEY,
            
            -- Document Content & Identification
            title TEXT NOT NULL,
            author TEXT,
            source TEXT,
            summary TEXT,
            
            -- Ownership & RBAC
            company_id INTEGER,
            dept_id INTEGER,
            rbac_namespace TEXT NOT NULL DEFAULT 'general',
            
            -- Chunking Strategy (document-level defaults)
            chunk_strategy TEXT NOT NULL DEFAULT 'recursive_splitter',
            chunk_size_char INTEGER NOT NULL DEFAULT 512,
            overlap_char INTEGER NOT NULL DEFAULT 50,
            
            -- Consolidated Metadata (sparse/optional fields as JSON)
            -- Stores: categories, keywords, doc_type, created_date, tags, etc.
            metadata_json TEXT,
            rbac_tags TEXT,
            meta_tags TEXT,
            
            -- Tracking
            last_ingested TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Constraints
            CONSTRAINT valid_chunk_size CHECK (chunk_size_char > 0),
            CONSTRAINT valid_overlap CHECK (overlap_char >= 0)
        )
    ''')
    
    # Create indexes for performance
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_document_metadata_rbac_namespace
        ON document_metadata(rbac_namespace)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_document_metadata_company_dept
        ON document_metadata(company_id, dept_id)
    ''')
    
    conn.commit()


def create_rag_history_table(conn: sqlite3.Connection):
    """Migration 009: Create rag_history_and_optimization table."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS rag_history_and_optimization (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- Event Classification
            event_type TEXT NOT NULL CHECK (event_type IN ('QUERY', 'HEAL', 'SYNTHETIC_TEST', 'GUARDRAIL_CHECK')),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            -- Query Information (populated if event_type='QUERY' or 'SYNTHETIC_TEST')
            query_text TEXT,
            
            -- Document & Chunk Context (populated if event_type='HEAL' or 'SYNTHETIC_TEST')
            target_doc_id TEXT,
            target_chunk_id TEXT,
            
            -- Performance & Metrics (consolidated as JSON for flexibility)
            -- For QUERY events: {frequency, avg_accuracy, cost, latency, user_feedback}
            -- For HEAL events: {strategy, before_metrics, after_metrics, improvement_delta}
            -- For SYNTHETIC_TEST: {expected_answer, generated_answer, accuracy, latency}
            metrics_json TEXT NOT NULL,
            
            -- Context & Actions (consolidated as JSON for additional details)
            -- Stores: source_attributions, actions_taken, reasoning, suggestions, etc.
            context_json TEXT,
            
            -- RL Agent Tracking (for reinforcement learning)
            reward_signal FLOAT,         -- Reward for this event (0.0-1.0)
            action_taken TEXT,           -- Action chosen by RL agent
            state_before TEXT,           -- System state before action (JSON)
            state_after TEXT,            -- System state after action (JSON)
            
            -- Traceability
            agent_id TEXT DEFAULT 'langgraph_agent',
            user_id TEXT,
            session_id TEXT
        )
    ''')
    
    # Create indexes for efficient querying
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_rag_history_event_type
        ON rag_history_and_optimization(event_type)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_rag_history_timestamp
        ON rag_history_and_optimization(timestamp DESC)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_rag_history_agent_id
        ON rag_history_and_optimization(agent_id)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_rag_history_session_id
        ON rag_history_and_optimization(session_id)
    ''')
    
    conn.commit()


def create_chunk_embedding_table(conn: sqlite3.Connection):
    """Migration 010: Create chunk_embedding_data table."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunk_embedding_data (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            
            -- Embedding Information
            embedding_model TEXT NOT NULL,
            embedding_version TEXT,
            
            -- Quality & Health (for Healing/RL Agent)
            quality_score FLOAT DEFAULT 0.8 CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
            reindex_count INTEGER DEFAULT 0,
            
            -- RL Agent Suggestions & Context
            healing_suggestions TEXT, -- JSON: {"strategy": "re_chunk", "reason": "...", "suggested_params": {...}}
            
            -- RBAC & Metadata Tags
            rbac_tags TEXT,
            meta_tags TEXT,
            
            -- Tracking
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_healed TIMESTAMP
        )
    ''')
    
    # Create indexes for performance
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_chunk_embedding_doc_id
        ON chunk_embedding_data(doc_id)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_chunk_embedding_quality
        ON chunk_embedding_data(quality_score)
    ''')
    
    conn.commit()


def create_agent_memory_table(conn: sqlite3.Connection):
    """Migration 012: Create agent_memory table."""
    conn.execute('''
        CREATE TABLE IF NOT EXISTS agent_memory (
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
    ''')
    
    # Create indexes for efficient queries
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_id
        ON agent_memory(agent_id)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_agent_memory_type
        ON agent_memory(agent_id, memory_type)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_agent_memory_expires
        ON agent_memory(expires_at)
    ''')
    
    conn.commit()


# ============================================================================
# Seeder System
# ============================================================================

class SeederRunner:
    """Manages initial data population (seeders)."""
    
    def __init__(self, conn: sqlite3.Connection):
        """Initialize seeder with connection."""
        self.conn = conn
    
    def run_all(self) -> bool:
        """Run all seeders and return success status."""
        print("\n" + "=" * 80)
        print("PHASE 2: RUNNING SEEDERS (Populating Initial Data)")
        print("=" * 80)
        
        try:
            print("\n[RUN] Seeding initial data...")
            self._seed_document_metadata()
            self._seed_rag_history()
            self._seed_agent_memory()
            
            self.conn.commit()
            print("\n[OK] All seeders completed successfully")
            return True
        except Exception as e:
            print(f"\n[FAIL] Seeding failed: {e}")
            return False
    
    def _seed_document_metadata(self):
        """Populate sample document metadata."""
        now = datetime.now().isoformat()
        
        sample_docs = [
            (
                "water_system_manual_001",
                "Water Distribution System Manual",
                "Engineering Team",
                "internal_documentation",
                "Comprehensive manual for water distribution system operations",
                1,
                1,
                "general",
                "recursive_splitter",
                512,
                50,
                json.dumps({"doc_type": "manual", "version": "1.0"}),
                json.dumps(["engineering", "operations"]),
                json.dumps(["system", "guide"]),
                now,
            ),
            (
                "anomaly_detection_guide",
                "Anomaly Detection Guide",
                "Data Science Team",
                "internal_documentation",
                "Guide for detecting anomalies in water pressure and flow",
                1,
                2,
                "general",
                "recursive_splitter",
                512,
                50,
                json.dumps({"doc_type": "guide", "version": "2.0"}),
                json.dumps(["data_science", "ml"]),
                json.dumps(["anomaly", "detection"]),
                now,
            ),
        ]
        
        try:
            self.conn.executemany('''
                INSERT OR IGNORE INTO document_metadata
                (doc_id, title, author, source, summary, company_id, dept_id,
                 rbac_namespace, chunk_strategy, chunk_size_char, overlap_char,
                 metadata_json, rbac_tags, meta_tags, last_ingested)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', sample_docs)
            
            print(f"  [OK] Seeded {len(sample_docs)} document metadata records")
        except sqlite3.IntegrityError:
            print("  [WARN] Document metadata already seeded (skipping)")
    
    def _seed_rag_history(self):
        """Populate sample RAG history records."""
        now = datetime.now().isoformat()
        
        sample_history = [
            (
                "QUERY",
                now,
                "What are the normal pressure ranges for zone A?",
                "water_system_manual_001",
                None,
                json.dumps({"accuracy": 0.92, "latency_ms": 245, "cost": 0.002}),
                json.dumps({"source": "user_query", "session": "sess_001"}),
                0.92,
                None,
                None,
                None,
                "langgraph_agent",
                "user_001",
                "sess_001",
            ),
        ]
        
        try:
            self.conn.executemany('''
                INSERT INTO rag_history_and_optimization
                (event_type, timestamp, query_text, target_doc_id, target_chunk_id,
                 metrics_json, context_json, reward_signal, action_taken,
                 state_before, state_after, agent_id, user_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', sample_history)
            
            print(f"  [OK] Seeded {len(sample_history)} RAG history records")
        except sqlite3.IntegrityError:
            print("  [WARN] RAG history already seeded (skipping)")
    
    def _seed_agent_memory(self):
        """Populate sample agent memory records."""
        now = datetime.now().isoformat()
        
        sample_memory = [
            (
                "langgraph_agent",
                "context",
                "user_profiles",
                json.dumps({"recent_users": ["user_001", "user_002"], "total_sessions": 15}),
                0.8,
                10,
                now,
                now,
                None,
            ),
            (
                "langgraph_agent",
                "performance",
                "query_accuracy",
                json.dumps({"avg_accuracy": 0.88, "total_queries": 150, "trend": "improving"}),
                0.9,
                5,
                now,
                now,
                None,
            ),
        ]
        
        try:
            self.conn.executemany('''
                INSERT OR IGNORE INTO agent_memory
                (agent_id, memory_type, memory_key, content, importance_score,
                 access_count, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', sample_memory)
            
            print(f"  [OK] Seeded {len(sample_memory)} agent memory records")
        except sqlite3.IntegrityError:
            print("  [WARN] Agent memory already seeded (skipping)")


# ============================================================================
# Main Setup Orchestrator
# ============================================================================

def get_db_path() -> str:
    """Get database path from environment or use default."""
    import os
    from pathlib import Path
    
    default_path = "src/data/RAG/rag_metadata.db"
    db_path = os.getenv("RAG_DB_PATH", default_path)
    
    # If relative path, make it absolute from project root
    if not os.path.isabs(db_path):
        # From db_setup in src/rag/rag_db_models -> project_root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        db_path = str(project_root / db_path)
    
    return db_path


def initialize_database(db_path: str = None) -> bool:
    """
    Quick database initialization - creates tables if they don't exist.
    Safe to call multiple times (idempotent).
    
    Args:
        db_path: Optional custom database path, uses default if not provided
        
    Returns:
        bool: True if successful, False otherwise
    """
    if db_path is None:
        db_path = get_db_path()
    
    try:
        with DatabaseConnection(db_path) as conn:
            # Check if main table already exists (quick check for already-initialized DB)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='rag_history_and_optimization'"
            )
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                # Only run migrations if tables don't exist (suppresses output on repeat calls)
                migration_runner = MigrationRunner(conn)
                migration_runner.run_all()
            
            return True
    except Exception as e:
        print(f"[WARNING] Database initialization failed: {e}")
        return False


def main():
    """Main setup orchestrator."""
    parser = argparse.ArgumentParser(
        description="RAG Database Setup - Initialize metadata storage"
    )
    parser.add_argument(
        "--migrations-only",
        action="store_true",
        help="Run only migrations (create tables)"
    )
    parser.add_argument(
        "--seed-only",
        action="store_true",
        help="Run only seeders (populate initial data)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all tables and recreate from scratch"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=get_db_path(),
        help="Custom database path"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("RAG DATABASE SETUP")
    print("=" * 80)
    print(f"Database: {args.db_path}\n")
    
    try:
        with DatabaseConnection(args.db_path) as conn:
            # Determine what to run
            run_migrations = not args.seed_only
            run_seeders = not args.migrations_only
            
            # Reset if requested
            if args.reset:
                migration_runner = MigrationRunner(conn)
                if not migration_runner.rollback_all():
                    print("\n[WARN] Reset encountered issues but continuing...")
                run_migrations = True
                run_seeders = True
            
            # Run migrations
            if run_migrations:
                migration_runner = MigrationRunner(conn)
                if not migration_runner.run_all():
                    print("\n✗ Migrations failed")
                    return 1
            
            # Run seeders
            if run_seeders:
                seeder_runner = SeederRunner(conn)
                if not seeder_runner.run_all():
                    print("\n✗ Seeders failed")
                    return 1
        
        print("\n" + "=" * 80)
        print("[OK] DATABASE SETUP COMPLETE")
        print("=" * 80)
        print(f"Database Location: {args.db_path}\n")
        
        return 0
    
    except Exception as e:
        print(f"\n[FAIL] Setup failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
