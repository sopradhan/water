"""Migration 012: Create agent_memory table for agent self-reflection and learning"""

def migrate(conn):
    """Create agent_memory table"""
    try:
        conn.execute("""
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
        """)
        
        # Create indexes for faster queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_id 
            ON agent_memory(agent_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_memory_type 
            ON agent_memory(agent_id, memory_type)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_memory_expires 
            ON agent_memory(expires_at)
        """)
        
        conn.commit()
        print("✓ Migration 012: agent_memory table created successfully")
        return True
    
    except Exception as e:
        print(f"✗ Migration 012 failed: {e}")
        return False


def rollback(conn):
    """Drop agent_memory table"""
    try:
        conn.execute("DROP TABLE IF EXISTS agent_memory")
        conn.commit()
        print("✓ Migration 012 rolled back successfully")
        return True
    except Exception as e:
        print(f"✗ Migration 012 rollback failed: {e}")
        return False
