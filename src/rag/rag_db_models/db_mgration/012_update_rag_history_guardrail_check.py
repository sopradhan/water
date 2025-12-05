"""Update rag_history_and_optimization table to allow GUARDRAIL_CHECK event type"""


def run(conn):
    try:
        # Check if the table needs updating by checking the column constraint
        cursor = conn.execute("PRAGMA table_info(rag_history_and_optimization)")
        columns = cursor.fetchall()
        
        # Try inserting a test row with GUARDRAIL_CHECK to see if it works
        try:
            conn.execute('''
                INSERT INTO rag_history_and_optimization 
                (event_type, metrics_json) VALUES ('GUARDRAIL_CHECK_TEST', '{}')
            ''')
            conn.rollback()
            # If we get here, the column already supports GUARDRAIL_CHECK
            print("[✓] rag_history_and_optimization already supports GUARDRAIL_CHECK")
            return
        except:
            # Need to update the table
            pass
        
        # Drop old indexes if they exist
        try:
            conn.execute('DROP INDEX IF EXISTS idx_event_type')
            conn.execute('DROP INDEX IF EXISTS idx_timestamp')
        except:
            pass
        
        # Recreate table with new constraint
        conn.execute('''
CREATE TABLE IF NOT EXISTS rag_history_and_optimization_new (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Event Classification (updated to include GUARDRAIL_CHECK)
    event_type TEXT NOT NULL CHECK (event_type IN ('QUERY', 'HEAL', 'SYNTHETIC_TEST', 'GUARDRAIL_CHECK')),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Query Information (populated if event_type='QUERY' or 'SYNTHETIC_TEST')
    query_text TEXT,
    
    -- Document & Chunk Context (populated if event_type='HEAL' or 'SYNTHETIC_TEST')
    target_doc_id TEXT,
    target_chunk_id TEXT,
    
    -- Performance & Metrics (consolidated as JSON for flexibility)
    metrics_json TEXT NOT NULL,
    
    -- Context & Actions (consolidated as JSON for additional details)
    context_json TEXT,
    
    -- RL Agent Tracking (for reinforcement learning)
    reward_signal FLOAT,
    action_taken TEXT,
    state_before TEXT,
    state_after TEXT,
    
    -- Traceability
    agent_id TEXT DEFAULT 'langgraph_agent',
    user_id TEXT,
    session_id TEXT
)
        ''')
        
        # Copy existing data
        conn.execute('''
INSERT INTO rag_history_and_optimization_new 
SELECT * FROM rag_history_and_optimization
        ''')
        
        # Drop old table
        conn.execute('DROP TABLE rag_history_and_optimization')
        
        # Rename new table
        conn.execute('ALTER TABLE rag_history_and_optimization_new RENAME TO rag_history_and_optimization')
        
        conn.commit()
        print("[✓] Successfully updated rag_history_and_optimization table to support GUARDRAIL_CHECK")
        
    except Exception as e:
        conn.rollback()
        print(f"[!] Migration note: {e}")
