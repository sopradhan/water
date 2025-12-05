"""Create chunk_embedding_data pivot table"""


def run(conn):
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
    
    -- Tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_healed TIMESTAMP
);
''')
    conn.commit()
