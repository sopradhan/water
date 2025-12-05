"""Create rag_history_and_optimization pivot table for rag"""


def run(conn):
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
    reward_signal FLOAT,         -- Reward for this event (0.0-1.0, calculated by RL agent)
    action_taken TEXT,           -- Action chosen by RL agent ("OPTIMIZE", "SKIP", "REINDEX")
    state_before TEXT,           -- System state before action (JSON)
    state_after TEXT,            -- System state after action (JSON)
    
    -- Traceability
    agent_id TEXT DEFAULT 'langgraph_agent',
    user_id TEXT,
    session_id TEXT
);
    ''')
    conn.commit()
