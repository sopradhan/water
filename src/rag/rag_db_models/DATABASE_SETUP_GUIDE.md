# RAG Database Setup Guide

## Overview

The RAG Database Setup system provides a two-phase initialization framework for SQLite metadata storage that powers the Water Anomaly Detection RAG system.

**Database Location:** `src/data/RAG/rag_metadata.db`

## Architecture

### Two-Phase Setup System

#### Phase 1: Migrations (Schema Creation)
Creates database tables with proper schema, indexes, and constraints. These are **idempotent** - safe to run multiple times.

**Tables Created:**
1. **document_metadata** - Document-level tracking with RBAC and chunking strategy
2. **rag_history_and_optimization** - Unified log for queries, healing, synthetic tests
3. **chunk_embedding_data** - Per-chunk embedding health and quality metrics
4. **agent_memory** - Agent self-reflection, learning, and decision history

#### Phase 2: Seeders (Initial Data Population)
Populates initial data for development, testing, and reference. Runs after migrations.

**Data Seeded:**
- 2 sample documents with metadata
- 1 sample query history record
- 2 sample agent memory records

### Abstraction Layer

The system provides a clean abstraction layer for metadata updates:

```
Application Code
    ↓
Model Classes (AgentMemoryModel, DocumentMetadataModel, etc.)
    ↓
Base Model (BaseModel) - Common DB Operations
    ↓
Connection Manager (get_connection())
    ↓
SQLite Database
```

## Quick Start

### Full Setup (Migrations + Seeders)

```bash
python -m src.rag.rag_db_models.db_setup
```

Output:
```
================================================================================
RAG DATABASE SETUP
================================================================================
Database: C:\Users\...\src\data\RAG\rag_metadata.db

✓ Database directory ready: C:\Users\...\src\data\RAG

================================================================================
PHASE 1: RUNNING MIGRATIONS (Creating Tables)
================================================================================

📋 Running: 008_document_metadata...
✓ 008_document_metadata completed successfully
...

✓ DATABASE SETUP COMPLETE
```

### Advanced Options

#### Migrations Only (Create Tables)
```bash
python -m src.rag.rag_db_models.db_setup --migrations-only
```

#### Seeders Only (Populate Data)
```bash
python -m src.rag.rag_db_models.db_setup --seed-only
```

#### Reset (Drop and Recreate)
```bash
python -m src.rag.rag_db_models.db_setup --reset
```

#### Custom Database Path
```bash
python -m src.rag.rag_db_models.db_setup --db-path "/custom/path/metadata.db"
```

## Database Schema

### 1. document_metadata

Stores document-level information with RBAC and chunking strategy.

```sql
CREATE TABLE document_metadata (
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
    
    -- Consolidated Metadata as JSON
    metadata_json TEXT,      -- {doc_type, version, categories, keywords}
    rbac_tags TEXT,          -- JSON array of RBAC access tags
    meta_tags TEXT,          -- JSON array of semantic tags
    
    -- Tracking
    last_ingested TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Indexes:**
- `idx_document_metadata_rbac_namespace` - For RBAC filtering
- `idx_document_metadata_company_dept` - For organizational filtering

### 2. rag_history_and_optimization

Unified historical log for queries, healing operations, and synthetic tests.

```sql
CREATE TABLE rag_history_and_optimization (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    -- Event Classification
    event_type TEXT NOT NULL,  -- QUERY, HEAL, SYNTHETIC_TEST, GUARDRAIL_CHECK
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Query Information
    query_text TEXT,
    
    -- Document & Chunk Context
    target_doc_id TEXT,
    target_chunk_id TEXT,
    
    -- Performance & Metrics (JSON)
    metrics_json TEXT NOT NULL,
    context_json TEXT,
    
    -- RL Agent Tracking
    reward_signal FLOAT,        -- 0.0-1.0 reward value
    action_taken TEXT,          -- OPTIMIZE, SKIP, REINDEX
    state_before TEXT,          -- JSON system state
    state_after TEXT,           -- JSON system state
    
    -- Traceability
    agent_id TEXT DEFAULT 'langgraph_agent',
    user_id TEXT,
    session_id TEXT
)
```

**Indexes:**
- `idx_rag_history_event_type` - Filter by event type
- `idx_rag_history_timestamp` - Time-based queries
- `idx_rag_history_agent_id` - Agent tracking
- `idx_rag_history_session_id` - Session analysis

### 3. chunk_embedding_data

Tracks per-chunk embedding health, versioning, and quality for RL healing agent.

```sql
CREATE TABLE chunk_embedding_data (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    
    -- Embedding Information
    embedding_model TEXT NOT NULL,
    embedding_version TEXT,
    
    -- Quality & Health
    quality_score FLOAT DEFAULT 0.8,  -- 0.0-1.0
    reindex_count INTEGER DEFAULT 0,
    
    -- RL Agent Suggestions
    healing_suggestions TEXT,  -- JSON: {strategy, reason, suggested_params}
    
    -- RBAC & Metadata
    rbac_tags TEXT,
    meta_tags TEXT,
    
    -- Tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_healed TIMESTAMP
)
```

**Indexes:**
- `idx_chunk_embedding_doc_id` - Document-level queries
- `idx_chunk_embedding_quality` - Quality filtering

### 4. agent_memory

Stores agent memories for self-reflection, learning, and debugging.

```sql
CREATE TABLE agent_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    memory_type TEXT NOT NULL,      -- context, log, decision, performance
    memory_key TEXT NOT NULL,
    content TEXT NOT NULL,
    importance_score REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    
    UNIQUE(agent_id, memory_type, memory_key)
)
```

**Indexes:**
- `idx_agent_memory_agent_id` - Agent lookup
- `idx_agent_memory_type` - Memory type filtering
- `idx_agent_memory_expires` - TTL cleanup

## Usage in Code

### Basic Usage - DocumentMetadataModel

```python
from src.rag.rag_db_models.db import get_connection
from src.rag.rag_db_models import DocumentMetadataModel

# Get connection
conn = get_connection()

# Create model instance
doc_model = DocumentMetadataModel(conn)

# Create/Update document metadata
doc_model.create(
    doc_id="water_pressure_guide_001",
    title="Water Pressure Monitoring Guide",
    author="Engineering Team",
    source="internal_docs",
    summary="Guide for monitoring water system pressure",
    company_id=1,
    dept_id=2,
    rbac_namespace="operations",
    chunk_strategy="recursive_splitter",
    chunk_size_char=512,
    overlap_char=50,
    metadata_json='{"doc_type": "guide", "version": "1.0"}',
    rbac_tags='["engineering", "operations"]',
    meta_tags='["pressure", "monitoring"]'
)
```

### RAG History Logging

```python
from src.rag.rag_db_models import RAGHistoryModel

history_model = RAGHistoryModel(conn)

# Log a query
history_id = history_model.log_query(
    query_text="What are normal pressure ranges?",
    target_doc_id="water_pressure_guide_001",
    metrics_json='{"accuracy": 0.95, "latency_ms": 234}',
    context_json='{"source": "user_query"}',
    agent_id="langgraph_agent",
    user_id="user_123",
    session_id="sess_456"
)
```

### Agent Memory Tracking

```python
from src.rag.rag_db_models import AgentMemoryModel

memory_model = AgentMemoryModel(conn)

# Record agent decision
memory_model.remember(
    agent_id="langgraph_agent",
    memory_type="decision",
    memory_key="pressure_check_strategy",
    content='{"strategy": "cross_reference", "confidence": 0.92}',
    importance_score=0.85
)
```

### Chunk Embedding Quality

```python
from src.rag.rag_db_models import ChunkEmbeddingDataModel

chunk_model = ChunkEmbeddingDataModel(conn)

# Track chunk embedding quality
chunk_model.create(
    chunk_id="chunk_001",
    doc_id="water_pressure_guide_001",
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    embedding_version="1.0",
    quality_score=0.88,
    reindex_count=0,
    healing_suggestions='{"strategy": "re_chunk", "reason": "low_quality"}'
)
```

## Environment Configuration

The database path can be controlled via environment variables:

### .env File

```env
RAG_DB_PATH=src/data/RAG/rag_metadata.db
CHROMA_DB_PATH=src/data/RAG/chroma_db
```

### Default Behavior

If no environment variable is set:
- Database: `src/data/RAG/rag_metadata.db`
- Vector DB: `src/data/RAG/chroma_db`

Both paths are resolved relative to the project root directory.

## Metadata Update Workflow

### Example: Document Ingestion with Full Metadata Capture

```python
import json
from datetime import datetime
from src.rag.rag_db_models.db import get_connection
from src.rag.rag_db_models import (
    DocumentMetadataModel,
    RAGHistoryModel,
    ChunkEmbeddingDataModel
)

conn = get_connection()

# 1. Register Document Metadata
doc_model = DocumentMetadataModel(conn)
doc_model.create(
    doc_id="system_manual_v2",
    title="Water System Operation Manual v2.0",
    author="Operations Team",
    rbac_namespace="operations",
    chunk_strategy="recursive_splitter",
    chunk_size_char=512,
    overlap_char=50,
    rbac_tags=json.dumps(["level_3", "dept_operations"]),
    meta_tags=json.dumps(["operational", "manual"])
)

# 2. Log Ingestion Event
history_model = RAGHistoryModel(conn)
history_model.log_query(
    query_text="Document ingestion: system_manual_v2",
    target_doc_id="system_manual_v2",
    metrics_json=json.dumps({
        "ingestion_time_ms": 1234,
        "chunks_created": 45,
        "embeddings_generated": 45
    }),
    event_type="INGESTION"
)

# 3. Track Chunk Embeddings
chunk_model = ChunkEmbeddingDataModel(conn)
for i in range(45):
    chunk_model.create(
        chunk_id=f"system_manual_v2_chunk_{i}",
        doc_id="system_manual_v2",
        embedding_model="all-mpnet-base-v2",
        embedding_version="1.0",
        quality_score=0.85,
        rbac_tags=json.dumps(["level_3"])
    )

conn.commit()
```

## Verification

Verify the database setup with:

```bash
python verify_db_setup.py
```

This will display:
- All created tables
- Record counts per table
- Complete schema information
- Sample data from each table

## Migration Files Structure

```
src/rag/rag_db_models/
├── db_setup.py                    # Main setup orchestrator
├── db_models/
│   ├── __init__.py
│   ├── base_model.py              # Base class for all models
│   ├── agent_memory_model.py      # Agent memory abstraction
│   ├── chunk_embedding_data_model.py
│   ├── document_metadata_model.py
│   └── rag_history_model.py
├── db_mgration/
│   ├── __init__.py
│   ├── 008_create_document_metadata_table_rag.py
│   ├── 009_create_rag_history_table_rag.py
│   ├── 010_create_chunk_embedding_data_table_rag.py
│   └── 012_create_agent_memory_table.py
├── db_seeder/
│   └── __init__.py                # Future: seeder implementations
├── db/
│   ├── __init__.py
│   └── connection.py              # Connection management
└── __init__.py
```

## Troubleshooting

### Issue: "Database directory not found"
**Solution:** Ensure the parent directory exists or run the setup script which creates it automatically.

### Issue: "Port already in use"
**Solution:** Check if another instance is using the database; close it or use a different path with `--db-path`.

### Issue: "UNIQUE constraint failed"
**Solution:** Data already exists in the table. Use `--reset` to clear and recreate, or use `--migrations-only` to skip seeders.

### Issue: "Module not found" for db_models
**Solution:** Ensure you're running from the project root directory and have `src/` in the Python path.

## Best Practices

1. **Initialize Before Usage** - Always run db_setup before starting the RAG system
2. **Use Transactions** - Wrap multiple operations in transactions for atomicity
3. **Close Connections** - Use context managers (`with get_connection() as conn:`)
4. **Monitor Growth** - Periodically check database size; consider archiving old history
5. **Backup Before Reset** - Always backup the database before running `--reset`

## Performance Considerations

### Indexes
- All tables include optimized indexes for common queries
- Add custom indexes for application-specific filters

### Query Examples

**Find all documents by department:**
```sql
SELECT * FROM document_metadata 
WHERE dept_id = ? AND rbac_namespace = ?
```

**Get recent query history:**
```sql
SELECT * FROM rag_history_and_optimization
WHERE event_type = 'QUERY'
ORDER BY timestamp DESC
LIMIT 100
```

**Identify low-quality chunks:**
```sql
SELECT chunk_id, doc_id, quality_score
FROM chunk_embedding_data
WHERE quality_score < 0.7
ORDER BY quality_score ASC
```

**Agent memory by type:**
```sql
SELECT memory_key, content, importance_score
FROM agent_memory
WHERE agent_id = ? AND memory_type = 'decision'
ORDER BY importance_score DESC
```

## Integration with RAG System

The database setup integrates with:
- **LangGraph RAG Agent** - Logs queries and decisions to history/memory tables
- **Streamlit Dashboard** - Displays metadata and historical analytics
- **Healing Agent** - Tracks chunk quality and healing suggestions
- **Vector Database** - Coordinates with ChromaDB for semantic search

---

**Created:** December 6, 2025
**Purpose:** SQLite metadata management for RAG system
**Location:** `src/data/RAG/rag_metadata.db`
