# RAG Database Setup - Quick Start

## 5-Minute Setup

### Step 1: Initialize Database
```bash
cd C:\Users\PRADHAN\OneDrive\Desktop\water
python -m src.rag.rag_db_models.db_setup
```

**Expected Output:**
```
[OK] Database directory ready
[RUN] 008_document_metadata...
[OK] 008_document_metadata completed successfully
[RUN] 009_rag_history...
[OK] 009_rag_history completed successfully
[RUN] 010_chunk_embedding...
[OK] 010_chunk_embedding completed successfully
[RUN] 012_agent_memory...
[OK] 012_agent_memory completed successfully
[RUN] Seeding initial data...
[OK] All seeders completed successfully
[OK] DATABASE SETUP COMPLETE
```

### Step 2: Verify Setup (Optional)
```bash
python verify_db_setup.py
```

**Shows:**
- ✅ All 4 tables created
- ✅ Sample data populated
- ✅ Schema details
- ✅ Record counts

### Step 3: Run Examples (Optional)
```bash
python examples_rag_database_usage.py
```

**Demonstrates:**
- Document registration
- Query logging
- Chunk tracking
- Agent memory management
- Historical analysis

## What Gets Created

### Database File
```
src/data/RAG/rag_metadata.db  (SQLite)
```

### Four Tables
1. **document_metadata** - Document tracking (2 sample docs)
2. **rag_history_and_optimization** - Event logs (5 sample events)
3. **chunk_embedding_data** - Chunk quality (5 sample chunks)
4. **agent_memory** - Agent learning (5 sample memories)

## Usage Examples

### Log a Query
```python
from src.rag.rag_db_models.db import get_connection
from src.rag.rag_db_models import RAGHistoryModel

conn = get_connection()
history = RAGHistoryModel(conn)

history.log_query(
    query_text="What is the pressure?",
    target_doc_id="water_manual_001",
    metrics_json='{"accuracy": 0.95}',
    user_id="user_123"
)
```

### Register a Document
```python
from src.rag.rag_db_models import DocumentMetadataModel

doc = DocumentMetadataModel(conn)
doc.create(
    doc_id="doc_001",
    title="My Document",
    author="Team",
    rbac_namespace="operations"
)
```

### Record Agent Memory
```python
from src.rag.rag_db_models import AgentMemoryModel

memory = AgentMemoryModel(conn)
memory.record_memory(
    agent_id="langgraph_agent",
    memory_type="decision",
    memory_key="strategy_1",
    content='{"method": "retrieval"}',
    importance_score=0.9
)
```

### Track Chunk Quality
```python
from src.rag.rag_db_models import ChunkEmbeddingDataModel

chunk = ChunkEmbeddingDataModel(conn)
chunk.create(
    chunk_id="chunk_001",
    doc_id="doc_001",
    embedding_model="all-mpnet-base-v2",
    quality_score=0.88
)
```

## Command Reference

```bash
# Full setup (create tables + seed data)
python -m src.rag.rag_db_models.db_setup

# Create tables only
python -m src.rag.rag_db_models.db_setup --migrations-only

# Populate data only
python -m src.rag.rag_db_models.db_setup --seed-only

# Reset everything (drop and recreate)
python -m src.rag.rag_db_models.db_setup --reset

# Use custom database path
python -m src.rag.rag_db_models.db_setup --db-path /custom/path/db.sqlite

# Verify installation
python verify_db_setup.py

# Run examples
python examples_rag_database_usage.py
```

## Database Structure

### Tables & Columns

#### document_metadata
```
doc_id (PRIMARY KEY)
title, author, source, summary
company_id, dept_id
rbac_namespace, chunk_strategy, chunk_size_char, overlap_char
metadata_json, rbac_tags, meta_tags
last_ingested (TIMESTAMP)
```

#### rag_history_and_optimization
```
history_id (PRIMARY KEY, AUTO-INCREMENT)
event_type (QUERY, HEAL, SYNTHETIC_TEST, GUARDRAIL_CHECK)
timestamp
query_text, target_doc_id, target_chunk_id
metrics_json, context_json
reward_signal, action_taken, state_before, state_after
agent_id, user_id, session_id
```

#### chunk_embedding_data
```
chunk_id (PRIMARY KEY)
doc_id
embedding_model, embedding_version
quality_score (0.0-1.0)
reindex_count
healing_suggestions, rbac_tags, meta_tags
created_at, last_healed (TIMESTAMP)
```

#### agent_memory
```
id (PRIMARY KEY, AUTO-INCREMENT)
agent_id, memory_type (context, log, decision, performance)
memory_key, content
importance_score (0.0-1.0)
access_count
created_at, updated_at, expires_at (TIMESTAMP)
UNIQUE(agent_id, memory_type, memory_key)
```

## Performance Indexes

11 indexes for fast queries:
- RBAC filtering
- Time-based searches
- Agent/document lookups
- Quality filtering
- Memory type filtering

## Integration

### Works With
- LangGraph RAG Agent
- Healing Agent
- Streamlit Dashboard
- Vector Database (ChromaDB)

### Environment Variables
```env
RAG_DB_PATH=src/data/RAG/rag_metadata.db
CHROMA_DB_PATH=src/data/RAG/chroma_db
```

## Troubleshooting

### "No module named 'src.rag.visualization'"
✅ **Fixed** - Visualization module now created during setup

### "ModuleNotFoundError: No module named 'src.rag'"
- Run from project root: `C:\Users\PRADHAN\OneDrive\Desktop\water`
- Ensure `src/` is in Python path

### "Database file not found"
- Run db_setup first: `python -m src.rag.rag_db_models.db_setup`
- Directory created automatically

### "Table already exists"
- This is normal for idempotent migrations
- Use `--reset` to drop and recreate

## Documentation

- **Full Guide:** `DATABASE_SETUP_GUIDE.md` (450+ lines)
- **Quick Reference:** `QUICK_REFERENCE.md` (250+ lines)
- **Examples:** `examples_rag_database_usage.py` (420 lines)
- **Verification:** `verify_db_setup.py`

## Location

**Database Path:** `src/data/RAG/rag_metadata.db`  
**Models:** `src/rag/rag_db_models/db_models/`  
**Setup Script:** `src/rag/rag_db_models/db_setup.py`  
**Visualization:** `src/rag/visualization/`

## Status

✅ **READY FOR PRODUCTION**

All components tested and working:
- ✅ Database creation
- ✅ Table schema
- ✅ Sample data
- ✅ Indexes
- ✅ Models
- ✅ Examples
- ✅ Verification

---

**Time to Setup:** 2-5 minutes  
**Database Size:** ~100KB (with samples)  
**Tables:** 4  
**Indexes:** 11  
**Sample Records:** 12+
