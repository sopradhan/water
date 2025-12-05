# RAG Database Setup - Quick Reference

## One-Line Setup

```bash
python -m src.rag.rag_db_models.db_setup
```

## Database Location
- **SQLite Metadata:** `src/data/RAG/rag_metadata.db`
- **Vector Database:** `src/data/RAG/chroma_db`

## What Gets Created

### 4 Tables
| Table | Purpose | Records |
|-------|---------|---------|
| `document_metadata` | Document tracking + RBAC + chunking | 2 sample |
| `rag_history_and_optimization` | Query/heal/test logs + RL tracking | 1 sample |
| `chunk_embedding_data` | Embedding health + quality metrics | 0 initial |
| `agent_memory` | Agent self-reflection + learning | 2 sample |

### 8 Performance Indexes
```
document_metadata:
  ├─ rbac_namespace (RBAC filtering)
  └─ company_id, dept_id (org filtering)

rag_history_and_optimization:
  ├─ event_type (filter by event)
  ├─ timestamp DESC (recent first)
  ├─ agent_id (agent tracking)
  └─ session_id (session analysis)

chunk_embedding_data:
  ├─ doc_id (document lookup)
  └─ quality_score (quality filtering)

agent_memory:
  ├─ agent_id (agent lookup)
  ├─ agent_id, memory_type (memory filtering)
  └─ expires_at (TTL cleanup)
```

## Usage Patterns

### Log a Query
```python
from src.rag.rag_db_models.db import get_connection
from src.rag.rag_db_models import RAGHistoryModel

conn = get_connection()
history_model = RAGHistoryModel(conn)

history_model.log_query(
    query_text="What is the pressure in zone A?",
    target_doc_id="water_manual_001",
    metrics_json='{"accuracy": 0.95, "latency_ms": 234}',
    agent_id="langgraph_agent",
    user_id="user_123",
    session_id="sess_456"
)
```

### Register a Document
```python
from src.rag.rag_db_models import DocumentMetadataModel

doc_model = DocumentMetadataModel(conn)
doc_model.create(
    doc_id="doc_001",
    title="System Manual",
    author="Team",
    rbac_namespace="operations",
    chunk_size_char=512,
    rbac_tags='["level_3"]',
    meta_tags='["manual", "operations"]'
)
```

### Record Agent Memory
```python
from src.rag.rag_db_models import AgentMemoryModel

memory_model = AgentMemoryModel(conn)
memory_model.remember(
    agent_id="langgraph_agent",
    memory_type="decision",
    memory_key="strategy_1",
    content='{"method": "cross_reference"}',
    importance_score=0.85
)
```

### Track Chunk Quality
```python
from src.rag.rag_db_models import ChunkEmbeddingDataModel

chunk_model = ChunkEmbeddingDataModel(conn)
chunk_model.create(
    chunk_id="chunk_001",
    doc_id="doc_001",
    embedding_model="all-mpnet-base-v2",
    quality_score=0.88
)
```

## Commands

### Full Setup
```bash
python -m src.rag.rag_db_models.db_setup
```
Creates tables + populates sample data

### Migrations Only
```bash
python -m src.rag.rag_db_models.db_setup --migrations-only
```
Creates tables only, no sample data

### Seeders Only
```bash
python -m src.rag.rag_db_models.db_setup --seed-only
```
Populates data (requires tables to exist)

### Reset Database
```bash
python -m src.rag.rag_db_models.db_setup --reset
```
Drops all tables + recreates everything

### Custom Path
```bash
python -m src.rag.rag_db_models.db_setup --db-path "/path/to/db.sqlite"
```

### Verify Setup
```bash
python verify_db_setup.py
```
Shows tables, schema, and sample data

## Environment Variables

```env
# .env file
RAG_DB_PATH=src/data/RAG/rag_metadata.db
CHROMA_DB_PATH=src/data/RAG/chroma_db
```

## Two-Phase Setup Architecture

### Phase 1: Migrations
```
DatabaseConnection (ensures directory + connection)
    ↓
MigrationRunner (runs all migration functions)
    ├─ 008_create_document_metadata_table_rag
    ├─ 009_create_rag_history_table_rag
    ├─ 010_create_chunk_embedding_data_table_rag
    └─ 012_create_agent_memory_table
```

### Phase 2: Seeders
```
SeederRunner (populates initial data)
    ├─ _seed_document_metadata (2 sample docs)
    ├─ _seed_rag_history (1 sample query)
    └─ _seed_agent_memory (2 sample memories)
```

## Abstraction Layer

```
Application Code
    ↓
[Model Classes]
  • AgentMemoryModel
  • ChunkEmbeddingDataModel
  • DocumentMetadataModel
  • RAGHistoryModel
    ↓
[BaseModel] - Common operations
  • execute(), executemany()
  • fetchone(), fetchall()
  • commit(), rollback()
    ↓
[Connection Manager]
  get_connection() → sqlite3.Connection
    ↓
SQLite Database
```

## File Structure

```
src/rag/rag_db_models/
├── db_setup.py ..................... Main orchestrator
├── __init__.py ..................... Package exports
├── __main__.py ..................... Module entry point
├── DATABASE_SETUP_GUIDE.md ......... Full documentation
├── db_models/
│   ├── base_model.py ............... Base class
│   ├── agent_memory_model.py ....... Agent memory CRUD
│   ├── chunk_embedding_data_model.py Chunk tracking
│   ├── document_metadata_model.py .. Document tracking
│   └── rag_history_model.py ........ Query/event logs
├── db_mgration/
│   ├── 008_create_document_metadata_table_rag.py
│   ├── 009_create_rag_history_table_rag.py
│   ├── 010_create_chunk_embedding_data_table_rag.py
│   └── 012_create_agent_memory_table.py
├── db_seeder/
│   └── (Future: sample data seeders)
└── db/
    ├── connection.py ............... Connection singleton
    └── __init__.py
```

## Integration Points

- **LangGraph Agent** → Logs queries to `rag_history_and_optimization`
- **Healing Agent** → Tracks chunk quality in `chunk_embedding_data`
- **Streamlit Dashboard** → Reads metadata and history for analytics
- **Vector DB** → Coordinates with ChromaDB for embeddings

## Key Features

✓ **Idempotent Migrations** - Safe to run multiple times
✓ **Two-Phase Setup** - Separate schema from data
✓ **Abstraction Layer** - Clean CRUD operations via models
✓ **RBAC Support** - Document-level access control
✓ **Performance Indexes** - 8 optimized indexes
✓ **RL Agent Tracking** - Reward signals and state capture
✓ **TTL Support** - Automatic memory expiration
✓ **JSON Storage** - Flexible metadata via JSON columns

---

**Database:** SQLite (local, portable)
**Purpose:** RAG metadata storage
**Status:** Ready for production use
