"""RAG Database Models and Setup - SQLite Metadata Management Layer.

This package provides:
1. Database Models (Abstraction Layer)
   - AgentMemoryModel: Agent self-reflection and learning
   - ChunkEmbeddingDataModel: Per-chunk embedding health and quality
   - DocumentMetadataModel: Document-level tracking with RBAC
   - RAGHistoryModel: Query logs, healing actions, synthetic tests

2. Database Migrations (Schema Creation)
   - Idempotent table creation scripts
   - Versioned migrations with rollback support

3. Database Seeders (Initial Data Population)
   - One-time setup data for tables
   - Sample records for testing and reference

4. Database Setup Script (Orchestration)
   - Two-phase initialization: migrations then seeders
   - Reset capability for development
   - CLI interface with options

Usage:
    # Full setup (migrations + seeders)
    python -m src.rag.rag_db_models.db_setup
    
    # Migrations only
    python -m src.rag.rag_db_models.db_setup --migrations-only
    
    # Seeders only
    python -m src.rag.rag_db_models.db_setup --seed-only
    
    # Reset (drop and recreate)
    python -m src.rag.rag_db_models.db_setup --reset

Database Location: src/data/RAG/rag_metadata.db
Vector DB Location: src/data/RAG/chroma_db/
"""

from .db_models import (
    AgentMemoryModel,
    ChunkEmbeddingDataModel,
    DocumentMetadataModel,
    RAGHistoryModel,
)

__all__ = [
    "AgentMemoryModel",
    "ChunkEmbeddingDataModel",
    "DocumentMetadataModel",
    "RAGHistoryModel",
]
