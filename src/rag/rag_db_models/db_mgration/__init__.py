"""RAG Database Migrations - Schema creation and updates."""

from .create_document_metadata_table_rag import run as create_document_metadata
from .create_rag_history_table_rag import run as create_rag_history
from .create_chunk_embedding_data_table_rag import run as create_chunk_embedding
from .create_agent_memory_table import run as create_agent_memory

__all__ = [
    "create_document_metadata",
    "create_rag_history",
    "create_chunk_embedding",
    "create_agent_memory",
]

MIGRATIONS = [
    ("008_create_document_metadata_table_rag", create_document_metadata),
    ("009_create_rag_history_table_rag", create_rag_history),
    ("010_create_chunk_embedding_data_table_rag", create_chunk_embedding),
    ("012_create_agent_memory_table", create_agent_memory),
]
