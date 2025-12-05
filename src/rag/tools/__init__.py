"""Agent Tools for Enhanced RAG System"""

# Import all tools from submodules
from .ingestion_tools import (
    ingest_documents_from_path_tool,
    extract_document_text_tool,
    extract_metadata_tool,
    chunk_document_tool,
    save_to_vectordb_tool,
    update_metadata_tracking_tool,
    ingest_sqlite_table_tool,
    record_agent_memory_tool,
    retrieve_agent_memory_tool,
    clear_agent_memory_tool,
)

from .retrieval_tools import (
    retrieve_context_tool,
    rerank_context_tool,
    answer_question_tool,
    traceability_tool,
    rewrite_query_tool,
    retrieve_relevant_context_tool,
    rerank_results_tool,
    synthesize_answer_tool,
)

from .healing_tools import (
    check_embedding_health_tool,
    get_context_cost_tool,
    optimize_chunk_size_tool,
)

from .adjust_config_tool import (
    adjust_config_tool,
)

# NEW: Enhanced RAG tools
from .document_classification_tool import (
    enhance_document_metadata_tool,
    DocumentClassificationAgent,
)

from .document_markdown_converter import (
    convert_to_markdown_tool,
    DocumentToMarkdownConverter,
)

from .rbac_retrieval_tool import (
    retrieve_with_rbac_tool,
    generate_response_with_mode_tool,
    UserRole,
    RBACRetrieval,
    ResponseMode,
)

__all__ = [
    # ============================================================================
    # INGESTION TOOLS
    # ============================================================================
    'ingest_documents_from_path_tool',      # Batch ingest from folder
    'extract_document_text_tool',           # Extract text from various formats
    'extract_metadata_tool',                # LLM-based semantic metadata extraction
    'chunk_document_tool',                  # Semantic chunking with overlap
    'save_to_vectordb_tool',                # Generate embeddings & save to ChromaDB
    'update_metadata_tracking_tool',        # Audit trail recording
    'ingest_sqlite_table_tool',             # Ingest database table rows
    'record_agent_memory_tool',             # Memory/context tracking
    
    # ============================================================================
    # RETRIEVAL TOOLS
    # ============================================================================
    'retrieve_context_tool',                # Semantic similarity search
    'rerank_context_tool',                  # LLM-based relevance reranking
    'answer_question_tool',                 # Answer synthesis from context
    'traceability_tool',                    # Audit trail for queries
    'rewrite_query_tool',                   # Query expansion/optimization
    'retrieve_relevant_context_tool',       # Multi-query retrieval
    'rerank_results_tool',                  # Advanced reranking
    'synthesize_answer_tool',               # Answer generation
    
    # ============================================================================
    # HEALING & OPTIMIZATION TOOLS
    # ============================================================================
    'check_embedding_health_tool',          # Embedding quality assessment
    'get_context_cost_tool',                # Token cost estimation
    'optimize_chunk_size_tool',             # Parameter optimization
    'adjust_config_tool',                   # Dynamic config adjustment
    
    # ============================================================================
    # NEW: ENHANCED RAG TOOLS (Generic & Scalable)
    # ============================================================================
    # Document Classification (Meta-prompting with RBAC)
    'enhance_document_metadata_tool',       # Auto-classify intent, dept, role, sensitivity
    'DocumentClassificationAgent',          # Meta-prompting classifier
    
    # Universal Markdown Conversion (Docling-parse powered)
    'convert_to_markdown_tool',             # Convert any format to markdown
    'DocumentToMarkdownConverter',          # Universal converter class
    
    # RBAC-Aware Retrieval (3 response modes + access control)
    'retrieve_with_rbac_tool',              # RBAC-filtered retrieval
    'generate_response_with_mode_tool',     # Mode-specific response generation
    'UserRole',                             # User context with permissions
    'RBACRetrieval',                        # RBAC retrieval orchestrator
    'ResponseMode',                         # Enum: concise|verbose|internal
]
