"""
LangGraph-based Agentic RAG System

Uses LangGraph for proper workflow orchestration with nodes and edges.
- Ingestion workflow (documents, tables, PDFs, Word, Text files)
- Retrieval workflow with traceability and Guardrails validation
- Optimization workflow
"""
import os
import sys
import json
import time
import uuid
import argparse
import traceback
from typing import Any, Dict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Import Custom Guardrails
from src.rag.guardrails.custom_guardrails import CustomGuardrails

# Import visualization
from src.rag.visualization.langgraph_visualizer import create_visualization, save_visualization
from src.rag.visualization.animated_graph_visualizer import create_ingestion_tracker, create_retrieval_tracker, AnimatedGraphTracker

# Import Enhanced RAG tools (new tools for generic RAG system)
from src.rag.tools.document_markdown_converter import convert_to_markdown_tool
from src.rag.tools.rbac_retrieval_tool import retrieve_with_rbac_tool, generate_response_with_mode_tool, UserRole

# Import RAG tools
# Ingestion tools: extract_metadata_tool, chunk_document_tool, save_to_vectordb_tool, 
#                  update_metadata_tracking_tool, ingest_sqlite_table_tool,
#                  ingest_documents_from_path_tool
# Optional tools: record_agent_memory_tool, extract_document_text_tool (for future use)
from src.rag.tools.ingestion_tools import (
    extract_metadata_tool,
    chunk_document_tool,
    save_to_vectordb_tool,
    update_metadata_tracking_tool,
    ingest_sqlite_table_tool,
    record_agent_memory_tool,
    ingest_documents_from_path_tool,
    extract_document_text_tool,
)
from src.rag.tools.retrieval_tools import (
    retrieve_context_tool,
    rerank_context_tool,
    answer_question_tool,
    traceability_tool,
)
from src.rag.tools.healing_tools import (
    get_context_cost_tool,
    optimize_chunk_size_tool,
)
from src.rag.tools.adjust_config_tool import adjust_config_tool
from src.rag.tools.services.llm_service import LLMService
from src.rag.tools.services.vectordb_service import VectorDBService
from src.rag.config.env_config import EnvConfig
from src.rag.config.prompt_loader import load_prompt, get_system_prompt
from src.rag.agents.healing_agent.rl_healing_agent import RLHealingAgent


class LangGraphRAGState:
    """State object for LangGraph workflow."""
    def __init__(self):
        self.document_text: str = ""
        self.doc_id: str = ""
        self.question: str = ""
        self.metadata: Dict[str, Any] = {}
        self.chunks: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {}
        self.reranked_context: Dict[str, Any] = {}
        self.answer: str = ""
        self.traceability: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.config_updates: Dict[str, Any] = {}
        self.errors: List[str] = []
        
        # New fields for enhanced RAG tools
        self.markdown_text: str = ""  # Converted markdown from document_markdown_converter
        self.classification_metadata: Dict[str, Any] = {}  # From document_classification_tool
        self.rbac_tags: List[str] = []  # RBAC tags for access control
        self.meta_tags: List[str] = []  # Meta tags for semantic retrieval
        self.user_context: Dict[str, Any] = {}  # User role, department for RBAC
        self.response_mode: str = "concise"  # concise|verbose|internal
        self.guardrail_checks: Dict[str, Any] = {}  # Guardrails validation results
        self.is_response_safe: bool = True  # Safety flag after guardrails


class LangGraphRAGAgent:
    """LangGraph-based RAG agent with workflow orchestration."""
    
    def __init__(self):
        """Initialize agent and build workflow graph."""
        self.llm_service, self.vectordb_service = self._init_services()
        self.rl_healing_agent = self._init_rl_agent()
        self.ingestion_graph = self._build_ingestion_graph()
        self.retrieval_graph = self._build_retrieval_graph()
        self.optimization_graph = self._build_optimization_graph()
        self.custom_guardrails = CustomGuardrails()
    
    def _init_services(self):
        """Initialize services using environment configuration."""
        # Get configuration paths from EnvConfig (plug-and-play, no hardcoded paths)
        rag_config_path = EnvConfig.get_rag_config_path()
        
        # Build the LLM config path - EnvConfig handles absolute path conversion
        llm_config_path = os.getenv("LLM_CONFIG_PATH", os.path.join(rag_config_path, "llm_config.json"))
        
        # Ensure absolute path using EnvConfig if needed
        if not os.path.isabs(llm_config_path):
            # EnvConfig already converts relative paths to absolute
            llm_config_path = os.path.join(rag_config_path, os.path.basename(llm_config_path))
        
        with open(llm_config_path, "r") as f:
            llm_config = json.load(f)
       
        llm_service = LLMService(llm_config)
        
        # Use EnvConfig for VectorDB paths
        chroma_db_path = EnvConfig.get_chroma_db_path()
        # Note: Collection name created dynamically per company_id via _get_collection_name()
        vectordb_service = VectorDBService(
            persist_directory=chroma_db_path,
            collection_name=os.getenv("CHROMA_COLLECTION", "rag_embeddings_default")
        )
        
        return llm_service, vectordb_service

    def _init_rl_agent(self):
        """Initialize RL Healing Agent using environment configuration."""
        try:
            db_path = EnvConfig.get_db_path()
            return RLHealingAgent(db_path, llm_service=self.llm_service)
        except Exception as e:
            print(f"Warning: Failed to initialize RL agent: {e}")
            return None

    def _get_collection_name(self, company_id: int = None) -> str:
        """Generate ChromaDB collection name based on company_id.
        
        Args:
            company_id: Company ID (default: from environment or 'default')
            
        Returns:
            Collection name in format: tenant_{company_id} or rag_embeddings_default
        """
        if company_id:
            return f"tenant_{company_id}"
        else:
            env_collection = os.getenv("CHROMA_COLLECTION", None)
            return env_collection if env_collection else "rag_embeddings_default"

    def generate_embeddings_parallel(self, chunks: list, llm_service, max_workers: int = 4) -> list:
        """
        Generate embeddings for multiple chunks in parallel.
        
        OPTIMIZATION: Uses ThreadPoolExecutor for concurrent embedding generation
        
        Args:
            chunks: List of chunk dicts with 'text' key
            llm_service: Service for embedding generation
            max_workers: Number of parallel workers (default: 4)
        
        Returns:
            List of embeddings in same order as input chunks
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        embeddings = [None] * len(chunks)
        
        if not chunks:
            print(f"[DEBUG EMBEDDINGS] No chunks to embed")
            return embeddings
        
        print(f"[DEBUG EMBEDDINGS] Starting parallel embedding generation for {len(chunks)} chunks with {max_workers} workers")
        
        def generate_chunk_embedding(chunk_info):
            chunk_index, chunk_text = chunk_info
            try:
                embedding = llm_service.generate_embedding(chunk_text)
                return chunk_index, embedding
            except Exception as e:
                print(f"[ERROR] Failed to generate embedding for chunk {chunk_index}: {e}")
                return chunk_index, None
        
        # Prepare tasks: (index, text) tuples
        tasks = [(i, chunk.get('text', '').strip()) for i, chunk in enumerate(chunks) if chunk.get('text', '').strip()]
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {executor.submit(generate_chunk_embedding, task): task[0] for task in tasks}
                
                # Collect results as they complete
                for future in as_completed(future_to_index):
                    chunk_index, embedding = future.result()
                    if embedding is not None:
                        embeddings[chunk_index] = embedding
        except Exception as e:
            print(f"[ERROR] Parallel embedding generation failed: {e}, falling back to sequential")
            # Fallback to sequential if parallelization fails
            for i, chunk in enumerate(chunks):
                chunk_text = chunk.get('text', '').strip()
                if chunk_text:
                    embeddings[i] = llm_service.generate_embedding(chunk_text)
        
        return embeddings

    def _build_ingestion_graph(self):
        """Build ingestion workflow graph.
        
        INGESTION WORKFLOW STAGES:
        1. extract_metadata_node: Uses extract_metadata_tool with LLM to generate semantic metadata
        2. chunk_document_node: Uses chunk_document_tool to split text into semantic chunks
        3. save_vectordb_node: Uses save_to_vectordb_tool to:
           - Generate embeddings for each chunk using LLM
           - Store embeddings in VectorDB (ChromaDB)
           - Persist metadata to SQLite (if available)
        4. update_tracking_node: Uses update_metadata_tracking_tool to record audit trail
        
        INGESTION PATHS:
        - Single Document: ingest_document(text, doc_id)
        - Table Ingestion: ingest_sqlite_table_tool for database rows
        - Batch Folder: ingest_documents_from_path_tool discovers files, then loops through each
        
        ENHANCED WITH NEW TOOLS:
        0. convert_markdown_node: Convert document to markdown using docling-parse
        0.5. generate_rbac_tags_node: Generate RBAC tags directly from company_id and department_id
        """
        graph = StateGraph(dict)
        
        # Define nodes
        def convert_markdown_node(state):
            """
            INGESTION STAGE 0: Convert Document to Markdown
            
            TOOL: convert_to_markdown_tool (using docling-parse)
            WHEN: Very first step - universal document normalization
            INPUT: source_path or source_type, content, title
            PROCESS:
              1. Detects document format (PDF, DOCX, XLSX, CSV, TXT, etc.)
              2. Uses docling-parse for professional-grade conversion
              3. Preserves: structure, tables, headers, hierarchy
              4. Outputs: normalized markdown format
            OUTPUT: state["markdown_text"] with normalized markdown
            PURPOSE: Ensure consistent format for all document types before processing
            BENEFIT: Better quality for semantic chunking and embeddings
            """
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("convert_markdown", "Convert Markdown", {
                        "text_length": len(state.get("text", ""))
                    })
                
                # Check if we have a source_path or need to convert from text
                source_path = state.get("source_path")
                source_type = state.get("source_type", "txt")
                title = state.get("title", state.get("doc_id", "document"))
                # Note: company_id and dept_id flow through state dict and passed to save_vectordb_node
                if source_path and os.path.exists(source_path):
                    # Convert from file path
                    markdown_response = convert_to_markdown_tool.invoke({
                        "source_type": source_type,
                        "source_path": source_path,
                        "title": title,
                        "auto_detect": True
                    })
                else:
                    # Convert from text content (already have document_text or text)
                    doc_text = state.get("document_text") or state.get("text", "")
                    markdown_response = convert_to_markdown_tool.invoke({
                        "content": doc_text,
                        "source_type": "txt",
                        "title": title
                    })
                
                markdown_data = json.loads(markdown_response) if isinstance(markdown_response, str) else markdown_response
                
                if markdown_data.get("success"):
                    state["markdown_text"] = markdown_data["markdown"]
                    state["text"] = markdown_data["markdown"]  # Update text field for downstream
                    state["status"] = "markdown_converted"
                else:
                    # Fallback: use original text
                    state["markdown_text"] = state.get("document_text") or state.get("text", "")
                    state["errors"] = state.get("errors", []) + [f"Markdown conversion failed: {markdown_data.get('error', 'unknown')}"]
                
                if tracker:
                    tracker.node_end("convert_markdown", {
                        "markdown_length": len(state.get("markdown_text", "")),
                        "status": state.get("status")
                    }, "success" if not state.get("errors") else "error")
                    tracker.edge_traversal("convert_markdown", "generate_rbac_tags", ["markdown_text", "text"])
                    
            except Exception as e:
                state["markdown_text"] = state.get("document_text") or state.get("text", "")
                state["errors"] = state.get("errors", []) + [f"Markdown conversion failed: {e}"]
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("convert_markdown", {"error": str(e)}, "error")
            
            return state

        def generate_rbac_tags_node(state: dict) -> dict:
            """
            INGESTION STAGE 0.5: Generate RBAC Tags and Meta Tags from Company & Department IDs
            
            WHEN: After markdown conversion, before chunking
            INPUT: company_id, dept_id, title, markdown_text from state
            PROCESS:
              1. Extract company_id and dept_id from state
              2. Generate RBAC tags: rbac:{company_id}:{dept_id}
              3. Extract semantic keywords from document (LLM)
              4. Generate Meta tags: static (dept, company) + dynamic (keywords)
              5. Store tags in state for vectordb storage
            OUTPUT: state["rbac_tags"] with RBAC access control tags (single string, not list)
                    state["meta_tags"] with semantic metadata tags (5 keywords from content)
            PURPOSE: Enable RBAC-based access control and semantic retrieval with content-aware tags
            BENEFIT: Dynamic tagging based on document content + fast deterministic RBAC
            """
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("generate_rbac_tags", "Generate RBAC Tags", {
                        "company_id": state.get("company_id"),
                        "dept_id": state.get("dept_id")
                    })
                
                company_id = state.get("company_id", "default")
                dept_id = state.get("dept_id", "default")
                doc_id = state.get("doc_id", "")
                title = state.get("title", "document")
                markdown_text = state.get("markdown_text", "")
                
                # Generate RBAC tag as a SINGLE STRING, not a list
                # Format: rbac:{company_id}:{dept_id}
                # ChromaDB metadata values must be: str, int, float, bool, SparseVector, or None (NOT lists)
                rbac_tag = f"rbac:{company_id}:{dept_id}"
                print(f"[DEBUG RBAC] INGESTION: RBAC tag set to company:dept format - Tag: {rbac_tag}")
                
                # Extract semantic keywords from document content using LLM
                # These are domain-specific tags like "policy", "incident", "vacation"
                semantic_keywords = []
                if markdown_text and len(markdown_text.strip()) > 100:  # Only if substantial content
                    try:
                        # Use LLM to extract 5 key semantic tags from document
                        keyword_prompt = load_prompt("extraction", "document_keywords", content=markdown_text[:2000])
                        keyword_response = self.llm_service.generate_response(keyword_prompt)
                        
                        # Parse LLM response to extract tags
                        try:
                            import re
                            # Try to extract JSON array from response
                            match = re.search(r'\[.*?\]', keyword_response, re.DOTALL)
                            if match:
                                keywords_str = match.group(0)
                                semantic_keywords = json.loads(keywords_str)
                                # Ensure we have exactly 5 tags and they are lowercase
                                semantic_keywords = [str(kw).lower().strip() for kw in semantic_keywords[:5]]
                        except:
                            # Fallback: split by comma or space
                            keywords_text = keyword_response.lower()
                            semantic_keywords = [kw.strip() for kw in keywords_text.replace('[', '').replace(']', '').replace('"', '').split(',') if kw.strip()][:5]
                    except Exception as e:
                        print(f"[WARNING] Semantic keyword extraction failed: {e}")
                        semantic_keywords = []
                
                # Generate Meta tags: ONLY semantic keywords from document content
                # Format: meta:keyword:{keyword} - extracted from document, no assumptions
                # These are semantic descriptors of WHAT the document is about (not WHO can see it)
                meta_tags = []
                
                # Add semantic keywords as meta tags (up to 5 keywords)
                # These are extracted from the document content via LLM
                # Examples: "policy", "incident", "vacation", "approval", "budget"
                meta_tags_list = []
                for keyword in semantic_keywords[:5]:
                    if keyword and len(keyword) > 0:
                        meta_tags_list.append(f"meta:keyword:{keyword}")
                
                # Store as SINGLE STRING (join with semicolon) since ChromaDB metadata values must be str, not list
                state["rbac_tags"] = rbac_tag  # Single string: "rbac:1:10"
                state["meta_tags"] = ";".join(meta_tags_list) if meta_tags_list else ""  # Single string
                state["semantic_keywords"] = semantic_keywords  # Store for reference
                state["status"] = "rbac_tags_generated"
                
                # DEBUG: Print tag creation
                print(f"[TAGS] Doc: {state.get('doc_id')}, Company: {company_id}, Dept: {dept_id}")
                print(f"[TAGS]   RBAC: {state['rbac_tags']}")
                print(f"[TAGS]   Meta: {state['meta_tags']}")
                
                if tracker:
                    tracker.node_end("generate_rbac_tags", {
                        "rbac_tags": state.get("rbac_tags"),
                        "meta_tags": state.get("meta_tags"),
                        "semantic_keywords": semantic_keywords
                    }, "success")
                    tracker.edge_traversal("generate_rbac_tags", "extract_metadata", ["rbac_tags", "meta_tags", "markdown_text"])
                
            except Exception as e:
                # Fallback to generic RBAC tag
                company_id = state.get("company_id", "default")
                dept_id = state.get("dept_id", "default")
                state["rbac_tags"] = f"rbac:{company_id}:{dept_id}"
                state["meta_tags"] = ""
                state["semantic_keywords"] = []
                state["errors"] = state.get("errors", []) + [f"RBAC/Meta tag generation failed: {e}"]
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("generate_rbac_tags", {"error": str(e)}, "error")
            
            return state

        def extract_metadata_node(state):
            """
            INGESTION STAGE 1: Extract Semantic Metadata
            
            TOOL: extract_metadata_tool
            WHEN: After RBAC tag generation
            INPUT: text (normalized markdown content), llm_service (for LLM processing)
            PROCESS:
              1. Uses LLM to analyze markdown document text
              2. Extracts: title, summary (2-3 sentences), keywords (5-10), topics, doc_type
              3. Returns structured JSON metadata
            OUTPUT: state["metadata"] with LLM-extracted fields
            PURPOSE: Enable semantic understanding and search of documents
            FALLBACK: Returns basic metadata if processing fails
            """
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("extract_metadata", "Extract Metadata", {
                        "text_length": len(state.get("markdown_text", ""))
                    })
                
                text = state.get("markdown_text") or state["text"]
                
                # Use LLM to extract metadata
                meta_response = extract_metadata_tool.invoke({
                    "text": text,
                    "llm_service": self.llm_service
                })
                state["metadata"] = json.loads(meta_response) if isinstance(meta_response, str) else meta_response
                state["status"] = "metadata_extracted"
                
                if tracker:
                    tracker.node_end("extract_metadata", {
                        "metadata": state.get("metadata", {})
                    }, "success")
                    tracker.edge_traversal("extract_metadata", "chunk_document", ["text", "metadata"])
                    
            except Exception as e:
                # Fallback metadata
                state["metadata"] = {
                    "success": True,
                    "metadata": {
                        "title": state.get("doc_id", "Document"),
                        "summary": "Document processed",
                        "keywords": [],
                        "topics": [],
                        "doc_type": "text"
                    }
                }
                state["errors"] = state.get("errors", []) + [f"Metadata extraction failed: {e}"]
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("extract_metadata", {"error": str(e)}, "error")
            
            return state

        def chunk_document_node(state):
            """
            INGESTION STAGE 2: Split Into Semantic Chunks
            
            TOOL: chunk_document_tool
            WHEN: After metadata extraction
            INPUT: text (document content), doc_id (unique identifier)
            PROCESS:
              1. Uses RecursiveCharacterTextSplitter with Markdown-aware separators
              2. Default: chunk_size=500 chars, overlap=50 chars
              3. Splits on: \\n\\n##, \\n\\n, \\n, ". ", " ", ""
            OUTPUT: state["chunks"] with num_chunks and list of chunks
            PURPOSE: Create semantically coherent pieces for embedding
            """
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("chunk_document", "Chunk Document", {
                        "text_length": len(state.get("text", ""))
                    })
                
                chunks_response = chunk_document_tool.invoke({"text": state["text"], "doc_id": state["doc_id"]})
                state["chunks"] = json.loads(chunks_response) if isinstance(chunks_response, str) else chunks_response
                state["status"] = "chunks_created"
                
                if tracker:
                    chunks_count = state.get("chunks", {}).get("num_chunks", 0)
                    tracker.node_end("chunk_document", {
                        "chunks_count": chunks_count
                    }, "success")
                    tracker.edge_traversal("chunk_document", "save_vectordb", ["chunks", "doc_id"])
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Chunking failed: {e}"]
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("chunk_document", {"error": str(e)}, "error")
            return state

        def save_vectordb_node(state):
            """
            INGESTION STAGE 3: Generate Embeddings (PARALLELIZED) & Store in VectorDB
            
            OPTIMIZATION: Uses parallel embedding generation before storing to VectorDB
            
            TOOL: save_to_vectordb_tool (modified to accept pre-generated embeddings)
            WHEN: After chunking
            INPUT: chunks (from chunk_document_tool), doc_id, company_id, dept_id, services, metadata
            PROCESS:
              1. Generate embeddings in PARALLEL for all chunks using ThreadPoolExecutor
              2. Batch adds to ChromaDB vector store with company-specific collection:
                 - collection_name = rag_company_{company_id}
                 - chunk_ids, embeddings (pre-generated), text, metadatas
              3. Optionally persists to SQLite (DocumentMetadataModel, ChunkEmbeddingDataModel)
            OUTPUT: state["save_result"] with chunks_saved count and collection_name
            PURPOSE: Make documents searchable via semantic similarity with PARALLELIZED embedding generation
            """
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("save_vectordb", "Save to VectorDB", {
                        "company_id": state.get("company_id"),
                        "dept_id": state.get("dept_id"),
                        "chunks_count": len(state.get("chunks", {}).get("chunks", []))
                    })
                
                meta_json = json.dumps(state.get("metadata", {}).get("metadata", {}))
                company_id = state.get("company_id")
                dept_id = state.get("dept_id")
                doc_id = state.get("doc_id")
                collection_name = self._get_collection_name(company_id)
                rbac_tags = state.get("rbac_tags", "")
                meta_tags = state.get("meta_tags", "")
                
                # Debug output
                print(f"\n[DEBUG] === INGESTION: Saving to VectorDB ===")
                print(f"[DEBUG] Doc ID: {doc_id}")
                print(f"[DEBUG] Company ID: {company_id}, Dept ID: {dept_id}")
                print(f"[DEBUG] Collection: {collection_name}")
                print(f"[DEBUG] RBAC Tags: {rbac_tags} (type: {type(rbac_tags).__name__})")
                print(f"[DEBUG] Meta Tags: {meta_tags} (type: {type(meta_tags).__name__})")
                
                # Create VectorDB service with company-specific collection
                vectordb_service = VectorDBService(
                    persist_directory=self.vectordb_service.persist_directory,
                    collection_name=collection_name
                )
                
                # OPTIMIZATION: Generate embeddings in parallel
                chunks_data = state.get("chunks", {})
                chunk_list = chunks_data.get("chunks", [])
                
                print(f"[DEBUG SAVE] Ingesting doc_id={doc_id}, company_id={company_id}, collection={collection_name}")
                print(f"[DEBUG SAVE] Total chunks to save: {len(chunk_list)}")
                
                if chunk_list:
                    # Generate all embeddings in parallel
                    embeddings_list = self.generate_embeddings_parallel(chunk_list, self.llm_service, max_workers=4)
                    
                    # Store embeddings back in chunks for save_to_vectordb_tool
                    embedded_count = 0
                    for i, embedding in enumerate(embeddings_list):
                        if i < len(chunk_list) and embedding is not None:
                            chunk_list[i]["embedding"] = embedding
                            embedded_count += 1
                    
                    print(f"[DEBUG SAVE] Embeddings generated: {embedded_count}/{len(chunk_list)}")
                
                print(f"[DEBUG SAVE] RBAC tags: {state.get('rbac_tags', '')}")
                print(f"[DEBUG SAVE] Meta tags: {state.get('meta_tags', '')}")
                
                save_response = save_to_vectordb_tool.invoke({
                    "chunks": json.dumps(state["chunks"]),
                    "doc_id": doc_id,
                    "company_id": company_id,
                    "dept_id": dept_id,
                    "rbac_namespace": "root",  # All documents get root namespace for root user access
                    "rbac_tags": rbac_tags,  # Regular users filter by rbac_tags (e.g., "rbac:1:1")
                    "meta_tags": meta_tags,  # Now a string, not list
                    "llm_service": self.llm_service,
                    "vectordb_service": vectordb_service,
                    "metadata": meta_json,
                    "pre_generated_embeddings": True  # Flag to skip embedding generation in tool
                })
                state["save_result"] = json.loads(save_response) if isinstance(save_response, str) else save_response
                state["collection_name"] = collection_name
                
                save_result = state.get("save_result", {})
                print(f"[DEBUG SAVE] Save result: success={save_result.get('success')}, chunks_saved={save_result.get('chunks_saved', 0)}, error={save_result.get('error', '')}")
                print(f"[DEBUG SAVE] Collection now has {vectordb_service.collection.count()} total vectors")
                state["status"] = "saved_to_vectordb"
                print(f"[DEBUG] Save result: {state['save_result']}")
                print(f"[DEBUG] === END VectorDB Save ===\n")
                
                if tracker:
                    tracker.node_end("save_vectordb", {
                        "chunks_saved": save_result.get("chunks_saved", 0),
                        "collection_name": collection_name,
                        "total_vectors": vectordb_service.collection.count()
                    }, "success" if save_result.get("success") else "error")
                    tracker.edge_traversal("save_vectordb", "update_tracking", ["save_result", "chunks_saved"])
            except Exception as e:
                print(f"[DEBUG ERROR] VectorDB save failed: {e}")
                state["errors"] = state.get("errors", []) + [f"VectorDB save failed: {e}"]
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("save_vectordb", {"error": str(e)}, "error")
            return state

        def update_tracking_node(state):
            """
            INGESTION STAGE 4: Record Audit Trail & Finalize
            
            TOOL: update_metadata_tracking_tool
            WHEN: After VectorDB persistence
            INPUT: doc_id, company_id, dept_id, source_path, rbac_tags, meta_tags, metadata, chunks_saved
            PROCESS:
              1. Updates DocumentTrackingModel in SQLite (if available)
              2. Records ingestion audit trail with timestamp
              3. Marks ingestion_status = 'COMPLETED'
              4. Stores metadata_tags (rbac_tags, meta_tags) as JSON for later querying
            OUTPUT: state["tracking_result"] with success status
            PURPOSE: Create immutable audit trail for compliance & debugging
            """
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("update_tracking", "Update Audit Trail", {
                        "doc_id": state.get("doc_id")
                    })
                
                meta_json = json.dumps(state.get("metadata", {}).get("metadata", {}))
                chunks_saved = state.get("save_result", {}).get("chunks_saved", 0)
                
                # Prepare tags JSON for tracking
                tags_json = json.dumps({
                    "rbac_tags": state.get("rbac_tags", []),
                    "meta_tags": state.get("meta_tags", [])
                })
                
                update_response = update_metadata_tracking_tool.invoke({
                    "doc_id": state["doc_id"],
                    "company_id": state.get("company_id"),
                    "dept_id": state.get("dept_id"),
                    "source_path": "document_ingestion",
                    "rbac_namespace": "root",
                    "metadata": meta_json,
                    "tags": tags_json,
                    "chunks_saved": chunks_saved
                })
                state["tracking_result"] = json.loads(update_response) if isinstance(update_response, str) else update_response
                state["status"] = "completed"
                
                if tracker:
                    tracker.node_end("update_tracking", {
                        "tracking_result": state.get("tracking_result"),
                        "status": state.get("status")
                    }, "success")
                    tracker.edge_traversal("update_tracking", "END", ["status", "tracking_result"])
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Tracking update failed: {e}"]
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("update_tracking", {"error": str(e)}, "error")
            return state

        # Add nodes to graph
        # Add nodes to graph
        graph.add_node("convert_markdown", convert_markdown_node)
        graph.add_node("generate_rbac_tags", generate_rbac_tags_node)
        graph.add_node("extract_metadata", extract_metadata_node)
        graph.add_node("chunk_document", chunk_document_node)
        graph.add_node("save_vectordb", save_vectordb_node)
        graph.add_node("update_tracking", update_tracking_node)

        # Add edges: Enhanced pipeline with docling markdown conversion and direct RBAC tags
        graph.add_edge(START, "convert_markdown")              # Docling markdown conversion
        graph.add_edge("convert_markdown", "generate_rbac_tags") # Generate RBAC from company/dept IDs
        graph.add_edge("generate_rbac_tags", "extract_metadata") # Then extract metadata
        graph.add_edge("extract_metadata", "chunk_document")
        graph.add_edge("chunk_document", "save_vectordb")
        graph.add_edge("save_vectordb", "update_tracking")
        graph.add_edge("update_tracking", END)

        return graph.compile()

    def _build_retrieval_graph(self):
        """Build retrieval workflow graph with intelligent healing integration.
        
        RETRIEVAL WORKFLOW STAGES:
        1. retrieve_context_node: Uses retrieve_context_tool to get top-k relevant docs
        2. rerank_context_node: Uses rerank_context_tool to sort by relevance
        3. check_optimization: Decides if optimization needed (RL or heuristic)
        4. optimize_context: Uses get_context_cost_tool + optimize_chunk_size_tool
        5. answer_question_node: Uses answer_question_tool to generate answer
        6. traceability_node: Uses traceability_tool for audit trail
        
        RESPONSE MODES:
        - concise: User-friendly, applies hallucination_check + security_incident_policy guardrails
        - internal: System integration, applies hallucination_check guardrails only
        - verbose: Engineering view, no guardrails, shows all debug info
        """
        graph = StateGraph(dict)

        def retrieve_context_node(state):
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("retrieve_context", "Retrieve Context", {
                        "question": state.get("question", "")[:100],
                        "company_id": state.get("company_id")
                    })
                
                response_mode = state.get("response_mode", "concise")
                show_debug = response_mode in ["verbose", "internal"]
                
                if show_debug:
                    print(f"\n[🔍 RETRIEVE CONTEXT NODE - {response_mode.upper()} MODE]")
                    print(f"  Question: {state['question'][:80]}...")
                    print(f"  Retrieving top-k=5 relevant documents with RBAC filtering...")
                
                # STEP 1: Extract intent from user query using meta prompting
                # This helps drill down the semantic search with 5 key keywords
                user_intent_keywords = []
                try:
                    intent_prompt = load_prompt("intent", "user_intent_keywords", question=state['question'])
                    intent_response = self.llm_service.generate_response(intent_prompt)
                    
                    try:
                        import re
                        match = re.search(r'\[.*?\]', intent_response, re.DOTALL)
                        if match:
                            keywords_str = match.group(0)
                            user_intent_keywords = json.loads(keywords_str)
                            user_intent_keywords = [str(kw).lower().strip() for kw in user_intent_keywords[:5]]
                    except:
                        keywords_text = intent_response.lower()
                        user_intent_keywords = [kw.strip() for kw in keywords_text.replace('[', '').replace(']', '').replace('"', '').split(',') if kw.strip()][:5]
                    
                    state["user_intent_keywords"] = user_intent_keywords
                    if show_debug:
                        print(f"  Extracted Intent Keywords: {user_intent_keywords}")
                except Exception as e:
                    if show_debug:
                        print(f"  [WARNING] Intent extraction failed: {e}")
                    user_intent_keywords = []
                
                # STEP 2: Build RBAC filter based on user's company/department context
                # Get user's company and department from state (set during ask_question initialization)
                user_company_id = state.get("user_company_id")
                user_dept_id = state.get("user_dept_id")
                is_root_user = state.get("is_root_user", False)  # Check if user has root/admin access
                
                # Build RBAC filter for ChromaDB where clause
                # Root users: filter by rbac_namespace='root' (grants access to all documents)
                # Regular users: filter by rbac_tags containing company:dept tag (e.g., "rbac:1:1")
                rbac_filter = None
                if is_root_user:
                    # Root user - can see all documents marked with root namespace
                    rbac_filter = {
                        "rbac_namespace": {"$eq": "root"}
                    }
                    if show_debug:
                        print(f"  [ROOT ACCESS] User can see all documents in root namespace")
                        print(f"  RBAC Filter: {rbac_filter}")
                else:
                    # Regular user - can only see their company/department data
                    # Filter by rbac_tags field which contains the actual company:dept tag
                    if user_company_id is not None and user_dept_id is not None:
                        user_rbac_tag = f"rbac:{user_company_id}:{user_dept_id}"
                        rbac_filter = {
                            "rbac_tags": {"$eq": user_rbac_tag}
                        }
                        if show_debug:
                            print(f"  [TENANT RBAC] User company_id={user_company_id}, dept_id={user_dept_id}")
                            print(f"  RBAC Filter: {rbac_filter} (rbac_tags={user_rbac_tag})")
                    else:
                        if show_debug:
                            print(f"  [WARNING] Regular user missing company_id={user_company_id} or dept_id={user_dept_id}")
                            print(f"  No RBAC filter applied - using default access")
                
                # Get company-specific collection - use user_company_id for tenant isolation
                company_id = user_company_id or state.get("company_id")
                
                # Determine which collections to query
                # Root users query ALL collections to see parent_company:root documents
                # Regular users query only their tenant collection
                collections_to_query = []
                if is_root_user:
                    # Root user - query ALL collections to find parent_company:root documents
                    print(f"[RETRIEVAL] ROOT USER ACCESS - will search across all tenant collections")
                    # Get all collections from VectorDB
                    try:
                        all_collections = self.vectordb_service.client.list_collections()
                        collections_to_query = [c.name for c in all_collections if c.name.startswith("tenant_")]
                        print(f"[RETRIEVAL] Root user can access collections: {collections_to_query}")
                    except Exception as e:
                        print(f"[WARNING] Failed to list collections: {e}, falling back to company_id collection")
                        if company_id:
                            collections_to_query = [self._get_collection_name(company_id)]
                else:
                    # Regular user - query only their tenant collection
                    collection_name = self._get_collection_name(company_id)
                    collections_to_query = [collection_name]
                    print(f"[RETRIEVAL] REGULAR USER - accessing collection: {collection_name}")
                
                # DEBUG: Print collection selection
                print(f"[RETRIEVAL] Question: {state['question'][:80]}")
                print(f"[RETRIEVAL] Company ID: {company_id}, Collections to query: {collections_to_query}")
                
                # STEP 3: Enhanced retrieval with intent keywords + RBAC filtering
                # Use modified question with intent keywords for better semantic search
                enhanced_question = state['question']
                if user_intent_keywords:
                    # Append intent keywords to question for better embedding matching
                    enhanced_question = f"{state['question']} {' '.join(user_intent_keywords)}"
                
                # Retrieve from all applicable collections and merge results
                all_context = {"context": []}
                
                for collection_name in collections_to_query:
                    # Create VectorDB service for this collection
                    vectordb_service = VectorDBService(
                        persist_directory=self.vectordb_service.persist_directory,
                        collection_name=collection_name
                    )
                    
                    print(f"[DEBUG RETRIEVE] Querying collection={collection_name}")
                    print(f"[DEBUG RETRIEVE] Company ID={user_company_id}, Dept ID={user_dept_id}, User ID={state.get('user_id')}")
                    print(f"[DEBUG RETRIEVE] RBAC Filter: {rbac_filter}")
                    print(f"[DEBUG RETRIEVE] Question: {enhanced_question[:100]}...")
                    print(f"[DEBUG RETRIEVE] Collection has {vectordb_service.collection.count()} vectors")
                    
                    # TOOL: retrieve_context_tool
                    # WHEN: Question answering starts - need to find relevant documents
                    # INPUT: question (user query with intent keywords), k (top results from state), 
                    #        services (LLM for embedding, VectorDB for search)
                    #        where_clause (RBAC filter for tenant isolation)
                    # PROCESS: 
                    #   1. Generates embedding for the enhanced question (original + intent keywords)
                    #   2. Performs semantic similarity search in VectorDB with company-specific collection
                    #   3. Applies WHERE clause filter to enforce RBAC access control
                    #   4. Returns top-k most similar chunks with similarity scores (only accessible to user)
                    top_k_value = state.get("top_k", 5)  # Use top_k from state, default to 5
                    context_response = retrieve_context_tool.invoke({
                        "question": enhanced_question,
                        "llm_service": self.llm_service,
                        "vectordb_service": vectordb_service,
                        "k": top_k_value,
                        "where_clause": rbac_filter,  # RBAC filtering applied here
                        "user_intent_keywords": user_intent_keywords  # For logging/tracking
                    })
                    collection_context = json.loads(context_response) if isinstance(context_response, str) else context_response
                    
                    # Merge results from this collection
                    if collection_context.get("context"):
                        all_context["context"].extend(collection_context["context"])
                        print(f"[DEBUG RETRIEVE] Retrieved {len(collection_context['context'])} chunks from {collection_name}")
                
                # Use merged results from all collections
                state["context"] = all_context
                state["status"] = "context_retrieved"
                state["collection_name"] = ", ".join(collections_to_query)  # Log all queried collections
                
                retrieved_count = len(state["context"].get("context", []))
                print(f"[DEBUG RETRIEVE] Retrieved {retrieved_count} total chunks from {len(collections_to_query)} collection(s): {collections_to_query}")
                
                state["retrieval_quality"] = min(1.0, retrieved_count / max(state.get("top_k", 5), 1))  # Normalize to 0-1
                state["rbac_filter_applied"] = rbac_filter
                state["is_root_access"] = is_root_user
                
                if tracker:
                    tracker.node_end("retrieve_context", {
                        "retrieved_count": retrieved_count,
                        "collection_name": state["collection_name"],
                        "retrieval_quality": state.get("retrieval_quality", 0),
                        "is_root_access": is_root_user
                    }, "success")
                    tracker.edge_traversal("retrieve_context", "rerank_context", ["context", "retrieval_quality"])
                
                if show_debug:
                    ctx_items = state["context"].get("context", [])
                    print(f"  ✓ Retrieved {len(ctx_items)} total documents from {len(collections_to_query)} collection(s)")
                    print(f"    Collections: {collections_to_query}")
                    print(f"    Quality score: {state['retrieval_quality']:.2f}")
                    print(f"    Root access: {is_root_user}")
                    print(f"    RBAC Filter Applied: {rbac_filter}")
                    if response_mode == "verbose":
                        for i, ctx in enumerate(ctx_items[:3], 1):
                            doc_id = ctx.get("metadata", {}).get("doc_id", "unknown")
                            company = ctx.get("metadata", {}).get("company_id", "N/A")
                            dept = ctx.get("metadata", {}).get("dept_id", "N/A")
                            text_preview = ctx.get("text", "")[:60]
                            print(f"    [{i}] Doc: {doc_id} | Company: {company}, Dept: {dept} | {text_preview}...")
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Context retrieval failed: {e}"]
                state["retrieval_quality"] = 0.0
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("retrieve_context", {"error": str(e)}, "error")
                
                if show_debug:
                    print(f"  ✗ Error: {e}")
            return state

        def rerank_context_node(state):
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("rerank_context", "Rerank Context", {
                        "context_count": len(state.get("context", {}).get("context", []))
                    })
                
                response_mode = state.get("response_mode", "concise")
                show_debug = response_mode in ["verbose", "internal"]
                
                if show_debug:
                    print(f"\n[📊 RERANK CONTEXT NODE - {response_mode.upper()} MODE]")
                    initial_count = len(state.get("context", {}).get("context", []))
                    print(f"  Reranking {initial_count} documents for relevance...")
                
                # TOOL: rerank_context_tool
                # WHEN: After initial retrieval - need to improve result ordering
                # INPUT: context (raw retrieved chunks from retrieve_context_tool), llm_service
                # PROCESS:
                #   1. Uses LLM to re-evaluate relevance of each chunk
                #   2. Reorders results by relevance score (highest first)
                #   3. Filters out low-confidence matches
                reranked_response = rerank_context_tool.invoke({
                    "context": json.dumps(state.get("context", {})),
                    "llm_service": self.llm_service
                })
                state["reranked_context"] = json.loads(reranked_response) if isinstance(reranked_response, str) else reranked_response
                state["status"] = "context_reranked"
                
                if tracker:
                    reranked_count = len(state.get("reranked_context", {}).get("reranked_context", []))
                    tracker.node_end("rerank_context", {
                        "reranked_count": reranked_count
                    }, "success")
                    tracker.edge_traversal("rerank_context", "check_optimization", ["reranked_context"])
                
                if show_debug:
                    reranked_items = state["reranked_context"].get("reranked_context", [])
                    print(f"  ✓ Reranked to {len(reranked_items)} documents (sorted by relevance)")
                    if response_mode == "verbose":
                        for i, item in enumerate(reranked_items[:3], 1):
                            relevance_score = item.get("metadata", {}).get("relevance_score", "N/A")
                            doc_id = item.get("metadata", {}).get("doc_id", "unknown")
                            print(f"    [{i}] Score: {relevance_score} | Doc: {doc_id}")
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Context reranking failed: {e}"]
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("rerank_context", {"error": str(e)}, "error")
                
                if show_debug:
                    print(f"  ✗ Error: {e}")
            return state

        def check_optimization_needed(state):
            """Intelligent decision node using RL agent to check if healing/optimization is needed."""
            reranked = state.get("reranked_context", {}).get("reranked_context", [])
            num_results = len(reranked)
            
            # Calculate retrieval quality
            quality = min(1.0, num_results / 5)  # Normalized quality (5 is optimal)
            state["retrieval_quality"] = quality
            
            # Use RL agent for intelligent decision
            if self.rl_healing_agent and state.get("doc_id"):
                try:
                    recommendation = self.rl_healing_agent.recommend_healing(
                        doc_id=state.get("doc_id", "unknown"),
                        current_quality=quality
                    )
                    state["rl_recommendation"] = recommendation
                    
                    # Extract decision
                    should_optimize = recommendation['recommended_action'] != 'SKIP'
                    state["should_optimize"] = should_optimize
                    state["optimization_reason"] = recommendation['reasoning']
                    state["rl_action"] = recommendation['recommended_action']
                    
                except Exception as e:
                    # Fallback to simple heuristic if RL fails
                    should_optimize = quality < 0.6 or num_results < 3
                    state["should_optimize"] = should_optimize
                    state["optimization_reason"] = f"Quality={quality:.2f}, Results={num_results}"
                    print(f"Warning: RL agent failed: {e}")
            else:
                # Simple heuristic when RL agent not available
                should_optimize = quality < 0.6 or num_results < 3
                state["should_optimize"] = should_optimize
                state["optimization_reason"] = f"Quality={quality:.2f}, Results={num_results}"
            
            return state

        def optimize_context_node(state):
            """Apply healing/optimization to improve context quality and reduce tokens."""
            try:
                # Get cost estimate
                reranked = state.get("reranked_context", {}).get("reranked_context", [])
                context_list = [{"text": c.get("text", ""), "source": f"Doc {c.get('metadata', {}).get('doc_id', 'N/A')}"} 
                               for c in reranked]
                
                cost_response = get_context_cost_tool.invoke({
                    "context": context_list,
                    "llm_service": self.llm_service,
                    "model_name": "ollama"
                })
                cost_data = json.loads(cost_response) if isinstance(cost_response, str) else cost_response
                
                # Get optimization suggestions
                perf_history = state.get("performance_history", [])
                if not perf_history:
                    perf_history = [{"params": {"k": 5, "chunk_size": 512}, "metrics": {"cost": float(cost_data.get("estimated_cost_usd", 0))}}]
                
                optimize_response = optimize_chunk_size_tool.invoke({
                    "performance_history": perf_history,
                    "llm_service": self.llm_service
                })
                optimize_data = json.loads(optimize_response) if isinstance(optimize_response, str) else optimize_response
                
                state["optimization_result"] = {
                    "cost_analysis": cost_data,
                    "suggested_params": optimize_data.get("suggested_params", {}),
                    "tokens_before": cost_data.get("total_tokens", 0)
                }
                state["status"] = "optimized"
                
                # Log healing action if RL agent made a recommendation
                try:
                    if state.get("should_optimize") and state.get("rl_action") and state.get("rl_action") != "SKIP":
                        from src.database.models.rag_history_model import RAGHistoryModel
                        
                        print(f"[DEBUG] Logging healing action: should_optimize={state.get('should_optimize')}, rl_action={state.get('rl_action')}")
                        
                        action_taken = state.get("rl_action", "OPTIMIZE")
                        metrics = {
                            "strategy": action_taken,
                            "before_metrics": {"avg_quality": state.get("retrieval_quality", 0.0), "total_chunks": len(reranked)},
                            "after_metrics": {"avg_quality": min(1.0, state.get("retrieval_quality", 0.0) + 0.15)},
                            "improvement_delta": 0.15,  # Estimated
                            "cost_tokens": cost_data.get("total_tokens", 0),
                            "duration_ms": 0
                        }
                        
                        print(f"[DEBUG] Instantiating RAGHistoryModel for healing...")
                        rag_history = RAGHistoryModel()
                        print(f"[DEBUG] RAGHistoryModel connected to: {rag_history.db_path}")
                        
                        # Get doc_id from state or context
                        doc_id_to_log = state.get("doc_id")
                        print(f"[DEBUG] doc_id from state: {doc_id_to_log}")
                        
                        if not doc_id_to_log:
                            reranked = state.get("reranked_context", {}).get("reranked_context", [])
                            if reranked and len(reranked) > 0:
                                doc_id_to_log = reranked[0].get("metadata", {}).get("doc_id") or reranked[0].get("source", "unknown")
                                print(f"[DEBUG] doc_id extracted from context: {doc_id_to_log}")
                        
                        doc_id_to_log = doc_id_to_log or "unknown"
                        print(f"[DEBUG] Final doc_id_to_log: {doc_id_to_log}")
                        
                        healing_id = rag_history.log_healing(
                            target_doc_id=doc_id_to_log,
                            target_chunk_id=f"{state.get('doc_id', 'unknown')}_chunk_0",
                            metrics_json=json.dumps(metrics),
                            context_json=json.dumps({
                                "reason": state.get("optimization_reason", "quality_improvement"),
                                "alternatives_considered": ["SKIP", "REINDEX", "RE_EMBED"],
                                "expected_reward": state.get("rl_recommendation", {}).get("estimated_improvement", 0)
                            }),
                            action_taken=action_taken,
                            reward_signal=0.12,  # Estimated reward
                            agent_id="langgraph_agent",
                            session_id=state.get("session_id", "session_default")
                        )
                        
                        print(f"[DEBUG] Healing action logged successfully: healing_id={healing_id}, action={action_taken}")
                        state["healing_logged_id"] = healing_id
                        
                        # Verify it was written
                        rag_history.cursor.execute("SELECT COUNT(*) FROM rag_history_and_optimization WHERE event_type = 'HEAL'")
                        count = rag_history.cursor.fetchone()[0]
                        print(f"[DEBUG] Total HEAL events in database: {count}")
                        rag_history.close()
                        
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Failed to log healing action: {e}")
                    print(f"[TRACEBACK] {traceback.format_exc()}")
                    # Don't fail optimization if logging fails
                    
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Optimization analysis failed: {e}"]
                state["optimization_result"] = {"error": str(e)}
            return state

        def answer_question_node(state):
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("answer_question", "Generate Answer", {
                        "question": state.get("question", "")[:100],
                        "context_count": len(state.get("reranked_context", {}).get("reranked_context", []))
                    })
                
                response_mode = state.get("response_mode", "concise")
                show_debug = response_mode in ["verbose", "internal"]  # Show debug for verbose and internal modes
                
                if show_debug:
                    print(f"\n[📋 ANSWER GENERATION NODE - {response_mode.upper()} MODE]")
                    print(f"  Question: {state['question'][:80]}...")
                    print(f"  Response Mode: {response_mode}")
                    reranked_items = state.get('reranked_context', {}).get('reranked_context', [])
                    print(f"  Reranked Context Items: {len(reranked_items)}")
                    print(f"  State reranked_context keys: {list(state.get('reranked_context', {}).keys())}")
                    if reranked_items:
                        print(f"  First item keys: {list(reranked_items[0].keys())}")
                
                # TOOL: answer_question_tool
                # WHEN: After reranking - ready to synthesize final answer
                # INPUT: question (original user query), context (reranked chunks), llm_service, rbac_context
                # PROCESS:
                #   1. Uses LLM to synthesize coherent answer from context
                #   2. Formats answer based on response_mode
                #   3. Applies guardrails validation (concise/internal modes)
                #   4. Detects RBAC access denial and generates intelligent error messages
                # Generate answer using tool
                rbac_context = {
                    "is_root_user": state.get("is_root_user", False),
                    "user_id": state.get("user_id"),
                    "user_company_id": state.get("user_company_id"),
                    "user_dept_id": state.get("user_dept_id")
                }
                answer_response = answer_question_tool.invoke({
                    "question": state["question"],
                    "context": json.dumps(state.get("reranked_context", {})),
                    "llm_service": self.llm_service,
                    "rbac_context": rbac_context
                })
                
                # Parse the JSON response from the tool
                try:
                    if isinstance(answer_response, str):
                        answer_parsed = json.loads(answer_response)
                        state["answer"] = answer_parsed.get("answer", answer_response)
                        if show_debug:
                            print(f"  [DEBUG] Parsed answer from JSON: {str(state['answer'])[:100]}")
                    else:
                        state["answer"] = answer_response
                        if show_debug:
                            print(f"  [DEBUG] Answer already parsed (not string): {str(answer_response)[:100]}")
                except Exception as e:
                    state["answer"] = answer_response
                    if show_debug:
                        print(f"  [DEBUG] Failed to parse answer JSON: {e}, using raw response: {str(answer_response)[:100]}")
                
                state["status"] = "answer_generated"
                
                if show_debug:
                    print(f"  ✓ Answer Generated ({len(str(state['answer']).split())} words)")
                    print(f"  Answer Preview: {str(state['answer'])[:100]}...")
                
                if tracker:
                    tracker.node_end("answer_question", {
                        "answer_length": len(str(answer_response).split()),
                        "status": state.get("status")
                    }, "success")
                    tracker.edge_traversal("answer_question", "validate_guardrails", ["answer", "question"])
                
                if show_debug:
                    print(f"  ✓ Answer Generated ({len(str(answer_response).split())} words)")
                    if response_mode == "verbose":
                        print(f"  Answer Preview: {str(answer_response)[:150]}...")
                
                # Log query to database with metrics
                try:
                    from src.database.models.rag_history_model import RAGHistoryModel
                    
                    reranked = state.get("reranked_context", {}).get("reranked_context", [])
                    
                    if show_debug:
                        print(f"\n[📊 LOGGING QUERY TO DATABASE]")
                        print(f"  Reranked Sources: {len(reranked)}")
                    
                    metrics = {
                        "frequency": 1,
                        "avg_accuracy": state.get("retrieval_quality", 0.7),
                        "cost_tokens": len(state["question"].split()) * 10,  # Rough estimate
                        "latency_ms": 0,  # Would need timing
                        "user_feedback": 0.7,  # Default, will be updated by user
                        "quality_category": "warm" if state.get("retrieval_quality", 0) > 0.6 else "cold",
                        "sources_count": len(reranked),
                        "response_mode": response_mode
                    }
                    
                    rag_history = RAGHistoryModel()
                    if show_debug:
                        print(f"  Database Path: {rag_history.db_path}")
                    
                    # Get doc_id from context if not in state
                    doc_id_to_log = state.get("doc_id")
                    
                    if not doc_id_to_log and reranked and len(reranked) > 0:
                        # Extract from first retrieved document
                        doc_id_to_log = reranked[0].get("metadata", {}).get("doc_id") or reranked[0].get("source", "unknown")
                    
                    doc_id_to_log = doc_id_to_log or "unknown"
                    if show_debug:
                        print(f"  Target Doc ID: {doc_id_to_log}")
                    
                    query_id = rag_history.log_query(
                        query_text=state["question"],
                        target_doc_id=doc_id_to_log,
                        metrics_json=json.dumps(metrics),
                        context_json=json.dumps({
                            "retrieval_quality": state.get("retrieval_quality", 0.7),
                            "sources": len(reranked),
                            "answer_length": len(state["answer"].split()) if state["answer"] else 0,
                            "response_mode": response_mode
                        }),
                        agent_id="langgraph_agent",
                        session_id=state.get("session_id", "session_default")
                    )
                    
                    if show_debug:
                        print(f"  ✓ Query Logged: ID={query_id}")
                    
                    state["query_logged_id"] = query_id
                    
                    # Verify it was written
                    if response_mode == "verbose":
                        cur = rag_history.conn.execute("SELECT COUNT(*) FROM rag_history_and_optimization WHERE event_type = 'QUERY'")
                        count = cur.fetchone()[0]
                        print(f"  Database Total QUERY Events: {count}")
                    
                    # Note: RAGHistoryModel manages its own connection, no need to close
                    
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Failed to log query: {e}")
                    if response_mode == "verbose":
                        print(f"[TRACEBACK] {traceback.format_exc()}")
                    # Don't fail the answer generation if logging fails
                    
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Answer generation failed: {e}"]
                state["answer"] = "Failed to generate answer"
                if show_debug:
                    print(f"  ✗ Error: {e}")
            
            return state

        def traceability_node(state):
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("traceability", "Generate Traceability", {
                        "question": state.get("question", "")[:100]
                    })
                
                trace_response = traceability_tool.invoke({
                    "question": state["question"],
                    "context": json.dumps(state.get("reranked_context", {})),
                    "vectordb_service": self.vectordb_service
                })
                state["traceability"] = json.loads(trace_response) if isinstance(trace_response, str) else trace_response
                state["status"] = "completed"
                
                if tracker:
                    tracker.node_end("traceability", {
                        "traceability_status": state.get("status")
                    }, "success")
                    tracker.edge_traversal("traceability", "END", ["traceability"])
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Traceability generation failed: {e}"]
                
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_end("traceability", {"error": str(e)}, "error")
            return state

        def validate_response_guardrails_node(state):
            """
            NEW NODE: Validate Response with Custom Guardrails
            
            TOOL: CustomGuardrails (simple, effective, no external dependencies)
            WHEN: After answer generation, before returning to user
            INPUT: answer (generated response), question (user query), mode (concise|verbose|internal)
            PROCESS:
              1. Input validation: Check user query is safe
              2. Output safety check: Verify response is complete, no repetition
              3. PII detection: Find and redact sensitive data
              4. Response filtering: Redact credentials, passwords, API keys
            OUTPUT: state["guardrail_checks"], state["is_response_safe"], filtered answer
            PURPOSE: Prevent hallucinations, PII leaks, and inappropriate responses
            BENEFIT: Simple pattern-based validation without external library dependencies
            """
            try:
                tracker = state.get("_tracker")
                if tracker:
                    tracker.node_start("validate_guardrails", "Validate Guardrails", {
                        "answer_length": len(str(state.get("answer", "")).split())
                    })
                
                response_mode = state.get("response_mode", "concise")
                show_debug = response_mode in ["verbose", "internal"]
                
                if show_debug:
                    print(f"\n[🛡️ GUARDRAILS VALIDATION NODE - {response_mode.upper()} MODE]")
                
                # Use custom guardrails instance
                if response_mode == "verbose":
                    # Skip validation for verbose mode
                    state["guardrail_checks"] = {"skipped": True, "reason": "verbose mode"}
                    state["is_response_safe"] = True
                    
                    if tracker:
                        tracker.node_end("validate_guardrails", {
                            "validation_skipped": True
                        }, "success")
                        tracker.edge_traversal("validate_guardrails", "traceability", ["guardrail_checks"])
                    
                    if show_debug:
                        print(f"  ⊘ Validation skipped (verbose mode)")
                    return state
                
                # Run custom guardrails validation
                validation_result = self.custom_guardrails.process_request(
                    user_input=state.get("question", ""),
                    llm_output=state.get("answer", "")
                )
                
                # Store validation results
                state["guardrail_checks"] = {
                    "is_safe": validation_result.get("is_safe", False),
                    "safety_level": validation_result.get("safety_level", "unknown"),
                    "pii_detected": validation_result.get("pii_detected", {}),
                    "input_errors": validation_result.get("input_errors", []),
                    "output_errors": validation_result.get("output_errors", []),
                    "message": validation_result.get("message", "")
                }
                
                if tracker:
                    tracker.node_end("validate_guardrails", {
                        "is_safe": validation_result.get("is_safe", False),
                        "safety_level": validation_result.get("safety_level", "unknown")
                    }, "success")
                    tracker.edge_traversal("validate_guardrails", "traceability", ["guardrail_checks"])
                # Update response with filtered version if needed
                if validation_result.get("filtered_output"):
                    state["answer"] = validation_result["filtered_output"]
                
                # Set safety flag
                state["is_response_safe"] = validation_result.get("success", False)
                
                if show_debug:
                    print(f"  Safety Level: {validation_result['safety_level']}")
                    print(f"  Is Safe: {validation_result['is_safe']}")
                    if validation_result['pii_detected']:
                        print(f"  PII Found: {list(validation_result['pii_detected'].keys())}")
                    print(f"  Message: {validation_result['message']}")
                
                # Log guardrail check to database
                try:
                    from src.database.models.rag_history_model import RAGHistoryModel
                    
                    rag_history = RAGHistoryModel()
                    reranked_context = state.get("reranked_context", {}).get("reranked_context", [])
                    doc_id_to_log = state.get("doc_id")
                    
                    if not doc_id_to_log and reranked_context and len(reranked_context) > 0:
                        doc_id_to_log = reranked_context[0].get("metadata", {}).get("doc_id") or reranked_context[0].get("source", "unknown")
                    
                    doc_id_to_log = doc_id_to_log or "unknown"
                    
                    try:
                        guardrail_id = rag_history.log_guardrail_check(
                            target_doc_id=doc_id_to_log,
                            checks_json=json.dumps(state["guardrail_checks"]),
                            is_safe=state["is_response_safe"],
                            agent_id="langgraph_agent",
                            session_id=state.get("session_id", "session_default")
                        )
                        
                        if show_debug:
                            print(f"  ✓ Guardrail check logged: ID={guardrail_id}")
                    except Exception as log_err:
                        # If logging fails (e.g., old database schema), just continue
                        if show_debug:
                            print(f"  [INFO] Guardrail logging skipped (database schema may be outdated): {log_err}")
                    
                except Exception as e:
                    if show_debug:
                        print(f"  [WARNING] Failed to initialize guardrail logging: {e}")
                
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Guardrails validation failed: {e}"]
                state["is_response_safe"] = False
                if show_debug:
                    print(f"  ✗ Error: {e}")
            
            return state

        # Add nodes
        graph.add_node("retrieve_context", retrieve_context_node)
        graph.add_node("rerank_context", rerank_context_node)
        graph.add_node("check_optimization", check_optimization_needed)
        graph.add_node("optimize_context", optimize_context_node)
        graph.add_node("answer_question", answer_question_node)
        graph.add_node("validate_guardrails", validate_response_guardrails_node)  # NEW NODE
        graph.add_node("traceability", traceability_node)

        # Add edges with conditional routing
        graph.add_edge(START, "retrieve_context")
        graph.add_edge("retrieve_context", "rerank_context")
        graph.add_edge("rerank_context", "check_optimization")
        
        # Conditional edge: if optimization needed, optimize; otherwise skip to answer
        def route_to_optimization(state):
            return "optimize_context" if state.get("should_optimize", False) else "answer_question"
        
        graph.add_conditional_edges("check_optimization", route_to_optimization, {
            "optimize_context": "optimize_context",
            "answer_question": "answer_question"
        })
        
        graph.add_edge("optimize_context", "answer_question")
        graph.add_edge("answer_question", "validate_guardrails")  # NEW EDGE: Answer -> Guardrails
        graph.add_edge("validate_guardrails", "traceability")     # NEW EDGE: Guardrails -> Traceability
        graph.add_edge("traceability", END)

        return graph.compile()

    def _build_optimization_graph(self):
        """Build optimization workflow graph."""
        graph = StateGraph(dict)

        def optimize_node(state):
            try:
                result = optimize_chunk_size_tool.invoke({
                    "performance_history": state["performance_history"],
                    "llm_service": self.llm_service
                })
                state["optimization_result"] = json.loads(result) if isinstance(result, str) else result
                state["status"] = "optimization_complete"
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Optimization failed: {e}"]
            return state

        def apply_config_node(state):
            try:
                # Pass config_service as None if not available
                result = adjust_config_tool.invoke({
                    "config_service": None,
                    "updates": state.get("config_updates", {})
                })
                state["config_result"] = json.loads(result) if isinstance(result, str) else result
                state["status"] = "completed"
            except Exception as e:
                state["errors"] = state.get("errors", []) + [f"Config update failed: {e}"]
            return state

        # Add nodes
        graph.add_node("optimize", optimize_node)
        graph.add_node("apply_config", apply_config_node)

        # Add edges
        graph.add_edge(START, "optimize")
        graph.add_edge("optimize", "apply_config")
        graph.add_edge("apply_config", END)

        return graph.compile()

    def ingest_document(self, text: str, doc_id: str, company_id: int = None, dept_id: int = None) -> Dict[str, Any]:
        """
        Ingest document using ingestion workflow.
        
        If doc_id already exists in the company collection, the old embeddings will be deleted first.
        This enables re-ingestion with updated content and embeddings.
        """
        collection_name = self._get_collection_name(company_id)
        
        # Create animated tracker for real-time workflow visualization
        ingestion_tracker = AnimatedGraphTracker(workflow_type="ingestion", workflow_id=f"ingest_{doc_id}")
        
        # Check if document already exists and delete old embeddings
        try:
            vectordb_service = VectorDBService(
                persist_directory=self.vectordb_service.persist_directory,
                collection_name=collection_name
            )
            
            # Try to delete existing doc_id embeddings
            existing_count_before = vectordb_service.collection.count()
            vectordb_service.delete_by_document(doc_id)
            existing_count_after = vectordb_service.collection.count()
            
            if existing_count_before > existing_count_after:
                print(f"[DEBUG REINGEST] Deleted {existing_count_before - existing_count_after} old embeddings for doc_id={doc_id}")
                print(f"[DEBUG REINGEST] Collection {collection_name} now has {existing_count_after} vectors")
        except Exception as e:
            print(f"[DEBUG REINGEST] Could not delete old embeddings: {e}")
        
        # Now ingest the new document
        initial_state = {
            "text": text,
            "doc_id": doc_id,
            "company_id": company_id,
            "dept_id": dept_id,
            "errors": [],
            "status": "started",
            "_tracker": ingestion_tracker  # Pass tracker through workflow state
        }
        result = self.ingestion_graph.invoke(initial_state)
        collection_name = result.get("collection_name", self._get_collection_name(company_id))
        
        # Extract tracker data for visualization
        tracker_data = None
        if ingestion_tracker:
            tracker_data = ingestion_tracker.get_graph_data()
        
        return {
            "success": len(result.get("errors", [])) == 0,
            "doc_id": doc_id,
            "company_id": company_id,
            "collection_name": collection_name,
            "chunks_count": result.get("chunks", {}).get("num_chunks", 0),
            "chunks_saved": result.get("save_result", {}).get("chunks_saved", 0),
            "metadata": result.get("metadata", {}),
            "rbac_tags": result.get("rbac_tags", ""),
            "meta_tags": result.get("meta_tags", ""),
            "errors": result.get("errors", []),
            "workflow_graph": tracker_data  # Include animation data
        }

    def _apply_guardrails_validation(self, answer: str, response_mode: str) -> Dict[str, Any]:
        """Apply custom guardrails validation based on response mode.
        
        Uses CustomGuardrails for all response modes:
        - concise: Full validation (user-friendly)
        - internal: Hallucination & safety checks (system integration)
        - verbose: Basic safety only (engineers need raw data)
        """
        if response_mode == "verbose":
            # No validation for verbose mode - engineers need full debug info
            return {
                "validated": True,
                "guardrails_applied": False,
                "answer": answer
            }
        
        try:
            # Use CustomGuardrails for all modes
            validation_result = self.custom_guardrails.process_request(
                user_input="",  # No user input validation needed here
                llm_output=answer
            )
            
            return {
                "validated": validation_result.get("success", False),
                "guardrails_applied": True,
                "answer": validation_result.get("filtered_output", answer),
                "validation_mode": response_mode,
                "safety_level": validation_result.get("safety_level", "unknown"),
                "issues": validation_result.get("output_errors", [])
            }
        except Exception as e:
            print(f"[WARNING] Guardrails validation failed: {e}")
            # Return original answer if validation fails
            return {
                "validated": False,
                "guardrails_applied": False,
                "answer": answer,
                "validation_error": str(e)
            }

    def ask_question(self, question: str, performance_history: List[Dict[str, Any]] = None, doc_id: str = None, response_mode: str = "concise", company_id: int = None, user_id: int = None, dept_id: int = None, top_k: int = 5) -> Dict[str, Any]:
        """Answer question using intelligent retrieval workflow with RL healing agent.
        
        Args:
            question: User question
            performance_history: Historical performance data
            doc_id: Optional document ID to query
            response_mode: Response format:
                - "concise": End-user friendly (answer only)
                - "verbose": Engineer/RAG Admin (all metadata, traceability, RL info)
                - "internal": System/Integration (answer + structured data for updating tables)
            company_id: Company context for RBAC filtering
            user_id: User making the query
            dept_id: Department context for RBAC filtering
            top_k: Number of top results to retrieve
            
        Returns:
            Response dict with answer and metadata based on response_mode
        """
        import uuid
        session_id = str(uuid.uuid4())  # Generate session ID for tracking
        
        # Create visualization tracker
        viz = create_visualization(session_id)
        start_time = time.time()
        
        # Create animated tracker for retrieval workflow
        retrieval_tracker = AnimatedGraphTracker(workflow_type="retrieval", workflow_id=f"retrieve_{session_id}")
        
        # Determine if user is root (root user has user_id=99)
        is_root_user = user_id == 99
        
        # Tenant assignment: Use passed parameters directly
        # Root users (user_id=99): Can access all documents if company_id provided, else no restriction
        # Regular users: Must have company_id and dept_id for tenant isolation
        user_company_id = company_id if company_id is not None else (None if is_root_user else 1)
        user_dept_id = dept_id if dept_id is not None else (None if is_root_user else 1)
        
        initial_state = {
            "question": question,
            "doc_id": doc_id,
            "session_id": session_id,
            "response_mode": response_mode,
            "performance_history": performance_history or [],
            "errors": [],
            "status": "started",
            "company_id": company_id,
            "user_id": user_id,
            "dept_id": dept_id,
            "top_k": top_k,
            "is_root_user": is_root_user,  # Flag for root user access
            "user_company_id": user_company_id,  # User's company context
            "user_dept_id": user_dept_id,  # User's department context
            "_tracker": retrieval_tracker  # Pass tracker through workflow state
        }
        
        # Track workflow execution
        viz.record_node_start("retrieve_and_answer_workflow", initial_state)
        
        try:
            result = self.retrieval_graph.invoke(initial_state)
            
            # Track successful completion
            result["execution_time_ms"] = (time.time() - start_time) * 1000







            viz.record_node_end("retrieve_and_answer_workflow", result)
            result["visualization"] = viz.get_trace_data()
            
            # Extract tracker data for animated visualization
            if retrieval_tracker:
                result["workflow_graph"] = retrieval_tracker.get_graph_data()
            
        except Exception as e:
            viz.record_error("retrieve_and_answer_workflow", str(e))
            result = {"errors": [str(e)]}
        
        # Save visualization to logs and session_graph (suppress output in concise mode)
        try:
            import sys
            from io import StringIO
            
            if response_mode == "concise":
                # Suppress stdout during visualization save for concise mode
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    viz_files = save_visualization(viz, output_dir="logs", graph=self.retrieval_graph)
                finally:
                    sys.stdout = old_stdout
            else:
                viz_files = save_visualization(viz, output_dir="logs", graph=self.retrieval_graph)
                print(f"[✓] Session visualization saved: {viz_files}")
        except Exception as e:
            print(f"[!] Failed to save visualization: {e}")
            import traceback
            traceback.print_exc()
        
        # Build response based on mode
        if response_mode == "concise":
            # End-user friendly: just answer, no metadata
            answer_text = result.get("answer", "")
            
            # Extract plain text answer (answer is already extracted from JSON by answer_question_node)
            # No need to parse JSON again, it's already been done
            
            # Apply Guardrails validation for concise mode (hallucination_check + security_incident_policy)
            validation_result = self._apply_guardrails_validation(answer_text, response_mode)
            if not validation_result.get("validated"):
                print(f"[⚠] Guardrails validation warning: {validation_result.get('validation_error')}")
            validated_answer = validation_result.get("answer", answer_text)
            
            # Get chunk count for reporting
            reranked = result.get("reranked_context", {}).get("reranked_context", [])
            context_chunks = len(reranked)
            
            return {
                "success": len(result.get("errors", [])) == 0,
                "question": question,
                "answer": validated_answer,
                "session_id": session_id,
                "context_chunks": context_chunks,
                "guardrails_applied": validation_result.get("guardrails_applied", False),
                "errors": result.get("errors", [])
            }
        elif response_mode == "internal":
            # System/Integration: clean answer text + structured metadata for database updates (no approval needed)
            reranked = result.get("reranked_context", {}).get("reranked_context", [])
            
            # Extract plain text answer (answer is already extracted from JSON by answer_question_node)
            answer_text = result.get("answer", "")
            
            # Apply Guardrails validation for internal mode (hallucination_check only)
            validation_result = self._apply_guardrails_validation(answer_text, response_mode)
            if not validation_result.get("validated"):
                print(f"[⚠] Guardrails validation warning: {validation_result.get('validation_error')}")
            validated_answer = validation_result.get("answer", answer_text)
            
            return {
                "success": len(result.get("errors", [])) == 0,
                "answer": validated_answer,  # Plain text only
                "quality_score": result.get("retrieval_quality", 0.0),
                "sources_count": len(reranked),
                "source_docs": [{"doc_id": s.get("metadata", {}).get("doc_id") or s.get("source", "unknown"), 
                                "chunk_id": s.get("metadata", {}).get("chunk_id")} for s in reranked],
                "metadata": {
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "model": "langgraph_rag_agent",
                    "execution_time_ms": result.get("execution_time_ms", 0)
                },
                "guardrails_applied": validation_result.get("guardrails_applied", False),
                "errors": result.get("errors", [])
            }
        else:  # verbose mode (default for engineers/admins)
            # Full business intelligence: all metadata, traceability, RL info
            # NOTE: No guardrails for verbose mode - engineers need raw data for debugging
            reranked = result.get("reranked_context", {}).get("reranked_context", [])
            return {
                "success": len(result.get("errors", [])) == 0,
                "question": question,
                "answer": result.get("answer", ""),
                "sources": reranked,
                "sources_count": len(reranked),
                "traceability": result.get("traceability", {}),
                "retrieval_quality": result.get("retrieval_quality", 0.0),
                "optimization_applied": result.get("should_optimize", False),
                "optimization_reason": result.get("optimization_reason", ""),
                "rl_action": result.get("rl_action", "SKIP"),
                "rl_recommendation": {
                    "action": result.get("rl_info", {}).get("recommended_action", "N/A"),
                    "confidence": result.get("rl_info", {}).get("confidence", 0),
                    "expected_improvement": result.get("rl_info", {}).get("expected_improvement", 0),
                    "learning_stats": result.get("rl_info", {}).get("learning_stats", {})
                },
                "optimization_result": result.get("optimization_result", {}),
                "execution_time_ms": result.get("execution_time_ms", 0),
                "session_id": session_id,
                "visualization_data": result.get("visualization", {}),
                "guardrails_applied": False,
                "errors": result.get("errors", [])
            }

    def optimize_system(self, performance_history: List[Dict[str, Any]], config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize system using optimization workflow."""
        initial_state = {
            "performance_history": performance_history,
            "config_updates": config_updates,
            "errors": [],
            "status": "started"
        }
        result = self.optimization_graph.invoke(initial_state)
        return {
            "success": len(result.get("errors", [])) == 0,
            "optimization": result.get("optimization_result", {}),
            "config_applied": result.get("config_result", {}),
            "errors": result.get("errors", [])
        }

    def invoke_chat(self, response_mode: str = "concise", show_history: bool = True) -> Dict[str, Any]:
        """
        Interactive chat mode with human feedback loop.
        
        CHAT WORKFLOW:
        1. Get user question
        2. Process through RAG agent with specified response_mode
        3. Display answer
        4. Ask for satisfaction feedback
        5. If satisfied -> ask new question, else -> try different approach
        6. Continue until user quits or satisfied
        
        Args:
            response_mode (str): "concise" (user), "internal" (system), "verbose" (engineer)
            show_history (bool): Show conversation history at end
        
        Returns:
            Dict with conversation_history, message_count, session_stats
        """
        from datetime import datetime
        import sys
        from io import StringIO
        
        print("\n" + "="*70)
        print("LANGGRAPH AGENT - INTERACTIVE CHAT MODE")
        print(f"Response Mode: {response_mode.upper()}")
        print("="*70 + "\n")
        
        conversation_history = []
        message_count = 0
        
        while True:
            # Get user question
            try:
                question = input("Ask a question (or 'quit' to exit): ").strip()
            except EOFError:
                question = "quit"
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                print("[ERROR] Please enter a valid question\n")
                continue
            
            # Process question
            print("\n[PROCESSING] Searching knowledge base...")
            try:
                # Suppress debug output
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                response = self.ask_question(
                    question=question,
                    response_mode=response_mode
                )
                
                sys.stdout = old_stdout
            except Exception as e:
                sys.stdout = old_stdout
                print(f"[ERROR] Query failed: {e}\n")
                continue
            
            if not response.get("success"):
                print(f"[ERROR] {response.get('errors', 'Unknown error')}\n")
                continue
            
            # Display answer
            answer = response.get("answer", "No answer available")
            print(f"\nQ: {question}")
            print("-" * 70)
            print(f"A: {answer}\n")
            print("-" * 70)
            
            # Add to history
            conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "response_mode": response_mode
            })
            message_count += 1
            
            # Ask for satisfaction
            while True:
                try:
                    feedback = input("\nSatisfied? (yes/no/followup): ").strip().lower()
                except EOFError:
                    feedback = "yes"
                
                if feedback in ['yes', 'y']:
                    print("[OK] Great! Ask another question or type 'quit' to exit.\n")
                    break
                elif feedback in ['no', 'n']:
                    print("[NOTE] Searching for alternative answers...\n")
                    # Retry with verbose mode for more details
                    try:
                        old_stdout = sys.stdout
                        sys.stdout = StringIO()
                        
                        response = self.ask_question(
                            question=question,
                            response_mode="verbose"
                        )
                        
                        sys.stdout = old_stdout
                    except Exception as e:
                        sys.stdout = old_stdout
                        print(f"[ERROR] Retry failed: {e}\n")
                        break
                    
                    answer = response.get("answer", "No answer available")
                    print(f"A (with more details): {answer}\n")
                    print("-" * 70)
                    break
                elif feedback in ['followup', 'f']:
                    print("[OK] Ask your follow-up question:\n")
                    break
                else:
                    print("[ERROR] Please enter: yes, no, or followup\n")
        
        # Session summary
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Total Questions: {message_count}")
        print(f"Response Mode: {response_mode}")
        
        if show_history and conversation_history:
            print("\nConversation History:")
            for i, entry in enumerate(conversation_history, 1):
                print(f"  {i}. Q: {entry['question'][:60]}...")
                print(f"     A: {entry['answer'][:80]}...")
        
        print("="*70 + "\n")
        
        # Save session
        import json
        import os
        history_dir = "data/chat_history"
        os.makedirs(history_dir, exist_ok=True)
        session_file = f"{history_dir}/chat_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "response_mode": response_mode,
            "message_count": message_count,
            "conversation_history": conversation_history
        }
        
        with open(session_file, "w") as f:
            json.dump(session_data, f, indent=2, default=str)
        
        print(f"[OK] Session saved to: {session_file}\n")
        
        return {
            "success": True,
            "message_count": message_count,
            "conversation_history": conversation_history,
            "session_file": session_file
        }

    def invoke(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Unified interface for all agent operations.
        
        Operations:
        - ingest_document: Ingest single document with company/dept ownership
        - ingest_sqlite_table: Ingest data from SQLite table
        - ingest_from_path: Batch ingest documents from file path
        - ask_question: Query knowledge base with multi-tenant filtering
        - optimize: Optimize system configuration
        - chat: Interactive chat with user feedback
        """
        if operation == "ingest_document":
            return self.ingest_document(
                text=kwargs.get("text", ""),
                doc_id=kwargs.get("doc_id", "doc_default"),
                company_id=kwargs.get("company_id"),
                dept_id=kwargs.get("dept_id")
            )
        
        elif operation == "ingest_sqlite_table":
            # Load data_sources.json from RAG config path (plug-and-play via EnvConfig)
            rag_config_path = EnvConfig.get_rag_config_path()
            config_path = os.path.join(rag_config_path, "data_sources.json")
            
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except:
                config = {}
            
            table_name = kwargs.get("table_name", "knowledge_base")
            sqlite_config = config.get("data_sources", {}).get("sqlite", {})
            tables_config = sqlite_config.get("ingestion_modes", {}).get("table_based", {}).get("tables_to_ingest", [])
            
            table_config = next((t for t in tables_config if t.get("name") == table_name), {})
            
            text_columns = kwargs.get("text_columns", table_config.get("text_columns", []))
            metadata_columns = kwargs.get("metadata_columns", table_config.get("metadata_columns", []))
            chunking_config = sqlite_config.get("chunking", {})
            
            result_json = ingest_sqlite_table_tool.invoke({
                "table_name": table_name,
                "doc_id": kwargs.get("doc_id", f"sqlite_{table_name}"),
                "rbac_namespace": kwargs.get("rbac_namespace", "general"),
                "text_columns": text_columns,
                "metadata_columns": metadata_columns,
                "db_path": EnvConfig.get_db_path(),
                "llm_service": self.llm_service,
                "vectordb_service": self.vectordb_service,
                "chunk_size": kwargs.get("chunk_size", chunking_config.get("chunk_size", 512)),
                "chunk_overlap": kwargs.get("chunk_overlap", chunking_config.get("overlap", 50))
            })
            
            return json.loads(result_json) if isinstance(result_json, str) else result_json
        
        elif operation == "ingest_from_path":
            # Ingest documents from file path or folder recursively
            # Calls ingest_documents_from_path_tool which returns discovered docs with generated doc_ids
            path = kwargs.get("path", "")
            doc_id_prefix = kwargs.get("doc_id_prefix", "doc")
            file_type = kwargs.get("file_type", "auto")  # auto, pdf, text, word
            recursive = kwargs.get("recursive", True)
            
            if not path:
                return {
                    "success": False,
                    "error": "path parameter required for ingest_from_path operation",
                    "documents_ingested": 0,
                    "errors": []
                }
            
            try:
                # Discover documents from path using the ingest_documents_from_path_tool
                discovery_result = ingest_documents_from_path_tool.invoke({
                    "path": path,
                    "doc_id_prefix": doc_id_prefix,
                    "file_type": file_type,
                    "recursive": recursive
                })
                
                discovery_data = json.loads(discovery_result) if isinstance(discovery_result, str) else discovery_result
                
                discovered_docs = discovery_data.get("discovered_documents", [])
                ingestion_errors = []
                ingestion_count = 0
                
                # Ingest each discovered document
                for doc in discovered_docs:
                    try:
                        doc_text = doc.get("text", "")
                        doc_id = doc.get("doc_id", f"{doc_id_prefix}_{ingestion_count}")
                        
                        # Call ingest_document to process each document
                        ingest_result = self.ingest_document(
                            text=doc_text,
                            doc_id=doc_id
                        )
                        
                        if ingest_result.get("success"):
                            ingestion_count += 1
                        else:
                            ingestion_errors.append({
                                "doc_id": doc_id,
                                "error": ingest_result.get("errors", ["Unknown error"])
                            })
                    except Exception as e:
                        ingestion_errors.append({
                            "doc_id": doc.get("doc_id", f"{doc_id_prefix}_{ingestion_count}"),
                            "error": str(e)
                        })
                
                return {
                    "success": len(ingestion_errors) == 0,
                    "documents_discovered": len(discovered_docs),
                    "documents_ingested": ingestion_count,
                    "documents_failed": len(ingestion_errors),
                    "errors": ingestion_errors,
                    "ingestion_details": {
                        "path": path,
                        "recursive": recursive,
                        "file_type": file_type,
                        "doc_id_prefix": doc_id_prefix
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "documents_ingested": 0,
                    "errors": [str(e)]
                }
        
        elif operation == "ask_question":
            return self.ask_question(
                question=kwargs.get("question", ""),
                performance_history=kwargs.get("performance_history"),
                doc_id=kwargs.get("doc_id"),
                response_mode=kwargs.get("response_mode", "concise")
            )
        
        elif operation == "optimize":
            return self.optimize_system(
                performance_history=kwargs.get("performance_history", []),
                config_updates=kwargs.get("config_updates", {})
            )
        
        elif operation == "chat":
            # Interactive chat mode with human feedback loop
            return self.invoke_chat(
                response_mode=kwargs.get("response_mode", "concise"),
                show_history=kwargs.get("show_history", True)
            )
        
        else:
            return {
                "success": False,
                "error": f"Unknown operation: {operation}. Available: ingest_document, ingest_sqlite_table, ingest_from_path, ask_question, optimize, chat"
            }

    def ingest_directory(self, directory_path: str, file_extensions: List[str], company_id: int = None, 
                        dept_id: int = None, recursive: bool = True, batch_size: int = 10) -> Dict[str, Any]:
        """
        Ingest all documents from a directory.
        
        Processes files with specified extensions recursively or within the directory.
        Files are processed in parallel batches for performance.
        """
        import os
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        try:
            directory_path = str(Path(directory_path).resolve())
            
            if not os.path.isdir(directory_path):
                return {
                    "success": False,
                    "files_processed": 0,
                    "chunks_created": 0,
                    "vectors_saved": 0,
                    "failed_files": [directory_path],
                    "errors": [f"Directory not found: {directory_path}"]
                }
            
            # Collect files with specified extensions
            files_to_process = []
            pattern = f"**/*" if recursive else "*"
            
            for ext in file_extensions:
                ext_pattern = f"{pattern}.{ext}"
                files_to_process.extend(Path(directory_path).glob(ext_pattern))
            
            files_to_process = sorted(set([str(f) for f in files_to_process if f.is_file()]))
            
            if not files_to_process:
                return {
                    "success": True,
                    "files_processed": 0,
                    "chunks_created": 0,
                    "vectors_saved": 0,
                    "failed_files": [],
                    "errors": [f"No files found with extensions: {file_extensions}"]
                }
            
            total_chunks = 0
            total_vectors = 0
            failed_files = []
            
            # Process files in batches
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {}
                
                for file_path in files_to_process:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            doc_id = os.path.basename(file_path)
                            future = executor.submit(
                                self.ingest_document,
                                text=content,
                                doc_id=doc_id,
                                company_id=company_id,
                                dept_id=dept_id
                            )
                            futures[future] = file_path
                    except Exception as e:
                        failed_files.append(file_path)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result.get("success"):
                            total_chunks += result.get("chunks_saved", 0)
                            total_vectors += result.get("chunks_saved", 0)
                        else:
                            failed_files.append(futures[future])
                    except Exception as e:
                        failed_files.append(futures[future])
            
            return {
                "success": len(failed_files) == 0,
                "files_processed": len(files_to_process) - len(failed_files),
                "chunks_created": total_chunks,
                "vectors_saved": total_vectors,
                "failed_files": failed_files,
                "errors": [] if not failed_files else [f"Failed to process {len(failed_files)} files"]
            }
        
        except Exception as e:
            return {
                "success": False,
                "files_processed": 0,
                "chunks_created": 0,
                "vectors_saved": 0,
                "failed_files": [],
                "errors": [f"Directory ingestion error: {str(e)}"]
            }

    def ingest_sqlite(self, database_path: str, table_name: str, content_column: str = "content",
                     id_column: str = "id", batch_size: int = 100, company_id: int = None,
                     dept_id: int = None) -> Dict[str, Any]:
        """
        Ingest data from a SQLite database table.
        
        Reads rows from the specified table and ingests content from the content_column.
        Each row is treated as a separate document identified by id_column.
        Rows are processed in batches for performance.
        """
        import sqlite3
        import os
        from pathlib import Path
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        try:
            database_path = str(Path(database_path).resolve())
            
            if not os.path.isfile(database_path):
                return {
                    "success": False,
                    "rows_processed": 0,
                    "chunks_created": 0,
                    "vectors_saved": 0,
                    "errors": [f"Database file not found: {database_path}"]
                }
            
            # Connect and fetch data
            conn = sqlite3.connect(database_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            try:
                # Get total row count
                cursor.execute(f"SELECT COUNT(*) as count FROM {table_name}")
                total_rows = cursor.fetchone()["count"]
                
                if total_rows == 0:
                    return {
                        "success": True,
                        "rows_processed": 0,
                        "chunks_created": 0,
                        "vectors_saved": 0,
                        "errors": [f"Table '{table_name}' is empty"]
                    }
                
                total_chunks = 0
                total_vectors = 0
                processed_rows = 0
                errors = []
                
                # Fetch and process rows in batches
                offset = 0
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {}
                    
                    while offset < total_rows:
                        cursor.execute(
                            f"SELECT {id_column}, {content_column} FROM {table_name} LIMIT ? OFFSET ?",
                            (batch_size, offset)
                        )
                        rows = cursor.fetchall()
                        
                        if not rows:
                            break
                        
                        # Submit batch for processing
                        for row in rows:
                            try:
                                doc_id = str(row[id_column])
                                content = str(row[content_column])
                                
                                future = executor.submit(
                                    self.ingest_document,
                                    text=content,
                                    doc_id=doc_id,
                                    company_id=company_id,
                                    dept_id=dept_id
                                )
                                futures[future] = doc_id
                            except Exception as e:
                                errors.append(f"Error processing row {row.get(id_column)}: {str(e)}")
                        
                        offset += batch_size
                    
                    # Collect results
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result.get("success"):
                                total_chunks += result.get("chunks_saved", 0)
                                total_vectors += result.get("chunks_saved", 0)
                                processed_rows += 1
                            else:
                                errors.append(f"Failed to ingest row {futures[future]}")
                        except Exception as e:
                            errors.append(f"Processing error for row {futures[future]}: {str(e)}")
                
                return {
                    "success": len(errors) == 0,
                    "rows_processed": processed_rows,
                    "chunks_created": total_chunks,
                    "vectors_saved": total_vectors,
                    "errors": errors
                }
            
            finally:
                conn.close()
        
        except Exception as e:
            return {
                "success": False,
                "rows_processed": 0,
                "chunks_created": 0,
                "vectors_saved": 0,
                "errors": [f"SQLite ingestion error: {str(e)}"]
            }


# ============================================================================
# CLI ENTRY POINT - Direct command-line invocation
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LangGraph RAG Agent - Interactive Chat & Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:
  # Interactive chat mode (concise)
  python -m rag.agents.langgraph_agent.langgraph_rag_agent --chat
  
  # Chat with verbose mode (full debug details)
  python -m rag.agents.langgraph_agent.langgraph_rag_agent --chat --verbose
  
  # Ask single question (concise)
  python -m rag.agents.langgraph_agent.langgraph_rag_agent --ask "What are main incident causes?"
  
  # Ask single question (verbose)
  python -m rag.agents.langgraph_agent.langgraph_rag_agent --ask "What are main incident causes?" --verbose
  
  # Ingest knowledge_base table
  python -m rag.agents.langgraph_agent.langgraph_rag_agent --ingest-table knowledge_base
  
RESPONSE MODES:
  --concise    User-friendly, concise mode (default)
  --internal   System integration mode, structured data
  --verbose    Full debug mode for engineers
        """
    )
    
    # Operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat mode with human feedback loop"
    )
    group.add_argument(
        "--ask",
        type=str,
        metavar="QUESTION",
        help="Ask a single question and exit"
    )
    group.add_argument(
        "--ingest-table",
        type=str,
        metavar="TABLE_NAME",
        help="Ingest SQLite table (default: knowledge_base)"
    )
    group.add_argument(
        "--ingest-path",
        type=str,
        metavar="PATH",
        help="Ingest documents from file path (PDF, TXT, DOCX)"
    )
    
    # Response mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--concise",
        action="store_const",
        const="concise",
        dest="mode",
        help="User-friendly, concise mode (default)"
    )
    mode_group.add_argument(
        "--internal",
        action="store_const",
        const="internal",
        dest="mode",
        help="System integration mode, structured data"
    )
    mode_group.add_argument(
        "--verbose",
        action="store_const",
        const="verbose",
        dest="mode",
        help="Full debug mode for engineers"
    )
    
    # Options
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't show conversation history at end (chat mode only)"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively ingest from subdirectories (ingest-path only)"
    )
    
    args = parser.parse_args()
    
    # Default mode
    mode = args.mode or "concise"
    
    try:
        print(f"\n[INIT] Initializing LangGraph RAG Agent...")
        agent = LangGraphRAGAgent()
        print(f"[OK] Agent ready\n")
        
        if args.chat:
            # Interactive chat mode
            result = agent.invoke(
                operation="chat",
                response_mode=mode,
                show_history=not args.no_history
            )
            print(f"[OK] Chat completed: {result['message_count']} questions")
            
        elif args.ask:
            # Single question
            print(f"[QUERY] {args.ask}\n")
            result = agent.invoke(
                operation="ask_question",
                question=args.ask,
                response_mode=mode
            )
            
            if result.get("success"):
                print(f"[ANSWER]\n{result.get('answer', 'No answer')}\n")
            else:
                print(f"[ERROR] {result.get('errors', 'Unknown error')}\n")
            
        elif args.ingest_table:
            # Ingest SQLite table
            table_name = args.ingest_table
            print(f"[INGEST] Starting table ingestion: {table_name}\n")
            
            result = agent.invoke(
                operation="ingest_sqlite_table",
                table_name=table_name,
                doc_id=f"sqlite_{table_name}",
                rbac_namespace="root"
            )
            
            if result.get("success"):
                print(f"[OK] Successfully ingested {table_name}")
                print(f"    Records: {result.get('records_processed', 0)}")
                print(f"    Chunks: {result.get('total_chunks_saved', 0)}\n")
            else:
                print(f"[ERROR] {result.get('error', 'Ingestion failed')}\n")
                
        elif args.ingest_path:
            # Ingest from file path
            path = args.ingest_path
            print(f"[INGEST] Starting path ingestion: {path}\n")
            
            result = agent.invoke(
                operation="ingest_from_path",
                path=path,
                recursive=args.recursive,
                file_type="auto"
            )
            
            if result.get("success"):
                print(f"[OK] Successfully ingested from {path}")
                print(f"    Documents: {result.get('documents_ingested', 0)}")
                print(f"    Failed: {result.get('documents_failed', 0)}\n")
            else:
                print(f"[ERROR] {result.get('error', 'Ingestion failed')}\n")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n[OK] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
