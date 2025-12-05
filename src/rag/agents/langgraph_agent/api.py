"""
FastAPI Server for LangGraph RAG Agent

Exposes agent functionality as REST API endpoints:
- POST /ingest - Ingest documents
- POST /ask - Ask questions
- POST /optimize - Optimize system
- GET /status - Check agent status
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import logging

from src.rag.agents.langgraph_agent.langgraph_rag_agent import LangGraphRAGAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class IngestRequest(BaseModel):
    """Request to ingest a document"""
    text: str
    doc_id: str
    company_id: Optional[int] = None
    dept_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    """Response from document ingestion"""
    success: bool
    doc_id: str
    chunks_created: int
    vectors_saved: int
    metadata_stored: bool
    message: str
    collection_name: Optional[str] = None
    company_id: Optional[int] = None
    errors: List[str] = []
    workflow_graph: Optional[Dict[str, Any]] = None  # Animated workflow visualization data


class AskRequest(BaseModel):
    """Request to ask a question"""
    question: str
    company_id: Optional[int] = None
    dept_id: Optional[int] = None
    user_id: Optional[int] = None
    response_mode: str = "concise"  # concise|verbose|internal
    top_k: int = 5
    user_role: Optional[str] = None
    doc_id: Optional[str] = None
    performance_history: Optional[List[Dict[str, Any]]] = None


class AskResponse(BaseModel):
    """Response to question"""
    success: bool
    question: str
    answer: Optional[str] = None
    response_mode: str
    context_chunks: int
    guardrails_passed: bool
    traceability: Dict[str, Any] = {}
    confidence_score: Optional[float] = None
    errors: List[str] = []
    workflow_graph: Optional[Dict[str, Any]] = None  # Animated workflow visualization data


class OptimizeRequest(BaseModel):
    """Request to optimize system"""
    performance_history: Optional[List[Dict[str, Any]]] = None
    config_updates: Dict[str, Any] = {}
    optimization_type: Optional[str] = None
    iterations: Optional[int] = None
    quality_threshold: Optional[float] = None


class OptimizeResponse(BaseModel):
    """Response from optimization"""
    success: bool
    optimizations_applied: int
    config_changes: Dict[str, Any]
    performance_improvement: Optional[float] = None
    message: str
    errors: List[str] = []


class StatusResponse(BaseModel):
    """Agent status response"""
    status: str
    initialized: bool
    services_ready: Optional[bool] = None
    vectordb_connected: Optional[bool] = None
    message: Optional[str] = None


class FeedbackRequest(BaseModel):
    """User feedback on answer quality"""
    session_id: Optional[str] = None
    question: str
    answer: str
    rating: int  # 1-5 scale
    user_id: Optional[int] = None
    company_id: Optional[int] = None
    feedback_text: Optional[str] = None
    is_helpful: Optional[bool] = None


class FeedbackResponse(BaseModel):
    """Response to feedback submission"""
    success: bool
    feedback_id: Optional[str] = None
    message: str
    rl_learning_applied: bool
    rating: int
    errors: List[str] = []
class IngestDirectoryRequest(BaseModel):
    """Request to ingest from directory"""
    directory_path: str
    file_extensions: List[str] = ["txt", "md", "pdf"]
    company_id: Optional[int] = None
    dept_id: Optional[int] = None
    recursive: bool = True
    batch_size: int = 10


class IngestDirectoryResponse(BaseModel):
    """Response from directory ingestion"""
    success: bool
    files_processed: int
    chunks_created: int
    vectors_saved: int
    company_id: Optional[int] = None
    dept_id: Optional[int] = None
    failed_files: List[str] = []
    message: str
    errors: List[str] = []


class IngestSQLiteRequest(BaseModel):
    """Request to ingest from SQLite table"""
    database_path: str
    table_name: str
    content_column: str = "content"
    id_column: str = "id"
    batch_size: int = 100
    company_id: Optional[int] = None
    dept_id: Optional[int] = None


class IngestSQLiteResponse(BaseModel):
    """Response from SQLite ingestion"""
    success: bool
    rows_processed: int
    chunks_created: int
    vectors_saved: int
    company_id: Optional[int] = None
    dept_id: Optional[int] = None
    errors: List[str] = []
    message: str


# ============================================================================
# FastAPI Application with Lifespan
# ============================================================================

agent: Optional[LangGraphRAGAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage agent lifecycle (startup/shutdown)
    """
    global agent
    
    # Startup
    try:
        logger.info("Initializing LangGraph RAG Agent...")
        agent = LangGraphRAGAgent()
        logger.info("✅ LangGraph RAG Agent initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {str(e)}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down LangGraph RAG Agent...")
    # Add cleanup if needed
    agent = None


# Create FastAPI app
app = FastAPI(
    title="LangGraph RAG Agent API",
    description="REST API for LangGraph-based RAG system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from HTML UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific URLs in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Check agent health and readiness"""
    if agent is None:
        return StatusResponse(
            status="error",
            initialized=False,
            services_ready=False,
            vectordb_connected=False,
            message="Agent not initialized"
        )
    
    try:
        # Check if services are initialized
        services_ready = agent.llm_service is not None and agent.vectordb_service is not None
        vectordb_ready = services_ready and agent.vectordb_service.client is not None
        
        return StatusResponse(
            status="ready" if services_ready else "degraded",
            initialized=True,
            services_ready=services_ready,
            vectordb_connected=vectordb_ready,
            message="Agent is ready to process requests"
        )
    except Exception as e:
        return StatusResponse(
            status="error",
            initialized=True,
            services_ready=False,
            vectordb_connected=False,
            message=f"Error checking status: {str(e)}"
        )


@app.get("/status", response_model=StatusResponse)
async def status():
    """Get agent status"""
    return await health_check()


# ============================================================================
# Ingestion Endpoints
# ============================================================================

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a document into the RAG system
    
    - **text**: Document content
    - **doc_id**: Unique document identifier
    - **company_id**: Optional company identifier for RBAC
    - **dept_id**: Optional department identifier for RBAC
    - **metadata**: Optional metadata dictionary
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(f"Ingesting document: {request.doc_id}")
        
        result = agent.ingest_document(
            text=request.text,
            doc_id=request.doc_id,
            company_id=request.company_id,
            dept_id=request.dept_id
        )
        
        if result.get("success"):
            # Map ingest_document return values to API response
            chunks_saved = result.get("chunks_saved", 0)
            return IngestResponse(
                success=True,
                doc_id=request.doc_id,
                chunks_created=chunks_saved,  # Use chunks_saved from ingest_document
                vectors_saved=chunks_saved,   # Vectors saved = chunks saved (1:1)
                metadata_stored=result.get("metadata_stored", False),
                message=f"Document {request.doc_id} ingested successfully",
                collection_name=result.get("collection_name"),
                company_id=request.company_id,
                errors=result.get("errors", []),
                workflow_graph=result.get("workflow_graph")  # Include animated workflow data
            )
        else:
            return IngestResponse(
                success=False,
                doc_id=request.doc_id,
                chunks_created=0,
                vectors_saved=0,
                metadata_stored=False,
                message="Document ingestion failed",
                collection_name=result.get("collection_name"),
                company_id=request.company_id,
                errors=result.get("errors", ["Unknown error"]),
                workflow_graph=result.get("workflow_graph")  # Include animated workflow data
            )
    
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ingest_directory", response_model=IngestDirectoryResponse)
async def ingest_directory(request: IngestDirectoryRequest):
    """
    Ingest all documents from a directory into the RAG system
    
    - **directory_path**: Path to directory containing documents
    - **file_extensions**: List of file extensions to process (default: txt, md, pdf)
    - **company_id**: Optional company identifier for RBAC
    - **dept_id**: Optional department identifier for RBAC
    - **recursive**: Recursively process subdirectories (default: True)
    - **batch_size**: Number of files to process in parallel (default: 10)
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(f"Ingesting directory: {request.directory_path}")
        
        result = agent.ingest_directory(
            directory_path=request.directory_path,
            file_extensions=request.file_extensions,
            company_id=request.company_id,
            dept_id=request.dept_id,
            recursive=request.recursive,
            batch_size=request.batch_size
        )
        
        if result.get("success"):
            return IngestDirectoryResponse(
                success=True,
                files_processed=result.get("files_processed", 0),
                chunks_created=result.get("chunks_created", 0),
                vectors_saved=result.get("vectors_saved", 0),
                company_id=request.company_id,
                dept_id=request.dept_id,
                failed_files=result.get("failed_files", []),
                message=f"Directory ingestion complete: {result.get('files_processed', 0)} files processed",
                errors=result.get("errors", [])
            )
        else:
            return IngestDirectoryResponse(
                success=False,
                files_processed=result.get("files_processed", 0),
                chunks_created=result.get("chunks_created", 0),
                vectors_saved=result.get("vectors_saved", 0),
                company_id=request.company_id,
                dept_id=request.dept_id,
                failed_files=result.get("failed_files", []),
                message="Directory ingestion failed",
                errors=result.get("errors", ["Directory ingestion failed"])
            )
    
    except Exception as e:
        logger.error(f"Error ingesting directory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Directory ingestion failed: {str(e)}")


@app.post("/ingest_sqlite", response_model=IngestSQLiteResponse)
async def ingest_sqlite(request: IngestSQLiteRequest):
    """
    Ingest data from a SQLite database table into the RAG system
    
    - **database_path**: Path to SQLite database file
    - **table_name**: Name of table to ingest
    - **content_column**: Column name containing content to ingest
    - **id_column**: Column name for row identifiers
    - **batch_size**: Number of rows to process per batch (default: 100)
    - **company_id**: Optional company identifier for RBAC
    - **dept_id**: Optional department identifier for RBAC
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(f"Ingesting SQLite table: {request.table_name}")
        
        result = agent.ingest_sqlite(
            database_path=request.database_path,
            table_name=request.table_name,
            content_column=request.content_column,
            id_column=request.id_column,
            batch_size=request.batch_size,
            company_id=request.company_id,
            dept_id=request.dept_id
        )
        
        if result.get("success"):
            return IngestSQLiteResponse(
                success=True,
                rows_processed=result.get("rows_processed", 0),
                chunks_created=result.get("chunks_created", 0),
                vectors_saved=result.get("vectors_saved", 0),
                company_id=request.company_id,
                dept_id=request.dept_id,
                errors=result.get("errors", []),
                message=f"SQLite ingestion complete: {result.get('rows_processed', 0)} rows processed"
            )
        else:
            return IngestSQLiteResponse(
                success=False,
                rows_processed=result.get("rows_processed", 0),
                chunks_created=result.get("chunks_created", 0),
                vectors_saved=result.get("vectors_saved", 0),
                company_id=request.company_id,
                dept_id=request.dept_id,
                errors=result.get("errors", ["SQLite ingestion failed"]),
                message="SQLite ingestion failed"
            )
    
    except Exception as e:
        logger.error(f"Error ingesting SQLite: {str(e)}")
        raise HTTPException(status_code=500, detail=f"SQLite ingestion failed: {str(e)}")


# ============================================================================
# Retrieval Endpoints
# ============================================================================

@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question to the RAG system
    
    - **question**: The question to ask
    - **response_mode**: Response format (concise|verbose|internal)
    - **top_k**: Number of context chunks to retrieve
    - **user_role**: User role for RBAC
    - **doc_id**: Specific document to search (optional)
    - **company_id**: Company ID for RBAC filtering
    - **dept_id**: Department ID for RBAC filtering
    - **user_id**: User ID for RBAC filtering
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(f"Processing question: {request.question}")
        
        result = agent.ask_question(
            question=request.question,
            performance_history=request.performance_history,
            doc_id=request.doc_id,
            response_mode=request.response_mode,
            company_id=request.company_id,
            dept_id=request.dept_id,
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        if result.get("success"):
            answer = result.get("answer") or "No answer generated. Please try rephrasing your question."
            
            # Map different response modes to consistent field names
            context_chunks = (
                result.get("context_chunks", 0) or  # concise/internal mode
                result.get("sources_count", 0) or   # verbose mode
                len(result.get("sources", []))      # verbose mode (fallback)
            )
            
            return {
                "success": True,
                "question": request.question,
                "answer": answer,
                "response_mode": request.response_mode,
                "context_chunks": context_chunks,
                "guardrails_passed": result.get("guardrails_passed", True),
                "traceability": result.get("traceability", {}),
                "confidence_score": result.get("confidence_score"),
                "errors": result.get("errors", []),
                "workflow_graph": result.get("workflow_graph")
            }
        else:
            return {
                "success": False,
                "question": request.question,
                "answer": "Failed to generate answer. Please check the logs.",
                "response_mode": request.response_mode,
                "context_chunks": 0,
                "guardrails_passed": False,
                "errors": result.get("errors", ["Failed to generate answer"]),
                "workflow_graph": result.get("workflow_graph")
            }
    
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")


# ============================================================================
# Feedback & Learning Endpoints
# ============================================================================

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on answer quality for RL learning
    
    - **rating**: Quality rating from 1 (very bad) to 5 (excellent)
    - **is_helpful**: Whether the answer was helpful (true/false)
    - **feedback_text**: Optional detailed feedback
    
    This endpoint feeds user satisfaction data into the RL agent to improve
    future answer generation and ranking strategies.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info(f"Processing feedback - rating: {request.rating}, question: {request.question[:50]}")
        
        # Validate rating
        if request.rating < 1 or request.rating > 5:
            raise ValueError("Rating must be between 1 and 5")
        
        # Store feedback in RL agent for learning
        feedback_data = {
            "session_id": request.session_id,
            "question": request.question,
            "answer": request.answer,
            "rating": request.rating,
            "user_id": request.user_id,
            "company_id": request.company_id,
            "feedback_text": request.feedback_text,
            "is_helpful": request.is_helpful,
            "timestamp": __import__('time').time()
        }
        
        # Feed to RL agent for learning
        rl_learning_applied = False
        if hasattr(agent, 'rl_agent') and agent.rl_agent:
            try:
                # Update RL agent with feedback signal
                agent.rl_agent.process_feedback(feedback_data)
                rl_learning_applied = True
                logger.info(f"RL agent updated with feedback - rating: {request.rating}")
            except Exception as e:
                logger.warning(f"Failed to apply RL learning: {e}")
        
        return {
            "success": True,
            "feedback_id": f"fb_{feedback_data['timestamp']}",
            "message": f"Feedback received - rating: {request.rating}/5",
            "rl_learning_applied": rl_learning_applied,
            "rating": request.rating,
            "errors": []
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")


# ============================================================================
# Optimization Endpoints
# ============================================================================

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_system(request: OptimizeRequest):
    """
    Optimize the system based on performance history
    
    - **performance_history**: List of performance metrics (optional)
    - **config_updates**: Configuration changes to apply
    - **optimization_type**: Type of optimization (optional)
    - **iterations**: Number of optimization iterations (optional)
    - **quality_threshold**: Quality threshold for optimization (optional)
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        logger.info("Starting system optimization")
        
        # Use provided performance_history or create an empty list
        perf_history = request.performance_history or []
        
        # Add optimization parameters to config_updates if provided
        config_updates = request.config_updates.copy()
        if request.optimization_type:
            config_updates["optimization_type"] = request.optimization_type
        if request.iterations is not None:
            config_updates["iterations"] = request.iterations
        if request.quality_threshold is not None:
            config_updates["quality_threshold"] = request.quality_threshold
        
        result = agent.optimize_system(
            performance_history=perf_history,
            config_updates=config_updates
        )
        
        if result.get("success"):
            return OptimizeResponse(
                success=True,
                optimizations_applied=result.get("optimizations_applied", 0),
                config_changes=result.get("config_changes", {}),
                performance_improvement=result.get("performance_improvement"),
                message="System optimized successfully",
                errors=result.get("errors", [])
            )
        else:
            return OptimizeResponse(
                success=False,
                optimizations_applied=0,
                config_changes={},
                message="System optimization failed",
                errors=result.get("errors", ["Optimization failed"])
            )
    
    except Exception as e:
        logger.error(f"Error optimizing system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "name": "LangGraph RAG Agent API",
        "version": "1.0.0",
        "status": "ready" if agent else "initializing",
        "endpoints": {
            "health": "GET /health",
            "status": "GET /status",
            "ingest": "POST /ingest",
            "ask": "POST /ask",
            "optimize": "POST /optimize",
            "docs": "GET /docs",
            "openapi": "GET /openapi.json"
        }
    }


# ============================================================================
@app.get("/vectordb/stats")
async def get_vectordb_stats():
    """Get vector database statistics and list all collections"""
    try:
        if agent is None or agent.vectordb_service is None:
            return {
                "status": "error",
                "message": "VectorDB service not available",
                "total_collections": 0,
                "total_vectors": 0,
                "collections": []
            }
        
        # Get all collections from ChromaDB
        client = agent.vectordb_service.client
        all_collections = client.list_collections()
        
        collections_info = []
        total_vectors = 0
        
        for collection in all_collections:
            try:
                count = collection.count()
                total_vectors += count
                collections_info.append({
                    "name": collection.name,
                    "vector_count": count,
                    "metadata": collection.metadata if hasattr(collection, 'metadata') else {}
                })
                print(f"[DEBUG] Collection: {collection.name}, Documents: {count}")
            except Exception as e:
                print(f"[ERROR] Failed to count collection {collection.name}: {e}")
        
        return {
            "status": "healthy",
            "total_collections": len(collections_info),
            "total_vectors": total_vectors,
            "persist_directory": agent.vectordb_service.persist_directory,
            "collections": collections_info
        }
    except Exception as e:
        print(f"[ERROR] Failed to get VectorDB stats: {e}")
        return {
            "status": "error",
            "message": str(e),
            "total_collections": 0,
            "total_vectors": 0,
            "collections": []
        }


@app.get("/vectordb/collection/{collection_name}")
async def get_collection_details(collection_name: str):
    """Get detailed statistics for a specific collection"""
    try:
        if agent is None or agent.vectordb_service is None:
            return {
                "status": "error",
                "message": "VectorDB service not available",
                "collection": collection_name,
                "vector_count": 0,
                "rbac_tags": [],
                "meta_tags": []
            }
        
        client = agent.vectordb_service.client
        collection = client.get_collection(collection_name)
        
        count = collection.count()
        
        # Get sample data to analyze RBAC and meta tags
        rbac_tags_set = set()
        meta_tags_set = set()
        doc_ids_set = set()
        
        if count > 0:
            sample_size = min(1000, count)
            sample = collection.get(limit=sample_size)
            
            if sample["metadatas"]:
                for meta in sample["metadatas"]:
                    if meta:
                        if "rbac_tags" in meta:
                            rbac_tags_set.add(meta["rbac_tags"])
                        if "meta_tags" in meta and meta["meta_tags"]:
                            # Split semicolon-joined meta tags
                            tags = meta["meta_tags"].split(";")
                            meta_tags_set.update([t.strip() for t in tags if t.strip()])
            
            if sample["ids"]:
                doc_ids_set.update(sample["ids"])
        
        return {
            "status": "success",
            "collection": collection_name,
            "vector_count": count,
            "unique_documents": len(doc_ids_set),
            "rbac_tags": sorted(list(rbac_tags_set)),
            "meta_tags": sorted(list(meta_tags_set)),
            "metadata": collection.metadata if hasattr(collection, 'metadata') else {}
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "collection": collection_name
        }


# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.get("/rag-history")
def get_rag_history(limit: int = 50, event_type: str = None):
    """
    Get RAG query history for analytics dashboard.
    
    Args:
        limit: Number of recent records to return
        event_type: Filter by event type (QUERY, HEAL, SYNTHETIC_TEST)
    
    Returns:
        List of history records with metrics for RAGAS analysis
    """
    try:
        from src.rag.rag_db_models.db_models import RAGHistoryModel
        
        rag_history = RAGHistoryModel()
        
        # Get recent records
        query = f"SELECT * FROM rag_history_and_optimization ORDER BY timestamp DESC LIMIT ?"
        
        if event_type:
            query = f"SELECT * FROM rag_history_and_optimization WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?"
            cur = rag_history.conn.execute(query, (event_type, limit))
        else:
            cur = rag_history.conn.execute(query, (limit,))
        
        rag_history.conn.row_factory = __import__('sqlite3').Row
        records = cur.fetchall()
        
        # Convert to list of dicts with all fields
        history_list = []
        for record in records:
            record_dict = dict(record)
            history_list.append(record_dict)
        
        return {
            "status": "success",
            "count": len(history_list),
            "records": history_list
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "records": []
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
