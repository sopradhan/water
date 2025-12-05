"""
Integration Guide: Enhanced Ingestion Pipeline

This document shows how to integrate the new classification, markdown conversion,
and guardrails components into the existing ingestion_tools.py
"""

# ============================================================================
# ENHANCED INGESTION PIPELINE INTEGRATION
# ============================================================================

"""
STEP 1: Update ingestion_tools.py imports
"""

# Add these imports at the top of ingestion_tools.py:
"""
from .document_classification_tool import (
    enhance_document_metadata_tool,
    DocumentClassificationAgent
)
from .document_markdown_converter import (
    convert_to_markdown_tool,
    DocumentToMarkdownConverter
)
from ..guardrails.guardrails_service import GuardrailsService
"""

# ============================================================================
# STEP 2: Enhanced chunk_document_tool
# ============================================================================

"""
Replace/enhance the existing chunk_document_tool to work with markdown:

@tool
def chunk_document_markdown_tool(
    markdown_text: str,
    doc_id: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> str:
    '''
    Chunk markdown-formatted document into semantic pieces.
    
    Advantages of markdown chunking:
    - Preserves structure (headers, lists, tables)
    - Better semantic boundaries
    - Improved embedding quality
    
    Args:
        markdown_text: Markdown formatted content
        doc_id: Document identifier
        chunk_size: Size of chunks in characters
        overlap: Character overlap between chunks
    
    Returns:
        JSON string with chunks
    '''
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        
        # First, split by markdown headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False
        )
        
        md_chunks = markdown_splitter.split_text(markdown_text)
        
        # Then, further split large sections
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        final_chunks = []
        for chunk in md_chunks:
            if len(chunk.page_content) > chunk_size:
                sub_chunks = text_splitter.split_text(chunk.page_content)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk.page_content)
        
        return json.dumps({
            "success": True,
            "num_chunks": len(final_chunks),
            "chunks": final_chunks,
            "doc_id": doc_id,
            "average_chunk_size": sum(len(c) for c in final_chunks) / max(len(final_chunks), 1)
        })
    
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
"""

# ============================================================================
# STEP 3: Enhanced save_to_vectordb_tool with tagging
# ============================================================================

"""
Enhance save_to_vectordb_tool to handle RBAC and meta tags:

@tool
def save_to_vectordb_enhanced_tool(
    chunks: str,
    doc_id: str,
    llm_service,
    vectordb_service,
    metadata: str,
    db_conn=None
) -> str:
    '''
    Save chunks to vector DB with RBAC and meta tags.
    
    Args:
        chunks: JSON string with list of chunks
        doc_id: Document ID
        llm_service: LLMService for embeddings
        vectordb_service: VectorDB service
        metadata: JSON string with enhanced metadata including tags
        db_conn: Database connection for RBAC lookup
    
    Returns:
        JSON string with save result
    '''
    try:
        chunks_data = json.loads(chunks)
        chunks_list = chunks_data if isinstance(chunks_data, list) else chunks_data.get("chunks", [])
        
        metadata_obj = json.loads(metadata)
        rbac_tags = metadata_obj.get("tags", {}).get("rbac", [])
        meta_tags = metadata_obj.get("tags", {}).get("meta", [])
        
        # Generate embeddings for each chunk
        chunk_ids = []
        embeddings = []
        texts = []
        metadatas = []
        
        for idx, chunk_text in enumerate(chunks_list):
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_ids.append(chunk_id)
            texts.append(chunk_text)
            
            # Generate embedding
            embedding = llm_service.generate_embedding(chunk_text)
            embeddings.append(embedding)
            
            # Build metadata with tags
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_idx": idx,
                "rbac_tags": rbac_tags,
                "meta_tags": meta_tags,
                "title": metadata_obj.get("title", ""),
                "source": metadata_obj.get("source", ""),
                "classification": metadata_obj.get("classification", {}),
                "keywords": metadata_obj.get("keywords", []),
                "created_at": datetime.now().isoformat()
            }
            metadatas.append(chunk_metadata)
        
        # Save to ChromaDB with metadata
        result = vectordb_service.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return json.dumps({
            "success": True,
            "chunks_saved": len(chunk_ids),
            "doc_id": doc_id,
            "rbac_tags_applied": rbac_tags,
            "meta_tags_applied": meta_tags,
            "message": f"Saved {len(chunk_id)} chunks with RBAC and meta tags"
        })
    
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})
"""

# ============================================================================
# STEP 4: New unified ingestion tool with all enhancements
# ============================================================================

"""
@tool
def ingest_document_enhanced_tool(
    file_path: str,
    doc_id: str,
    doc_title: str,
    source_type: str,  # pdf, xlsx, csv, docx, txt, database_table
    llm_service,
    vectordb_service,
    db_conn=None,
    table_data: List[Dict] = None
) -> str:
    '''
    Complete enhanced ingestion pipeline:
    1. Convert to markdown
    2. Classify document (intent, dept, role, sensitivity)
    3. Generate RBAC + meta tags
    4. Chunk document
    5. Generate embeddings
    6. Save to vector DB with tags
    7. Store metadata in database
    
    Args:
        file_path: Path to file or content
        doc_id: Document identifier
        doc_title: Document title
        source_type: Type of source (pdf, xlsx, docx, etc.)
        llm_service: LLM service
        vectordb_service: Vector DB service
        db_conn: Database connection
        table_data: For database table ingestion
    
    Returns:
        JSON string with ingestion result
    '''
    try:
        # STEP 1: Convert to markdown
        converter = DocumentToMarkdownConverter()
        
        if source_type == "pdf":
            markdown = converter.pdf_to_markdown(file_path, doc_title)
        elif source_type == "xlsx":
            markdown = converter.excel_to_markdown(file_path, title=doc_title)
        elif source_type == "csv":
            markdown = converter.csv_to_markdown(file_path, doc_title)
        elif source_type == "docx":
            markdown = converter.word_to_markdown(file_path, doc_title)
        elif source_type == "database_table":
            markdown = converter.database_table_to_markdown(table_data, doc_title)
        else:  # txt
            with open(file_path, 'r') as f:
                content = f.read()
            markdown = converter.text_to_markdown(content, doc_title)
        
        # STEP 2: Classify document
        classifier = DocumentClassificationAgent(llm_service, db_conn)
        classification_result = classifier.classify_document_intent(markdown[:2000], doc_title)
        
        if not classification_result.get("success"):
            classification = classification_result.get("fallback_classification", {})
        else:
            classification = classification_result.get("classification", {})
        
        # STEP 3: Generate tags
        tags = classifier.generate_all_tags(classification)
        
        # STEP 4: Chunk document
        chunk_result = chunk_document_markdown_tool.invoke({
            "markdown_text": markdown,
            "doc_id": doc_id,
            "chunk_size": 500,
            "overlap": 50
        })
        chunks_data = json.loads(chunk_result)
        chunks_list = chunks_data.get("chunks", [])
        
        # STEP 5 & 6 & 7: Save to vector DB with enhanced metadata
        metadata = {
            "title": doc_title,
            "source": source_type,
            "tags": tags,
            "classification": classification,
            "keywords": classification.get("keywords", []),
            "created_at": datetime.now().isoformat()
        }
        
        save_result = save_to_vectordb_enhanced_tool.invoke({
            "chunks": json.dumps(chunks_list),
            "doc_id": doc_id,
            "llm_service": llm_service,
            "vectordb_service": vectordb_service,
            "metadata": json.dumps(metadata),
            "db_conn": db_conn
        })
        
        # STEP 8: Store metadata in SQLite database
        if db_conn:
            from ...database.models.document_metadata_model import DocumentMetadataModel
            
            doc_model = DocumentMetadataModel(db_conn)
            doc_model.create(
                doc_id=doc_id,
                title=doc_title,
                source=source_type,
                summary=classification.get("reasoning", ""),
                rbac_namespace=",".join(classification.get("relevant_departments", [])),
                metadata_json=json.dumps({
                    "classification": classification,
                    "rbac_tags": tags.get("rbac_tags", []),
                    "meta_tags": tags.get("meta_tags", []),
                    "keywords": classification.get("keywords", []),
                    "confidence_score": classification.get("confidence_score", 0.8)
                })
            )
        
        return json.dumps({
            "success": True,
            "doc_id": doc_id,
            "title": doc_title,
            "source_type": source_type,
            "chunks_created": len(chunks_list),
            "classification": classification,
            "rbac_tags": tags.get("rbac_tags", []),
            "meta_tags": tags.get("meta_tags", []),
            "message": "Document successfully ingested with classification and tagging"
        })
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "doc_id": doc_id
        })
"""

# ============================================================================
# STEP 5: Integration with Retrieval
# ============================================================================

"""
Update retrieval pipeline to use RBAC + guardrails:

@tool
def retrieve_and_respond_with_guardrails_tool(
    question: str,
    user_id: str,
    username: str,
    department: str,
    role: str,
    is_admin: bool,
    response_mode: str,
    llm_service,
    vectordb_service,
    db_conn=None,
    k: int = 5
) -> str:
    '''
    Complete retrieval + response generation + guardrails pipeline.
    '''
    try:
        from .rbac_retrieval_tool import retrieve_with_rbac_tool, generate_response_with_mode_tool
        from ..guardrails.guardrails_service import GuardrailsService
        
        # Step 1: Retrieve with RBAC
        retrieval_result = retrieve_with_rbac_tool.invoke({
            "question": question,
            "user_id": user_id,
            "username": username,
            "department": department,
            "role": role,
            "is_admin": is_admin,
            "response_mode": response_mode,
            "vectordb_service": vectordb_service,
            "llm_service": llm_service,
            "k": k
        })
        
        retrieval_data = json.loads(retrieval_result)
        
        if not retrieval_data.get("success"):
            return retrieval_result  # Return retrieval error
        
        # Step 2: Build context
        results = retrieval_data.get("results", [])
        context = "\\n\\n".join([r["text"] for r in results])
        
        # Step 3: Generate response in specified mode
        response_result = generate_response_with_mode_tool.invoke({
            "question": question,
            "context": context,
            "response_mode": response_mode,
            "llm_service": llm_service
        })
        
        response_data = json.loads(response_result)
        generated_response = response_data.get("response", "")
        
        # Step 4: Validate with guardrails
        guardrails = GuardrailsService(llm_service)
        validation = guardrails.validate_response(
            response=generated_response,
            context=context,
            question=question
        )
        
        # Step 5: Return result with guardrail info
        return json.dumps({
            "success": True,
            "question": question,
            "response": generated_response,
            "response_mode": response_mode,
            "user": {
                "user_id": user_id,
                "username": username,
                "department": department,
                "role": role
            },
            "guardrail_validation": validation,
            "is_safe": validation.get("is_safe", False),
            "risk_level": validation.get("max_risk_level", "unknown"),
            "recommendation": validation.get("recommendation", ""),
            "sources": [{"doc_id": r.get("metadata", {}).get("doc_id"), 
                        "relevance": r.get("relevance_score")} for r in results],
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })
"""

# ============================================================================
# STEP 6: Usage in LangGraph Agent
# ============================================================================

"""
Update LangGraph agent to use enhanced tools:

def _build_ingestion_graph(self):
    '''Updated ingestion graph with classification, markdown, and tagging.'''
    graph = StateGraph(dict)
    
    def ingestion_node(state):
        try:
            result = ingest_document_enhanced_tool.invoke({
                "file_path": state.get("file_path"),
                "doc_id": state.get("doc_id"),
                "doc_title": state.get("title"),
                "source_type": state.get("source_type"),
                "llm_service": self.llm_service,
                "vectordb_service": self.vectordb_service,
                "db_conn": self.db_conn,
                "table_data": state.get("table_data")
            })
            
            result_data = json.loads(result)
            state["ingestion_result"] = result_data
            state["status"] = "ingestion_complete" if result_data.get("success") else "ingestion_failed"
            
        except Exception as e:
            state["errors"] = state.get("errors", []) + [str(e)]
            state["status"] = "ingestion_failed"
        
        return state
    
    graph.add_node("ingestion", ingestion_node)
    graph.add_edge(START, "ingestion")
    graph.add_edge("ingestion", END)
    
    return graph.compile()
"""

# ============================================================================
# STEP 7: Usage in DeepAgents
# ============================================================================

"""
Add as ingestion subagent in DeepAgents:

ingestion_subagent = create_deep_agent(
    "Ingestion Agent",
    "Ingests documents with classification, markdown conversion, and RBAC tagging",
    tools=[
        ingest_document_enhanced_tool,
        convert_to_markdown_tool,
        enhance_document_metadata_tool
    ],
    system_prompt='''You are an ingestion specialist. Your job is to:
1. Accept documents of any type (PDF, Excel, Word, CSV, etc.)
2. Convert to markdown format
3. Classify the document (intent, department, role, sensitivity)
4. Generate RBAC and meta tags
5. Save to vector database with appropriate access controls
'''
)
"""

print(__doc__)
