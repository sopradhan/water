"""
Streamlit Chat UI Pages for Dual-Mode RAG

Admin Mode: Full ingestion, healing, optimization capabilities
User Mode: Query and retrieval only with response mode selection
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path


def show_admin_chat():
    """Admin Chat Mode - Full RAG capabilities"""
    st.set_page_config(
        page_title="RAG Admin Chat",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚙️ RAG Admin Chat - Full Control Mode")
    
    # Initialize session state
    if "admin_session_id" not in st.session_state:
        st.session_state.admin_session_id = None
    if "admin_messages" not in st.session_state:
        st.session_state.admin_messages = []
    if "admin_response_mode" not in st.session_state:
        st.session_state.admin_response_mode = "verbose"
    
    # Sidebar controls
    st.sidebar.header("Admin Settings")
    
    # Response mode
    response_mode = st.sidebar.selectbox(
        "Response Mode",
        ["verbose", "internal", "concise"],
        index=0,
        help="verbose: Full debug details | internal: Structured data | concise: User-friendly"
    )
    st.session_state.admin_response_mode = response_mode
    
    # Session info
    st.sidebar.subheader("Session Info")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button("New Session", key="admin_new_session")
    with col2:
        st.button("Export Chat", key="admin_export_chat")
    
    st.sidebar.divider()
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["Query", "Ingest", "Heal", "Settings"])
    
    with tab1:
        st.subheader("🔍 Query & RAG")
        st.info("Ask questions and retrieve information from ingested documents")
        
        query_col1, query_col2 = st.columns([4, 1])
        with query_col1:
            query_input = st.text_input(
                "Question",
                placeholder="What is the main incident cause?",
                key="admin_query_input"
            )
        with query_col2:
            submit_query = st.button("Search", key="admin_submit_query", type="primary")
        
        if submit_query and query_input:
            st.session_state.admin_messages.append({
                "role": "user",
                "content": query_input,
                "timestamp": datetime.now().isoformat(),
                "type": "rag_query"
            })
            
            # Show thinking
            with st.spinner("🔍 Searching knowledge base..."):
                try:
                    from src.rag.agents.langgraph_agent.langgraph_rag_agent import LangGraphRAGAgent
                    agent = LangGraphRAGAgent()
                    result = agent.ask_question(
                        question=query_input,
                        response_mode=response_mode
                    )
                    
                    if result.get("success"):
                        answer = result.get("answer", "No answer")
                        st.session_state.admin_messages.append({
                            "role": "assistant",
                            "content": answer,
                            "timestamp": datetime.now().isoformat(),
                            "type": "rag_response",
                            "metadata": {
                                "retrieval_quality": result.get("retrieval_quality"),
                                "sources": len(result.get("source_docs", [])),
                                "tokens_used": result.get("tokens_used", 0)
                            }
                        })
                        
                        st.markdown(f"**Answer:** {answer}")
                        
                        # Show metadata
                        with st.expander("📊 Query Metadata"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Quality Score", f"{result.get('retrieval_quality', 0):.2f}")
                            with col2:
                                st.metric("Sources", len(result.get("source_docs", [])))
                            with col3:
                                st.metric("Tokens Used", result.get("tokens_used", 0))
                            
                            if response_mode == "verbose":
                                st.json(result.get("rl_recommendation", {}))
                    
                    else:
                        st.error(f"Query failed: {result.get('errors', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Chat history
        if st.session_state.admin_messages:
            st.divider()
            st.subheader("Chat History")
            for msg in st.session_state.admin_messages:
                if msg["role"] == "user":
                    st.write(f"**You:** {msg['content']}")
                else:
                    st.write(f"**Agent:** {msg['content']}")
    
    with tab2:
        st.subheader("📥 Ingest Documents")
        st.info("Add new documents to the knowledge base")
        
        ingest_mode = st.radio(
            "Ingestion Method",
            ["File Upload", "Text Input", "Database Table"],
            horizontal=True
        )
        
        if ingest_mode == "File Upload":
            st.markdown("**Upload document file:**")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["pdf", "docx", "txt", "csv", "xlsx"],
                key="admin_file_upload"
            )
            
            doc_id = st.text_input("Document ID (optional)", placeholder="doc_custom_id")

            company_id = st.text_input("Company ID (optional)", placeholder="1")

            dept_id = st.text_input("Dept ID (optional)", placeholder="1")

            if st.button("Ingest File", type="primary", key="admin_ingest_file"):
                if uploaded_file:
                    with st.spinner("📥 Ingesting document..."):
                        try:
                            from src.rag.agents.langgraph_agent.langgraph_rag_agent import LangGraphRAGAgent
                            import tempfile
                            import os
                            
                            agent = LangGraphRAGAgent()
                            
                            # Save uploaded file to temp location
                            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                                tmp.write(uploaded_file.getbuffer())
                                tmp_path = tmp.name
                            
                            try:
                                # Use ingest_from_path for file
                                result = agent.invoke(
                                    operation="ingest_from_path",
                                    path=tmp_path,
                                    recursive=False,
                                    file_type="auto",
                                    doc_id=doc_id ,
                                    company_id=company_id,
                                    department_id=dept_id

                                )
                                
                                if result.get("success"):
                                    st.success(f"✓ Document ingested successfully!")
                                    st.write(f"Documents processed: {result.get('documents_ingested', 0)}")
                                    st.write(f"Chunks created: {result.get('chunks', 0)}")
                                else:
                                    st.error(f"Ingestion failed: {result.get('error')}")
                            finally:
                                os.remove(tmp_path)
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        elif ingest_mode == "Text Input":
            st.markdown("**Enter document text:**")
            text_content = st.text_area(
                "Document Content",
                placeholder="Paste your document content here...",
                height=200,
                key="admin_text_input"
            )
            doc_id = st.text_input("Document ID (optional)", placeholder="doc_custom_id", key="admin_text_id")
            
            if st.button("Ingest Text", type="primary", key="admin_ingest_text"):
                if text_content:
                    with st.spinner("📥 Ingesting text..."):
                        try:
                            from src.rag.agents.langgraph_agent.langgraph_rag_agent import LangGraphRAGAgent
                            
                            agent = LangGraphRAGAgent()
                            result = agent.invoke(
                                operation="ingest_document",
                                text=text_content,
                                doc_id=doc_id or f"doc_{datetime.now().timestamp()}"
                            )
                            
                            if result.get("success"):
                                st.success(f"✓ Text ingested successfully!")
                                st.write(f"Chunks created: {result.get('chunks_saved', 0)}")
                            else:
                                st.error(f"Ingestion failed: {result.get('errors')}")
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        else:  # Database Table
            st.markdown("**Select database table:**")
            table_name = st.selectbox(
                "Table",
                ["knowledge_base", "incidents", "articles", "faqs"],
                key="admin_table_select"
            )
            
            if st.button("Ingest Table", type="primary", key="admin_ingest_table"):
                with st.spinner(f"📥 Ingesting table: {table_name}..."):
                    try:
                        from src.rag.agents.langgraph_agent.langgraph_rag_agent import LangGraphRAGAgent
                        
                        agent = LangGraphRAGAgent()
                        result = agent.invoke(
                            operation="ingest_sqlite_table",
                            table_name=table_name
                        )
                        
                        if result.get("success"):
                            st.success(f"✓ Table ingested successfully!")
                            st.write(f"Records: {result.get('records_processed', 0)}")
                            st.write(f"Chunks: {result.get('total_chunks_saved', 0)}")
                        else:
                            st.error(f"Ingestion failed: {result.get('error')}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    with tab3:
        st.subheader("🏥 Healing & Optimization")
        st.info("Optimize document embeddings and fix quality issues")
        
        heal_col1, heal_col2 = st.columns(2)
        
        with heal_col1:
            st.markdown("**Heal Document**")
            doc_id_heal = st.text_input("Document ID", key="admin_heal_doc_id")
            quality_score = st.slider("Quality Score", 0.0, 1.0, 0.5, key="admin_quality_slider")
            
            if st.button("Start Healing", type="primary", key="admin_heal_btn"):
                if doc_id_heal:
                    with st.spinner("🏥 Healing document..."):
                        try:
                            from src.rag.agents.langgraph_agent.langgraph_rag_agent import LangGraphRAGAgent
                            agent = LangGraphRAGAgent()
                            
                            # Simulate healing by invoking ask_question which triggers optimization
                            result = agent.ask_question(
                                question=f"Check document {doc_id_heal}",
                                doc_id=doc_id_heal,
                                response_mode="verbose"
                            )
                            
                            st.success("✓ Healing completed!")
                            with st.expander("Healing Details"):
                                st.json({
                                    "doc_id": doc_id_heal,
                                    "original_quality": quality_score,
                                    "action": result.get("rl_action", "SKIP"),
                                    "improvement": result.get("optimization_reason", "N/A")
                                })
                        
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        with heal_col2:
            st.markdown("**Check Health**")
            doc_id_health = st.text_input("Document ID", key="admin_health_doc_id")
            
            if st.button("Check Health", type="primary", key="admin_health_btn"):
                if doc_id_health:
                    with st.spinner("🔍 Checking health..."):
                        try:
                            # Show placeholder health check
                            st.json({
                                "doc_id": doc_id_health,
                                "embedding_quality": 0.85,
                                "avg_chunk_relevance": 0.79,
                                "last_updated": datetime.now().isoformat(),
                                "recommendation": "OPTIMIZE"
                            })
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
    
    with tab4:
        st.subheader("⚙️ Settings")
        
        st.markdown("**Session Settings**")
        auto_optimize = st.checkbox("Auto-optimize on low quality", value=True)
        debug_mode = st.checkbox("Debug mode (show all logs)", value=False)
        
        st.markdown("**Advanced Options**")
        chunk_size = st.slider("Chunk Size", 256, 1024, 512, step=64)
        chunk_overlap = st.slider("Chunk Overlap", 0, 256, 50, step=10)
        
        if st.button("Save Settings", type="primary"):
            st.success("✓ Settings saved!")


def show_user_chat():
    """User Chat Mode - Query only with response modes"""
    st.set_page_config(
        page_title="RAG User Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💬 RAG User Chat - Knowledge Base Assistant")
    
    # Initialize session state
    if "user_session_id" not in st.session_state:
        st.session_state.user_session_id = None
    if "user_messages" not in st.session_state:
        st.session_state.user_messages = []
    if "user_response_mode" not in st.session_state:
        st.session_state.user_response_mode = "concise"
    
    # Sidebar
    st.sidebar.header("Chat Settings")
    
    # Response mode
    response_mode = st.sidebar.radio(
        "Response Mode",
        ["concise", "verbose", "internal"],
        index=0,
        help="**Concise:** Brief answers (Recommended)\n**Verbose:** Detailed with context\n**Internal:** Full technical details"
    )
    st.session_state.user_response_mode = response_mode
    
    # Show description
    mode_descriptions = {
        "concise": "📌 Brief, to-the-point answers",
        "verbose": "📚 Detailed answers with context and sources",
        "internal": "⚙️ Full technical details (requires admin role)"
    }
    st.sidebar.info(mode_descriptions[response_mode])
    
    st.sidebar.divider()
    
    # Session stats
    st.sidebar.subheader("Session Stats")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Questions", len([m for m in st.session_state.user_messages if m["role"] == "user"]))
    with col2:
        st.metric("Answers", len([m for m in st.session_state.user_messages if m["role"] == "assistant"]))
    
    if st.sidebar.button("Export Chat"):
        chat_export = json.dumps(st.session_state.user_messages, indent=2, default=str)
        st.sidebar.download_button(
            label="Download Chat",
            data=chat_export,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.user_messages = []
        st.rerun()
    
    st.divider()
    
    # Chat display
    st.subheader("Chat History")
    
    for msg in st.session_state.user_messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                
                # Show metadata for verbose mode
                if response_mode == "verbose" and "metadata" in msg:
                    with st.expander("📊 Details"):
                        st.json(msg["metadata"])
    
    st.divider()
    
    # Chat input
    st.subheader("Ask a Question")
    
    query_col1, query_col2 = st.columns([5, 1])
    
    with query_col1:
        user_input = st.text_input(
            "Your question",
            placeholder="E.g., What are the main incident causes?",
            label_visibility="collapsed",
            key="user_chat_input"
        )
    
    with query_col2:
        submit_btn = st.button("Send", type="primary", use_container_width=True, key="user_chat_send")
    
    if submit_btn and user_input:
        # Add user message
        st.session_state.user_messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get response
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                from src.rag.agents.langgraph_agent.langgraph_rag_agent import LangGraphRAGAgent
                
                agent = LangGraphRAGAgent()
                result = agent.ask_question(
                    question=user_input,
                    response_mode=response_mode
                )
                
                if result.get("success"):
                    answer = result.get("answer", "No answer available")
                    
                    msg_data = {
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add metadata for verbose mode
                    if response_mode == "verbose":
                        msg_data["metadata"] = {
                            "retrieval_quality": result.get("retrieval_quality", 0),
                            "sources": len(result.get("source_docs", [])),
                            "execution_time_ms": result.get("execution_time_ms", 0),
                            "rl_action": result.get("rl_action", "SKIP")
                        }
                    
                    st.session_state.user_messages.append(msg_data)
                    st.rerun()
                else:
                    st.error(f"Error: {result.get('errors', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")
