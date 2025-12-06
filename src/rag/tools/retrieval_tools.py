"""Retrieval Agent Tools

A complete set of tools for a RAG Retrieval Agent with unified, working functions:
- retrieve_context_tool: Retrieve relevant documents with meta-tagging
- rerank_context_tool: Rerank by relevance with confidence scoring
- answer_question_tool: Generate answer with keyword matching and fallback
- traceability_tool: Full provenance tracking
"""
import json
import re
from typing import List, Dict, Any, Tuple
from langchain_core.tools import tool
from collections import Counter


# ============================================================================
# META-TAGGING & KEYWORD EXTRACTION UTILITIES
# ============================================================================

def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract key concepts/keywords from text."""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # Extract words (lowercase, alphanumeric + hyphens/underscores)
    words = re.findall(r'\b[a-z][a-z\-_]*\b', text.lower())
    
    # Filter stop words and short words
    meaningful = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Get most common
    if meaningful:
        counter = Counter(meaningful)
        return [word for word, _ in counter.most_common(top_n)]
    return []


def calculate_keyword_match_score(question_keywords: List[str], context_keywords: List[str]) -> float:
    """Calculate keyword overlap score (0-1)."""
    if not question_keywords or not context_keywords:
        return 0.0
    
    # Use Jaccard similarity
    overlap = len(set(question_keywords) & set(context_keywords))
    union = len(set(question_keywords) | set(context_keywords))
    
    return overlap / union if union > 0 else 0.0


def extract_meta_tags(text: str) -> List[str]:
    """
    Extract meta tags from text based on predefined categories.
    Meta tags help classify content for better matching.
    """
    meta_tags = []
    
    # Semantic domain detection
    financial_terms = {'revenue', 'profit', 'expense', 'cost', 'budget', 'invoice', 'payment', 'finance', 'accounting'}
    hr_terms = {'employee', 'hiring', 'recruitment', 'salary', 'benefits', 'leave', 'performance', 'appraisal'}
    technical_terms = {'system', 'database', 'server', 'software', 'api', 'integration', 'code', 'deployment'}
    product_terms = {'product', 'feature', 'release', 'version', 'update', 'specification', 'requirement'}
    process_terms = {'process', 'workflow', 'procedure', 'step', 'stage', 'phase', 'schedule', 'timeline'}
    
    text_lower = text.lower()
    
    if any(term in text_lower for term in financial_terms):
        meta_tags.append('financial')
    if any(term in text_lower for term in hr_terms):
        meta_tags.append('hr')
    if any(term in text_lower for term in technical_terms):
        meta_tags.append('technical')
    if any(term in text_lower for term in product_terms):
        meta_tags.append('product')
    if any(term in text_lower for term in process_terms):
        meta_tags.append('process')
    
    return meta_tags if meta_tags else ['general']


def calculate_meta_tag_confidence(question_meta_tags: List[str], context_meta_tags: List[str]) -> float:
    """
    Calculate confidence that context matches question based on meta tags.
    Returns 0-1 confidence score.
    """
    if not question_meta_tags or not context_meta_tags:
        return 0.5  # Medium confidence if no tags
    
    # Check for tag overlap
    overlap_tags = set(question_meta_tags) & set(context_meta_tags)
    
    if overlap_tags:
        # Strong match - both have overlapping tags
        return min(1.0, len(overlap_tags) / max(len(question_meta_tags), len(context_meta_tags)))
    
    # Check for 'general' tag - it matches anything
    if 'general' in context_meta_tags:
        return 0.7
    
    # No overlap - weak confidence
    return 0.3


# --- Unified retrieval tools ---

@tool
def retrieve_context_tool(question: str, llm_service, vectordb_service, rbac_namespace: str = "general", k: int = 5, where_clause: dict = None, user_intent_keywords: list = None) -> str:
    """Retrieve relevant context from vector database with RBAC filtering."""
    try:
        query_embedding = llm_service.generate_embedding(question)
        # Apply where_clause (RBAC filter) if provided
        results = vectordb_service.search(query_embedding, top_k=k, where_filter=where_clause)
        
        context_list = []
        if results.get('documents') and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                context_list.append({
                    "text": doc,
                    "score": float(results['distances'][0][i]) if results['distances'] else 0,
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else {}
                })
        
        return json.dumps({"success": True, "context": context_list})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@tool
def rerank_context_tool(context: str, llm_service) -> str:
    """Rerank retrieved context using improved relevance scoring."""
    try:
        context_data = json.loads(context) if isinstance(context, str) else context
        context_items = context_data.get('context', [])
        
        if not context_items:
            return json.dumps({"success": True, "reranked_context": []})
        
        # Improved reranking using multiple factors
        reranked_items = []
        for i, item in enumerate(context_items):
            text_content = item.get('text', '')
            doc_id = item.get('metadata', {}).get('doc_id', f'doc_{i}')
            original_distance = item.get('score', 0.5)  # ChromaDB distance (0=perfect match, higher=less similar)
            
            # Convert distance to similarity score (0-1, higher=more relevant)
            similarity_score = max(0.0, 1.0 - original_distance)
            
            # Additional scoring factors
            text_length = len(text_content)
            # Longer texts (up to reasonable limit) might be more informative
            length_score = min(1.0, text_length / 500.0)  # Normalize by ideal chunk size
            
            # Combine scores (weighted average)
            relevance_score = (
                0.7 * similarity_score +   # 70% weight on vector similarity
                0.3 * length_score         # 30% weight on content richness
            )
            
            # Ensure score is between 0 and 1
            relevance_score = max(0.0, min(1.0, relevance_score))
            
            # Create reranked item with calculated relevance score
            reranked_item = {
                "text": text_content,
                "metadata": {
                    **item.get('metadata', {}),
                    "doc_id": doc_id,
                    "relevance_score": round(relevance_score, 3),
                    "similarity_score": round(similarity_score, 3),
                    "original_distance": round(original_distance, 3),
                    "text_length": text_length
                }
            }
            reranked_items.append(reranked_item)
        
        # Sort by relevance score (highest first)
        reranked_items.sort(key=lambda x: x.get('metadata', {}).get('relevance_score', 0), reverse=True)
        
        return json.dumps({"success": True, "reranked_context": reranked_items})
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@tool
def answer_question_tool(question: str, context: str, llm_service, rbac_context: dict = None, rl_agent=None) -> str:
    """
    Generate answer based on context with multi-level confidence analysis.
    If keyword confidence is low, ask user for clarification instead of hallucinating.
    """
    try:
        context_data = json.loads(context) if isinstance(context, str) else context
        context_texts = context_data.get('reranked_context', [])
        
        # ========================================================================
        # MULTI-LEVEL CONFIDENCE ANALYSIS (Embedding + Keywords + Meta-Tags)
        # ========================================================================
        keyword_match_score = 0.0
        meta_tag_confidence = 0.5
        embedding_confidence = 0.5
        combined_confidence = 0.5
        meta_tags = []
        missing_keywords = []
        context_meta_tags = []
        
        if len(context_texts) > 0:
            # Level 1: EMBEDDING SIMILARITY (from ChromaDB reranking)
            # The reranked_context already has relevance_score from vector similarity
            embedding_scores = [c.get('metadata', {}).get('relevance_score', 0.5) for c in context_texts]
            embedding_confidence = sum(embedding_scores) / len(embedding_scores) if embedding_scores else 0.5
            
            # Level 2: KEYWORD MATCHING (exact word overlap)
            if rl_agent:
                context_only = [c['text'] for c in context_texts]
                keyword_match_score, meta_tags, missing_keywords = rl_agent.analyze_keyword_match(question, context_only)
                
                # Level 3: META-TAG MATCHING (semantic domain)
                combined_context = " ".join(context_only)
                context_meta_tags = rl_agent.extract_meta_tags(combined_context)
                question_meta_tags = rl_agent.extract_meta_tags(question)
                meta_tag_confidence = rl_agent.calculate_meta_tag_confidence(question_meta_tags, context_meta_tags)
                
            
            # Combined confidence: Weight all three factors
            # Embedding similarity is most important (if ChromaDB found it, it's relevant)
            combined_confidence = (
                0.5 * embedding_confidence +      # 50% - Embedding/vector similarity (most important)
                0.3 * meta_tag_confidence +       # 30% - Semantic domain matching
                0.2 * keyword_match_score        # 20% - Exact keyword overlap (least important, can be low for synonyms)
            )
        
        # ========================================================================
        # LOW CONFIDENCE FALLBACK: ASK USER FOR CLARIFICATION
        # ========================================================================
        # Only ask for clarification if BOTH embedding AND meta-tags are weak
        # (Don't reject just because keywords don't match - synonyms won't match)
        if embedding_confidence < 0.4 and meta_tag_confidence < 0.4:
            # Both vector similarity AND semantic domain are weak - likely irrelevant
            
            # Generate clarification request
            clarification_message = f"""I found some documents, but they don't seem to match your question well.

Your Question: "{question}"
Confidence Score: {combined_confidence:.1%}
Detected Topic: {', '.join(meta_tags) if meta_tags else 'general'}

The available documents are about: {', '.join(context_meta_tags) if context_meta_tags else 'general topics'}

Could you please:
1. Rephrase your question to be more specific, or
2. Provide more context about what you're looking for, or
3. Try one of these alternative searches:
   - {' | '.join(rl_agent._generate_alternative_queries(meta_tags) if rl_agent else ['general search'])}
"""
            
            return json.dumps({
                "success": True,
                "answer": clarification_message,
                "confidence": "LOW",
                "requires_clarification": True,
                "combined_confidence": combined_confidence,
                "embedding_confidence": embedding_confidence,
                "keyword_match_score": keyword_match_score,
                "meta_tag_confidence": meta_tag_confidence,
                "missing_keywords": missing_keywords,
                "suggested_topics": context_meta_tags
            })
        
        # Handle case where no context is available (could be RBAC-related)
        if not context_texts:
            # Check if this is likely a RBAC access denial
            if rbac_context and not rbac_context.get('is_root_user', False):
                # User is not root and has no results - likely RBAC denial
                user_id = rbac_context.get('user_id', 'Unknown')
                user_company = rbac_context.get('user_company_id', 'Unknown')
                user_dept = rbac_context.get('user_dept_id', 'Unknown')
                
                # Generate intelligent error message via LLM
                error_prompt = f"""You are a helpful assistant explaining access control restrictions.

A user (User ID: {user_id}) with access to Company {user_company}, Department {user_dept} asked:
"{question}"

However, no matching documents were found. This could be because:
1. The requested information is in a different company or department they don't have access to
2. The information doesn't exist in their authorized scope
3. They need additional permissions

Generate a brief, professional message explaining:
- They don't have access to the requested information
- Their current authorization level (Company {user_company}, Department {user_dept})
- Suggestion to contact their administrator for additional access

Keep the message concise and helpful."""
                
                error_answer = llm_service.generate_response(error_prompt)
                return json.dumps({
                    "success": True, 
                    "answer": error_answer,
                    "access_denied": True,
                    "reason": "RBAC access restriction",
                    "user_context": {
                        "user_id": user_id,
                        "authorized_company": user_company,
                        "authorized_department": user_dept
                    }
                })
            else:
                # Generic no context message
                return json.dumps({"success": True, "answer": "No context available to answer the question."})
        
        context_text = "\n\n".join([f"[Source: {c.get('metadata', {}).get('doc_id', 'N/A')}]\n{c['text']}" 
                                      for c in context_texts])
        
        # Check relevance scores - warn if low quality matches
        relevance_scores = [c.get('metadata', {}).get('relevance_score', 0) for c in context_texts]
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        
        # If average relevance is very low, add warning to prompt
        low_relevance_warning = ""
        if avg_relevance < 0.5:
            low_relevance_warning = "\n[QUALITY WARNING] Retrieved documents have low relevance scores (avg={:.3f}). Carefully verify context relevance before answering.".format(avg_relevance)
        
        # Build a meta-cognitive prompt with Chain-of-Thought reasoning
        prompt = f"""You are a helpful assistant that answers questions ONLY based on provided context.

=== META-COGNITIVE REASONING (Think step-by-step) ===
1. READ the context carefully
2. IDENTIFY: What is this context about? (main topics, subjects)
3. CHECK: Does the question ask about the same topics?
4. ANALYZE: Does the context contain specific information to answer the question?
5. DECIDE: Can I answer truthfully from this context, or should I say I don't have the information?

=== CRITICAL RULES ===
- NEVER make up or hallucinate information
- NEVER use external knowledge that's not in the context
- If the context doesn't match the question, MUST say: "I don't have information about [topic]. The available documents are about: [what they actually cover]"
- If context is unclear or low quality, admit it rather than guess{low_relevance_warning}

=== YOUR REASONING PROCESS ===
Before answering, show your thinking:
1. "Context is about: [identify main topics]"
2. "Question asks about: [question topic]"
3. "Match? [yes/no and why]"
4. "Can answer? [yes/no]"

Context:
{context_text}

Question: {question}

Let me think through this step by step:
1. Context is about: """
        
        answer = llm_service.generate_response(prompt)
        confidence_level = "HIGH" if combined_confidence > 0.7 else "MEDIUM" if combined_confidence > 0.5 else "LOW"
        return json.dumps({
            "success": True, 
            "answer": answer, 
            "relevance_score": avg_relevance,
            "combined_confidence": combined_confidence,
            "embedding_confidence": embedding_confidence,
            "keyword_match_score": keyword_match_score,
            "meta_tag_confidence": meta_tag_confidence,
            "confidence": confidence_level
        })
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@tool
def traceability_tool(question: str, context: str, vectordb_service) -> str:
    """Provide full traceability for the answer."""
    try:
        context_data = json.loads(context) if isinstance(context, str) else context
        context_texts = context_data.get('reranked_context', [])
        
        trace = {
            "question": question,
            "sources_used": len(context_texts),
            "documents": [
                {
                    "doc_id": c.get('metadata', {}).get('doc_id', 'N/A'),
                    "chunk_index": c.get('metadata', {}).get('chunk_index', 'N/A'),
                    "similarity_score": float(c.get('score', 0)),
                    "text_preview": (c['text'][:100] + "...") if len(c['text']) > 100 else c['text']
                }
                for c in context_texts
            ]
        }
        return json.dumps({"success": True, "traceability": trace})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


# --- Deprecated: Pre-Retrieval Tools (for reference) ---
    """
    Retrieves the list of authorized hierarchical RBAC tags (permissions) for a given user ID.
    This is the first step in retrieval to establish security context.

    Args:
        user_id (str): The unique identifier for the querying user.
        mapping_service: Service object providing a .resolve_user_roles(user_id) method
                         that accesses the centralized role mapping JSON/DB.

    Returns:
        JSON string with 'success' and 'authorized_rbac_tags' list 
        (e.g., ["1", "1-3", "1-3-1"]).
    """
    try:
        # Example: user_id -> mapping_service -> ["1", "1-3", "1-3-1"]
        authorized_tags = mapping_service.resolve_user_roles(user_id)

        if not authorized_tags:
            return json.dumps({"success": True, "authorized_rbac_tags": ["general_public"]})

        return json.dumps({"success": True, "authorized_rbac_tags": authorized_tags})

    except Exception as e:
        return json.dumps({"success": False, "error": f"Failed to resolve permissions: {str(e)}"})


@tool
def rewrite_query_tool(query: str, llm_service) -> str:
    """
    Generates multiple, diverse queries (including keyword queries and declarative statements) 
    from a single user question to improve search recall and handle ambiguity.

    Args:
        query (str): The original user question.

    Returns:
        JSON string with 'success' and a list of 'optimized_queries'.
    """
    try:
        prompt = f"""Generate 3-5 alternative search queries (including keyword lists and rephrased questions) 
        that would help find the most relevant context for the user's question.

        Original Query: "{query}"

        Return JSON array of strings."""
        
        # Assume llm_service.generate_json handles the structured output.
        result = llm_service.generate_json(prompt)
        
        # Ensure result is a list of strings
        optimized_queries = result if isinstance(result, list) else json.loads(result)
        optimized_queries = [q for q in optimized_queries if isinstance(q, str)]

        # Always include the original query
        if query not in optimized_queries:
            optimized_queries.append(query)
            
        return json.dumps({"success": True, "optimized_queries": list(set(optimized_queries))})

    except Exception as e:
        # Fallback to just the original query
        return json.dumps({"success": True, "optimized_queries": [query]})


# --- 2. Core Retrieval Tool (with Hierarchical RBAC) ---

@tool
def retrieve_relevant_context_tool(queries: List[str], authorized_rbac_tags: List[str], llm_service, vectordb_service, k_per_query: int = 3) -> str:
    """
    Executes multiple vector searches across authorized RBAC collections (hierarchical VDB structure).
    
    Args:
        queries (List[str]): List of optimized search queries (from rewrite_query_tool).
        authorized_rbac_tags (List[str]): List of hierarchical access tags (e.g., ["1", "1-3", "1-3-1"]).
        llm_service: The language model service for generating embeddings.
        vectordb_service: The vector database service for querying.
        k_per_query (int): Number of top chunks to retrieve per query, per authorized tag.

    Returns:
        JSON string containing 'success', a list of unique 'raw_results' (chunks), 
        and the total number of chunks retrieved.
    """
    all_results = []
    
    # Set to track unique chunk_id + tag combinations to avoid duplicates
    seen_chunks = set() 
    
    try:
        # NOTE: vectordb_service and llm_service are assumed to be available in the context.
        # This function iterates through all authorized access levels (collections) and all queries.
        
        for rbac_tag in authorized_rbac_tags:
            # VDB Filtering: Use the RBAC tag to identify the specific VDB collection/index.
            # This relies on the ingestion process (Strategy 1) having created multiple 
            # collections based on the hierarchical tags.
            collection_name = f"rbac_{rbac_tag.replace('*', 'all')}" 
            
            try:
                collection = vectordb_service.get_collection(collection_name)
                
                for query in queries:
                    # Generate embedding for the query
                    query_embedding = llm_service.generate_embedding(query)
                    
                    # Perform search within the authorized collection
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=k_per_query,
                    )
                    
                    # Process Results
                    if results and results.get('documents'):
                        for i in range(len(results['documents'][0])):
                            metadata = results['metadatas'][0][i]
                            
                            chunk_id = metadata.get('doc_id', 'unknown') + metadata.get('chunk_index', '0')
                            unique_key = (chunk_id, rbac_tag)
                            
                            if unique_key not in seen_chunks:
                                seen_chunks.add(unique_key)
                                all_results.append({
                                    "chunk_text": results['documents'][0][i],
                                    "doc_id": metadata.get('doc_id', 'N/A'),
                                    "rbac_source_tag": rbac_tag,
                                    "score": results['distances'][0][i]
                                })
                                
            except Exception as e:
                # Log warning for collection access failure but continue
                print(f"Warning: Collection {collection_name} failed or is empty: {e}")


        return json.dumps({
            "success": True,
            "total_raw_results": len(all_results),
            "raw_results": all_results 
        })

    except Exception as e:
        return json.dumps({"success": False, "error": f"Retrieval failed: {str(e)}"})


# --- 3. Post-Retrieval Tools ---

@tool
def rerank_results_tool(query: str, raw_results: List[Dict[str, Any]], k_final: int = 5) -> str:
    """
    Reranks the raw retrieved chunks using a powerful cross-encoder model 
    to filter noise and select the most relevant chunks for synthesis.
    (Note: reranker_service is not implemented in this direct import version.)
    """
    if not raw_results:
        return json.dumps({"success": True, "final_context": []})

    try:
        # Placeholder: just return top k by score
        fallback_context = sorted(raw_results, key=lambda x: x['score'], reverse=True)[:k_final]
        return json.dumps({"success": True, "final_context": [
            {"text": item['chunk_text'], "source": f"Doc ID: {item['doc_id']} (RBAC: {item.get('rbac_source_tag', '')})"}
            for item in fallback_context
        ]})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Reranking failed: {str(e)}"})

@tool
def synthesize_answer_tool(query: str, context: List[Dict[str, str]], llm_service) -> str:
    """
    Generates the final, grounded answer using the retrieved context.
    """
    if not context:
        return json.dumps({"success": False, "answer": "I do not have enough specific context to answer that question.", "sources": []})

    try:
        context_text = "\n\n--- Source Document ---\n\n".join([c['text'] for c in context])
        sources = [c['source'] for c in context]
        system_prompt = (
            "You are a professional corporate knowledge assistant. Your task is to provide a "
            "clear, concise, and accurate answer based ONLY on the CONTEXT provided below. "
            "If the context does not contain the answer, state explicitly that you cannot answer."
        )
        user_prompt = f"""
        CONTEXT:
        {context_text}
        
        QUESTION: 
        {query}
        """
        # Use llm_service directly
        final_answer = llm_service.generate_response(
            prompt=user_prompt
        )
        return json.dumps({
            "success": True,
            "answer": final_answer,
            "sources": list(set(sources))
        })
    except Exception as e:
        return json.dumps({"success": False, "error": f"Answer synthesis failed: {str(e)}"})
