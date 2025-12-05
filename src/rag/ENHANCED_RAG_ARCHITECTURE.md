"""
Enhanced RAG System Architecture - Integration Guide

This document outlines the comprehensive, scalable, and generic RAG system with:
1. RBAC-based access control with automatic department/role classification
2. Universal markdown conversion for all document types
3. Tag-based retrieval (RBAC + Meta tags)
4. Three response modes (Concise, Verbose, Internal)
5. Comprehensive guardrails validation

Architecture Flow:
```
INPUT DOCUMENT
    ↓
[Classification Agent] ← Database (Departments/Roles)
    ↓ (Outputs: Intent, Dept, Role, Keywords)
[Markdown Converter] ← Universal format conversion
    ↓
[Tag Generator]
    ├─ RBAC Tags: rbac:dept:{dept}:role:{role}
    └─ Meta Tags: meta:intent:{intent}, meta:sensitivity:{level}, meta:keyword:{kw}
    ↓
[Vector DB] (ChromaDB with tags)
    ├─ Documents
    ├─ Embeddings
    └─ Metadata (RBAC + Meta tags)

QUERY FROM USER
    ↓
[User Context] (dept, role, user_id)
    ↓
[RBAC Retrieval]
    ├─ Check: user.can_access(doc_rbac_tags)?
    └─ Filter: sensitivity_level per response_mode
    ↓
[Response Generation] (Concise/Verbose/Internal)
    ↓
[Guardrails Validation]
    ├─ Hallucination check
    ├─ Security check (PII, credentials)
    ├─ Factual accuracy
    ├─ Tone check
    └─ Completeness check
    ↓
OUTPUT TO USER
```
"""

# ============================================================================
# INTEGRATION COMPONENTS
# ============================================================================

"""
1. DOCUMENT CLASSIFICATION & TAGGING
   Location: src/rag/tools/document_classification_tool.py
   
   Components:
   - DocumentClassificationAgent: Meta-prompting based classification
   - Uses database to fetch available departments/roles
   - Outputs: intent, primary_dept, required_roles, sensitivity_level, keywords
   - enhance_document_metadata_tool: Tool wrapper for integration
   
   Integration:
   ```python
   from src.rag.tools.document_classification_tool import enhance_document_metadata_tool
   
   metadata = enhance_document_metadata_tool.invoke({
       "doc_id": "doc_123",
       "title": "Budget Review Process",
       "text": document_content,
       "llm_service": llm_service,
       "db_conn": db_connection
   })
   ```

2. MARKDOWN CONVERSION
   Location: src/rag/tools/document_markdown_converter.py
   
   Supported Formats:
   - Text (TXT): Plain text to markdown
   - Tabular (CSV, XLSX): Tables to markdown with statistics
   - Documents (PDF, DOCX): Full document structure with sections
   - Database Tables: Structured data to markdown
   
   Universal Interface:
   ```python
   from src.rag.tools.document_markdown_converter import convert_to_markdown_tool
   
   markdown = convert_to_markdown_tool.invoke({
       "content": data_or_path,
       "source_type": "pdf|csv|xlsx|docx|txt|database_table",
       "title": "Document Title",
       "source_path": "path/to/file.pdf"  # For file-based
   })
   ```

3. RBAC-AWARE RETRIEVAL
   Location: src/rag/tools/rbac_retrieval_tool.py
   
   Components:
   - UserRole: Represents user with department/role
   - RBACRetrieval: Enforces access control during retrieval
   - retrieve_with_rbac_tool: Tool for retrieval with RBAC
   - generate_response_with_mode_tool: Response generation with modes
   
   Integration:
   ```python
   from src.rag.tools.rbac_retrieval_tool import retrieve_with_rbac_tool, UserRole
   
   # Create user context
   user = UserRole(
       user_id="user_123",
       username="John Doe",
       department="Finance",
       role="analyst",
       is_admin=False
   )
   
   # Retrieve with RBAC
   results = retrieve_with_rbac_tool.invoke({
       "question": "What is the budget for Q4?",
       "user_id": user.user_id,
       "username": user.username,
       "department": user.department,
       "role": user.role,
       "is_admin": user.is_admin,
       "response_mode": "concise",  # or "verbose", "internal"
       "vectordb_service": vectordb_service,
       "llm_service": llm_service
   })
   ```

4. GUARDRAILS VALIDATION
   Location: src/rag/guardrails/guardrails_service.py
   
   Validation Types:
   - Hallucination: Is response grounded in context?
   - Security: Does response expose PII/credentials?
   - Accuracy: Are claims factually correct?
   - Tone: Is response appropriate?
   - Completeness: Does response answer the question?
   
   Integration:
   ```python
   from src.rag.guardrails.guardrails_service import GuardrailsService
   
   guardrails = GuardrailsService(llm_service)
   validation = guardrails.validate_response(
       response=generated_response,
       context=retrieved_context,
       question=user_question,
       check_types=["hallucination", "security", "accuracy", "tone", "completeness"]
   )
   
   if validation["is_safe"]:
       return generated_response
   else:
       # Handle based on max_risk_level
       if validation["max_risk_level"] == "critical":
           # Block response
           return error_response
   ```
"""

# ============================================================================
# INGESTION PIPELINE (Enhanced)
# ============================================================================

"""
UPDATED INGESTION PIPELINE:

1. Document Input
   ↓
2. [CLASSIFICATION] Extract intent, department, role requirements
   └─ Uses meta-prompting against database departments/roles
   ↓
3. [MARKDOWN CONVERSION] Convert to markdown format
   └─ Universal support for all document types
   └─ Preserves structure, headers, tables
   ↓
4. [METADATA ENRICHMENT]
   └─ Generate RBAC tags: rbac:dept:{dept}:role:{role}
   └─ Generate Meta tags: meta:intent:{intent}, meta:sensitivity:{level}
   └─ Add keywords as meta tags
   ↓
5. [CHUNKING] Split markdown into semantic chunks
   ├─ Preserve RBAC/Meta tags per chunk
   └─ Maintain hierarchy and context
   ↓
6. [EMBEDDING & STORAGE]
   └─ Generate embeddings for chunks
   └─ Store in ChromaDB with tags in metadata
   └─ Metadata structure:
       {
           "rbac_tags": ["rbac:dept:Finance:role:analyst"],
           "meta_tags": ["meta:intent:report", "meta:sensitivity:internal"],
           "doc_id": "doc_123",
           "title": "Quarterly Report",
           "source": "pdf",
           "classification": {...}
       }
   ↓
7. [DATABASE TRACKING]
   └─ Store metadata in SQLite
   └─ Track classification confidence
   └─ Enable auditing and analytics
"""

# ============================================================================
# RETRIEVAL PIPELINE (Enhanced)
# ============================================================================

"""
UPDATED RETRIEVAL PIPELINE:

1. User Query
   ├─ Question text
   ├─ User context: department, role, is_admin
   └─ Response mode: concise|verbose|internal
   ↓
2. [RBAC ENFORCEMENT]
   ├─ Check: user.can_access(doc_rbac_tags)?
   ├─ Filter: sensitivity_level per response_mode
   │  ├─ CONCISE: Only "public" documents
   │  ├─ VERBOSE: "public" + "internal"
   │  └─ INTERNAL: "internal" + "confidential" (role required)
   └─ Admin/Root: Bypass all filters
   ↓
3. [SEMANTIC RETRIEVAL]
   ├─ Generate query embedding
   └─ Search ChromaDB (filtered by RBAC tags)
   ↓
4. [RERANKING]
   ├─ Use meta tags for semantic matching
   └─ Consider: intent match, keyword match, relevance
   ↓
5. [RESPONSE GENERATION]
   ├─ Generate response in selected mode
   ├─ CONCISE: 1-2 sentences, direct answer
   ├─ VERBOSE: Detailed with context and sources
   └─ INTERNAL: Implementation details, caveats
   ↓
6. [GUARDRAILS VALIDATION]
   ├─ Check: Hallucination
   ├─ Check: Security (PII/credentials)
   ├─ Check: Factual accuracy
   ├─ Check: Tone
   └─ Check: Completeness
   ↓
7. [OUTPUT]
   ├─ If safe: Return response + metadata
   └─ If unsafe: Block or flag based on risk level
"""

# ============================================================================
# ENHANCED METADATA STORAGE
# ============================================================================

"""
METADATA TABLE ENHANCEMENTS:

document_metadata table (enhanced):
├─ doc_id (PK)
├─ title
├─ author
├─ source
├─ summary
├─ rbac_namespace (deprecated, replaced by tags)
├─ classification_intent (from classifier)
├─ classification_primary_dept (from classifier)
├─ classification_required_roles (JSON list)
├─ classification_sensitivity (public|internal|confidential|secret)
├─ classification_confidence (0-1)
├─ classification_reasoning
├─ rbac_tags (JSON: ["rbac:dept:Finance:role:analyst"])
├─ meta_tags (JSON: ["meta:intent:report", "meta:keyword:budget"])
├─ keywords (JSON: ["quarterly", "budget", "finance"])
└─ markdown_format (bool - document converted to markdown)

chunk_embedding_data table (enhanced):
├─ chunk_id (PK)
├─ doc_id (FK)
├─ chunk_text (markdown formatted)
├─ embedding_model
├─ embedding_version
├─ quality_score
├─ reindex_count
├─ healing_suggestions
├─ rbac_tags (inherited from document)
├─ meta_tags (inherited + chunk-specific)
├─ intent_match_score (0-1)
└─ keyword_match_score (0-1)

rag_history_and_optimization (enhanced):
├─ history_id (PK)
├─ event_type (QUERY|HEAL|SYNTHETIC_TEST|GUARDRAIL_VIOLATION)
├─ timestamp
├─ query_text
├─ target_doc_id
├─ target_chunk_id
├─ user_id
├─ user_department
├─ user_role
├─ response_mode (concise|verbose|internal)
├─ metrics_json (includes: accuracy, latency, token_cost, etc.)
├─ guardrail_checks (JSON: {hallucination: safe, security: safe, ...})
├─ guardrail_violations (JSON: [...violations found...])
├─ guardrail_risk_level (safe|low|medium|high|critical)
├─ reward_signal
├─ action_taken
└─ session_id
"""

# ============================================================================
# RESPONSE MODES DETAILED
# ============================================================================

"""
CONCISE MODE
Purpose: Brief, to-the-point answers for quick decisions
Format: 1-2 sentences maximum
Sensitivity Filter: public only
Use Cases: Quick lookups, high-level summaries
Example:
Q: "What is our Q4 budget?"
A: "The Q4 budget is $2.5M with $1M allocated to operations and $1.5M to marketing."

VERBOSE MODE
Purpose: Detailed answers with context and sources
Format: 1-2 paragraphs with structure
Sensitivity Filter: public + internal
Use Cases: Research, understanding context, decision making
Example:
Q: "What is our Q4 budget?"
A: "The Q4 budget is $2.5M. This includes:
- Operations: $1M (40%)
- Marketing: $1.5M (60%)

This represents a 15% increase from Q3 due to expansion plans outlined
in the strategic plan. Additional details in the Budget Review Process document."

INTERNAL MODE
Purpose: Internal documentation with detailed reasoning
Format: Structured with sections, implementation details, caveats
Sensitivity Filter: internal + confidential + secret (role required)
Use Cases: Internal team collaboration, implementation guidance, strategic planning
Requirements: Must have admin or internal-designated role
Example:
Q: "What is our Q4 budget?"
A: "## Q4 Budget Overview
**Total**: $2.5M
**Allocation Strategy**: Based on strategic expansion initiative

### Breakdown
- Operations: $1M (cost reduction through automation)
- Marketing: $1.5M (new market penetration)

### Assumptions
- Headcount: 20 new hires (ops)
- Market growth: 25% YoY

### Risk Factors
- Supply chain disruptions could impact ops
- Market competition may require additional marketing spend

### Implementation Timeline
Q4 Week 1-2: Budget reviews
Q4 Week 3: Final approvals
Q4 Week 4: Implementation begins"
"""

# ============================================================================
# USAGE EXAMPLE - COMPLETE FLOW
# ============================================================================

"""
COMPLETE WORKFLOW EXAMPLE:

# 1. INGESTION (One-time per document)

from src.rag.tools.document_classification_tool import enhance_document_metadata_tool
from src.rag.tools.document_markdown_converter import convert_to_markdown_tool
from src.rag.tools.ingestion_tools import save_to_vectordb_tool

# Read document
with open("budget_report.pdf", "rb") as f:
    pdf_content = f.read()

# Step 1: Convert to markdown
markdown_result = convert_to_markdown_tool.invoke({
    "content": None,
    "source_type": "pdf",
    "title": "Q4 Budget Report",
    "source_path": "budget_report.pdf"
})
markdown_text = json.loads(markdown_result)["markdown"]

# Step 2: Classify and generate tags
metadata_result = enhance_document_metadata_tool.invoke({
    "doc_id": "budget_2025_q4",
    "title": "Q4 Budget Report",
    "text": markdown_text[:2000],  # Use excerpt for classification
    "llm_service": llm_service,
    "db_conn": db_connection
})
classification = json.loads(metadata_result)["metadata"]

# Step 3: Store in vector DB with tags
save_result = save_to_vectordb_tool.invoke({
    "chunks": markdown_text,  # Now in markdown
    "doc_id": "budget_2025_q4",
    "llm_service": llm_service,
    "vectordb_service": vectordb_service,
    "metadata": json.dumps({
        "rbac_tags": classification["tags"]["rbac"],
        "meta_tags": classification["tags"]["meta"],
        "title": classification["title"],
        "classification": classification["classification"]
    })
})

# 2. RETRIEVAL (Per user query)

from src.rag.tools.rbac_retrieval_tool import retrieve_with_rbac_tool, UserRole
from src.rag.guardrails.guardrails_service import GuardrailsService

# Create user context
user = UserRole(
    user_id="emp_456",
    username="Alice Smith",
    department="Finance",
    role="analyst",
    is_admin=False
)

# Retrieve with RBAC
retrieval_result = retrieve_with_rbac_tool.invoke({
    "question": "What is the Q4 budget?",
    "user_id": user.user_id,
    "username": user.username,
    "department": user.department,
    "role": user.role,
    "is_admin": user.is_admin,
    "response_mode": "verbose",
    "vectordb_service": vectordb_service,
    "llm_service": llm_service,
    "k": 5
})

# Parse results
retrieval_data = json.loads(retrieval_result)
if retrieval_data["success"]:
    context = "\\n\\n".join([r["text"] for r in retrieval_data["results"]])
    
    # Generate response in verbose mode
    from src.rag.tools.rbac_retrieval_tool import generate_response_with_mode_tool
    
    response_result = generate_response_with_mode_tool.invoke({
        "question": "What is the Q4 budget?",
        "context": context,
        "response_mode": "verbose",
        "llm_service": llm_service
    })
    
    response_data = json.loads(response_result)
    generated_response = response_data["response"]
    
    # Validate with guardrails
    guardrails = GuardrailsService(llm_service)
    validation = guardrails.validate_response(
        response=generated_response,
        context=context,
        question="What is the Q4 budget?"
    )
    
    if validation["is_safe"]:
        print(f"✓ Response: {generated_response}")
    else:
        print(f"✗ Response blocked: {validation['recommendation']}")
        print(f"  Risk level: {validation['max_risk_level']}")
else:
    print(f"✗ Retrieval failed: {retrieval_data['error']}")
"""

print(__doc__)
