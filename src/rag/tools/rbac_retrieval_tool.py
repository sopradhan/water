"""
RBAC-Aware Retrieval with Tag-Based Filtering

Implements secure, role-based document retrieval with three response modes:
- CONCISE: Brief, to-the-point answers
- VERBOSE: Detailed, comprehensive answers with sources
- INTERNAL: Internal documentation retrieval (requires special role)

RBAC Implementation:
- User has department and role
- Documents tagged with rbac:dept:{dept}:role:{role}
- Admin/Root/SuperUser can access all documents
- Meta tags enable pinpoint retrieval accuracy
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain_core.tools import tool
from enum import Enum


class ResponseMode(Enum):
    """Response generation modes."""
    CONCISE = "concise"
    VERBOSE = "verbose"
    INTERNAL = "internal"


class UserRole:
    """Represents user with department and role."""
    
    def __init__(self, user_id: str, username: str, department: str, role: str, is_admin: bool = False):
        """
        Initialize user role.
        
        Args:
            user_id: Unique user identifier
            username: Display name
            department: Department name
            role: Role name
            is_admin: Whether user is admin/root/superuser
        """
        self.user_id = user_id
        self.username = username
        self.department = department
        self.role = role
        self.is_admin = is_admin
    
    def get_accessible_rbac_tags(self) -> List[str]:
        """Get all RBAC tags user can access."""
        if self.is_admin:
            return ["rbac:admin:all"]  # Admin can access everything
        
        return [f"rbac:dept:{self.department}:role:{self.role}"]
    
    def can_access_document(self, doc_rbac_tags: List[str]) -> bool:
        """Check if user can access document based on RBAC tags."""
        if self.is_admin:
            return True
        
        accessible_tags = self.get_accessible_rbac_tags()
        return any(tag in doc_rbac_tags for tag in accessible_tags)


class RBACRetrieval:
    """Secure retrieval with RBAC and tag-based filtering."""
    
    def __init__(self, vectordb_service, llm_service):
        """
        Initialize RBAC retrieval.
        
        Args:
            vectordb_service: VectorDB service (ChromaDB)
            llm_service: LLM service for response generation
        """
        self.vectordb = vectordb_service
        self.llm = llm_service
    
    def retrieve_with_rbac_filtering(
        self,
        question: str,
        user: UserRole,
        k: int = 5,
        sensitivity_filters: List[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents with RBAC filtering.
        
        Args:
            question: User query
            user: UserRole object with access rights
            k: Number of results to retrieve
            sensitivity_filters: Optional sensitivity levels to filter (e.g., ["public", "internal"])
        
        Returns:
            Dict with filtered results
        """
        try:
            # Generate query embedding
            query_embedding = self.llm.generate_embedding(question)
            
            # Raw retrieval from vectordb (without filtering)
            raw_results = self.vectordb.search(query_embedding, top_k=k*2)  # Get more to account for filtering
            
            # Filter results by RBAC and sensitivity
            filtered_results = self._filter_by_rbac_and_sensitivity(
                raw_results,
                user,
                sensitivity_filters
            )
            
            # Truncate to k results
            filtered_results = self._truncate_results(filtered_results, k)
            
            return {
                "success": True,
                "user": {
                    "user_id": user.user_id,
                    "username": user.username,
                    "department": user.department,
                    "role": user.role,
                    "is_admin": user.is_admin
                },
                "query": question,
                "results": filtered_results,
                "count": len(filtered_results),
                "filtered": True,
                "access_enforcement": "RBAC" if not user.is_admin else "ADMIN_OVERRIDE",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "user": {"user_id": user.user_id}
            }
    
    def _filter_by_rbac_and_sensitivity(
        self,
        results: Dict[str, Any],
        user: UserRole,
        sensitivity_filters: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter retrieved results by RBAC tags and sensitivity.
        
        Args:
            results: Raw results from vectordb
            user: User with access rights
            sensitivity_filters: Optional sensitivity level filter
        
        Returns:
            Filtered list of results
        """
        filtered = []
        
        if not results.get('documents') or not results['documents'][0]:
            return filtered
        
        documents = results['documents'][0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        
        for doc_text, metadata, distance in zip(documents, metadatas, distances):
            # Extract RBAC tags from metadata
            rbac_tags = metadata.get('rbac_tags', []) if isinstance(metadata, dict) else []
            
            # Check RBAC access
            if not user.can_access_document(rbac_tags):
                continue  # User cannot access this document
            
            # Check sensitivity level if filter provided
            if sensitivity_filters:
                doc_sensitivity = None
                if isinstance(metadata, dict):
                    # Try to extract from meta tags
                    meta_tags = metadata.get('meta_tags', [])
                    for tag in meta_tags:
                        if tag.startswith("meta:sensitivity:"):
                            doc_sensitivity = tag.replace("meta:sensitivity:", "")
                            break
                
                if doc_sensitivity and doc_sensitivity not in sensitivity_filters:
                    continue  # Document sensitivity not in allowed list
            
            # Add to filtered results
            filtered.append({
                "text": doc_text,
                "metadata": metadata,
                "relevance_score": max(0.0, 1.0 - distance),
                "access_granted": True
            })
        
        return filtered
    
    def _truncate_results(self, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Truncate results to k items."""
        return results[:k]


@tool
def retrieve_with_rbac_tool(
    question: str,
    user_id: str,
    username: str,
    department: str,
    role: str,
    is_admin: bool = False,
    response_mode: str = "concise",
    vectordb_service=None,
    llm_service=None,
    k: int = 5
) -> str:
    """
    Retrieve documents with RBAC enforcement.
    
    Args:
        question: User query
        user_id: User identifier
        username: Display name
        department: User's department
        role: User's role
        is_admin: Whether user is admin
        response_mode: "concise", "verbose", or "internal"
        vectordb_service: VectorDB service
        llm_service: LLM service
        k: Number of results
    
    Returns:
        JSON string with results
    """
    try:
        # Create user role object
        user = UserRole(user_id, username, department, role, is_admin)
        
        # Initialize RBAC retrieval
        retrieval = RBACRetrieval(vectordb_service, llm_service)
        
        # Determine sensitivity filters based on response mode
        sensitivity_filters = None
        if response_mode == "internal":
            # Internal mode only allows internal and above
            if not user.is_admin and role != "admin":
                return json.dumps({
                    "success": False,
                    "error": "Internal mode requires admin or special role",
                    "access_denied": True
                })
            sensitivity_filters = ["internal", "confidential", "secret"]
        elif response_mode == "verbose":
            # Verbose allows public and internal
            sensitivity_filters = ["public", "internal"]
        else:  # concise
            # Concise allows only public
            sensitivity_filters = ["public"]
        
        # Retrieve with RBAC filtering
        results = retrieval.retrieve_with_rbac_filtering(
            question,
            user,
            k,
            sensitivity_filters
        )
        
        # Add response mode to results
        results["response_mode"] = response_mode
        
        return json.dumps(results)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@tool
def generate_response_with_mode_tool(
    question: str,
    context: str,
    response_mode: str = "concise",
    llm_service=None
) -> str:
    """
    Generate response in specified mode: concise, verbose, or internal.
    
    CONCISE: Brief answer (1-2 sentences), direct to the point
    VERBOSE: Detailed answer (1-2 paragraphs) with context and sources
    INTERNAL: Internal documentation format with detailed reasoning
    
    Args:
        question: User question
        context: Retrieved context/documents
        response_mode: Response mode
        llm_service: LLM service
    
    Returns:
        JSON string with generated response
    """
    try:
        mode = ResponseMode[response_mode.upper()] if response_mode.upper() in ResponseMode.__members__ else ResponseMode.CONCISE
        
        # Build prompts based on mode
        if mode == ResponseMode.CONCISE:
            system_prompt = """You are a helpful assistant providing concise answers.
Keep responses brief (1-2 sentences maximum). Be direct and to the point.
Focus on the most important information."""
            
        elif mode == ResponseMode.VERBOSE:
            system_prompt = """You are a helpful assistant providing comprehensive answers.
Provide detailed explanations (1-2 paragraphs). Include relevant context and sources.
Structure the answer clearly with key points."""
            
        else:  # INTERNAL
            system_prompt = """You are an internal documentation specialist.
Provide internal-level documentation with detailed reasoning and explanations.
Include implementation details, considerations, and any caveats.
Format with clear sections and structured information."""
        
        # Generate response
        response = llm_service.generate_response(
            f"""{system_prompt}

Question: {question}

Context/Documents:
{context}

Provide the response:"""
        )
        
        return json.dumps({
            "success": True,
            "response": response,
            "mode": response_mode,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example user roles
    finance_user = UserRole("user123", "John Doe", "Finance", "analyst", is_admin=False)
    admin_user = UserRole("admin456", "Admin", "IT", "admin", is_admin=True)
    
    # Example RBAC tags
    finance_doc_tags = ["rbac:dept:Finance:role:analyst", "meta:intent:report"]
    hr_doc_tags = ["rbac:dept:HR:role:manager", "meta:intent:policy"]
    
    print("RBAC Access Examples:")
    print(f"Finance analyst accessing Finance doc: {finance_user.can_access_document(finance_doc_tags)}")  # True
    print(f"Finance analyst accessing HR doc: {finance_user.can_access_document(hr_doc_tags)}")  # False
    print(f"Admin accessing Finance doc: {admin_user.can_access_document(finance_doc_tags)}")  # True
    print(f"Admin accessing HR doc: {admin_user.can_access_document(hr_doc_tags)}")  # True
