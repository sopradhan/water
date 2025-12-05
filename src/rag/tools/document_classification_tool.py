"""
Enhanced Metadata-Driven Document Classification & RBAC Tagging System

Purpose: Classify documents by intent, department, and role during ingestion
Uses meta-prompting with agentic classification to understand document context
and map to available departments/roles for RBAC-based access control.

Features:
- Multi-level classification (intent, department, role, sensitivity)
- Generic department/role mapping from database
- RBAC tag generation for access control
- Meta tag generation for retrieval optimization
- Markdown-optimized metadata structure
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.tools import tool
from datetime import datetime


# ============================================================================
# METADATA CLASSIFICATION AGENT (Meta-Prompting Approach)
# ============================================================================

class DocumentClassificationAgent:
    """
    Agentic classifier that uses meta-prompting to understand document intent
    and classify it against available departments/roles from database.
    """
    
    def __init__(self, llm_service, db_conn=None):
        """
        Initialize classifier with LLM service and optional database connection.
        
        Args:
            llm_service: LLMService instance for embeddings and classification
            db_conn: Database connection to fetch available departments/roles
        """
        self.llm_service = llm_service
        self.db_conn = db_conn
    
    def get_available_departments_roles(self) -> Dict[str, List[str]]:
        """
        Fetch available departments and their roles from database.
        Creates dynamic context for classification.
        
        Returns:
            Dict: {dept_id: {"name": "...", "roles": [...]}, ...}
        """
        if not self.db_conn:
            return {
                "default": {"name": "General", "roles": ["viewer", "editor", "admin"]}
            }
        
        try:
            cursor = self.db_conn.cursor()
            
            # Get departments
            cursor.execute("SELECT id, name FROM departments")
            depts = {str(row[0]): {"name": row[1], "roles": []} for row in cursor.fetchall()}
            
            # Get roles per department (via company_department_role mapping)
            cursor.execute("""
                SELECT DISTINCT cdr.department_id, r.name
                FROM company_department_role cdr
                JOIN roles r ON cdr.role_id = r.id
                ORDER BY cdr.department_id, r.name
            """)
            
            for dept_id, role_name in cursor.fetchall():
                if str(dept_id) in depts:
                    depts[str(dept_id)]["roles"].append(role_name)
            
            return depts
        except Exception as e:
            print(f"Error fetching departments/roles: {e}")
            return {"default": {"name": "General", "roles": ["viewer", "editor", "admin"]}}
    
    def classify_document_intent(self, document_text: str, doc_title: str = "") -> Dict[str, Any]:
        """
        Use meta-prompting to classify document intent, department, and role.
        
        Classification outputs:
        - intent: Primary purpose of document
        - departments: List of relevant departments
        - roles: Required roles to access
        - sensitivity: Data sensitivity level (public, internal, confidential, secret)
        - keywords: Key topics for meta tagging
        """
        try:
            departments = self.get_available_departments_roles()
            dept_names = ", ".join([d["name"] for d in departments.values()])
            roles_list = set()
            for d in departments.values():
                roles_list.update(d["roles"])
            roles_str = ", ".join(sorted(roles_list))
            
            # Meta-prompting for document classification
            classification_prompt = f"""
You are an intelligent document classification agent. Analyze the following document
and classify it based on intent, department, role requirements, and sensitivity.

AVAILABLE DEPARTMENTS: {dept_names}
AVAILABLE ROLES: {roles_str}

DOCUMENT TITLE: {doc_title}
DOCUMENT EXCERPT (first 1000 chars): {document_text[:1000]}

Respond with ONLY valid JSON (no markdown, no extra text):
{{
    "intent": "<primary purpose: e.g., procedure, policy, guide, faq, report, etc.>",
    "primary_department": "<best matching department from available>",
    "relevant_departments": ["<dept1>", "<dept2>"],
    "required_roles": ["<role1>", "<role2>"],
    "sensitivity_level": "<public|internal|confidential|secret>",
    "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],
    "confidence_score": 0.85,
    "reasoning": "<brief explanation of classification>"
}}
"""
            
            response = self.llm_service.generate_response(classification_prompt)
            
            # Parse JSON response
            classification = json.loads(response)
            
            return {
                "success": True,
                "classification": classification,
                "timestamp": datetime.now().isoformat()
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse classification response: {e}",
                "fallback_classification": {
                    "intent": "document",
                    "primary_department": "General",
                    "relevant_departments": ["General"],
                    "required_roles": ["viewer"],
                    "sensitivity_level": "internal",
                    "keywords": [],
                    "confidence_score": 0.0
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fallback_classification": {
                    "intent": "document",
                    "primary_department": "General",
                    "relevant_departments": ["General"],
                    "required_roles": ["viewer"],
                    "sensitivity_level": "internal",
                    "keywords": [],
                    "confidence_score": 0.0
                }
            }
    
    def generate_rbac_tags(self, classification: Dict[str, Any]) -> List[str]:
        """
        Generate RBAC tags for access control.
        
        Format: "dept:{department_id}:role:{role}"
        Allows filtering in vector DB by department and role.
        
        Args:
            classification: Output from classify_document_intent
        
        Returns:
            List of RBAC tags
        """
        rbac_tags = []
        
        primary_dept = classification.get("primary_department", "General")
        roles = classification.get("required_roles", ["viewer"])
        
        # Create RBAC tags for each role
        for role in roles:
            rbac_tag = f"rbac:dept:{primary_dept}:role:{role}"
            rbac_tags.append(rbac_tag)
        
        # Add admin override tag (admin can always access)
        rbac_tags.append("rbac:admin:all")
        
        return rbac_tags
    
    def generate_meta_tags(self, classification: Dict[str, Any]) -> List[str]:
        """
        Generate meta tags for retrieval optimization and pinpoint accuracy.
        
        Format: "meta:{category}:{value}"
        Used for semantic filtering and accurate retrieval.
        
        Args:
            classification: Output from classify_document_intent
        
        Returns:
            List of meta tags
        """
        meta_tags = []
        
        # Intent tag
        intent = classification.get("intent", "document")
        meta_tags.append(f"meta:intent:{intent}")
        
        # Sensitivity tag
        sensitivity = classification.get("sensitivity_level", "internal")
        meta_tags.append(f"meta:sensitivity:{sensitivity}")
        
        # Department tags
        for dept in classification.get("relevant_departments", []):
            meta_tags.append(f"meta:dept:{dept}")
        
        # Custom keywords
        for keyword in classification.get("keywords", []):
            # Clean keyword for tag format
            clean_keyword = keyword.lower().replace(" ", "_").replace("-", "_")
            meta_tags.append(f"meta:keyword:{clean_keyword}")
        
        return meta_tags
    
    def generate_all_tags(self, classification: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate both RBAC and meta tags.
        
        Returns:
            Dict: {"rbac_tags": [...], "meta_tags": [...]}
        """
        return {
            "rbac_tags": self.generate_rbac_tags(classification),
            "meta_tags": self.generate_meta_tags(classification)
        }


# ============================================================================
# ENHANCED METADATA STORAGE
# ============================================================================

@tool
def enhance_document_metadata_tool(
    doc_id: str,
    title: str,
    text: str,
    llm_service,
    db_conn=None,
    source: str = "ingestion"
) -> str:
    """
    Enhanced metadata extraction and tagging.
    
    Uses meta-prompting to classify document and generate RBAC + meta tags.
    
    Args:
        doc_id: Document identifier
        title: Document title
        text: Document content
        llm_service: LLMService instance
        db_conn: Database connection for department/role lookup
        source: Document source
    
    Returns:
        JSON string with enhanced metadata including tags
    """
    try:
        # Initialize classifier
        classifier = DocumentClassificationAgent(llm_service, db_conn)
        
        # Classify document
        classification_result = classifier.classify_document_intent(text, title)
        
        if not classification_result.get("success"):
            # Use fallback classification
            classification = classification_result.get("fallback_classification", {})
            confidence = 0.0
        else:
            classification = classification_result.get("classification", {})
            confidence = classification.get("confidence_score", 0.8)
        
        # Generate tags
        tags = classifier.generate_all_tags(classification)
        
        # Build enhanced metadata
        enhanced_metadata = {
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "classification": {
                "intent": classification.get("intent", "document"),
                "primary_department": classification.get("primary_department", "General"),
                "relevant_departments": classification.get("relevant_departments", ["General"]),
                "required_roles": classification.get("required_roles", ["viewer"]),
                "sensitivity_level": classification.get("sensitivity_level", "internal"),
                "confidence_score": confidence,
                "reasoning": classification.get("reasoning", "")
            },
            "tags": {
                "rbac": tags.get("rbac_tags", []),
                "meta": tags.get("meta_tags", [])
            },
            "keywords": classification.get("keywords", [])
        }
        
        return json.dumps({
            "success": True,
            "metadata": enhanced_metadata,
            "message": f"Document classified with {len(tags.get('rbac_tags', []))} RBAC tags and {len(tags.get('meta_tags', []))} meta tags"
        })
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "metadata": {
                "doc_id": doc_id,
                "title": title,
                "classification": {
                    "intent": "document",
                    "primary_department": "General",
                    "sensitivity_level": "internal",
                    "confidence_score": 0.0
                },
                "tags": {"rbac": [], "meta": []}
            }
        })


if __name__ == "__main__":
    # Example usage
    from src.rag.tools.services.llm_service import LLMService
    
    llm_service = LLMService({})
    classifier = DocumentClassificationAgent(llm_service)
    
    sample_text = """
    This document outlines the quarterly budget review process for the Finance department.
    All managers must submit their budgets by the end of the month. Finance team approval required.
    """
    
    result = classifier.classify_document_intent(sample_text, "Quarterly Budget Review Process")
    print(json.dumps(result, indent=2))
