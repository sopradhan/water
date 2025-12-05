"""
RAG Database Usage Examples
===========================

This script demonstrates common patterns for using the RAG database system.
Run these examples to understand how to interact with the metadata layer.

Usage:
    python examples_rag_database_usage.py
"""

import json
import sys
import io
from datetime import datetime, timedelta
from src.rag.rag_db_models.db import get_connection, close_connection
from src.rag.rag_db_models import (
    DocumentMetadataModel,
    RAGHistoryModel,
    ChunkEmbeddingDataModel,
    AgentMemoryModel,
)

# Fix encoding for Windows cmd
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def example_1_document_registration():
    """Example 1: Register a new document with full metadata."""
    print_section("Example 1: Document Registration")
    
    conn = get_connection()
    doc_model = DocumentMetadataModel(conn)
    
    # Register multiple documents
    documents = [
        {
            "doc_id": "water_pressure_monitoring_v1",
            "title": "Water Pressure Monitoring and Control",
            "author": "Engineering Team",
            "source": "internal_documentation",
            "summary": "Complete guide for monitoring water pressure across all zones",
            "company_id": 1,
            "dept_id": 1,
            "rbac_namespace": "engineering",
            "chunk_strategy": "recursive_splitter",
            "chunk_size_char": 512,
            "overlap_char": 50,
            "metadata_json": json.dumps({
                "doc_type": "technical_guide",
                "version": "1.0",
                "published_date": "2024-01-15",
                "categories": ["pressure", "monitoring", "control"]
            }),
            "rbac_tags": json.dumps(["level_3", "dept_engineering"]),
            "meta_tags": json.dumps(["operational", "technical"])
        },
        {
            "doc_id": "anomaly_detection_procedures",
            "title": "Anomaly Detection Standard Operating Procedures",
            "author": "Data Science Team",
            "source": "internal_documentation",
            "summary": "SOP for detecting and responding to anomalies",
            "company_id": 1,
            "dept_id": 2,
            "rbac_namespace": "operations",
            "chunk_strategy": "recursive_splitter",
            "chunk_size_char": 512,
            "overlap_char": 50,
            "metadata_json": json.dumps({
                "doc_type": "standard_operating_procedure",
                "version": "2.0",
                "effective_date": "2024-03-01"
            }),
            "rbac_tags": json.dumps(["level_2", "dept_operations"]),
            "meta_tags": json.dumps(["anomaly", "detection", "procedures"])
        }
    ]
    
    for doc in documents:
        doc_model.create(**doc)
        print(f"[OK] Registered: {doc['title']}")
        print(f"  ID: {doc['doc_id']}")
        print(f"  RBAC Namespace: {doc['rbac_namespace']}")
    
    conn.commit()
    print("\n[OK] Document registration complete")


def example_2_query_logging():
    """Example 2: Log RAG queries with metrics and context."""
    print_section("Example 2: Query Logging and Metrics")
    
    conn = get_connection()
    history_model = RAGHistoryModel(conn)
    
    # Example 1: Log a regular query
    print("[1] Logging user query:")
    history_id = history_model.log_query(
        query_text="What is the normal pressure range for Zone A?",
        target_doc_id="water_pressure_monitoring_v1",
        metrics_json=json.dumps({
            "accuracy": 0.95,
            "latency_ms": 245,
            "retrieval_time_ms": 120,
            "response_time_ms": 125,
            "cost": 0.0023,
            "model_confidence": 0.92
        }),
        context_json=json.dumps({
            "source": "user_query",
            "user_level": "level_2",
            "attempt": 1
        }),
        agent_id="langgraph_agent",
        user_id="eng_user_001",
        session_id="sess_2024_001"
    )
    print(f"[OK] Logged QUERY event (ID: {history_id})")
    print(f"  Query: What is the normal pressure range for Zone A?")
    
    # Example 2: Log a healing action
    print("\n[2] Logging healing action:")
    history_id = history_model.log_healing(
        target_doc_id="water_pressure_monitoring_v1",
        target_chunk_id="chunk_zone_c_001",
        metrics_json=json.dumps({
            "before_quality": 0.72,
            "after_quality": 0.89,
            "improvement_delta": 0.17,
            "healing_strategy": "re_chunk_with_better_boundaries",
            "time_ms": 1234
        }),
        context_json=json.dumps({
            "reason": "low_retrieval_accuracy",
            "trigger": "RL_agent",
            "suggested_params": {
                "new_chunk_size": 384,
                "new_overlap": 64
            }
        }),
        reward_signal=0.85,
        action_taken="RE_CHUNK_AND_REEMBED",
        agent_id="langgraph_agent",
        session_id="sess_2024_heal_001"
    )
    print(f"[OK] Logged HEAL event (ID: {history_id})")
    print(f"  Improvement: 0.72 → 0.89 (+0.17)")
    print(f"  Reward: 0.85")
    
    conn.commit()
    print("\n[OK] Query logging complete")


def example_3_chunk_embedding_tracking():
    """Example 3: Track chunk embedding quality and health."""
    print_section("Example 3: Chunk Embedding Quality Tracking")
    
    conn = get_connection()
    chunk_model = ChunkEmbeddingDataModel(conn)
    
    # Simulate tracking chunks during ingestion
    document_id = "water_pressure_monitoring_v1"
    chunk_count = 5
    
    print(f"Tracking {chunk_count} chunks from document: {document_id}\n")
    
    for i in range(chunk_count):
        chunk_id = f"{document_id}_chunk_{i+1:03d}"
        
        # Simulate quality scores (some good, some need healing)
        quality_score = 0.92 - (i * 0.05)  # Decreasing quality
        needs_healing = quality_score < 0.75
        
        healing_suggestions = None
        if needs_healing:
            healing_suggestions = json.dumps({
                "strategy": "re_chunk_with_overlap_increase",
                "reason": f"quality_score_below_threshold ({quality_score:.2f})",
                "suggested_params": {
                    "new_chunk_size": 384,
                    "new_overlap": 64,
                    "new_overlap_pct": 16.7
                },
                "confidence": 0.78
            })
        
        chunk_model.create(
            chunk_id=chunk_id,
            doc_id=document_id,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            embedding_version="1.0",
            quality_score=quality_score,
            reindex_count=0,
            healing_suggestions=healing_suggestions,
            rbac_tags=json.dumps(["level_3", "dept_engineering"]),
            meta_tags=json.dumps(["pressure_monitoring"])
        )
        
        status = "[WARN] NEEDS_HEALING" if needs_healing else "[OK] HEALTHY"
        print(f"{status}  Chunk {i+1}: quality={quality_score:.2f}")
    
    conn.commit()
    print("\n[OK] Chunk tracking complete")


def example_4_agent_memory_management():
    """Example 4: Agent self-reflection and learning via memory."""
    print_section("Example 4: Agent Memory Management")
    
    conn = get_connection()
    memory_model = AgentMemoryModel(conn)
    
    now = datetime.now().isoformat()
    
    # Example 1: Context memory - User profiles
    print("[1] CONTEXT MEMORY: User Profiles")
    memory_model.record_memory(
        agent_id="langgraph_agent",
        memory_type="context",
        memory_key="user_profiles",
        content=json.dumps({
            "active_users": ["user_001", "user_002", "user_003"],
            "user_skill_levels": {
                "user_001": "expert",
                "user_002": "intermediate",
                "user_003": "beginner"
            },
            "preferred_answer_depth": {
                "user_001": "comprehensive",
                "user_002": "balanced",
                "user_003": "concise"
            }
        }),
        importance_score=0.8
    )
    print("  [OK] Recorded user profile patterns")
    
    # Example 2: Decision memory - Query strategy
    print("\n[2] DECISION MEMORY: Query Strategies")
    memory_model.record_memory(
        agent_id="langgraph_agent",
        memory_type="decision",
        memory_key="pressure_query_strategy",
        content=json.dumps({
            "strategy_name": "cross_reference_verification",
            "success_rate": 0.94,
            "typical_latency_ms": 245,
            "cost_per_query": 0.0023,
            "learned_from": ["query_history_1", "query_history_2"],
            "last_successful_use": now
        }),
        importance_score=0.9
    )
    print("  [OK] Recorded high-performing query strategy")
    
    # Example 3: Performance memory - System metrics
    print("\n[3] PERFORMANCE MEMORY: System Metrics")
    memory_model.record_memory(
        agent_id="langgraph_agent",
        memory_type="performance",
        memory_key="system_health_metrics",
        content=json.dumps({
            "avg_query_accuracy": 0.92,
            "total_queries_processed": 1256,
            "healing_operations_count": 34,
            "successful_healing_rate": 0.88,
            "avg_response_time_ms": 267,
            "cost_per_query_avg": 0.00245,
            "last_24h_trends": {
                "accuracy_trend": "improving",
                "performance_trend": "stable",
                "cost_trend": "decreasing"
            }
        }),
        importance_score=0.95
    )
    print("  [OK] Recorded system performance baseline")
    
    # Example 4: Log memory - Error handling
    print("\n[4] LOG MEMORY: Error Patterns")
    memory_model.record_memory(
        agent_id="langgraph_agent",
        memory_type="log",
        memory_key="error_patterns_week_1",
        content=json.dumps({
            "error_type": "low_retrieval_accuracy",
            "frequency": 12,
            "affected_documents": ["doc_002", "doc_005"],
            "root_causes": [
                "overlapping_chunks",
                "poor_chunk_boundaries",
                "missing_context"
            ],
            "applied_solutions": [
                "re_chunking",
                "overlap_adjustment",
                "additional_context_injection"
            ],
            "resolution_success_rate": 0.75
        }),
        importance_score=0.7
    )
    print("  [OK] Recorded error pattern analysis")
    
    conn.commit()
    print("\n[OK] Agent memory management complete")


def example_5_query_historical_patterns():
    """Example 5: Query and analyze historical patterns."""
    print_section("Example 5: Historical Pattern Analysis")
    
    conn = get_connection()
    
    # Query document metadata with RBAC
    print("[SEARCH] Finding documents by RBAC namespace:")
    cursor = conn.execute('''
        SELECT doc_id, title, rbac_namespace, chunk_size_char
        FROM document_metadata
        WHERE rbac_namespace = ?
    ''', ('engineering',))
    
    for row in cursor.fetchall():
        print(f"  * {row['title']}")
        print(f"    ID: {row['doc_id']}")
        print(f"    Chunk Size: {row['chunk_size_char']} chars")
    
    # Query recent RAG history
    print("\n[INFO] Recent query events:")
    cursor = conn.execute('''
        SELECT event_type, query_text, timestamp, 
               json_extract(metrics_json, '$.accuracy') as accuracy
        FROM rag_history_and_optimization
        WHERE event_type = 'QUERY'
        ORDER BY timestamp DESC
        LIMIT 3
    ''')
    
    for row in cursor.fetchall():
        print(f"  * {row['event_type']}: {row['query_text'][:50]}...")
        if row['accuracy']:
            print(f"    Accuracy: {float(row['accuracy']):.2%}")
    
    # Find chunks needing healing
    print("\n[WARN] Chunks needing attention (quality < 0.80):")
    cursor = conn.execute('''
        SELECT chunk_id, doc_id, quality_score
        FROM chunk_embedding_data
        WHERE quality_score < 0.8
        ORDER BY quality_score ASC
    ''')
    
    results = cursor.fetchall()
    if results:
        for row in results:
            print(f"  * {row['chunk_id']}")
            print(f"    Quality Score: {row['quality_score']:.2f}")
    else:
        print("  [OK] No chunks below quality threshold")
    
    # Agent memory summary
    print("\n[INFO] Agent memory summary:")
    cursor = conn.execute('''
        SELECT memory_type, COUNT(*) as count, AVG(importance_score) as avg_importance
        FROM agent_memory
        GROUP BY memory_type
    ''')
    
    for row in cursor.fetchall():
        print(f"  - {row['memory_type'].upper()}: {row['count']} memories")
        print(f"    Avg Importance: {row['avg_importance']:.2f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("  RAG DATABASE USAGE EXAMPLES")
    print("=" * 70)
    
    try:
        # Example 1: Document Registration
        example_1_document_registration()
        
        # Example 2: Query Logging
        example_2_query_logging()
        
        # Example 3: Chunk Embedding Tracking
        example_3_chunk_embedding_tracking()
        
        # Example 4: Agent Memory Management
        example_4_agent_memory_management()
        
        # Example 5: Historical Pattern Analysis
        example_5_query_historical_patterns()
        
        print("\n" + "=" * 70)
        print("  [OK] ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nDatabase: src/data/RAG/rag_metadata.db")
        print("Next: Review DATABASE_SETUP_GUIDE.md for comprehensive documentation\n")
        
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        raise
    finally:
        close_connection()


if __name__ == "__main__":
    main()
