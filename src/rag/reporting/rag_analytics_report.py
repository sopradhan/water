"""
RAG Analytics & Reporting Module

Generates comprehensive reports from RAG metadata tables:
1. Document Ingestion Report - Track what was ingested and chunking performance
2. Query Performance Report - Track query metrics, accuracy, cost
3. Healing & Optimization Report - Track system healing actions and improvements
4. Embedding Health Report - Monitor chunk embedding quality and re-indexing needs
5. User & Session Analytics - Query patterns by user/session
6. Cost & Token Analysis - Token usage and estimated costs
7. Quality Dashboard - Overall RAG system health metrics
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from pathlib import Path


class RAGAnalyticsReport:
    """Generate comprehensive analytics reports from RAG metadata tables."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    # ========================================================================
    # REPORT 1: DOCUMENT INGESTION REPORT
    # ========================================================================
    
    def generate_ingestion_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Document Ingestion Report
        
        Metrics:
        - Total documents ingested
        - Documents by namespace (RBAC)
        - Average chunks per document
        - Chunking strategy distribution
        - Ingestion timeline (last N days)
        - Documents by source
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_docs,
                    COUNT(DISTINCT rbac_namespace) as namespaces,
                    AVG(CAST((SELECT COUNT(*) FROM chunk_embedding_data 
                              WHERE chunk_embedding_data.doc_id = document_metadata.doc_id) AS FLOAT)) as avg_chunks_per_doc,
                    MIN(last_ingested) as first_ingestion,
                    MAX(last_ingested) as last_ingestion
                FROM document_metadata
                WHERE last_ingested >= ?
            """, (since.isoformat(),))
            
            summary = dict(self.cursor.fetchone() or {})
            
            # Documents by namespace
            self.cursor.execute("""
                SELECT rbac_namespace, COUNT(*) as count
                FROM document_metadata
                WHERE last_ingested >= ?
                GROUP BY rbac_namespace
                ORDER BY count DESC
            """, (since.isoformat(),))
            
            namespace_dist = [dict(row) for row in self.cursor.fetchall()]
            
            # Chunking strategy distribution
            self.cursor.execute("""
                SELECT chunk_strategy, COUNT(*) as count, 
                       AVG(chunk_size_char) as avg_chunk_size
                FROM document_metadata
                WHERE last_ingested >= ?
                GROUP BY chunk_strategy
            """, (since.isoformat(),))
            
            strategy_dist = [dict(row) for row in self.cursor.fetchall()]
            
            # Documents by source
            self.cursor.execute("""
                SELECT source, COUNT(*) as count
                FROM document_metadata
                WHERE last_ingested >= ?
                GROUP BY source
                ORDER BY count DESC
            """, (since.isoformat(),))
            
            source_dist = [dict(row) for row in self.cursor.fetchall()]
            
            return {
                "report_type": "DOCUMENT_INGESTION",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "summary": summary,
                "namespace_distribution": namespace_dist,
                "chunking_strategy_distribution": strategy_dist,
                "source_distribution": source_dist
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # REPORT 2: QUERY PERFORMANCE REPORT
    # ========================================================================
    
    def generate_query_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Query Performance Report
        
        Metrics:
        - Total queries executed
        - Query frequency timeline
        - Average accuracy/relevance
        - Average latency
        - Average cost per query
        - Top performing queries
        - Queries by user/session
        - Query success rate (reward signal)
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_queries,
                    AVG(CAST(json_extract(metrics_json, '$.accuracy') AS FLOAT)) as avg_accuracy,
                    AVG(CAST(json_extract(metrics_json, '$.latency_ms') AS FLOAT)) as avg_latency_ms,
                    AVG(CAST(json_extract(metrics_json, '$.cost') AS FLOAT)) as avg_cost,
                    AVG(reward_signal) as avg_reward
                FROM rag_history_and_optimization
                WHERE event_type = 'QUERY' AND timestamp >= ?
            """, (since.isoformat(),))
            
            summary = dict(self.cursor.fetchone() or {})
            
            # Query frequency over time (daily)
            self.cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as query_count,
                    AVG(CAST(json_extract(metrics_json, '$.accuracy') AS FLOAT)) as daily_avg_accuracy
                FROM rag_history_and_optimization
                WHERE event_type = 'QUERY' AND timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (since.isoformat(),))
            
            timeline = [dict(row) for row in self.cursor.fetchall()]
            
            # Top performing queries
            self.cursor.execute("""
                SELECT 
                    query_text,
                    COUNT(*) as frequency,
                    AVG(CAST(json_extract(metrics_json, '$.accuracy') AS FLOAT)) as avg_accuracy,
                    AVG(reward_signal) as avg_reward
                FROM rag_history_and_optimization
                WHERE event_type = 'QUERY' AND timestamp >= ?
                GROUP BY query_text
                ORDER BY frequency DESC
                LIMIT 10
            """, (since.isoformat(),))
            
            top_queries = [dict(row) for row in self.cursor.fetchall()]
            
            # Query performance by user
            self.cursor.execute("""
                SELECT 
                    user_id,
                    COUNT(*) as query_count,
                    AVG(CAST(json_extract(metrics_json, '$.accuracy') AS FLOAT)) as avg_accuracy,
                    AVG(reward_signal) as avg_reward
                FROM rag_history_and_optimization
                WHERE event_type = 'QUERY' AND timestamp >= ? AND user_id IS NOT NULL
                GROUP BY user_id
                ORDER BY query_count DESC
            """, (since.isoformat(),))
            
            user_performance = [dict(row) for row in self.cursor.fetchall()]
            
            return {
                "report_type": "QUERY_PERFORMANCE",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "summary": summary,
                "timeline": timeline,
                "top_queries": top_queries,
                "user_performance": user_performance
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # REPORT 3: HEALING & OPTIMIZATION REPORT
    # ========================================================================
    
    def generate_healing_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Healing & Optimization Report
        
        Metrics:
        - Total healing events
        - Healing actions taken (OPTIMIZE, REINDEX, RE_EMBED, SKIP)
        - Average improvement delta
        - Before/after metrics comparison
        - Documents most frequently healed
        - Cost vs benefit of healing actions
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Total healing events
            self.cursor.execute("""
                SELECT COUNT(*) as total_healing_events
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ?
            """, (since.isoformat(),))
            
            total_events = dict(self.cursor.fetchone() or {})
            
            # Healing actions distribution
            self.cursor.execute("""
                SELECT 
                    action_taken,
                    COUNT(*) as count,
                    AVG(reward_signal) as avg_reward,
                    AVG(CAST(json_extract(metrics_json, '$.improvement_delta') AS FLOAT)) as avg_improvement
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ?
                GROUP BY action_taken
                ORDER BY count DESC
            """, (since.isoformat(),))
            
            actions = [dict(row) for row in self.cursor.fetchall()]
            
            # Documents most frequently healed
            self.cursor.execute("""
                SELECT 
                    target_doc_id,
                    COUNT(*) as healing_count,
                    AVG(reward_signal) as avg_reward
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ? AND target_doc_id IS NOT NULL
                GROUP BY target_doc_id
                ORDER BY healing_count DESC
                LIMIT 10
            """, (since.isoformat(),))
            
            most_healed_docs = [dict(row) for row in self.cursor.fetchall()]
            
            # Healing effectiveness (average reward)
            self.cursor.execute("""
                SELECT 
                    AVG(reward_signal) as avg_reward,
                    MIN(reward_signal) as min_reward,
                    MAX(reward_signal) as max_reward,
                    SUM(CASE WHEN reward_signal > 0.5 THEN 1 ELSE 0 END) as effective_count
                FROM rag_history_and_optimization
                WHERE event_type = 'HEAL' AND timestamp >= ?
            """, (since.isoformat(),))
            
            effectiveness = dict(self.cursor.fetchone() or {})
            
            return {
                "report_type": "HEALING_OPTIMIZATION",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "total_events": total_events,
                "actions_distribution": actions,
                "most_healed_documents": most_healed_docs,
                "effectiveness_metrics": effectiveness
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # REPORT 4: EMBEDDING HEALTH REPORT
    # ========================================================================
    
    def generate_embedding_health_report(self) -> Dict[str, Any]:
        """
        Embedding Health Report
        
        Metrics:
        - Total chunks tracked
        - Average quality score
        - Quality score distribution
        - Chunks needing re-indexing (quality < threshold)
        - Re-indexing frequency
        - Embedding model distribution
        - Chunks by healing suggestion
        """
        try:
            # Overall health summary
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(quality_score) as avg_quality_score,
                    MIN(quality_score) as min_quality,
                    MAX(quality_score) as max_quality,
                    AVG(reindex_count) as avg_reindex_count,
                    SUM(CASE WHEN quality_score < 0.7 THEN 1 ELSE 0 END) as chunks_needing_reindex
                FROM chunk_embedding_data
            """)
            
            summary = dict(self.cursor.fetchone() or {})
            
            # Quality score distribution
            self.cursor.execute("""
                SELECT 
                    CASE 
                        WHEN quality_score >= 0.9 THEN 'EXCELLENT (0.9+)'
                        WHEN quality_score >= 0.8 THEN 'GOOD (0.8-0.89)'
                        WHEN quality_score >= 0.7 THEN 'FAIR (0.7-0.79)'
                        ELSE 'POOR (<0.7)'
                    END as quality_band,
                    COUNT(*) as count
                FROM chunk_embedding_data
                GROUP BY quality_band
            """)
            
            quality_distribution = [dict(row) for row in self.cursor.fetchall()]
            
            # Embedding models in use
            self.cursor.execute("""
                SELECT 
                    embedding_model,
                    COUNT(*) as chunk_count,
                    AVG(quality_score) as avg_quality
                FROM chunk_embedding_data
                GROUP BY embedding_model
            """)
            
            embedding_models = [dict(row) for row in self.cursor.fetchall()]
            
            # Chunks with re-indexing needs
            self.cursor.execute("""
                SELECT 
                    chunk_id,
                    doc_id,
                    quality_score,
                    reindex_count,
                    healing_suggestions
                FROM chunk_embedding_data
                WHERE quality_score < 0.7
                ORDER BY quality_score ASC
                LIMIT 20
            """)
            
            poor_quality_chunks = [dict(row) for row in self.cursor.fetchall()]
            
            return {
                "report_type": "EMBEDDING_HEALTH",
                "generated_at": datetime.now().isoformat(),
                "summary": summary,
                "quality_distribution": quality_distribution,
                "embedding_models": embedding_models,
                "poor_quality_chunks": poor_quality_chunks
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # REPORT 5: COST & TOKEN ANALYSIS REPORT
    # ========================================================================
    
    def generate_cost_analysis_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Cost & Token Analysis Report
        
        Metrics:
        - Total token usage
        - Total estimated cost
        - Cost by event type (QUERY, HEAL, SYNTHETIC_TEST)
        - Cost per query
        - Most expensive queries
        - Cost trends over time
        - Cost optimization opportunities
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Total cost by event type
            self.cursor.execute("""
                SELECT 
                    event_type,
                    COUNT(*) as event_count,
                    SUM(CAST(json_extract(metrics_json, '$.cost') AS FLOAT)) as total_cost,
                    AVG(CAST(json_extract(metrics_json, '$.cost') AS FLOAT)) as avg_cost,
                    SUM(CAST(json_extract(metrics_json, '$.tokens') AS FLOAT)) as total_tokens
                FROM rag_history_and_optimization
                WHERE timestamp >= ?
                GROUP BY event_type
            """, (since.isoformat(),))
            
            cost_by_type = [dict(row) for row in self.cursor.fetchall()]
            
            # Daily cost trend
            self.cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    SUM(CAST(json_extract(metrics_json, '$.cost') AS FLOAT)) as daily_cost,
                    SUM(CAST(json_extract(metrics_json, '$.tokens') AS FLOAT)) as daily_tokens,
                    COUNT(*) as event_count
                FROM rag_history_and_optimization
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """, (since.isoformat(),))
            
            daily_trend = [dict(row) for row in self.cursor.fetchall()]
            
            # Most expensive queries
            self.cursor.execute("""
                SELECT 
                    query_text,
                    CAST(json_extract(metrics_json, '$.cost') AS FLOAT) as cost,
                    CAST(json_extract(metrics_json, '$.tokens') AS FLOAT) as tokens,
                    CAST(json_extract(metrics_json, '$.accuracy') AS FLOAT) as accuracy
                FROM rag_history_and_optimization
                WHERE event_type = 'QUERY' AND timestamp >= ?
                ORDER BY cost DESC
                LIMIT 10
            """, (since.isoformat(),))
            
            expensive_queries = [dict(row) for row in self.cursor.fetchall()]
            
            # Total costs summary
            self.cursor.execute("""
                SELECT 
                    SUM(CAST(json_extract(metrics_json, '$.cost') AS FLOAT)) as total_cost,
                    SUM(CAST(json_extract(metrics_json, '$.tokens') AS FLOAT)) as total_tokens,
                    COUNT(*) as total_events
                FROM rag_history_and_optimization
                WHERE timestamp >= ?
            """, (since.isoformat(),))
            
            totals = dict(self.cursor.fetchone() or {})
            
            return {
                "report_type": "COST_ANALYSIS",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "total_summary": totals,
                "cost_by_event_type": cost_by_type,
                "daily_trend": daily_trend,
                "expensive_queries": expensive_queries
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # REPORT 6: SYSTEM HEALTH DASHBOARD
    # ========================================================================
    
    def generate_health_dashboard(self, days: int = 7) -> Dict[str, Any]:
        """
        System Health Dashboard - Quick overview of RAG system health
        
        Metrics:
        - Documents ingested (last 7 days)
        - Queries processed (last 7 days)
        - Average query accuracy
        - Healing actions taken
        - Embedding health score
        - Cost efficiency (revenue per token or accuracy per dollar)
        - System reliability (success rate)
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            # Get all metrics
            docs_stats = self.generate_ingestion_report(days)
            query_stats = self.generate_query_performance_report(days)
            healing_stats = self.generate_healing_report(days)
            embedding_health = self.generate_embedding_health_report()
            cost_stats = self.generate_cost_analysis_report(days)
            
            # Calculate health score (0-100)
            try:
                query_accuracy = query_stats.get("summary", {}).get("avg_accuracy", 0.5) * 100
                embedding_quality = embedding_health.get("summary", {}).get("avg_quality_score", 0.5) * 100
                healing_effectiveness = healing_stats.get("effectiveness_metrics", {}).get("avg_reward", 0.5) * 100
                
                health_score = (query_accuracy * 0.5 + embedding_quality * 0.3 + healing_effectiveness * 0.2)
            except:
                health_score = 0
            
            return {
                "report_type": "HEALTH_DASHBOARD",
                "period_days": days,
                "generated_at": datetime.now().isoformat(),
                "health_score": round(health_score, 2),
                "ingestion_stats": docs_stats.get("summary", {}),
                "query_stats": query_stats.get("summary", {}),
                "healing_stats": healing_stats.get("total_events", {}),
                "embedding_health": embedding_health.get("summary", {}),
                "cost_summary": cost_stats.get("total_summary", {})
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ========================================================================
    # EXPORT UTILITIES
    # ========================================================================
    
    def export_report_json(self, report: Dict[str, Any], filename: str) -> bool:
        """Export report to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False
    
    def export_report_csv(self, report: Dict[str, Any], filename: str) -> bool:
        """Export report data to CSV (for table-like data)."""
        try:
            import csv
            # Simplified CSV export for list-like data in reports
            with open(filename, 'w', newline='') as f:
                # Find first list in report to export
                for key, value in report.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        writer = csv.DictWriter(f, fieldnames=value[0].keys())
                        writer.writeheader()
                        writer.writerows(value)
                        return True
            return False
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    db_path = "path/to/your/database.db"
    analytics = RAGAnalyticsReport(db_path)
    
    # Generate all reports
    print("Generating RAG Analytics Reports...\n")
    
    ingestion_report = analytics.generate_ingestion_report(days=30)
    print("1. Ingestion Report:", ingestion_report)
    
    query_report = analytics.generate_query_performance_report(days=30)
    print("\n2. Query Performance Report:", query_report)
    
    healing_report = analytics.generate_healing_report(days=30)
    print("\n3. Healing Report:", healing_report)
    
    embedding_report = analytics.generate_embedding_health_report()
    print("\n4. Embedding Health Report:", embedding_report)
    
    cost_report = analytics.generate_cost_analysis_report(days=30)
    print("\n5. Cost Analysis Report:", cost_report)
    
    health_dashboard = analytics.generate_health_dashboard(days=7)
    print("\n6. Health Dashboard:", health_dashboard)
    
    # Export reports
    analytics.export_report_json(ingestion_report, "ingestion_report.json")
    analytics.export_report_json(query_report, "query_report.json")
    analytics.export_report_json(health_dashboard, "health_dashboard.json")
