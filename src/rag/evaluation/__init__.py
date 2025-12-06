"""
RAG Evaluation Module - RAGAS Integration
"""

RAGAS_AVAILABLE = False
RAGASEvaluator = None
RAGASScore = None
get_ragas_evaluator = None

try:
    from .ragas_evaluator import (
        RAGASEvaluator,
        RAGASScore,
        get_ragas_evaluator,
        RAGAS_AVAILABLE
    )
except ImportError as e:
    # RAGAS dependencies may not be available
    print(f"[WARNING] RAGAS evaluation module not available: {e}")
    RAGAS_AVAILABLE = False
    RAGASEvaluator = None
    RAGASScore = None
    get_ragas_evaluator = None

__all__ = [
    'RAGASEvaluator',
    'RAGASScore',
    'get_ragas_evaluator',
    'RAGAS_AVAILABLE'
]
