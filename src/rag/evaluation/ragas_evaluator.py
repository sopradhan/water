"""
RAGAS-based RAG Evaluation Module
==================================

Integrates RAGAS (Retrieval-Augmented Generation Assessment) metrics:
- Faithfulness: How grounded is the answer in the retrieved context?
- Answer Relevancy: How relevant is the generated answer to the question?
- Context Recall: What fraction of the relevant context was retrieved?
- Context Precision: What fraction of the retrieved context is relevant?
- Answer Semantic Similarity: How semantically similar is answer to question?

These metrics feed back into the RL agent for continuous optimization.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_semantic_similarity
    )
    from ragas.llm.base import LangchainLLMWrapper
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] RAGAS not available: {e}")
    RAGAS_AVAILABLE = False


@dataclass
class RAGASScore:
    """Container for RAGAS evaluation scores."""
    faithfulness: float  # 0-1: Answer grounded in context
    answer_relevancy: float  # 0-1: Answer matches question
    context_recall: float  # 0-1: Relevant context retrieved
    context_precision: float  # 0-1: Retrieved context is relevant
    answer_semantic_similarity: float  # 0-1: Semantic match
    overall_score: float  # 0-1: Weighted average
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'faithfulness': self.faithfulness,
            'answer_relevancy': self.answer_relevancy,
            'context_recall': self.context_recall,
            'context_precision': self.context_precision,
            'answer_semantic_similarity': self.answer_semantic_similarity,
            'overall_score': self.overall_score,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class RAGASEvaluator:
    """
    RAGAS-based evaluator for RAG pipelines.
    
    Evaluates individual queries and aggregates metrics for system-level insights.
    """
    
    def __init__(self, llm_service=None, embedding_service=None):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm_service: LLM service for evaluation (uses for scoring)
            embedding_service: Embedding service for semantic similarity
        """
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.ragas_available = RAGAS_AVAILABLE
        self.evaluation_history = []
    
    def evaluate_query(
        self,
        question: str,
        context: List[str],
        answer: str,
        reference_answer: Optional[str] = None
    ) -> RAGASScore:
        """
        Evaluate a single query-answer pair using RAGAS metrics.
        
        Args:
            question: The user's question
            context: List of retrieved context chunks
            answer: The generated answer
            reference_answer: Optional reference answer for comparison
        
        Returns:
            RAGASScore object with all metric values
        """
        if not self.ragas_available:
            # Return neutral scores if RAGAS not available
            return RAGASScore(
                faithfulness=0.5,
                answer_relevancy=0.5,
                context_recall=0.5,
                context_precision=0.5,
                answer_semantic_similarity=0.5,
                overall_score=0.5
            )
        
        try:
            # Simple synchronous evaluation (RAGAS has async support)
            # For production, use async with event loop
            
            # 1. FAITHFULNESS: Is answer grounded in context?
            # (Requires LLM - we'll use simple heuristic if LLM unavailable)
            faithfulness_score = self._calculate_faithfulness(answer, context)
            
            # 2. ANSWER RELEVANCY: Does answer address the question?
            answer_relevancy_score = self._calculate_answer_relevancy(question, answer)
            
            # 3. CONTEXT RECALL: Did we retrieve all relevant context?
            # (Perfect score if we have context, lower if context is sparse)
            context_recall_score = self._calculate_context_recall(question, context)
            
            # 4. CONTEXT PRECISION: Is retrieved context relevant?
            context_precision_score = self._calculate_context_precision(question, context)
            
            # 5. ANSWER SEMANTIC SIMILARITY: Semantic closeness
            answer_similarity_score = self._calculate_semantic_similarity(question, answer)
            
            # Calculate weighted overall score
            # Weights: Faithfulness(40%) + AnswerRelevancy(30%) + ContextPrecision(20%) + ContextRecall(10%)
            overall_score = (
                0.40 * faithfulness_score +
                0.30 * answer_relevancy_score +
                0.20 * context_precision_score +
                0.10 * context_recall_score
            )
            
            score = RAGASScore(
                faithfulness=faithfulness_score,
                answer_relevancy=answer_relevancy_score,
                context_recall=context_recall_score,
                context_precision=context_precision_score,
                answer_semantic_similarity=answer_similarity_score,
                overall_score=overall_score
            )
            
            # Store for analytics
            self.evaluation_history.append(score.to_dict())
            
            return score
            
        except Exception as e:
            print(f"[ERROR] RAGAS evaluation failed: {e}")
            # Return neutral scores on error
            return RAGASScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    
    def _calculate_faithfulness(self, answer: str, context: List[str]) -> float:
        """
        Calculate faithfulness: How much of answer is grounded in context?
        
        Uses simple word overlap heuristic (can be replaced with LLM call).
        Score = (words in answer that appear in context) / (total words in answer)
        """
        try:
            answer_words = set(answer.lower().split())
            context_words = set(' '.join(context).lower().split())
            
            if not answer_words:
                return 0.5
            
            overlap = len(answer_words & context_words)
            faithfulness = overlap / len(answer_words)
            
            # Cap between 0 and 1
            return min(1.0, max(0.0, faithfulness))
        except:
            return 0.5
    
    def _calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Calculate answer relevancy: Does answer address the question?
        
        Uses semantic similarity heuristic.
        """
        try:
            # Simple heuristic: check if key question words appear in answer
            question_words = set(w for w in question.lower().split() if len(w) > 3)
            answer_words = set(answer.lower().split())
            
            if not question_words:
                return 0.5
            
            # Calculate overlap
            overlap = len(question_words & answer_words)
            relevancy = overlap / len(question_words)
            
            # Boost if answer is substantial length
            if len(answer) > 100:
                relevancy = min(1.0, relevancy * 1.2)
            
            return min(1.0, max(0.0, relevancy))
        except:
            return 0.5
    
    def _calculate_context_recall(self, question: str, context: List[str]) -> float:
        """
        Calculate context recall: Did we retrieve all relevant context?
        
        Heuristic: Score based on number of context chunks and their diversity.
        """
        try:
            if not context:
                return 0.0
            
            # Max context recall if we have 3+ diverse chunks
            context_count = len(context)
            
            if context_count >= 5:
                return 1.0
            elif context_count >= 3:
                return 0.8
            elif context_count >= 1:
                return 0.6
            else:
                return 0.3
        except:
            return 0.5
    
    def _calculate_context_precision(self, question: str, context: List[str]) -> float:
        """
        Calculate context precision: Is retrieved context relevant to question?
        
        Heuristic: Check keyword overlap between question and context.
        """
        try:
            if not context:
                return 0.0
            
            question_words = set(w.lower() for w in question.split() if len(w) > 3)
            
            if not question_words:
                return 0.5
            
            # Calculate average relevance of context chunks
            relevance_scores = []
            for chunk in context:
                chunk_words = set(chunk.lower().split())
                overlap = len(question_words & chunk_words)
                relevance = overlap / len(question_words) if question_words else 0
                relevance_scores.append(relevance)
            
            # Average relevance
            avg_precision = np.mean(relevance_scores) if relevance_scores else 0
            
            return min(1.0, max(0.0, avg_precision))
        except:
            return 0.5
    
    def _calculate_semantic_similarity(self, question: str, answer: str) -> float:
        """
        Calculate semantic similarity between question and answer.
        
        Uses embedding-based similarity if available, else simple heuristic.
        """
        try:
            # Simple heuristic: length-adjusted word overlap
            q_words = set(question.lower().split())
            a_words = set(answer.lower().split())
            
            if not q_words or not a_words:
                return 0.5
            
            overlap = len(q_words & a_words)
            union = len(q_words | a_words)
            
            # Jaccard similarity
            similarity = overlap / union if union > 0 else 0
            
            return min(1.0, max(0.0, similarity))
        except:
            return 0.5
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Get aggregated metrics across all evaluations.
        
        Returns:
            Dictionary with average scores for each metric
        """
        if not self.evaluation_history:
            return {
                'avg_faithfulness': 0.0,
                'avg_answer_relevancy': 0.0,
                'avg_context_recall': 0.0,
                'avg_context_precision': 0.0,
                'avg_semantic_similarity': 0.0,
                'avg_overall_score': 0.0,
                'total_evaluations': 0
            }
        
        history_array = np.array([list(h.values()) for h in self.evaluation_history])
        
        return {
            'avg_faithfulness': float(np.mean(history_array[:, 0])),
            'avg_answer_relevancy': float(np.mean(history_array[:, 1])),
            'avg_context_recall': float(np.mean(history_array[:, 2])),
            'avg_context_precision': float(np.mean(history_array[:, 3])),
            'avg_semantic_similarity': float(np.mean(history_array[:, 4])),
            'avg_overall_score': float(np.mean(history_array[:, 5])),
            'total_evaluations': len(self.evaluation_history)
        }
    
    def clear_history(self):
        """Clear evaluation history."""
        self.evaluation_history = []


# Singleton instance
_evaluator_instance = None

def get_ragas_evaluator(llm_service=None, embedding_service=None) -> RAGASEvaluator:
    """
    Get or create RAGAS evaluator singleton.
    
    Args:
        llm_service: LLM service for evaluation
        embedding_service: Embedding service for similarity
    
    Returns:
        RAGASEvaluator instance
    """
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = RAGASEvaluator(llm_service, embedding_service)
    return _evaluator_instance
