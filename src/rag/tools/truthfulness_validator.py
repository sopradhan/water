"""
Truthfulness Validation Module

Purpose: Validates whether LLM-generated answers are truthful based on:
1. Source alignment - Does the answer text appear in the source?
2. Semantic coherence - Does the answer address the question?
3. Factual consistency - Is the answer logically consistent?
4. User feedback - Learn from past corrections

This allows us to have LOWER thresholds at retrieval time since answers are 
validated AFTER generation, rather than rejecting based on confidence alone.
"""

import json
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass


@dataclass
class TruthfulnessScore:
    """Truthfulness assessment result"""
    overall_score: float  # 0-1, higher = more truthful
    source_alignment: float  # Does answer come from source?
    semantic_coherence: float  # Does answer address question?
    factual_consistency: float  # Is answer logically sound?
    confidence: str  # HIGH/MEDIUM/LOW
    reasons: List[str]  # Why this score?
    suggestions: List[str]  # How to improve?


class TruthfulnessValidator:
    """Validates LLM answers for truthfulness and hallucination detection"""
    
    def __init__(self, llm_service, vectordb_service=None):
        """
        Args:
            llm_service: LLM service for semantic analysis
            vectordb_service: Optional VDB for source verification
        """
        self.llm_service = llm_service
        self.vectordb_service = vectordb_service
        self.feedback_history = []  # Track user feedback for learning
    
    def validate_answer(self, 
                       question: str,
                       answer: str, 
                       context_texts: List[str],
                       embedding_confidence: float = 0.5) -> TruthfulnessScore:
        """
        Comprehensive truthfulness validation of LLM answer.
        
        Args:
            question: Original user question
            answer: LLM-generated answer
            context_texts: Retrieved source documents
            embedding_confidence: Confidence from retrieval (0-1)
        
        Returns:
            TruthfulnessScore with detailed assessment
        """
        reasons = []
        suggestions = []
        
        # Level 1: SOURCE ALIGNMENT - Does answer contain information from source?
        source_alignment = self._check_source_alignment(answer, context_texts)
        if source_alignment < 0.3:
            reasons.append("❌ Answer content not found in source documents")
            suggestions.append("Answer may be hallucinated - check source texts")
        elif source_alignment > 0.7:
            reasons.append("✅ Answer well-grounded in source documents")
        else:
            reasons.append("⚠️ Answer partially grounded in sources")
        
        # Level 2: SEMANTIC COHERENCE - Does answer actually address the question?
        semantic_coherence = self._check_semantic_coherence(question, answer)
        if semantic_coherence < 0.4:
            reasons.append("❌ Answer doesn't address the question")
            suggestions.append("Reframe the answer to directly address: " + question[:50])
        elif semantic_coherence > 0.8:
            reasons.append("✅ Answer directly addresses the question")
        else:
            reasons.append("⚠️ Answer partially addresses the question")
        
        # Level 3: FACTUAL CONSISTENCY - Is answer internally consistent?
        factual_consistency = self._check_factual_consistency(answer, context_texts)
        if factual_consistency < 0.5:
            reasons.append("❌ Answer contains contradictions or inconsistencies")
            suggestions.append("Verify facts and remove contradictory statements")
        elif factual_consistency > 0.8:
            reasons.append("✅ Answer is internally consistent")
        else:
            reasons.append("⚠️ Answer has some internal inconsistencies")
        
        # Level 4: COMBINED TRUTHFULNESS SCORE
        # Weight factors: source alignment (40%), semantic coherence (35%), factual consistency (25%)
        overall_score = (
            0.4 * source_alignment +
            0.35 * semantic_coherence +
            0.25 * factual_consistency
        )
        
        # Adjust based on retrieval confidence
        # If retrieval was weak but answer is still good, boost score
        # If retrieval was strong but answer is weak, lower score
        retrieval_factor = (embedding_confidence - 0.5) * 0.1  # -0.05 to +0.05
        overall_score = max(0.0, min(1.0, overall_score + retrieval_factor))
        
        # Determine confidence level
        if overall_score > 0.75:
            confidence_level = "HIGH"
        elif overall_score > 0.5:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        
        # Add meta-reasoning
        reasons.insert(0, f"📊 Truthfulness Score: {overall_score:.1%}")
        reasons.append(f"📈 Embedding confidence: {embedding_confidence:.1%}")
        
        return TruthfulnessScore(
            overall_score=overall_score,
            source_alignment=source_alignment,
            semantic_coherence=semantic_coherence,
            factual_consistency=factual_consistency,
            confidence=confidence_level,
            reasons=reasons,
            suggestions=suggestions
        )
    
    def _check_source_alignment(self, answer: str, context_texts: List[str]) -> float:
        """
        Check if answer text appears in or is derived from source.
        Uses substring matching + semantic similarity.
        
        Returns: 0-1 score
        """
        if not context_texts or not answer:
            return 0.0
        
        combined_context = " ".join(context_texts).lower()
        answer_lower = answer.lower()
        
        # Step 1: Substring matching (most reliable - direct copy from source)
        # Extract key phrases from answer (2-4 word chunks)
        answer_words = answer_lower.split()
        substring_matches = 0
        
        for i in range(len(answer_words) - 1):
            phrase = " ".join(answer_words[i:i+3])
            if phrase in combined_context:
                substring_matches += 1
        
        substring_score = min(1.0, substring_matches / max(1, len(answer_words) - 2))
        
        # Step 2: Semantic similarity (LLM-based verification)
        try:
            check_prompt = f"""Does the following answer contain information from the source documents?
Rate from 0-1 where:
- 1.0 = Answer directly quotes or closely paraphrases source
- 0.5 = Answer infers from source with reasonable logic
- 0.0 = Answer contradicts or is unrelated to source

Source (first 500 chars):
{combined_context[:500]}

Answer:
{answer[:300]}

Return ONLY a decimal number 0.0-1.0, nothing else."""
            
            semantic_score_str = self.llm_service.generate_response(check_prompt).strip()
            semantic_score = float(semantic_score_str)
        except:
            semantic_score = 0.5  # Default to neutral if check fails
        
        # Combine: substring matching is more reliable (70%), semantic is backup (30%)
        alignment_score = 0.7 * substring_score + 0.3 * semantic_score
        return min(1.0, alignment_score)
    
    def _check_semantic_coherence(self, question: str, answer: str) -> float:
        """
        Check if answer actually addresses the question.
        Uses semantic similarity between question and answer.
        
        Returns: 0-1 score
        """
        if not question or not answer:
            return 0.0
        
        # Simple heuristic: check if key concepts from question appear in answer
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'is', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'on', 'at', 'for'}
        q_words = q_words - common_words
        a_words = a_words - common_words
        
        # Calculate Jaccard similarity
        if not q_words:
            return 0.5
        
        overlap = len(q_words & a_words)
        union = len(q_words | a_words)
        word_overlap = overlap / union if union > 0 else 0
        
        # Use LLM for semantic coherence check
        try:
            coherence_prompt = f"""Does this answer address the question? Rate 0-1.
            
Question: {question}
Answer: {answer[:300]}

Return ONLY a decimal 0.0-1.0."""
            
            llm_coherence_str = self.llm_service.generate_response(coherence_prompt).strip()
            llm_coherence = float(llm_coherence_str)
        except:
            llm_coherence = 0.5
        
        # Combine: word overlap (40%) + LLM coherence (60%)
        coherence_score = 0.4 * word_overlap + 0.6 * llm_coherence
        return min(1.0, coherence_score)
    
    def _check_factual_consistency(self, answer: str, context_texts: List[str]) -> float:
        """
        Check for contradictions and logical inconsistencies in answer.
        
        Returns: 0-1 score
        """
        if not answer or not context_texts:
            return 0.5
        
        combined_context = " ".join(context_texts)
        
        try:
            consistency_prompt = f"""Check if the answer is internally consistent and doesn't contradict the source.
            
Source: {combined_context[:400]}

Answer: {answer[:300]}

Are there any contradictions or factual errors? Rate consistency 0-1:
- 1.0 = Fully consistent, no contradictions
- 0.5 = Some minor inconsistencies
- 0.0 = Major contradictions or factual errors

Return ONLY a decimal 0.0-1.0."""
            
            consistency_str = self.llm_service.generate_response(consistency_prompt).strip()
            consistency_score = float(consistency_str)
        except:
            consistency_score = 0.7  # Default to mostly consistent if check fails
        
        return consistency_score
    
    def record_feedback(self, question: str, answer: str, is_truthful: bool, 
                       predicted_score: float, user_feedback: str = ""):
        """
        Record user feedback to learn which validation signals matter most.
        
        Args:
            question: Original question
            answer: LLM answer
            is_truthful: User indicated if answer was truthful
            predicted_score: Our predicted truthfulness score
            user_feedback: Optional user comment
        """
        self.feedback_history.append({
            "question": question,
            "answer": answer,
            "is_truthful": is_truthful,
            "predicted_score": predicted_score,
            "error": is_truthful - (1 if predicted_score > 0.5 else 0),
            "feedback": user_feedback
        })
    
    def get_feedback_analysis(self) -> Dict[str, Any]:
        """Analyze feedback to improve validation thresholds"""
        if not self.feedback_history:
            return {"total_feedback": 0, "analysis": "No feedback yet"}
        
        correct = sum(1 for f in self.feedback_history if abs(f["error"]) < 0.5)
        accuracy = correct / len(self.feedback_history)
        
        avg_predicted = sum(f["predicted_score"] for f in self.feedback_history) / len(self.feedback_history)
        truthful_rate = sum(1 for f in self.feedback_history if f["is_truthful"]) / len(self.feedback_history)
        
        return {
            "total_feedback": len(self.feedback_history),
            "validation_accuracy": accuracy,
            "average_predicted_score": avg_predicted,
            "actual_truthfulness_rate": truthful_rate,
            "recommendation": "Thresholds are good" if accuracy > 0.8 else "Adjust thresholds"
        }
