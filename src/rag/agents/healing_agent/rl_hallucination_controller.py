"""
Enhanced RL Healing Agent for Hallucination Control & Dynamic RAG Configuration
Purpose: Use RL to dynamically adjust LLM behavior, prompt strategies, and RAG parameters
to minimize hallucination while maintaining answer quality

Features:
- Detects hallucination patterns (context mismatch, low relevance)
- Dynamically adjusts prompt temperature, top_p, CoT reasoning
- Adapts retrieval parameters based on quality feedback
- Learns from past hallucination/accuracy patterns
"""

import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class HallucinationDetection:
    """Detected hallucination metrics"""
    is_hallucination: bool
    confidence: float  # 0-1 probability it's hallucination
    indicators: List[str]  # What caused the detection
    context_relevance: float  # 0-1, how well context matches question
    answer_grounding: float  # 0-1, how well answer is grounded in context
    semantic_drift: float  # 0-1, how much answer diverges from context topics


@dataclass
class DynamicRAGConfig:
    """Dynamic configuration parameters"""
    # LLM Behavior
    llm_temperature: float = 0.3  # Lower = more deterministic (less hallucination)
    llm_top_p: float = 0.7  # Lower = more focused responses
    use_cot_reasoning: bool = True  # Chain-of-Thought
    cot_depth: int = 3  # How many reasoning steps
    
    # Prompt Strategy
    require_grounding: bool = True  # Force answer grounding in context
    hallucination_warnings: bool = True  # Add warnings about low quality
    context_matching_check: bool = True  # Verify context matches question
    
    # Retrieval Parameters
    retrieval_k: int = 5  # Top-k documents
    relevance_threshold: float = 0.5  # Min relevance score
    diversity_multiplier: float = 1.0  # Balance relevance vs diversity
    rerank_enabled: bool = True
    
    # Response Strategy
    refuse_on_low_relevance: bool = True  # Don't answer if relevance too low
    max_tokens: int = 256  # Limit response length
    confidence_threshold: float = 0.6  # Min confidence to answer


class RLHallucinationController:
    """
    RL agent that controls hallucination by:
    1. Detecting potential hallucinations
    2. Adjusting LLM/RAG parameters dynamically
    3. Learning from feedback over time
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.epsilon = 0.2  # Exploration rate
        self.learning_rate = 0.1
        
        # Track hallucination events and their outcomes
        self.hallucination_history: List[Dict[str, Any]] = []
        
        # Q-values for configuration choices
        self.config_q_values: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'hallucinations_detected': 0,
            'hallucinations_prevented': 0,
            'false_positives': 0,
            'avg_confidence': 0.0
        }
        
        # Action history for learning
        self.action_outcomes = {
            'reduce_temp': [],  # Reducing temperature outcomes
            'increase_cot': [],  # Adding CoT reasoning outcomes
            'raise_threshold': [],  # Raising relevance threshold outcomes
            'refuse_answer': []  # Refusing to answer outcomes
        }
    
    def detect_hallucination(self, 
                            question: str,
                            answer: str,
                            context: List[Dict[str, Any]],
                            relevance_scores: List[float]) -> HallucinationDetection:
        """
        Detect if answer is likely a hallucination
        
        Returns:
            HallucinationDetection with probability and indicators
        """
        indicators = []
        confidence_sum = 0.0
        
        # Calculate context relevance
        if relevance_scores:
            avg_relevance = np.mean(relevance_scores)
            context_relevance = avg_relevance
        else:
            context_relevance = 0.0
        
        if context_relevance < 0.3:
            indicators.append("Very low context relevance")
            confidence_sum += 0.4
        elif context_relevance < 0.5:
            indicators.append("Low context relevance")
            confidence_sum += 0.2
        
        # Check if answer is grounded in context
        answer_grounding = self._check_answer_grounding(answer, context)
        if answer_grounding < 0.3:
            indicators.append("Answer poorly grounded in context")
            confidence_sum += 0.35
        elif answer_grounding < 0.6:
            indicators.append("Answer weakly grounded in context")
            confidence_sum += 0.15
        
        # Check for semantic drift (answer topic different from context topic)
        semantic_drift = self._calculate_semantic_drift(question, answer, context)
        if semantic_drift > 0.7:
            indicators.append("High semantic drift from context")
            confidence_sum += 0.25
        
        # Check context-question match
        context_question_match = self._check_context_question_match(question, context)
        if context_question_match < 0.4:
            indicators.append("Question doesn't match context topics")
            confidence_sum += 0.3
        
        # Combine signals
        is_likely_hallucination = confidence_sum > 0.5
        
        detection = HallucinationDetection(
            is_hallucination=is_likely_hallucination,
            confidence=min(1.0, confidence_sum),
            indicators=indicators,
            context_relevance=context_relevance,
            answer_grounding=answer_grounding,
            semantic_drift=semantic_drift
        )
        
        return detection
    
    def _check_answer_grounding(self, answer: str, context: List[Dict[str, Any]]) -> float:
        """
        Check how well answer is grounded in provided context
        Returns 0-1 score of grounding quality
        """
        if not context or not answer:
            return 0.0
        
        # Extract key phrases from context
        context_text = " ".join([c.get('text', '') for c in context])
        context_words = set(context_text.lower().split())
        
        # Extract key phrases from answer
        answer_words = set(answer.lower().split())
        
        # Calculate word overlap
        overlap = len(context_words & answer_words) / max(len(answer_words), 1)
        
        # Check for sentences with context references
        sentences = answer.split('.')
        grounded_sentences = sum(1 for s in sentences if any(w in s.lower() for w in context_words))
        sentence_grounding = grounded_sentences / max(len(sentences), 1)
        
        # Combined score
        grounding_score = 0.6 * overlap + 0.4 * sentence_grounding
        return grounding_score
    
    def _calculate_semantic_drift(self, question: str, answer: str, context: List[Dict[str, Any]]) -> float:
        """
        Calculate how much answer topic diverges from context topics
        Returns 0-1 drift score (1 = maximum drift/hallucination)
        """
        # Extract key concepts
        context_text = " ".join([c.get('text', '')[:100] for c in context])  # First 100 chars of each
        
        # Simple semantic similarity using word overlap
        context_words = set(context_text.lower().split())
        answer_words = set(answer.lower().split())
        question_words = set(question.lower().split())
        
        # How much does answer relate to question vs context?
        question_match = len(answer_words & question_words) / max(len(answer_words), 1)
        context_match = len(answer_words & context_words) / max(len(answer_words), 1)
        
        # If answer matches question but not context, it's drift
        if question_match > 0.5 and context_match < 0.3:
            drift = 0.8
        elif context_match < 0.2:
            drift = 0.6
        elif context_match < 0.4:
            drift = 0.3
        else:
            drift = 0.1
        
        return drift
    
    def _check_context_question_match(self, question: str, context: List[Dict[str, Any]]) -> float:
        """
        Check if question topics match context document topics
        Returns 0-1 match score
        """
        if not context:
            return 0.0
        
        context_text = " ".join([c.get('text', '')[:200] for c in context])  # First 200 chars
        
        # Extract key terms from context (first few words after common words)
        stop_words = {'the', 'is', 'a', 'and', 'or', 'to', 'of', 'in', 'for', 'by', 'with'}
        context_key_words = set(w for w in context_text.lower().split() if w not in stop_words and len(w) > 3)
        question_key_words = set(w for w in question.lower().split() if w not in stop_words and len(w) > 3)
        
        # Calculate overlap
        if not question_key_words:
            return 0.5
        
        match_score = len(context_key_words & question_key_words) / len(question_key_words)
        return match_score
    
    def decide_config_adjustment(self, 
                                detection: HallucinationDetection,
                                current_config: DynamicRAGConfig) -> Tuple[DynamicRAGConfig, str]:
        """
        Decide how to adjust configuration to prevent hallucination
        
        Returns:
            (new_config, reasoning)
        """
        new_config = DynamicRAGConfig(**vars(current_config))
        actions = []
        
        if detection.is_hallucination and detection.confidence > 0.7:
            # High confidence hallucination - aggressive correction
            
            # 1. Reduce LLM temperature for more deterministic responses
            new_config.llm_temperature = max(0.1, current_config.llm_temperature - 0.2)
            actions.append(f"Reduce LLM temperature to {new_config.llm_temperature:.2f}")
            
            # 2. Increase CoT reasoning depth
            new_config.cot_depth = min(5, current_config.cot_depth + 1)
            new_config.use_cot_reasoning = True
            actions.append(f"Enable/increase CoT reasoning to depth {new_config.cot_depth}")
            
            # 3. Raise relevance threshold
            new_config.relevance_threshold = min(0.8, current_config.relevance_threshold + 0.2)
            actions.append(f"Raise relevance threshold to {new_config.relevance_threshold:.2f}")
            
            # 4. Reduce answer length
            new_config.max_tokens = int(current_config.max_tokens * 0.7)
            actions.append(f"Reduce max tokens to {new_config.max_tokens}")
        
        elif detection.is_hallucination and detection.confidence > 0.5:
            # Moderate hallucination risk - gentle correction
            
            # Add hallucination warnings
            new_config.hallucination_warnings = True
            actions.append("Enable hallucination warnings in prompt")
            
            # Require stronger grounding
            new_config.require_grounding = True
            actions.append("Require strict answer grounding in context")
            
            # Slightly reduce temperature
            new_config.llm_temperature = max(0.15, current_config.llm_temperature - 0.1)
            actions.append(f"Slightly reduce temperature to {new_config.llm_temperature:.2f}")
        
        elif detection.context_relevance < 0.4:
            # Low relevance context - refuse to answer
            new_config.refuse_on_low_relevance = True
            new_config.relevance_threshold = max(0.5, current_config.relevance_threshold + 0.1)
            actions.append(f"Set refuse threshold to {new_config.relevance_threshold:.2f}")
        
        elif detection.answer_grounding < 0.5:
            # Poor grounding - enforce context matching
            new_config.context_matching_check = True
            new_config.use_cot_reasoning = True
            actions.append("Enforce context matching check with CoT reasoning")
        
        reasoning = "Hallucination prevention: " + " | ".join(actions) if actions else "No adjustment needed"
        
        return new_config, reasoning
    
    def get_rl_recommendations(self, 
                              query_results: Dict[str, Any],
                              session_id: str = None) -> Dict[str, Any]:
        """
        Get RL-based recommendations for system behavior
        
        Analyzes recent performance and suggests optimal configuration
        """
        detection = self.detect_hallucination(
            question=query_results.get('question', ''),
            answer=query_results.get('answer', ''),
            context=query_results.get('context', []),
            relevance_scores=query_results.get('relevance_scores', [])
        )
        
        current_config = DynamicRAGConfig()  # Get from state in real usage
        new_config, reasoning = self.decide_config_adjustment(detection, current_config)
        
        return {
            'hallucination_detected': detection.is_hallucination,
            'hallucination_confidence': detection.confidence,
            'indicators': detection.indicators,
            'grounding_score': detection.answer_grounding,
            'relevance_score': detection.context_relevance,
            'semantic_drift': detection.semantic_drift,
            'recommended_config': vars(new_config),
            'reasoning': reasoning,
            'session_id': session_id
        }


# Integration with answer_question_tool
def integrate_rl_hallucination_control(llm_service, rl_controller: RLHallucinationController):
    """
    Factory to wrap answer_question_tool with RL hallucination control
    """
    def controlled_answer_question(question: str, context: str, rbac_context: dict = None) -> str:
        """
        Enhanced answer generation with RL hallucination control
        """
        # Parse context
        context_data = json.loads(context) if isinstance(context, str) else context
        context_texts = context_data.get('reranked_context', [])
        relevance_scores = [c.get('metadata', {}).get('relevance_score', 0) for c in context_texts]
        
        # Detect hallucination risk
        detection = rl_controller.detect_hallucination(
            question=question,
            answer="",  # Detect before generation
            context=context_texts,
            relevance_scores=relevance_scores
        )
        
        print(f"\n[RL HALLUCINATION DETECTION]")
        print(f"  Hallucination Risk: {detection.confidence:.2%}")
        print(f"  Context Relevance: {detection.context_relevance:.2f}")
        print(f"  Indicators: {', '.join(detection.indicators) if detection.indicators else 'None'}")
        
        # If high risk, refuse to answer
        if detection.is_hallucination and detection.confidence > 0.8:
            print(f"  ✓ REFUSED: High hallucination risk detected")
            return json.dumps({
                "success": True,
                "answer": f"I cannot reliably answer this question. The available documents don't contain relevant information about '{question}'. Please try a more specific query or ensure documents related to your topic are indexed.",
                "hallucination_prevented": True,
                "detection": {
                    "confidence": detection.confidence,
                    "indicators": detection.indicators
                }
            })
        
        # Otherwise, generate with grounding prompt
        context_text = "\n\n".join([f"[{c.get('metadata', {}).get('doc_id', 'N/A')}]\n{c['text']}" 
                                     for c in context_texts])
        
        prompt = f"""Answer based ONLY on provided context. Show your reasoning.

Context Documents:
{context_text}

Question: {question}

Step 1 - Identify: What is this context about?
Step 2 - Match: Does question match these topics?
Step 3 - Find: What specific information answers the question?
Step 4 - Answer: Based on Step 3 above, provide answer.

If context doesn't cover the question, say: "I don't have relevant information."

Answer:"""
        
        answer = llm_service.generate_response(prompt)
        
        # Detect if answer might still be hallucinating
        post_generation_detection = rl_controller.detect_hallucination(
            question=question,
            answer=answer,
            context=context_texts,
            relevance_scores=relevance_scores
        )
        
        if post_generation_detection.is_hallucination and post_generation_detection.confidence > 0.6:
            print(f"  ⚠ POST-GENERATION HALLUCINATION DETECTED: {post_generation_detection.confidence:.2%}")
            # Refund answer to safer response
            answer = f"Based on the available documents, I cannot provide a confident answer to '{question}'. The documents appear to be about: {', '.join([c.get('metadata', {}).get('doc_id', 'unknown') for c in context_texts])}"
        
        return json.dumps({
            "success": True,
            "answer": answer,
            "hallucination_detected": detection.is_hallucination,
            "hallucination_score": detection.confidence,
            "grounding_score": detection.answer_grounding
        })
    
    return controlled_answer_question
