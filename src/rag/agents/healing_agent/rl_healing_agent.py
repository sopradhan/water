"""
Reinforcement Learning-based Healing Agent
Purpose: Intelligently optimize RAG system using RL with meta-tagging and keyword matching
Date: 2025-11-26

RL Architecture:
- State: Document quality, query accuracy, token cost, meta-tag confidence
- Actions: SKIP, OPTIMIZE, REINDEX, RE_EMBED, CLARIFY_USER, SUGGEST_TAGS
- Reward: (Quality_Improvement - Cost) * Confidence * MetaTagScore
- Learning: Track effectiveness and improve decision-making over time
"""
import json
import sqlite3
import numpy as np
import re
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import Counter
import os


@dataclass
class RLState:
    """Current system state for RL agent decision-making"""
    quality_score: float
    query_accuracy: float
    chunk_count: int
    avg_token_cost: float
    reindex_count: int
    last_healing_delta: float
    query_frequency: int
    user_feedback: float
    keyword_match_score: float = 0.0  # NEW: Keyword overlap score (0-1)
    meta_tag_confidence: float = 0.5  # NEW: Meta-tag matching confidence (0-1)
    meta_tags: List[str] = None  # NEW: Detected semantic tags


@dataclass
class RLAction:
    """Action to be taken by the agent"""
    action: str  # SKIP, OPTIMIZE, REINDEX, RE_EMBED
    params: Dict[str, Any]
    estimated_improvement: float
    estimated_cost: float
    confidence: float


class RLHealingAgent:
    """
    Reinforcement Learning-based healing agent for RAG optimization
    
    Learning Strategy:
    - Epsilon-greedy: Explore new strategies vs exploit known good ones
    - Q-learning: Update value estimates based on observed rewards
    - Contextual bandits: Consider document characteristics for decisions
    """
    
    def __init__(self, db_path: str, initial_epsilon: float = 0.3, llm_service=None):
        """
        Initialize RL Healing Agent
        
        Args:
            db_path: Path to SQLite database
            initial_epsilon: Initial exploration rate (0-1)
            llm_service: LLM service for dynamic meta-tag extraction (optional)
        """
        self.db_path = db_path
        self.epsilon = initial_epsilon  # Exploration rate
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.llm_service = llm_service  # For dynamic meta-tag extraction
        
        # Q-values: state_action_pair -> value (learning memory)
        self.q_values: Dict[str, float] = {}
        
        # Action statistics for learning
        self.action_history: Dict[str, Dict[str, float]] = {
            'SKIP': {'count': 0, 'total_reward': 0, 'avg_reward': 0},
            'OPTIMIZE': {'count': 0, 'total_reward': 0, 'avg_reward': 0},
            'REINDEX': {'count': 0, 'total_reward': 0, 'avg_reward': 0},
            'RE_EMBED': {'count': 0, 'total_reward': 0, 'avg_reward': 0},
            'CLARIFY_USER': {'count': 0, 'total_reward': 0, 'avg_reward': 0},  # NEW
            'SUGGEST_TAGS': {'count': 0, 'total_reward': 0, 'avg_reward': 0},  # NEW
        }
        
        # Cache for learned meta tags to avoid repeated LLM calls
        self.meta_tag_cache: Dict[str, List[str]] = {}
        
        self._init_db()
    
    # ========================================================================
    # META-TAGGING & KEYWORD MATCHING METHODS (NEW)
    # ========================================================================
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Dynamically extract key concepts/keywords from text using LLM when available,
        fallback to simple regex-based extraction.
        """
        # Check cache first
        cache_key = f"kw_{hash(text[:100]) if len(text) > 100 else hash(text)}"
        if cache_key in self.meta_tag_cache:
            return self.meta_tag_cache[cache_key]
        
        # Try LLM-based extraction if service available
        if self.llm_service:
            try:
                prompt = f"""Extract the top {top_n} most important keywords/concepts from this text.
Focus on:
- Main topics and entities
- Proper nouns (locations, names, organizations)
- Key concepts relevant to understanding the content
- Ignore common words like articles, prepositions, pronouns

Text: "{text}"

Return ONLY a JSON array of keywords (max {top_n}), like: ["keyword1", "keyword2", "keyword3"]
"""
                result = self.llm_service.generate_response(prompt)
                
                # Parse JSON response
                try:
                    keywords = json.loads(result.strip())
                    if isinstance(keywords, list):
                        # Cache and return
                        self.meta_tag_cache[cache_key] = keywords
                        return keywords[:top_n]
                except json.JSONDecodeError:
                    pass  # Fall through to fallback extraction
            
            except Exception as e:
                print(f"[DEBUG] LLM keyword extraction failed: {e}, falling back to regex extraction")
        
        # Fallback: Regex-based keyword extraction (no hardcoded stop words)
        return self._extract_keywords_by_regex(text, top_n)
    
    def _extract_keywords_by_regex(self, text: str, top_n: int = 5) -> List[str]:
        """
        Fallback regex-based keyword extraction without hardcoded stop words.
        Extracts words by frequency and importance.
        """
        # Extract all words (including proper nouns - capitalized words)
        all_words = re.findall(r'\b[a-zA-Z][a-zA-Z\-_]*\b', text)
        
        if not all_words:
            return []
        
        # Simple filtering: 
        # - Keep capitalized words (proper nouns like "Haldia", "Mumbai")
        # - Keep longer words (usually more meaningful)
        # - Count frequency
        
        keywords = []
        for word in all_words:
            # Keep capitalized words regardless of length
            if word[0].isupper():
                keywords.append(word.lower())
            # Keep lowercase words with length >= 3
            elif len(word) >= 3:
                keywords.append(word.lower())
        
        # Get most common keywords by frequency
        if keywords:
            counter = Counter(keywords)
            # Return top N by frequency
            return [word for word, _ in counter.most_common(top_n)]
        
        return []
    
    def extract_meta_tags(self, text: str) -> List[str]:
        """
        Dynamically extract meta tags from text using LLM when available,
        fallback to simple keyword-based extraction.
        """
        # Check cache first
        cache_key = hash(text[:100]) if len(text) > 100 else hash(text)
        if cache_key in self.meta_tag_cache:
            return self.meta_tag_cache[cache_key]
        
        # Try LLM-based extraction if service available
        if self.llm_service:
            try:
                prompt = f"""Analyze the following text and identify its semantic domain/category.
                
Text: "{text}"

Return ONLY a JSON array of relevant categories (max 3) from these examples:
- financial: revenue, profit, expense, budget, accounting
- hr: employee, hiring, recruitment, salary, benefits
- technical: system, database, API, deployment, code
- product: feature, release, version, specification
- process: workflow, procedure, schedule, timeline
- location: city, address, geographic location, region
- operations: logistics, supply chain, inventory
- marketing: campaign, brand, customer, engagement
- compliance: policy, regulation, audit, governance
- general: if none of above

Return as JSON like: ["location", "operations"]
"""
                result = self.llm_service.generate_response(prompt)
                
                # Parse JSON response
                try:
                    tags = json.loads(result.strip())
                    if isinstance(tags, list):
                        # Cache and return
                        self.meta_tag_cache[cache_key] = tags
                        return tags
                except json.JSONDecodeError:
                    pass  # Fall through to keyword-based extraction
            
            except Exception as e:
                print(f"[DEBUG] LLM meta-tag extraction failed: {e}, falling back to keyword extraction")
        
        # Fallback: Simple keyword-based extraction (non-hardcoded)
        return self._extract_tags_by_keywords(text)
    
    def _extract_tags_by_keywords(self, text: str) -> List[str]:
        """Fallback keyword-based meta-tag extraction."""
        text_lower = text.lower()
        
        # Dynamic keyword dictionary - can be easily updated
        keyword_map = {
            'location': ['where', 'location', 'city', 'address', 'region', 'state', 'country', 'place'],
            'financial': ['revenue', 'profit', 'expense', 'cost', 'budget', 'invoice', 'payment'],
            'hr': ['employee', 'hiring', 'recruitment', 'salary', 'benefits', 'leave'],
            'technical': ['system', 'database', 'api', 'code', 'deployment', 'server'],
            'product': ['product', 'feature', 'version', 'release', 'specification'],
            'process': ['process', 'workflow', 'procedure', 'schedule', 'timeline'],
        }
        
        detected_tags = []
        
        for tag, keywords in keyword_map.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_tags.append(tag)
        
        return detected_tags if detected_tags else ['general']
    
    def calculate_keyword_match_score(self, question_keywords: List[str], context_keywords: List[str]) -> float:
        """Calculate keyword overlap score (Jaccard similarity)."""
        if not question_keywords or not context_keywords:
            return 0.0
        
        overlap = len(set(question_keywords) & set(context_keywords))
        union = len(set(question_keywords) | set(context_keywords))
        
        return overlap / union if union > 0 else 0.0
    
    def calculate_meta_tag_confidence(self, question_meta_tags: List[str], context_meta_tags: List[str]) -> float:
        """
        Calculate confidence that context matches question based on meta tags.
        Returns 0-1 confidence score.
        """
        if not question_meta_tags or not context_meta_tags:
            return 0.5
        
        overlap_tags = set(question_meta_tags) & set(context_meta_tags)
        
        if overlap_tags:
            return min(1.0, len(overlap_tags) / max(len(question_meta_tags), len(context_meta_tags)))
        
        if 'general' in context_meta_tags:
            return 0.7
        
        return 0.3
    
    def analyze_keyword_match(self, question: str, context_texts: List[str]) -> Tuple[float, List[str], List[str]]:
        """
        Analyze keyword matching between question and context.
        Returns (match_score, suggested_tags, missing_keywords)
        """
        question_keywords = self.extract_keywords(question)
        question_meta_tags = self.extract_meta_tags(question)
        
        context_combined = " ".join(context_texts)
        context_keywords = self.extract_keywords(context_combined)
        context_meta_tags = self.extract_meta_tags(context_combined)
        
        # Calculate scores
        keyword_score = self.calculate_keyword_match_score(question_keywords, context_keywords)
        meta_confidence = self.calculate_meta_tag_confidence(question_meta_tags, context_meta_tags)
        
        # Find missing keywords
        missing_keywords = list(set(question_keywords) - set(context_keywords))
        
        # DEBUG: Log extracted keywords
        print(f"[DEBUG KEYWORDS] Question: '{question}'")
        print(f"[DEBUG KEYWORDS] Question keywords: {question_keywords}")
        print(f"[DEBUG KEYWORDS] Question meta tags: {question_meta_tags}")
        print(f"[DEBUG KEYWORDS] Context combined length: {len(context_combined)}")
        print(f"[DEBUG KEYWORDS] Context keywords: {context_keywords}")
        print(f"[DEBUG KEYWORDS] Context meta tags: {context_meta_tags}")
        print(f"[DEBUG KEYWORDS] Keyword score: {keyword_score:.3f}")
        print(f"[DEBUG KEYWORDS] Meta confidence: {meta_confidence:.3f}")
        print(f"[DEBUG KEYWORDS] Missing keywords: {missing_keywords}")
        
        # Average score
        avg_score = (keyword_score + meta_confidence) / 2
        
        return avg_score, context_meta_tags, missing_keywords
    
    def _init_db(self):
        """Ensure database tables exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Verify tables exist
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='document_metadata'"
            )
            if not cursor.fetchone():
                raise RuntimeError("Database schema not initialized. Run migration first!")
            
            conn.close()
        except Exception as e:
            raise RuntimeError(f"Database initialization failed: {e}")
    
    def decide_action(self, state: RLState, doc_id: str) -> RLAction:
        """
        Use RL to decide which action to take, considering meta-tag confidence.
        
        Args:
            state: Current system state (includes keyword_match_score and meta_tag_confidence)
            doc_id: Document being evaluated
        
        Returns:
            RLAction with chosen action and parameters
        """
        # Check meta-tag confidence first
        if state.meta_tag_confidence < 0.4:
            # Low confidence - ask user for clarification or suggest tags
            if np.random.random() < 0.6:
                action = 'CLARIFY_USER'
            else:
                action = 'SUGGEST_TAGS'
        elif state.keyword_match_score < 0.3:
            # Keywords don't match well - need clarification
            action = 'CLARIFY_USER'
        else:
            # Normal RL decision
            if np.random.random() < self.epsilon:
                # Explore: randomly choose action
                action = np.random.choice(['SKIP', 'OPTIMIZE', 'REINDEX', 'RE_EMBED'])
            else:
                # Exploit: choose best known action
                action = self._get_best_action(state, doc_id)
        
        # Generate action details
        rl_action = self._generate_action_details(action, state, doc_id)
        
        return rl_action
    
    def _get_best_action(self, state: RLState, doc_id: str) -> str:
        """
        Choose action with highest expected value based on learning history
        and meta-tag confidence scoring.
        """
        # Score each action
        action_scores = {}
        
        for action in ['SKIP', 'OPTIMIZE', 'REINDEX', 'RE_EMBED']:
            stats = self.action_history[action]
            
            if stats['count'] == 0:
                # No data yet, neutral score
                action_scores[action] = 0.5
            else:
                # Q-value approach: average reward adjusted for action cost
                base_reward = stats['avg_reward']
                
                # Meta-tag confidence multiplier (NEW)
                # Low confidence -> boost OPTIMIZE/REINDEX actions
                meta_multiplier = 1.0 - (0.5 * (1.0 - state.meta_tag_confidence))
                
                # Adjust based on state
                if action == 'SKIP':
                    # Only good if quality is high AND meta-tag confidence is high
                    adjustment = 1.0 if state.quality_score > 0.75 and state.meta_tag_confidence > 0.7 else -1.0
                
                elif action == 'OPTIMIZE':
                    # Good if quality is poor or meta confidence is low
                    if (state.quality_score < 0.6 and state.avg_token_cost < 2000) or state.meta_tag_confidence < 0.5:
                        adjustment = 1.5
                    elif state.quality_score < 0.6:
                        adjustment = 0.8
                    else:
                        adjustment = -0.5
                
                elif action == 'REINDEX':
                    # Good if re-indexing hasn't been done much or meta confidence is low
                    if state.reindex_count < 3:
                        adjustment = 1.0 if (state.quality_score < 0.65 or state.meta_tag_confidence < 0.5) else -0.5
                    else:
                        adjustment = -1.0
                
                elif action == 'RE_EMBED':
                    # Good for fresh perspectives when quality is very low
                    if state.quality_score < 0.5 and state.meta_tag_confidence < 0.4:
                        adjustment = 2.5  # Boosted for low confidence
                    elif state.quality_score < 0.5:
                        adjustment = 2.0
                    elif state.avg_token_cost < 1000:
                        adjustment = 0.5
                    else:
                        adjustment = -1.5
                
                action_scores[action] = (base_reward + adjustment) * meta_multiplier
        
        # Choose action with highest score
        best_action = max(action_scores, key=action_scores.get)
        return best_action
    
    def _generate_action_details(self, action: str, state: RLState, doc_id: str) -> RLAction:
        """
        Generate detailed parameters and estimates for the chosen action
        """
        if action == 'SKIP':
            return RLAction(
                action='SKIP',
                params={},
                estimated_improvement=0,
                estimated_cost=0,
                confidence=0.95 if state.quality_score > 0.75 else 0.5
            )
        
        elif action == 'OPTIMIZE':
            # Suggest chunk size optimization
            current_size = 512
            if state.quality_score < 0.6:
                suggested_size = 256  # Smaller chunks for low quality
                improvement = 0.15
                confidence = 0.82
            else:
                suggested_size = 384  # Balance
                improvement = 0.08
                confidence = 0.70
            
            return RLAction(
                action='OPTIMIZE',
                params={
                    'new_chunk_size': suggested_size,
                    'new_overlap': int(suggested_size * 0.1),
                    'strategy': 'recursive_splitter'
                },
                estimated_improvement=improvement,
                estimated_cost=500,  # tokens
                confidence=confidence
            )
        
        elif action == 'REINDEX':
            # Re-index with same parameters
            return RLAction(
                action='REINDEX',
                params={
                    'clear_cache': True,
                    'recompute_embeddings': True
                },
                estimated_improvement=0.12 if state.reindex_count < 2 else 0.05,
                estimated_cost=300,
                confidence=0.75 if state.reindex_count < 2 else 0.55
            )
        
        elif action == 'RE_EMBED':
            # Use different embedding model
            return RLAction(
                action='RE_EMBED',
                params={
                    'new_model': 'mistral',  # Switch from ollama default
                    'preserve_old_embeddings': True
                },
                estimated_improvement=0.25,
                estimated_cost=800,
                confidence=0.68
            )
        
        elif action == 'CLARIFY_USER':
            # NEW: Ask user for clarification when keywords don't match
            return RLAction(
                action='CLARIFY_USER',
                params={
                    'message': 'The retrieved documents do not seem to match your question. Could you please rephrase or provide more details?',
                    'meta_tags': state.meta_tags if state.meta_tags else [],
                    'confidence_level': state.meta_tag_confidence,
                    'keyword_match': state.keyword_match_score
                },
                estimated_improvement=0.3,
                estimated_cost=0,  # No cost - user clarification
                confidence=0.85
            )
        
        elif action == 'SUGGEST_TAGS':
            # NEW: Suggest relevant tags when confidence is low
            return RLAction(
                action='SUGGEST_TAGS',
                params={
                    'message': 'I found some documents, but they may not be exactly what you\'re looking for. Did you mean to ask about:',
                    'suggested_tags': state.meta_tags if state.meta_tags else ['general'],
                    'confidence': state.meta_tag_confidence,
                    'alternative_queries': self._generate_alternative_queries(state.meta_tags)
                },
                estimated_improvement=0.2,
                estimated_cost=0,
                confidence=0.75
            )
    
    def _generate_alternative_queries(self, meta_tags: List[str]) -> List[str]:
        """Generate alternative queries based on detected meta tags."""
        tag_templates = {
            'financial': ['revenue analysis', 'expense reports', 'budget planning', 'financial statements'],
            'hr': ['employee records', 'recruitment process', 'salary management', 'performance review'],
            'technical': ['system architecture', 'database design', 'API documentation', 'deployment guide'],
            'product': ['feature overview', 'product roadmap', 'release notes', 'specifications'],
            'process': ['workflow documentation', 'standard procedures', 'process flow', 'timeline'],
            'general': ['general information', 'overview', 'documentation', 'summary']
        }
        
        queries = []
        for tag in meta_tags if meta_tags else ['general']:
            if tag in tag_templates:
                queries.extend(tag_templates[tag][:2])
        
        return queries[:5]  # Limit to top 5
        """
        Update RL values based on observed reward
        
        Args:
            action: Action that was taken
            actual_reward: Observed reward (quality improvement - cost)
            session_id: Session identifier for tracking
        """
        # Update Q-values
        action_name = action.action
        stats = self.action_history[action_name]
        
        # Update statistics
        stats['count'] += 1
        stats['total_reward'] += actual_reward
        stats['avg_reward'] = stats['total_reward'] / stats['count']
        
        # Decay epsilon (explore less over time)
        self.epsilon = max(0.05, self.epsilon * 0.995)
        
        # Log to database
        self._log_rl_decision(action, actual_reward, session_id)
    
    def _log_rl_decision(self, action: RLAction, reward: float, session_id: str):
        """Log RL decision to database for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            state_json = json.dumps({
                'action': action.action,
                'params': action.params,
                'estimated_improvement': action.estimated_improvement,
                'confidence': action.confidence
            })
            
            context_json = json.dumps({
                'reward_achieved': reward,
                'q_values': self.action_history,
                'epsilon': self.epsilon
            })
            
            cursor.execute("""
                INSERT INTO rag_history_and_optimization
                (event_type, timestamp, action_taken, reward_signal, context_json, agent_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                'HEAL',
                datetime.now().isoformat(),
                action.action,
                reward,
                context_json,
                'rl_healing_agent',
                session_id
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Failed to log RL decision: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get current learning statistics
        """
        total_decisions = sum(stats['count'] for stats in self.action_history.values())
        
        return {
            'total_decisions': total_decisions,
            'epsilon': self.epsilon,
            'actions': {
                action: {
                    'count': stats['count'],
                    'percentage': (stats['count'] / total_decisions * 100) if total_decisions > 0 else 0,
                    'avg_reward': round(stats['avg_reward'], 4),
                    'total_reward': round(stats['total_reward'], 2)
                }
                for action, stats in self.action_history.items()
            },
            'best_action': max(
                self.action_history.items(),
                key=lambda x: x[1]['avg_reward'] if x[1]['count'] > 0 else 0
            )[0] if any(s['count'] > 0 for s in self.action_history.values()) else 'N/A'
        }
    
    def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user feedback and update RL learning
        
        Args:
            feedback_data: Dictionary containing:
                - rating: 1-5 user satisfaction rating
                - question: Original user question
                - answer: Generated answer
                - user_id: User identifier
                - timestamp: Feedback submission time
                - session_id (optional): Session identifier
                - feedback_text (optional): Additional user comments
        
        Returns:
            Dictionary with processing result and learning metrics
        """
        try:
            # Convert rating (1-5) to reward signal (0-1)
            # 1-2: poor (reward 0.0-0.3), 3: neutral (0.5), 4-5: good (0.7-1.0)
            rating = feedback_data.get('rating', 3)
            if rating < 1 or rating > 5:
                rating = 3
            
            reward = (rating - 1) / 4.0  # Converts 1-5 scale to 0-1 scale
            
            # Extract session ID
            session_id = feedback_data.get('session_id', f"feedback_{feedback_data.get('user_id', 'unknown')}_{datetime.now().timestamp()}")
            
            # Log feedback to database
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                feedback_json = json.dumps({
                    'rating': rating,
                    'feedback_text': feedback_data.get('feedback_text', ''),
                    'question': feedback_data.get('question', ''),
                    'answer_length': len(feedback_data.get('answer', '')),
                    'user_id': feedback_data.get('user_id', 'unknown')
                })
                
                cursor.execute("""
                    INSERT INTO rag_history_and_optimization
                    (event_type, timestamp, action_taken, reward_signal, context_json, agent_id, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    'FEEDBACK',
                    datetime.now().isoformat(),
                    'USER_RATING',
                    reward,
                    feedback_json,
                    'rl_healing_agent',
                    session_id
                ))
                
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[FEEDBACK] Warning: Failed to log feedback: {e}")
            
            # Update action history with feedback signal
            if 'USER_RATING' not in self.action_history:
                self.action_history['USER_RATING'] = {
                    'count': 0,
                    'total_reward': 0,
                    'avg_reward': 0
                }
            
            stats = self.action_history['USER_RATING']
            stats['count'] += 1
            stats['total_reward'] += reward
            stats['avg_reward'] = stats['total_reward'] / stats['count']
            
            return {
                'success': True,
                'feedback_id': session_id,
                'rating': rating,
                'reward_signal': round(reward, 4),
                'learning_updated': True,
                'feedback_count': stats['count'],
                'avg_feedback_reward': round(stats['avg_reward'], 4),
                'rl_learning_applied': True
            }
        
        except Exception as e:
            print(f"[FEEDBACK] Error processing feedback: {e}")
            return {
                'success': False,
                'error': str(e),
                'rl_learning_applied': False
            }
    
    def recommend_healing(self, doc_id: str, current_quality: float) -> Dict[str, Any]:
        """
        Get healing recommendation for a specific document
        
        Args:
            doc_id: Document ID
            current_quality: Current quality score (0-1)
        
        Returns:
            Recommendation with action and reasoning
        """
        # Create state
        state = self._build_state_from_db(doc_id, current_quality)
        
        # Get action
        action = self.decide_action(state, doc_id)
        
        # Return recommendation
        return {
            'doc_id': doc_id,
            'current_quality': current_quality,
            'recommended_action': action.action,
            'parameters': action.params,
            'expected_improvement': action.estimated_improvement,
            'estimated_cost': action.estimated_cost,
            'confidence': action.confidence,
            'reasoning': self._generate_reasoning(action, state),
            'learning_stats': self.get_learning_stats()
        }
    
    def _build_state_from_db(self, doc_id: str, current_quality: float) -> RLState:
        """Build state from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get document info
            cursor.execute("""
                SELECT chunk_size_char FROM document_metadata WHERE doc_id = ?
            """, (doc_id,))
            doc_info = cursor.fetchone()
            
            # Get chunk stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as chunk_count,
                    AVG(reindex_count) as avg_reindex
                FROM chunk_embedding_data
                WHERE doc_id = ?
            """, (doc_id,))
            chunk_stats = cursor.fetchone()
            
            # Get query performance
            cursor.execute("""
                SELECT 
                    COUNT(*) as query_count,
                    AVG(CAST(json_extract(metrics_json, '$.avg_accuracy') AS FLOAT)) as avg_accuracy,
                    AVG(CAST(json_extract(metrics_json, '$.cost_tokens') AS FLOAT)) as avg_cost,
                    AVG(CAST(json_extract(metrics_json, '$.user_feedback') AS FLOAT)) as avg_feedback
                FROM rag_history_and_optimization
                WHERE target_doc_id = ? AND event_type = 'QUERY'
            """, (doc_id,))
            query_stats = cursor.fetchone()
            
            conn.close()
            
            return RLState(
                quality_score=current_quality,
                query_accuracy=query_stats[1] or 0.7 if query_stats else 0.7,
                chunk_count=chunk_stats[0] if chunk_stats else 0,
                avg_token_cost=query_stats[2] or 1000 if query_stats else 1000,
                reindex_count=int(chunk_stats[1] or 0) if chunk_stats else 0,
                last_healing_delta=0.1,  # Default
                query_frequency=query_stats[0] if query_stats else 0,
                user_feedback=query_stats[3] or 0.7 if query_stats else 0.7
            )
        except Exception as e:
            # Return default state on error
            return RLState(
                quality_score=current_quality,
                query_accuracy=0.7,
                chunk_count=0,
                avg_token_cost=1000,
                reindex_count=0,
                last_healing_delta=0.1,
                query_frequency=0,
                user_feedback=0.7
            )
    
    def _generate_reasoning(self, action: RLAction, state: RLState) -> str:
        """Generate human-readable reasoning for the action"""
        reasons = {
            'SKIP': "System quality is good. No action needed.",
            'OPTIMIZE': "Quality is below target. Optimizing chunk parameters for better retrieval.",
            'REINDEX': "Regenerating embeddings to refresh semantic understanding.",
            'RE_EMBED': "Switching embedding model for better quality understanding."
        }
        
        return reasons.get(action.action, "Action selected based on learning history.")
    
    def observe_reward(self, action: RLAction, reward: float, session_id: str) -> Dict[str, Any]:
        """
        Observe actual reward from action execution and update learning
        
        Args:
            action: The action that was taken (RLAction)
            reward: Actual reward achieved (0-1 scale)
            session_id: Session identifier for tracking
        
        Returns:
            Dictionary with learning update status
        """
        try:
            # Update action statistics
            if action.action not in self.action_history:
                self.action_history[action.action] = {
                    'count': 0,
                    'total_reward': 0,
                    'avg_reward': 0
                }
            
            stats = self.action_history[action.action]
            stats['count'] += 1
            stats['total_reward'] += reward
            stats['avg_reward'] = stats['total_reward'] / stats['count']
            
            # Decay epsilon (explore less over time)
            self.epsilon = max(0.05, self.epsilon * 0.995)
            
            # Log to database
            self._log_rl_decision(action, reward, session_id)
            
            return {
                'success': True,
                'action': action.action,
                'reward': round(reward, 4),
                'avg_reward': round(stats['avg_reward'], 4),
                'action_count': stats['count'],
                'epsilon': round(self.epsilon, 4)
            }
        except Exception as e:
            print(f"[OBSERVE] Error recording reward: {e}")
            return {
                'success': False,
                'error': str(e)
            }



def example_usage():
    """Example of how to use the RL Healing Agent"""
    from ..config.env_config import EnvConfig
    
    # Initialize agent
    db_path = EnvConfig.get_rag_db_path()
    agent = RLHealingAgent(db_path)
    
    # Get recommendation
    recommendation = agent.recommend_healing(
        doc_id="doc_001",
        current_quality=0.55
    )
    
    print("\n📊 RL Healing Agent Recommendation:")
    print(json.dumps(recommendation, indent=2))
    
    # Simulate action and observe reward
    action = RLAction(
        action=recommendation['recommended_action'],
        params=recommendation['parameters'],
        estimated_improvement=recommendation['expected_improvement'],
        estimated_cost=recommendation['estimated_cost'],
        confidence=recommendation['confidence']
    )
    
    # Observe actual reward (in real system, this comes from actual healing results)
    actual_reward = 0.12  # Achieved 12% improvement
    agent.observe_reward(action, actual_reward, session_id="session_123")
    
    # Check learning progress
    stats = agent.get_learning_stats()
    print("\n📈 RL Agent Learning Stats:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    example_usage()
