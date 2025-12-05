"""
Custom Guardrails Service for RAG System

Simple but effective guardrails without external dependencies (like guardrails-ai).
Implements pattern-based validation for:
- Input validation (harmful patterns, length limits)
- Output safety checks (PII, repetition, empty responses)
- Response filtering (sensitive data redaction)

Inspired by: https://github.com/roy-souvik/agent-zero/blob/guardrails/agent/pages/rag_qa.py
"""

import re
from typing import Dict, Optional, Tuple, List
from enum import Enum


class SafetyLevel(Enum):
    """Safety levels for responses."""
    SAFE = "safe"
    WARNING = "warning"
    BLOCKED = "blocked"


class CustomGuardrails:
    """Simple guardrails for RAG system without external dependencies."""
    
    def __init__(self):
        """Initialize guardrails with patterns and thresholds."""
        # Harmful patterns to detect in input
        self.harmful_patterns = [
            r'(?i)(password|secret|api.?key|token|credential)',
            r'(?i)(execute|eval|exec|system|__import__|compile)',
            r'(?i)(drop\s+table|delete\s+from|insert\s+into|update\s+)',
            r'(?i)(select\s+\*|union\s+select)',
        ]
        
        # Blocked keywords
        self.blocked_keywords = [
            'violence', 'harassment', 'illegal', 'exploit',
            'malware', 'ransomware', 'backdoor', 'trojan'
        ]
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            'ssn': r'\b(?!000|666)[0-9]{3}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b',
            'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
            'api_key': r'(?i)(api[_-]?key|apikey)\s*[:=]\s*([a-zA-Z0-9_\-]{20,})',
            'password': r'(?i)(password|passwd|pwd)\s*[:=]\s*[^\s,}]+'
        }
        
        # Thresholds
        self.max_input_length = 10000
        self.max_output_length = 5000
        self.repetition_threshold = 0.2  # 20% unique words minimum
    
    def validate_input(self, user_input: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user input for safety and appropriateness.
        
        Args:
            user_input: User's input text
        
        Returns:
            (is_valid, error_message)
        """
        # Check input length
        if len(user_input) > self.max_input_length:
            return False, f"Input exceeds maximum length of {self.max_input_length}"
        
        # Check for empty input
        if not user_input.strip():
            return False, "Input cannot be empty"
        
        # Check for harmful patterns
        for pattern in self.harmful_patterns:
            if re.search(pattern, user_input):
                return False, "Input contains potentially harmful content"
        
        # Check for blocked keywords
        input_lower = user_input.lower()
        for keyword in self.blocked_keywords:
            if keyword in input_lower:
                return False, f"Input contains blocked keyword: '{keyword}'"
        
        return True, None
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Detect PII patterns in text.
        
        Args:
            text: Text to check for PII
        
        Returns:
            Dictionary with PII type -> list of matches
        """
        findings = {}
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[pii_type] = matches if isinstance(matches[0], str) and len(matches[0]) > 5 else [str(m) for m in matches]
        
        return findings
    
    def check_output_safety(self, output: str) -> Tuple[bool, Optional[str]]:
        """
        Verify output safety before returning to user.
        
        Args:
            output: LLM generated output
        
        Returns:
            (is_safe, error_message)
        """
        if not output or not output.strip():
            return False, "Model generated empty output"
        
        # Check output length
        if len(output) > self.max_output_length:
            # Don't fail, just warn - truncate will happen later
            pass
        
        # Check for repetition (could indicate loop or error)
        words = output.split()
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words > 10:  # Only check if enough content
            unique_ratio = unique_words / total_words
            if unique_ratio < self.repetition_threshold:
                return False, f"Output contains excessive repetition (only {unique_ratio*100:.1f}% unique words)"
        
        # Check for harmful keywords in output
        output_lower = output.lower()
        for keyword in self.blocked_keywords:
            if keyword in output_lower:
                return False, f"Output contains blocked keyword: '{keyword}'"
        
        return True, None
    
    def filter_output(self, output: str) -> str:
        """
        Post-process LLM output for safety.
        
        Args:
            output: Raw LLM output
        
        Returns:
            Filtered output
        """
        # Truncate if too long
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length] + "\n\n[Response truncated for length]"
        
        # Redact sensitive patterns
        output = re.sub(
            r'(password|api.?key|secret|token)\s*[:=]\s*[^\s,\n]+',
            r'\1: [REDACTED]',
            output,
            flags=re.IGNORECASE
        )
        
        # Redact PII patterns
        output = re.sub(self.pii_patterns['email'], '[EMAIL_REDACTED]', output)
        output = re.sub(self.pii_patterns['phone'], '[PHONE_REDACTED]', output)
        output = re.sub(self.pii_patterns['ssn'], '[SSN_REDACTED]', output)
        output = re.sub(self.pii_patterns['credit_card'], '[CC_REDACTED]', output)
        
        return output
    
    def process_request(self, user_input: str, llm_output: str) -> Dict:
        """
        Complete guardrails pipeline: validate input -> check safety -> filter output.
        
        Args:
            user_input: Original user query
            llm_output: Generated response from LLM
        
        Returns:
            Dict with keys:
            - success (bool): Overall result
            - is_safe (bool): Safety check result
            - safety_level (str): safe|warning|blocked
            - input_errors (list): Input validation errors
            - output_errors (list): Output validation errors
            - pii_detected (dict): PII patterns found
            - filtered_output (str): Safe to return to user
            - message (str): Human-readable summary
        """
        result = {
            'success': True,
            'is_safe': True,
            'safety_level': 'safe',
            'input_errors': [],
            'output_errors': [],
            'pii_detected': {},
            'filtered_output': None,
            'message': None
        }
        
        # Step 1: Validate input
        is_valid, error = self.validate_input(user_input)
        if not is_valid:
            result['success'] = False
            result['is_safe'] = False
            result['safety_level'] = 'blocked'
            result['input_errors'] = [error]
            result['message'] = f"Input validation failed: {error}"
            return result
        
        # Step 2: Check output safety
        is_safe, error = self.check_output_safety(llm_output)
        if not is_safe:
            result['success'] = False
            result['is_safe'] = False
            result['safety_level'] = 'blocked'
            result['output_errors'] = [error]
            result['message'] = f"Output validation failed: {error}"
            return result
        
        # Step 3: Detect PII
        pii_found = self.detect_pii(llm_output)
        if pii_found:
            result['is_safe'] = False
            result['safety_level'] = 'warning'
            result['pii_detected'] = pii_found
            result['message'] = f"PII detected: {list(pii_found.keys())}"
        
        # Step 4: Filter output for safety
        filtered_output = self.filter_output(llm_output)
        result['filtered_output'] = filtered_output
        
        # Final message
        if result['pii_detected']:
            result['message'] = f"Response filtered (PII redacted: {list(result['pii_detected'].keys())})"
        else:
            result['message'] = "Response validated and safe"
        
        return result
    
    def get_safety_report(self) -> Dict:
        """Get a report of guardrails configuration."""
        return {
            'max_input_length': self.max_input_length,
            'max_output_length': self.max_output_length,
            'harmful_patterns': len(self.harmful_patterns),
            'blocked_keywords': self.blocked_keywords,
            'pii_detectors': list(self.pii_patterns.keys()),
            'repetition_threshold': f"{self.repetition_threshold*100:.0f}%"
        }
