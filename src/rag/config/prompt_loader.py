"""
PromptLoader: Centralized prompt management for LLM interactions.

This module loads prompts from prompts.json configuration file,
allowing easy maintenance and modification of LLM prompts without
changing code.
"""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class PromptLoader:
    """Load and manage LLM prompts from JSON configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize PromptLoader with prompts configuration.
        
        Args:
            config_path: Path to prompts.json. If None, searches in standard locations.
        """
        self.config_path = config_path or self._find_prompts_config()
        self.prompts = self._load_prompts()
    
    def _find_prompts_config(self) -> str:
        """Find prompts.json in standard locations."""
        # Try relative to this file
        current_dir = Path(__file__).parent
        candidates = [
            current_dir / "prompts.json",
            current_dir.parent.parent / "config" / "prompts.json",
            Path("src/rag/config/prompts.json"),
            Path("config/prompts.json"),
        ]
        
        for path in candidates:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            f"Could not find prompts.json. Tried: {[str(p) for p in candidates]}"
        )
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from JSON configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts configuration not found at {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts.json: {e}")
    
    def get_prompt(self, category: str, prompt_key: str) -> str:
        """
        Get a prompt template by category and key.
        
        Args:
            category: Category of prompt (e.g., 'extraction', 'intent', 'answer_generation')
            prompt_key: Key within category (e.g., 'document_keywords', 'user_intent_keywords')
        
        Returns:
            Template string that can be formatted with variables
        """
        try:
            return self.prompts[category][prompt_key]["template"]
        except KeyError as e:
            raise KeyError(
                f"Prompt not found: {category}.{prompt_key}. "
                f"Available categories: {list(self.prompts.keys())}"
            )
    
    def get_prompt_info(self, category: str, prompt_key: str) -> Dict[str, Any]:
        """
        Get complete prompt information including template, description, and metadata.
        
        Args:
            category: Category of prompt
            prompt_key: Key within category
        
        Returns:
            Dict with 'template', 'description', 'required_vars', 'output_format'
        """
        try:
            return self.prompts[category][prompt_key]
        except KeyError as e:
            raise KeyError(f"Prompt not found: {category}.{prompt_key}")
    
    def format_prompt(self, category: str, prompt_key: str, **variables) -> str:
        """
        Get a prompt template and format it with variables.
        
        Args:
            category: Category of prompt
            prompt_key: Key within category
            **variables: Variables to substitute in template
        
        Returns:
            Formatted prompt string
        """
        template = self.get_prompt(category, prompt_key)
        prompt_info = self.get_prompt_info(category, prompt_key)
        
        # Validate required variables
        required = prompt_info.get("required_vars", [])
        missing = [v for v in required if v not in variables]
        if missing:
            raise ValueError(f"Missing required variables for {category}.{prompt_key}: {missing}")
        
        return template.format(**variables)
    
    def get_system_prompt(self, prompt_key: str) -> str:
        """
        Get a system prompt by key.
        
        Args:
            prompt_key: Key in system_prompts (e.g., 'ingestion_specialist')
        
        Returns:
            System prompt template
        """
        return self.get_prompt("system_prompts", prompt_key)
    
    def list_categories(self) -> list:
        """Return list of available prompt categories."""
        return list(self.prompts.keys())
    
    def list_prompts_in_category(self, category: str) -> list:
        """Return list of prompt keys in a category."""
        if category not in self.prompts:
            raise KeyError(f"Category not found: {category}")
        return list(self.prompts[category].keys())
    
    def get_all_prompt_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all prompts (useful for documentation/debugging)."""
        metadata = {}
        for category, prompts_dict in self.prompts.items():
            metadata[category] = {}
            for key, info in prompts_dict.items():
                metadata[category][key] = {
                    "description": info.get("description", ""),
                    "required_vars": info.get("required_vars", []),
                    "output_format": info.get("output_format", "")
                }
        return metadata


# Singleton instance for global access
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """
    Get or create the global PromptLoader instance.
    
    Returns:
        Global PromptLoader instance
    """
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


def load_prompt(category: str, prompt_key: str, **variables) -> str:
    """
    Convenience function to load and format a prompt in one call.
    
    Args:
        category: Category of prompt
        prompt_key: Key within category
        **variables: Variables to substitute in template
    
    Returns:
        Formatted prompt string
    """
    loader = get_prompt_loader()
    return loader.format_prompt(category, prompt_key, **variables)


def get_system_prompt(prompt_key: str) -> str:
    """
    Convenience function to get a system prompt.
    
    Args:
        prompt_key: Key in system_prompts
    
    Returns:
        System prompt
    """
    loader = get_prompt_loader()
    return loader.get_system_prompt(prompt_key)
