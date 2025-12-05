"""RAG Configuration module."""

from .env_config import EnvConfig
from .prompt_loader import PromptLoader

# Backward compatibility aliases
RAGConfig = EnvConfig
get_rag_config = lambda: EnvConfig()

__all__ = ["EnvConfig", "RAGConfig", "get_rag_config", "PromptLoader"]
