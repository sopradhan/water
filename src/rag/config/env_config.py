import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class EnvConfig:
    """Load configuration for Water Anomaly Detection RAG System
    
    Database Strategy:
    - SQLite DB: Local metadata storage at src/data/RAG/rag_metadata.db
      Used for: Document tracking, query logs, agent memory, audit trails
    - ChromaDB: Local vector database at src/data/RAG/chroma_db
      Used for: Semantic embeddings and similarity search
    """
    
    @staticmethod
    def _get_project_root() -> Path:
        """Get absolute path to project root directory"""
        # From rag/config/env_config.py -> rag/ -> src/ -> project_root/
        current_file = Path(__file__).resolve()
        return current_file.parent.parent.parent.parent
    
    @staticmethod
    def get_rag_db_path() -> str:
        """Get SQLite database path for RAG metadata storage"""
        # Use absolute path from project root
        path = os.getenv('RAG_DB_PATH', 'src/data/RAG/rag_metadata.db')
        if not os.path.isabs(path):
            project_root = EnvConfig._get_project_root()
            path = str(project_root / path)
        return path
    
    @staticmethod
    def get_chroma_db_path() -> str:
        """Get ChromaDB persistence path for local vector database"""
        # Use absolute path from project root
        path = os.getenv('CHROMA_DB_PATH', 'src/data/RAG/chroma_db')
        if not os.path.isabs(path):
            project_root = EnvConfig._get_project_root()
            path = str(project_root / path)
        return path
    
    @staticmethod
    def get_rag_config_path() -> str:
        """Get RAG configuration directory path"""
        # Use absolute path from project root to prevent working directory confusion
        path = os.getenv('RAG_CONFIG_PATH', 'src/rag/config')
        if not os.path.isabs(path):
            project_root = EnvConfig._get_project_root()
            path = str(project_root / path)
        return path
    
    @staticmethod
    def get_app_env() -> str:
        """Get application environment (development, staging, production)"""
        return os.getenv('APP_ENV', 'development')
    
    @staticmethod
    def get_log_level() -> str:
        """Get logging level"""
        return os.getenv('LOG_LEVEL', 'info')
        """Get logging level"""
        return os.getenv('LOG_LEVEL', 'info')
