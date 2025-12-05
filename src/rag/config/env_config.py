import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class EnvConfig:
    """Load configuration from environment variables
    
    Database Strategy:
    - KB_DB_PATH / connection_string_env in data_sources.json: Knowledge base database (incident_iq.db) 
      READ ONLY during ingestion - used to read: incidents, knowledge_base, resource tables
      NOT used for: metadata storage, embedding storage
    - DB_PATH (rag.db): Internal RAG metadata and embeddings storage
      Used for: query logs, document tracking, agent memory
      NOT used for: reading source data
    - CHROMA_DB_PATH: Vector embeddings store
    """
    
    @staticmethod
    def _get_project_root() -> Path:
        """Get absolute path to project root directory"""
        # From rag/config/env_config.py -> rag/ -> src/ -> project_root/
        current_file = Path(__file__).resolve()
        return current_file.parent.parent.parent.parent
    
    @staticmethod
    def get_kb_db_path() -> str:
        """Get knowledge base database path (for reading source data during ingestion)
        Default: reads from data_sources.json connection_string_env or uses incident_iq.db
        """
        return os.getenv('KB_DB_PATH', 'incident_iq.db')
    
    @staticmethod
    def get_db_path() -> str:
        """Get RAG metadata database path (for internal metadata, embeddings tracking)"""
        # Use absolute path from project root to prevent working directory confusion
        path = os.getenv('DB_PATH', 'src/database/data/incident_iq.db')
        if not os.path.isabs(path):
            project_root = EnvConfig._get_project_root()
            path = str(project_root / path)
        return path
    
    @staticmethod
    def get_chroma_db_path() -> str:
        """Get ChromaDB vector store path"""
        # Fetch path directly from EnvConfig
        path = EnvConfig.get_chroma_db_path()
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
