"""
Unified RAG API Server Launcher
================================

This script launches the LangGraph RAG Agent API server.

Usage:
    python -m src.rag.api.launcher                 # Start LangGraph API on port 8001
    python -m src.rag.api.launcher --port 8000     # Custom port
    python -m src.rag.api.launcher --reload         # With auto-reload (development)

Environment:
    RAG_HOST: Server host (default: 0.0.0.0)
    RAG_PORT: Server port (default: 8001)
    APP_ENV: Environment type (development/staging/production)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Set TensorFlow optimization flag BEFORE any imports that might use it
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from dotenv import load_dotenv

# Load environment variables
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_langgraph(host: str = "0.0.0.0", 
                  port: int = 8001, 
                  reload: bool = False,
                  log_level: str = "info"):
    """Run LangGraph RAG Agent API"""
    from src.rag.agents.langgraph_agent.api import app
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("🚀 Starting LangGraph RAG Agent API")
    logger.info("=" * 80)
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"URL: http://localhost:{port}")
    logger.info("")
    logger.info("📚 API Documentation:")
    logger.info(f"  - Swagger UI: http://localhost:{port}/docs")
    logger.info(f"  - ReDoc: http://localhost:{port}/redoc")
    logger.info(f"  - OpenAPI Schema: http://localhost:{port}/openapi.json")
    logger.info("")
    logger.info("📡 Main Endpoints:")
    logger.info(f"  - Health Check: GET http://localhost:{port}/health")
    logger.info(f"  - Ingest: POST http://localhost:{port}/ingest")
    logger.info(f"  - Ask: POST http://localhost:{port}/ask")
    logger.info(f"  - VectorDB Stats: GET http://localhost:{port}/vectordb/stats")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            reload_dirs=["src/rag"] if reload else None,
            log_level=log_level,
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("\n⏹️  Server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Error starting server: {str(e)}")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LangGraph RAG Agent API Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings (0.0.0.0:8001)
  python -m src.rag.api.launcher
  
  # Start with auto-reload (development)
  python -m src.rag.api.launcher --reload
  
  # Start on custom port
  python -m src.rag.api.launcher --port 8000
  
  # Start with specific host and port
  python -m src.rag.api.launcher --host 127.0.0.1 --port 8001
  
  # Debug mode
  python -m src.rag.api.launcher --log-level debug
        """
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("RAG_HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("RAG_PORT", 8001)),
        help="Server port (default: 8001)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload on code changes (development only)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("RAG_LOG_LEVEL", "info"),
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level (default: info)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Verify dependencies
    try:
        import fastapi
        import uvicorn
        import chromadb
        import langchain
        import langgraph
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {str(e)}")
        logger.error("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Verify RAG setup
    api_file = project_root / "src" / "rag" / "agents" / "langgraph_agent" / "api.py"
    if not api_file.exists():
        logger.error(f"❌ API file not found: {api_file}")
        sys.exit(1)
    
    # Run LangGraph API
    run_langgraph(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
