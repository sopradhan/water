"""
RAG Complete Stack Launcher
============================

Launches both the RAG API server and Streamlit dashboard in one command.

Usage:
    python -m src.rag.api.startup              # Start both API and Streamlit
    python -m src.rag.api.startup --api-only   # Start only API server
    python -m src.rag.api.startup --ui-only    # Start only Streamlit dashboard

Environment:
    RAG_API_URL: API server URL (default: http://localhost:8001)
    STREAMLIT_PORT: Streamlit port (default: 8501)
"""

import os
import sys
import subprocess
import time
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def start_api_server(port=8001, reload=False):
    """Start RAG API server in subprocess"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("[START] Starting RAG API Server...")
    logger.info("=" * 80)
    logger.info(f"API will be available at: http://localhost:{port}")
    logger.info(f"API Docs: http://localhost:{port}/docs")
    logger.info("")
    
    cmd = [
        sys.executable,
        "-m", "src.rag.api.launcher",
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        return process
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {str(e)}")
        return None


def start_streamlit_dashboard(api_url="http://localhost:8001", streamlit_port=8501):
    """Start Streamlit dashboard in subprocess"""
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎨 Starting Streamlit Dashboard...")
    logger.info("=" * 80)
    logger.info(f"Dashboard will be available at: http://localhost:{streamlit_port}")
    logger.info(f"API URL configured as: {api_url}")
    logger.info("")
    
    # Set environment variables for Streamlit
    env = os.environ.copy()
    env["RAG_API_URL"] = api_url
    
    streamlit_app = project_root / "src" / "rag" / "pages" / "streamlit_app.py"
    
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m", "streamlit",
                "run", str(streamlit_app),
                "--server.port", str(streamlit_port),
                "--server.headless", "false",
                "--logger.level", "info"
            ],
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        return process
    except Exception as e:
        logger.error(f"❌ Failed to start Streamlit dashboard: {str(e)}")
        return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RAG Complete Stack Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start both API and Streamlit
  python -m src.rag.api.startup
  
  # Start only API server
  python -m src.rag.api.startup --api-only
  
  # Start only Streamlit dashboard
  python -m src.rag.api.startup --ui-only
  
  # Start API with auto-reload
  python -m src.rag.api.startup --reload
        """
    )
    
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start only the API server"
    )
    
    parser.add_argument(
        "--ui-only",
        action="store_true",
        help="Start only the Streamlit dashboard"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for API server (development)"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8001,
        help="API server port (default: 8001)"
    )
    
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8501,
        help="Streamlit port (default: 8501)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    processes = []
    
    try:
        logger.info("")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " RAG Complete Stack Launcher ".center(78) + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("")
        
        # Start API server unless UI-only mode
        if not args.ui_only:
            api_process = start_api_server(port=args.api_port, reload=args.reload)
            if api_process:
                processes.append(api_process)
            else:
                logger.error("Failed to start API server")
                return 1
        
        # Start Streamlit unless API-only mode
        if not args.api_only:
            time.sleep(2)  # Wait for API to fully start
            api_url = f"http://localhost:{args.api_port}"
            ui_process = start_streamlit_dashboard(
                api_url=api_url,
                streamlit_port=args.ui_port
            )
            if ui_process:
                processes.append(ui_process)
            else:
                logger.error("Failed to start Streamlit dashboard")
                return 1
        
        # Display startup summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ All services started successfully!")
        logger.info("=" * 80)
        
        if not args.ui_only:
            logger.info(f"📡 API Server: http://localhost:{args.api_port}")
            logger.info(f"   Docs: http://localhost:{args.api_port}/docs")
            logger.info(f"   Health: http://localhost:{args.api_port}/health")
        
        if not args.api_only:
            logger.info(f"🎨 Dashboard: http://localhost:{args.ui_port}")
        
        logger.info("")
        logger.info("Press Ctrl+C to stop all services")
        logger.info("=" * 80)
        logger.info("")
        
        # Wait for all processes
        for process in processes:
            process.wait()
    
    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("⏹️  Shutting down all services...")
        logger.info("=" * 80)
        
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        logger.info("✅ All services stopped")
        return 0
    
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        for process in processes:
            try:
                process.terminate()
            except:
                pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
