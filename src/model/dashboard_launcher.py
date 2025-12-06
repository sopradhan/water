#!/usr/bin/env python3
"""
Dashboard Launcher
Starts the Streamlit dashboard for data exploration and model retraining
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Launch the dashboard"""
    
    logger.info("=" * 70)
    logger.info("[START] Water Anomaly Detection - Advanced Dashboard")
    logger.info("=" * 70)
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    dashboard_script = project_root / "src" / "model" / "dashboard.py"
    
    # Check if dashboard script exists
    if not dashboard_script.exists():
        logger.error(f"[ERROR] Dashboard script not found at {dashboard_script}")
        sys.exit(1)
    
    logger.info(f"[INFO] Dashboard script: {dashboard_script}")
    logger.info("[INFO] Starting Streamlit application...")
    logger.info("=" * 70)
    logger.info("[URL] Dashboard: http://localhost:8501")
    logger.info("[URL] API Documentation: http://localhost:8501/docs")
    logger.info("=" * 70)
    
    # Change to project root
    os.chdir(project_root)
    
    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_script),
            "--logger.level=info",
            "--client.showErrorDetails=true"
        ], check=False)
    except KeyboardInterrupt:
        logger.info("[STOP] Dashboard stopped by user")
    except Exception as e:
        logger.error(f"[ERROR] Failed to start dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
