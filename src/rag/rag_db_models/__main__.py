"""Entry point for RAG database setup when run as module.

Usage:
    python -m src.rag.rag_db_models  [options]
    python -m src.rag.rag_db_models.db_setup  [options]

Options:
    --migrations-only    Run only migrations (create tables)
    --seed-only         Run only seeders (populate initial data)
    --reset             Drop and recreate all tables
    --db-path PATH      Custom database path
"""

from .db_setup import main
import sys

if __name__ == "__main__":
    sys.exit(main())
