"""Verify RAG database setup."""

import sqlite3
from pathlib import Path

db_path = Path("src/data/RAG/rag_metadata.db")

if db_path.exists():
    print(f"\n[OK] Database file found: {db_path.absolute()}\n")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("=" * 60)
    print("TABLES CREATED:")
    print("=" * 60)
    for table in tables:
        table_name = table[0]
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"[OK] {table_name:<40} ({count} rows)")
    
    print("\n" + "=" * 60)
    print("SCHEMA DETAILS:")
    print("=" * 60)
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"\n{table_name}:")
        for col in columns:
            col_id, col_name, col_type, notnull, default, pk = col
            null_str = "NOT NULL" if notnull else "NULL"
            print(f"  - {col_name:<30} {col_type:<15} {null_str}")
    
    # Sample data from document_metadata
    print("\n" + "=" * 60)
    print("SAMPLE DATA - document_metadata:")
    print("=" * 60)
    cursor.execute("SELECT doc_id, title, author FROM document_metadata LIMIT 5")
    for row in cursor.fetchall():
        print(f"  • {row[0]}: {row[1]} (by {row[2]})")
    
    # Sample data from agent_memory
    print("\n" + "=" * 60)
    print("SAMPLE DATA - agent_memory:")
    print("=" * 60)
    cursor.execute("SELECT agent_id, memory_type, memory_key FROM agent_memory LIMIT 5")
    for row in cursor.fetchall():
        print(f"  • {row[0]} | {row[1]}: {row[2]}")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("[OK] RAG DATABASE VERIFICATION SUCCESSFUL")
    print("=" * 60 + "\n")
else:
    print(f"\n[FAIL] Database not found: {db_path}")
