#!/usr/bin/env python3
import sqlite3

db_path = 'src/data/RAG/rag_metadata.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()

print(f"Tables in {db_path}:")
print(f"Total: {len(tables)}")
for table in tables:
    print(f"  - {table[0]}")

conn.close()

if len(tables) == 0:
    print("\n⚠ No tables found! Running initialization...")
    from src.rag.rag_db_models.db_setup import initialize_database
    result = initialize_database()
    print(f"Initialization result: {result}")
    
    # Check again
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"\nTables after initialization: {len(tables)}")
    for table in tables:
        print(f"  - {table[0]}")
    conn.close()
