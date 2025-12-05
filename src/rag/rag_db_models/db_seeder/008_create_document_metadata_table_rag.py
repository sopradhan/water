"""Create document_metadata pivot  table for RAG """


def run(conn):
    conn.execute('''
    CREATE TABLE IF NOT EXISTS document_metadata (
    doc_id TEXT PRIMARY KEY,
    
    -- Document Content & Identification
    title TEXT NOT NULL,
    author TEXT,
    source TEXT,
    summary TEXT,
    -- Ownership & RBAC
    company_id INTEGER,
    dept_id INTEGER,
    rbac_namespace TEXT NOT NULL DEFAULT 'general', 
    -- Chunking Strategy (document-level defaults)
    chunk_strategy TEXT NOT NULL DEFAULT 'recursive_splitter',
    chunk_size_char INTEGER NOT NULL DEFAULT 512,
    overlap_char INTEGER NOT NULL DEFAULT 50, 
    -- Consolidated Metadata (sparse/optional fields as JSON)
    -- Stores: categories, keywords, doc_type, created_date, tags, etc.
    metadata_json TEXT, 
    -- Tracking
    last_ingested TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Foreign Keys
    FOREIGN KEY(company_id) REFERENCES companies(id),
    FOREIGN KEY(dept_id) REFERENCES departments(id),
    -- Indexes for performance
    CONSTRAINT valid_chunk_size CHECK (chunk_size_char > 0),
    CONSTRAINT valid_overlap CHECK (overlap_char >= 0)
);
    ''')
    conn.commit()
