"""
Vector Database Service - ChromaDB abstraction
Provides unified interface for vector operations with configurable backend
"""
import chromadb
from chromadb import PersistentClient, HttpClient
from typing import List, Dict, Any, Optional
from pathlib import Path
import os


class VectorDBService:
    """ChromaDB vector database service for Water Anomaly Detection RAG
    
    Supports two modes:
    1. Local mode: ChromaDB persistence directory at src/data/RAG/chroma_db
    2. Server mode: Remote ChromaDB server via HTTP (for external hosting)
    """
    
    def __init__(self, persist_directory: Optional[str] = None, 
                 collection_name: str = "water_anomaly_detection",
                 server_url: Optional[str] = None):
        """
        Initialize vector database service
        
        Args:
            persist_directory: Path to ChromaDB persistence directory (local mode)
                             If None and server_url is None, uses src/data/RAG/chroma_db
            collection_name: Name of the collection to use
            server_url: URL for remote ChromaDB server (e.g., http://localhost:8000)
                       If provided, connects to server instead of local persistence
        """
        # Determine which mode to use
        if server_url:
            # Server mode: Connect to remote ChromaDB server
            self._init_server_mode(server_url, collection_name)
        else:
            # Local mode: Use ChromaDB persistence directory
            if persist_directory is None:
                # Use default RAG directory
                persist_directory = self._get_default_persist_dir()
            self._init_local_mode(persist_directory, collection_name)
        
        self.collection_name = collection_name
        print(f"[VectorDBService] Collection '{collection_name}' ready")
        print(f"[VectorDBService] Current document count: {self.collection.count()}")
    
    @staticmethod
    def _get_default_persist_dir() -> str:
        """Get default persistence directory from environment or use src/data/RAG"""
        # Try to get from environment variable first
        persist_dir = os.getenv('CHROMA_DB_PATH')
        if persist_dir:
            return persist_dir
        
        # Fall back to default: src/data/RAG/chroma_db
        # Get project root and construct path
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent.parent
        default_path = str(project_root / "src" / "data" / "RAG" / "chroma_db")
        return default_path
    
    def _init_local_mode(self, persist_directory: str, collection_name: str):
        """Initialize ChromaDB in local persistence mode"""
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        self.persist_directory = persist_directory
        self.server_url = None
        self.mode = "local"
        
        # Initialize ChromaDB with persistence
        self.client = PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"[VectorDBService] Local mode - Persistence directory: {persist_directory}")
    
    def _init_server_mode(self, server_url: str, collection_name: str):
        """Initialize ChromaDB in server mode (connect to remote instance)"""
        self.persist_directory = None
        self.server_url = server_url
        self.mode = "server"
        
        # Connect to remote ChromaDB server
        try:
            self.client = HttpClient(host=server_url.split("://")[-1].split(":")[0],
                                     port=int(server_url.split(":")[-1]) if ":" in server_url else 8000)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[VectorDBService] Server mode - Connected to: {server_url}")
        except Exception as e:
            print(f"[ERROR] Failed to connect to server: {server_url}")
            print(f"[ERROR] {e}")
            raise
    
    def add_documents(self, ids: List[str], metadatas: List[Dict], documents: List[str],
                     embeddings: Optional[List[List[float]]] = None):
        """
        Add documents to vector database (with or without embeddings)
        
        Args:
            ids: List of unique IDs for chunks
            metadatas: List of metadata dicts
            documents: List of text content
            embeddings: Optional list of embedding vectors (if None, ChromaDB auto-embeds)
        """
        try:
            if embeddings is not None:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            else:
                # Let ChromaDB auto-embed the documents
                self.collection.add(
                    ids=ids,
                    metadatas=metadatas,
                    documents=documents
                )
            print(f"[VectorDB] Added {len(ids)} documents {'with' if embeddings else 'without'} embeddings")
        except Exception as e:
            print(f"[ERROR] Failed to add documents: {e}")
            raise
    
  
    
    def search(self, query_embedding: List[float], top_k: int = 10,
              where_filter: Optional[Dict] = None,
              where_document: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            where_filter: Metadata filter
            where_document: Document content filter
            
        Returns:
            Dictionary with ids, distances, metadatas, documents
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                where_document=where_document
            )
            return results
        except Exception as e:
            print(f"[ERROR] Failed to search vectors: {e}")
            return {'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]}
    
    def search_by_text(self, query_text: str, top_k: int = 10,
                      embedding_fn = None,
                      where_filter: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search using text query (requires embedding function)
        
        Args:
            query_text: Query text
            top_k: Number of results
            embedding_fn: Function to generate embeddings
            where_filter: Metadata filter
            
        Returns:
            Search results
        """
        if embedding_fn is None:
            raise ValueError("embedding_fn required for text search")
        
        query_embedding = embedding_fn(query_text)
        return self.search(query_embedding, top_k, where_filter)
    
    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve documents by IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            Documents and metadata
        """
        try:
            return self.collection.get(ids=ids)
        except Exception as e:
            print(f"[ERROR] Failed to get documents: {e}")
            return {'ids': [], 'metadatas': [], 'documents': []}
    
    def delete_by_ids(self, ids: List[str]):
        """
        Delete documents by IDs
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            print(f"[VectorDB] Deleted {len(ids)} documents")
        except Exception as e:
            print(f"[ERROR] Failed to delete documents: {e}")
    
    def delete_by_document(self, doc_id: str):
        """
        Delete all chunks for a document
        
        Args:
            doc_id: Document ID
        """
        try:
            self.collection.delete(where={"document_id": doc_id})
            print(f"[VectorDB] Deleted all chunks for document: {doc_id}")
        except Exception as e:
            print(f"[ERROR] Failed to delete document chunks: {e}")
    
    def delete_by_filter(self, where_filter: Dict):
        """
        Delete documents matching filter
        
        Args:
            where_filter: Metadata filter
        """
        try:
            self.collection.delete(where=where_filter)
            print(f"[VectorDB] Deleted documents matching filter")
        except Exception as e:
            print(f"[ERROR] Failed to delete by filter: {e}")
    
    def update_metadata(self, ids: List[str], metadatas: List[Dict]):
        """
        Update metadata for existing documents
        
        Args:
            ids: List of document IDs
            metadatas: List of new metadata dicts
        """
        try:
            self.collection.update(ids=ids, metadatas=metadatas)
            print(f"[VectorDB] Updated metadata for {len(ids)} documents")
        except Exception as e:
            print(f"[ERROR] Failed to update metadata: {e}")
    
    def count(self) -> int:
        """Get total number of documents in collection"""
        return self.collection.count()
    
    def peek(self, limit: int = 10) -> Dict[str, Any]:
        """
        Peek at first N documents
        
        Args:
            limit: Number of documents to return
            
        Returns:
            Sample documents
        """
        return self.collection.peek(limit=limit)
    
    def reset_collection(self):
        """Delete all documents in collection (use with caution!)"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[VectorDB] Collection '{self.collection_name}' reset")
        except Exception as e:
            print(f"[ERROR] Failed to reset collection: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about current collection
        
        Returns:
            Dictionary with collection metadata and stats
        """
        try:
            return {
                "name": self.collection_name,
                "document_count": self.collection.count(),
                "metadata": self.collection.metadata if hasattr(self.collection, 'metadata') else {},
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            print(f"[ERROR] Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def list_all_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections in the ChromaDB instance
        
        Returns:
            List of collection dictionaries with stats
        """
        try:
            all_collections = self.client.list_collections()
            collections = []
            for collection in all_collections:
                try:
                    count = collection.count()
                    collections.append({
                        "name": collection.name,
                        "vector_count": count,
                        "metadata": collection.metadata if hasattr(collection, 'metadata') else {}
                    })
                except Exception as e:
                    print(f"[WARNING] Error counting collection {collection.name}: {e}")
            return collections
        except Exception as e:
            print(f"[ERROR] Failed to list collections: {e}")
            return []
    
    def get_collection_metadata_summary(self) -> Dict[str, Any]:
        """
        Get summary of metadata fields and their values in collection
        
        Returns:
            Dictionary with metadata field summaries (RBAC tags, company IDs, dept IDs, etc.)
        """
        try:
            # Peek at up to 100 documents to analyze metadata
            sample_docs = self.collection.peek(limit=100)
            
            metadata_summary = {
                "total_documents": self.collection.count(),
                "rbac_tags": set(),
                "companies": set(),
                "departments": set(),
                "document_ids": set(),
                "metadata_fields": set()
            }
            
            if sample_docs and sample_docs.get("metadatas"):
                for metadata in sample_docs["metadatas"]:
                    if metadata:
                        # Extract RBAC tags
                        if "rbac_tags" in metadata:
                            rbac = metadata["rbac_tags"]
                            if isinstance(rbac, str) and rbac.startswith("rbac:"):
                                metadata_summary["rbac_tags"].add(rbac)
                                # Parse company and dept from rbac tag
                                parts = rbac.split(":")
                                if len(parts) >= 3:
                                    metadata_summary["companies"].add(parts[1])
                                    metadata_summary["departments"].add(parts[2])
                        
                        # Extract document ID
                        if "document_id" in metadata:
                            metadata_summary["document_ids"].add(metadata["document_id"])
                        
                        # Track all metadata field names
                        metadata_summary["metadata_fields"].update(metadata.keys())
            
            # Convert sets to lists for JSON serialization
            return {
                "total_documents": metadata_summary["total_documents"],
                "unique_rbac_tags": list(metadata_summary["rbac_tags"]),
                "unique_companies": list(metadata_summary["companies"]),
                "unique_departments": list(metadata_summary["departments"]),
                "unique_document_ids": list(metadata_summary["document_ids"]),
                "metadata_field_names": list(metadata_summary["metadata_fields"])
            }
        except Exception as e:
            print(f"[ERROR] Failed to get metadata summary: {e}")
            return {"error": str(e)}
