"""Database package initialization."""

from .connection import get_connection, close_connection, RAGDatabaseConnection

__all__ = ["get_connection", "close_connection", "RAGDatabaseConnection"]
