"""
Custom exceptions for document processing in Authormaton core engine.
"""

class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass

class DocumentParseError(DocumentProcessingError):
    """Raised when document parsing fails."""
    pass

class DocumentSaveError(DocumentProcessingError):
    """Raised when saving a document fails."""
    pass

class DocumentChunkError(DocumentProcessingError):
    """Raised when chunking a document fails."""
    pass

class DocumentEmbeddingError(DocumentProcessingError):
    """Raised when embedding generation fails."""
    pass
