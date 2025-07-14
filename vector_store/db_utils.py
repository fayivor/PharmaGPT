"""Vector database utilities using FAISS for PharmaGPT."""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from dataclasses import dataclass
from config import config
from .embedder import PharmaTextEmbedder

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document representation for vector store."""
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    embedding: Optional[np.ndarray] = None


class VectorStore:
    """FAISS-based vector store for pharmaceutical documents."""
    
    def __init__(self, store_path: Optional[str] = None, embedder: Optional[PharmaTextEmbedder] = None):
        """Initialize the vector store.
        
        Args:
            store_path: Path to store/load the vector database.
            embedder: Text embedder instance.
        """
        self.store_path = store_path or config.VECTOR_STORE_PATH
        self.embedder = embedder or PharmaTextEmbedder()
        self.index = None
        self.documents: List[Document] = []
        self.dimension = None
        
        # Ensure store directory exists
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        
        logger.info(f"Initialized VectorStore with path: {self.store_path}")
    
    def _initialize_index(self, dimension: int) -> None:
        """Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension.
        """
        self.dimension = dimension
        # Use IndexFlatIP for cosine similarity (after L2 normalization)
        self.index = faiss.IndexFlatIP(dimension)
        logger.info(f"Initialized FAISS index with dimension: {dimension}")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add.
        """
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Extract texts for embedding
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts)
        
        # Initialize index if needed
        if self.index is None:
            self._initialize_index(embeddings.shape[1])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documents with embeddings
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
            self.documents.append(doc)
        
        logger.info(f"Added {len(documents)} documents. Total documents: {len(self.documents)}")
    
    def search(self, query: str, k: int = 10, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Search for similar documents.
        
        Args:
            query: Search query.
            k: Number of results to return.
            filter_metadata: Optional metadata filters.
            
        Returns:
            List of (document, similarity_score) tuples.
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("Vector store is empty")
            return []
        
        # Embed query
        query_embedding = self.embedder.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            doc = self.documents[idx]
            
            # Apply metadata filters if provided
            if filter_metadata:
                if not self._matches_filter(doc.metadata, filter_metadata):
                    continue
            
            results.append((doc, float(score)))
        
        logger.debug(f"Found {len(results)} results for query: {query[:50]}...")
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document metadata matches filters.
        
        Args:
            metadata: Document metadata.
            filters: Filter criteria.
            
        Returns:
            True if metadata matches all filters.
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def save(self) -> None:
        """Save the vector store to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        try:
            # Save FAISS index
            index_path = f"{self.store_path}.index"
            faiss.write_index(self.index, index_path)
            
            # Save documents and metadata
            metadata_path = f"{self.store_path}.metadata"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'dimension': self.dimension
                }, f)
            
            logger.info(f"Saved vector store to {self.store_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load(self) -> bool:
        """Load the vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise.
        """
        index_path = f"{self.store_path}.index"
        metadata_path = f"{self.store_path}.metadata"
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.info("No existing vector store found")
            return False
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents and metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.dimension = data['dimension']
            
            logger.info(f"Loaded vector store with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics.
        
        Returns:
            Dictionary with store statistics.
        """
        return {
            'total_documents': len(self.documents),
            'dimension': self.dimension,
            'sources': list(set(doc.source for doc in self.documents)),
            'index_size': self.index.ntotal if self.index else 0
        }
