"""Document reranking utilities for improved retrieval relevance."""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from dataclasses import dataclass
from config import config

logger = logging.getLogger(__name__)


@dataclass
class RankedDocument:
    """Document with relevance score."""
    content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    source: str


class CrossEncoderReranker:
    """Cross-encoder based document reranker."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use.
        """
        self.model_name = model_name or config.RERANKER_MODEL
        self.model = None
        logger.info(f"Initializing CrossEncoderReranker with model: {self.model_name}")
    
    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        if self.model is None:
            try:
                self.model = CrossEncoder(self.model_name)
                logger.info(f"Loaded reranker model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load reranker model: {e}")
                raise
    
    def rerank(self, query: str, documents: List[Tuple[str, float, Dict[str, Any]]], 
               top_k: Optional[int] = None) -> List[RankedDocument]:
        """Rerank documents based on query relevance.
        
        Args:
            query: Search query.
            documents: List of (content, score, metadata) tuples.
            top_k: Number of top documents to return.
            
        Returns:
            List of reranked documents.
        """
        if not documents:
            return []
        
        self._load_model()
        
        # Prepare query-document pairs
        pairs = []
        doc_data = []
        
        for content, original_score, metadata in documents:
            # Truncate content if too long
            truncated_content = self._truncate_text(content)
            pairs.append([query, truncated_content])
            doc_data.append((content, original_score, metadata))
        
        try:
            # Get reranking scores
            rerank_scores = self.model.predict(pairs)
            
            # Create ranked documents
            ranked_docs = []
            for (content, original_score, metadata), rerank_score in zip(doc_data, rerank_scores):
                ranked_doc = RankedDocument(
                    content=content,
                    metadata=metadata,
                    original_score=original_score,
                    rerank_score=float(rerank_score),
                    source=metadata.get('source', 'unknown')
                )
                ranked_docs.append(ranked_doc)
            
            # Sort by rerank score (descending)
            ranked_docs.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Return top_k if specified
            if top_k:
                ranked_docs = ranked_docs[:top_k]
            
            logger.debug(f"Reranked {len(documents)} documents, returning top {len(ranked_docs)}")
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback: return original ranking
            return [
                RankedDocument(
                    content=content,
                    metadata=metadata,
                    original_score=original_score,
                    rerank_score=original_score,
                    source=metadata.get('source', 'unknown')
                )
                for content, original_score, metadata in documents
            ]
    
    def _truncate_text(self, text: str, max_length: int = 512) -> str:
        """Truncate text to fit model limits.
        
        Args:
            text: Text to truncate.
            max_length: Maximum character length.
            
        Returns:
            Truncated text.
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')
        
        if last_period > max_length * 0.8:
            return truncated[:last_period + 1]
        elif last_space > max_length * 0.8:
            return truncated[:last_space]
        else:
            return truncated


class HybridReranker:
    """Hybrid reranker combining multiple signals."""
    
    def __init__(self, cross_encoder_model: Optional[str] = None):
        """Initialize hybrid reranker.
        
        Args:
            cross_encoder_model: Cross-encoder model name.
        """
        self.cross_encoder = CrossEncoderReranker(cross_encoder_model)
        logger.info("Initialized HybridReranker")
    
    def rerank(self, query: str, documents: List[Tuple[str, float, Dict[str, Any]]], 
               top_k: Optional[int] = None, weights: Optional[Dict[str, float]] = None) -> List[RankedDocument]:
        """Rerank documents using hybrid approach.
        
        Args:
            query: Search query.
            documents: List of (content, score, metadata) tuples.
            top_k: Number of top documents to return.
            weights: Weights for different ranking signals.
            
        Returns:
            List of reranked documents.
        """
        if not documents:
            return []
        
        # Default weights
        if weights is None:
            weights = {
                'semantic': 0.6,      # Cross-encoder score
                'retrieval': 0.2,     # Original retrieval score
                'source': 0.1,        # Source reliability
                'recency': 0.1        # Publication recency
            }
        
        # Get cross-encoder scores
        ranked_docs = self.cross_encoder.rerank(query, documents)
        
        # Apply hybrid scoring
        for doc in ranked_docs:
            hybrid_score = self._calculate_hybrid_score(doc, weights)
            doc.rerank_score = hybrid_score
        
        # Re-sort by hybrid score
        ranked_docs.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Return top_k if specified
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        
        logger.debug(f"Hybrid reranking completed, returning {len(ranked_docs)} documents")
        return ranked_docs
    
    def _calculate_hybrid_score(self, doc: RankedDocument, weights: Dict[str, float]) -> float:
        """Calculate hybrid score for a document.
        
        Args:
            doc: Document to score.
            weights: Scoring weights.
            
        Returns:
            Hybrid score.
        """
        # Normalize scores to [0, 1] range
        semantic_score = self._normalize_score(doc.rerank_score, -10, 10)
        retrieval_score = self._normalize_score(doc.original_score, 0, 1)
        source_score = self._get_source_score(doc.source)
        recency_score = self._get_recency_score(doc.metadata)
        
        # Calculate weighted sum
        hybrid_score = (
            weights.get('semantic', 0.6) * semantic_score +
            weights.get('retrieval', 0.2) * retrieval_score +
            weights.get('source', 0.1) * source_score +
            weights.get('recency', 0.1) * recency_score
        )
        
        return hybrid_score
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to [0, 1] range.
        
        Args:
            score: Score to normalize.
            min_val: Minimum possible value.
            max_val: Maximum possible value.
            
        Returns:
            Normalized score.
        """
        if max_val == min_val:
            return 0.5
        
        normalized = (score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def _get_source_score(self, source: str) -> float:
        """Get reliability score for a source.
        
        Args:
            source: Source name.
            
        Returns:
            Source reliability score [0, 1].
        """
        # Source reliability mapping
        source_scores = {
            'pubmed': 0.9,
            'drugbank': 0.95,
            'fda': 0.95,
            'clinicaltrials': 0.85,
            'cochrane': 0.95,
            'nejm': 0.9,
            'lancet': 0.9,
            'jama': 0.9,
            'unknown': 0.5
        }
        
        source_lower = source.lower()
        for key, score in source_scores.items():
            if key in source_lower:
                return score
        
        return source_scores['unknown']
    
    def _get_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Get recency score based on publication date.
        
        Args:
            metadata: Document metadata.
            
        Returns:
            Recency score [0, 1].
        """
        # This is a simplified implementation
        # In practice, you'd parse publication dates and calculate age
        
        pub_date = metadata.get('publication_date', '')
        if not pub_date:
            return 0.5
        
        try:
            # Extract year (simplified)
            if len(pub_date) >= 4 and pub_date[:4].isdigit():
                year = int(pub_date[:4])
                current_year = 2024  # Update as needed
                age = current_year - year
                
                # Score decreases with age
                if age <= 1:
                    return 1.0
                elif age <= 3:
                    return 0.8
                elif age <= 5:
                    return 0.6
                elif age <= 10:
                    return 0.4
                else:
                    return 0.2
        except:
            pass
        
        return 0.5
