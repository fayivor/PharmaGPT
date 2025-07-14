"""Text embedding utilities for PharmaGPT."""

import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from config import config

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Text embedding utility using sentence transformers."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the text embedder.
        
        Args:
            model_name: Name of the sentence transformer model to use.
                       Defaults to config.EMBEDDING_MODEL.
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing TextEmbedder with model: {self.model_name}")
        
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as numpy array.
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            Array of embedding vectors.
        """
        self._load_model()
        
        try:
            # Clean and preprocess texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                cleaned_texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
                batch_size=32
            )
            
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding.
        
        Args:
            text: Raw text to preprocess.
            
        Returns:
            Preprocessed text.
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (most models have token limits)
        max_length = 512  # Conservative limit for most sentence transformers
        if len(text) > max_length:
            text = text[:max_length]
            logger.debug("Text truncated for embedding")
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension.
        """
        self._load_model()
        return self.model.get_sentence_embedding_dimension()
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.
            
        Returns:
            Cosine similarity score.
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)


class PharmaTextEmbedder(TextEmbedder):
    """Specialized text embedder for pharmaceutical content."""
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess pharmaceutical text with domain-specific handling.
        
        Args:
            text: Raw pharmaceutical text.
            
        Returns:
            Preprocessed text optimized for pharmaceutical content.
        """
        text = super()._preprocess_text(text)
        
        if not text:
            return text
        
        # Pharmaceutical-specific preprocessing
        # Preserve drug names, dosages, and medical terms
        # This is a simplified version - in practice, you'd use more sophisticated
        # medical text preprocessing
        
        # Convert common abbreviations
        abbreviations = {
            "mg": "milligrams",
            "ml": "milliliters", 
            "kg": "kilograms",
            "FDA": "Food and Drug Administration",
            "NIH": "National Institutes of Health"
        }
        
        for abbrev, full_form in abbreviations.items():
            text = text.replace(f" {abbrev} ", f" {full_form} ")
            text = text.replace(f" {abbrev}.", f" {full_form}")
        
        return text
