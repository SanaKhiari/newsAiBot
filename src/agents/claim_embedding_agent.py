"""
Claim Embedding Agent: Generates semantic embeddings for claims.
Uses sentence-transformers for efficient, high-quality embeddings.
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.config import settings


class ClaimEmbeddingAgent:
    """Agent for generating embeddings from claims."""
    
    def __init__(self):
        """Initialize sentence transformer model."""
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        
        try:
            self.model = SentenceTransformer(settings.embedding_model)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded successfully")
            logger.debug(f"  - Model: {settings.embedding_model}")
            logger.debug(f"  - Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_claims(self, claims: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for all claims.
        
        Args:
            claims: List of claim dictionaries
            
        Returns:
            Claims with added 'embedding' field
        """
        if not claims:
            logger.warning("No claims to embed")
            return []
        
        logger.info(f"Generating embeddings for {len(claims)} claims")
        
        try:
            # Extract claim texts
            claim_texts = [claim['text'] for claim in claims]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(
                claim_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Add embeddings to claims
            for claim, embedding in zip(claims, embeddings):
                claim['embedding'] = embedding.tolist()
            
            logger.info(f"Successfully embedded {len(claims)} claims")
            
            return claims
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized embedding vector
        """
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=True
            )
            return embedding
        except Exception as e:
            logger.error(f"Single text embedding failed: {e}")
            raise
