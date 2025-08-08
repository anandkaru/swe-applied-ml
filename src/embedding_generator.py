"""
Semantic embedding generation for the PrimeApple Review Insight Pipeline.
"""

import numpy as np
import pandas as pd
import logging
import joblib
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Optional
import os

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates semantic embeddings for review text."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        self.model = None
    
    def generate_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate embeddings for review text.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            Embeddings array
        """
        cache_path = self.config.get_embedding_cache_path(self.config.embedding_model)
        
        # Check if embeddings are cached and force_recompute is False
        if not self.config.force_recompute and os.path.exists(cache_path):
            self.logger.info(f"Loading cached embeddings from {cache_path}")
            return joblib.load(cache_path)
        
        self.logger.info(f"Generating embeddings using {self.config.embedding_model}")
        
        # Load model
        self.model = SentenceTransformer(self.config.embedding_model)
        
        # Generate embeddings
        texts = df['text_clean'].tolist()
        embeddings = self._generate_embeddings_batch(texts)
        
        # Cache embeddings
        self.logger.info(f"Caching embeddings to {cache_path}")
        joblib.dump(embeddings, cache_path)
        
        return embeddings
    
    def _generate_embeddings_batch(self, texts: list) -> np.ndarray:
        """
        Generate embeddings in batches.
        
        Args:
            texts: List of text strings
            
        Returns:
            Embeddings array
        """
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.config.batch_size), 
                     desc="Generating embeddings"):
            batch_texts = texts[i:i + self.config.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        embeddings = np.vstack(embeddings)
        
        self.logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def get_embedding_stats(self, embeddings: np.ndarray) -> dict:
        """
        Get statistics about the embeddings.
        
        Args:
            embeddings: Embeddings array
            
        Returns:
            Dictionary with embedding statistics
        """
        return {
            'shape': embeddings.shape,
            'mean': float(np.mean(embeddings)),
            'std': float(np.std(embeddings)),
            'min': float(np.min(embeddings)),
            'max': float(np.max(embeddings)),
            'model': self.config.embedding_model
        }
    
    def compute_similarity_matrix(self, embeddings: np.ndarray, 
                                sample_size: Optional[int] = None) -> np.ndarray:
        """
        Compute similarity matrix for a sample of embeddings.
        
        Args:
            embeddings: Full embeddings array
            sample_size: Number of samples to use (for memory efficiency)
            
        Returns:
            Similarity matrix
        """
        if sample_size and sample_size < len(embeddings):
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[indices]
        else:
            sample_embeddings = embeddings
        
        # Normalize embeddings
        normalized_embeddings = sample_embeddings / np.linalg.norm(
            sample_embeddings, axis=1, keepdims=True
        )
        
        # Compute cosine similarity
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix 