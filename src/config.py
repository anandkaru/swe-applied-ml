"""
Configuration management for the PrimeApple Review Insight Pipeline.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Pipeline configuration settings."""
    
    # LLM Settings
    llm_model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 200
    top_p: float = 0.9
    
    # Clustering Settings
    min_clusters: int = 3  # Updated based on experiment results
    max_clusters: int = 5  # Updated based on experiment results
    random_seed: int = 42
    silhouette_threshold: float = 0.3
    
    # Embedding Settings
    embedding_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    force_recompute: bool = False
    
    # Storage Settings
    db_path: str = "insights.db"
    cache_dir: str = "cache"
    
    # Sentiment Analysis
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Only require OpenAI API key if we're using LLM features
        if not os.getenv("OPENAI_API_KEY") and self.llm_model.startswith("gpt"):
            raise ValueError("OPENAI_API_KEY environment variable is required for LLM features")
        
        if self.temperature < 0 or self.temperature > 1:
            raise ValueError("Temperature must be between 0 and 1")
        
        if self.min_clusters > self.max_clusters:
            raise ValueError("min_clusters cannot be greater than max_clusters")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")
    
    def get_embedding_cache_path(self, model_name: str) -> str:
        """Get cache path for embeddings."""
        return os.path.join(self.cache_dir, f"embeddings_{model_name}.joblib")
    
    def get_cluster_cache_path(self, model_name: str, k: int) -> str:
        """Get cache path for clustering results."""
        return os.path.join(self.cache_dir, f"clusters_{model_name}_k{k}.joblib") 