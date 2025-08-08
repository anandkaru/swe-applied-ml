"""
Data loading and preprocessing for the PrimeApple Review Insight Pipeline.
"""

import pandas as pd
import logging
from typing import Optional
import re

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of review data."""
    
    def __init__(self):
        self.logger = logger
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load review data from CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info(f"Loading data from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['review_id', 'product', 'rating', 'created_at', 'text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Preprocess data
        df = self._preprocess_data(df)
        
        self.logger.info(f"Loaded {len(df)} reviews")
        self.logger.info(f"Products: {df['product'].value_counts().to_dict()}")
        self.logger.info(f"Rating distribution: {df['rating'].value_counts().sort_index().to_dict()}")
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the review data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Convert created_at to datetime
        df['created_at'] = pd.to_datetime(df['created_at'])
        
        # Clean text
        df['text_clean'] = df['text'].apply(self._clean_text)
        
        # Remove reviews with empty text after cleaning
        initial_count = len(df)
        df = df[df['text_clean'].str.len() > 10].copy()
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            self.logger.warning(f"Removed {removed_count} reviews with insufficient text")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """
        Clean review text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_product_stats(self, df: pd.DataFrame) -> dict:
        """
        Get statistics by product.
        
        Args:
            df: Review DataFrame
            
        Returns:
            Dictionary with product statistics
        """
        stats = {}
        
        for product in df['product'].unique():
            product_df = df[df['product'] == product]
            stats[product] = {
                'count': len(product_df),
                'avg_rating': product_df['rating'].mean(),
                'rating_distribution': product_df['rating'].value_counts().sort_index().to_dict(),
                'date_range': {
                    'start': product_df['created_at'].min(),
                    'end': product_df['created_at'].max()
                }
            }
        
        return stats 