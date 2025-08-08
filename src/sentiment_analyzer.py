"""
Sentiment analysis for the PrimeApple Review Insight Pipeline.
"""

import pandas as pd
import numpy as np
import logging
from transformers import pipeline
from typing import Dict, List, Tuple
import re

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyzes sentiment of review text."""
    
    def __init__(self):
        self.logger = logger
        self.sentiment_pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentiment analysis model."""
        try:
            self.logger.info("Loading sentiment analysis model...")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            self.logger.info("Sentiment model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load sentiment model: {e}")
            raise
    
    def analyze_sentiment(self, df: pd.DataFrame) -> Dict:
        """
        Analyze sentiment for all reviews.
        
        Args:
            df: DataFrame with review data
            
        Returns:
            Dictionary with sentiment analysis results
        """
        self.logger.info("Starting sentiment analysis")
        
        # Analyze sentiment for each review
        sentiments = []
        confidence_scores = []
        
        for idx, row in df.iterrows():
            sentiment_result = self._analyze_single_review(row['text_clean'])
            sentiments.append(sentiment_result['label'])
            confidence_scores.append(sentiment_result['score'])
            
            if (idx + 1) % 100 == 0:
                self.logger.info(f"Processed {idx + 1} reviews")
        
        # Create results dictionary
        results = {
            'sentiments': sentiments,
            'confidence_scores': confidence_scores,
            'sentiment_distribution': self._get_sentiment_distribution(sentiments),
            'avg_confidence': np.mean(confidence_scores),
            'sentiment_by_rating': self._get_sentiment_by_rating(df, sentiments),
            'sentiment_by_product': self._get_sentiment_by_product(df, sentiments)
        }
        
        self.logger.info("Sentiment analysis completed")
        self.logger.info(f"Sentiment distribution: {results['sentiment_distribution']}")
        
        return results
    
    def _analyze_single_review(self, text: str) -> Dict:
        """
        Analyze sentiment for a single review.
        
        Args:
            text: Review text
            
        Returns:
            Dictionary with sentiment label and confidence score
        """
        # Truncate text if too long (model has token limits)
        if len(text) > 500:
            text = text[:500]
        
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # Find the highest scoring sentiment
            best_sentiment = max(result, key=lambda x: x['score'])
            
            return {
                'label': best_sentiment['label'],
                'score': best_sentiment['score']
            }
        except Exception as e:
            self.logger.warning(f"Failed to analyze sentiment for text: {e}")
            return {
                'label': 'neutral',
                'score': 0.5
            }
    
    def _get_sentiment_distribution(self, sentiments: List[str]) -> Dict[str, float]:
        """
        Get distribution of sentiments.
        
        Args:
            sentiments: List of sentiment labels
            
        Returns:
            Dictionary with sentiment distribution
        """
        sentiment_counts = pd.Series(sentiments).value_counts()
        total = len(sentiments)
        
        distribution = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            distribution[sentiment] = (count / total) * 100
        
        return distribution
    
    def _get_sentiment_by_rating(self, df: pd.DataFrame, 
                                sentiments: List[str]) -> Dict[int, Dict[str, float]]:
        """
        Get sentiment distribution by rating.
        
        Args:
            df: Review DataFrame
            sentiments: List of sentiment labels
            
        Returns:
            Dictionary with sentiment distribution by rating
        """
        df_with_sentiment = df.copy()
        df_with_sentiment['sentiment'] = sentiments
        
        sentiment_by_rating = {}
        
        for rating in sorted(df['rating'].unique()):
            rating_df = df_with_sentiment[df_with_sentiment['rating'] == rating]
            if len(rating_df) > 0:
                sentiment_counts = rating_df['sentiment'].value_counts()
                total = len(rating_df)
                
                distribution = {}
                for sentiment in ['positive', 'negative', 'neutral']:
                    count = sentiment_counts.get(sentiment, 0)
                    distribution[sentiment] = (count / total) * 100
                
                sentiment_by_rating[rating] = distribution
        
        return sentiment_by_rating
    
    def _get_sentiment_by_product(self, df: pd.DataFrame, 
                                 sentiments: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get sentiment distribution by product.
        
        Args:
            df: Review DataFrame
            sentiments: List of sentiment labels
            
        Returns:
            Dictionary with sentiment distribution by product
        """
        df_with_sentiment = df.copy()
        df_with_sentiment['sentiment'] = sentiments
        
        sentiment_by_product = {}
        
        for product in df['product'].unique():
            product_df = df_with_sentiment[df_with_sentiment['product'] == product]
            if len(product_df) > 0:
                sentiment_counts = product_df['sentiment'].value_counts()
                total = len(product_df)
                
                distribution = {}
                for sentiment in ['positive', 'negative', 'neutral']:
                    count = sentiment_counts.get(sentiment, 0)
                    distribution[sentiment] = (count / total) * 100
                
                sentiment_by_product[product] = distribution
        
        return sentiment_by_product
    
    def get_sentiment_summary(self, results: Dict) -> str:
        """
        Generate a summary of sentiment analysis results.
        
        Args:
            results: Sentiment analysis results
            
        Returns:
            Summary string
        """
        distribution = results['sentiment_distribution']
        
        summary = f"Overall sentiment: {distribution['positive']:.1f}% positive, "
        summary += f"{distribution['negative']:.1f}% negative, "
        summary += f"{distribution['neutral']:.1f}% neutral"
        
        return summary 