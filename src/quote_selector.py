"""
Quote selection for the PrimeApple Review Insight Pipeline.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import re

logger = logging.getLogger(__name__)


class QuoteSelector:
    """Selects representative quotes for each theme."""
    
    def __init__(self):
        self.logger = logger
    
    def select_quotes(self, df: pd.DataFrame, clusters: np.ndarray,
                      sentiment_results: Dict,
                      embeddings: Optional[np.ndarray] = None) -> Dict[int, List[Dict]]:
        """
        Select representative quotes for each theme.
        
        Args:
            df: Review DataFrame
            clusters: Cluster assignments
            sentiment_results: Sentiment analysis results
            
        Returns:
            Dictionary mapping cluster_id to list of selected quotes
        """
        self.logger.info("Selecting representative quotes")
        
        theme_quotes = {}
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            self.logger.info(f"Selecting quotes for cluster {cluster_id}")
            
            # Get reviews for this cluster
            cluster_mask = clusters == cluster_id
            cluster_reviews = df[cluster_mask]
            # Get sentiment indices that correspond to the cluster reviews
            cluster_indices = np.where(cluster_mask)[0]
            cluster_sentiments = [sentiment_results['sentiments'][i] for i in cluster_indices]
            # Subset embeddings for this cluster if available
            cluster_embeddings = None
            if embeddings is not None:
                try:
                    cluster_embeddings = embeddings[cluster_mask]
                except Exception:
                    cluster_embeddings = None
            
            # Select quotes
            quotes = self._select_cluster_quotes(
                cluster_reviews, cluster_sentiments, cluster_id, cluster_embeddings
            )
            
            theme_quotes[cluster_id] = quotes
        
        self.logger.info(f"Selected quotes for {len(theme_quotes)} themes")
        
        return theme_quotes
    
    def _select_cluster_quotes(self, cluster_reviews: pd.DataFrame,
                               cluster_sentiments: List[str],
                               cluster_id: int,
                               cluster_embeddings: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Select representative quotes for a single cluster.
        
        Args:
            cluster_reviews: Reviews in the cluster
            cluster_sentiments: Sentiment labels for the cluster
            cluster_id: Cluster identifier
            
        Returns:
            List of selected quotes
        """
        # Add sentiment to DataFrame
        cluster_df = cluster_reviews.copy()
        cluster_df['sentiment'] = cluster_sentiments
        
        # Score all reviews
        scored_reviews = self._score_reviews(cluster_df)
        
        # Select diverse quotes
        if cluster_embeddings is not None and len(cluster_embeddings) == len(scored_reviews):
            selected_quotes = self._select_diverse_quotes_mmr(scored_reviews, cluster_embeddings)
        else:
            selected_quotes = self._select_diverse_quotes_jaccard(scored_reviews)
        
        return selected_quotes
    
    def _score_reviews(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score reviews based on multiple criteria.
        
        Args:
            cluster_df: Cluster reviews with sentiment
            
        Returns:
            DataFrame with scores
        """
        scored_df = cluster_df.copy()
        
        # Initialize scores
        scored_df['length_score'] = 0.0
        scored_df['specificity_score'] = 0.0
        scored_df['sentiment_score'] = 0.0
        scored_df['clarity_score'] = 0.0
        scored_df['total_score'] = 0.0
        
        for idx, row in scored_df.iterrows():
            text = row['text_clean']
            
            # Length score (prefer medium length)
            length = len(text)
            if 50 <= length <= 200:
                length_score = 1.0
            elif 30 <= length <= 300:
                length_score = 0.7
            else:
                length_score = 0.3
            scored_df.at[idx, 'length_score'] = length_score
            
            # Specificity score (mentions specific features)
            specificity_score = self._calculate_specificity_score(text)
            scored_df.at[idx, 'specificity_score'] = specificity_score
            
            # Sentiment score (diversity)
            sentiment_score = self._calculate_sentiment_score(row['sentiment'], row['rating'])
            scored_df.at[idx, 'sentiment_score'] = sentiment_score
            
            # Clarity score (well-written)
            clarity_score = self._calculate_clarity_score(text)
            scored_df.at[idx, 'clarity_score'] = clarity_score
            
            # Total score (weighted average)
            total_score = (
                0.2 * length_score +
                0.3 * specificity_score +
                0.3 * sentiment_score +
                0.2 * clarity_score
            )
            scored_df.at[idx, 'total_score'] = total_score
        
        return scored_df
    
    def _calculate_specificity_score(self, text: str) -> float:
        """
        Calculate specificity score based on feature mentions.
        
        Args:
            text: Review text
            
        Returns:
            Specificity score (0-1)
        """
        # Define specific features to look for
        features = [
            'battery', 'life', 'charging', 'power',
            'screen', 'display', 'glare', 'resolution',
            'pen', 'stylus', 'latency', 'pressure',
            'firmware', 'update', 'software', 'app',
            'packaging', 'box', 'unboxing', 'accessories',
            'wifi', 'connectivity', 'bluetooth', 'network',
            'performance', 'speed', 'lag', 'responsive'
        ]
        
        text_lower = text.lower()
        feature_count = sum(1 for feature in features if feature in text_lower)
        
        # Score based on feature mentions
        if feature_count >= 3:
            return 1.0
        elif feature_count == 2:
            return 0.8
        elif feature_count == 1:
            return 0.6
        else:
            return 0.3
    
    def _calculate_sentiment_score(self, sentiment: str, rating: int) -> float:
        """
        Calculate sentiment score for diversity.
        
        Args:
            sentiment: Sentiment label
            rating: Original rating
            
        Returns:
            Sentiment score (0-1)
        """
        # Prefer diverse sentiment representation
        # Higher score for reviews that don't align with rating
        if sentiment == 'positive' and rating <= 2:
            return 1.0  # Positive sentiment with low rating
        elif sentiment == 'negative' and rating >= 4:
            return 1.0  # Negative sentiment with high rating
        elif sentiment == 'neutral':
            return 0.8  # Neutral sentiment
        else:
            return 0.5  # Expected sentiment-rating alignment
    
    def _calculate_clarity_score(self, text: str) -> float:
        """
        Calculate clarity score based on writing quality.
        
        Args:
            text: Review text
            
        Returns:
            Clarity score (0-1)
        """
        # Check for good writing indicators
        good_indicators = [
            len(text.split()) >= 10,  # Sufficient length
            text.count('.') >= 1,     # Has sentences
            text.count(',') >= 1,     # Has structure
            not text.isupper(),       # Not all caps
            len(text) >= 30           # Minimum length
        ]
        
        score = sum(good_indicators) / len(good_indicators)
        
        # Bonus for specific details
        if any(word in text.lower() for word in ['because', 'since', 'when', 'while']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _select_diverse_quotes_jaccard(self, scored_df: pd.DataFrame) -> List[Dict]:
        """
        Select diverse quotes ensuring sentiment and rating diversity.
        
        Args:
            scored_df: DataFrame with scored reviews
            
        Returns:
            List of selected quotes
        """
        selected_quotes = []
        
        # Sort by total score
        scored_df = scored_df.sort_values('total_score', ascending=False)
        
        # Select quotes ensuring diversity and avoiding repetition
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        selected_texts = set()  # Track selected text content to avoid repetition
        
        # First pass: Try to get one quote from each sentiment category
        for sentiment in ['positive', 'negative', 'neutral']:
            if len(selected_quotes) >= 3:
                break
                
            sentiment_df = scored_df[scored_df['sentiment'] == sentiment]
            if len(sentiment_df) == 0:
                continue
                
            for _, row in sentiment_df.iterrows():
                if len(selected_quotes) >= 3:
                    break
                    
                text = row['text_clean']
                
                # Check for text similarity to avoid repetitive quotes
                is_similar = False
                for selected_text in selected_texts:
                    similarity = self._calculate_text_similarity(text, selected_text)
                    if similarity > 0.6:  # Lower threshold for better diversity
                        is_similar = True
                        break
                
                if not is_similar:
                    quote = {
                        'text': text,
                        'rating': int(row['rating']),
                        'sentiment': sentiment,
                        'score': float(row['total_score']),
                        'created_at': row['created_at'].isoformat() if pd.notna(row['created_at']) else None
                    }
                    
                    selected_quotes.append(quote)
                    sentiment_counts[sentiment] += 1
                    rating_counts[row['rating']] += 1
                    selected_texts.add(text)
                    break  # Only take one from each sentiment initially
        
        # Second pass: Fill remaining slots with best available quotes
        if len(selected_quotes) < 3:
            remaining_df = scored_df[~scored_df.index.isin([q.get('index', -1) for q in selected_quotes])]
            for _, row in remaining_df.head(10).iterrows():  # Check more candidates
                if len(selected_quotes) >= 3:
                    break
                    
                text = row['text_clean']
                sentiment = row['sentiment']
                rating = row['rating']
                
                # Check for similarity
                is_similar = False
                for selected_text in selected_texts:
                    similarity = self._calculate_text_similarity(text, selected_text)
                    if similarity > 0.6:
                        is_similar = True
                        break
                
                if not is_similar:
                    quote = {
                        'text': text,
                        'rating': int(rating),
                        'sentiment': sentiment,
                        'score': float(row['total_score']),
                        'created_at': row['created_at'].isoformat() if pd.notna(row['created_at']) else None
                    }
                    selected_quotes.append(quote)
                    selected_texts.add(text)
        
        # Sort by score for final order
        selected_quotes.sort(key=lambda x: x['score'], reverse=True)
        
        return selected_quotes[:3]

    def _select_diverse_quotes_mmr(self, scored_df: pd.DataFrame,
                                   cluster_embeddings: np.ndarray,
                                   max_quotes: int = 3,
                                   alpha: float = 0.7) -> List[Dict]:
        """Select quotes using Maximal Marginal Relevance with embeddings.

        Args:
            scored_df: DataFrame with per-review scores
            cluster_embeddings: Embeddings aligned to scored_df rows
            max_quotes: Number of quotes to return
            alpha: Trade-off between quality and diversity (0..1)

        Returns:
            List of selected quote dicts
        """
        if len(scored_df) == 0:
            return []

        # Normalize embeddings to unit vectors for cosine similarity
        emb = cluster_embeddings.astype(float)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        sim = emb @ emb.T  # cosine similarity matrix

        # Normalize quality scores to [0,1]
        q = scored_df['total_score'].to_numpy(dtype=float)
        q_min, q_max = float(np.min(q)), float(np.max(q))
        denom = (q_max - q_min) if (q_max - q_min) > 1e-9 else 1.0
        quality = (q - q_min) / denom

        candidates = list(range(len(scored_df)))
        selected_idx: List[int] = []

        while candidates and len(selected_idx) < max_quotes:
            mmr_scores: List[float] = []
            for i in candidates:
                diversity = 0.0 if not selected_idx else float(np.max(sim[i, selected_idx]))
                mmr = alpha * quality[i] - (1.0 - alpha) * diversity
                mmr_scores.append(mmr)
            best_local = candidates[int(np.argmax(mmr_scores))]
            selected_idx.append(best_local)
            candidates.remove(best_local)

        # Build quote dicts in selected order
        selected_quotes: List[Dict] = []
        for i in selected_idx:
            row = scored_df.iloc[i]
            quote = {
                'text': row['text_clean'],
                'rating': int(row['rating']),
                'sentiment': row['sentiment'],
                'score': float(row['total_score']),
                'created_at': row['created_at'].isoformat() if pd.notna(row.get('created_at', None)) else None
            }
            selected_quotes.append(quote)

        # Keep highest score first
        selected_quotes.sort(key=lambda x: x['score'], reverse=True)
        return selected_quotes[:max_quotes]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Convert to sets of words
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_quote_statistics(self, theme_quotes: Dict[int, List[Dict]]) -> Dict:
        """
        Get statistics about selected quotes.
        
        Args:
            theme_quotes: Dictionary of theme quotes
            
        Returns:
            Statistics dictionary
        """
        all_quotes = []
        for quotes in theme_quotes.values():
            all_quotes.extend(quotes)
        
        if not all_quotes:
            return {}
        
        # Calculate statistics
        avg_length = np.mean([len(q['text']) for q in all_quotes])
        sentiment_dist = pd.Series([q['sentiment'] for q in all_quotes]).value_counts().to_dict()
        rating_dist = pd.Series([q['rating'] for q in all_quotes]).value_counts().sort_index().to_dict()
        avg_score = np.mean([q['score'] for q in all_quotes])
        
        return {
            'total_quotes': len(all_quotes),
            'avg_length': avg_length,
            'sentiment_distribution': sentiment_dist,
            'rating_distribution': rating_dist,
            'avg_score': avg_score
        } 