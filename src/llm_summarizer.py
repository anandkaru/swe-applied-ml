"""
LLM-based theme summarization for the PrimeApple Review Insight Pipeline.
"""

import pandas as pd
import numpy as np
import logging
import openai
from typing import Dict, List, Tuple
import time
import json

logger = logging.getLogger(__name__)

class LLMSummarizer:
    """Generates theme titles and summaries using LLM."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        # Set OpenAI API key globally for compatibility
        openai.api_key = config.openai_api_key
    
    def generate_summaries(self, df: pd.DataFrame, clusters: np.ndarray, 
                          sentiment_results: Dict) -> List[Dict]:
        """
        Generate theme summaries using LLM.
        
        Args:
            df: Review DataFrame
            clusters: Cluster assignments
            sentiment_results: Sentiment analysis results
            
        Returns:
            List of theme summaries
        """
        self.logger.info("Generating theme summaries with LLM")
        
        theme_summaries = []
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            self.logger.info(f"Generating summary for cluster {cluster_id}")
            
            # Get reviews for this cluster
            cluster_reviews = df[clusters == cluster_id]
            cluster_sentiments = [sentiment_results['sentiments'][i] 
                                for i in range(len(df)) if clusters[i] == cluster_id]
            
            # Generate summary
            summary = self._generate_cluster_summary(
                cluster_reviews, cluster_sentiments, cluster_id
            )
            
            theme_summaries.append(summary)
            
            # Rate limiting
            time.sleep(0.5)
        
        self.logger.info(f"Generated summaries for {len(theme_summaries)} themes")
        
        return theme_summaries
    
    def _generate_cluster_summary(self, cluster_reviews: pd.DataFrame, 
                                cluster_sentiments: List[str], 
                                cluster_id: int) -> Dict:
        """
        Generate summary for a single cluster.
        
        Args:
            cluster_reviews: Reviews in the cluster
            cluster_sentiments: Sentiment labels for the cluster
            cluster_id: Cluster identifier
            
        Returns:
            Dictionary with theme summary
        """
        # Prepare cluster data
        sample_reviews = self._get_sample_reviews(cluster_reviews, 10)
        sentiment_dist = self._get_cluster_sentiment_distribution(cluster_sentiments)
        product_dist = cluster_reviews['product'].value_counts().to_dict()
        
        # Create prompt
        prompt = self._create_summary_prompt(
            sample_reviews, sentiment_dist, product_dist, cluster_id
        )
        
        try:
            # Generate summary using OpenAI ChatCompletion
            response = openai.ChatCompletion.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            summary_text = response.choices[0].message['content'].strip()
            
            # Parse the response
            parsed_summary = self._parse_summary_response(summary_text)
            
            # Add metadata
            parsed_summary.update({
                'cluster_id': cluster_id,
                'review_count': len(cluster_reviews),
                'volume_pct': (len(cluster_reviews) / len(cluster_reviews)) * 100,
                'positive_pct': sentiment_dist.get('positive', 0),
                'negative_pct': sentiment_dist.get('negative', 0),
                'neutral_pct': sentiment_dist.get('neutral', 0),
                'product_distribution': product_dist
            })
            
            return parsed_summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary for cluster {cluster_id}: {e}")
            self.logger.error(f"API Key set: {bool(self.config.openai_api_key)}")
            self.logger.error(f"Model: {self.config.llm_model}")
            return self._get_fallback_summary(cluster_id, cluster_reviews, sentiment_dist)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM."""
        return """You are an expert business analyst specializing in customer feedback analysis for PrimeApple, a technology company.

Your task is to analyze customer reviews and create concise, actionable theme summaries that help leadership understand what customers love or dislike about their products.

IMPORTANT GUIDELINES:
1. Titles must be ≤8 words and use plain English
2. Summaries must be ≤50 words and focus on actionable insights
3. Be specific about product features mentioned
4. Highlight both positive and negative feedback
5. Use business-friendly language
6. Focus on insights that can drive product decisions

Format your response as JSON with these fields:
{
    "title": "Theme title (≤8 words)",
    "summary": "Theme explanation (≤50 words)"
}"""
    
    def _create_summary_prompt(self, sample_reviews: List[str], 
                             sentiment_dist: Dict[str, float],
                             product_dist: Dict[str, int],
                             cluster_id: int) -> str:
        """
        Create the prompt for generating a theme summary.
        
        Args:
            sample_reviews: Sample reviews from the cluster
            sentiment_dist: Sentiment distribution
            product_dist: Product distribution
            cluster_id: Cluster identifier
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze the following customer reviews for PrimeApple's EchoPad products and create a theme summary.

CLUSTER {cluster_id} ANALYSIS:

SAMPLE REVIEWS:
"""
        
        for i, review in enumerate(sample_reviews, 1):
            prompt += f"{i}. \"{review}\"\n"
        
        prompt += f"""
SENTIMENT DISTRIBUTION:
- Positive: {sentiment_dist.get('positive', 0):.1f}%
- Negative: {sentiment_dist.get('negative', 0):.1f}%
- Neutral: {sentiment_dist.get('neutral', 0):.1f}%

PRODUCT DISTRIBUTION:
"""
        
        for product, count in product_dist.items():
            prompt += f"- {product}: {count} reviews\n"
        
        prompt += """
Based on this analysis, create a theme title and summary that captures the main customer feedback pattern.

Respond with JSON only:
{
    "title": "Theme title (≤8 words)",
    "summary": "Theme explanation (≤50 words)"
}"""
        
        return prompt
    
    def _get_sample_reviews(self, cluster_reviews: pd.DataFrame, 
                           sample_size: int) -> List[str]:
        """
        Get a sample of reviews from the cluster.
        
        Args:
            cluster_reviews: Reviews in the cluster
            sample_size: Number of reviews to sample
            
        Returns:
            List of sample review texts
        """
        if len(cluster_reviews) <= sample_size:
            return cluster_reviews['text_clean'].tolist()
        
        # Sample reviews with diversity in ratings
        sample_reviews = []
        for rating in sorted(cluster_reviews['rating'].unique()):
            rating_reviews = cluster_reviews[cluster_reviews['rating'] == rating]
            if len(rating_reviews) > 0:
                sample = rating_reviews.sample(
                    min(len(rating_reviews), sample_size // 5 + 1)
                )
                sample_reviews.extend(sample['text_clean'].tolist())
        
        # If we don't have enough, add more random samples
        if len(sample_reviews) < sample_size:
            remaining = cluster_reviews[~cluster_reviews.index.isin(
                [i for i, _ in enumerate(sample_reviews)]
            )]
            additional = remaining.sample(min(len(remaining), sample_size - len(sample_reviews)))
            sample_reviews.extend(additional['text_clean'].tolist())
        
        return sample_reviews[:sample_size]
    
    def _get_cluster_sentiment_distribution(self, cluster_sentiments: List[str]) -> Dict[str, float]:
        """
        Get sentiment distribution for a cluster.
        
        Args:
            cluster_sentiments: Sentiment labels for the cluster
            
        Returns:
            Dictionary with sentiment distribution
        """
        sentiment_counts = pd.Series(cluster_sentiments).value_counts()
        total = len(cluster_sentiments)
        
        distribution = {}
        for sentiment in ['positive', 'negative', 'neutral']:
            count = sentiment_counts.get(sentiment, 0)
            distribution[sentiment] = (count / total) * 100
        
        return distribution
    
    def _parse_summary_response(self, response_text: str) -> Dict:
        """
        Parse the LLM response into structured format.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed summary dictionary
        """
        try:
            # Try to parse as JSON
            if response_text.startswith('{') and response_text.endswith('}'):
                parsed = json.loads(response_text)
                return {
                    'title': parsed.get('title', 'Unknown Theme'),
                    'summary': parsed.get('summary', 'No summary available')
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        lines = response_text.split('\n')
        title = 'Unknown Theme'
        summary = 'No summary available'
        
        for line in lines:
            line = line.strip()
            if 'title' in line.lower() and ':' in line:
                title = line.split(':', 1)[1].strip().strip('"')
            elif 'summary' in line.lower() and ':' in line:
                summary = line.split(':', 1)[1].strip().strip('"')
        
        return {
            'title': title,
            'summary': summary
        }
    
    def _get_fallback_summary(self, cluster_id: int, cluster_reviews: pd.DataFrame,
                            sentiment_dist: Dict[str, float]) -> Dict:
        """
        Generate a fallback summary when LLM fails.
        
        Args:
            cluster_id: Cluster identifier
            cluster_reviews: Reviews in the cluster
            sentiment_dist: Sentiment distribution
            
        Returns:
            Fallback summary dictionary
        """
        # Extract common themes from the actual review content
        common_themes = self._extract_common_themes(cluster_reviews)
        dominant_sentiment = max(sentiment_dist.items(), key=lambda x: x[1])[0]
        avg_rating = cluster_reviews['rating'].mean()
        
        if common_themes:
            # Use the most common theme as the title
            title = f"{common_themes[0].title()} Issues"
            summary = f"Customer feedback about {common_themes[0]} with {dominant_sentiment} sentiment (avg rating: {avg_rating:.1f})"
        else:
            title = f"Cluster {cluster_id} Theme"
            summary = f"Reviews with {dominant_sentiment} sentiment (avg rating: {avg_rating:.1f})"
        
        return {
            'title': title,
            'summary': summary,
            'cluster_id': cluster_id,
            'review_count': len(cluster_reviews),
            'volume_pct': (len(cluster_reviews) / len(cluster_reviews)) * 100,
            'positive_pct': sentiment_dist.get('positive', 0),
            'negative_pct': sentiment_dist.get('negative', 0),
            'neutral_pct': sentiment_dist.get('neutral', 0)
        }
    
    def _extract_common_themes(self, cluster_reviews: pd.DataFrame) -> List[str]:
        """
        Extract common themes from review text using keyword analysis.
        
        Args:
            cluster_reviews: Reviews in the cluster
            
        Returns:
            List of common themes found in the reviews
        """
        import re
        from collections import Counter
        
        # Define feature keywords
        feature_keywords = {
            'battery': ['battery', 'battery life', 'power', 'charge', 'drain'],
            'screen': ['screen', 'display', 'glare', 'reflection', 'brightness'],
            'pen': ['pen', 'stylus', 'latency', 'pressure', 'drawing'],
            'firmware': ['firmware', 'update', 'software', 'patch', 'bug'],
            'packaging': ['packaging', 'box', 'unboxing', 'arrived', 'shipping']
        }
        
        # Count occurrences of each feature in the reviews
        text_combined = ' '.join(cluster_reviews['text_clean'].astype(str)).lower()
        feature_counts = {}
        
        for feature, keywords in feature_keywords.items():
            count = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_combined)) 
                       for keyword in keywords)
            feature_counts[feature] = count
        
        # Return features that appear frequently (more than 2 times)
        common_features = [feature for feature, count in feature_counts.items() if count > 2]
        return sorted(common_features, key=lambda x: feature_counts[x], reverse=True) 