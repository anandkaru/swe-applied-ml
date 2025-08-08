#!/usr/bin/env python3
"""
Clustering Granularity Experiment
=================================

This experiment tests the impact of different K values in K-Means clustering
on theme quality, readability, and cost.

Hypothesis: Different values of k significantly impact the clarity and 
usefulness of resulting themes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
import time
from typing import Dict, List, Tuple
import logging

from src.config import Config
from src.data_loader import DataLoader
from src.embedding_generator import EmbeddingGenerator
from src.llm_summarizer import LLMSummarizer
from src.storage_manager import StorageManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringGranularityExperiment:
    """Experiment to test different K values in K-Means clustering."""
    
    def __init__(self, config: Config):
        self.config = config
        self.results = []
        
    def run_experiment(self, k_values: List[int] = [3, 5, 7, 9, 11]) -> Dict:
        """
        Run the clustering granularity experiment.
        
        Args:
            k_values: List of K values to test
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Starting clustering granularity experiment with K values: {k_values}")
        
        # Load data and generate embeddings (reuse from cache)
        data_loader = DataLoader()
        df = data_loader.load_data('reviews.csv')
        
        embedding_gen = EmbeddingGenerator(self.config)
        embeddings = embedding_gen.generate_embeddings(df)
        
        # Scale embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        
        # Test each K value
        for k in k_values:
            logger.info(f"Testing K = {k}")
            result = self._test_k_value(k, scaled_embeddings, df)
            self.results.append(result)
            
        return self._analyze_results()
    
    def _test_k_value(self, k: int, embeddings: np.ndarray, df: pd.DataFrame) -> Dict:
        """
        Test a specific K value and measure performance metrics.
        
        Args:
            k: Number of clusters
            embeddings: Scaled embeddings
            df: Original dataframe
            
        Returns:
            Dictionary with metrics for this K value
        """
        start_time = time.time()
        
        # Perform clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(embeddings, clusters)
        
        # Calculate cluster balance (standard deviation of cluster sizes)
        cluster_sizes = np.bincount(clusters)
        cluster_balance = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        
        # Generate theme summaries and measure token usage
        token_usage = self._measure_token_usage(k, clusters, df)
        
        # Calculate theme diversity (number of unique themes)
        theme_diversity = self._calculate_theme_diversity(clusters, df)
        
        execution_time = time.time() - start_time
        
        return {
            'k': k,
            'silhouette_score': silhouette_avg,
            'cluster_balance': cluster_balance,
            'token_usage': token_usage,
            'theme_diversity': theme_diversity,
            'execution_time': execution_time,
            'cluster_sizes': cluster_sizes.tolist()
        }
    
    def _measure_token_usage(self, k: int, clusters: np.ndarray, df: pd.DataFrame) -> int:
        """
        Measure token usage for generating theme summaries.
        
        Args:
            k: Number of clusters
            clusters: Cluster assignments
            df: Original dataframe
            
        Returns:
            Total token usage
        """
        try:
            llm_summarizer = LLMSummarizer(self.config)
            total_tokens = 0
            
            for cluster_id in range(k):
                cluster_mask = clusters == cluster_id
                cluster_reviews = df[cluster_mask]
                
                # Create a simple sentiment distribution for testing
                sentiment_dist = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
                
                # Generate summary and estimate tokens (simplified)
                # In a real implementation, you'd track actual token usage
                avg_review_length = cluster_reviews['text'].str.len().mean()
                estimated_tokens = int(avg_review_length / 4) + 200  # Rough estimation
                total_tokens += estimated_tokens
                
        except Exception as e:
            logger.warning(f"Could not measure token usage for K={k}: {e}")
            total_tokens = k * 500  # Fallback estimation
            
        return total_tokens
    
    def _calculate_theme_diversity(self, clusters: np.ndarray, df: pd.DataFrame) -> float:
        """
        Calculate theme diversity based on review content.
        
        Args:
            clusters: Cluster assignments
            df: Original dataframe
            
        Returns:
            Diversity score (0-1)
        """
        # Extract common words from each cluster
        cluster_keywords = []
        
        for cluster_id in range(len(np.unique(clusters))):
            cluster_mask = clusters == cluster_id
            cluster_reviews = df[cluster_mask]['text'].str.lower().str.cat(sep=' ')
            
            # Extract common words (simplified)
            words = cluster_reviews.split()
            word_counts = pd.Series(words).value_counts()
            top_words = word_counts.head(5).index.tolist()
            cluster_keywords.append(set(top_words))
        
        # Calculate overlap between clusters
        total_overlap = 0
        comparisons = 0
        
        for i in range(len(cluster_keywords)):
            for j in range(i + 1, len(cluster_keywords)):
                overlap = len(cluster_keywords[i] & cluster_keywords[j])
                total_overlap += overlap
                comparisons += 1
        
        avg_overlap = total_overlap / comparisons if comparisons > 0 else 0
        diversity = 1 - (avg_overlap / 5)  # Normalize by max possible overlap
        
        return max(0, min(1, diversity))
    
    def _analyze_results(self) -> Dict:
        """
        Analyze and summarize experiment results.
        
        Returns:
            Dictionary with analysis results
        """
        results_df = pd.DataFrame(self.results)
        
        # Find optimal K based on different criteria
        best_silhouette = results_df.loc[results_df['silhouette_score'].idxmax()]
        best_balance = results_df.loc[results_df['cluster_balance'].idxmax()]
        best_diversity = results_df.loc[results_df['theme_diversity'].idxmax()]
        lowest_cost = results_df.loc[results_df['token_usage'].idxmin()]
        
        analysis = {
            'results': self.results,
            'summary': {
                'best_silhouette_k': int(best_silhouette['k']),
                'best_balance_k': int(best_balance['k']),
                'best_diversity_k': int(best_diversity['k']),
                'lowest_cost_k': int(lowest_cost['k']),
                'recommended_k': self._get_recommended_k(results_df)
            },
            'metrics': {
                'silhouette_scores': results_df['silhouette_score'].tolist(),
                'cluster_balances': results_df['cluster_balance'].tolist(),
                'token_usage': results_df['token_usage'].tolist(),
                'theme_diversity': results_df['theme_diversity'].tolist()
            }
        }
        
        return analysis
    
    def _get_recommended_k(self, results_df: pd.DataFrame) -> int:
        """
        Get recommended K value based on multiple criteria.
        
        Args:
            results_df: DataFrame with experiment results
            
        Returns:
            Recommended K value
        """
        # Normalize metrics to 0-1 scale
        normalized = results_df.copy()
        normalized['silhouette_norm'] = (results_df['silhouette_score'] - results_df['silhouette_score'].min()) / (results_df['silhouette_score'].max() - results_df['silhouette_score'].min())
        normalized['balance_norm'] = (results_df['cluster_balance'] - results_df['cluster_balance'].min()) / (results_df['cluster_balance'].max() - results_df['cluster_balance'].min())
        normalized['diversity_norm'] = (results_df['theme_diversity'] - results_df['theme_diversity'].min()) / (results_df['theme_diversity'].max() - results_df['theme_diversity'].min())
        normalized['cost_norm'] = 1 - (results_df['token_usage'] - results_df['token_usage'].min()) / (results_df['token_usage'].max() - results_df['token_usage'].min())
        
        # Calculate composite score (weighted average)
        normalized['composite_score'] = (
            normalized['silhouette_norm'] * 0.3 +
            normalized['balance_norm'] * 0.2 +
            normalized['diversity_norm'] * 0.3 +
            normalized['cost_norm'] * 0.2
        )
        
        best_k = int(normalized.loc[normalized['composite_score'].idxmax(), 'k'])
        return best_k
    
    def create_visualizations(self, analysis: Dict, output_dir: str = "experiment_results"):
        """
        Create visualizations for the experiment results.
        
        Args:
            analysis: Analysis results dictionary
            output_dir: Directory to save visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results_df = pd.DataFrame(analysis['results'])
        k_values = results_df['k'].tolist()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clustering Granularity Experiment Results', fontsize=16, fontweight='bold')
        
        # 1. Silhouette Score
        axes[0, 0].plot(k_values, results_df['silhouette_score'], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Silhouette Score vs K')
        axes[0, 0].set_xlabel('Number of Clusters (K)')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(x=analysis['summary']['best_silhouette_k'], color='red', linestyle='--', alpha=0.7, label=f'Best: K={analysis["summary"]["best_silhouette_k"]}')
        axes[0, 0].legend()
        
        # 2. Cluster Balance
        axes[0, 1].plot(k_values, results_df['cluster_balance'], 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Cluster Balance vs K')
        axes[0, 1].set_xlabel('Number of Clusters (K)')
        axes[0, 1].set_ylabel('Cluster Balance (1 - CV)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=analysis['summary']['best_balance_k'], color='red', linestyle='--', alpha=0.7, label=f'Best: K={analysis["summary"]["best_balance_k"]}')
        axes[0, 1].legend()
        
        # 3. Token Usage
        axes[1, 0].plot(k_values, results_df['token_usage'], 'ro-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Token Usage vs K')
        axes[1, 0].set_xlabel('Number of Clusters (K)')
        axes[1, 0].set_ylabel('Estimated Token Usage')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(x=analysis['summary']['lowest_cost_k'], color='red', linestyle='--', alpha=0.7, label=f'Lowest: K={analysis["summary"]["lowest_cost_k"]}')
        axes[1, 0].legend()
        
        # 4. Theme Diversity
        axes[1, 1].plot(k_values, results_df['theme_diversity'], 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Theme Diversity vs K')
        axes[1, 1].set_xlabel('Number of Clusters (K)')
        axes[1, 1].set_ylabel('Theme Diversity Score')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(x=analysis['summary']['best_diversity_k'], color='red', linestyle='--', alpha=0.7, label=f'Best: K={analysis["summary"]["best_diversity_k"]}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/clustering_granularity_results.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure instead of showing it
        
        # Save results to JSON (convert numpy types to native Python types)
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        analysis_serializable = convert_numpy_types(analysis)
        with open(f'{output_dir}/experiment_results.json', 'w') as f:
            json.dump(analysis_serializable, f, indent=2)
        
        # Create results table
        results_table = results_df[['k', 'silhouette_score', 'cluster_balance', 'token_usage', 'theme_diversity']].round(4)
        results_table.to_csv(f'{output_dir}/results_table.csv', index=False)
        
        logger.info(f"Visualizations saved to {output_dir}/")

def main():
    """Run the clustering granularity experiment."""
    config = Config()
    
    # Create experiment
    experiment = ClusteringGranularityExperiment(config)
    
    # Run experiment with different K values
    k_values = [3, 5, 7, 9, 11]
    analysis = experiment.run_experiment(k_values)
    
    # Create visualizations
    experiment.create_visualizations(analysis)
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING GRANULARITY EXPERIMENT RESULTS")
    print("="*60)
    
    summary = analysis['summary']
    print(f"\n📊 EXPERIMENT SUMMARY:")
    print(f"   • Best Silhouette Score: K = {summary['best_silhouette_k']}")
    print(f"   • Best Cluster Balance: K = {summary['best_balance_k']}")
    print(f"   • Best Theme Diversity: K = {summary['best_diversity_k']}")
    print(f"   • Lowest Token Cost: K = {summary['lowest_cost_k']}")
    print(f"   • 🎯 RECOMMENDED K: {summary['recommended_k']}")
    
    print(f"\n📈 DETAILED RESULTS:")
    results_df = pd.DataFrame(analysis['results'])
    print(results_df[['k', 'silhouette_score', 'cluster_balance', 'token_usage', 'theme_diversity']].to_string(index=False))
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Based on the experiment results, we recommend using K = {summary['recommended_k']}")
    print(f"   This provides the best balance of clustering quality, theme diversity, and cost efficiency.")
    
    return analysis

if __name__ == "__main__":
    main() 