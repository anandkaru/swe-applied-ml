"""
Theme discovery using clustering for the PrimeApple Review Insight Pipeline.
"""

import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import os

logger = logging.getLogger(__name__)


class ThemeDiscovery:
    """Discovers themes in reviews using clustering."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logger
        self.scaler = StandardScaler()
        self.optimal_k = None
        self.cluster_model = None
    
    def discover_themes(self, embeddings: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Discover themes using clustering.
        
        Args:
            embeddings: Review embeddings
            
        Returns:
            Tuple of (cluster_labels, optimal_k)
        """
        self.logger.info("Starting theme discovery")
        
        # Scale embeddings
        scaled_embeddings = self.scaler.fit_transform(embeddings)
        
        # Find optimal number of clusters
        optimal_k = self._find_optimal_k(scaled_embeddings)
        self.optimal_k = optimal_k
        
        self.logger.info(f"Optimal number of clusters: {optimal_k}")
        
        # Perform clustering
        cluster_labels = self._perform_clustering(scaled_embeddings, optimal_k)
        
        # Log cluster statistics
        self._log_cluster_stats(cluster_labels)
        
        return cluster_labels, optimal_k
    
    def _find_optimal_k(self, embeddings: np.ndarray) -> int:
        """
        Find optimal number of clusters using silhouette score and elbow method.
        
        Args:
            embeddings: Scaled embeddings
            
        Returns:
            Optimal number of clusters
        """
        k_range = range(self.config.min_clusters, self.config.max_clusters + 1)
        silhouette_scores = []
        inertias = []
        
        self.logger.info(f"Testing K values from {self.config.min_clusters} to {self.config.max_clusters}")
        
        for k in k_range:
            # Perform clustering
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.config.random_seed,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate metrics
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
            
            self.logger.debug(f"K={k}: Silhouette={silhouette_avg:.3f}, Inertia={kmeans.inertia_:.0f}")
        
        # Find optimal K using silhouette score
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
        # Find optimal K using elbow method
        optimal_k_elbow = self._find_elbow_point(k_range, inertias)
        
        # Choose the K that appears in both methods or prefer silhouette
        if optimal_k_silhouette == optimal_k_elbow:
            optimal_k = optimal_k_silhouette
        else:
            # Prefer silhouette score if it's above threshold
            max_silhouette = max(silhouette_scores)
            if max_silhouette >= self.config.silhouette_threshold:
                optimal_k = optimal_k_silhouette
            else:
                optimal_k = optimal_k_elbow
        
        self.logger.info(f"Silhouette-based K: {optimal_k_silhouette}")
        self.logger.info(f"Elbow-based K: {optimal_k_elbow}")
        self.logger.info(f"Selected K: {optimal_k}")
        
        return optimal_k
    
    def _find_elbow_point(self, k_range: range, inertias: List[float]) -> int:
        """
        Find elbow point in inertia curve.
        
        Args:
            k_range: Range of K values
            inertias: List of inertia values
            
        Returns:
            K value at elbow point
        """
        # Calculate second derivative
        second_derivative = np.diff(np.diff(inertias))
        
        # Find the point with maximum second derivative (elbow)
        elbow_idx = np.argmax(second_derivative) + 1
        
        return k_range[elbow_idx]
    
    def _perform_clustering(self, embeddings: np.ndarray, k: int) -> np.ndarray:
        """
        Perform K-means clustering.
        
        Args:
            embeddings: Scaled embeddings
            k: Number of clusters
            
        Returns:
            Cluster labels
        """
        self.cluster_model = KMeans(
            n_clusters=k,
            random_state=self.config.random_seed,
            n_init=10
        )
        
        cluster_labels = self.cluster_model.fit_predict(embeddings)
        
        return cluster_labels
    
    def _log_cluster_stats(self, cluster_labels: np.ndarray):
        """
        Log statistics about the clusters.
        
        Args:
            cluster_labels: Cluster assignments
        """
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        self.logger.info("Cluster sizes:")
        for label, count in zip(unique_labels, counts):
            percentage = (count / len(cluster_labels)) * 100
            self.logger.info(f"  Cluster {label}: {count} reviews ({percentage:.1f}%)")
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centers.
        
        Returns:
            Cluster centers array
        """
        if self.cluster_model is None:
            raise ValueError("Clustering must be performed first")
        
        return self.cluster_model.cluster_centers_
    
    def get_cluster_quality_metrics(self, embeddings: np.ndarray, 
                                  cluster_labels: np.ndarray) -> dict:
        """
        Get quality metrics for the clustering.
        
        Args:
            embeddings: Scaled embeddings
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary with quality metrics
        """
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        
        return {
            'silhouette_score': silhouette_avg,
            'num_clusters': len(np.unique(cluster_labels)),
            'cluster_sizes': np.bincount(cluster_labels).tolist(),
            'cluster_size_std': float(np.std(np.bincount(cluster_labels)))
        } 