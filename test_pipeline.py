#!/usr/bin/env python3
"""
Test script for the PrimeApple Review Insight Pipeline.

This script tests individual components without requiring OpenAI API.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.embedding_generator import EmbeddingGenerator
from src.theme_discovery import ThemeDiscovery
from src.sentiment_analyzer import SentimentAnalyzer
from src.quote_selector import QuoteSelector
from src.storage_manager import StorageManager
from src.config import Config


def test_data_loading():
    """Test data loading and preprocessing."""
    print("🧪 Testing data loading...")
    
    data_loader = DataLoader()
    df = data_loader.load_data('reviews.csv')
    
    assert len(df) > 0, "DataFrame should not be empty"
    assert 'text_clean' in df.columns, "Should have cleaned text column"
    assert df['text_clean'].str.len().min() > 10, "Should have meaningful text length"
    
    print(f"✅ Data loading test passed: {len(df)} reviews loaded")
    return df


def test_embedding_generation(df):
    """Test embedding generation."""
    print("🧪 Testing embedding generation...")
    
    # Use unique cache path for testing to avoid conflicts
    config = Config(llm_model="gpt-4", force_recompute=True)
    config.cache_dir = "test_cache"
    os.makedirs(config.cache_dir, exist_ok=True)
    
    embedding_gen = EmbeddingGenerator(config)
    test_df = df.head(50)  # Test with subset
    embeddings = embedding_gen.generate_embeddings(test_df)
    
    assert embeddings.shape[0] == len(test_df), f"Should have embeddings for all reviews, got {embeddings.shape[0]} for {len(test_df)} reviews"
    assert embeddings.shape[1] > 0, "Should have embedding dimensions"
    
    print(f"✅ Embedding generation test passed: {embeddings.shape}")
    
    # Clean up test cache
    import shutil
    if os.path.exists(config.cache_dir):
        shutil.rmtree(config.cache_dir)
    
    return embeddings


def test_theme_discovery(embeddings):
    """Test theme discovery."""
    print("🧪 Testing theme discovery...")
    
    config = Config(min_clusters=2, max_clusters=4, llm_model="gpt-4")
    theme_discovery = ThemeDiscovery(config)
    clusters, optimal_k = theme_discovery.discover_themes(embeddings)
    
    assert len(clusters) == len(embeddings), "Should have cluster for each review"
    assert optimal_k >= 2 and optimal_k <= 4, "Should be within specified range"
    assert len(np.unique(clusters)) == optimal_k, "Should have correct number of clusters"
    
    print(f"✅ Theme discovery test passed: {optimal_k} clusters found")
    return clusters, optimal_k


def test_sentiment_analysis(df):
    """Test sentiment analysis."""
    print("🧪 Testing sentiment analysis...")
    
    sentiment_analyzer = SentimentAnalyzer()
    results = sentiment_analyzer.analyze_sentiment(df.head(20))  # Test with subset
    
    assert len(results['sentiments']) == 20, "Should have sentiment for each review"
    assert all(s in ['positive', 'negative', 'neutral'] for s in results['sentiments']), \
        "Should have valid sentiment labels"
    
    print(f"✅ Sentiment analysis test passed: {len(results['sentiments'])} reviews analyzed")
    return results


def test_quote_selection(df, clusters, sentiment_results):
    """Test quote selection."""
    print("🧪 Testing quote selection...")
    
    quote_selector = QuoteSelector()
    # Use the same subset that was used for sentiment analysis
    test_df = df.head(20)
    test_clusters = clusters[:20]  # Use first 20 clusters
    theme_quotes = quote_selector.select_quotes(test_df, test_clusters, sentiment_results)
    
    assert len(theme_quotes) > 0, "Should have quotes for themes"
    for cluster_id, quotes in theme_quotes.items():
        assert len(quotes) <= 3, "Should have at most 3 quotes per theme"
        for quote in quotes:
            assert 'text' in quote, "Quote should have text"
            assert 'rating' in quote, "Quote should have rating"
            assert 'sentiment' in quote, "Quote should have sentiment"
    
    print(f"✅ Quote selection test passed: {len(theme_quotes)} themes with quotes")
    return theme_quotes


def test_storage(df, clusters, sentiment_results, theme_quotes):
    """Test storage functionality."""
    print("🧪 Testing storage...")
    
    storage_manager = StorageManager("test_insights.db")
    
    # Use the same subset as quote selection
    test_df = df.head(20)
    test_clusters = clusters[:20]
    
    # Create dummy theme summaries for testing
    theme_summaries = []
    for cluster_id in np.unique(test_clusters):
        theme_summaries.append({
            'cluster_id': cluster_id,
            'title': f'Test Theme {cluster_id}',
            'summary': f'Test summary for theme {cluster_id}',
            'positive_pct': 50.0,
            'negative_pct': 30.0,
            'neutral_pct': 20.0
        })
    
    config = Config(llm_model="gpt-4")
    run_id = storage_manager.store_results(
        test_df, test_clusters, theme_summaries, theme_quotes, 
        sentiment_results, config
    )
    
    # Test retrieval
    results = storage_manager.get_run_results(run_id)
    assert results is not None, "Should be able to retrieve results"
    assert 'themes' in results, "Should have themes in results"
    
    print(f"✅ Storage test passed: run_id {run_id}")
    
    # Clean up test database
    if os.path.exists("test_insights.db"):
        os.remove("test_insights.db")
    
    return results


def run_tests():
    """Run all tests."""
    print("🚀 Running PrimeApple Review Insight Pipeline Tests")
    print("=" * 60)
    
    try:
        # Test each component
        df = test_data_loading()
        embeddings = test_embedding_generation(df)
        clusters, optimal_k = test_theme_discovery(embeddings)
        sentiment_results = test_sentiment_analysis(df)
        theme_quotes = test_quote_selection(df.head(50), clusters, sentiment_results)
        results = test_storage(df, clusters, sentiment_results, theme_quotes)
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Pipeline components are working correctly.")
        print("=" * 60)
        
        # Show sample results
        print("\n📊 Sample Results:")
        if results and 'themes' in results:
            for i, theme in enumerate(results['themes'][:2], 1):
                print(f"\n{i}. {theme['title']}")
                print(f"   Summary: {theme['summary']}")
                print(f"   Volume: {theme['volume_pct']:.1f}%")
                print(f"   Quotes: {len(theme['quotes'])}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 