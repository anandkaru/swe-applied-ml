#!/usr/bin/env python3
"""
Demo script for the PrimeApple Review Insight Pipeline.

This script runs the pipeline on a small subset of data for testing purposes.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.embedding_generator import EmbeddingGenerator
from src.theme_discovery import ThemeDiscovery
from src.llm_summarizer import LLMSummarizer
from src.sentiment_analyzer import SentimentAnalyzer
from src.quote_selector import QuoteSelector
from src.storage_manager import StorageManager
from src.config import Config


def run_demo():
    """Run a demo of the pipeline with a small dataset."""
    
    print("🚀 PrimeApple Review Insight Pipeline Demo")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize configuration for demo
    config = Config(
        llm_model="gpt-3.5-turbo",  # Use cheaper model for demo
        temperature=0.3,
        max_clusters=5,
        min_clusters=3,
        force_recompute=True
    )
    
    try:
        # Step 1: Load and preprocess data
        print("\n📊 Step 1: Loading and preprocessing data")
        data_loader = DataLoader()
        reviews_df = data_loader.load_data('reviews.csv')
        
        # Use only first 100 reviews for demo
        demo_df = reviews_df.head(100).copy()
        print(f"Using {len(demo_df)} reviews for demo")
        
        # Step 2: Generate embeddings
        print("\n🔤 Step 2: Generating semantic embeddings")
        embedding_gen = EmbeddingGenerator(config)
        embeddings = embedding_gen.generate_embeddings(demo_df)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Step 3: Discover themes
        print("\n🎯 Step 3: Discovering themes via clustering")
        theme_discovery = ThemeDiscovery(config)
        clusters, optimal_k = theme_discovery.discover_themes(embeddings)
        print(f"Discovered {optimal_k} themes")
        
        # Step 4: Analyze sentiment
        print("\n😊 Step 4: Analyzing sentiment")
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_results = sentiment_analyzer.analyze_sentiment(demo_df)
        print("Sentiment analysis complete")
        
        # Step 5: Generate theme summaries
        print("\n🤖 Step 5: Generating theme summaries with LLM")
        llm_summarizer = LLMSummarizer(config)
        theme_summaries = llm_summarizer.generate_summaries(
            demo_df, clusters, sentiment_results
        )
        print("Theme summaries generated")
        
        # Step 6: Select representative quotes
        print("\n💬 Step 6: Selecting representative quotes")
        quote_selector = QuoteSelector()
        theme_quotes = quote_selector.select_quotes(
            demo_df, clusters, sentiment_results
        )
        print("Representative quotes selected")
        
        # Step 7: Store results
        print("\n💾 Step 7: Storing results")
        storage_manager = StorageManager("demo_insights.db")
        run_id = storage_manager.store_results(
            demo_df, clusters, theme_summaries, theme_quotes, 
            sentiment_results, config
        )
        print(f"Results stored with run_id: {run_id}")
        
        # Display results
        print("\n" + "=" * 50)
        print("📈 DEMO RESULTS")
        print("=" * 50)
        
        results = storage_manager.get_run_results(run_id)
        if results:
            for i, theme in enumerate(results['themes'], 1):
                print(f"\n{i}. {theme['title']}")
                print(f"   Summary: {theme['summary']}")
                print(f"   Volume: {theme['volume_pct']:.1f}%")
                print(f"   Sentiment: {theme['positive_pct']:.0f}% positive, "
                      f"{theme['negative_pct']:.0f}% negative, "
                      f"{theme['neutral_pct']:.0f}% neutral")
                
                print("   Representative Quotes:")
                for j, quote in enumerate(theme['quotes'], 1):
                    print(f"     {j}. \"{quote['text'][:80]}...\" "
                          f"(Rating: {quote['rating']}, Sentiment: {quote['sentiment']})")
        
        print(f"\n✅ Demo completed successfully!")
        print(f"Database: demo_insights.db")
        print(f"Run ID: {run_id}")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demo() 