#!/usr/bin/env python3
"""
PrimeApple Review Insight Pipeline

End-to-end pipeline for transforming customer reviews into actionable insights.
"""

import os
import sys
import click
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """PrimeApple Review Insight Pipeline CLI"""
    pass


@cli.command()
@click.option('--model', default='gpt-4', help='LLM model to use')
@click.option('--temperature', default=0.3, type=float, help='LLM temperature')
@click.option('--max-clusters', default=10, type=int, help='Maximum number of themes')
@click.option('--min-clusters', default=5, type=int, help='Minimum number of themes')
@click.option('--force-recompute', is_flag=True, help='Force recompute embeddings')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def run(model, temperature, max_clusters, min_clusters, force_recompute, debug):
    """Run the complete insight pipeline"""
    
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting PrimeApple Review Insight Pipeline")
    
    # Initialize configuration
    config = Config(
        llm_model=model,
        temperature=temperature,
        max_clusters=max_clusters,
        min_clusters=min_clusters,
        force_recompute=force_recompute
    )
    
    try:
        # Step 1: Load and preprocess data
        logger.info("Step 1: Loading and preprocessing data")
        data_loader = DataLoader()
        reviews_df = data_loader.load_data('reviews.csv')
        logger.info(f"Loaded {len(reviews_df)} reviews")
        
        # Step 2: Generate embeddings
        logger.info("Step 2: Generating semantic embeddings")
        embedding_gen = EmbeddingGenerator(config)
        embeddings = embedding_gen.generate_embeddings(reviews_df)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Step 3: Discover themes
        logger.info("Step 3: Discovering themes via clustering")
        theme_discovery = ThemeDiscovery(config)
        clusters, optimal_k = theme_discovery.discover_themes(embeddings)
        logger.info(f"Discovered {optimal_k} themes")
        
        # Step 4: Analyze sentiment
        logger.info("Step 4: Analyzing sentiment")
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_results = sentiment_analyzer.analyze_sentiment(reviews_df)
        logger.info("Sentiment analysis complete")
        
        # Step 5: Generate theme summaries
        logger.info("Step 5: Generating theme summaries with LLM")
        llm_summarizer = LLMSummarizer(config)
        theme_summaries = llm_summarizer.generate_summaries(
            reviews_df, clusters, sentiment_results
        )
        logger.info("Theme summaries generated")
        
        # Step 6: Select representative quotes
        logger.info("Step 6: Selecting representative quotes")
        quote_selector = QuoteSelector()
        theme_quotes = quote_selector.select_quotes(
            reviews_df, clusters, sentiment_results, embeddings
        )
        logger.info("Representative quotes selected")
        
        # Step 7: Store results
        logger.info("Step 7: Storing results")
        storage_manager = StorageManager()
        run_id = storage_manager.store_results(
            reviews_df, clusters, theme_summaries, theme_quotes, 
            sentiment_results, config
        )
        logger.info(f"Results stored with run_id: {run_id}")
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        if debug:
            raise
        sys.exit(1)


@cli.command()
@click.option('--run-id', help='Specific run ID to show')
@click.option('--limit', default=10, help='Number of themes to show')
def show_results(run_id, limit):
    """Show pipeline results"""
    storage_manager = StorageManager()
    
    if run_id:
        results = storage_manager.get_run_results(run_id)
    else:
        results = storage_manager.get_latest_results()
    
    if not results:
        click.echo("No results found. Run the pipeline first.")
        return
    
    click.echo("\n" + "="*80)
    click.echo("PRIMEAPPLE REVIEW INSIGHT RESULTS")
    click.echo("="*80)
    
    for i, theme in enumerate(results['themes'][:limit], 1):
        click.echo(f"\n{i}. {theme['title']}")
        click.echo(f"   Summary: {theme['summary']}")
        click.echo(f"   Volume: {theme['volume_pct']:.1f}%")
        click.echo(f"   Sentiment: {theme['positive_pct']:.0f}% positive, "
                  f"{theme['negative_pct']:.0f}% negative, "
                  f"{theme['neutral_pct']:.0f}% neutral")
        
        click.echo("   Representative Quotes:")
        for j, quote in enumerate(theme['quotes'], 1):
                            click.echo(f"     {j}. \"{quote['text']}\" "
                      f"(Rating: {quote['rating']}, Sentiment: {quote['sentiment']})")
    
    click.echo(f"\nTotal themes: {len(results['themes'])}")
    click.echo(f"Run ID: {results['run_id']}")
    click.echo(f"Generated: {results['timestamp']}")


@cli.command()
@click.option('--format', 'export_format', default='csv', 
              type=click.Choice(['csv', 'json']), help='Export format')
@click.option('--run-id', help='Specific run ID to export')
@click.option('--output', default='insights_export', help='Output filename')
def export(export_format, run_id, output):
    """Export results to file"""
    storage_manager = StorageManager()
    
    if run_id:
        results = storage_manager.get_run_results(run_id)
    else:
        results = storage_manager.get_latest_results()
    
    if not results:
        click.echo("No results found. Run the pipeline first.")
        return
    
    if export_format == 'csv':
        storage_manager.export_to_csv(results, f"{output}.csv")
        click.echo(f"Results exported to {output}.csv")
    else:
        storage_manager.export_to_json(results, f"{output}.json")
        click.echo(f"Results exported to {output}.json")


@cli.command()
def list_runs():
    """List all pipeline runs"""
    storage_manager = StorageManager()
    runs = storage_manager.list_runs()
    
    if not runs:
        click.echo("No runs found.")
        return
    
    click.echo("\nPipeline Runs:")
    click.echo("-" * 80)
    for run in runs:
        click.echo(f"Run ID: {run['run_id']}")
        click.echo(f"Timestamp: {run['timestamp']}")
        click.echo(f"Model: {run['llm_model']}")
        click.echo(f"Themes: {run['num_clusters']}")
        click.echo(f"Reviews: {run['total_reviews']}")
        if run['notes']:
            click.echo(f"Notes: {run['notes']}")
        click.echo("-" * 80)


if __name__ == '__main__':
    cli() 