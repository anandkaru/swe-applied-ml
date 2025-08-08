"""
Storage management for the PrimeApple Review Insight Pipeline.
"""

import sqlite3
import pandas as pd
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage and retrieval of pipeline results."""
    
    def __init__(self, db_path: str = "insights.db"):
        self.db_path = db_path
        self.logger = logger
        self._init_database()
    
    def _init_database(self):
        """Initialize the database with required tables."""
        self.logger.info(f"Initializing database: {self.db_path}")
        
        with sqlite3.connect(self.db_path) as conn:
            # Create runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    llm_model TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    num_clusters INTEGER NOT NULL,
                    total_reviews INTEGER NOT NULL,
                    token_cost REAL,
                    notes TEXT
                )
            """)
            
            # Create clusters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    theme_id INTEGER,
                    run_id TEXT,
                    product TEXT,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    volume_pct REAL NOT NULL,
                    positive_pct REAL NOT NULL,
                    negative_pct REAL NOT NULL,
                    neutral_pct REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (theme_id, run_id),
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
            """)
            
            # Create quotes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quotes (
                    quote_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theme_id INTEGER,
                    run_id TEXT,
                    review_text TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    sentiment TEXT NOT NULL,
                    created_at TEXT,
                    FOREIGN KEY (theme_id, run_id) REFERENCES clusters (theme_id, run_id)
                )
            """)
            
            conn.commit()
    
    def store_results(self, df: pd.DataFrame, clusters: np.ndarray, 
                     theme_summaries: List[Dict], theme_quotes: Dict[int, List[Dict]],
                     sentiment_results: Dict, config) -> str:
        """
        Store pipeline results in the database.
        
        Args:
            df: Review DataFrame
            clusters: Cluster assignments
            theme_summaries: Theme summaries
            theme_quotes: Theme quotes
            sentiment_results: Sentiment analysis results
            config: Pipeline configuration
            
        Returns:
            Run ID
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Storing results with run_id: {run_id}")
        
        with sqlite3.connect(self.db_path) as conn:
            # Store run metadata
            self._store_run_metadata(conn, run_id, df, clusters, config)
            
            # Store theme summaries
            self._store_theme_summaries(conn, run_id, theme_summaries, df, clusters, sentiment_results)
            
            # Store quotes
            self._store_quotes(conn, run_id, theme_quotes)
            
            conn.commit()
        
        self.logger.info(f"Results stored successfully for run_id: {run_id}")
        
        return run_id
    
    def _store_run_metadata(self, conn: sqlite3.Connection, run_id: str, 
                           df: pd.DataFrame, clusters: np.ndarray, config):
        """Store run metadata."""
        conn.execute("""
            INSERT INTO runs (run_id, timestamp, llm_model, temperature, 
                            num_clusters, total_reviews, token_cost, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            datetime.now().isoformat(),
            config.llm_model,
            config.temperature,
            len(np.unique(clusters)),
            len(df),
            None,  # TODO: Calculate token cost
            f"Pipeline run with {len(np.unique(clusters))} themes"
        ))
    
    def _store_theme_summaries(self, conn: sqlite3.Connection, run_id: str,
                              theme_summaries: List[Dict], df: pd.DataFrame,
                              clusters: np.ndarray, sentiment_results: Dict):
        """Store theme summaries."""
        for theme in theme_summaries:
            cluster_id = theme['cluster_id']
            
            # Get cluster statistics
            cluster_mask = clusters == cluster_id
            cluster_reviews = df[cluster_mask]
            cluster_sentiments = [sentiment_results['sentiments'][i] 
                                for i in range(len(df)) if clusters[i] == cluster_id]
            
            # Calculate volume percentage
            volume_pct = (len(cluster_reviews) / len(df)) * 100
            
            # Get dominant product
            product_dist = cluster_reviews['product'].value_counts()
            dominant_product = product_dist.index[0] if len(product_dist) > 0 else "Unknown"
            
            conn.execute("""
                INSERT INTO clusters (theme_id, run_id, product, title, summary,
                                    volume_pct, positive_pct, negative_pct, neutral_pct, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cluster_id,
                run_id,
                dominant_product,
                theme['title'],
                theme['summary'],
                volume_pct,
                theme.get('positive_pct', 0),
                theme.get('negative_pct', 0),
                theme.get('neutral_pct', 0),
                datetime.now().isoformat()
            ))
    
    def _store_quotes(self, conn: sqlite3.Connection, run_id: str, 
                     theme_quotes: Dict[int, List[Dict]]):
        """Store quotes."""
        for cluster_id, quotes in theme_quotes.items():
            for quote in quotes:
                conn.execute("""
                    INSERT INTO quotes (theme_id, run_id, review_text, rating, 
                                      sentiment, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    cluster_id,
                    run_id,
                    quote['text'],
                    quote['rating'],
                    quote['sentiment'],
                    quote.get('created_at')
                ))
    
    def get_latest_results(self) -> Optional[Dict]:
        """Get the most recent pipeline results."""
        with sqlite3.connect(self.db_path) as conn:
            # Get latest run
            cursor = conn.execute("""
                SELECT run_id, timestamp FROM runs 
                ORDER BY timestamp DESC LIMIT 1
            """)
            result = cursor.fetchone()
            
            if not result:
                return None
            
            run_id, timestamp = result
            return self.get_run_results(run_id)
    
    def get_run_results(self, run_id: str) -> Optional[Dict]:
        """Get results for a specific run."""
        with sqlite3.connect(self.db_path) as conn:
            # Get run metadata
            cursor = conn.execute("""
                SELECT * FROM runs WHERE run_id = ?
            """, (run_id,))
            run_data = cursor.fetchone()
            
            if not run_data:
                return None
            
            # Get themes
            cursor = conn.execute("""
                SELECT * FROM clusters WHERE run_id = ? ORDER BY volume_pct DESC
            """, (run_id,))
            themes_data = cursor.fetchall()
            
            # Get quotes for each theme
            themes = []
            for theme_row in themes_data:
                theme_id = theme_row[0]
                
                cursor = conn.execute("""
                    SELECT * FROM quotes WHERE theme_id = ? AND run_id = ?
                    ORDER BY quote_id
                """, (theme_id, run_id))
                quotes_data = cursor.fetchall()
                
                # Convert to dictionaries
                quotes = []
                for quote_row in quotes_data:
                    quotes.append({
                        'text': quote_row[3],
                        'rating': quote_row[4],
                        'sentiment': quote_row[5],
                        'created_at': quote_row[6]
                    })
                
                themes.append({
                    'theme_id': theme_id,
                    'title': theme_row[3],
                    'summary': theme_row[4],
                    'volume_pct': theme_row[5],
                    'positive_pct': theme_row[6],
                    'negative_pct': theme_row[7],
                    'neutral_pct': theme_row[8],
                    'quotes': quotes
                })
            
            return {
                'run_id': run_id,
                'timestamp': run_data[1],
                'llm_model': run_data[2],
                'temperature': run_data[3],
                'num_clusters': run_data[4],
                'total_reviews': run_data[5],
                'themes': themes
            }
    
    def list_runs(self) -> List[Dict]:
        """List all pipeline runs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT run_id, timestamp, llm_model, temperature, 
                       num_clusters, total_reviews, notes
                FROM runs ORDER BY timestamp DESC
            """)
            
            runs = []
            for row in cursor.fetchall():
                runs.append({
                    'run_id': row[0],
                    'timestamp': row[1],
                    'llm_model': row[2],
                    'temperature': row[3],
                    'num_clusters': row[4],
                    'total_reviews': row[5],
                    'notes': row[6]
                })
            
            return runs
    
    def export_to_csv(self, results: Dict, filename: str):
        """Export results to CSV format."""
        if not results or 'themes' not in results:
            raise ValueError("No results to export")
        
        # Create DataFrame for themes
        themes_data = []
        for theme in results['themes']:
            theme_row = {
                'theme_id': theme['theme_id'],
                'title': theme['title'],
                'summary': theme['summary'],
                'volume_pct': theme['volume_pct'],
                'positive_pct': theme['positive_pct'],
                'negative_pct': theme['negative_pct'],
                'neutral_pct': theme['neutral_pct']
            }
            
            # Add quotes
            for i, quote in enumerate(theme['quotes'], 1):
                theme_row[f'quote_{i}_text'] = quote['text']
                theme_row[f'quote_{i}_rating'] = quote['rating']
                theme_row[f'quote_{i}_sentiment'] = quote['sentiment']
            
            themes_data.append(theme_row)
        
        # Create DataFrame and export
        df = pd.DataFrame(themes_data)
        df.to_csv(filename, index=False)
        
        self.logger.info(f"Results exported to {filename}")
    
    def export_to_json(self, results: Dict, filename: str):
        """Export results to JSON format."""
        if not results:
            raise ValueError("No results to export")
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {filename}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Count records in each table
            cursor = conn.execute("SELECT COUNT(*) FROM runs")
            runs_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM clusters")
            clusters_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM quotes")
            quotes_count = cursor.fetchone()[0]
            
            return {
                'runs_count': runs_count,
                'clusters_count': clusters_count,
                'quotes_count': quotes_count,
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024)
            } 