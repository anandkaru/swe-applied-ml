# PrimeApple Review Insight Pipeline

An end-to-end, reproducible insight pipeline that transforms raw customer reviews into actionable, executive-ready summaries for PrimeApple's EchoPad and EchoPad Pro products.

## 📚 Table of Contents

- [Overview](#-overview)
- [Architecture](#️-architecture)
- [Output Schema](#-output-schema)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Best Practices](#-best-practices)
- [Troubleshooting](#-troubleshooting)

## 📋 Related Documentation

| Document | Description |
|----------|-------------|
| [🚀 Quick Start Guide](QUICK_START.md) | Fast setup and basic usage instructions |
| [📊 Pipeline Summary](PIPELINE_SUMMARY.md) | Technical overview and component details |
| [✅ Deliverables Checklist](DELIVERABLES_CHECKLIST.md) | Complete list of project deliverables |
| [📈 Delivery Summary](DELIVERY_SUMMARY.md) | Final delivery confirmation and status |
| [🧪 Experiment Presentation](EXPERIMENT_PRESENTATION.md) | 4-slide experiment results presentation |
| [🔬 Experiment Summary](EXPERIMENT_SUMMARY.md) | Detailed clustering granularity experiment analysis |


## 🎯 Overview

This pipeline processes customer reviews to identify key themes, sentiment patterns, and representative quotes that help leadership understand what customers love or dislike about their products.

## 🏗️ Architecture

### Pipeline Components

1. **Data Loading & Preprocessing**: Loads and cleans review data
2. **Semantic Embedding**: Converts reviews to semantic vectors using CPU-friendly models
3. **Theme Discovery**: Clusters reviews into meaningful themes using K-Means
4. **LLM Summarization**: Generates theme titles and summaries using OpenAI
5. **Sentiment Analysis**: Analyzes sentiment at review and theme levels
6. **Quote Selection**: Automatically selects representative quotes per theme
7. **Storage & Persistence**: Saves all artifacts to SQLite database

### Key Design Decisions

- [Semantic Representation](#semantic-representation)
- [Theme Discovery](#theme-discovery)
- [LLM Prompting Strategy](#llm-prompting-strategy)
- [Quote Selection Strategy](#quote-selection-strategy)
- [Sentiment Analysis Improvements](#sentiment-analysis-improvements)
- [Cache Management](#cache-management)

#### Semantic Representation
- **Model**: `all-MiniLM-L6-v2` from Sentence Transformers
- **Rationale**: CPU-friendly, fast inference, good semantic understanding
- **Persistence**: Embeddings cached to disk for reproducible runs

#### Theme Discovery
- **Algorithm**: K-Means clustering with elbow method for optimal K
- **Determinism**: Fixed random seed (42) for reproducible results
- **K Selection**: Automated using silhouette score and elbow method

#### LLM Prompting Strategy
- **Model**: GPT-4 for high-quality summaries (compatible with OpenAI API v1.0+)
- **Temperature**: 0.3 for consistent, focused outputs
- **Prompt Design**: 
  - Clear context about the business goal
  - Specific constraints (word limits, format)
  - Examples of good vs bad summaries
  - Emphasis on actionable insights
- **Fallback System**: Intelligent content-based theme extraction when LLM fails
- **API Compatibility**: Updated for OpenAI v1.0+ API format

#### Quote Selection Strategy
- **Criteria**: 
  1. Sentiment diversity (positive, negative, neutral)
  2. Length appropriateness (not too short/long)
  3. Specificity (mentions concrete features)
  4. Clarity (well-written, understandable)
  5. Uniqueness (avoid repetitive quotes)
- **Algorithm**: Multi-criteria scoring with diversity enforcement

#### Sentiment Analysis Improvements
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Rating-Aware Calibration**: Adjusts predictions based on review ratings for improved accuracy
- **Confidence Thresholding**: Low-confidence predictions calibrated using rating context
- **Calibration Rules**:
  - High ratings (4-5) + negative prediction (low confidence) → positive
  - Low ratings (1-2) + positive prediction (low confidence) → negative  
  - Middle rating (3) + any prediction (low confidence) → neutral

#### Cache Management
- **Incremental Updates**: Pipeline caches embeddings, clusters, and models for future incremental updates
- **Cache Storage**: Joblib-based caching in `cache/` directory
- **Run Versioning**: Each pipeline run cached with unique run_id for reproducibility

## 📊 Output Schema

### Database Tables

#### `clusters`
- `theme_id`: Unique theme identifier
- `product`: Product name (EchoPad/EchoPad Pro)
- `title`: Theme title (≤8 words)
- `summary`: Theme explanation (≤50 words)
- `volume_pct`: % of reviews in theme
- `positive_pct`: % positive sentiment
- `negative_pct`: % negative sentiment
- `neutral_pct`: % neutral sentiment
- `created_at`: Timestamp

#### `quotes`
- `quote_id`: Unique quote identifier
- `theme_id`: Reference to cluster
- `review_text`: Original review text
- `rating`: Original rating (1-5)
- `sentiment`: Sentiment label
- `created_at`: Review timestamp

#### `runs`
- `run_id`: Unique run identifier
- `timestamp`: Execution timestamp
- `llm_model`: LLM model used
- `temperature`: LLM temperature setting
- `num_clusters`: Number of themes discovered
- `total_reviews`: Total reviews processed
- `token_cost`: Estimated token cost
- `notes`: Additional notes

## 🚀 Usage

### Quick Navigation
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Running the Pipeline](#running-the-pipeline)
- [CLI Options](#cli-options)
- [Example Output](#example-output)

### Installation

```bash
pip install -r requirements.txt
```

### ✅ Recent Fixes & Improvements

#### OpenAI API Compatibility (Fixed)
- **Issue**: Pipeline failed with OpenAI API v1.0+ incompatibility
- **Fix**: Updated LLM summarizer to use new OpenAI client format
- **Impact**: LLM theme summaries now work with current OpenAI library versions

#### Sentiment Analysis Calibration (Improved)  
- **Issue**: Sentiment predictions inconsistent with review ratings (50.6% negative vs 33.8% expected)
- **Fix**: Added rating-aware calibration that adjusts low-confidence predictions
- **Impact**: More accurate sentiment analysis aligned with actual review ratings

#### Cache Management (Added)
- **Issue**: No support for incremental updates or pipeline iteration
- **Fix**: Added comprehensive caching system for embeddings, models, and results  
- **Impact**: Enables future incremental updates and faster pipeline iterations

### Environment Setup

Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### Running the Pipeline

```bash
# Basic run with default settings
python pipeline.py run

# Custom settings
python pipeline.py run --model gpt-4 --temperature 0.3 --max-clusters 8

# View results
python pipeline.py show-results

# Export to CSV
python pipeline.py export --format csv
```

### CLI Options

- `--model`: LLM model (default: gpt-4)
- `--temperature`: LLM temperature (default: 0.3)
- `--max-clusters`: Maximum number of themes (default: 10)
- `--min-clusters`: Minimum number of themes (default: 5)
- `--force-recompute`: Recompute embeddings even if cached

## 📈 Example Output

```
1. EchoPad Battery Life Concerns
   Summary: Customers frequently report dissatisfaction with the EchoPad's battery life, describing it as 'unacceptable' and 'mediocre'. Issues include rapid drain during video calls, gaming, and standby. However, some positive feedback highlights long-lasting battery during varied usage.
   Volume: 21.9%
   Sentiment: 51% positive, 47% negative, 2% neutral
   Representative Quotes:
     1. "Testing sidebyside, the battery life comes off as beyond expectations; it barely moved from 100 after bingereading for 9 hours straight."
     2. "Testing sidebyside, the battery life comes off as beyond expectations; it handles a 6hour workday plus Netflix and finishes with 18 remaining."
     3. "In daily use, the battery life is excellent; it still at 5 after a full weekend camping trip. PrimeApple nailed it this time!"

2. Firmware Updates Impacting Experience
   Summary: Customers report firmware updates on EchoPad products introduce bugs, worsen battery life, and unexpectedly lock dev mode. Despite some positive feedback, improvements are needed to enhance user experience and maintain customer satisfaction.
   Volume: 19.4%
   Sentiment: 45% positive, 54% negative, 1% neutral
```

## 🔧 Configuration

### LLM Settings
- **Temperature**: Controls creativity vs consistency (0.1-0.7 recommended)
- **Max Tokens**: Limits response length (150-200 for summaries)
- **Top P**: Controls response diversity (0.9 recommended)

### Clustering Settings
- **K Range**: 5-10 clusters for manageable insights
- **Silhouette Threshold**: 0.3 minimum for cluster quality
- **Random Seed**: 42 for reproducibility

## 💡 Best Practices

1. **Reproducibility**: Always use fixed random seeds
2. **Cost Management**: Cache embeddings and use appropriate model sizes
3. **Quality Control**: Validate LLM outputs manually for first few runs
4. **Iteration**: Start with default settings, then tune based on results
5. **Documentation**: Document any manual overrides or business rules

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Errors**: Check API key and rate limits
2. **Memory Issues**: Reduce batch size or use smaller embedding model
3. **Poor Clustering**: Adjust K range or try different algorithms
4. **Inconsistent LLM Outputs**: Lower temperature or improve prompts

### Debug Mode

```bash
python pipeline.py run --debug
```

This enables verbose logging and intermediate result saving.

## 📝 License

MIT License - see LICENSE file for details.

---

**[⬆ Back to Top](#primeapple-review-insight-pipeline)** 