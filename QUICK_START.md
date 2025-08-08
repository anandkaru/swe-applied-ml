# 🚀 Quick Start Guide

Get the PrimeApple Review Insight Pipeline running in minutes!

## Prerequisites

- Python 3.8+
- OpenAI API key
- 4GB+ RAM (for model loading)

## 1. Setup Environment

```bash
# Clone or download the project
cd swe-applied-ai

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env and add your OpenAI API key
```

## 2. Test the Pipeline

```bash
# Run tests to verify everything works
python test_pipeline.py
```

## 3. Run Demo

```bash
# Run demo with small dataset (100 reviews)
python demo.py
```

## 4. Run Full Pipeline

```bash
# Run on full dataset (10,000 reviews)
python pipeline.py run

# View results
python pipeline.py show-results

# Export to CSV
python pipeline.py export --format csv
```

## 5. Customize Settings

```bash
# Use different LLM model
python pipeline.py run --model gpt-3.5-turbo

# Adjust clustering
python pipeline.py run --min-clusters 3 --max-clusters 8

# Change temperature for more/less creative summaries
python pipeline.py run --temperature 0.5
```

## Expected Output

After running the pipeline, you'll get:

- **5-10 data-backed themes** with titles and summaries
- **Sentiment analysis** for each theme
- **Representative quotes** from customer reviews
- **Volume metrics** showing theme importance
- **SQLite database** with all results

## Troubleshooting

### Common Issues

1. **OpenAI API Error**: Check your API key in `.env`
2. **Memory Issues**: Reduce batch size in `src/config.py`
3. **Model Download**: First run will download models (~500MB)

### Getting Help

- Check logs for detailed error messages
- Run `python test_pipeline.py` to isolate issues
- Review `README.md` for detailed documentation

## Next Steps

- Review `PIPELINE_SUMMARY.md` for technical details
- Customize prompts in `src/llm_summarizer.py`
- Adjust clustering parameters in `src/theme_discovery.py`
- Add your own data preprocessing in `src/data_loader.py`

Happy analyzing! 🎉 