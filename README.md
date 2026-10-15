# Sector Sentiment

Scrapes recent news for the 11 GICS sectors, classifies each headline with FinBERT,
and rolls the results up into a recency-weighted sentiment score per sector (-1 to 1).
Optionally sizes sector-ETF hedges against your portfolio.

## Setup
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

## Run
    .venv/bin/python run.py                                       # sector scores
    .venv/bin/python run.py --portfolio data/portfolio.csv        # + hedge sizing
    .venv/bin/python run.py --show-articles "Energy"              # inspect headlines
    .venv/bin/python run.py --model vader                         # fast fallback model

Copy `data/portfolio.example.csv` to `data/portfolio.csv` and fill in your holdings.

## Layout
- `sector_sentiment/config.py`: sectors, ETFs, search queries, thresholds
- `sector_sentiment/scraper.py`: Google News + Yahoo Finance RSS
- `sector_sentiment/sentiment.py`: FinBERT / VADER classifiers
- `sector_sentiment/aggregate.py`: sector scoring, SQLite history, z-scores
- `sector_sentiment/hedge.py`: signal and hedge-ratio logic

Every run adds to `data/sentiment.db`. Once a sector has about 10 runs of history, the `zscore`
column shows how unusual today's score is *for that sector*.

Research tool only. This is not investment advice.
