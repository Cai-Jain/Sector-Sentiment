"""Sector definitions and tunable parameters.

GICS has 11 sectors. Each maps to its SPDR Select Sector ETF (used both as a
news source and as the natural hedge instrument), a set of search queries,
and a few bellwether tickers whose news tends to move the whole sector.
"""

SECTORS = {
    "Information Technology": {
        "etf": "XLK",
        "queries": ["technology stocks", "semiconductor stocks", "software stocks"],
        "tickers": ["AAPL", "MSFT", "NVDA", "AVGO"],
    },
    "Health Care": {
        "etf": "XLV",
        "queries": ["healthcare stocks", "pharmaceutical stocks", "biotech stocks"],
        "tickers": ["LLY", "UNH", "JNJ", "MRK"],
    },
    "Financials": {
        "etf": "XLF",
        "queries": ["bank stocks", "financial sector stocks", "insurance stocks"],
        "tickers": ["JPM", "BRK-B", "V", "GS"],
    },
    "Consumer Discretionary": {
        "etf": "XLY",
        "queries": ["retail stocks", "consumer discretionary stocks", "automaker stocks"],
        "tickers": ["AMZN", "TSLA", "HD", "MCD"],
    },
    "Communication Services": {
        "etf": "XLC",
        "queries": ["communication services stocks", "media stocks", "telecom stocks"],
        "tickers": ["GOOGL", "META", "NFLX", "DIS"],
    },
    "Industrials": {
        "etf": "XLI",
        "queries": ["industrial stocks", "aerospace defense stocks", "airline stocks"],
        "tickers": ["GE", "CAT", "RTX", "UNP"],
    },
    "Consumer Staples": {
        "etf": "XLP",
        "queries": ["consumer staples stocks", "food beverage stocks", "grocery retail stocks"],
        "tickers": ["PG", "COST", "WMT", "KO"],
    },
    "Energy": {
        "etf": "XLE",
        "queries": ["energy stocks", "oil prices", "natural gas prices"],
        "tickers": ["XOM", "CVX", "COP", "SLB"],
    },
    "Utilities": {
        "etf": "XLU",
        "queries": ["utility stocks", "electric utilities", "power grid demand"],
        "tickers": ["NEE", "SO", "DUK", "CEG"],
    },
    "Real Estate": {
        "etf": "XLRE",
        "queries": ["REIT stocks", "commercial real estate", "real estate stocks"],
        "tickers": ["PLD", "AMT", "EQIX", "SPG"],
    },
    "Materials": {
        "etf": "XLB",
        "queries": ["materials stocks", "chemical stocks", "mining stocks"],
        "tickers": ["LIN", "SHW", "FCX", "NEM"],
    },
}

# --- Scraping ---
LOOKBACK_HOURS = 48          # ignore articles older than this
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (sector-sentiment research script)"

# --- Scoring ---
HALF_LIFE_HOURS = 12         # an article's weight halves every N hours
MIN_ARTICLES = 5             # below this, a sector's signal is flagged low-confidence

# --- Signals (score is in [-1, 1]) ---
BEARISH_THRESHOLD = -0.15
BULLISH_THRESHOLD = 0.15
MAX_HEDGE_RATIO = 0.5        # never suggest hedging more than 50% of a sector exposure

DB_PATH = "data/sentiment.db"
