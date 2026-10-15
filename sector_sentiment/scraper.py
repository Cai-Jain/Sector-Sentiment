"""Collect recent sector news headlines from free RSS feeds.

Sources:
  * Google News RSS search (one feed per sector query)
  * Yahoo Finance headline RSS (sector ETF + bellwether tickers)

Headlines + RSS summaries are used rather than full article bodies: most
financial sites paywall or block scraping, and FinBERT was trained on
sentence-level financial text, so headlines are a strong signal on their own.
"""

from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser
import requests

from . import config

GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={q}+when:2d&hl=en-US&gl=US&ceid=US:en"
YAHOO_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={tickers}&region=US&lang=en-US"


@dataclass
class Article:
    sector: str
    title: str
    summary: str
    url: str
    source: str
    published: datetime

    @property
    def text(self) -> str:
        # Summary is often just the headline repeated (Google News); avoid doubling.
        if self.summary and self.summary.lower() not in self.title.lower():
            return f"{self.title}. {self.summary}"[:512]
        return self.title


def _clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", html.unescape(s or ""))
    return re.sub(r"\s+", " ", s).strip()


def _fetch_feed(url: str) -> feedparser.FeedParserDict:
    resp = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=config.REQUEST_TIMEOUT)
    resp.raise_for_status()
    return feedparser.parse(resp.content)


def _parse(sector: str, url: str, source_label: str, cutoff: datetime) -> list[Article]:
    try:
        feed = _fetch_feed(url)
    except Exception as e:
        print(f"  ! {sector}: failed to fetch {source_label} ({e})")
        return []

    out = []
    for entry in feed.entries:
        ts = entry.get("published_parsed") or entry.get("updated_parsed")
        if not ts:
            continue
        published = datetime(*ts[:6], tzinfo=timezone.utc)
        if published < cutoff:
            continue
        title = _clean(entry.get("title", ""))
        # Google News appends " - Publisher" to titles; split it off.
        publisher = source_label
        if " - " in title and source_label == "google":
            title, publisher = title.rsplit(" - ", 1)
        out.append(Article(
            sector=sector,
            title=title,
            summary=_clean(entry.get("summary", "")),
            url=entry.get("link", ""),
            source=publisher,
            published=published,
        ))
    return out


def _sector_urls(sector: str, spec: dict) -> list[tuple[str, str]]:
    urls = [(GOOGLE_NEWS_URL.format(q=quote_plus(q)), "google") for q in spec["queries"]]
    tickers = ",".join([spec["etf"], *spec["tickers"]])
    urls.append((YAHOO_URL.format(tickers=tickers), "yahoo"))
    return urls


def _dedupe(articles: list[Article]) -> list[Article]:
    seen, out = set(), []
    for a in sorted(articles, key=lambda a: a.published, reverse=True):
        key = re.sub(r"[^a-z0-9]", "", a.title.lower())[:80]
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out


def scrape_all(sectors: dict | None = None) -> dict[str, list[Article]]:
    """Return {sector: [Article, ...]} for every sector, fetched in parallel."""
    sectors = sectors or config.SECTORS
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.LOOKBACK_HOURS)

    jobs = [(name, url, label) for name, spec in sectors.items() for url, label in _sector_urls(name, spec)]
    results: dict[str, list[Article]] = {name: [] for name in sectors}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (name, _, _), arts in zip(jobs, pool.map(lambda j: _parse(j[0], j[1], j[2], cutoff), jobs)):
            results[name].extend(arts)

    return {name: _dedupe(arts) for name, arts in results.items()}
