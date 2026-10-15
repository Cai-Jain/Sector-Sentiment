"""Roll article-level sentiment up to a sector score and persist history."""

from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from . import config


def articles_frame(scored: dict[str, list[tuple]]) -> pd.DataFrame:
    """scored = {sector: [(Article, Sentiment), ...]} -> one row per article."""
    now = datetime.now(timezone.utc)
    rows = []
    for sector, pairs in scored.items():
        for art, sent in pairs:
            age_h = (now - art.published).total_seconds() / 3600
            rows.append({
                "sector": sector,
                "published": art.published,
                "age_hours": round(age_h, 1),
                "source": art.source,
                "title": art.title,
                "url": art.url,
                "label": sent.label,
                "positive": sent.positive,
                "negative": sent.negative,
                "neutral": sent.neutral,
                "score": sent.score,
                # Recency decay; neutral articles carry little weight automatically
                # because their score is near zero.
                "weight": 0.5 ** (age_h / config.HALF_LIFE_HOURS),
            })
    return pd.DataFrame(rows)


def sector_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector, spec in config.SECTORS.items():
        g = df[df["sector"] == sector] if not df.empty else df
        n = len(g)
        if n:
            w = g["weight"]
            score = float((g["score"] * w).sum() / w.sum())
            pos, neg = int((g["label"] == "positive").sum()), int((g["label"] == "negative").sum())
        else:
            score, pos, neg = 0.0, 0, 0
        # Shrink toward zero when there are few articles so thin news days
        # don't produce extreme readings.
        shrink = n / (n + config.MIN_ARTICLES)
        rows.append({
            "sector": sector,
            "etf": spec["etf"],
            "articles": n,
            "positive": pos,
            "negative": neg,
            "neutral": n - pos - neg,
            "raw_score": round(score, 3),
            "score": round(score * shrink, 3),
            "low_confidence": n < config.MIN_ARTICLES,
        })
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


# ---------- history ----------

def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    c = sqlite3.connect(config.DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS sector_scores (
        run_at TEXT, sector TEXT, etf TEXT, articles INTEGER, score REAL, raw_score REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS articles (
        run_at TEXT, sector TEXT, published TEXT, source TEXT, title TEXT, url TEXT,
        label TEXT, score REAL, UNIQUE(sector, title))""")
    return c


def save(run_at: str, scores: pd.DataFrame, articles: pd.DataFrame) -> None:
    with _conn() as c:
        c.executemany("INSERT INTO sector_scores VALUES (?,?,?,?,?,?)",
                      scores[["sector", "etf", "articles", "score", "raw_score"]]
                      .assign(run_at=run_at)[["run_at", "sector", "etf", "articles", "score", "raw_score"]]
                      .itertuples(index=False, name=None))
        if not articles.empty:
            c.executemany("INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?,?,?,?)",
                          [(run_at, r.sector, r.published.isoformat(), r.source, r.title, r.url, r.label, r.score)
                           for r in articles.itertuples()])


def add_trend(scores: pd.DataFrame) -> pd.DataFrame:
    """Add change vs. previous run and a z-score vs. the sector's own history.

    Sector news has persistent biases (tech headlines skew positive, energy
    skews negative), so the z-score -- "unusually negative *for this sector*"
    -- is often a better hedge trigger than the raw level. It needs ~10 runs
    of history before it's meaningful.
    """
    with _conn() as c:
        hist = pd.read_sql("SELECT run_at, sector, score FROM sector_scores", c)
    scores = scores.copy()
    scores["delta"], scores["zscore"] = math.nan, math.nan
    if hist.empty:
        return scores
    for i, row in scores.iterrows():
        h = hist[hist["sector"] == row["sector"]].sort_values("run_at")["score"]
        if len(h):
            scores.at[i, "delta"] = round(row["score"] - h.iloc[-1], 3)
        if len(h) >= 10 and h.std() > 0:
            scores.at[i, "zscore"] = round((row["score"] - h.mean()) / h.std(), 2)
    return scores
