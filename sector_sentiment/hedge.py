"""Turn sector sentiment into hedge suggestions for a portfolio.

Portfolio CSV format (data/portfolio.csv):
    ticker,sector,market_value
    AAPL,Information Technology,25000
    JPM,Financials,12000

The hedge ratio scales linearly from 0 at BEARISH_THRESHOLD to MAX_HEDGE_RATIO
at a score of -1. The suggested instrument is the sector's SPDR ETF (short it,
or buy puts of equivalent delta). These are inputs to your own judgment, not
orders. Sentiment is noisy and should be combined with price/risk signals.
"""

from __future__ import annotations

import pandas as pd

from . import config


def signal(score: float) -> str:
    if score <= config.BEARISH_THRESHOLD:
        return "BEARISH"
    if score >= config.BULLISH_THRESHOLD:
        return "BULLISH"
    return "NEUTRAL"


def hedge_ratio(score: float) -> float:
    if score > config.BEARISH_THRESHOLD:
        return 0.0
    span = -1.0 - config.BEARISH_THRESHOLD
    frac = (score - config.BEARISH_THRESHOLD) / span
    return round(min(1.0, frac) * config.MAX_HEDGE_RATIO, 3)


def load_portfolio(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"ticker", "sector", "market_value"} - set(df.columns)
    if missing:
        raise ValueError(f"portfolio file missing columns: {missing}")
    unknown = set(df["sector"]) - set(config.SECTORS)
    if unknown:
        raise ValueError(f"unknown sectors in portfolio: {unknown}. Use one of: {list(config.SECTORS)}")
    return df


def build_hedges(scores: pd.DataFrame, portfolio: pd.DataFrame | None) -> pd.DataFrame:
    out = scores[["sector", "etf", "score", "low_confidence"]].copy()
    out["signal"] = out["score"].map(signal)
    out["hedge_ratio"] = out["score"].map(hedge_ratio)

    if portfolio is not None:
        exposure = portfolio.groupby("sector")["market_value"].sum()
        total = exposure.sum()
        out["exposure"] = out["sector"].map(exposure).fillna(0.0)
        out["weight_pct"] = (out["exposure"] / total * 100).round(1)
        out["hedge_notional"] = (out["exposure"] * out["hedge_ratio"]).round(0)
        out["action"] = out.apply(_action, axis=1)
    return out


def _action(r) -> str:
    if r["hedge_notional"] > 0:
        note = " (low article count; confirm manually)" if r["low_confidence"] else ""
        return f"Hedge ~${r['hedge_notional']:,.0f}: short {r['etf']} or buy {r['etf']} puts{note}"
    if r["signal"] == "BEARISH":
        return "Bearish, but no exposure"
    if r["signal"] == "BULLISH" and r["exposure"] > 0:
        return "Bullish: consider lifting existing hedges"
    return "-"
