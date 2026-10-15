"""Scrape sector news, classify sentiment, and print hedge suggestions.

Usage:
    python run.py                                  # FinBERT, no portfolio
    python run.py --portfolio data/portfolio.csv   # add hedge sizing
    python run.py --model vader                    # fast, lightweight model
    python run.py --show-articles Energy           # inspect headlines for a sector
"""

import argparse
import os
from datetime import datetime, timezone

import pandas as pd
from tabulate import tabulate

from sector_sentiment import aggregate, config, hedge, scraper, sentiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["finbert", "vader"], default="finbert")
    ap.add_argument("--portfolio", help="CSV with ticker,sector,market_value")
    ap.add_argument("--show-articles", metavar="SECTOR", help="print scored headlines for one sector")
    ap.add_argument("--no-save", action="store_true", help="don't write results to history DB")
    args = ap.parse_args()

    run_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{run_at}] Scraping news for {len(config.SECTORS)} sectors...")
    articles = scraper.scrape_all()
    total = sum(len(a) for a in articles.values())
    print(f"  {total} unique articles from the last {config.LOOKBACK_HOURS}h")

    print(f"Classifying with {args.model}...")
    clf = sentiment.get_classifier(args.model)
    flat = [a for arts in articles.values() for a in arts]
    sents = clf.classify([a.text for a in flat]) if flat else []
    scored = {s: [] for s in config.SECTORS}
    for a, s in zip(flat, sents):
        scored[a.sector].append((a, s))

    df = aggregate.articles_frame(scored)
    scores = aggregate.add_trend(aggregate.sector_scores(df))
    portfolio = hedge.load_portfolio(args.portfolio) if args.portfolio else None
    plan = hedge.build_hedges(scores, portfolio)

    report = scores.merge(plan[["sector", "signal", "hedge_ratio"]], on="sector")
    cols = ["sector", "etf", "articles", "positive", "negative", "neutral", "score", "delta", "zscore", "signal", "hedge_ratio"]
    print("\nSECTOR SENTIMENT (score in [-1, 1], recency-weighted; * = low article count)")
    report["sector"] = report.apply(lambda r: r["sector"] + (" *" if r["low_confidence"] else ""), axis=1)
    print(tabulate(report[cols], headers="keys", tablefmt="simple", showindex=False, floatfmt=".3f", missingval="-"))

    if portfolio is not None:
        print("\nHEDGE SUGGESTIONS")
        hcols = ["sector", "weight_pct", "exposure", "score", "hedge_ratio", "action"]
        print(tabulate(plan[plan["exposure"] > 0][hcols], headers="keys", tablefmt="simple",
                       showindex=False, floatfmt=",.2f"))
        print(f"\nTotal suggested hedge notional: ${plan['hedge_notional'].sum():,.0f}")

    if args.show_articles:
        sub = df[df["sector"].str.lower() == args.show_articles.lower()].sort_values("score")
        print(f"\nARTICLES: {args.show_articles} ({len(sub)})")
        print(tabulate(sub[["age_hours", "label", "score", "source", "title"]].assign(title=sub["title"].str[:90]),
                       headers="keys", tablefmt="simple", showindex=False, floatfmt=".2f"))

    os.makedirs("data/runs", exist_ok=True)
    stamp = run_at.replace(":", "")
    report.to_csv(f"data/runs/{stamp}_sectors.csv", index=False)
    if not df.empty:
        df.to_csv(f"data/runs/{stamp}_articles.csv", index=False)
    if not args.no_save:
        aggregate.save(run_at, scores, df)
    print(f"\nSaved CSVs to data/runs/ and history to {config.DB_PATH}")


if __name__ == "__main__":
    main()
