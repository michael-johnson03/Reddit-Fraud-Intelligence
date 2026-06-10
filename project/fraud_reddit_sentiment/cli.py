# fraud_reddit_sentiment/cli.py
"""CLI runner for RSS-based Reddit fraud monitoring pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from fraud_reddit_sentiment.ingestion.rss_pull import run as run_rss
from fraud_reddit_sentiment.ingestion.normalize import run as run_normalize
from fraud_reddit_sentiment.inference.theme_classify import run as run_theme_classify
from fraud_reddit_sentiment.retrieval.build_embeddings import run as run_build_embeddings
from fraud_reddit_sentiment.summarizer.theme_summaries import run as run_theme_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSS-based Reddit fraud monitoring pipeline with per-theme RAG + Mistral synthesis"
    )
    parser.add_argument("--date", default=None, help="Run date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation.")
    parser.add_argument("--skip-theme-summaries", action="store_true", help="Skip theme summary generation.")
    parser.add_argument("--top-k", type=int, default=8, help="Top-k chunks to retrieve per theme.")
    parser.add_argument("--min-posts-per-theme", type=int, default=3, help="Minimum posts required to summarize a theme.")
    args = parser.parse_args()

    run_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"=== Running pipeline for {run_date} ===")

    # 1) RSS ingest
    df_raw = run_rss(run_date)
    if df_raw.empty:
        print("No raw data pulled; exiting.")
        return

    # 2) Normalize + dedupe
    df_curated = run_normalize(run_date)
    if df_curated.empty:
        print("No new curated rows after cross-run dedupe; exiting.")
        return

    # 3) Theme classification
    run_theme_classify(run_date)

    # 4) Build embeddings
    if not args.skip_embeddings:
        try:
            run_build_embeddings(run_date)
        except Exception as e:
            print(f"Embedding build skipped: {type(e).__name__}: {e}")

    # 5) Per-theme RAG + Mistral synthesis
    if not args.skip_theme_summaries:
        try:
            run_theme_summaries(
                run_date,
                top_k=args.top_k,
                min_posts_per_theme=args.min_posts_per_theme,
            )
        except Exception as e:
            print(f"Theme summaries skipped: {type(e).__name__}: {e}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
