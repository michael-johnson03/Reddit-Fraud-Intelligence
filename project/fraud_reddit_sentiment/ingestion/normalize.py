# fraud_reddit_sentiment/ingestion/normalize.py
"""Normalize + curate Reddit RSS raw pulls.

- Reads raw posts for run_date
- Standardizes schema
- Applies clean_text to rss_summary_text so all downstream files have clean text
- Dedupes across runs using persisted state file
- Writes curated flat files to data/curated/...
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Set

import pandas as pd

from fraud_reddit_sentiment.config_utils import load_pipeline_config
from fraud_reddit_sentiment.io_utils import (
    resolve_data_paths, curated_posts_path, read_parquet,
    write_csv, write_parquet, load_json, save_json,
)
from fraud_reddit_sentiment.preprocessing.clean_text import clean


def run(run_date: str) -> pd.DataFrame:
    cfg = load_pipeline_config("config.yaml")
    paths = resolve_data_paths(cfg.data_root, run_date, dataset="reddit")

    raw_path = paths.raw_dir / "reddit_posts_raw.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    df = read_parquet(raw_path)

    # Standard schema
    df["text"] = df.get("text", "").fillna("")
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["post_id"] = df["post_id"].fillna(df["permalink"]).fillna(df["url"])
    df["source"] = "reddit_rss"
    df["pulled_at_utc"] = df.get("pulled_at_utc", datetime.now(timezone.utc).isoformat())
    df["run_date"] = run_date

    # Clean rss_summary_text here so all downstream files have clean text
    df["rss_summary_text"] = df["rss_summary_text"].fillna("").astype(str).apply(
        lambda s: clean(s, max_words=150)
    )

    # Load seen IDs for cross-run dedupe
    state = load_json(cfg.state_path)
    seen: Set[str] = set(state.get("seen_post_ids", []))

    before = len(df)
    df = df[df["post_id"].notna()].copy()
    df = df[~df["post_id"].isin(seen)].copy()
    after = len(df)
    print(f"Cross-run new posts: {before} -> {after}")

    # Update state
    newly_seen = df["post_id"].astype(str).tolist()
    state["seen_post_ids"] = list(seen.union(newly_seen))
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(cfg.state_path, state)

    # Write curated
    out = curated_posts_path(paths, filename="reddit_posts_curated")
    write_csv(df, out["csv"])
    write_parquet(df, out["parquet"])
    print("Wrote:", out["csv"])
    print("Wrote:", out["parquet"])
    return df


if __name__ == "__main__":
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run(today)
