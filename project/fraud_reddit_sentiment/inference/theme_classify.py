# fraud_reddit_sentiment/inference/theme_classify.py
"""Two-tier theme classification for Reddit RSS fraud monitoring.

Tier 1: Exact phrase matching (high confidence, higher weight per match)
Tier 2: Theme word bag scoring (broader coverage, lower weight per word hit)

Final theme = highest combined score across both tiers.
Prefers specific themes over General_Scam on ties.

Inputs:
- data/curated/reddit/run_date=YYYY-MM-DD/reddit_posts_curated.parquet

Outputs:
- data/scored/reddit/run_date=YYYY-MM-DD/reddit_posts_themed.(csv|parquet)
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Tuple

import pandas as pd

from fraud_reddit_sentiment.config_utils import load_pipeline_config
from fraud_reddit_sentiment.io_utils import (
    resolve_data_paths,
    read_parquet,
    write_csv,
    write_parquet,
)
from fraud_reddit_sentiment.inference.fraud_taxonomy import THEME_KEYWORDS, THEME_WORD_BAGS
from fraud_reddit_sentiment.preprocessing.clean_text import clean

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9\s]+")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _word_boundary_pattern(phrase: str) -> re.Pattern:
    """Compile word-boundary regex — prevents substring false positives."""
    return re.compile(r"\b" + re.escape(phrase.lower()) + r"\b")


# Pre-compile all tier 1 phrase patterns at import time
_COMPILED_KEYWORDS = [
    (_word_boundary_pattern(kw), kw, theme, weight)
    for kw, theme, weight in THEME_KEYWORDS
]

# Pre-compile tier 2 word patterns at import time
_COMPILED_WORD_BAGS = {
    theme: (
        [_word_boundary_pattern(w) for w in words],
        words,
        weight,
    )
    for theme, (words, weight) in THEME_WORD_BAGS.items()
}


def score_themes(text: str) -> Tuple[str, float, List[str]]:
    """Score text against both tiers and return best theme assignment.

    Returns:
        (best_theme, best_score, matched_phrases_for_best_theme)
    """
    t = normalize_text(text)

    theme_scores: dict = {}
    theme_phrase_matches: dict = {}

    # --- Tier 1: exact phrase matching ---
    for pattern, kw, theme, weight in _COMPILED_KEYWORDS:
        if pattern.search(t):
            theme_scores[theme] = theme_scores.get(theme, 0.0) + weight
            theme_phrase_matches.setdefault(theme, [])
            if kw not in theme_phrase_matches[theme]:
                theme_phrase_matches[theme].append(kw)

    # --- Tier 2: word bag scoring ---
    for theme, (patterns, words, weight_per_hit) in _COMPILED_WORD_BAGS.items():
        hits = sum(1 for p in patterns if p.search(t))
        if hits > 0:
            # Only count tier 2 if at least 2 word hits to avoid noise
            if hits >= 2:
                theme_scores[theme] = theme_scores.get(theme, 0.0) + (hits * weight_per_hit)

    if not theme_scores:
        return "Other_Unclear", 0.0, []

    # Pick highest scoring theme
    # On ties: prefer specific themes over General_Scam
    best_theme = max(
        theme_scores,
        key=lambda th: (
            theme_scores[th],
            0 if th == "General_Scam" else 1,
        ),
    )

    matched = theme_phrase_matches.get(best_theme, [])
    return best_theme, round(theme_scores[best_theme], 3), matched


def build_theme_input(df: pd.DataFrame) -> pd.Series:
    """Combine cleaned title + rss_summary_text as classification input."""
    title = df.get("title", "").fillna("").astype(str).apply(lambda s: clean(s, max_words=30))
    summary = df.get("rss_summary_text", "").fillna("").astype(str)
    return (title + " " + summary).str.strip()


def run(run_date: str) -> pd.DataFrame:
    cfg = load_pipeline_config("config.yaml")
    paths = resolve_data_paths(cfg.data_root, run_date, dataset="reddit")

    curated_path = paths.curated_dir / "reddit_posts_curated.parquet"
    if not curated_path.exists():
        raise FileNotFoundError(f"Missing curated file: {curated_path}")

    df = read_parquet(curated_path).copy()
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["rss_summary_text"] = df.get("rss_summary_text", "").fillna("").astype(str)

    df["theme_input_text"] = build_theme_input(df)

    results = df["theme_input_text"].apply(score_themes)
    df["theme"] = results.apply(lambda x: x[0])
    df["theme_score"] = results.apply(lambda x: x[1])
    df["theme_matches"] = results.apply(lambda x: ", ".join(x[2]) if x[2] else "")
    df["themed_at_utc"] = datetime.now(timezone.utc).isoformat()

    out_csv = paths.scored_dir / "reddit_posts_themed.csv"
    out_parquet = paths.scored_dir / "reddit_posts_themed.parquet"
    write_csv(df, out_csv)
    write_parquet(df, out_parquet)

    print("Wrote:", out_csv)
    print("Wrote:", out_parquet)

    print("\nTheme counts:")
    print(df["theme"].value_counts())

    print("\nOther_Unclear %: {:.1f}%".format(
        100 * (df["theme"] == "Other_Unclear").sum() / len(df)
    ))

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Theme classification for Reddit posts.")
    parser.add_argument("--run-date", default=None, help="Run date YYYY-MM-DD (default: today UTC)")
    args = parser.parse_args()
    run_date = args.run_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run(run_date)
