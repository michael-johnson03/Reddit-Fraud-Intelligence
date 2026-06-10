# fraud_reddit_sentiment/io_utils.py
"""
I/O utilities for Fraud Reddit Sentiment pipeline.

Responsibilities:
- Create folder structure automatically (data/raw, curated, scored, summaries + run_date partition)
- Provide standard path builders so all pipeline steps write to consistent locations
- Read/write CSV and Parquet in a Snowflake-friendly way (flat files)

Designed to work in:
- Google Colab (Drive-mounted paths)
- Local environments
- Production runs (same folder structure)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd


@dataclass(frozen=True)
class DataPaths:
    """Resolved output paths for a given run_date."""
    root: Path
    run_date: str
    raw_dir: Path
    curated_dir: Path
    scored_dir: Path
    summaries_dir: Path


def ensure_dir(path: Path) -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    """Create parent directory for a file path if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def resolve_data_paths(data_root: str, run_date: str, dataset: str = "reddit") -> DataPaths:
    """
    Build and create the standard folder structure for a specific run_date.

    Layout under data_root:
      raw/<dataset>/run_date=YYYY-MM-DD/
      curated/<dataset>/run_date=YYYY-MM-DD/
      scored/<dataset>/run_date=YYYY-MM-DD/
      summaries/<dataset>/run_date=YYYY-MM-DD/
    """
    root = Path(data_root).expanduser().resolve()

    raw_dir = root / "raw" / dataset / f"run_date={run_date}"
    curated_dir = root / "curated" / dataset / f"run_date={run_date}"
    scored_dir = root / "scored" / dataset / f"run_date={run_date}"
    summaries_dir = root / "summaries" / dataset / f"run_date={run_date}"

    for d in (raw_dir, curated_dir, scored_dir, summaries_dir):
        ensure_dir(d)

    return DataPaths(
        root=root,
        run_date=run_date,
        raw_dir=raw_dir,
        curated_dir=curated_dir,
        scored_dir=scored_dir,
        summaries_dir=summaries_dir,
    )


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file. If missing, return empty dict."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    """Save JSON file (pretty printed). Creates parent dir."""
    p = Path(path).expanduser().resolve()
    ensure_parent(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_csv(df: pd.DataFrame, filepath: Path, index: bool = False) -> None:
    """Write dataframe to CSV, ensuring parent folder exists."""
    ensure_parent(filepath)
    df.to_csv(filepath, index=index)


def write_parquet(df: pd.DataFrame, filepath: Path) -> None:
    """Write dataframe to Parquet, ensuring parent folder exists."""
    ensure_parent(filepath)
    df.to_parquet(filepath, index=False)


def read_parquet(filepath: Path) -> pd.DataFrame:
    """Read Parquet file."""
    return pd.read_parquet(filepath)


def read_csv(filepath: Path) -> pd.DataFrame:
    """Read CSV file."""
    return pd.read_csv(filepath)


def raw_posts_path(paths: DataPaths, filename: str = "reddit_posts_raw") -> Dict[str, Path]:
    """Standard raw post output paths (CSV + Parquet)."""
    return {
        "csv": paths.raw_dir / f"{filename}.csv",
        "parquet": paths.raw_dir / f"{filename}.parquet",
    }


def curated_posts_path(paths: DataPaths, filename: str = "reddit_posts_curated") -> Dict[str, Path]:
    """Standard curated post output paths (CSV + Parquet)."""
    return {
        "csv": paths.curated_dir / f"{filename}.csv",
        "parquet": paths.curated_dir / f"{filename}.parquet",
    }


def scored_posts_path(paths: DataPaths, filename: str = "reddit_posts_scored") -> Dict[str, Path]:
    """Standard scored post output paths (CSV + Parquet)."""
    return {
        "csv": paths.scored_dir / f"{filename}.csv",
        "parquet": paths.scored_dir / f"{filename}.parquet",
    }
