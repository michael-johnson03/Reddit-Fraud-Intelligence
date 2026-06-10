# fraud_reddit_sentiment/config_utils.py
"""Config utilities (RSS-only).

Loads config.yaml and validates required fields.
No OAuth credentials or sentiment/weekly switches are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class RSSConfig:
    feed_type: str           # "new" (recommended)
    per_subreddit_limit: int  # cap number of items pulled per feed
    throttle_seconds: float  # polite delay between requests


@dataclass(frozen=True)
class PipelineConfig:
    data_root: str
    state_path: str
    subreddits: List[str]
    rss: RSSConfig


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse to a dict at the top level.")
    return cfg


def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]


def load_pipeline_config(config_path: str = "config.yaml") -> PipelineConfig:
    raw = load_yaml(config_path)

    data_root = str(_require(raw, "data_root")).strip()
    state_path = str(raw.get("state_path", "../state/reddit_state.json")).strip()

    subreddits = _require(raw, "subreddits")
    if not isinstance(subreddits, list) or not subreddits:
        raise ValueError("subreddits must be a non-empty list")
    subreddits = [str(s).strip() for s in subreddits if str(s).strip()]

    rss_raw = raw.get("rss", {}) or {}
    feed_type = str(rss_raw.get("feed_type", "new")).strip().lower()
    if feed_type not in {"new", "hot"}:
        raise ValueError("rss.feed_type must be 'new' or 'hot'")

    per_subreddit_limit = int(rss_raw.get("per_subreddit_limit", 50))
    if per_subreddit_limit < 1 or per_subreddit_limit > 100:
        raise ValueError("rss.per_subreddit_limit must be between 1 and 100")

    throttle_seconds = float(rss_raw.get("throttle_seconds", 0.6))
    if throttle_seconds < 0:
        raise ValueError("rss.throttle_seconds must be >= 0")

    return PipelineConfig(
        data_root=data_root,
        state_path=state_path,
        subreddits=subreddits,
        rss=RSSConfig(
            feed_type=feed_type,
            per_subreddit_limit=per_subreddit_limit,
            throttle_seconds=throttle_seconds,
        ),
    )
