# fraud_reddit_sentiment/ingestion/rss_pull.py
"""Reddit RSS ingestion (subreddit feeds only) with summary capture.

Pulls:
  https://www.reddit.com/r/<subreddit>/<feed>/.rss
Where <feed> is typically 'new' (recommended).

Writes raw flat files (CSV + Parquet) to:
  data/raw/reddit/run_date=YYYY-MM-DD/reddit_posts_raw.*

Fields:
- title: post title
- post_id: permalink (stable surrogate key)
- published_at: post publish timestamp
- pulled_at_utc: ingestion timestamp
- rss_summary_html: raw HTML summary from RSS entry
- rss_summary_text: cleaned plain text from RSS summary
"""

from __future__ import annotations

import re
import html as _html
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List

import requests
import pandas as pd

from fraud_reddit_sentiment.config_utils import load_pipeline_config
from fraud_reddit_sentiment.io_utils import resolve_data_paths, raw_posts_path, write_csv, write_parquet

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
USER_AGENT = "Mozilla/5.0 (compatible; fraud-monitor/1.0)"


def rss_url(subreddit: str, feed_type: str = "new") -> str:
    return f"https://www.reddit.com/r/{subreddit}/{feed_type}/.rss"


def _html_to_text(html_str: str) -> str:
    if not html_str:
        return ""
    text = re.sub(r"<[^>]+>", " ", html_str)
    text = _html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _get_text(element, tag: str, ns: dict) -> str:
    node = element.find(tag, ns)
    if node is not None and node.text:
        return node.text.strip()
    return ""


def pull_feed(subreddit: str, feed_type: str, limit: int) -> List[Dict]:
    url = rss_url(subreddit, feed_type)
    headers = {"User-Agent": USER_AGENT}
    pulled_at = datetime.now(timezone.utc).isoformat()

    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch RSS for r/{subreddit}: {e}")

    root = ET.fromstring(r.text)
    entries = root.findall("atom:entry", ATOM_NS)[:limit]

    rows: List[Dict] = []
    for e in entries:
        link_node = e.find("atom:link", ATOM_NS)
        link = link_node.attrib.get("href", "") if link_node is not None else ""

        title = _get_text(e, "atom:title", ATOM_NS)
        published = _get_text(e, "atom:published", ATOM_NS) or _get_text(e, "atom:updated", ATOM_NS)

        content_node = e.find("atom:content", ATOM_NS)
        summary_node = e.find("atom:summary", ATOM_NS)

        if content_node is not None and content_node.text:
            summary_html = content_node.text.strip()
        elif summary_node is not None and summary_node.text:
            summary_html = summary_node.text.strip()
        else:
            summary_html = ""

        summary_text = _html_to_text(summary_html)

        rows.append({
            "source": "reddit_rss",
            "subreddit": subreddit,
            "post_id": link,
            "title": str(title),
            "text": "",
            "rss_summary_html": summary_html,
            "rss_summary_text": summary_text,
            "permalink": link,
            "url": link,
            "published_at": published,
            "pulled_at_utc": pulled_at,
        })

    return rows


def run(run_date: str) -> pd.DataFrame:
    cfg = load_pipeline_config("config.yaml")
    paths = resolve_data_paths(cfg.data_root, run_date, dataset="reddit")

    all_rows: List[Dict] = []
    per_sr_limit = min(int(cfg.rss.per_subreddit_limit), 100)

    for sr in cfg.subreddits:
        try:
            rows = pull_feed(sr, cfg.rss.feed_type, per_sr_limit)
            for r in rows:
                r["run_date"] = run_date
            all_rows.extend(rows)
            print(f"[rss] r/{sr}: {len(rows)}")
        except Exception as e:
            print(f"[rss] r/{sr}: ERROR {type(e).__name__}: {e}")
        time.sleep(cfg.rss.throttle_seconds)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No RSS rows pulled.")
        return df

    before = len(df)
    df = df.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)
    print(f"Deduped within-run: {before} -> {len(df)}")

    out = raw_posts_path(paths, filename="reddit_posts_raw")
    write_csv(df, out["csv"])
    write_parquet(df, out["parquet"])
    print("Wrote:", out["csv"])
    print("Wrote:", out["parquet"])
    return df


if __name__ == "__main__":
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run(today)
