from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_reddit_sentiment.config_utils import load_pipeline_config
from fraud_reddit_sentiment.io_utils import resolve_data_paths
from fraud_reddit_sentiment.preprocessing.clean_text import clean


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _build_retrieval_text(row: pd.Series) -> str:
    title = clean(str(row.get("title", "") or ""))
    rss_summary = clean(str(row.get("rss_summary_text", "") or ""))
    theme = clean(str(row.get("theme", "") or ""))
    signals = clean(str(row.get("theme_matches", "") or ""))

    parts = [
        f"title: {title}",
        f"rss_summary: {rss_summary}",
        f"theme: {theme}",
        f"signals: {signals}",
    ]
    return "\n".join(parts).strip()


def _load_embedder(model_name: str = EMBED_MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return vectors / norms


def _collect_all_themed_parquets(data_root: str, dataset: str = "reddit") -> pd.DataFrame:
    """Aggregate themed parquets from all available run_date partitions."""
    root = Path(data_root).expanduser().resolve()
    scored_root = root / "scored" / dataset
    pattern = "run_date=*/reddit_posts_themed.parquet"

    files = sorted(scored_root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No themed parquets found under {scored_root}")

    print(f"Found {len(files)} run_date partition(s):")
    frames = []
    for f in files:
        run_date = f.parent.name.replace("run_date=", "")
        df = pd.read_parquet(f).copy()
        df["run_date"] = run_date
        print(f"  {run_date}: {len(df)} posts")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate across run dates by post_id
    before = len(combined)
    combined = combined.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)
    after = len(combined)
    print(f"\nCombined: {before} rows → {after} unique posts after cross-date dedup")

    return combined


def build_embeddings(
    data_root: str,
    output_parquet_path: str | Path,
    output_index_path: str | Path,
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = 64,
    dataset: str = "reddit",
) -> pd.DataFrame:
    import faiss

    output_parquet_path = Path(output_parquet_path)
    output_index_path = Path(output_index_path)

    # Aggregate all run dates
    df = _collect_all_themed_parquets(data_root, dataset=dataset)

    # Ensure required fields exist
    for col, default in {
        "post_id": "",
        "run_date": "",
        "theme": "Other_Unclear",
        "theme_score": 0.0,
        "theme_matches": "",
        "title": "",
        "rss_summary_text": "",
    }.items():
        if col not in df.columns:
            df[col] = default

    df["retrieval_text"] = df.apply(_build_retrieval_text, axis=1)

    print(f"\nEmbedding {len(df)} posts...")
    model = _load_embedder(model_name)
    texts = df["retrieval_text"].fillna("").astype(str).tolist()

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    vectors = _normalize_vectors(vectors).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    output_index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_index_path))

    out_df = df[[
        "post_id",
        "run_date",
        "theme",
        "theme_score",
        "theme_matches",
        "title",
        "rss_summary_text",
        "retrieval_text",
    ]].copy()

    out_df["embedding_id"] = np.arange(len(out_df), dtype=np.int64)
    out_df["embedding_model"] = model_name
    out_df["embedding_dim"] = dim
    out_df["embedded_at_utc"] = datetime.now(timezone.utc).isoformat()

    out_df.to_parquet(output_parquet_path, index=False)

    print(f"Wrote embeddings parquet: {output_parquet_path}")
    print(f"Wrote FAISS index:        {output_index_path}")
    print(f"Total posts embedded:     {len(out_df)}")
    print(f"Embedding dim:            {dim}")

    theme_counts = out_df["theme"].value_counts()
    print(f"\nPosts per theme:")
    print(theme_counts.to_string())

    return out_df


def run(run_date: str, config_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_pipeline_config(config_path)
    paths = resolve_data_paths(cfg.data_root, run_date, dataset="reddit")

    # Write the aggregated index to the current run_date scored dir
    output_parquet_path = paths.scored_dir / "reddit_post_embeddings.parquet"
    output_index_path = paths.scored_dir / "reddit_post_embeddings.faiss"

    return build_embeddings(
        data_root=cfg.data_root,
        output_parquet_path=output_parquet_path,
        output_index_path=output_index_path,
        model_name=EMBED_MODEL_NAME,
        dataset="reddit",
    )


if __name__ == "__main__":
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run(today)
