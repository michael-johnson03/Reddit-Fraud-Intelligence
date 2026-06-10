from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fraud_reddit_sentiment.config_utils import load_pipeline_config
from fraud_reddit_sentiment.io_utils import resolve_data_paths


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

THEME_QUERY_MAP = {
    "ATO_Account_Takeover": "Find repeated account takeover patterns, unauthorized access behavior, and likely user impact.",
    "Phishing_Smishing": "Find repeated phishing, smishing, or vishing patterns, contact methods, requested actions, and likely risk.",
    "Identity_Theft": "Find repeated identity theft behavior, fraudulent account creation, and downstream victim impact.",
    "Benefits_Fraud": "Find repeated government benefits fraud, unemployment fraud, tax fraud, and likely victim or agency impact.",
    "Payment_Scams_P2P": "Find repeated peer-to-peer payment scam behavior, fake payments, reversals, and likely victim losses.",
    "Check_Fraud": "Find repeated check fraud, mail theft, check washing, or forged check behavior and likely victim impact.",
    "Crypto_Fraud": "Find repeated crypto scam or wallet theft behavior, what victims are persuaded to do, and likely financial impact.",
    "Tech_Support_Scam": "Find repeated tech support scam behavior, remote-access tactics, and likely victim impact.",
    "Investment_Scam": "Find repeated fraudulent investment solicitations, promised returns, and likely investor losses.",
    "BEC_Business_Email_Compromise": "Find repeated business email compromise patterns, impersonation attempts, and payment redirection risk.",
    "Money_Laundering": "Find repeated money laundering structuring, mule account behavior, and layering patterns.",
    "Sanctions": "Find repeated sanctions evasion, export control violations, and sanctioned entity behavior.",
    "Terrorist_Financing": "Find repeated terrorist financing patterns and methods.",
    "Human_Trafficking": "Find repeated human or labor trafficking patterns, recruitment methods, and victim impact.",
    "Elder_Fraud": "Find repeated fraud targeting elderly or senior victims, impersonation of government agencies, and financial or identity impact.",
    "Military_Scam": "Find repeated scams targeting military members or veterans including romance scams, VA fraud, and BAH fraud.",
    "Data_Breach": "Find repeated data breach incidents, credential exposure, and downstream fraud risk from leaked personal information.",
    "Consumer_Billing_Fraud": "Find repeated unauthorized charges, fake subscriptions, rental scams, and marketplace fraud affecting consumers.",
    "General_Scam": "Find repeated scam behavior, the deception used, and likely victim impact.",
    "Other_Unclear": "Find the most repeated suspicious behavior, if any, and likely user risk.",
}


def _load_embedder(model_name: str = EMBED_MODEL_NAME):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _build_query(theme: str, custom_query: Optional[str] = None) -> str:
    if custom_query and str(custom_query).strip():
        return str(custom_query).strip()
    return THEME_QUERY_MAP.get(
        theme,
        f"Find the repeated pattern, affected users, and likely risk for theme {theme}."
    )


def _text_len(value) -> int:
    if pd.isna(value):
        return 0
    return len(str(value).strip())


def _evidence_quality(row: pd.Series) -> float:
    rss_len = _text_len(row.get("rss_summary_text", ""))
    title_len = _text_len(row.get("title", ""))

    quality = 0.0
    quality += min(rss_len / 200.0, 1.0) * 0.90
    quality += min(title_len / 80.0, 1.0) * 0.10
    return float(quality)


def _has_minimum_evidence(row: pd.Series) -> bool:
    rss_len = _text_len(row.get("rss_summary_text", ""))
    title_len = _text_len(row.get("title", ""))

    if rss_len >= 80:
        return True
    if rss_len >= 40 and title_len >= 20:
        return True
    if title_len >= 20:
        return True
    return False


def retrieve_theme_evidence(
    embeddings_parquet_path: str | Path,
    faiss_index_path: str | Path,
    theme: str,
    top_k: int = 8,
    custom_query: Optional[str] = None,
    min_theme_score: float = 0.0,
    include_general_backfill: bool = False,
    model_name: str = EMBED_MODEL_NAME,
) -> pd.DataFrame:
    import faiss

    embeddings_parquet_path = Path(embeddings_parquet_path)
    faiss_index_path = Path(faiss_index_path)

    if not embeddings_parquet_path.exists():
        raise FileNotFoundError(f"Missing embeddings parquet: {embeddings_parquet_path}")
    if not faiss_index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {faiss_index_path}")

    df = pd.read_parquet(embeddings_parquet_path).copy()
    if df.empty:
        return df

    if "embedding_id" not in df.columns:
        raise ValueError("Embeddings parquet must contain 'embedding_id'.")

    df["theme"] = df.get("theme", "Other_Unclear").fillna("Other_Unclear").astype(str)
    df["theme_score"] = pd.to_numeric(df.get("theme_score", 0.0), errors="coerce").fillna(0.0)
    df["rss_summary_text"] = df.get("rss_summary_text", "").fillna("").astype(str)
    df["title"] = df.get("title", "").fillna("").astype(str)

    if include_general_backfill:
        candidate_mask = (
            (df["theme"] == theme) |
            ((df["theme"].isin(["General_Scam", "Other_Unclear"])) & (df["theme_score"] >= min_theme_score))
        )
    else:
        candidate_mask = (df["theme"] == theme)

    candidates = df.loc[candidate_mask].copy()

    if min_theme_score > 0:
        candidates = candidates.loc[candidates["theme_score"] >= min_theme_score].copy()

    if candidates.empty:
        return candidates

    candidates["evidence_quality"] = candidates.apply(_evidence_quality, axis=1)
    candidates["has_min_evidence"] = candidates.apply(_has_minimum_evidence, axis=1)
    candidates = candidates.loc[candidates["has_min_evidence"]].copy()

    if candidates.empty:
        return candidates

    index = faiss.read_index(str(faiss_index_path))

    embedder = _load_embedder(model_name)
    query_text = _build_query(theme=theme, custom_query=custom_query)

    query_vector = embedder.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")[0]

    query_vector = _normalize_vector(query_vector).astype("float32").reshape(1, -1)

    candidate_ids = candidates["embedding_id"].astype(int).tolist()
    candidate_vectors = np.vstack([index.reconstruct(int(i)) for i in candidate_ids]).astype("float32")

    temp_index = faiss.IndexFlatIP(candidate_vectors.shape[1])
    temp_index.add(candidate_vectors)

    search_k = min(max(top_k * 3, top_k), len(candidates))
    scores, positions = temp_index.search(query_vector, search_k)

    ranked = candidates.iloc[positions[0]].copy().reset_index(drop=True)
    ranked["retrieval_score"] = scores[0]

    max_theme_score = ranked["theme_score"].max() if len(ranked) else 0.0
    if max_theme_score <= 0:
        ranked["normalized_theme_score"] = 0.0
    else:
        ranked["normalized_theme_score"] = ranked["theme_score"] / max_theme_score

    ranked["final_rank_score"] = (
        0.60 * ranked["retrieval_score"].astype(float) +
        0.25 * ranked["normalized_theme_score"].astype(float) +
        0.15 * ranked["evidence_quality"].astype(float)
    )

    ranked = ranked.sort_values(
        ["final_rank_score", "retrieval_score", "theme_score", "evidence_quality"],
        ascending=[False, False, False, False]
    ).head(top_k).reset_index(drop=True)

    ranked["retrieval_rank"] = np.arange(1, len(ranked) + 1)
    ranked["retrieval_query"] = query_text
    ranked["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()

    preferred_cols = [
        "retrieval_rank",
        "final_rank_score",
        "retrieval_score",
        "normalized_theme_score",
        "evidence_quality",
        "retrieval_query",
        "post_id",
        "run_date",
        "theme",
        "theme_score",
        "theme_matches",
        "title",
        "rss_summary_text",
        "retrieval_text",
        "embedding_id",
        "embedding_model",
    ]
    ordered_cols = [c for c in preferred_cols if c in ranked.columns] + [c for c in ranked.columns if c not in preferred_cols]
    ranked = ranked[ordered_cols]

    return ranked


def run(
    run_date: str,
    theme: str,
    top_k: int = 8,
    custom_query: Optional[str] = None,
    include_general_backfill: bool = False,
    min_theme_score: float = 0.0,
    config_path: str = "config.yaml",
) -> pd.DataFrame:
    cfg = load_pipeline_config(config_path)
    paths = resolve_data_paths(cfg.data_root, run_date, dataset="reddit")

    embeddings_parquet_path = paths.scored_dir / "reddit_post_embeddings.parquet"
    faiss_index_path = paths.scored_dir / "reddit_post_embeddings.faiss"
    output_csv_path = paths.summaries_dir / f"reddit_theme_evidence_{theme}.csv"
    output_parquet_path = paths.summaries_dir / f"reddit_theme_evidence_{theme}.parquet"

    result = retrieve_theme_evidence(
        embeddings_parquet_path=embeddings_parquet_path,
        faiss_index_path=faiss_index_path,
        theme=theme,
        top_k=top_k,
        custom_query=custom_query,
        include_general_backfill=include_general_backfill,
        min_theme_score=min_theme_score,
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_csv_path, index=False)
    result.to_parquet(output_parquet_path, index=False)

    print("Wrote evidence CSV:", output_csv_path)
    print("Wrote evidence Parquet:", output_parquet_path)
    print("Rows returned:", len(result))

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve top-k semantic evidence rows for a theme.")
    parser.add_argument("--run-date", required=True, help="Run date YYYY-MM-DD")
    parser.add_argument("--theme", required=True, help="Theme to retrieve evidence for")
    parser.add_argument("--top-k", type=int, default=8, help="Number of rows to retrieve")
    parser.add_argument("--custom-query", default=None, help="Optional custom semantic retrieval query")
    parser.add_argument("--include-general-backfill", action="store_true", help="Allow General_Scam and Other_Unclear rows into the candidate pool")
    parser.add_argument("--min-theme-score", type=float, default=0.0, help="Minimum theme_score for candidates")
    parser.add_argument("--config-path", default="config.yaml", help="Pipeline config path")
    args = parser.parse_args()

    run(
        run_date=args.run_date,
        theme=args.theme,
        top_k=args.top_k,
        custom_query=args.custom_query,
        include_general_backfill=args.include_general_backfill,
        min_theme_score=args.min_theme_score,
        config_path=args.config_path,
    )
