from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from fraud_reddit_sentiment.config_utils import load_pipeline_config
from fraud_reddit_sentiment.io_utils import resolve_data_paths
from fraud_reddit_sentiment.preprocessing.clean_text import clean

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SYNTH_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
TOP_K = 8


# ---------------------------------------------------------------------------
# Per-theme FAISS index (in memory)
# ---------------------------------------------------------------------------

def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return vectors / norms


def _build_theme_index(
    theme_df: pd.DataFrame,
    embedder: SentenceTransformer,
) -> tuple[faiss.IndexFlatIP, pd.DataFrame]:
    """Build an in-memory FAISS index for one theme's posts."""
    theme_df = theme_df.copy().reset_index(drop=True)

    texts = []
    for _, row in theme_df.iterrows():
        title = clean(str(row.get("title", "") or ""))
        summary = clean(str(row.get("rss_summary_text", "") or ""))
        detail = summary if len(summary) > 30 else title
        texts.append(detail)

    theme_df["chunk_text"] = texts

    vectors = embedder.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype("float32")

    vectors = _normalize(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, theme_df


def _retrieve_chunks(
    theme: str,
    index: faiss.IndexFlatIP,
    theme_df: pd.DataFrame,
    embedder: SentenceTransformer,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Query the per-theme index with a theme-specific analytical query."""
    query_map = {
        "ATO_Account_Takeover": "repeated account takeover patterns unauthorized access and victim impact",
        "Phishing_Smishing": "repeated phishing smishing vishing tactics contact methods and victim risk",
        "Identity_Theft": "repeated identity theft behavior fraudulent account creation victim impact",
        "Benefits_Fraud": "repeated government benefits fraud unemployment tax fraud and victim impact",
        "Payment_Scams_P2P": "repeated peer to peer payment scam tactics fake payments and victim losses",
        "Check_Fraud": "repeated check fraud mail theft check washing forged check and victim impact",
        "Crypto_Fraud": "repeated crypto scam wallet theft tactics and financial victim impact",
        "Tech_Support_Scam": "repeated tech support scam remote access tactics and victim impact",
        "Investment_Scam": "repeated fraudulent investment solicitations promised returns and investor losses",
        "BEC_Business_Email_Compromise": "repeated business email compromise impersonation payment redirection risk",
        "Money_Laundering": "repeated money laundering structuring mule account layering behavior",
        "Sanctions": "repeated sanctions evasion export control violations and sanctioned entity behavior",
        "Terrorist_Financing": "repeated terrorist financing patterns and funding methods",
        "Human_Trafficking": "repeated human trafficking labor trafficking recruitment and victim impact",
        "Elder_Fraud": "repeated fraud targeting elderly seniors veterans fake government agency impersonation and financial impact",
        "Military_Scam": "repeated scams targeting military members veterans romance scams VA benefit fraud BAH fraud",
        "Data_Breach": "repeated data breach credential exposure personal information leaked and downstream fraud risk",
        "Consumer_Billing_Fraud": "repeated unauthorized charges fake subscriptions rental scams marketplace fraud consumer billing",
        "General_Scam": "repeated scam deception tactics and victim impact",
        "Other_Unclear": "repeated suspicious behavior and likely user risk",
    }
    query = query_map.get(theme, f"repeated fraud pattern and victim risk for {theme}")

    q_vec = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    q_vec = _normalize(q_vec)

    k = min(top_k, len(theme_df))
    scores, positions = index.search(q_vec, k)

    retrieved = theme_df.iloc[positions[0]].copy().reset_index(drop=True)
    retrieved["retrieval_score"] = scores[0]
    return retrieved


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _extract_watch_signals(theme_df: pd.DataFrame, top_k: int = 8) -> str:
    counter = Counter()
    for val in theme_df.get("theme_matches", pd.Series(dtype=str)).fillna("").astype(str):
        parts = [p.strip() for p in re.split(r"[|,;]", val) if p.strip()]
        counter.update(parts)
    repeated = [s for s, c in counter.most_common(top_k) if c >= 2]
    if not repeated:
        repeated = [s for s, _ in counter.most_common(top_k)]
    return ", ".join(repeated) if repeated else "insufficient signal data"


# ---------------------------------------------------------------------------
# Evidence block for synthesis
# ---------------------------------------------------------------------------

def _is_narrative_post(text: str) -> bool:
    """Detect single-story narrative posts that will cause Mistral to copy instead of synthesize."""
    if not text:
        return False
    s = text.strip()
    s_lower = s.lower()

    # Truncated fragment — starts with lowercase (mid-sentence cut)
    if s and s[0].islower():
        return True

    # Starts with ellipsis or continuation marker
    if s.startswith(("...", "…", "- ", "* ")):
        return True

    # First-person narrative starters
    bad_starts = [
        "i ", "i'", "i've", "i was", "i got", "i am", "i had", "i just",
        "my ", "so i", "ok so", "okay so", "hi ", "hello ", "hey ",
        "well ", "update:", "edit:", "tldr", "tl;dr",
    ]
    if any(s_lower.startswith(p) for p in bad_starts):
        return True

    return False


def _truncate_for_summary(text: str, max_words: int = 150) -> str:
    """Truncate cleaned post text for inclusion in Mistral evidence block."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def _build_evidence_block(retrieved_df: pd.DataFrame, max_words_per_post: int = 150) -> str:
    """Build evidence block for Mistral.

    Posts are fully cleaned (no truncation at ingest) but truncated here
    before passing to Mistral so the total context stays within token limits.
    20 posts x 150 words = ~3000 words, well within Mistral's 3000 token prompt limit.
    """
    parts = []
    skipped = 0
    for _, row in retrieved_df.iterrows():
        text = clean(str(row.get("chunk_text", "") or ""))
        if not text or len(text.split()) < 6:
            continue
        if _is_narrative_post(text):
            skipped += 1
            continue
        parts.append(f"- {_truncate_for_summary(text, max_words_per_post)}")

    if len(parts) < 3:
        # Fallback: include narrative posts if not enough non-narrative evidence
        for _, row in retrieved_df.iterrows():
            text = clean(str(row.get("chunk_text", "") or ""))
            if text and len(text.split()) >= 6:
                entry = f"- {_truncate_for_summary(text, max_words_per_post)}"
                if entry not in parts:
                    parts.append(entry)

    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Mistral-7B synthesis
# ---------------------------------------------------------------------------

def _load_mistral(model_name: str = SYNTH_MODEL_NAME):
    print(f"Loading synthesis model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()
    print("Synthesis model loaded.\n")
    return tokenizer, model


def _synthesize(
    theme: str,
    evidence_block: str,
    tokenizer,
    model,
    max_new_tokens: int = 200,
    do_sample: bool = False,
) -> str:
    prompt = f"""<s>[INST] You are a fraud analyst writing a briefing for a financial crime dashboard.

Below are reports from online fraud communities grouped under the theme: {theme}

Evidence:
{evidence_block}

Write a structured analytical summary with exactly these three sections:

Pattern: [2-3 sentences describing the repeated tactic or behavior seen across multiple reports. Do not describe a single post. Focus on what is common across reports.]

Targets & Impact: [1-2 sentences on who is being affected and what harm occurs - financial loss, account access, identity exposure.]

Watch signals: [comma separated list of the key terms or behaviors that indicate this fraud type is active]

Do not copy posts verbatim. Use neutral analyst language. [/INST]"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3000,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7 if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _is_low_quality(text: str) -> bool:
    if not text:
        return True
    s = text.strip().lower()
    if len(s.split()) < 15:
        return True
    # Prompt leakage or unsolicited sections
    bad = [
        "[inst]", "evidence:", "below are reports", "[link]", "[comments]",
        "submitted by", "recommendations:", "recommendation:", "note:",
        "disclaimer:", "write a structured",
    ]
    if any(p in s for p in bad):
        return True
    # Must start with Pattern: section — anything else is raw copy or prompt leakage
    if not s.startswith("pattern:"):
        return True
    # Repetition detection — if same sentence appears 3+ times it's looping
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]
    if len(sentences) != len(set(sentences)) and len(sentences) > 3:
        from collections import Counter
        counts = Counter(sentences)
        if max(counts.values()) >= 3:
            return True
    return False


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_theme_summaries(
    run_date: str,
    embeddings_parquet_path: str | Path,
    output_csv_path: str | Path,
    output_parquet_path: str | Path,
    embed_model_name: str = EMBED_MODEL_NAME,
    synth_model_name: str = SYNTH_MODEL_NAME,
    top_k: int = TOP_K,
    min_posts_per_theme: int = 3,
) -> pd.DataFrame:

    embeddings_parquet_path = Path(embeddings_parquet_path)
    df = pd.read_parquet(embeddings_parquet_path).copy()
    if df.empty:
        raise ValueError(f"No rows found in {embeddings_parquet_path}")

    df["theme"] = df["theme"].fillna("Other_Unclear").astype(str)
    themes = sorted(df["theme"].unique().tolist())

    # Load embedding model
    print(f"Loading embedding model: {embed_model_name}")
    embedder = SentenceTransformer(embed_model_name)
    print("Embedding model loaded.\n")

    # Load synthesis model
    tokenizer, synth_model = _load_mistral(synth_model_name)

    rows = []

    for theme in themes:
        print(f"{'='*50}")
        print(f"Theme: {theme}")

        theme_df = df[df["theme"] == theme].copy().reset_index(drop=True)
        post_count = len(theme_df)
        print(f"Posts in theme: {post_count}")

        watch_signals = _extract_watch_signals(theme_df)

        # For BEC, backfill with General_Scam and Payment posts if sparse
        if theme == "BEC_Business_Email_Compromise" and post_count < 10:
            backfill = df[df["theme"].isin(["General_Scam", "Payment_Scams_P2P"])].copy()
            theme_df = pd.concat([theme_df, backfill], ignore_index=True).drop_duplicates(subset=["post_id"])
            post_count = len(theme_df)
            print(f"  BEC backfill applied — total posts: {post_count}")

        if post_count < min_posts_per_theme:
            print(f"  Skipping — fewer than {min_posts_per_theme} posts.")
            summary_text = (
                f"Pattern: Insufficient posts ({post_count}) to detect a reliable pattern.\n\n"
                f"Targets & Impact: Insufficient evidence.\n\n"
                f"Watch signals: {watch_signals}"
            )
        else:
            # Dynamic top_k based on post count
            if post_count >= 50:
                dynamic_k = 20
            elif post_count >= 20:
                dynamic_k = 15
            else:
                dynamic_k = min(top_k, post_count)

            # Build per-theme FAISS index
            index, indexed_df = _build_theme_index(theme_df, embedder)

            # Retrieve most relevant chunks
            retrieved = _retrieve_chunks(theme, index, indexed_df, embedder, top_k=dynamic_k)
            evidence_block = _build_evidence_block(retrieved)

            print(f"  Retrieved chunks: {len(retrieved)} (dynamic_k={dynamic_k})")

            # Synthesize with retry on low quality output
            raw = _synthesize(theme, evidence_block, tokenizer, synth_model)
            print(f"  Raw output preview: {raw[:80].strip()!r}")
            print(f"  Low quality: {_is_low_quality(raw)}")

            if _is_low_quality(raw):
                # Retry once with do_sample=True to break pattern latch
                print(f"  Retrying generation...")
                raw = _synthesize(theme, evidence_block, tokenizer, synth_model, do_sample=True)
                print(f"  Retry preview: {raw[:80].strip()!r}")
                print(f"  Retry low quality: {_is_low_quality(raw)}")

            if _is_low_quality(raw):
                summary_text = (
                    f"Pattern: No consistent pattern detected from available evidence.\n\n"
                    f"Targets & Impact: Insufficient evidence to determine victim impact.\n\n"
                    f"Watch signals: {watch_signals}"
                )
            else:
                # Always replace Watch signals with aggregated taxonomy signals
                # Split on Watch signals: and rebuild cleanly
                if "Watch signals:" in raw:
                    body = raw[:raw.index("Watch signals:")].rstrip()
                    summary_text = f"{body}\n\nWatch signals: {watch_signals}"
                else:
                    # Mistral didn't include Watch signals section — append it
                    summary_text = raw.strip() + f"\n\nWatch signals: {watch_signals}"

        rows.append({
            "run_date": run_date,
            "theme": theme,
            "post_count": post_count,
            "theme_summary_text": summary_text,
            "model_name": synth_model_name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"  Done.\n")

    out_df = pd.DataFrame(rows).sort_values(["run_date", "theme"]).reset_index(drop=True)

    output_csv_path = Path(output_csv_path)
    output_parquet_path = Path(output_parquet_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_csv_path, index=False)
    out_df.to_parquet(output_parquet_path, index=False)

    return out_df


def run(
    run_date: str,
    config_path: str = "config.yaml",
    top_k: int = TOP_K,
    min_posts_per_theme: int = 3,
) -> pd.DataFrame:
    cfg = load_pipeline_config(config_path)
    paths = resolve_data_paths(cfg.data_root, run_date, dataset="reddit")

    embeddings_parquet_path = paths.scored_dir / "reddit_post_embeddings.parquet"
    output_csv_path = paths.summaries_dir / "reddit_theme_summaries.csv"
    output_parquet_path = paths.summaries_dir / "reddit_theme_summaries.parquet"

    if not embeddings_parquet_path.exists():
        raise FileNotFoundError(f"Embeddings parquet not found: {embeddings_parquet_path}")

    result = build_theme_summaries(
        run_date=run_date,
        embeddings_parquet_path=embeddings_parquet_path,
        output_csv_path=output_csv_path,
        output_parquet_path=output_parquet_path,
        top_k=top_k,
        min_posts_per_theme=min_posts_per_theme,
    )

    print(f"Wrote theme summaries CSV:     {output_csv_path}")
    print(f"Wrote theme summaries Parquet: {output_parquet_path}")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Per-theme RAG summarization with Mistral-7B 4-bit.")
    parser.add_argument("--run-date", required=True)
    parser.add_argument("--config-path", default="config.yaml")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-posts-per-theme", type=int, default=3)
    args = parser.parse_args()

    run(
        run_date=args.run_date,
        config_path=args.config_path,
        top_k=args.top_k,
        min_posts_per_theme=args.min_posts_per_theme,
    )
