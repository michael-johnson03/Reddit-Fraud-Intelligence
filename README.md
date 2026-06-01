# reddit-fraud-intelligence

End-to-end Retrieval-Augmented Generation pipeline that ingests Reddit posts from fraud-relevant subreddits via RSS, classifies each post against an 18-theme fraud taxonomy with transparent keyword matching, builds FAISS-indexed semantic embeddings, and uses a locally-hosted **Mistral-7B-Instruct-v0.2 (4-bit NF4 quantized)** to generate one structured analytical brief per fraud theme per run.

> 🚧 **Repository under preparation.** Built as part of a graduate capstone in partnership with a financial-services institution. Public release is in progress — sponsor-specific configuration and demographic targeting are being sanitized before publication.

---

## What this project will contain

- Full Python package (`fraud_reddit_sentiment/`) with CLI entry point
- Production-style pipeline modules: ingestion, preprocessing, classification, retrieval, summarization
- 18-theme weighted fraud taxonomy with auditable theme-match traces
- FAISS vector store with sentence-transformers/all-MiniLM-L6-v2 embeddings
- Grounded LLM synthesis via Mistral-7B with verbatim-copy quality gates
- 5 dated production runs (sample outputs and architecture documentation)
- Streamlit dashboard integration
- Tests for core classification and preprocessing logic

## Tech stack

Python · FAISS · sentence-transformers · Mistral-7B-Instruct-v0.2 · 4-bit NF4 quantization · Streamlit · Google Colab (A100) · Parquet

---

For early discussion or questions about the work, reach out via [LinkedIn](https://www.linkedin.com/in/michael-d-johnson3/).

