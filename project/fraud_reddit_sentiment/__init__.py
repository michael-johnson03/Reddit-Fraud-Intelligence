"""Reddit fraud theme intelligence pipeline.

End-to-end RAG pipeline that ingests Reddit posts from fraud-relevant
subreddits, classifies them against an 18-theme fraud taxonomy, embeds
them with sentence-transformers + FAISS, and synthesizes structured
analytical briefs via Mistral-7B-Instruct-v0.2 (4-bit NF4 quantized).

Entry point: fraud_reddit_sentiment.cli
"""

__version__ = "0.1.0"
