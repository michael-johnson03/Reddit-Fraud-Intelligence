# fraud_reddit_sentiment/preprocessing/clean_text.py
"""Text cleaning for Reddit RSS social text."""

from __future__ import annotations
import html
import re

_URL = re.compile(r"http\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_REDDIT_SUBMITTED = re.compile(r"submitted by\s+/u/\S+", re.IGNORECASE)
_REDDIT_USER = re.compile(r"/u/\S+")
_REDDIT_LINK = re.compile(r"\[link\]|\[comments\]", re.IGNORECASE)
_SC_TAGS = re.compile(r"<!--\s*SC_OFF\s*-->|<!--\s*SC_ON\s*-->", re.IGNORECASE)
_EXTRA_PUNCT = re.compile(r"\s+([.,!?])")
_WS = re.compile(r"\s+")


def clean(s: str, max_words: int = 120) -> str:
    """Clean Reddit RSS text and truncate to max_words.

    Removes:
    - HTML tags and entities
    - URLs
    - Reddit 'submitted by /u/username' footers
    - [link] [comments] artifacts
    - SC_OFF / SC_ON comment markers
    - Excess whitespace

    Truncates to max_words to keep evidence blocks focused.
    """
    if not s:
        return ""
    s = _SC_TAGS.sub(" ", s)         # remove SC_OFF/SC_ON markers
    s = _HTML_TAG.sub(" ", s)        # strip residual HTML tags
    s = html.unescape(s)             # unescape &amp; &quot; &#39; etc.
    s = _URL.sub("", s)              # remove URLs
    s = _REDDIT_SUBMITTED.sub("", s) # remove "submitted by /u/username"
    s = _REDDIT_USER.sub("", s)      # remove remaining /u/username
    s = _REDDIT_LINK.sub("", s)      # remove [link] [comments]
    s = _EXTRA_PUNCT.sub(r"\1", s)   # fix spacing before punctuation
    s = _WS.sub(" ", s).strip()      # collapse whitespace

    # Truncate to max_words
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).rstrip(",;:") + "..."

    return s
