"""Text normalisation and tokenisation shared by the lexical and vector paths.

Retrieval quality is decided here as much as in the ranking function: if the
query tokeniser and the index tokeniser disagree, recall silently collapses.
"""
from __future__ import annotations

import re
import unicodedata

# A compact English stop list. Kept small on purpose: aggressive stop-word
# removal hurts phrase queries like "how to roll back a release".
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "does",
    "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is",
    "it", "its", "of", "on", "or", "our", "that", "the", "their", "then",
    "there", "these", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}

# Interior dots/dashes are kept (rds-4471, v1.2, x-ratelimit) but a token may
# never end in punctuation - otherwise "release." and "release" are different terms.
TOKEN_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9._+-]*[a-z0-9])?")

# Rule-based suffix stripping. Cheap, dependency-free, and enough to make
# "deploys"/"deployment"/"deploying" match "deploy".
_SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness", "ments",
             "ingly", "edly", "ation", "ment", "ness", "ance", "ence", "able",
             "ible", "ing", "ies", "ied", "ers", "est", "ly", "ed", "es", "s")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def stem(token: str) -> str:
    """Light suffix stripping; never shortens a token below four characters."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            base = token[: -len(suffix)]
            # "running" -> "runn" -> "run"
            if len(base) >= 4 and base[-1] == base[-2]:
                base = base[:-1]
            return base
    return token


def tokenize(text: str, *, remove_stopwords: bool = True, apply_stemming: bool = True) -> list[str]:
    tokens = TOKEN_PATTERN.findall(normalise(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if apply_stemming:
        tokens = [stem(t) for t in tokens]
    return [t for t in tokens if len(t) > 1]


def sentences(text: str) -> list[str]:
    """Split into sentences without dragging in an NLP dependency."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}", text.strip())
    return [part.strip() for part in parts if part and part.strip()]


def estimate_tokens(text: str) -> int:
    """~4 characters per token: close enough for context-budget arithmetic."""
    return max(1, len(text) // 4)
