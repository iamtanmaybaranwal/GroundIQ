"""Structure-aware chunking.

Chunk boundaries decide what the model can cite. Splitting every N characters
cuts sentences in half and destroys the heading context a passage needs to make
sense on its own, so this splitter works down the document's own structure:
sections (markdown headings) → paragraphs → sentences, with overlap so an answer
that straddles a boundary is still retrievable from one chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .text import estimate_tokens, sentences

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    section: str
    ordinal: int
    token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "section": self.section,
            "ordinal": self.ordinal,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Chunk":
        return Chunk(**raw)


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading path, body) pairs, keeping the heading hierarchy."""
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    current: list[str] = []
    heading_path = ""

    for line in markdown.splitlines():
        match = HEADING.match(line.strip())
        if match:
            if current:
                sections.append((heading_path, "\n".join(current).strip()))
                current = []
            level = len(match.group(1))
            title = match.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            heading_path = " > ".join(stack)
        else:
            current.append(line)

    if current:
        sections.append((heading_path, "\n".join(current).strip()))
    return [(path, body) for path, body in sections if body]


def chunk_document(
    doc_id: str,
    text: str,
    *,
    target_tokens: int = 220,
    overlap_sentences: int = 1,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split one document into retrievable, self-describing chunks."""
    chunks: list[Chunk] = []
    ordinal = 0

    for section, body in split_sections(text) or [("", text)]:
        buffer: list[str] = []
        buffer_tokens = 0

        def flush(carry: list[str]) -> list[str]:
            nonlocal ordinal, buffer_tokens
            if not buffer:
                return []
            body_text = " ".join(buffer).strip()
            # The heading is prepended so the chunk carries its own context into
            # the embedding *and* into the model's prompt.
            full_text = (section + "\n" + body_text).strip() if section else body_text
            chunks.append(
                Chunk(
                    id=f"{doc_id}#{ordinal}",
                    doc_id=doc_id,
                    text=full_text,
                    section=section,
                    ordinal=ordinal,
                    token_estimate=estimate_tokens(full_text),
                    metadata=dict(metadata or {}),
                )
            )
            ordinal += 1
            buffer_tokens = sum(estimate_tokens(s) for s in carry)
            return list(carry)

        for sentence in sentences(body):
            sentence_tokens = estimate_tokens(sentence)
            if buffer and buffer_tokens + sentence_tokens > target_tokens:
                carry = buffer[-overlap_sentences:] if overlap_sentences else []
                buffer = flush(carry)
            buffer.append(sentence)
            buffer_tokens += sentence_tokens

        flush([])

    return chunks
