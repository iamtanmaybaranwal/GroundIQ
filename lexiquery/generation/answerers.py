"""Answer generation.

Three interchangeable answerers behind one interface:

* :class:`ExtractiveAnswerer` - the **default**. It composes the answer from
  sentences that actually appear in the retrieved passages, so it is fully
  grounded by construction, runs offline, and is deterministic (which is what
  makes the evaluation harness meaningful).
* :class:`AnthropicAnswerer` / :class:`OpenAIAnswerer` - hosted models for
  fluent synthesis. Both are strictly optional: fill in the API key in `.env`
  and switch one config value.

Every answerer returns citations, and the pipeline refuses to answer when
retrieval confidence is too low - the single most effective guard against a
RAG system confidently inventing policy that does not exist.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..retrieval.hybrid import ScoredChunk
from ..text import sentences, tokenize

REFUSAL = (
    "I could not find this in the knowledge base. Try rephrasing, or check with the "
    "owning team - answering without a supporting source would risk inventing policy."
)

SYSTEM_PROMPT = (
    "You are a precise assistant for an internal engineering knowledge base. "
    "Answer ONLY from the numbered context passages. Cite every claim with the "
    "passage number in square brackets, e.g. [2]. If the passages do not contain "
    "the answer, reply exactly: NOT_IN_CONTEXT."
)


@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    grounded: bool = True
    refused: bool = False
    provider: str = "extractive"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.text,
            "citations": self.citations,
            "grounded": self.grounded,
            "refused": self.refused,
            "provider": self.provider,
            "confidence": round(self.confidence, 4),
        }


def build_context(passages: list[ScoredChunk], token_budget: int = 1_800) -> tuple[str, list[dict]]:
    """Numbered context block plus the citation table, inside a token budget."""
    blocks: list[str] = []
    citations: list[dict] = []
    used = 0

    for index, scored in enumerate(passages, start=1):
        chunk = scored.chunk
        if used + chunk.token_estimate > token_budget:
            break
        blocks.append(f"[{index}] ({chunk.doc_id} :: {chunk.section or 'body'})\n{chunk.text}")
        citations.append(
            {
                "n": index,
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "score": round(scored.score, 6),
            }
        )
        used += chunk.token_estimate

    return "\n\n".join(blocks), citations


class ExtractiveAnswerer:
    """Selects and stitches the sentences that best answer the question."""

    name = "extractive"

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def answer(
        self,
        question: str,
        passages: list[ScoredChunk],
        citations: list[dict],
        term_weights: dict[str, float] | None = None,
    ) -> Answer:
        query_terms = list(dict.fromkeys(tokenize(question)))
        # Weight query terms by IDF so a rare identifier (RDS-4471) outweighs a
        # common verb (means) when choosing which sentences to quote.
        default_weight = max(term_weights.values()) if term_weights else 1.0
        weights = {
            term: (term_weights.get(term, default_weight) if term_weights else 1.0)
            for term in query_terms
        }
        total_weight = sum(weights.values()) or 1.0
        candidates: list[tuple[float, int, str]] = []

        for citation, scored in zip(citations, passages):
            body = scored.chunk.text
            # Chunks carry their heading for retrieval; strip it before quoting.
            if scored.chunk.section and body.startswith(scored.chunk.section):
                body = body[len(scored.chunk.section) :].strip()

            for sentence in sentences(body):
                terms = set(tokenize(sentence))
                if not terms:
                    continue
                overlap = sum(w for term, w in weights.items() if term in terms) / total_weight
                if overlap == 0:
                    continue  # never pad the answer with unrelated text
                # Retrieval rank is a prior; term overlap picks the sentence.
                relevance = 0.75 * overlap + 0.25 * scored.score / (passages[0].score or 1.0)
                length_penalty = 1.0 if 4 <= len(terms) <= 60 else 0.85
                candidates.append((relevance * length_penalty, citation["n"], sentence.strip()))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        chosen: list[tuple[int, str]] = []
        seen: set[str] = set()
        # Only keep sentences close to the best one: a long tail of weak matches
        # reads as padding and dilutes the answer.
        cutoff = candidates[0][0] * 0.5 if candidates else 0.0
        for relevance, number, sentence in candidates:
            if relevance < cutoff:
                break
            key = sentence.lower()[:90]
            if key in seen:
                continue
            seen.add(key)
            chosen.append((number, sentence))
            if len(chosen) >= self.max_sentences:
                break

        if not chosen:
            return Answer(REFUSAL, [], grounded=False, refused=True, provider=self.name)

        # Keep the original passage order so the answer reads coherently.
        chosen.sort(key=lambda item: item[0])
        text = " ".join(f"{sentence} [{number}]" for number, sentence in chosen)
        used = sorted({number for number, _ in chosen})
        return Answer(
            text=text,
            citations=[c for c in citations if c["n"] in used],
            grounded=True,
            provider=self.name,
        )


class AnthropicAnswerer:  # pragma: no cover - requires network + credentials
    """Claude via the official Anthropic SDK (`pip install anthropic`).

    Set ``ANTHROPIC_API_KEY`` in `.env` and ``LEXIQUERY_ANSWERER=anthropic``.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    def answer(
        self,
        question: str,
        passages: list[ScoredChunk],
        citations: list[dict],
        term_weights: dict[str, float] | None = None,
    ) -> Answer:
        import anthropic

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set - fill it in .env")

        context, _ = build_context(passages)
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
                }
            ],
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        refused = text.strip() == "NOT_IN_CONTEXT"
        return Answer(
            text=REFUSAL if refused else text,
            citations=[] if refused else citations,
            grounded=not refused,
            refused=refused,
            provider=self.name,
        )


class OpenAIAnswerer:  # pragma: no cover - requires network + credentials
    """Any OpenAI-compatible chat completions endpoint.

    Set ``LEXIQUERY_OPENAI_API_KEY`` in `.env` and ``LEXIQUERY_ANSWERER=openai``.
    """

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("LEXIQUERY_OPENAI_API_KEY", "")
        self.base_url = os.environ.get("LEXIQUERY_OPENAI_BASE_URL", "https://api.openai.com/v1")

    def answer(
        self,
        question: str,
        passages: list[ScoredChunk],
        citations: list[dict],
        term_weights: dict[str, float] | None = None,
    ) -> Answer:
        import httpx

        if not self.api_key:
            raise RuntimeError("LEXIQUERY_OPENAI_API_KEY is not set - fill it in .env")

        context, _ = build_context(passages)
        response = httpx.post(
            self.base_url + "/chat/completions",
            headers={"Authorization": "Bearer " + self.api_key},
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
                    },
                ],
            },
            timeout=60.0,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
        refused = text == "NOT_IN_CONTEXT"
        return Answer(
            text=REFUSAL if refused else text,
            citations=[] if refused else citations,
            grounded=not refused,
            refused=refused,
            provider=self.name,
        )


def build_answerer(kind: str = "extractive"):
    if kind == "anthropic":
        return AnthropicAnswerer()
    if kind == "openai":
        return OpenAIAnswerer()
    return ExtractiveAnswerer()
