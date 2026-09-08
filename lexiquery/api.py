"""FastAPI service: ingest, search, ask, evaluate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .config import Settings, settings as default_settings
from .evaluation.harness import evaluate, load_questions
from .pipeline import Document, LexiQuery


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


class IngestDocument(BaseModel):
    id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1)


def create_app(engine: LexiQuery | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or default_settings
    app = FastAPI(
        title="LexiQuery",
        description="Hybrid RAG engine: BM25 + dense retrieval, RRF fusion, reranking, "
        "grounded answers with citations, and a built-in evaluation harness.",
        version="1.0.0",
    )

    if engine is None:
        engine = LexiQuery(settings)
        corpus = Path(settings.corpus_path)
        if corpus.exists():
            engine.add_directory(corpus)
            engine.build()

    app.state.engine = engine
    app.state.settings = settings

    def require_api_key(x_api_key: str = Header(default="")) -> None:
        if x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="valid X-API-Key header required")

    guard = [Depends(require_api_key)]

    @app.get("/health", tags=["ops"])
    def health(request: Request):
        return {"status": "ok", **request.app.state.engine.stats()}

    @app.post("/v1/ask", dependencies=guard, tags=["rag"])
    def ask(payload: AskRequest, request: Request):
        return request.app.state.engine.ask(payload.question, top_k=payload.top_k)

    @app.post("/v1/search", dependencies=guard, tags=["rag"])
    def search(payload: SearchRequest, request: Request):
        results = request.app.state.engine.retrieve(payload.query, top_k=payload.top_k)
        return {"query": payload.query, "results": [r.to_dict() for r in results]}

    @app.post("/v1/documents", status_code=201, dependencies=guard, tags=["corpus"])
    def ingest(payload: IngestRequest, request: Request):
        engine: LexiQuery = request.app.state.engine
        added = 0
        for document in payload.documents:
            added += len(
                engine.add_document(
                    Document(
                        id=document.id,
                        text=document.text,
                        title=document.title,
                        metadata=document.metadata,
                    )
                )
            )
        engine.build()  # re-fit: the corpus vocabulary changed
        return {"documents": len(payload.documents), "chunks_added": added, **engine.stats()}

    @app.get("/v1/documents", dependencies=guard, tags=["corpus"])
    def list_documents(request: Request):
        engine: LexiQuery = request.app.state.engine
        return {
            "documents": [
                {
                    "id": document.id,
                    "title": document.title,
                    "chunks": sum(1 for c in engine.chunks.values() if c.doc_id == document.id),
                }
                for document in engine.documents.values()
            ]
        }

    @app.post("/v1/evaluate", dependencies=guard, tags=["evaluation"])
    def run_evaluation(request: Request, path: str = "eval/questions.json"):
        cases, out_of_scope = load_questions(path)
        report = evaluate(request.app.state.engine, cases, out_of_scope)
        return report["summary"]

    return app


app = create_app  # uvicorn lexiquery.api:app --factory
