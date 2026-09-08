"""LexiQuery CLI.

    python -m lexiquery.cli ask "how do I roll back a release?"
    python -m lexiquery.cli search "rate limit headers"
    python -m lexiquery.cli evaluate
    python -m lexiquery.cli ablation
    python -m lexiquery.cli index --save artifacts/index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import settings
from .evaluation.harness import compare_strategies, evaluate, load_questions
from .pipeline import LexiQuery


def build_engine(corpus: str) -> LexiQuery:
    engine = LexiQuery(settings)
    count = engine.add_directory(corpus)
    if count == 0:
        raise SystemExit("no documents found in " + corpus)
    engine.build()
    return engine


def print_result(result: dict) -> None:
    print("\nQ: " + result["question"])
    print("-" * 78)
    print(result["answer"])
    print("-" * 78)
    if result["citations"]:
        print("sources:")
        for citation in result["citations"]:
            print("  [{0}] {1} :: {2}".format(citation["n"], citation["doc_id"], citation["section"]))
    print(
        "confidence {0} | refused {1} | {2} ms | provider {3}".format(
            result["confidence"], result["refused"], result["latency_ms"], result["provider"]
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lexiquery", description="Hybrid RAG over a document corpus")
    parser.add_argument("--corpus", default=settings.corpus_path)
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--top-k", type=int, default=settings.top_k)

    search = sub.add_parser("search", help="show the retrieved passages and their signals")
    search.add_argument("query", nargs="+")
    search.add_argument("--top-k", type=int, default=settings.top_k)

    sub.add_parser("evaluate", help="run the labelled evaluation set")
    sub.add_parser("ablation", help="compare bm25 / vector / hybrid retrieval")
    sub.add_parser("stats", help="show index statistics")

    index = sub.add_parser("index", help="build and persist the index")
    index.add_argument("--save", default=settings.index_path)

    args = parser.parse_args(argv)
    engine = build_engine(args.corpus)

    if args.command == "ask":
        print_result(engine.ask(" ".join(args.question), top_k=args.top_k))
        return 0

    if args.command == "search":
        query = " ".join(args.query)
        print("\nquery: " + query)
        print("{0:<28} {1:>8} {2:>8} {3:>8} {4:>9}".format("chunk", "fused", "bm25", "vector", "rerank"))
        for scored in engine.retrieve(query, top_k=args.top_k):
            print(
                "{0:<28} {1:>8.4f} {2:>8.3f} {3:>8.3f} {4:>9.3f}".format(
                    scored.chunk.id[:28],
                    scored.score,
                    scored.bm25_score,
                    scored.vector_score,
                    scored.rerank_score or 0.0,
                )
            )
        return 0

    if args.command == "evaluate":
        cases, out_of_scope = load_questions("eval/questions.json")
        report = evaluate(engine, cases, out_of_scope)
        print("\nRetrieval and answer quality on {0} labelled questions".format(len(cases)))
        print("-" * 78)
        for metric, value in report["summary"].items():
            print("  {0:<28} {1}".format(metric, value))
        failures = [q for q in report["per_question"] if q["hit@3"] == 0]
        if failures:
            print("\n  misses:")
            for failure in failures:
                print("    {0}: expected {1}, got {2}".format(failure["id"], failure["expected"], failure["retrieved"][:3]))
        return 0

    if args.command == "ablation":
        cases, _ = load_questions("eval/questions.json")
        table = compare_strategies(engine, cases)
        print("\nRetrieval strategy comparison ({0} questions)".format(len(cases)))
        print("-" * 78)
        print("  {0:<16} {1:>8} {2:>8} {3:>10} {4:>8} {5:>8}".format("strategy", "hit@1", "hit@3", "recall@5", "mrr", "ndcg@5"))
        for name, metrics in table.items():
            print(
                "  {0:<16} {1:>8} {2:>8} {3:>10} {4:>8} {5:>8}".format(
                    name, metrics["hit@1"], metrics["hit@3"], metrics["recall@5"], metrics["mrr"], metrics["ndcg@5"]
                )
            )
        return 0

    if args.command == "stats":
        print(json.dumps(engine.stats(), indent=2))
        return 0

    if args.command == "index":
        path = engine.save(args.save)
        print("index written to " + str(Path(path).resolve()))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
