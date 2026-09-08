# LexiQuery — Hybrid RAG Engine with Grounded Answers and a Real Eval Harness

> A retrieval-augmented question answering engine for internal knowledge bases: BM25 + dense
> retrieval fused with Reciprocal Rank Fusion, an explainable reranker, citation-backed answers,
> and a **refusal guardrail that says "I don't know" instead of inventing policy** — with a
> measured evaluation harness that proves every design choice.

[![tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)]()
[![hit@3](https://img.shields.io/badge/hit%403-1.00-brightgreen)]()
[![MRR](https://img.shields.io/badge/MRR-0.977-brightgreen)]()
[![hallucination](https://img.shields.io/badge/out--of--scope%20refusal-100%25-blue)]()

---

## The real-world problem

Every company has the same failure: the answer *is* written down, and nobody can find it. So they
bolt an LLM onto their docs, and hit the four problems that make most RAG demos unusable in
production:

| Problem | What goes wrong | How LexiQuery solves it |
|---|---|---|
| **Hallucination** | The model answers confidently from nothing, and someone follows an invented policy | Every claim carries a citation to a retrieved passage, and the pipeline **refuses** when retrieval confidence is low. Measured: **100% refusal on out-of-scope questions**, 100% citation validity |
| **Vectors miss exact identifiers** | "What is RDS-4471?" returns semantically similar prose about databases, not the runbook line that defines it | BM25 runs alongside embeddings; the fused ranking finds the exact token. Measured: BM25 gets MRR **1.00** on identifier/keyword questions |
| **Keyword search misses paraphrases** | "How do I undo a bad release?" finds nothing, because the doc says "rollback procedure" | Dense retrieval covers the paraphrase. Measured on a deliberately paraphrased set: hybrid lifts hit@3 from **0.67 (BM25) to 0.83** |
| **Chunking destroys context** | A passage split mid-sentence, with no heading, is unciteable and often unusable | Structure-aware chunking down the heading hierarchy, with sentence overlap; every chunk carries its heading path |
| **"Is it any good?"** | Teams ship RAG with no metrics at all | A labelled eval set (22 keyword + 6 paraphrase + 4 out-of-scope questions), retrieval metrics, citation validity, and an **ablation harness** comparing BM25 / vectors / hybrid |

**The whole system runs offline with no API key.** Hosted embeddings, rerankers and LLMs
(Anthropic, OpenAI-compatible) are drop-in providers behind the same interfaces.

---

## Architecture

```
  corpus/*.md
      │
      ▼
┌───────────────────────┐   sections → paragraphs → sentences, with overlap;
│ structure-aware       │   every chunk keeps its heading path so it can stand
│ chunking              │   alone in a prompt and be cited
└──────────┬────────────┘
           ▼
   ┌───────────────┬─────────────────────┐
   ▼               ▼                     │
┌──────────┐  ┌──────────────────┐       │  indexing
│  BM25    │  │ embeddings       │       │
│ (from    │  │ local: TF-IDF+SVD│       │
│ scratch) │  │ or hosted API    │       │
└────┬─────┘  └────────┬─────────┘       │
     │ ranks           │ ranks           │
     └────────┬────────┘                 │
              ▼                          │
   ┌────────────────────────┐            │  retrieval
   │ Reciprocal Rank Fusion │  RRF(d) = Σ 1/(k + rank_i(d)),  k=60
   │ (scale-free)           │  fuses *ranks*, so incomparable score
   └───────────┬────────────┘  scales never need normalising
               ▼
   ┌────────────────────────┐   coverage · proximity · phrase · heading · density
   │ feature reranker       │   → an explainable 0-1 score per candidate
   │ (or hosted x-encoder)  │
   └───────────┬────────────┘
               ▼
   ┌────────────────────────┐   IDF-weighted coverage of the query's information.
   │ GROUNDING GUARDRAIL    │   Below threshold → refuse, with no citations.
   └───────────┬────────────┘
               ▼
   ┌────────────────────────┐   extractive (default, offline, deterministic)
   │ answer generation      │   │ anthropic (claude-opus-5) │ openai-compatible
   └───────────┬────────────┘
               ▼
     answer + citations + per-source ranking signals + confidence + latency
```

### Design decisions, and the numbers behind them

| Decision | Why | Evidence |
|---|---|---|
| **Hybrid, not vectors-only** | Dense retrieval blurs rare identifiers | vector-only MRR 0.943 vs hybrid 0.977 on the keyword set |
| **Hybrid, not BM25-only** | Lexical search dies on paraphrase | On the paraphrase set: BM25 hit@3 0.667 → hybrid **0.833** |
| **RRF over score normalisation** | BM25 scores are unbounded, cosine is [-1,1]; normalising is fragile and corpus-dependent | `test_rrf_is_scale_free` |
| **IDF-weighted grounding signal** | Plain term overlap treats "the" like "RDS-4471", so off-topic questions look grounded | Out-of-scope questions score **0.000** coverage vs 0.26–1.00 for in-scope |
| **Refuse instead of guessing** | A wrong internal policy answer is worse than no answer | 100% refusal on 4 out-of-scope questions, 0% false refusals on 22 in-scope |
| **Extractive answerer as default** | Grounded by construction, deterministic, so the eval harness measures *retrieval*, not model variance | citation validity 1.0 |
| **Local embeddings as default** | The project must be runnable and reproducible by anyone, with no key and no GPU | `pip install -r requirements.txt && pytest` |

> **An honest result:** on the keyword-style eval set, BM25 alone scores a perfect MRR of 1.00 —
> slightly *above* hybrid's 0.977. Small, well-written corpora with keyword-ish questions are
> BM25's best case. That is exactly why the paraphrase set exists: it isolates the case hybrid is
> for, and there hybrid wins on both hit@3 and nDCG. Reporting only the flattering number would
> have hidden the real trade-off.

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Retrieval | BM25 (implemented from the ranking function), exact cosine over a NumPy matrix, RRF fusion |
| Embeddings | scikit-learn TF-IDF + TruncatedSVD (LSA), L2-normalised — or a hosted provider |
| Reranking | Hand-built feature reranker (coverage, min-span proximity, phrase, heading, density) |
| Generation | Extractive (default) · Anthropic `claude-opus-5` via the official SDK · any OpenAI-compatible endpoint |
| Serving | FastAPI + uvicorn, API-key auth, Pydantic schemas |
| Persistence | JSON chunks + `.npy` vectors + joblib embedder — start a service without re-indexing |
| Evaluation | Labelled question set, IR metrics (hit@k, recall@k, MRR, nDCG), citation validity, ablation harness |
| Testing | pytest — 35 tests including **quality gates that fail CI on a retrieval regression** |

---

## Project layout

```
01-lexiquery/
├── lexiquery/
│   ├── text.py                  # tokenisation, stemming, sentence splitting
│   ├── chunking.py              # heading-aware chunking with overlap
│   ├── index/
│   │   ├── bm25.py              # BM25 from scratch (IDF, saturation, length norm)
│   │   ├── embeddings.py        # local LSA embedder + hosted provider
│   │   └── vector.py            # exact cosine search
│   ├── retrieval/
│   │   ├── hybrid.py            # Reciprocal Rank Fusion
│   │   └── rerank.py            # explainable feature reranker (+ hosted option)
│   ├── generation/answerers.py  # extractive / Anthropic / OpenAI-compatible
│   ├── pipeline.py              # the RAG engine + grounding guardrail + persistence
│   ├── evaluation/
│   │   ├── metrics.py           # hit@k, recall@k, precision@k, MRR, nDCG, citation validity
│   │   └── harness.py           # eval runner + retrieval-strategy ablation
│   ├── api.py                   # FastAPI service
│   └── cli.py                   # ask / search / evaluate / ablation / index
├── corpus/                      # 11-document engineering knowledge base
├── eval/questions.json          # 22 labelled + 6 paraphrase + 4 out-of-scope questions
├── tests/                       # 35 tests
└── requirements.txt
```

---

## Quickstart

```bash
cd ai-ml/01-lexiquery

python -m venv .venv
# Windows:  .venv\Scripts\activate     macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: nothing in it is required to run
```

### Ask a question

```bash
python -m lexiquery.cli ask "what does RDS-4471 mean and how do I fix it"
```

```
Q: what does RDS-4471 mean and how do I fix it
------------------------------------------------------------------------------
Error `RDS-4471` means the connection pool is exhausted. [1]
------------------------------------------------------------------------------
sources:
  [1] database-runbook :: PostgreSQL Runbook > Common errors
confidence 0.6786 | refused False | 4.55 ms | provider extractive
```

### The guardrail in action

```bash
python -m lexiquery.cli ask "what is the capital of France"
```

```
I could not find this in the knowledge base. Try rephrasing, or check with the owning team -
answering without a supporting source would risk inventing policy.
confidence 0.1592 | refused True | 3.52 ms | provider extractive
```

### Measure it

```bash
python -m lexiquery.cli evaluate
```

```
Retrieval and answer quality on 22 labelled questions
------------------------------------------------------------------------------
  hit@1                        0.9545
  hit@3                        1.0
  recall@5                     1.0
  precision@1                  0.9545
  mrr                          0.9773
  ndcg@5                       0.9832
  citation_validity            1.0
  keyword_recall               0.9318
  answered                     22
  out_of_scope_refusal_rate    1.0
```

### Ablation — does each stage earn its place?

```bash
python -m lexiquery.cli ablation
```

| Strategy | hit@1 | hit@3 | recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| **Keyword-style questions (22)** | | | | | |
| bm25_only | 1.000 | 1.000 | 1.000 | **1.000** | 1.000 |
| vector_only | 0.909 | 0.955 | 1.000 | 0.943 | 0.957 |
| hybrid + rerank | 0.955 | 1.000 | 1.000 | 0.977 | 0.983 |
| **Paraphrased questions (6)** | | | | | |
| bm25_only | 0.167 | 0.667 | 0.667 | 0.472 | 0.604 |
| vector_only | 0.333 | 0.500 | 0.500 | 0.507 | 0.510 |
| hybrid + rerank | 0.167 | **0.833** | 0.833 | **0.514** | **0.637** |

### Inspect the ranking signals

```bash
python -m lexiquery.cli search "postgres failover"
```

```
chunk                           fused     bm25   vector    rerank
database-runbook#1             0.6912   11.284    0.831     0.712
database-runbook#0             0.4013    6.117    0.604     0.402
...
```

### Run the service

```bash
uvicorn lexiquery.api:app --factory --port 8000
# docs: http://127.0.0.1:8000/docs
```

```bash
KEY="change-me-lexiquery-api-key"

curl -s -XPOST localhost:8000/v1/ask -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"question":"How long are audit logs retained?"}'

curl -s -XPOST localhost:8000/v1/search -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"query":"rate limit headers","top_k":3}'

# add documents at runtime (the index re-fits)
curl -s -XPOST localhost:8000/v1/documents -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"documents":[{"id":"laptops","title":"IT","text":"# IT\n## Laptops\nLaptops are replaced every three years."}]}'

curl -s -XPOST "localhost:8000/v1/evaluate" -H "X-API-Key: $KEY"
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/ask` | Grounded answer with citations, sources, confidence and latency |
| `POST /v1/search` | Retrieved passages with every ranking signal exposed |
| `POST /v1/documents` · `GET /v1/documents` | Ingest / list corpus documents |
| `POST /v1/evaluate` | Run the labelled eval set and return the metric summary |
| `GET /health` | Public liveness + index statistics |

### Switching to hosted providers

Everything is one config change (keys live in `.env`, never in code):

```ini
LEXIQUERY_ANSWERER=anthropic          # Claude via the official SDK (claude-opus-5)
ANTHROPIC_API_KEY=                    # ==> your key here

LEXIQUERY_EMBEDDING_PROVIDER=openai   # hosted embeddings
LEXIQUERY_OPENAI_API_KEY=             # ==> your key here

LEXIQUERY_RERANKER=cross-encoder      # hosted reranker
LEXIQUERY_RERANK_API_KEY=             # ==> your key here
```

---

## Testing

```bash
python -m pytest -q          # 35 tests, ~3s
```

* **Text & chunking** — tokenisation/stemming/stop-words; heading hierarchy; chunks carry their
  heading, respect the token budget and overlap so a boundary answer stays retrievable.
* **BM25** — rare-term ranking, IDF ordering, and a test that term-frequency **saturates**
  (10× the term must not give 10× the score — that is the `k1` term doing its job).
* **Vectors & fusion** — exact cosine neighbours; RRF rewards documents both retrievers found,
  prefers consistent ranks, and is scale-free.
* **Reranker** — prefers tight term proximity over a long passage that merely mentions the terms;
  signals are individually asserted so the score stays explainable.
* **Metrics** — hit@k, recall@k, precision@k, MRR, nDCG and citation validity verified against
  hand-computed values.
* **Pipeline** — grounded, cited answers; exact-identifier lookup (the lexical path); paraphrase
  retrieval (the dense path); **out-of-scope refusal**; index save/reload; runtime ingestion.
* **Quality gates** — `test_retrieval_quality_meets_the_bar` fails CI if hit@1 < 0.80, hit@3 <
  0.90, MRR < 0.85, citation validity < 1.0, or any out-of-scope question is answered.
* **API** — auth, ask/search/list/validation behaviour.

---

## What this project demonstrates

* RAG as an engineering discipline rather than a prompt: chunking, hybrid retrieval, fusion,
  reranking, grounding and refusal are separate, testable stages.
* Information retrieval from first principles — BM25, RRF and the IR metric suite implemented and
  verified, not imported.
* Hallucination control that can be *measured*, and a guardrail signal (IDF-weighted coverage)
  designed specifically because the naive one failed on real out-of-scope questions.
* Honest evaluation, including publishing the ablation where the simplest baseline wins.
* Provider abstraction so an offline default and a hosted stack share one code path.

## Roadmap

- [ ] Approximate nearest neighbours (HNSW/FAISS) once the corpus outgrows an exact matrix
- [ ] Query rewriting and multi-hop retrieval for compound questions
- [ ] Sentence-transformer embeddings as a third local provider
- [ ] Per-tenant corpora with document-level ACLs enforced at retrieval time
- [ ] Answer-level LLM-as-judge scoring alongside the deterministic metrics

## License

MIT
