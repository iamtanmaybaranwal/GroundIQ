# GroundIQ — Grounded Knowledge Search & RAG Engine

> **Ask questions about your company's documents — get answers backed by evidence, or get told "I don't know."**

LexiQuery is a **retrieval-augmented question answering (RAG) engine** for internal knowledge bases.

Think of it as **an AI search assistant for company documentation**.

Instead of giving an LLM access to a collection of documents and trusting it to answer, LexiQuery first searches the knowledge base, finds the most relevant information, checks whether the evidence is strong enough, and then produces an answer with citations.

If the information cannot be reliably found, **LexiQuery refuses to guess.**

### In simple terms

Imagine a company has thousands of documents containing:

- Engineering runbooks
- Security policies
- HR documentation
- Internal FAQs
- Deployment guides
- Database troubleshooting guides

An engineer asks:

> **"What does RDS-4471 mean and how do I fix it?"**

LexiQuery searches the company's knowledge base, finds the relevant runbook, and answers:

> **RDS-4471 means the PostgreSQL connection pool is exhausted.**

It also shows **where the answer came from**.

If someone instead asks:

> **"What is the capital of France?"**

and that information does not exist in the company's knowledge base, LexiQuery does not invent an answer.

It says:

> **"I don't know — I couldn't find supporting information in the knowledge base."**

This makes LexiQuery focused on **grounded answers rather than confident guesses.**

---

## Why LexiQuery?

Most simple RAG applications look like:

```text
Documents
    ↓
Embeddings
    ↓
Vector Database
    ↓
LLM
    ↓
Answer
````

That approach works for demos, but real knowledge systems have harder problems.

LexiQuery addresses them as separate, measurable engineering problems:

| Problem                                                | LexiQuery's approach                        |
| ------------------------------------------------------ | ------------------------------------------- |
| Exact identifiers are difficult for semantic search    | **BM25 lexical retrieval**                  |
| Keywords fail on paraphrased questions                 | **Dense retrieval**                         |
| Different retrieval methods produce different rankings | **Reciprocal Rank Fusion (RRF)**            |
| Some retrieved passages are only loosely relevant      | **Explainable reranking**                   |
| LLMs may answer without sufficient evidence            | **Grounding guardrail**                     |
| Answers need to be verifiable                          | **Citation-backed responses**               |
| Nobody knows whether a RAG system actually works       | **Automated evaluation + ablation testing** |
| Out-of-scope questions can cause hallucinations        | **Explicit refusal mechanism**              |

---

# Core Pipeline

LexiQuery treats RAG as an **engineering pipeline**, not simply a prompt.

```text
                    ┌──────────────────┐
                    │   Documents      │
                    │   Markdown/Docs  │
                    └────────┬─────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Structure-Aware     │
                  │ Chunking            │
                  └─────────┬───────────┘
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          ┌─────────────┐       ┌─────────────┐
          │    BM25     │       │    Dense    │
          │   Search    │       │  Retrieval  │
          └──────┬──────┘       └──────┬──────┘
                 │                     │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Reciprocal Rank     │
                 │ Fusion (RRF)        │
                 └─────────┬───────────┘
                           ▼
                 ┌─────────────────────┐
                 │ Explainable         │
                 │ Reranker             │
                 └─────────┬───────────┘
                           ▼
                 ┌─────────────────────┐
                 │ Grounding Guardrail │
                 └─────────┬───────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Sufficient Evidence    Weak Evidence
                 │                   │
                 ▼                   ▼
          Answer + Citations     "I don't know"
```

---

# Key Features

## 🔎 Hybrid Retrieval

LexiQuery combines two fundamentally different search strategies.

### BM25

BM25 handles **exact words, identifiers, and technical terminology**.

For example:

```text
RDS-4471
PostgreSQL
connection pool
Kubernetes
```

This is particularly useful when the user asks about a specific error code, configuration key, service name, or technical identifier.

### Dense Retrieval

Dense retrieval handles **meaning and paraphrases**.

For example:

```text
"How do I undo a bad release?"
```

can retrieve a document containing:

```text
"Production deployment rollback procedure"
```

even though the exact words are different.

---

# Reciprocal Rank Fusion

BM25 and dense retrieval produce different rankings.

LexiQuery combines those rankings using **Reciprocal Rank Fusion (RRF)**.

```text
BM25 results
      +
Dense results
      ↓
    RRF
      ↓
Combined ranking
```

RRF works with rankings rather than raw scores, avoiding fragile score normalization between retrieval systems.

The implementation uses:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

with `k = 60`.

---

# Explainable Reranking

After retrieval and fusion, LexiQuery reranks candidate passages using interpretable signals such as:

* Query term coverage
* IDF-weighted coverage
* Term proximity
* Phrase matching
* Heading relevance
* Content density

Instead of producing an unexplained black-box score, the system exposes the ranking signals used for each candidate.

This makes retrieval behavior easier to inspect and debug.

---

# Grounding Guardrail

This is one of the most important parts of LexiQuery.

Before generating an answer, LexiQuery checks whether the retrieved evidence provides enough support for the question.

```text
Question
   ↓
Retrieved evidence
   ↓
Grounding score
   ↓
 ┌───────────────┬────────────────┐
 │ Strong enough │ Too weak       │
 ▼               ▼
Answer          Refuse
 + citations    to answer
```

If the evidence falls below the configured threshold:

```text
I could not find this in the knowledge base.
Answering without a supporting source would risk inventing policy.
```

The goal is simple:

> **A missing answer is better than a fabricated internal policy.**

On the included out-of-scope evaluation set, LexiQuery achieved a **100% refusal rate**.

---

# Citation-Backed Answers

Every generated answer can be traced back to the retrieved document passages.

Example:

```text
Q:
What does RDS-4471 mean?

A:
RDS-4471 means the PostgreSQL connection pool is exhausted.

Source:
database-runbook
→ PostgreSQL Runbook
→ Common errors

Confidence: 0.6786
```

This allows users to verify the answer instead of blindly trusting the model.

---

# Evaluation

LexiQuery includes a dedicated **evaluation harness** instead of simply claiming that the RAG system is accurate.

The evaluation set contains:

* 22 keyword-style questions
* 6 paraphrased questions
* 4 out-of-scope questions

It measures:

* Hit@K
* Recall@K
* Precision@K
* Mean Reciprocal Rank (MRR)
* nDCG
* Citation validity
* Out-of-scope refusal rate

### Current results

| Metric               |     Result |
| -------------------- | ---------: |
| Hit@1                | **95.45%** |
| Hit@3                |   **100%** |
| Recall@5             |   **100%** |
| MRR                  | **97.73%** |
| nDCG@5               | **98.32%** |
| Citation validity    |   **100%** |
| Out-of-scope refusal |   **100%** |

These results are from the included labelled evaluation set.

---

# Retrieval Ablation

LexiQuery also measures whether each component actually improves the system.

### Keyword-style questions

| Strategy        |     Hit@3 |       MRR |
| --------------- | --------: | --------: |
| BM25 only       | **1.000** | **1.000** |
| Vector only     |     0.955 |     0.943 |
| Hybrid + rerank | **1.000** |     0.977 |

### Paraphrased questions

| Strategy        |     Hit@3 |       MRR |
| --------------- | --------: | --------: |
| BM25 only       |     0.667 |     0.472 |
| Vector only     |     0.500 |     0.507 |
| Hybrid + rerank | **0.833** | **0.514** |

An important result is that **BM25 actually performs better on the keyword-style dataset**.

LexiQuery does not hide this result.

Instead, the paraphrase evaluation demonstrates the specific situation where hybrid retrieval provides value.

This makes the evaluation an actual engineering experiment rather than a collection of flattering benchmark numbers.

---

# Architecture

```text
                         ┌─────────────────┐
                         │     Corpus      │
                         │    Markdown     │
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌────────────────────────┐
                    │ Structure-Aware        │
                    │ Chunking               │
                    │                        │
                    │ Sections → paragraphs  │
                    │ → sentences + overlap │
                    └───────────┬────────────┘
                                │
                  ┌─────────────┴──────────────┐
                  │                            │
                  ▼                            ▼
          ┌──────────────┐              ┌──────────────┐
          │     BM25     │              │  Embeddings  │
          │              │              │              │
          │ Implemented  │              │ TF-IDF + SVD │
          │ from scratch │              │ or hosted    │
          └──────┬───────┘              └──────┬───────┘
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Reciprocal Rank Fusion │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Explainable Reranker   │
                    │                        │
                    │ coverage               │
                    │ proximity              │
                    │ phrase                 │
                    │ heading                │
                    │ density                │
                    └───────────┬────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Grounding Guardrail    │
                    └───────────┬────────────┘
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
              ┌─────────────┐       ┌────────────┐
              │   Answer    │       │  Refusal   │
              │ + citations │       │ "I don't  │
              │ + confidence│       │   know"   │
              └─────────────┘       └────────────┘
```

---

# Design Decisions

## Why Hybrid Retrieval?

BM25 is strong for exact technical terms and identifiers.

Dense retrieval is stronger for semantic similarity and paraphrases.

Using both makes the system robust to different types of questions.

---

## Why RRF?

BM25 and vector similarity produce scores on completely different scales.

Rather than trying to normalize those scores, LexiQuery combines their **rank positions** using Reciprocal Rank Fusion.

---

## Why Structure-Aware Chunking?

Naively splitting documents can separate:

```text
Heading
+
Important explanation
```

from each other.

LexiQuery preserves the document's heading hierarchy and uses sentence overlap so that retrieved chunks retain enough context to be useful and citeable.

---

## Why Refuse?

In an internal knowledge system, an incorrect answer can be worse than no answer.

For example:

```text
"What is our production data retention policy?"
```

Inventing a policy could cause a real operational or compliance problem.

LexiQuery therefore treats **insufficient evidence as a reason not to answer**.

---

## Why an Extractive Answerer?

The default answerer is extractive and deterministic.

This makes the system:

* Offline
* Reproducible
* Fast
* Easy to evaluate

It also allows the evaluation harness to measure retrieval quality without introducing additional variability from an LLM.

Hosted LLM providers can be enabled when desired.

---

# Tech Stack

| Layer             | Technology                                  |
| ----------------- | ------------------------------------------- |
| Language          | Python 3.11+                                |
| API               | FastAPI + Uvicorn                           |
| Lexical Retrieval | BM25 implemented from scratch               |
| Dense Retrieval   | TF-IDF + TruncatedSVD (LSA)                 |
| Vector Search     | NumPy cosine similarity                     |
| Rank Fusion       | Reciprocal Rank Fusion                      |
| Reranking         | Explainable feature-based reranker          |
| Generation        | Extractive / Anthropic / OpenAI-compatible  |
| Validation        | Pydantic                                    |
| Persistence       | JSON + NumPy + Joblib                       |
| Evaluation        | Hit@K, Recall, MRR, nDCG, citation validity |
| Testing           | pytest                                      |
| Authentication    | API-key authentication                      |

---

# Project Structure

```text
lexiquery/
├── lexiquery/
│   ├── text.py
│   ├── chunking.py
│   │
│   ├── index/
│   │   ├── bm25.py
│   │   ├── embeddings.py
│   │   └── vector.py
│   │
│   ├── retrieval/
│   │   ├── hybrid.py
│   │   └── rerank.py
│   │
│   ├── generation/
│   │   └── answerers.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── harness.py
│   │
│   ├── pipeline.py
│   ├── api.py
│   └── cli.py
│
├── corpus/
│   └── engineering knowledge base
│
├── eval/
│   └── questions.json
│
├── tests/
│   └── 35 tests
│
└── requirements.txt
```

---

# Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/iamtanmaybaranwal/LexiQuery.git
cd LexiQuery
```

## 2. Create a virtual environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

No API key is required for the default offline configuration.

---

# Ask a Question

```bash
python -m lexiquery.cli ask \
  "what does RDS-4471 mean and how do I fix it"
```

Example:

```text
Q: what does RDS-4471 mean and how do I fix it

Error RDS-4471 means the connection pool is exhausted.

sources:
  [1] database-runbook
      PostgreSQL Runbook > Common errors

confidence 0.6786
refused False
provider extractive
```

---

# Test the Refusal Guardrail

```bash
python -m lexiquery.cli ask \
  "what is the capital of France"
```

Example:

```text
I could not find this in the knowledge base.

Try rephrasing, or check with the owning team -
answering without a supporting source would risk inventing policy.

confidence 0.1592
refused True
```

---

# Run the Evaluation

```bash
python -m lexiquery.cli evaluate
```

---

# Run the Ablation Study

```bash
python -m lexiquery.cli ablation
```

This compares:

```text
BM25 only
Vector only
Hybrid + reranking
```

and reports retrieval metrics for each strategy.

---

# Run the API

```bash
uvicorn lexiquery.api:app --factory --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# API

| Method | Endpoint        | Purpose                                       |
| ------ | --------------- | --------------------------------------------- |
| POST   | `/v1/ask`       | Grounded answer with citations and confidence |
| POST   | `/v1/search`    | Search and expose ranking signals             |
| POST   | `/v1/documents` | Add documents to the corpus                   |
| GET    | `/v1/documents` | List documents                                |
| POST   | `/v1/evaluate`  | Run the evaluation set                        |
| GET    | `/health`       | Health and index statistics                   |

---

# Example API Request

```bash
curl -X POST localhost:8000/v1/ask \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How long are audit logs retained?"
  }'
```

A response contains:

```text
answer
citations
sources
confidence
latency
refused
```

---

# Hosted Providers

The architecture uses provider interfaces, so hosted services can replace the local components without changing the overall pipeline.

Supported options include:

* Anthropic for answer generation
* OpenAI-compatible embedding providers
* Hosted cross-encoder reranking

The offline configuration remains the default so the project can be reproduced without API keys.

---

# Testing

Run the complete test suite:

```bash
python -m pytest -q
```

LexiQuery currently contains **35 tests** covering:

* Text processing
* Structure-aware chunking
* BM25 ranking
* Term-frequency saturation
* Vector search
* RRF fusion
* Reranking
* Retrieval metrics
* Citation validity
* Grounding
* Out-of-scope refusal
* Runtime document ingestion
* Index persistence
* API authentication
* API behavior

The project also contains **quality gates** that fail CI when retrieval quality falls below the configured thresholds.

---

# What This Project Demonstrates

LexiQuery was designed to demonstrate that building a useful RAG system involves much more than connecting an LLM to a vector database.

### Information Retrieval

* BM25 from first principles
* Dense retrieval
* Reciprocal Rank Fusion
* Reranking
* IR evaluation metrics

### AI Engineering

* RAG architecture
* Grounded generation
* Hallucination mitigation
* Provider abstraction
* Citation-backed answers

### Backend Engineering

* FastAPI service
* API authentication
* Runtime document ingestion
* Persistence
* Configurable providers

### Software Engineering

* Modular architecture
* Deterministic testing
* Automated quality gates
* Ablation experiments
* Reproducible evaluation

---

# Honest Evaluation

One of the goals of LexiQuery is to make RAG evaluation transparent.

The included experiments show that:

* BM25 is extremely strong for keyword-heavy questions.
* Dense retrieval helps with semantic variation.
* Hybrid retrieval improves paraphrased-question retrieval.
* Reranking provides an additional explainable ranking stage.
* A grounding signal can be used to reject unsupported questions.
* Evaluation should expose weaknesses instead of reporting only the best-looking metric.

The system is therefore designed around a simple principle:

> **Measure every important component instead of assuming that a more complicated pipeline is automatically better.**

---

# Roadmap

* [ ] Approximate nearest-neighbor search using HNSW / FAISS
* [ ] Query rewriting
* [ ] Multi-hop retrieval
* [ ] Sentence-transformer embeddings
* [ ] Per-tenant document collections
* [ ] Document-level access control
* [ ] Answer-level LLM-as-judge evaluation
* [ ] Larger-scale retrieval benchmarks

---

# License

MIT
