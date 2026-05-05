# EquityIQ

AI-powered equity research assistant that reads SEC filings and answers analyst-style questions in plain English. Ask "what are NVDA's data center risk factors?" and it retrieves the relevant 10-K passages, reasons across them, and streams a grounded thesis with citations.

Designed, built, and maintained solo as a portfolio project demonstrating production-shape AI engineering.

## Built solo

- Multi-agent pipeline: planner, retriever, analyst, and critic agents working in sequence
- Hybrid search: vector similarity + full-text search + cross-encoder reranking, fused with RRF
- SEC EDGAR ingestion: fetches, parses, and chunks real filings into a queryable knowledge base
- LLM-as-judge CI gate: every PR is regression-tested against a golden Q/A set; a metric drop >3% fails the build
- Zero local GPU required: LLM runs via OpenRouter free tier, embeddings via fastembed ONNX runtime
- 7-service Docker Compose stack: Postgres + pgvector, Neo4j, Redis, TEI reranker, Langfuse, FastAPI

## Skills demonstrated

`Python 3.12` `FastAPI` `async/await` `RAG pipelines` `pgvector` `hybrid search` `multi-agent systems` `LLM evaluation` `SEC EDGAR` `Docker Compose` `uv monorepo` `CI/CD` `Mypy strict` `Pytest`

## Stack

| Layer | Choice |
|-------|--------|
| LLM serving | OpenRouter (GPT-OSS 120B free tier, zero local GPU required) |
| Embeddings | nomic-embed-text v1.5 via fastembed (ONNX, runs locally, no GPU) |
| Reranker | TEI + BAAI/bge-reranker-v2-m3 |
| Storage | Postgres 16 + pgvector + HNSW + GIN |
| Graph | Neo4j (entity + filing relationships) |
| Tracing | Langfuse self-hosted |
| Cache | Redis |
| API | FastAPI + sse-starlette |
| CLI | Typer + Rich |
| Build | uv workspace, Hatchling, Ruff, Mypy, Pytest |

## Workspace layout

```
equityiq/
├── apps/
│   ├── api/             FastAPI gateway (/health, /retrieve, /thesis/stream)
│   └── cli/             `equityiq` CLI: ingest, query, health
├── packages/
│   ├── llm/             OpenRouter LLM client + ModelRouter
│   ├── observability/   structlog + Langfuse decorators
│   ├── ingestion/       EDGAR client, SEC parsers, semantic chunker, pipeline
│   ├── retrieval/       Hybrid retriever, RRF fusion, TEI reranker
│   ├── agents/          Planner / analyst / critic loop, tool registry
│   └── eval/            Golden-set runner, LLM-as-judge, CI regression gate
├── infra/postgres/      pgvector schema bootstrap
├── docker-compose.yml   Full 7-service local stack
└── .github/workflows/   ci.yml (lint+type+unit), eval.yml (regression gate)
```

## Quick start

```bash
make install            # uv sync --all-packages
cp .env.example .env    # add OPENROUTER_API_KEY (free at openrouter.ai)
make up                 # docker compose up -d
make api                # uvicorn dev server
```

Run a thesis query:

```bash
curl -N -X POST http://localhost:8000/thesis/stream \
  -H 'content-type: application/json' \
  -d '{"ticker":"NVDA","question":"summarize recent data center risk factors"}'
```

CLI:

```bash
uv run equityiq ingest --ticker AAPL --limit 4
uv run equityiq query "supply chain concentration risk" --ticker AAPL
uv run equityiq health
```

## Eval CI gate

`packages/eval/golden/qa_2024.jsonl` holds the golden Q/A set. On each PR touching `agents`, `retrieval`, `eval`, `llm`, or `api`, GitHub Actions runs:

```bash
uv run python -m equityiq_eval.ci_gate \
  --dataset packages/eval/golden/qa_2024.jsonl \
  --baseline-from-main \
  --max-regression 0.03
```

Metrics: `faithfulness` (LLM-as-judge), `answer_relevance` (LLM-as-judge), `context_precision` (deterministic set math). Any metric regressing >3% vs `origin/main`'s last green run fails the build.

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Repo scaffold + CI + Docker stack | done |
| 2 | Ingestion + retrieval + CLI + `/retrieve` | done |
| 3 | Agent loop + eval harness + `/thesis/stream` + CI gate | done |
| 4 | Knowledge graph (Neo4j) + entity linking | planned |
| 5 | Quality-aware learned router | planned |

## Development

```bash
make fmt            # ruff format + autofix
make lint           # ruff check
make typecheck      # mypy strict
make test           # pytest (excludes integration)
make test-int       # integration tests (needs `make up`)
```

## License

MIT.
