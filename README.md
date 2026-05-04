# EquityIQ

Multi-agent equity research platform on local Ollama. Ingests SEC EDGAR filings, runs hybrid retrieval (pgvector + Postgres FTS + cross-encoder rerank), and orchestrates a planner→retriever→analyst→critic agent loop with LLM-as-judge evaluation gating CI.

Built as a portfolio project demonstrating production-shape AI engineering: uv workspace monorepo, async Python 3.12, FastAPI streaming, golden-set regression testing, Langfuse tracing, Docker Compose stack.

## Stack

| Layer | Choice |
|------|--------|
| LLM serving | Ollama (Llama 3.3 70B primary, Qwen 2.5 32B fallback/judge) |
| Embeddings | nomic-embed-text v1.5 (768d) |
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
│   ├── llm/             Ollama async client + ModelRouter
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
cp .env.example .env    # then fill secrets if any
make up                 # docker compose up -d
make models             # pull Ollama models
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

Metrics — `faithfulness` (LLM-as-judge), `answer_relevance` (LLM-as-judge), `context_precision` (deterministic set math). Any metric regressing >3% vs `origin/main`'s last green report fails the build.

## Status

| Phase | Scope | Status |
|------|-------|--------|
| 0 | Repo scaffold + CI + docker stack | done |
| 2 | Ingestion + retrieval + CLI + /retrieve | done |
| 3 | Agent loop + eval harness + /thesis/stream + CI gate | done |
| 4 | Knowledge graph (Neo4j) + entity linking | planned |
| 5 | Quality-aware learned router | planned |

## Development

```bash
make fmt            # ruff format + autofix
make lint           # ruff check
make typecheck      # mypy strict
make test           # pytest (excludes integration)
make test-int      # integration tests (needs `make up`)
```

## License

MIT.
