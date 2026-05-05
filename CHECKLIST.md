# EquityIQ - Tomorrow's Checklist (2026-05-05)

Repo: https://github.com/Chinh091/equityiq
Last green CI: commit `87bbe91`, run 25317847353 (27s)

Format: `[ ]` pending · `[x]` done · `[-]` deferred · `[!]` blocked

---

## Block 1 - Live stack bring-up (90 min)

- [ ] `make up` - start docker-compose stack (postgres+pgvector, ollama, TEI, redis, neo4j, langfuse)
- [ ] `docker compose ps` - verify 7 services healthy
- [ ] `make models` - pull Ollama models (~50GB total)
  - llama3.3:70b-instruct-q4_K_M (~40GB)
  - qwen2.5:32b-instruct (~20GB)
  - nomic-embed-text:v1.5 (~280MB)
- [ ] If RAM/disk insufficient, swap primary → `qwen2.5:14b-instruct` in `.env`, defer 70B
- [ ] `make api` - start FastAPI; hit `GET /health` returns `{"status":"ok"}`
- [ ] `uv run equityiq health` - confirm all three components green

## Block 2 - Real ingestion smoke (60 min)

- [ ] Edit `.env` → set `SEC_EDGAR_USER_AGENT=EquityIQ research-portfolio chinh@comansservices.com.au` (SEC requires real contact)
- [ ] `uv run equityiq ingest --ticker NVDA --limit 4` - pull recent NVDA 10-K + 10-Q
- [ ] Verify `chunks` table: `SELECT count(*) FROM chunks WHERE filing_id IN (SELECT id FROM filings WHERE ticker='NVDA');` - expect >500 rows
- [ ] Repeat for 4 more tickers: AAPL, MSFT, AMZN, TSLA
- [ ] `uv run equityiq query "data center capex" --ticker NVDA --k 5` - confirm hybrid retrieve returns NVDA chunks first
- [ ] `curl /thesis/stream` smoke (Makefile `make smoke`) - confirm SSE: plan → tool_call → tool_result → draft → critique → final

## Block 3 - Golden dataset expansion (2-3 hr)

Goal: replace 10 stub items in `packages/eval/golden/qa_2024.jsonl` with ~50 real-grounded items (200 is overkill for now).

- [ ] Write `packages/eval/scripts/seed_from_chunks.py`:
  - Iterate ingested filings, sample 5 chunks per filing per item_code
  - Use `ModelTier.PRIMARY` to generate Q/A pair from each chunk
  - Output draft `qa_2024_draft.jsonl` with real accession + item_code
- [ ] Run seed script on the 5 ingested tickers → ~50 draft items
- [ ] Manual review pass: open draft, fix Q phrasing, kill bad ones, target 30-40 keepers
- [ ] Move keepers to `qa_2024.jsonl`, commit
- [ ] Add a `tests/test_golden_dataset.py` schema test: every row parses as `GoldenItem`, every accession matches `\d{10}-\d{2}-\d{6}` regex

## Block 4 - Baseline eval + CI gate live (45 min)

- [ ] Set GH secrets via `gh secret set --repo Chinh091/equityiq`:
  - `OLLAMA_BASE_URL` - defer if no public Ollama yet (use ngrok or skip)
  - `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` - register at cloud.langfuse.com (free tier)
- [ ] OR: gate eval CI behind a label `run-eval` so it only runs when explicitly requested (avoids needing a public Ollama)
- [ ] Run `uv run python -m equityiq_eval.ci_gate --dataset packages/eval/golden/qa_2024.jsonl` locally → produces `eval-report.json`
- [ ] Commit `eval-report.json` on main (this becomes the baseline that `--baseline-from-main` reads)
- [ ] Open a noop PR; verify eval-gate workflow runs green and prints metric diff

## Block 5 - Phase 4 kickoff: Neo4j knowledge graph (90 min, optional if Block 3 over-runs)

- [ ] Create `packages/graph` skeleton (pyproject, src/equityiq_graph)
- [ ] `entities.py` - Pydantic types: Company, Person, Risk, Segment, Product
- [ ] `extractor.py` - LLM call per chunk: extract entities + relationships, returns `list[Edge]`
- [ ] `loader.py` - async neo4j writer with idempotent MERGE
- [ ] CLI add: `equityiq graph build --ticker NVDA` - runs extractor over chunks, loads to neo4j
- [ ] Read-only test: `MATCH (c:Company {ticker:'NVDA'})-[:HAS_RISK]->(r) RETURN r LIMIT 10`

## Block 6 - Housekeeping (30 min, parallel-able)

- [ ] Add `eval-report.json` to `.gitignore`? NO - it's the baseline, must be tracked
- [ ] Add `data/` to `.gitignore` (already done, verify)
- [ ] Replace `datetime.utcnow()` deprecation in `equityiq_eval/types.py` (12 warnings in pytest)
- [ ] Add `coverage` thresholds to `pyproject.toml` `[tool.coverage.report]` - fail-under=70
- [ ] Push branch protection on `main`: require CI + 1 review (manual GH UI step)

---

## Blocked / parking lot

- [!] Eval CI gate end-to-end on PR - needs public-reachable Ollama. Options: ngrok tunnel, RunPod endpoint, or run eval as nightly cron on self-hosted runner instead of per-PR
- [!] Resume update - user explicitly deferred ("I do not need resume right now"), revisit after Block 5
- [-] Expand golden dataset to 200 items - defer until Phase 4 graph data lands; bigger Q/A diversity comes free from graph relationships

## Definition of Done for tomorrow

Minimum: Blocks 1-2 complete, real chunks in DB, /thesis/stream works against real Ollama.
Stretch: Blocks 3-4 done, eval-report.json on main, CI gate fires on next PR.
Hero: Block 5 entity extractor lands first NVDA risk graph in Neo4j.

## Quick commands

```bash
make up && make models           # bring stack online
make api                          # FastAPI dev server
uv run equityiq health            # smoke all 3 components
uv run equityiq ingest -t NVDA -n 4
uv run equityiq query "supply risk" -t NVDA
make smoke                        # curl /thesis/stream

# After Block 3:
uv run python -m equityiq_eval.ci_gate \
  --dataset packages/eval/golden/qa_2024.jsonl \
  --report-path eval-report.json
git add eval-report.json && git commit -m "chore: seed eval baseline"
```
