-- Bootstrap schema for EquityIQ.
-- Idempotent. Runs on first container start (volume empty).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Filings: one row per accepted SEC filing.
CREATE TABLE IF NOT EXISTS filings (
    id              BIGSERIAL PRIMARY KEY,
    accession       TEXT NOT NULL UNIQUE,
    cik             TEXT NOT NULL,
    ticker          TEXT,
    form_type       TEXT NOT NULL,
    filed_at        TIMESTAMPTZ NOT NULL,
    period_of_report DATE,
    source_url      TEXT NOT NULL,
    raw_html_path   TEXT,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_filings_ticker_form ON filings (ticker, form_type, filed_at DESC);
CREATE INDEX IF NOT EXISTS idx_filings_cik         ON filings (cik, filed_at DESC);

-- Sections: parsed sections within a filing (Item 1A, Item 7, etc.).
CREATE TABLE IF NOT EXISTS filing_sections (
    id          BIGSERIAL PRIMARY KEY,
    filing_id   BIGINT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    item_code   TEXT NOT NULL,
    title       TEXT,
    text        TEXT NOT NULL,
    char_start  INT,
    char_end    INT
);

CREATE INDEX IF NOT EXISTS idx_filing_sections_filing ON filing_sections (filing_id, item_code);

-- Chunks: retrieval-time units, semantic-split.
-- Embedding dim 768 = nomic-embed-text-v1.5. Adjust if you change model.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    section_id  BIGINT NOT NULL REFERENCES filing_sections(id) ON DELETE CASCADE,
    filing_id   BIGINT NOT NULL REFERENCES filings(id) ON DELETE CASCADE,
    ord         INT NOT NULL,
    text        TEXT NOT NULL,
    tokens      INT,
    embedding   vector(768)
);

CREATE INDEX IF NOT EXISTS idx_chunks_filing  ON chunks (filing_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks (section_id);
CREATE INDEX IF NOT EXISTS idx_chunks_text_fts
    ON chunks USING gin (to_tsvector('english', text));
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm
    ON chunks USING gin (text gin_trgm_ops);
-- HNSW for ANN. Built lazily; k tuned per workload.
CREATE INDEX IF NOT EXISTS idx_chunks_embed_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Earnings calls (transcripts; speaker turns).
CREATE TABLE IF NOT EXISTS earnings_calls (
    id              BIGSERIAL PRIMARY KEY,
    ticker          TEXT NOT NULL,
    fiscal_period   TEXT NOT NULL,
    held_at         TIMESTAMPTZ NOT NULL,
    audio_url       TEXT,
    transcript_path TEXT,
    UNIQUE (ticker, fiscal_period)
);

CREATE TABLE IF NOT EXISTS call_turns (
    id          BIGSERIAL PRIMARY KEY,
    call_id     BIGINT NOT NULL REFERENCES earnings_calls(id) ON DELETE CASCADE,
    speaker     TEXT,
    role        TEXT,  -- 'management' | 'analyst' | 'operator'
    ord         INT NOT NULL,
    start_ms    INT,
    end_ms      INT,
    text        TEXT NOT NULL,
    embedding   vector(768)
);

CREATE INDEX IF NOT EXISTS idx_call_turns_call ON call_turns (call_id, ord);
CREATE INDEX IF NOT EXISTS idx_call_turns_embed_hnsw
    ON call_turns USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Eval golden set lives in repo as JSONL but mirrored here for run records.
CREATE TABLE IF NOT EXISTS eval_runs (
    id              BIGSERIAL PRIMARY KEY,
    git_sha         TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    metrics_json    JSONB
);
