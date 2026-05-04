# Golden Q/A datasets

Each `.jsonl` row is a `GoldenItem` (see `equityiq_eval.types`):

```json
{
  "id": "unique-slug",
  "question": "natural-language question",
  "reference_answer": "ground-truth answer for relevance judging",
  "expected_accessions": ["0000320193-24-000123"],
  "expected_item_codes": ["1A"],
  "ticker": "AAPL",
  "tags": ["risk-factors"]
}
```

`expected_accessions` powers the deterministic `context_precision` metric. Leave empty
when the answer is not tied to a specific filing — precision will return 1.0.

`qa_2024.jsonl` is a starter set of 10 items spanning sectors. Expand to ~200 before
relying on the CI gate as a real regression signal.
