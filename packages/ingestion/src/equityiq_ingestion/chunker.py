"""Sentence-aware, token-bounded chunker.

Why not just use LlamaIndex's SemanticSplitterNodeParser?
    - LlamaIndex's parser requires an embedding callback per split; it's slow
      and pulls a heavy dep tree.
    - For 10-K body text, paragraph + sentence boundaries are nearly always
      correct semantic breakpoints. A token-aware sliding-window chunker with
      sentence boundary preference wins on both speed and quality.

Properties:
    - Targets `target_tokens` per chunk (default 320).
    - Overlap of `overlap_tokens` (default 48) between adjacent chunks → improves
      retrieval recall on facts that span boundaries.
    - Hard cap at `max_tokens` to avoid huge chunks even on a long sentence.
    - Token counts use tiktoken cl100k_base (close enough for non-OpenAI models).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z(\"'\[])"  # split on terminal punct followed by Cap
    r"|\n{2,}"  # or blank line
)


@dataclass(slots=True)
class Chunk:
    ord: int
    text: str
    tokens: int


class SemanticChunker:
    def __init__(
        self,
        *,
        target_tokens: int = 320,
        overlap_tokens: int = 48,
        max_tokens: int = 480,
        encoding: str = "cl100k_base",
    ) -> None:
        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens must be < target_tokens")
        if max_tokens < target_tokens:
            raise ValueError("max_tokens must be >= target_tokens")
        self._target = target_tokens
        self._overlap = overlap_tokens
        self._max = max_tokens
        self._enc = tiktoken.get_encoding(encoding)

    def _tok_count(self, s: str) -> int:
        return len(self._enc.encode(s, disallowed_special=()))

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sents = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
        return sents

    def chunk(self, text: str) -> list[Chunk]:
        if not text.strip():
            return []
        sents = self._split_sentences(text)
        out: list[Chunk] = []
        buf: list[str] = []
        buf_tokens = 0
        ord_ = 0

        def flush() -> None:
            nonlocal ord_
            if not buf:
                return
            joined = " ".join(buf).strip()
            if joined:
                out.append(Chunk(ord=ord_, text=joined, tokens=self._tok_count(joined)))
                ord_ += 1

        for sent in sents:
            t = self._tok_count(sent)
            # Sentence too big alone → emit what we have and split it as one chunk
            if t > self._max:
                flush()
                buf = [sent]
                flush()
                buf = []
                buf_tokens = 0
                continue

            if buf_tokens + t <= self._target:
                buf.append(sent)
                buf_tokens += t
                continue

            # Over target — emit and start new buffer with overlap tail.
            flush()
            tail = self._tail_for_overlap(buf)
            buf = [*tail, sent]
            buf_tokens = sum(self._tok_count(s) for s in buf)
            # Hard cap safety
            while buf_tokens > self._max and len(buf) > 1:
                buf.pop(0)
                buf_tokens = sum(self._tok_count(s) for s in buf)

        flush()
        return out

    def _tail_for_overlap(self, buf: list[str]) -> list[str]:
        if self._overlap <= 0 or not buf:
            return []
        tail: list[str] = []
        toks = 0
        for s in reversed(buf):
            t = self._tok_count(s)
            if toks + t > self._overlap:
                break
            tail.insert(0, s)
            toks += t
        return tail
