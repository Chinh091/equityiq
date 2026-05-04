from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from equityiq_llm.config import LLMSettings, ModelTier


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    """Async Ollama HTTP client. Streaming + non-streaming + embeddings."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or LLMSettings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=self._settings.ollama_request_timeout_s,
        )

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def generate(
        self,
        *,
        prompt: str,
        tier: ModelTier = ModelTier.PRIMARY,
        system: str | None = None,
        options: Mapping[str, Any] | None = None,
        format: str | Mapping[str, Any] | None = None,
    ) -> str:
        body = self._chat_body(prompt, tier, system, options, format, stream=False)
        resp = await self._client.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {}).get("content")
        if not isinstance(msg, str):
            raise OllamaError(f"unexpected response shape: {data!r}")
        return msg

    async def stream(
        self,
        *,
        prompt: str,
        tier: ModelTier = ModelTier.PRIMARY,
        system: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        body = self._chat_body(prompt, tier, system, options, format=None, stream=True)
        async with self._client.stream("POST", "/api/chat", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as e:
                    raise OllamaError(f"bad ndjson chunk: {line!r}") from e
                content = chunk.get("message", {}).get("content")
                if content:
                    yield content
                if chunk.get("done"):
                    return

    async def embed(self, texts: list[str]) -> list[list[float]]:
        body = {
            "model": self._settings.model_for(ModelTier.EMBED),
            "input": texts,
        }
        resp = await self._client.post("/api/embed", json=body)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list):
            raise OllamaError(f"unexpected embed response: {data!r}")
        return embeddings

    def _chat_body(
        self,
        prompt: str,
        tier: ModelTier,
        system: str | None,
        options: Mapping[str, Any] | None,
        format: str | Mapping[str, Any] | None,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self._settings.model_for(tier),
            "messages": messages,
            "stream": stream,
            "keep_alive": self._settings.ollama_keep_alive,
        }
        if options:
            body["options"] = dict(options)
        if format is not None:
            body["format"] = format
        return body
