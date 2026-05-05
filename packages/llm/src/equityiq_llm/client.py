from __future__ import annotations

import asyncio
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


class LLMError(RuntimeError):
    pass


class LLMClient:
    """OpenRouter-backed LLM client. Uses fastembed locally for embeddings."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or LLMSettings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._settings.openrouter_base_url,
            headers={"Authorization": f"Bearer {self._settings.openrouter_api_key}"},
            timeout=self._settings.openrouter_request_timeout_s,
        )
        self._embedder: Any = None

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=self._settings.openrouter_embed_model)
        return self._embedder

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
        resp = await self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        try:
            msg = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected response shape: {data!r}") from e
        if not isinstance(msg, str):
            raise LLMError(f"unexpected response shape: {data!r}")
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
        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as e:
                    raise LLMError(f"bad SSE chunk: {line!r}") from e
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    yield content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        embedder = self._get_embedder()
        embeddings = await loop.run_in_executor(None, lambda: list(embedder.embed(texts)))
        return [e.tolist() for e in embeddings]

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
        }
        if options:
            for key in ("temperature", "top_p", "max_tokens", "top_k"):
                if key in options:
                    body[key] = options[key]
        if format == "json":
            body["response_format"] = {"type": "json_object"}
        return body
