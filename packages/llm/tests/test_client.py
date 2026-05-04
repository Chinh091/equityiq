import json

import httpx
import pytest
import respx

from equityiq_llm import LLMSettings, ModelTier, OllamaClient
from equityiq_llm.client import OllamaError


@pytest.fixture
def settings() -> LLMSettings:
    return LLMSettings(
        ollama_base_url="http://ollama.test",
        ollama_primary_model="primary-x",
        ollama_fallback_model="fallback-x",
        ollama_judge_model="judge-x",
        ollama_embed_model="embed-x",
        ollama_request_timeout_s=5.0,
    )


@pytest.mark.asyncio
@respx.mock
async def test_generate_returns_message_content(settings: LLMSettings) -> None:
    respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"content": "hello"}, "done": True},
        )
    )
    async with OllamaClient(settings) as c:
        out = await c.generate(prompt="hi", tier=ModelTier.FALLBACK)
    assert out == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_generate_uses_correct_model_for_tier(settings: LLMSettings) -> None:
    route = respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": "ok"}})
    )
    async with OllamaClient(settings) as c:
        await c.generate(prompt="hi", tier=ModelTier.PRIMARY)
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "primary-x"
    assert body["stream"] is False


@pytest.mark.asyncio
@respx.mock
async def test_stream_yields_chunks_until_done(settings: LLMSettings) -> None:
    chunks = [
        json.dumps({"message": {"content": "he"}}) + "\n",
        json.dumps({"message": {"content": "llo"}}) + "\n",
        json.dumps({"message": {"content": ""}, "done": True}) + "\n",
    ]
    respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(200, content="".join(chunks).encode())
    )
    async with OllamaClient(settings) as c:
        out = [tok async for tok in c.stream(prompt="hi")]
    assert "".join(out) == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_embed_returns_vectors(settings: LLMSettings) -> None:
    respx.post("http://ollama.test/api/embed").mock(
        return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})
    )
    async with OllamaClient(settings) as c:
        out = await c.embed(["hello"])
    assert out == [[0.1, 0.2]]


@pytest.mark.asyncio
@respx.mock
async def test_unexpected_shape_raises(settings: LLMSettings) -> None:
    respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(200, json={"unexpected": True})
    )
    async with OllamaClient(settings) as c, pytest.raises(OllamaError):
        await c.generate(prompt="hi")
