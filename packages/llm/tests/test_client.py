import json
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import respx
from equityiq_llm import LLMSettings, ModelTier, LLMClient
from equityiq_llm.client import LLMError

BASE = "https://openrouter.ai/api/v1"


@pytest.fixture
def settings() -> LLMSettings:
    return LLMSettings(
        openrouter_api_key="test-key",
        openrouter_base_url=BASE,
        openrouter_primary_model="primary-x",
        openrouter_fallback_model="fallback-x",
        openrouter_judge_model="judge-x",
        openrouter_embed_model="nomic-ai/nomic-embed-text-v1.5",
        openrouter_request_timeout_s=5.0,
    )


def _chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]},
    )


@pytest.mark.asyncio
@respx.mock
async def test_generate_returns_message_content(settings: LLMSettings) -> None:
    respx.post(f"{BASE}/chat/completions").mock(return_value=_chat_response("hello"))
    async with LLMClient(settings) as c:
        out = await c.generate(prompt="hi", tier=ModelTier.FALLBACK)
    assert out == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_generate_uses_correct_model_for_tier(settings: LLMSettings) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(return_value=_chat_response("ok"))
    async with LLMClient(settings) as c:
        await c.generate(prompt="hi", tier=ModelTier.PRIMARY)
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "primary-x"
    assert body["stream"] is False


@pytest.mark.asyncio
@respx.mock
async def test_stream_yields_chunks_until_done(settings: LLMSettings) -> None:
    sse_lines = (
        "data: " + json.dumps({"choices": [{"delta": {"content": "he"}}]}) + "\n"
        "data: " + json.dumps({"choices": [{"delta": {"content": "llo"}}]}) + "\n"
        "data: [DONE]\n"
    )
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, content=sse_lines.encode())
    )
    async with LLMClient(settings) as c:
        out = [tok async for tok in c.stream(prompt="hi")]
    assert "".join(out) == "hello"


@pytest.mark.asyncio
async def test_embed_returns_vectors(settings: LLMSettings) -> None:
    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [np.array([0.1, 0.2])]
    with patch.object(LLMClient, "_get_embedder", return_value=mock_embedder):
        async with LLMClient(settings) as c:
            out = await c.embed(["hello"])
    assert out == [[0.1, 0.2]]


@pytest.mark.asyncio
@respx.mock
async def test_unexpected_shape_raises(settings: LLMSettings) -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json={"unexpected": True})
    )
    async with LLMClient(settings) as c:
        with pytest.raises(LLMError):
            await c.generate(prompt="hi")
