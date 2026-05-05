from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from equityiq_agents import AgentLoop, AgentSettings, RetrieveTool
from equityiq_llm import LLMClient, LLMSettings, ModelRouter
from equityiq_retrieval import HybridRetriever, RetrievalSettings, TEIReranker
from fastapi import FastAPI, Request

LLM_KEY = "llm_client"
ROUTER_KEY = "model_router"
RETRIEVER_KEY = "retriever"
RERANKER_KEY = "reranker"
AGENT_LOOP_KEY = "agent_loop"


@asynccontextmanager
async def lifespan_state(app: FastAPI) -> AsyncIterator[None]:
    llm_settings = LLMSettings()
    retr_settings = RetrievalSettings()

    ollama = LLMClient(llm_settings)
    reranker = TEIReranker(retr_settings)
    retriever = HybridRetriever(llm=ollama, reranker=reranker, settings=retr_settings)
    agent_loop = AgentLoop(
        llm=ollama,
        retrieve=RetrieveTool(retriever),
        settings=AgentSettings(),
    )

    app.state.__dict__[LLM_KEY] = ollama
    app.state.__dict__[ROUTER_KEY] = ModelRouter()
    app.state.__dict__[RERANKER_KEY] = reranker
    app.state.__dict__[RETRIEVER_KEY] = retriever
    app.state.__dict__[AGENT_LOOP_KEY] = agent_loop

    try:
        yield
    finally:
        await retriever.aclose()
        await reranker.aclose()
        await ollama.aclose()


def get_ollama(request: Request) -> LLMClient:
    return request.app.state.__dict__[LLM_KEY]  # type: ignore[no-any-return]


def get_router(request: Request) -> ModelRouter:
    return request.app.state.__dict__[ROUTER_KEY]  # type: ignore[no-any-return]


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.__dict__[RETRIEVER_KEY]  # type: ignore[no-any-return]


def get_agent_loop(request: Request) -> AgentLoop:
    return request.app.state.__dict__[AGENT_LOOP_KEY]  # type: ignore[no-any-return]
