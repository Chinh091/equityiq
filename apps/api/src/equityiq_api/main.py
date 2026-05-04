from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sse_starlette.sse import EventSourceResponse

from equityiq_agents import AgentLoop
from equityiq_agents.events import serialize_event
from equityiq_api.deps import get_agent_loop, get_retriever, lifespan_state
from equityiq_api.schemas import (
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    ThesisRequest,
)
from equityiq_observability import (
    configure_logging,
    get_logger,
    observed,
    shutdown_langfuse,
)
from equityiq_retrieval import HybridRetriever, RetrievalQuery

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    log.info("api.startup")
    async with lifespan_state(app):
        yield
    shutdown_langfuse()
    log.info("api.shutdown")


app = FastAPI(
    title="EquityIQ API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/retrieve", response_model=RetrieveResponse)
@observed("api.retrieve")
async def retrieve(
    req: RetrieveRequest,
    retriever: HybridRetriever = Depends(get_retriever),
) -> RetrieveResponse:
    """Hybrid retrieval (pgvector + FTS + RRF) with optional cross-encoder rerank."""
    q = RetrievalQuery(
        query=req.query,
        ticker=req.ticker,
        form_types=req.form_types,
        item_codes=req.item_codes,
        top_k=req.top_k,
        pool=req.pool,
        use_reranker=req.use_reranker,
    )
    results = await retriever.search(q)
    return RetrieveResponse(
        results=results,
        count=len(results),
        used_reranker=req.use_reranker,
    )


@app.post("/thesis/stream")
async def thesis_stream(
    req: ThesisRequest,
    loop: AgentLoop = Depends(get_agent_loop),
) -> EventSourceResponse:
    """Multi-agent thesis pipeline: planner → retriever → analyst → critic.

    Streams plan, tool_call, tool_result, draft, critique, and final events as SSE.
    """
    log.info("thesis.start", ticker=req.ticker, q_chars=len(req.question))
    return EventSourceResponse(_stream_agent(loop, req), ping=15)


@observed("thesis.agent_stream")
async def _stream_agent(
    loop: AgentLoop, req: ThesisRequest
) -> AsyncIterator[dict[str, str]]:
    async for evt in loop.run(question=req.question, ticker=req.ticker):
        yield {"event": evt.type, "data": serialize_event(evt)}
    yield {"event": "done", "data": "1"}
