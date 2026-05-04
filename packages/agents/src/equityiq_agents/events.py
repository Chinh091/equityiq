from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel


class PlanEvent(BaseModel):
    type: Literal["plan"] = "plan"
    subqueries: list[str]


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    name: str
    args: dict[str, object]


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    name: str
    summary: str
    n_results: int


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    role: str
    text: str


class DraftEvent(BaseModel):
    type: Literal["draft"] = "draft"
    text: str


class CritiqueEvent(BaseModel):
    type: Literal["critique"] = "critique"
    faithfulness: float
    notes: str
    accepted: bool


class FinalAnswerEvent(BaseModel):
    type: Literal["final"] = "final"
    text: str
    citations: list[str]


AgentEvent = (
    PlanEvent
    | ToolCallEvent
    | ToolResultEvent
    | TokenEvent
    | DraftEvent
    | CritiqueEvent
    | FinalAnswerEvent
)


def serialize_event(event: AgentEvent) -> str:
    """SSE-friendly JSON serialization."""
    return json.dumps(event.model_dump(), ensure_ascii=False)
