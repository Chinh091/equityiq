from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast

from pydantic_settings import BaseSettings, SettingsConfigDict

P = ParamSpec("P")
R = TypeVar("R")


class LangfuseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    langfuse_host: str = "http://localhost:3001"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = True


_client: Any = None


def get_langfuse() -> Any:
    """Return a process-wide Langfuse client, or None if disabled / unconfigured.

    Lazy-imported so unit tests don't need the package installed.
    """
    global _client
    if _client is not None:
        return _client
    settings = LangfuseSettings()
    if not (settings.langfuse_enabled and settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        return None
    _client = Langfuse(
        host=settings.langfuse_host,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )
    return _client


def shutdown_langfuse() -> None:
    global _client
    if _client is None:
        return
    try:
        _client.flush()
    finally:
        _client = None


def observed(name: str | None = None) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator that wires a coroutine into Langfuse tracing if available.

    Falls back to a no-op when Langfuse isn't configured, so tests don't need it.
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        try:
            from langfuse.decorators import observe  # type: ignore[import-untyped]
        except ImportError:
            return fn

        wrapped = observe(name=name or fn.__name__)(fn)
        return cast(Callable[P, Awaitable[R]], functools.wraps(fn)(wrapped))

    return decorator
