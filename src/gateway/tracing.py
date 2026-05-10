"""
Request-id propagation across the async/sync boundary.

ContextVars work inside asyncio but do not propagate automatically across the
ThreadPoolExecutor boundary. The executor wrapper copies the current context
into the worker thread; this module provides the variable and the helpers
needed for that.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str | None:
    return _request_id.get()
