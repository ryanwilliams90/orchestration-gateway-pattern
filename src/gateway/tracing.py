"""Request-id propagation.

ContextVars work inside asyncio but do not propagate across the
ThreadPoolExecutor boundary; the executor copies the current context into
the worker thread. This module owns the variable, the accessors, and a
``logging.Filter`` that injects the current request id into every log
record so the request id appears in log lines without each call site
having to remember.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar, Token

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def set_request_id(value: str) -> Token[str | None]:
    """Set the request id and return a Token for ``reset_request_id``.

    The middleware overwrites this on every request, but library callers
    that want scoped propagation should hold the Token and pass it back to
    ``reset_request_id`` when the scope ends — without that, a long-lived
    task would inherit the parent's request id indefinitely.
    """
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    _request_id.reset(token)


def get_request_id() -> str | None:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Inject the current request id into every log record as ``request_id``.

    Use with a formatter that references ``%(request_id)s``. The filter
    sets ``"-"`` when no request id is in scope, so the format string never
    raises ``KeyError`` on missing attributes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True
