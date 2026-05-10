"""Orchestration gateway pattern — async/sync executor boundary.

The package is organized around the boundaries described in the case study:

- ``app`` — FastAPI application: lifespan, middleware, request surface.
- ``executor`` — bounded thread-pool boundary with admission control,
  ContextVar propagation, and submitter-side timeout.
- ``runtime`` — synchronous workflow runtime stub. In production this is
  the framework's supported entrypoint (e.g. ``crewai run``).
- ``provider`` — provider wrapper with retry, error normalization, and
  per-call instrumentation. A ``FakeProvider`` is included for tests.
- ``metrics`` — Prometheus definitions for the three metric layers.
- ``secrets`` — lifespan-scoped loader for Kubernetes-mounted secrets.
- ``tracing`` — request-id ContextVar.
"""

from gateway.executor import BoundedExecutor, ExecutorRejected, ExecutorTimeout
from gateway.provider import (
    NormalizedError,
    ProviderResponse,
    ProviderWrapper,
    Throttled,
    Transient,
    Unrecoverable,
)
from gateway.runtime import WorkflowRuntime

__all__ = [
    "BoundedExecutor",
    "ExecutorRejected",
    "ExecutorTimeout",
    "NormalizedError",
    "ProviderResponse",
    "ProviderWrapper",
    "Throttled",
    "Transient",
    "Unrecoverable",
    "WorkflowRuntime",
]
