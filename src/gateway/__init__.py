"""
Orchestration gateway pattern — a reference implementation of the async/sync
executor-boundary pattern for placing a synchronous, framework-driven agent
runtime behind a FastAPI service surface.

The package is organized around the boundaries described in the case study:

- `app`        — FastAPI application, lifespan secret loading, route surface.
- `executor`   — ThreadPoolExecutor boundary with bounded concurrency and
                 timeout enforcement.
- `runtime`    — synchronous workflow runtime stub. In production this would be
                 the framework's supported entrypoint (e.g. `crewai run`).
- `provider`   — provider wrapper with retry/jitter, error normalization, and
                 per-call instrumentation. A fake provider is included for
                 tests and local runs.
- `metrics`    — Prometheus histogram/counter definitions for the three
                 metric layers (gateway, executor, provider).
- `secrets`    — lifespan-scoped secret loader; reads from a Kubernetes-style
                 mount path or an environment shim.
- `tracing`    — request-id propagation across the async/sync boundary.
"""

from gateway.executor import BoundedExecutor, ExecutorTimeout
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
    "ExecutorTimeout",
    "NormalizedError",
    "ProviderResponse",
    "ProviderWrapper",
    "Throttled",
    "Transient",
    "Unrecoverable",
    "WorkflowRuntime",
]
