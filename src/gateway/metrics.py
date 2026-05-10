"""
Three-layer metrics surface: gateway, executor, provider.

Histogram buckets are chosen for AI workload latencies — orchestration runs
routinely take seconds to tens of seconds, and provider calls span hundreds of
milliseconds to many seconds. Default Prometheus buckets compress everything
above one second into a single bucket, which makes tail behavior invisible
exactly where it matters.

The module-level singletons follow prometheus_client's intended usage. Tests
clear them via `prometheus_client.REGISTRY.unregister` if needed; the registry
is process-global by design.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Latency buckets (seconds) tuned for AI workload distributions: short
# instrumentation calls at the low end, long workflow runs at the top.
_WORKFLOW_BUCKETS = (
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
    300.0,
)
_PROVIDER_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    20.0,
    30.0,
)
_GATEWAY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

# Gateway layer: HTTP-side observations.
gateway_request_duration = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end request duration at the FastAPI gateway, including executor wait time.",
    labelnames=("route", "outcome"),
    buckets=_GATEWAY_BUCKETS,
)
gateway_in_flight = Gauge(
    "gateway_in_flight_requests",
    "Number of requests currently being served by the gateway.",
    labelnames=("route",),
)

# Executor layer: where saturation actually happens.
executor_queue_depth = Gauge(
    "executor_queue_depth",
    "Tasks queued behind the bounded executor pool, by pool name.",
    labelnames=("pool",),
)
executor_active_workers = Gauge(
    "executor_active_workers",
    "Worker threads currently executing a task, by pool name.",
    labelnames=("pool",),
)
executor_task_duration = Histogram(
    "executor_task_duration_seconds",
    "Duration of work submitted to the bounded executor (sync runtime time).",
    labelnames=("pool", "outcome"),
    buckets=_WORKFLOW_BUCKETS,
)
executor_rejections = Counter(
    "executor_rejections_total",
    "Tasks rejected because the executor refused admission (e.g. shutdown).",
    labelnames=("pool", "reason"),
)

# Provider layer: where Bedrock-equivalent failures live.
provider_call_duration = Histogram(
    "provider_call_duration_seconds",
    "Per-call duration of provider invocations (post-retry, end-to-end).",
    labelnames=("provider", "model", "outcome"),
    buckets=_PROVIDER_BUCKETS,
)
provider_retries = Counter(
    "provider_retries_total",
    "Number of retry attempts performed by the provider wrapper.",
    labelnames=("provider", "model", "reason"),
)
provider_errors = Counter(
    "provider_errors_total",
    "Provider errors after retry budget is exhausted, by normalized class.",
    labelnames=("provider", "model", "error_class"),
)
