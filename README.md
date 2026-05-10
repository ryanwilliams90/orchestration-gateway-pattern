# Orchestration Gateway Pattern

[![ci](https://github.com/ryanwilliams90/orchestration-gateway-pattern/actions/workflows/ci.yml/badge.svg)](https://github.com/ryanwilliams90/orchestration-gateway-pattern/actions/workflows/ci.yml)

A reference implementation of the async/sync executor-boundary pattern for placing a synchronous, framework-driven agent runtime behind a FastAPI service.

> **Python 3.11+** · `mypy --strict` clean · `ruff` clean · 90+ contract-driven tests including a real-uvicorn smoke test

This is a **distilled pattern**, not a production system. It accompanies the [Production AI Orchestration Gateway](https://github.com/ryanwilliams90/portfolio/blob/main/case-studies/01-ai-orchestration-gateway.md) case study so a reader can verify the architecture is grounded in working code.

## What this demonstrates

**The async/sync boundary itself.** A synchronous, framework-driven agent runtime cannot be invoked directly from an async handler without blocking the event loop. The executor (`gateway/executor.py`) is a thread pool with admission control on both axes — concurrent workers and queued submissions — so saturation is observable rather than emergent. ContextVars (request id) are captured at submit time and restored inside the worker via `ctx.run`. Submitter-side timeouts cancel the await; the worker continues to run because `concurrent.futures` cannot cancel arbitrary blocking work, and the duration histogram records both the submitter's abandonment and the worker's eventual completion under separate outcome labels.

**Lifespan-scoped state.** Secrets are loaded once at FastAPI lifespan startup from a Kubernetes-style mount path (or an env-var shim for local development). The runtime never reads secrets on the request path. Startup that can't resolve secrets fails before the readiness probe passes; running-but-misconfigured is structurally impossible.

**Provider wrapper with normalized errors.** Provider-specific exception zoos collapse into `Throttled` / `Transient` / `Unrecoverable`. Retry uses bounded exponential backoff with full jitter and an upper bound on `max_delay` so a typo can't hold a worker thread for half an hour. Orchestration code calls a provider-neutral interface; instrumentation lives in one place.

**Three metric layers.** Histograms tuned for AI workload latencies, exposed at `/metrics`. Gateway, executor, and provider each contribute distinct signals — executor saturation is visible before it becomes elevated latency at the gateway.

**Liveness vs readiness.** `/healthz` reports liveness (the process is up). `/readyz` reports readiness — it returns 503 once the executor is closed, so Kubernetes removes the pod from service endpoints during the graceful-shutdown window even though the process is still alive.

**Request id end-to-end.** The middleware honors a client-supplied `x-request-id` or generates one. The id propagates into every log record via a `RequestIdFilter`, into worker threads via the captured `contextvars.Context`, and back to the client via the response header — including on error paths, where the middleware synthesizes a 500 response with the id attached so correlation isn't lost when handlers raise.

## What this is not

- A drop-in production gateway. The `FakeProvider` is a stand-in; real deployments wire AWS Bedrock / Anthropic / OpenAI clients via the same `Provider` protocol.
- A workflow framework. The `WorkflowRuntime` stub stands in for a real synchronous agent runtime (CrewAI, LangGraph, etc.). Production code calls the framework's supported lifecycle entrypoint, not direct module imports.
- Proprietary code. Everything is clean-room; the pattern is the point.

## Layout

```
src/gateway/
  app.py        FastAPI factory, lifespan, middleware, /v1/run, /healthz, /readyz
  executor.py   BoundedExecutor — the async/sync boundary with admission control
  provider.py   Provider protocol, wrapper, retry policy, FakeProvider
  runtime.py    Synchronous workflow runtime (stub)
  secrets.py    Lifespan-scoped secret loader (MappingProxyType-immutable)
  metrics.py    Three-layer Prometheus metric definitions
  tracing.py    Request-id ContextVar + RequestIdFilter for log records
  py.typed      PEP 561 marker
tests/
  test_app.py       End-to-end through the ASGI surface; readiness, cancellation, log integration
  test_executor.py  Concurrency bounds, admission control, timeout honesty, ContextVar propagation
  test_provider.py  Retry semantics, error normalization, jitter envelope, name validation
  test_runtime.py   Workflow request validation, result round-trip
  test_secrets.py   Mount-path / env-prefix loading, dotfile skipping, immutability
  test_smoke.py     Real-uvicorn subprocess test driving HTTP through the full stack
examples/
  run_local.py  Runs the gateway with the FakeProvider
```

## Running

```
make install
make all          # lint + type-check + tests
```

Or run the example app:

```
GATEWAY_FAKE_KEY=any-value .venv/bin/python examples/run_local.py
```

```
curl -s localhost:8080/healthz
curl -s localhost:8080/readyz
curl -s -X POST localhost:8080/v1/run \
    -H 'content-type: application/json' \
    -H 'x-request-id: my-trace-id' \
    -d '{"project":"demo","model":"claude-test","prompt":"hello"}'
curl -s localhost:8080/metrics | grep -E '(gateway|executor|provider)_'
```

The example app's logs will include `rid=my-trace-id` on every record produced during that request, so log correlation works without further configuration.

## Related

- Portfolio: [`ryanwilliams90/portfolio`](https://github.com/ryanwilliams90/portfolio)
- Case study: [Production AI Orchestration Gateway](https://github.com/ryanwilliams90/portfolio/blob/main/case-studies/01-ai-orchestration-gateway.md)
