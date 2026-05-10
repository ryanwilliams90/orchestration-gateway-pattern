# Orchestration Gateway Pattern

A reference implementation of the async/sync executor-boundary pattern for placing a synchronous, framework-driven agent runtime behind a FastAPI service.

This repo is a **distilled pattern**, not a production system. It is the companion code to the [Production AI Orchestration Gateway](https://github.com/ryanwilliams90/portfolio/blob/main/case-studies/01-ai-orchestration-gateway.md) case study and exists so a reader can check that the architecture described there is grounded in working code.

## What this demonstrates

The case study claims four runtime properties. Each is exercised by code and tests in this repo:

1. **Lifespan-scoped startup.** Secrets are loaded once at FastAPI lifespan startup from a Kubernetes-style mount path (or an env-var shim for local development). The runtime never reads secrets on the request path. If startup fails, the process exits before the readiness probe passes — `running but misconfigured` is structurally impossible.
   See [`gateway/secrets.py`](src/gateway/secrets.py) and [`tests/test_secrets.py`](tests/test_secrets.py).

2. **Async/sync executor boundary.** Synchronous agent runtimes do not fit the async-everywhere model. A `ThreadPoolExecutor` with a fixed pool size sits between the FastAPI handlers and the runtime. ContextVars (request id) propagate into the worker thread; saturation manifests as observable queue depth, not as event-loop lag.
   See [`gateway/executor.py`](src/gateway/executor.py) and [`tests/test_executor.py`](tests/test_executor.py).

3. **Provider wrapper with normalized errors and bounded retry.** Provider-specific exception zoos collapse into `Throttled` / `Transient` / `Unrecoverable`. Retry uses bounded exponential backoff with full jitter. Orchestration code calls a provider-neutral interface; instrumentation lives in one place.
   See [`gateway/provider.py`](src/gateway/provider.py) and [`tests/test_provider.py`](tests/test_provider.py).

4. **Three-layer metrics.** Histograms tuned for AI workload latencies (workflow runs measured in seconds-to-minutes, provider calls in tens-of-milliseconds-to-seconds), exposed at `/metrics`. Gateway, executor, and provider each contribute distinct signals — saturation is visible at the executor layer before it becomes elevated latency at the gateway.
   See [`gateway/metrics.py`](src/gateway/metrics.py).

## What this is not

- A drop-in production gateway. The `FakeProvider` is a stand-in; real deployments wire AWS Bedrock / Anthropic / OpenAI clients via the same `Provider` protocol.
- A workflow framework. The `WorkflowRuntime` stub stands in for a real synchronous agent runtime (CrewAI, LangGraph, etc.). Production code calls the framework's supported lifecycle entrypoint, not direct module imports.
- An attempt at proprietary code. Everything here is clean-room; the pattern is the point.

## Layout

```
src/gateway/
  app.py        FastAPI app factory, lifespan, request-id middleware, /v1/run
  executor.py   BoundedExecutor — the async/sync boundary
  provider.py   Provider protocol, wrapper, retry policy, FakeProvider
  runtime.py    Synchronous workflow runtime (stub)
  secrets.py    Lifespan-scoped secret loader
  metrics.py    Three-layer Prometheus metric definitions
  tracing.py    Request-id ContextVar
tests/
  test_app.py       End-to-end through the ASGI surface
  test_executor.py  Concurrency bounds, timeout honesty, ContextVar propagation
  test_provider.py  Retry semantics, error normalization, jitter envelope
  test_secrets.py   Mount-path and env-prefix loading, immutability
examples/
  run_local.py  Runs the gateway with the FakeProvider
```

## Running

```
make install
make all          # lint + type-check + tests
make test
```

Or run the example app:

```
GATEWAY_FAKE_KEY=any-value .venv/bin/python examples/run_local.py
```

```
curl -s localhost:8080/healthz
curl -s -X POST localhost:8080/v1/run \
    -H 'content-type: application/json' \
    -d '{"project":"demo","model":"claude-test","prompt":"hello"}'
curl -s localhost:8080/metrics | grep -E '(gateway|executor|provider)_'
```

## Notes for readers

A few things worth pointing out for a careful reader:

- **Timeout honesty.** `concurrent.futures` cannot cancel running tasks. The executor stops waiting on timeout; the underlying thread continues until the work finishes. The `ExecutorTimeout` test pins this behavior, and the case study addresses why this is the right tradeoff (admission control, not cancellation, is the lever).
- **Counter locking.** `_active` and `_submitted` are mutated from worker threads; a `threading.Lock` guards the increment/decrement and the derived queue-depth gauge. Without it, the depth metric would race under load.
- **Context propagation.** `loop.run_in_executor` does not copy the current ContextVar context. The executor uses `contextvars.copy_context()` and runs the callable inside `ctx.run()` so request ids flow through.
- **Mypy strict, ruff clean.** The intent is for this to read like production code, not a sketch. `make all` enforces it.

## Related

- Portfolio: [`ryanwilliams90/portfolio`](https://github.com/ryanwilliams90/portfolio)
- Case study: [Production AI Orchestration Gateway](https://github.com/ryanwilliams90/portfolio/blob/main/case-studies/01-ai-orchestration-gateway.md)
