"""
End-to-end tests through the FastAPI app.

These pin contracts the case study and module docstrings make:

- Lifespan startup populates state before traffic is accepted; a startup
  failure (e.g., missing secrets) prevents the app from serving requests.
- The middleware attaches `x-request-id` to every response, including
  responses produced when a handler raises before producing one.
- A client-supplied `x-request-id` is honored; absence triggers generation.
- /v1/run maps:
    ExecutorTimeout    -> 504
    ExecutorRejected   -> 503
    Unrecoverable      -> 502
    invalid payload    -> 422
- /healthz returns 200 with a status field.
- /metrics exposes the three documented metric layers.

`httpx.ASGITransport` drives the app without a real server.
`asgi_lifespan.LifespanManager` triggers the lifespan hook (ASGITransport
otherwise skips it).
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport

from gateway.app import GatewayConfig, create_app
from gateway.provider import (
    FakeProvider,
    Provider,
    ProviderResponse,
    Throttled,
    Unrecoverable,
)
from gateway.secrets import Secrets


def _factory(*, fail_first: int = 0) -> object:
    def make(_: Secrets) -> Provider:
        return FakeProvider(latency_seconds=0.0, fail_first=fail_first, failure=Throttled)

    return make


@pytest.fixture(autouse=True)
def _set_env_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default secret loader requires either a mount path or env-prefixed vars.
    monkeypatch.setenv("GATEWAY_TEST_KEY", "value")


@pytest.fixture
async def client() -> httpx.AsyncClient:
    app = create_app(
        config=GatewayConfig(
            pool_name="t",
            pool_size=2,
            workflow_timeout_seconds=5.0,
        ),
        provider_factory=_factory(),  # type: ignore[arg-type]
    )
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# ----- Health and metrics surfaces -----------------------------------------


async def test_healthz_returns_ok(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_metrics_exposes_three_documented_layers(client: httpx.AsyncClient) -> None:
    """
    The case study claim: gateway, executor, and provider each contribute
    distinct signals. Drive one request through to populate samples, then
    assert the documented metric names appear with values.
    """
    await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "hello"},
    )

    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.text

    # Gateway layer.
    assert "gateway_request_duration_seconds_count" in body
    assert "gateway_in_flight_requests" in body
    # Executor layer.
    assert "executor_queue_depth" in body
    assert "executor_active_workers" in body
    assert "executor_task_duration_seconds_count" in body
    # Provider layer.
    assert 'provider_call_duration_seconds_count{model="m"' in body
    assert 'outcome="ok"' in body
    assert 'provider="fake"' in body


# ----- Happy path and request-id propagation -------------------------------


async def test_run_happy_path_returns_full_response(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "hello"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["project"] == "demo"
    assert body["provider"] == "fake"
    assert body["model"] == "m"
    assert "echo[m]" in body["output"]
    assert isinstance(body["input_tokens"], int)
    assert isinstance(body["output_tokens"], int)
    assert isinstance(body["request_id"], str) and len(body["request_id"]) > 0


async def test_client_supplied_request_id_is_honored(client: httpx.AsyncClient) -> None:
    rid = "rid-abc-123"
    r = await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "hello"},
        headers={"x-request-id": rid},
    )
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == rid
    assert r.json()["request_id"] == rid


async def test_request_id_is_generated_when_absent(client: httpx.AsyncClient) -> None:
    """No header supplied → middleware generates a fresh id and surfaces it."""
    r = await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "hello"},
    )
    rid = r.headers.get("x-request-id")
    assert rid is not None and len(rid) > 0
    # The generated id round-trips into the response body.
    assert r.json()["request_id"] == rid


async def test_distinct_requests_get_distinct_generated_ids(client: httpx.AsyncClient) -> None:
    """Generated ids must not collide; this also catches accidental caching."""
    r1 = await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "hello"},
    )
    r2 = await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "hello"},
    )
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


# ----- Validation ----------------------------------------------------------


async def test_run_rejects_empty_project(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/run",
        json={"project": "", "model": "m", "prompt": "hello"},
    )
    assert r.status_code == 422


async def test_run_rejects_missing_field(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/run", json={"project": "demo", "model": "m"})
    assert r.status_code == 422


async def test_run_rejects_oversized_prompt(client: httpx.AsyncClient) -> None:
    """Prompt size cap exists to bound per-request memory; pin it."""
    r = await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "x" * 70_000},
    )
    assert r.status_code == 422


# ----- Error mapping -------------------------------------------------------


async def test_run_returns_504_on_executor_timeout() -> None:
    """A workflow that exceeds the orchestration budget surfaces as 504, not 500."""

    def slow_factory(_: Secrets) -> Provider:
        class SlowProvider:
            name = "slow"

            def invoke(self, model: str, prompt: str) -> ProviderResponse:
                time.sleep(0.5)
                raise RuntimeError("unreachable")

        return SlowProvider()  # type: ignore[return-value]

    app = create_app(
        config=GatewayConfig(pool_name="t", pool_size=1, workflow_timeout_seconds=0.05),
        provider_factory=slow_factory,
    )
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/v1/run",
                json={"project": "demo", "model": "m", "prompt": "hello"},
            )
            assert r.status_code == 504
            # The request id must still be present on error paths.
            assert "x-request-id" in r.headers


async def test_run_returns_502_on_unrecoverable_provider_error() -> None:
    """
    Unrecoverable upstream error → 502 Bad Gateway. The semantics: the
    *gateway* received a bad response from an upstream dependency.
    """

    def factory(_: Secrets) -> Provider:
        return FakeProvider(latency_seconds=0.0, fail_first=10, failure=Unrecoverable)

    app = create_app(
        config=GatewayConfig(pool_name="t", pool_size=1, workflow_timeout_seconds=5.0),
        provider_factory=factory,
    )
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/v1/run",
                json={"project": "demo", "model": "m", "prompt": "hello"},
            )
            assert r.status_code == 502
            assert "x-request-id" in r.headers


async def test_run_returns_503_after_executor_close() -> None:
    """
    A request that arrives after the executor has been closed (e.g.,
    in-flight during shutdown) maps to ExecutorRejected → 503 Service
    Unavailable.
    """

    def factory(_: Secrets) -> Provider:
        return FakeProvider(latency_seconds=0.0)

    app = create_app(
        config=GatewayConfig(pool_name="t", pool_size=1, workflow_timeout_seconds=5.0),
        provider_factory=factory,
    )
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            # Reach into the running app to close the executor mid-life,
            # simulating the window between shutdown begin and shutdown
            # complete where readiness has flipped but the app is still
            # serving.
            from gateway.app import _STATE_KEY  # noqa: PLC0415

            state = getattr(app.state, _STATE_KEY)
            await state.executor.aclose()

            r = await c.post(
                "/v1/run",
                json={"project": "demo", "model": "m", "prompt": "hello"},
            )
            assert r.status_code == 503
            assert "x-request-id" in r.headers


async def test_request_id_survives_unhandled_handler_exception() -> None:
    """
    If a handler raises before producing a response, the middleware must
    still attach `x-request-id`. Losing the header on exception paths
    breaks correlation exactly when correlation matters most.
    """
    rid = "rid-exc-test"

    def factory(_: Secrets) -> Provider:
        return FakeProvider(latency_seconds=0.0)

    app = create_app(
        config=GatewayConfig(pool_name="t", pool_size=1, workflow_timeout_seconds=5.0),
        provider_factory=factory,
    )

    @app.get("/boom")
    async def boom() -> dict[str, str]:
        raise RuntimeError("intentional handler failure")

    async with LifespanManager(app):
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/boom", headers={"x-request-id": rid})
            assert r.status_code == 500
            assert r.headers.get("x-request-id") == rid
            assert r.json().get("request_id") == rid


# ----- Lifespan contract ---------------------------------------------------


async def test_lifespan_failure_blocks_traffic(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A lifespan startup that fails (e.g., no secrets resolvable) must prevent
    the app from accepting traffic. The case study claim: 'running but
    misconfigured is structurally impossible.' Pin it.
    """
    monkeypatch.delenv("GATEWAY_TEST_KEY", raising=False)

    app = create_app(
        config=GatewayConfig(
            pool_name="t",
            pool_size=1,
            workflow_timeout_seconds=5.0,
            secret_env_prefix="ABSENT_PREFIX_DEFINITELY_NOT_SET",
            secret_mount_path=None,
        ),
        provider_factory=_factory(),  # type: ignore[arg-type]
    )

    from gateway.secrets import SecretSourceUnavailable  # noqa: PLC0415

    with pytest.raises(SecretSourceUnavailable):
        async with LifespanManager(app):
            # Should never reach here: lifespan startup must fail before yield.
            pass


async def test_handler_fails_loudly_if_state_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The typed state accessor must raise a clear error if the lifespan
    didn't populate state. This makes "code reached production with
    lifespan misconfigured" produce a fast, clear failure instead of a
    confusing AttributeError.
    """
    from types import SimpleNamespace

    from gateway.app import _get_state  # noqa: PLC0415

    fake_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="not initialized"):
        _get_state(fake_request)  # type: ignore[arg-type]


# ----- Concurrent traffic --------------------------------------------------


async def test_concurrent_requests_do_not_corrupt_request_ids(
    client: httpx.AsyncClient,
) -> None:
    """
    Ten concurrent requests, each with a distinct supplied id. Each
    response must carry its own id back, with no cross-talk. This pins
    the ContextVar isolation property end-to-end through the gateway.
    """
    rids = [f"rid-{i:04d}" for i in range(10)]

    async def one(rid: str) -> tuple[str, str]:
        r = await client.post(
            "/v1/run",
            json={"project": "demo", "model": "m", "prompt": rid},
            headers={"x-request-id": rid},
        )
        assert r.status_code == 200
        body = r.json()
        return rid, body["request_id"]

    results = await asyncio.gather(*(one(rid) for rid in rids))
    for sent, returned in results:
        assert sent == returned, f"request id corrupted: sent={sent!r} returned={returned!r}"


# ----- /readyz: liveness vs readiness separation --------------------------


async def test_readyz_reports_ready_under_normal_operation(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


async def test_readyz_reports_draining_after_executor_closed() -> None:
    """
    Once the executor is closed (graceful shutdown window), /readyz must
    flip to 503 so Kubernetes removes the pod from service endpoints.
    /healthz must still return 200 — the process is alive, just draining.
    """

    def factory(_: Secrets) -> Provider:
        return FakeProvider(latency_seconds=0.0)

    app = create_app(
        config=GatewayConfig(pool_name="t", pool_size=1, workflow_timeout_seconds=5.0),
        provider_factory=factory,
    )
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            from gateway.app import _STATE_KEY  # noqa: PLC0415

            state = getattr(app.state, _STATE_KEY)
            await state.executor.aclose()

            healthz = await c.get("/healthz")
            assert healthz.status_code == 200, "liveness must remain 200 during drain"

            readyz = await c.get("/readyz")
            assert readyz.status_code == 503
            assert readyz.json()["status"] == "draining"


# ----- Logging integration ------------------------------------------------


async def test_request_id_propagates_into_log_records(
    client: httpx.AsyncClient, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Log records emitted during request handling must carry the request id
    via the RequestIdFilter. Without this, log correlation across services
    is impossible — the response header alone doesn't help an operator
    grepping logs for a failed request.
    """
    from gateway.tracing import RequestIdFilter  # noqa: PLC0415

    rid = "rid-log-test"
    rid_filter = RequestIdFilter()
    caplog.handler.addFilter(rid_filter)

    with caplog.at_level("INFO"):
        r = await client.post(
            "/v1/run",
            json={"project": "demo", "model": "m", "prompt": "hello"},
            headers={"x-request-id": rid},
        )
    assert r.status_code == 200

    runtime_records = [rec for rec in caplog.records if rec.name == "gateway.runtime"]
    assert runtime_records, "expected at least one log record from gateway.runtime"
    assert all(getattr(rec, "request_id", None) == rid for rec in runtime_records), (
        f"request_id missing or wrong on records: "
        f"{[(r.name, getattr(r, 'request_id', None)) for r in runtime_records]}"
    )


# ----- Cancellation -------------------------------------------------------


async def test_cancellation_propagates_and_is_logged() -> None:
    """
    When the awaiting coroutine is cancelled (client disconnect, ASGI
    shutdown), the middleware must record the outcome as 'cancelled' and
    re-raise. CancelledError must NOT be swallowed by the middleware's
    `except Exception` clause — that would convert a cancellation into a
    500 response and leave a worker thread running with no awareness that
    the upstream gave up.
    """
    from prometheus_client import REGISTRY  # noqa: PLC0415

    def factory(_: Secrets) -> Provider:
        return FakeProvider(latency_seconds=0.5)

    app = create_app(
        config=GatewayConfig(pool_name="t", pool_size=1, workflow_timeout_seconds=10.0),
        provider_factory=factory,
    )
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            before = REGISTRY.get_sample_value(
                "gateway_request_duration_seconds_count",
                labels={"route": "/v1/run", "outcome": "cancelled"},
            )

            task = asyncio.create_task(
                c.post(
                    "/v1/run",
                    json={"project": "demo", "model": "m", "prompt": "hello"},
                )
            )
            # Give the request time to enter the handler before cancelling.
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            after = REGISTRY.get_sample_value(
                "gateway_request_duration_seconds_count",
                labels={"route": "/v1/run", "outcome": "cancelled"},
            )
            assert (after or 0) - (before or 0) >= 1.0, (
                "cancellation outcome was not recorded by the middleware"
            )
