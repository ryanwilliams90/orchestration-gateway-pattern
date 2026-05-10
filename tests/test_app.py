"""
End-to-end tests through the FastAPI app. These exercise the integrated
path: handler → executor → runtime → provider wrapper → fake provider.

`httpx.ASGITransport` lets us drive the app without a real server, which
keeps tests deterministic and fast.
"""

from __future__ import annotations

import httpx
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport

from gateway.app import GatewayConfig, create_app
from gateway.provider import FakeProvider, Provider, Throttled, Unrecoverable
from gateway.secrets import Secrets


def _factory(*, fail_first: int = 0) -> object:
    def make(_: Secrets) -> Provider:
        return FakeProvider(latency_seconds=0.0, fail_first=fail_first, failure=Throttled)

    return make


@pytest.fixture(autouse=True)
def _set_env_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    # The default secret loader expects either a mount path or env-prefixed vars.
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


async def test_healthz(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_metrics_exposed(client: httpx.AsyncClient) -> None:
    r = await client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert "gateway_request_duration_seconds" in body
    assert "executor_queue_depth" in body
    assert "provider_call_duration_seconds" in body


async def test_run_happy_path(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "hello"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["project"] == "demo"
    assert body["provider"] == "fake"
    assert "request_id" in body
    assert "echo[m]" in body["output"]


async def test_run_propagates_request_id(client: httpx.AsyncClient) -> None:
    rid = "rid-abc-123"
    r = await client.post(
        "/v1/run",
        json={"project": "demo", "model": "m", "prompt": "hello"},
        headers={"x-request-id": rid},
    )
    assert r.status_code == 200
    assert r.headers.get("x-request-id") == rid
    assert r.json()["request_id"] == rid


async def test_run_rejects_invalid_payload(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/run",
        json={"project": "", "model": "m", "prompt": "hello"},
    )
    assert r.status_code == 422


async def test_run_returns_504_on_timeout() -> None:
    # A workflow that always exceeds the budget must surface as 504, not 500.
    import time

    def slow_provider_factory(_: Secrets) -> Provider:
        class SlowProvider:
            name = "slow"

            def invoke(self, model: str, prompt: str):  # type: ignore[no-untyped-def]
                time.sleep(0.5)
                raise RuntimeError("unreachable")

        return SlowProvider()  # type: ignore[return-value]

    app = create_app(
        config=GatewayConfig(
            pool_name="t",
            pool_size=1,
            workflow_timeout_seconds=0.05,
        ),
        provider_factory=slow_provider_factory,
    )
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/v1/run",
                json={"project": "demo", "model": "m", "prompt": "hello"},
            )
            assert r.status_code == 504


async def test_run_returns_502_on_unrecoverable_provider_error() -> None:
    """
    The wrapper's `Unrecoverable` taxonomy maps to 502 Bad Gateway: the
    upstream returned a permanent failure, so the *gateway* received a bad
    response, which is what 502 means.
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


async def test_request_id_survives_handler_exception() -> None:
    """
    If a handler raises before producing a response, the middleware must
    still attach `x-request-id` to whatever response is returned to the
    client. Losing the header on exception paths breaks correlation
    exactly when correlation matters most.
    """
    rid = "rid-exc-test"

    # Build a minimal app via the same factory and add a route that raises.
    # The middleware is the unit under test; we don't care about /v1/run here.
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
