"""FastAPI application: lifespan, middleware, request surface.

Async handlers do nothing blocking. They validate, hand off to the
executor, await the result, and serialize. Anything that blocks crosses
the executor boundary first.

The lifespan resolves secrets, constructs the executor pool and provider
wrapper, and stores them on ``app.state`` once. Startup failure exits the
process before the readiness probe passes.

The app factory takes an optional ``provider_factory`` so tests inject a
``FakeProvider`` and run the full path without external credentials.
``configure_logging`` is exposed but not invoked at import — call it from
the entrypoint, as ``examples/run_local.py`` does.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from gateway.executor import BoundedExecutor, ExecutorRejected, ExecutorTimeout
from gateway.metrics import gateway_in_flight, gateway_request_duration
from gateway.provider import (
    FakeProvider,
    Provider,
    ProviderWrapper,
    Unrecoverable,
)
from gateway.runtime import WorkflowRequest, WorkflowRuntime
from gateway.secrets import Secrets, load_secrets
from gateway.tracing import (
    RequestIdFilter,
    get_request_id,
    new_request_id,
    set_request_id,
)

log = logging.getLogger(__name__)


# ---- Configuration --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    pool_name: str = "workflows"
    pool_size: int = 4
    workflow_timeout_seconds: float = 30.0
    secret_mount_path: str | None = None
    secret_env_prefix: str | None = "GATEWAY"


# ---- Request / response models -------------------------------------------


class RunRequest(BaseModel):
    project: str = Field(min_length=1, max_length=128)
    model: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1, max_length=64_000)


class RunResponse(BaseModel):
    request_id: str
    project: str
    output: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


# ---- App state ------------------------------------------------------------


@dataclass(slots=True)
class AppState:
    config: GatewayConfig
    secrets: Secrets
    executor: BoundedExecutor
    runtime: WorkflowRuntime


_STATE_KEY = "gateway"


def _get_state(request: Request) -> AppState:
    """Typed accessor for the lifespan-initialized AppState.

    Starlette's ``app.state`` is a dynamic namespace returning ``Any``, and a
    bare ``app.state.gateway`` raises ``AttributeError`` with a confusing
    message if lifespan startup hasn't run. This helper returns the typed
    state and raises a clear ``RuntimeError`` if it's missing.
    """
    state = getattr(request.app.state, _STATE_KEY, None)
    if state is None:
        raise RuntimeError(
            "gateway state not initialized — lifespan startup did not run "
            "or did not complete successfully"
        )
    if not isinstance(state, AppState):  # pragma: no cover  (defensive)
        raise RuntimeError(f"gateway state has unexpected type {type(state).__name__!r}")
    return state


# ---- App factory ----------------------------------------------------------


ProviderFactory = Callable[[Secrets], Provider]


def _default_provider_factory(_: Secrets) -> Provider:
    """Default factory used when none is supplied. Returns the fake provider."""
    return FakeProvider()


def create_app(
    *,
    config: GatewayConfig | None = None,
    provider_factory: ProviderFactory | None = None,
) -> FastAPI:
    """
    Construct the FastAPI app. The factory pattern is intentional: tests
    inject a fake provider; production wires a real Bedrock/Anthropic/OpenAI
    client. The wiring lives at the edge, not inside handlers.
    """
    cfg = config or GatewayConfig()
    factory = provider_factory or _default_provider_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("lifespan startup begin")
        secrets = load_secrets(
            mount_path=cfg.secret_mount_path,
            env_prefix=cfg.secret_env_prefix,
        )
        provider = factory(secrets)
        wrapper = ProviderWrapper(provider)
        executor = BoundedExecutor(name=cfg.pool_name, max_workers=cfg.pool_size)
        runtime = WorkflowRuntime(wrapper)

        setattr(
            app.state,
            _STATE_KEY,
            AppState(
                config=cfg,
                secrets=secrets,
                executor=executor,
                runtime=runtime,
            ),
        )
        log.info(
            "lifespan startup complete pool=%s pool_size=%d",
            cfg.pool_name,
            cfg.pool_size,
        )
        try:
            yield
        finally:
            log.info("lifespan shutdown begin")
            await executor.aclose()
            log.info("lifespan shutdown complete")

    app = FastAPI(
        title="Orchestration Gateway Pattern",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def _request_id_and_timing(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get("x-request-id") or new_request_id()
        set_request_id(rid)
        route = request.url.path
        gateway_in_flight.labels(route=route).inc()
        started = time.perf_counter()
        outcome = "ok"
        response: Response
        try:
            try:
                response = await call_next(request)
            except asyncio.CancelledError:
                # Client disconnect or ASGI shutdown propagation. Record the
                # cancellation as its own outcome so dashboards distinguish
                # "the request gave up" from "the handler raised", and
                # re-raise — cancellation must propagate, not be swallowed.
                outcome = "cancelled"
                log.info("request cancelled route=%s rid=%s", route, rid)
                raise
            except Exception:
                # The handler raised before producing a response. Synthesize
                # one so the client still sees the request id; without this,
                # FastAPI's default 500 response would lack x-request-id and
                # break end-to-end correlation.
                outcome = "exception"
                log.exception("unhandled exception in handler route=%s rid=%s", route, rid)
                response = JSONResponse(
                    status_code=500,
                    content={"detail": "internal server error", "request_id": rid},
                )
            else:
                if response.status_code >= 500:
                    outcome = "server_error"
                elif response.status_code >= 400:
                    outcome = "client_error"
        finally:
            gateway_request_duration.labels(route=route, outcome=outcome).observe(
                time.perf_counter() - started
            )
            gateway_in_flight.labels(route=route).dec()
        response.headers["x-request-id"] = rid
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Liveness probe: process is up.

        Distinct from readiness — this returns 200 even during graceful
        shutdown, because the process is still answering. Use ``/readyz``
        to gate traffic.
        """
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> Response:
        """Readiness probe: ready to accept work.

        Returns 503 once the executor has been closed (graceful shutdown
        window: lifespan teardown has begun, the process is still running
        but new work is being rejected). Kubernetes should remove this pod
        from service endpoints when ``/readyz`` flips to 503.
        """
        state = _get_state(request)
        if state.executor.is_closed:
            return JSONResponse(status_code=503, content={"status": "draining"})
        return JSONResponse(status_code=200, content={"status": "ready"})

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/run", response_model=RunResponse)
    async def run(payload: RunRequest, request: Request) -> RunResponse:
        state = _get_state(request)
        rid = get_request_id() or new_request_id()
        try:
            result = await state.executor.submit(
                state.runtime.run,
                WorkflowRequest(
                    project=payload.project,
                    model=payload.model,
                    prompt=payload.prompt,
                ),
                timeout=state.config.workflow_timeout_seconds,
            )
        except ExecutorTimeout as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except ExecutorRejected as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Unrecoverable as exc:
            # The upstream provider returned a non-retryable error
            # (validation, auth, permanent failure). 502 Bad Gateway is the
            # correct semantic — the *gateway* received an invalid response
            # from an upstream dependency. Mapping this to 4xx would imply
            # the *client* sent a bad request, which is not what happened.
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return RunResponse(
            request_id=rid,
            project=result.project,
            output=result.output,
            provider=result.provider,
            model=result.model,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    return app


def configure_logging() -> None:
    """Install a structured-ish format that includes the request id.

    Idempotent: if the root logger already has handlers, replace their
    formatter and attach the request-id filter — don't add another handler.
    Call this from your entrypoint (the example runner does); not called
    at import time so importing ``gateway.app`` has no side effects.
    """
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)s %(levelname)s %(name)s rid=%(request_id)s %(message)s"
    formatter = logging.Formatter(fmt)
    rid_filter = RequestIdFilter()

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(rid_filter)
        root.addHandler(stream_handler)
    else:
        for existing in root.handlers:
            existing.setFormatter(formatter)
            existing.addFilter(rid_filter)
