"""
FastAPI application.

Demonstrates the four runtime properties the case study describes:

1. Lifespan-scoped startup: secrets resolved, executor pool created, provider
   wrapper constructed once. Readiness probe stays red until startup
   completes successfully.

2. Async handlers do nothing blocking. Validation, executor submission, await
   the result, serialize. Any blocking call goes through the executor.

3. Per-route timeout enforced at the executor boundary. Client-side, ingress,
   and gateway timeouts are documented in the README; the orchestration
   budget is the inner deadline and must be the shortest.

4. Three-layer metrics surfaced at `/metrics`: gateway, executor, provider.

The app accepts a `provider_factory` parameter so tests can construct it with
a `FakeProvider` and run end-to-end without external dependencies.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
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
from gateway.tracing import get_request_id, new_request_id, set_request_id

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

        app.state.gateway = AppState(
            config=cfg,
            secrets=secrets,
            executor=executor,
            runtime=runtime,
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
        response: Response | None = None
        try:
            response = await call_next(request)
            if response.status_code >= 500:
                outcome = "server_error"
            elif response.status_code >= 400:
                outcome = "client_error"
        except Exception:
            outcome = "exception"
            raise
        finally:
            gateway_request_duration.labels(route=route, outcome=outcome).observe(
                time.perf_counter() - started
            )
            gateway_in_flight.labels(route=route).dec()
        response.headers["x-request-id"] = rid
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/run", response_model=RunResponse)
    async def run(payload: RunRequest, request: Request) -> RunResponse:
        state: AppState = request.app.state.gateway
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
            # Provider validation/auth failure — surfaces to the caller as 4xx.
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


def _logging_setup() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


_logging_setup()
