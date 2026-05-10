"""
Provider wrapper.

A thin abstraction over a model provider that owns three concerns the case
study identifies as drift-prone if scattered across call sites:

1. **Error normalization** — provider-specific exception zoos collapse into a
   small, retryable/non-retryable taxonomy: `Throttled`, `Transient`,
   `Unrecoverable`. Orchestration code never imports provider SDK exceptions.

2. **Retry policy** — bounded exponential backoff with full jitter, applied
   uniformly. Orchestration code does not retry on its own; compounded retries
   across layers are the failure mode this defends against.

3. **Instrumentation** — per-call latency, retry count, model id, outcome.
   Emitted from one place so dashboards are coherent.

The `Provider` protocol is the seam. Real implementations (Bedrock, OpenAI,
Anthropic, etc.) plug in behind it. A `FakeProvider` is included for tests
and for the example app — the wrapper is exercisable end-to-end without any
cloud credentials.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

from gateway.metrics import (
    provider_call_duration,
    provider_errors,
    provider_retries,
)

log = logging.getLogger(__name__)


# ---- Normalized error taxonomy --------------------------------------------


class NormalizedError(Exception):
    """Base class for the wrapper's normalized error surface."""

    retryable: bool = False


class Throttled(NormalizedError):
    """Provider signaled rate limiting. Retry with backoff."""

    retryable = True


class Transient(NormalizedError):
    """Transport-layer or transient server-side failure. Retry with backoff."""

    retryable = True


class Unrecoverable(NormalizedError):
    """Validation, auth, or permanent failure. Do not retry."""

    retryable = False


# ---- Provider protocol and response shape ---------------------------------


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Normalized response shape. Provider-specific fields go in `raw` if needed."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    raw: Mapping[str, object] = field(default_factory=dict)


class Provider(Protocol):
    """
    The seam between the wrapper and a concrete model backend.

    Implementations raise `NormalizedError` subclasses; SDK-specific exceptions
    must not escape. This is enforced by convention (and tests).
    """

    name: str

    def invoke(self, model: str, prompt: str) -> ProviderResponse: ...


# ---- Wrapper --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff with full jitter.

    `max_delay` has an enforced upper bound (`MAX_REASONABLE_DELAY`) so that
    a typo or misconfiguration cannot produce a retry that holds an executor
    thread for an unreasonable amount of time. The executor's submitter-side
    timeout is the outer deadline, but a thread that's sleeping inside a
    retry won't be reclaimed until the sleep ends.
    """

    MAX_REASONABLE_DELAY: ClassVar[float] = 60.0

    max_attempts: int = 4
    base_delay: float = 0.25
    max_delay: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("base_delay and max_delay must be non-negative")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.max_delay > self.MAX_REASONABLE_DELAY:
            raise ValueError(
                f"max_delay {self.max_delay}s exceeds the reasonable cap "
                f"{self.MAX_REASONABLE_DELAY}s — a retry sleep this long will "
                f"hold an executor thread well past any sane request budget"
            )

    def delay_for(self, attempt: int) -> float:
        """Delay before attempt N (1-indexed). Full jitter per AWS guidance."""
        ceiling = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return random.uniform(0.0, ceiling)


class ProviderWrapper:
    """Wraps a `Provider` with retry, error normalization, and metrics."""

    def __init__(
        self,
        provider: Provider,
        *,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        # `provider.name` becomes a Prometheus label on every metric the
        # wrapper emits. An empty or whitespace-only label produces samples
        # that won't match label matchers like `provider!=""` and silently
        # disappear from dashboards. Fail fast at construction.
        if not provider.name or not provider.name.strip():
            raise ValueError(
                "provider.name must be a non-empty string; metrics labels "
                "depend on it being a stable, identifying value"
            )
        self._provider = provider
        self._retry = retry_policy or RetryPolicy()
        # `sleep` is parameterized so tests can inject a no-op without
        # patching the time module.
        self._sleep = sleep

    @property
    def provider_name(self) -> str:
        return self._provider.name

    def complete(self, model: str, prompt: str) -> ProviderResponse:
        """
        Invoke the model with bounded retry. Synchronous by design — this
        runs inside the executor pool.
        """
        last_error: NormalizedError | None = None
        started = time.perf_counter()
        outcome = "ok"

        for attempt in range(1, self._retry.max_attempts + 1):
            try:
                response = self._provider.invoke(model, prompt)
                provider_call_duration.labels(
                    provider=self._provider.name,
                    model=model,
                    outcome="ok",
                ).observe(time.perf_counter() - started)
                return response
            except NormalizedError as exc:
                last_error = exc
                if not exc.retryable or attempt == self._retry.max_attempts:
                    outcome = type(exc).__name__.lower()
                    provider_errors.labels(
                        provider=self._provider.name,
                        model=model,
                        error_class=type(exc).__name__,
                    ).inc()
                    break
                provider_retries.labels(
                    provider=self._provider.name,
                    model=model,
                    reason=type(exc).__name__,
                ).inc()
                delay = self._retry.delay_for(attempt)
                log.info(
                    "provider retry attempt=%d delay=%.3fs error=%s",
                    attempt,
                    delay,
                    type(exc).__name__,
                )
                self._sleep(delay)

        provider_call_duration.labels(
            provider=self._provider.name,
            model=model,
            outcome=outcome,
        ).observe(time.perf_counter() - started)
        # `last_error` is assigned on every path that exits the loop without
        # returning. `RetryPolicy.__post_init__` guarantees `max_attempts >= 1`,
        # so the loop body always executes at least once. The check below
        # exists for type-narrowing (mypy) and as a defensive backstop.
        if last_error is None:  # pragma: no cover
            raise RuntimeError("retry loop exited without error or success")
        raise last_error


# ---- Fake provider for tests and the example app -------------------------


class FakeProvider:
    """
    In-memory provider for tests and demos.

    Deterministic by construction: same input → same output. Can be configured
    to inject `Throttled` / `Transient` / `Unrecoverable` failures for the
    first N calls, which is what the retry tests exercise.
    """

    name: str = "fake"

    def __init__(
        self,
        *,
        latency_seconds: float = 0.01,
        fail_first: int = 0,
        failure: type[NormalizedError] = Throttled,
    ) -> None:
        self._latency = latency_seconds
        self._fail_first = fail_first
        self._failure = failure
        self._calls = 0

    @property
    def call_count(self) -> int:
        return self._calls

    def invoke(self, model: str, prompt: str) -> ProviderResponse:
        # Real providers reject empty model identifiers with their own
        # validation error. Mirror that contract so test fixtures don't
        # silently mask "we forgot to set the model" bugs.
        if not model:
            raise Unrecoverable("model must be a non-empty string")
        if not prompt:
            raise Unrecoverable("prompt must be a non-empty string")
        self._calls += 1
        if self._calls <= self._fail_first:
            raise self._failure(f"injected {self._failure.__name__} on call {self._calls}")
        time.sleep(self._latency)
        # Deterministic-but-prompt-dependent response, useful for tests that
        # want to verify the response wired through unchanged.
        synthetic = f"echo[{model}]:{prompt[:64]}"
        return ProviderResponse(
            text=synthetic,
            model=model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(synthetic) // 4,
            latency_seconds=self._latency,
        )
