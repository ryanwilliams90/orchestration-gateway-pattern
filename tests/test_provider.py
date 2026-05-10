"""
Tests for ProviderWrapper.

These pin the wrapper's documented contracts:

- Successful invocation returns the normalized response unchanged. Token
  counts, latency, and model id round-trip from the provider.
- Retryable errors (Throttled, Transient) are retried with bounded backoff.
  The retry budget is `max_attempts`; on exhaustion, the last error is
  raised — not the first.
- Non-retryable errors (Unrecoverable) are raised on first occurrence with
  no backoff sleep.
- Mixed retryable failures retry independent of which retryable type fired
  (the retry decision is the `.retryable` flag, not the exact class).
- Backoff sleeps follow full-jitter bounds: U(0, min(max_delay, base*2^(n-1))).
- The wrapper emits provider_call_duration on every exit path,
  provider_retries on each retry, and provider_errors on each
  budget-exhausting or non-retryable failure.
- RetryPolicy validates its invariants at construction time.
- The wrapper does not catch non-NormalizedError exceptions: provider SDK
  bugs that escape the contract surface to the caller as-is.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from gateway.provider import (
    FakeProvider,
    NormalizedError,
    Provider,
    ProviderResponse,
    ProviderWrapper,
    RetryPolicy,
    Throttled,
    Transient,
    Unrecoverable,
)

# ----- RetryPolicy contract ------------------------------------------------


def test_retry_policy_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)


def test_retry_policy_rejects_negative_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=-1)


def test_retry_policy_rejects_negative_delay() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RetryPolicy(base_delay=-1.0)


def test_retry_policy_rejects_inverted_delays() -> None:
    with pytest.raises(ValueError, match="max_delay"):
        RetryPolicy(base_delay=1.0, max_delay=0.5)


def test_retry_policy_rejects_unreasonable_max_delay() -> None:
    """
    A `max_delay` larger than the policy's reasonable cap cannot be
    constructed. This bounds the worst-case time a retry can hold an
    executor thread, defending against typos like `max_delay=300.0`.
    """
    cap = RetryPolicy.MAX_REASONABLE_DELAY
    with pytest.raises(ValueError, match="reasonable cap"):
        RetryPolicy(max_delay=cap + 1.0)


def test_retry_policy_accepts_max_delay_at_cap() -> None:
    """The cap itself is allowed (boundary condition)."""
    cap = RetryPolicy.MAX_REASONABLE_DELAY
    policy = RetryPolicy(max_delay=cap, base_delay=0.1)
    assert policy.max_delay == cap


def test_retry_policy_full_jitter_envelope() -> None:
    """
    Full-jitter delays are bounded by min(max_delay, base * 2^(n-1)).
    Sample many draws to exercise the distribution rather than just one
    point per attempt.
    """
    policy = RetryPolicy(max_attempts=5, base_delay=0.1, max_delay=1.0)
    for attempt in range(1, 6):
        ceiling = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
        for _ in range(50):
            delay = policy.delay_for(attempt)
            assert 0.0 <= delay <= ceiling, f"attempt={attempt} delay={delay} ceiling={ceiling}"


def test_retry_policy_distribution_is_jittered() -> None:
    """
    A full-jitter policy must produce variation across draws. If the policy
    were deterministic (e.g., always returned the ceiling), the pattern
    would silently degrade backoff effectiveness under thundering herd.
    """
    policy = RetryPolicy(max_attempts=4, base_delay=0.5, max_delay=4.0)
    samples = [policy.delay_for(3) for _ in range(20)]
    assert len(set(samples)) > 1, "delay_for produced no variation across draws"


# ----- Successful invocation -----------------------------------------------


def test_successful_invocation_returns_normalized_response() -> None:
    """
    The full ProviderResponse round-trips. Token counts, model id, and
    latency are part of the contract because they are what the metrics
    layer keys off.
    """
    wrapper = ProviderWrapper(FakeProvider(latency_seconds=0.0))
    result = wrapper.complete(model="claude-test", prompt="hello")

    assert isinstance(result, ProviderResponse)
    assert result.model == "claude-test"
    assert result.text.startswith("echo[claude-test]")
    assert result.input_tokens >= 0
    assert result.output_tokens >= 0
    assert result.latency_seconds >= 0


def test_provider_name_is_exposed() -> None:
    """The wrapper exposes provider_name so callers / metrics can label."""
    wrapper = ProviderWrapper(FakeProvider())
    assert wrapper.provider_name == "fake"


def test_provider_name_is_forwarded_from_underlying_provider() -> None:
    """
    `wrapper.provider_name` returns whatever the underlying provider's
    `name` attribute holds — no hardcoded fallback. Pin this so the
    wrapper cannot accidentally substitute a constant under refactor.
    """

    class CustomProvider:
        name = "bedrock-us-east-1"

        def invoke(self, model: str, prompt: str) -> ProviderResponse:
            return ProviderResponse(
                text="ok",
                model=model,
                input_tokens=0,
                output_tokens=0,
                latency_seconds=0.0,
            )

    wrapper = ProviderWrapper(CustomProvider())
    assert wrapper.provider_name == "bedrock-us-east-1"


def test_wrapper_rejects_provider_with_empty_name() -> None:
    """
    An empty `provider.name` would produce empty Prometheus labels and
    silently break dashboard label matchers. The wrapper must fail fast
    at construction.
    """

    class NamelessProvider:
        name = ""

        def invoke(self, model: str, prompt: str) -> ProviderResponse:
            raise NotImplementedError

    with pytest.raises(ValueError, match="non-empty"):
        ProviderWrapper(NamelessProvider())


def test_wrapper_rejects_provider_with_whitespace_name() -> None:
    class WhitespaceProvider:
        name = "   "

        def invoke(self, model: str, prompt: str) -> ProviderResponse:
            raise NotImplementedError

    with pytest.raises(ValueError, match="non-empty"):
        ProviderWrapper(WhitespaceProvider())


def test_fake_provider_rejects_empty_model() -> None:
    """
    The fake mirrors a real provider's contract: empty model is a hard
    Unrecoverable error. This ensures test fixtures don't accidentally
    mask a 'forgot to set the model' bug elsewhere.
    """
    provider = FakeProvider(latency_seconds=0.0)
    with pytest.raises(Unrecoverable, match="model"):
        provider.invoke(model="", prompt="hi")


def test_fake_provider_rejects_empty_prompt() -> None:
    provider = FakeProvider(latency_seconds=0.0)
    with pytest.raises(Unrecoverable, match="prompt"):
        provider.invoke(model="m", prompt="")


# ----- Retry contract ------------------------------------------------------


def test_retries_throttled_until_success() -> None:
    sleeps: list[float] = []
    provider = FakeProvider(latency_seconds=0.0, fail_first=2, failure=Throttled)
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.05),
        sleep=sleeps.append,
    )
    result = wrapper.complete(model="m", prompt="p")
    assert result.text.startswith("echo[m]")
    assert provider.call_count == 3
    assert len(sleeps) == 2  # Two backoffs between the three attempts.


def test_retries_transient_until_success() -> None:
    sleeps: list[float] = []
    provider = FakeProvider(latency_seconds=0.0, fail_first=1, failure=Transient)
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05),
        sleep=sleeps.append,
    )
    wrapper.complete(model="m", prompt="p")
    assert provider.call_count == 2
    assert len(sleeps) == 1


def test_retry_decision_is_made_on_retryable_flag_not_exact_class() -> None:
    """
    The wrapper's contract is to retry on the `.retryable` flag, so a
    custom retryable subclass is honored exactly like the canonical types.
    """

    class CustomRetryable(NormalizedError):
        retryable = True

    class FlakyProvider:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, model: str, prompt: str) -> ProviderResponse:
            self.calls += 1
            if self.calls < 3:
                raise CustomRetryable("transient-ish")
            return ProviderResponse(
                text="ok", model=model, input_tokens=0, output_tokens=0, latency_seconds=0.0
            )

    sleeps: list[float] = []
    provider: Provider = FlakyProvider()
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.02),
        sleep=sleeps.append,
    )
    wrapper.complete(model="m", prompt="p")
    assert provider.calls == 3  # type: ignore[attr-defined]
    assert len(sleeps) == 2


def test_does_not_retry_unrecoverable() -> None:
    sleeps: list[float] = []
    provider = FakeProvider(latency_seconds=0.0, fail_first=1, failure=Unrecoverable)
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.05),
        sleep=sleeps.append,
    )
    with pytest.raises(Unrecoverable):
        wrapper.complete(model="m", prompt="p")
    assert provider.call_count == 1
    assert sleeps == [], "no backoff should occur before raising a non-retryable"


def test_exhausts_budget_then_raises_last_error() -> None:
    """
    Budget exhaustion raises the *last* error encountered, not the first.
    If the provider returns increasingly informative messages across
    retries, the operator gets the most recent one.
    """

    class IndexedProvider:
        name = "indexed"

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, model: str, prompt: str) -> ProviderResponse:
            self.calls += 1
            raise Throttled(f"attempt {self.calls}")

    sleeps: list[float] = []
    provider: Provider = IndexedProvider()
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.02),
        sleep=sleeps.append,
    )

    with pytest.raises(Throttled, match="attempt 3"):
        wrapper.complete(model="m", prompt="p")
    assert provider.calls == 3  # type: ignore[attr-defined]
    assert len(sleeps) == 2  # Two backoffs; no sleep after the final failure.


def test_mixed_retryable_failures_are_handled_uniformly() -> None:
    """
    A sequence of different retryable error types must retry through to
    success — the retry decision is the flag, not the type.
    """

    class MixedProvider:
        name = "mixed"

        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, model: str, prompt: str) -> ProviderResponse:
            self.calls += 1
            if self.calls == 1:
                raise Throttled("throttled")
            if self.calls == 2:
                raise Transient("transient")
            return ProviderResponse(
                text="ok", model=model, input_tokens=0, output_tokens=0, latency_seconds=0.0
            )

    provider: Provider = MixedProvider()
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.02),
        sleep=lambda _: None,
    )
    result = wrapper.complete(model="m", prompt="p")
    assert result.text == "ok"
    assert provider.calls == 3  # type: ignore[attr-defined]


def test_unexpected_provider_exception_propagates() -> None:
    """
    The wrapper's documented surface is `NormalizedError`. Provider SDK
    bugs that leak non-NormalizedError exceptions are not silently
    swallowed — they propagate to the caller, where the executor's worker
    will surface them as a generic exception.

    This test exists to pin the contract: "if your provider violates the
    Protocol, the failure is loud, not silent."
    """

    class BadProvider:
        name = "bad"

        def invoke(self, model: str, prompt: str) -> ProviderResponse:
            raise RuntimeError("SDK bug — should never escape in production")

    wrapper = ProviderWrapper(
        BadProvider(),  # type: ignore[arg-type]
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.02),
        sleep=lambda _: None,
    )
    with pytest.raises(RuntimeError, match="SDK bug"):
        wrapper.complete(model="m", prompt="p")


# ----- Metrics emission ----------------------------------------------------


def _provider_calls_count(*, provider: str, model: str, outcome: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "provider_call_duration_seconds_count",
            labels={"provider": provider, "model": model, "outcome": outcome},
        )
        or 0.0
    )


def _provider_retries(*, provider: str, model: str, reason: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "provider_retries_total",
            labels={"provider": provider, "model": model, "reason": reason},
        )
        or 0.0
    )


def _provider_errors(*, provider: str, model: str, error_class: str) -> float:
    return (
        REGISTRY.get_sample_value(
            "provider_errors_total",
            labels={"provider": provider, "model": model, "error_class": error_class},
        )
        or 0.0
    )


def test_success_emits_duration_with_ok_outcome() -> None:
    wrapper = ProviderWrapper(FakeProvider(latency_seconds=0.0))
    before = _provider_calls_count(provider="fake", model="m", outcome="ok")
    wrapper.complete(model="m", prompt="p")
    after = _provider_calls_count(provider="fake", model="m", outcome="ok")
    assert after - before == 1.0


def test_retry_emits_retries_counter_per_attempt() -> None:
    """Two failures + one success → exactly two retry events."""
    provider = FakeProvider(latency_seconds=0.0, fail_first=2, failure=Throttled)
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.0, max_delay=0.0),
        sleep=lambda _: None,
    )
    before = _provider_retries(provider="fake", model="m", reason="Throttled")
    wrapper.complete(model="m", prompt="p")
    after = _provider_retries(provider="fake", model="m", reason="Throttled")
    assert after - before == 2.0


def test_budget_exhaustion_emits_errors_counter() -> None:
    provider = FakeProvider(latency_seconds=0.0, fail_first=10, failure=Throttled)
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, max_delay=0.0),
        sleep=lambda _: None,
    )
    before = _provider_errors(provider="fake", model="m", error_class="Throttled")
    with pytest.raises(Throttled):
        wrapper.complete(model="m", prompt="p")
    after = _provider_errors(provider="fake", model="m", error_class="Throttled")
    assert after - before == 1.0


def test_unrecoverable_emits_errors_not_retries() -> None:
    """A non-retryable failure must increment errors but not retries."""
    provider = FakeProvider(latency_seconds=0.0, fail_first=1, failure=Unrecoverable)
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.0, max_delay=0.0),
        sleep=lambda _: None,
    )
    retries_before = _provider_retries(provider="fake", model="m", reason="Unrecoverable")
    errors_before = _provider_errors(provider="fake", model="m", error_class="Unrecoverable")
    with pytest.raises(Unrecoverable):
        wrapper.complete(model="m", prompt="p")
    retries_after = _provider_retries(provider="fake", model="m", reason="Unrecoverable")
    errors_after = _provider_errors(provider="fake", model="m", error_class="Unrecoverable")

    assert retries_after - retries_before == 0.0, "non-retryable must not increment retries"
    assert errors_after - errors_before == 1.0
