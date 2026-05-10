"""
Tests for the provider wrapper. The behavior under test:

- Successful invocation returns the normalized response unchanged.
- Retryable errors are retried, up to the budget.
- Non-retryable errors are not retried.
- Retry budget is bounded — exhaustion raises the last error.
- Backoff delays are applied between attempts (verified via injected sleep).
"""

from __future__ import annotations

import pytest

from gateway.provider import (
    FakeProvider,
    ProviderWrapper,
    RetryPolicy,
    Throttled,
    Transient,
    Unrecoverable,
)


def test_successful_invocation_returns_response() -> None:
    wrapper = ProviderWrapper(FakeProvider(latency_seconds=0.0))
    result = wrapper.complete(model="claude-test", prompt="hello")
    assert "echo[claude-test]" in result.text
    assert result.model == "claude-test"


def test_retries_on_throttling_until_success() -> None:
    sleeps: list[float] = []
    provider = FakeProvider(latency_seconds=0.0, fail_first=2, failure=Throttled)
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.05),
        sleep=sleeps.append,
    )
    result = wrapper.complete(model="m", prompt="p")
    assert result.text.startswith("echo[m]")
    # Two failures + one success = two backoff sleeps.
    assert len(sleeps) == 2
    assert provider.call_count == 3


def test_retries_on_transient_until_success() -> None:
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
    assert sleeps == []


def test_exhausts_budget_then_raises() -> None:
    sleeps: list[float] = []
    provider = FakeProvider(latency_seconds=0.0, fail_first=10, failure=Throttled)
    wrapper = ProviderWrapper(
        provider,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05),
        sleep=sleeps.append,
    )
    with pytest.raises(Throttled):
        wrapper.complete(model="m", prompt="p")
    assert provider.call_count == 3
    # Two backoffs between three attempts; no sleep after the final failure.
    assert len(sleeps) == 2


def test_backoff_delays_are_within_jitter_envelope() -> None:
    """Full-jitter delays are bounded by `min(max_delay, base * 2^(n-1))`."""
    policy = RetryPolicy(max_attempts=5, base_delay=0.1, max_delay=1.0)
    for attempt in range(1, 6):
        delay = policy.delay_for(attempt)
        ceiling = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
        assert 0.0 <= delay <= ceiling
