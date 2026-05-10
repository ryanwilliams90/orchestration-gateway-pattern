"""
Tests for BoundedExecutor.

These pin the *contracts* the module documents:

- Concurrency is capped by `max_workers`. (test_concurrency_is_bounded_*)
- Submitter timeout raises ExecutorTimeout but does not cancel the worker;
  counter state recovers when the worker eventually finishes.
  (test_timeout_*, test_counters_recover_after_timed_out_task_completes)
- ContextVars captured at submit() are visible inside the worker, isolated
  per-submission, and do not leak back into the event-loop context.
  (test_request_id_*)
- Submission after close raises ExecutorRejected and increments the
  rejections counter. (test_rejects_after_close, test_close_emits_rejection_metric)
- Counter and gauge invariants hold under concurrent submission and
  completion: 0 <= active_workers <= max_workers, queue_depth >= 0.
  (test_counters_remain_consistent_under_concurrent_load)
- Constructor validates max_workers >= 1. (test_constructor_rejects_*)
- aclose is idempotent. (test_aclose_is_idempotent)

Tests synchronize with worker threads via threading.Semaphore rather than
asyncio.sleep so they are deterministic on slow CI runners.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest
from prometheus_client import REGISTRY

from gateway.executor import BoundedExecutor, ExecutorRejected, ExecutorTimeout
from gateway.tracing import get_request_id, set_request_id


@pytest.fixture
async def executor() -> BoundedExecutor:
    ex = BoundedExecutor(name="test", max_workers=2)
    yield ex
    await ex.aclose()


# ----- Constructor contract ------------------------------------------------


def test_constructor_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        BoundedExecutor(name="x", max_workers=0)


def test_constructor_rejects_negative_workers() -> None:
    with pytest.raises(ValueError, match="max_workers"):
        BoundedExecutor(name="x", max_workers=-3)


def test_constructor_accepts_minimal_pool() -> None:
    ex = BoundedExecutor(name="x", max_workers=1)
    assert ex.max_workers == 1
    assert ex.name == "x"


# ----- Return value and exception propagation ------------------------------


async def test_returns_value_from_sync_callable(executor: BoundedExecutor) -> None:
    result = await executor.submit(lambda x: x * 2, 21, timeout=1.0)
    assert result == 42


async def test_kwargs_pass_through(executor: BoundedExecutor) -> None:
    """The submit signature must forward both positional and keyword args."""

    def fn(a: int, *, b: int) -> int:
        return a + b

    result = await executor.submit(fn, 10, timeout=1.0, b=5)
    assert result == 15


async def test_propagates_synchronous_exception(executor: BoundedExecutor) -> None:
    """The exact exception raised in the worker reaches the awaiter."""

    class Specific(RuntimeError):
        pass

    def boom() -> None:
        raise Specific("expected")

    with pytest.raises(Specific, match="expected"):
        await executor.submit(boom, timeout=1.0)


# ----- Concurrency bound ---------------------------------------------------


async def test_concurrency_is_bounded_by_pool_size(executor: BoundedExecutor) -> None:
    """
    Pool size is 2. Three submissions: only two may run concurrently. The
    third must not start until a slot is freed.

    Synchronization is deterministic. Each worker releases a 'started'
    permit on entry and waits on a 'release' event on exit. The test
    acquires exactly two permits, asserts no third permit is available,
    then releases.
    """
    started_sem = threading.Semaphore(0)
    release = threading.Event()

    def held() -> None:
        started_sem.release()
        release.wait(timeout=5.0)

    loop = asyncio.get_running_loop()
    t1 = asyncio.create_task(executor.submit(held, timeout=5.0))
    t2 = asyncio.create_task(executor.submit(held, timeout=5.0))
    t3 = asyncio.create_task(executor.submit(held, timeout=5.0))

    await asyncio.wait_for(loop.run_in_executor(None, started_sem.acquire), timeout=2.0)
    await asyncio.wait_for(loop.run_in_executor(None, started_sem.acquire), timeout=2.0)

    assert not started_sem.acquire(blocking=False), (
        "third worker started before pool freed a slot — concurrency unbounded"
    )

    release.set()
    await asyncio.gather(t1, t2, t3)


# ----- Timeout contract ----------------------------------------------------


async def test_timeout_raises_executor_timeout(executor: BoundedExecutor) -> None:
    def slow() -> None:
        time.sleep(0.5)

    with pytest.raises(ExecutorTimeout):
        await executor.submit(slow, timeout=0.05)


async def test_timeout_message_includes_pool_name_and_deadline(
    executor: BoundedExecutor,
) -> None:
    """The error message is the operator's first signal; pin its content."""

    def slow() -> None:
        time.sleep(0.3)

    with pytest.raises(ExecutorTimeout) as exc_info:
        await executor.submit(slow, timeout=0.02)

    msg = str(exc_info.value)
    assert "test" in msg, f"pool name missing from: {msg!r}"
    assert "0.02" in msg, f"deadline missing from: {msg!r}"


async def test_timeout_does_not_cancel_underlying_thread(
    executor: BoundedExecutor,
) -> None:
    """
    `concurrent.futures` cannot cancel running tasks. The submitter stops
    waiting; the worker continues. This test pins that behavior — if it
    changes, callers' assumptions about pool occupancy must be revisited.
    """
    completed = threading.Event()

    def slow() -> None:
        time.sleep(0.2)
        completed.set()

    with pytest.raises(ExecutorTimeout):
        await executor.submit(slow, timeout=0.05)

    assert completed.wait(timeout=2.0), "underlying thread did not complete"


async def test_counters_recover_after_timed_out_task_completes(
    executor: BoundedExecutor,
) -> None:
    """
    A submitter timeout must not leak counter state. The worker's `finally`
    block decrements `_active` and `_submitted` when the work eventually
    finishes — even though the submitter has long since given up.

    Without this, repeated timeouts would saturate the pool with phantom
    occupancy and the queue_depth gauge would grow without bound.
    """
    finished = threading.Event()

    def slow() -> None:
        time.sleep(0.15)
        finished.set()

    with pytest.raises(ExecutorTimeout):
        await executor.submit(slow, timeout=0.02)

    assert finished.wait(timeout=2.0)
    for _ in range(50):
        if executor.active_workers() == 0 and executor.queue_depth() == 0:
            break
        await asyncio.sleep(0.01)
    assert executor.active_workers() == 0
    assert executor.queue_depth() == 0


# ----- ContextVar propagation and isolation --------------------------------


async def test_request_id_propagates_into_worker_thread(
    executor: BoundedExecutor,
) -> None:
    set_request_id("test-rid-123")

    def read_rid() -> str | None:
        return get_request_id()

    result = await executor.submit(read_rid, timeout=1.0)
    assert result == "test-rid-123"


async def test_request_id_captured_at_submit_time(executor: BoundedExecutor) -> None:
    """
    The context is captured at submit() time. Mutating the ContextVar after
    submit() (but before the worker reads it) must not affect what the
    worker sees.
    """
    set_request_id("at-submit-time")

    seen_in_worker = threading.Event()
    proceed = threading.Event()
    seen: list[str | None] = []

    def read_rid() -> None:
        seen.append(get_request_id())
        seen_in_worker.set()
        proceed.wait(timeout=2.0)

    task = asyncio.create_task(executor.submit(read_rid, timeout=5.0))

    # Mutate the event-loop context after the worker has been submitted but
    # before the worker reads the var. Worker must still see the value
    # captured at submit time.
    set_request_id("after-submit-time")
    proceed.set()
    await task

    assert seen == ["at-submit-time"]


async def test_request_id_isolated_across_concurrent_submissions(
    executor: BoundedExecutor,
) -> None:
    """
    Two concurrent submissions, each carrying its own request id. The
    worker for submission A must see A's id; the worker for B must see B's.
    No bleed-through.
    """

    def read_rid() -> str | None:
        # A small sleep to maximise interleaving between the two workers.
        time.sleep(0.02)
        return get_request_id()

    async def submit_with(rid: str) -> str | None:
        set_request_id(rid)
        return await executor.submit(read_rid, timeout=2.0)

    a, b = await asyncio.gather(submit_with("rid-A"), submit_with("rid-B"))
    assert {a, b} == {"rid-A", "rid-B"}


async def test_worker_context_does_not_leak_back_to_event_loop(
    executor: BoundedExecutor,
) -> None:
    """
    A ContextVar set *inside* the worker via the captured context must not
    propagate back into the caller's context. The submitter's view of the
    var is unchanged after submit() returns.
    """
    set_request_id("caller")

    def mutate_inside_worker() -> None:
        set_request_id("worker-mutated")

    await executor.submit(mutate_inside_worker, timeout=1.0)
    assert get_request_id() == "caller"


# ----- Shutdown and rejection ----------------------------------------------


async def test_rejects_after_close() -> None:
    ex = BoundedExecutor(name="closed", max_workers=1)
    await ex.aclose()
    with pytest.raises(ExecutorRejected):
        await ex.submit(lambda: 1, timeout=1.0)


async def test_close_emits_rejection_metric() -> None:
    """
    A submission against a closed pool increments the rejections counter
    with reason='closed'. Operators rely on this to detect requests that
    arrived after the readiness probe started failing.
    """
    ex = BoundedExecutor(name="reject-metric", max_workers=1)
    await ex.aclose()

    before = REGISTRY.get_sample_value(
        "executor_rejections_total",
        labels={"pool": "reject-metric", "reason": "closed"},
    )

    with pytest.raises(ExecutorRejected):
        await ex.submit(lambda: 1, timeout=1.0)

    after = REGISTRY.get_sample_value(
        "executor_rejections_total",
        labels={"pool": "reject-metric", "reason": "closed"},
    )
    assert (after or 0) - (before or 0) == 1.0


async def test_aclose_is_idempotent() -> None:
    """Closing twice must not raise; production graceful shutdown may double-close."""
    ex = BoundedExecutor(name="double-close", max_workers=1)
    await ex.aclose()
    await ex.aclose()


# ----- Saturation and counter invariants -----------------------------------


async def test_queue_depth_reflects_saturation(executor: BoundedExecutor) -> None:
    """
    Saturate a pool of size 2 with three tasks. After both workers are
    running, the third waits, so queue_depth >= 1.
    """
    started_sem = threading.Semaphore(0)
    release = threading.Event()

    def held() -> None:
        started_sem.release()
        release.wait(timeout=5.0)

    loop = asyncio.get_running_loop()
    t1 = asyncio.create_task(executor.submit(held, timeout=5.0))
    t2 = asyncio.create_task(executor.submit(held, timeout=5.0))
    t3 = asyncio.create_task(executor.submit(held, timeout=5.0))

    await loop.run_in_executor(None, started_sem.acquire)
    await loop.run_in_executor(None, started_sem.acquire)

    assert executor.active_workers() == 2
    assert executor.queue_depth() >= 1, (
        f"expected queue depth >=1 with 3 tasks and pool size 2, got {executor.queue_depth()}"
    )

    release.set()
    await asyncio.gather(t1, t2, t3)
    assert executor.queue_depth() == 0
    assert executor.active_workers() == 0


async def test_counters_remain_consistent_under_concurrent_load() -> None:
    """
    Stress test: many short-lived tasks on a small pool. Sample the
    counters throughout and assert the bounds invariant always holds.

    This is the test that pins the queue-depth race fix — without
    lock-protected gauge updates, the active_workers gauge can transiently
    exceed pool_size during a torn read.
    """
    pool_size = 3
    total_tasks = 60
    ex = BoundedExecutor(name="stress", max_workers=pool_size)

    completed = 0
    completed_lock = threading.Lock()

    def task() -> None:
        nonlocal completed
        time.sleep(0.005)
        with completed_lock:
            completed += 1

    violations: list[str] = []
    stop = asyncio.Event()

    async def sampler() -> None:
        while not stop.is_set():
            active = ex.active_workers()
            depth = ex.queue_depth()
            if active < 0 or active > pool_size:
                violations.append(f"active_workers={active} out of [0, {pool_size}]")
            if depth < 0:
                violations.append(f"queue_depth={depth} negative")
            await asyncio.sleep(0)

    sampler_task = asyncio.create_task(sampler())
    try:
        await asyncio.gather(*(ex.submit(task, timeout=5.0) for _ in range(total_tasks)))
    finally:
        stop.set()
        await sampler_task
        await ex.aclose()

    assert completed == total_tasks
    assert ex.active_workers() == 0
    assert ex.queue_depth() == 0
    assert violations == [], f"counter invariants violated: {violations[:5]}"


# ----- Metrics surface -----------------------------------------------------


async def test_task_duration_metric_is_emitted_on_success(
    executor: BoundedExecutor,
) -> None:
    """
    The duration histogram must increment on successful task completion
    with outcome='ok'. Operators rely on this to compute saturation latency.
    """
    before = REGISTRY.get_sample_value(
        "executor_task_duration_seconds_count",
        labels={"pool": "test", "outcome": "ok"},
    )
    await executor.submit(lambda: None, timeout=1.0)
    after = REGISTRY.get_sample_value(
        "executor_task_duration_seconds_count",
        labels={"pool": "test", "outcome": "ok"},
    )
    assert (after or 0) - (before or 0) == 1.0


async def test_task_duration_metric_records_submitter_timeout(
    executor: BoundedExecutor,
) -> None:
    """
    On submitter timeout, the duration histogram increments with
    outcome='submitter_timeout'. Worker-side completion later increments
    again with outcome='ok' or 'error' — together they describe
    timed-out-but-still-running work honestly.
    """
    before = REGISTRY.get_sample_value(
        "executor_task_duration_seconds_count",
        labels={"pool": "test", "outcome": "submitter_timeout"},
    )

    finished = threading.Event()

    def slow() -> None:
        time.sleep(0.1)
        finished.set()

    with pytest.raises(ExecutorTimeout):
        await executor.submit(slow, timeout=0.02)

    after = REGISTRY.get_sample_value(
        "executor_task_duration_seconds_count",
        labels={"pool": "test", "outcome": "submitter_timeout"},
    )
    assert (after or 0) - (before or 0) == 1.0

    # Wait for worker to finish so its `finally` runs and the second
    # observation is emitted. We don't assert on the value here — that's
    # already covered by the success-path test.
    assert finished.wait(timeout=2.0)


# ----- Admission control --------------------------------------------------


def test_constructor_rejects_negative_queue_depth() -> None:
    with pytest.raises(ValueError, match="max_queue_depth"):
        BoundedExecutor(name="x", max_workers=1, max_queue_depth=-1)


async def test_admission_rejects_when_queue_is_full() -> None:
    """
    Pool size 1, queue cap 1: one task can run, one can wait, the third
    must be rejected with ExecutorRejected and reason='queue_full'. This
    is the bound the README claims and the OOM defence under sustained
    overload.
    """
    ex = BoundedExecutor(name="cap", max_workers=1, max_queue_depth=1)
    started_sem = threading.Semaphore(0)
    release = threading.Event()

    def held() -> None:
        started_sem.release()
        release.wait(timeout=5.0)

    loop = asyncio.get_running_loop()
    try:
        running = asyncio.create_task(ex.submit(held, timeout=5.0))
        await loop.run_in_executor(None, started_sem.acquire)

        # The pool is full (1 active). Submit one more — fits in the queue.
        queued = asyncio.create_task(ex.submit(held, timeout=5.0))
        # Wait for queue depth to reach 1 deterministically.
        for _ in range(50):
            if ex.queue_depth() == 1:
                break
            await asyncio.sleep(0.005)
        assert ex.queue_depth() == 1

        # Third submission must be rejected: queue is full.
        with pytest.raises(ExecutorRejected, match="queue full"):
            await ex.submit(lambda: None, timeout=1.0)

        release.set()
        await asyncio.gather(running, queued)
    finally:
        await ex.aclose()


async def test_queue_full_increments_rejection_metric() -> None:
    ex = BoundedExecutor(name="cap-metric", max_workers=1, max_queue_depth=1)
    started_sem = threading.Semaphore(0)
    release = threading.Event()

    def held() -> None:
        started_sem.release()
        release.wait(timeout=5.0)

    loop = asyncio.get_running_loop()
    try:
        running = asyncio.create_task(ex.submit(held, timeout=5.0))
        await loop.run_in_executor(None, started_sem.acquire)
        queued = asyncio.create_task(ex.submit(held, timeout=5.0))
        for _ in range(50):
            if ex.queue_depth() == 1:
                break
            await asyncio.sleep(0.005)

        before = REGISTRY.get_sample_value(
            "executor_rejections_total",
            labels={"pool": "cap-metric", "reason": "queue_full"},
        )
        with pytest.raises(ExecutorRejected):
            await ex.submit(lambda: None, timeout=1.0)
        after = REGISTRY.get_sample_value(
            "executor_rejections_total",
            labels={"pool": "cap-metric", "reason": "queue_full"},
        )
        assert (after or 0) - (before or 0) == 1.0

        release.set()
        await asyncio.gather(running, queued)
    finally:
        await ex.aclose()


async def test_zero_queue_depth_disables_admission_control() -> None:
    """
    ``max_queue_depth=0`` (the default) disables the cap; any number of
    submissions can queue. This preserves the prior behavior for callers
    who haven't opted in.
    """
    ex = BoundedExecutor(name="uncapped", max_workers=1, max_queue_depth=0)
    started_sem = threading.Semaphore(0)
    release = threading.Event()

    def held() -> None:
        started_sem.release()
        release.wait(timeout=5.0)

    loop = asyncio.get_running_loop()
    try:
        running = asyncio.create_task(ex.submit(held, timeout=5.0))
        await loop.run_in_executor(None, started_sem.acquire)
        # Submit five more; none should be rejected.
        queued = [asyncio.create_task(ex.submit(held, timeout=5.0)) for _ in range(5)]
        for _ in range(50):
            if ex.queue_depth() >= 5:
                break
            await asyncio.sleep(0.005)
        assert ex.queue_depth() == 5

        release.set()
        await asyncio.gather(running, *queued)
    finally:
        await ex.aclose()
