"""
Tests for the BoundedExecutor. The interesting properties are not "does it
return a value" but "does it bound concurrency, propagate context, and
handle timeout honestly."

Tests synchronize with worker threads via `threading.Semaphore` rather than
`asyncio.sleep`, so they are deterministic on slow CI runners. A test that
sleeps "long enough for the workers to start" is a test that flakes once a
quarter; tests here block until a specific worker-thread observation has
occurred.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from gateway.executor import BoundedExecutor, ExecutorRejected, ExecutorTimeout
from gateway.tracing import get_request_id, set_request_id


@pytest.fixture
async def executor() -> BoundedExecutor:
    ex = BoundedExecutor(name="test", max_workers=2)
    yield ex
    await ex.aclose()


async def test_returns_value_from_sync_callable(executor: BoundedExecutor) -> None:
    result = await executor.submit(lambda x: x * 2, 21, timeout=1.0)
    assert result == 42


async def test_propagates_synchronous_exception(executor: BoundedExecutor) -> None:
    def boom() -> None:
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        await executor.submit(boom, timeout=1.0)


async def test_concurrency_is_bounded_by_pool_size(executor: BoundedExecutor) -> None:
    """
    Pool size is 2. The third submission must wait until one of the first
    two finishes before it starts.

    Synchronization is deterministic: each worker increments a `started`
    semaphore when it begins, and waits on a `release` event before exiting.
    The test acquires the semaphore exactly twice (both initial workers
    started), asserts the third worker has not yet started, then releases.
    """
    started_sem = threading.Semaphore(0)
    release = threading.Event()
    starts: list[float] = []
    starts_lock = threading.Lock()

    def held() -> float:
        ts = time.perf_counter()
        with starts_lock:
            starts.append(ts)
        started_sem.release()
        release.wait(timeout=5.0)
        return ts

    loop = asyncio.get_running_loop()

    async def acquire_started() -> None:
        # `Semaphore.acquire` is blocking; run on the default executor so the
        # event loop continues servicing the submitted tasks.
        await loop.run_in_executor(None, started_sem.acquire)

    t1 = asyncio.create_task(executor.submit(held, timeout=5.0))
    t2 = asyncio.create_task(executor.submit(held, timeout=5.0))
    t3 = asyncio.create_task(executor.submit(held, timeout=5.0))

    # Wait deterministically for exactly two workers to have started.
    await asyncio.wait_for(acquire_started(), timeout=2.0)
    await asyncio.wait_for(acquire_started(), timeout=2.0)

    # The third worker must not have started — pool size is 2, both slots
    # occupied. If `started_sem` has a third permit waiting, concurrency
    # is unbounded.
    assert not started_sem.acquire(blocking=False), (
        "third worker started before pool freed a slot — concurrency unbounded"
    )
    with starts_lock:
        assert len(starts) == 2

    release.set()
    await asyncio.gather(t1, t2, t3)
    with starts_lock:
        assert len(starts) == 3


async def test_timeout_raises_executor_timeout(executor: BoundedExecutor) -> None:
    def slow() -> None:
        time.sleep(0.5)

    with pytest.raises(ExecutorTimeout):
        await executor.submit(slow, timeout=0.05)


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

    # Underlying work continues; wait deterministically for completion.
    assert completed.wait(timeout=2.0), "underlying thread did not complete"


async def test_counters_recover_after_timed_out_task_completes(
    executor: BoundedExecutor,
) -> None:
    """
    A submitter timeout must not leak counter state. The worker's `finally`
    block decrements `_active` and `_submitted` when the work eventually
    finishes — even though the submitter has long since given up.

    This is the test that pins the contract behind the README's claim that
    "the executor slot is treated as occupied until the work actually
    finishes" — and that, once it finishes, the slot is reclaimed.
    """
    finished = threading.Event()

    def slow() -> None:
        time.sleep(0.15)
        finished.set()

    with pytest.raises(ExecutorTimeout):
        await executor.submit(slow, timeout=0.02)

    assert finished.wait(timeout=2.0)
    # Allow the worker's finally block to run.
    for _ in range(50):
        if executor.active_workers() == 0 and executor.queue_depth() == 0:
            break
        await asyncio.sleep(0.01)
    assert executor.active_workers() == 0
    assert executor.queue_depth() == 0


async def test_request_id_propagates_into_worker_thread(
    executor: BoundedExecutor,
) -> None:
    set_request_id("test-rid-123")

    def read_rid() -> str | None:
        return get_request_id()

    result = await executor.submit(read_rid, timeout=1.0)
    assert result == "test-rid-123"


async def test_rejects_after_close() -> None:
    ex = BoundedExecutor(name="closed", max_workers=1)
    await ex.aclose()
    with pytest.raises(ExecutorRejected):
        await ex.submit(lambda: 1, timeout=1.0)


async def test_queue_depth_reflects_saturation(executor: BoundedExecutor) -> None:
    """
    Saturate a pool of size 2 with three tasks. Synchronize on the workers
    actually entering execution so the assertion is deterministic.
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
    Stress test: submit many tasks against a small pool, sample
    queue_depth and active_workers throughout, and assert that the bounds
    invariant always holds: 0 <= active_workers <= pool_size and
    0 <= queue_depth <= submitted - completed.

    This is the test that would have caught the queue-depth race condition.
    Without lock-protected gauge updates, the active_workers gauge can
    transiently exceed pool_size during a torn read.
    """
    pool_size = 3
    total_tasks = 60
    ex = BoundedExecutor(name="stress", max_workers=pool_size)

    completed = 0
    completed_lock = threading.Lock()

    def task() -> None:
        nonlocal completed
        # Brief work; we want many tasks moving through the pool quickly.
        time.sleep(0.005)
        with completed_lock:
            completed += 1

    async def sampler(stop: asyncio.Event, violations: list[str]) -> None:
        while not stop.is_set():
            active = ex.active_workers()
            depth = ex.queue_depth()
            if active < 0 or active > pool_size:
                violations.append(f"active_workers={active} out of [0, {pool_size}]")
            if depth < 0:
                violations.append(f"queue_depth={depth} negative")
            await asyncio.sleep(0)  # yield without sleeping a real interval

    violations: list[str] = []
    stop = asyncio.Event()
    sampler_task = asyncio.create_task(sampler(stop, violations))

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
