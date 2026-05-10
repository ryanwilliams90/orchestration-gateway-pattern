"""
Tests for the BoundedExecutor. The interesting properties are not "does it
return a value" but "does it bound concurrency, propagate context, and
handle timeout honestly."
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
    Pool size is 2. Submitting three blocking tasks of 0.4s each should take
    at least ~0.8s — the third task waits behind the first two — but not
    1.2s, which would be serial. This is the property the case study sells.
    """
    holding_event = threading.Event()

    def held(timestamp_holder: list[float]) -> float:
        ts = time.perf_counter()
        timestamp_holder.append(ts)
        holding_event.wait(timeout=2.0)
        return ts

    starts: list[float] = []
    holding_event.clear()

    async def submit_held() -> float:
        return await executor.submit(held, starts, timeout=5.0)

    # Submit three jobs concurrently; the third should not start until one
    # of the first two finishes.
    task1 = asyncio.create_task(submit_held())
    task2 = asyncio.create_task(submit_held())
    task3 = asyncio.create_task(submit_held())

    # Give the first two time to enter the worker pool.
    await asyncio.sleep(0.1)
    assert len(starts) == 2, f"expected 2 active workers, got {len(starts)}"

    # Release the held tasks.
    holding_event.set()
    await asyncio.gather(task1, task2, task3)

    # The third task should have started after the first two finished, not
    # at submission time.
    assert len(starts) == 3
    third_start = starts[2]
    earliest_two = min(starts[0], starts[1])
    assert third_start > earliest_two + 0.05, (
        "third task started concurrently with first two — concurrency unbounded?"
    )


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

    # Underlying work continues; wait for it to actually finish.
    assert completed.wait(timeout=1.0), "underlying thread did not complete"


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
    holding_event = threading.Event()

    def held() -> None:
        holding_event.wait(timeout=2.0)

    # Saturate the pool (size=2) with two tasks, then submit a third that
    # must wait in the queue.
    t1 = asyncio.create_task(executor.submit(held, timeout=5.0))
    t2 = asyncio.create_task(executor.submit(held, timeout=5.0))
    t3 = asyncio.create_task(executor.submit(held, timeout=5.0))

    await asyncio.sleep(0.1)
    assert executor.active_workers() == 2
    assert executor.queue_depth() >= 1, f"expected queue depth >=1, got {executor.queue_depth()}"

    holding_event.set()
    await asyncio.gather(t1, t2, t3)
    assert executor.queue_depth() == 0
    assert executor.active_workers() == 0
