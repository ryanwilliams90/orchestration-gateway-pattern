"""
The async/sync boundary.

Async handlers submit a synchronous callable and ``await`` its completion.
Submission is bounded on two axes — concurrent workers and queued
admissions — so saturation is observable rather than emergent.

The properties this module enforces:

- Concurrency cap. ``max_workers`` is a hard ceiling; the cap is the number
  the operator configured, not whatever the event loop happens to schedule.
- Admission cap. ``max_queue_depth`` rejects submissions when the wait queue
  is full, with ``ExecutorRejected``. Without this, the underlying
  ``ThreadPoolExecutor`` queues unboundedly under sustained overload and
  the process OOMs before the queue-depth gauge catches up.
- Submitter-side timeout. ``asyncio.wait_for`` cancels the await; the
  underlying thread continues to run because ``concurrent.futures`` cannot
  cancel arbitrary blocking work. Counter bookkeeping is preserved by the
  worker's ``finally``, and the histogram records both the submitter's
  abandonment and the worker's eventual completion under separate outcome
  labels.
- ContextVar propagation. ``loop.run_in_executor`` does not copy the
  current context across the executor boundary; this module captures it at
  submit time and restores it inside the worker via ``ctx.run``.

Counter mutations and the derived ``gauge.set`` calls are performed under
the same lock so a stale derived value can't clobber a fresh one.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from gateway.metrics import (
    executor_active_workers,
    executor_queue_depth,
    executor_rejections,
    executor_task_duration,
)

log = logging.getLogger(__name__)

R = TypeVar("R")


class ExecutorTimeout(TimeoutError):
    """The submitter exceeded its per-call deadline.

    The underlying thread is *not* cancelled; the slot is treated as
    occupied until the work eventually completes. Defending against runaway
    tasks is the job of admission control and time budgets, not cancellation.
    """


class ExecutorRejected(RuntimeError):
    """Submission refused.

    Raised when the executor is shut down or the wait queue is full.
    """


class BoundedExecutor:
    """Thread-pool executor with admission control and instrumentation.

    Pool sizing is a deployment-time decision driven by per-task memory,
    upstream concurrency limits, and CPU/memory requests — not request
    volume alone. Queue sizing should reflect how much overload the gateway
    is willing to absorb before shedding load; a queue that's too large
    just delays the rejection signal until memory is gone.
    """

    def __init__(
        self,
        *,
        name: str,
        max_workers: int,
        max_queue_depth: int = 0,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_queue_depth < 0:
            raise ValueError("max_queue_depth must be >= 0 (0 disables admission control)")
        self._name = name
        self._max_workers = max_workers
        self._max_queue_depth = max_queue_depth
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"gateway-{name}",
        )
        # Counters are mutated from worker threads and the event loop. The
        # lock guards both the counters and the gauge.set call that derives
        # from them, keeping the gauge consistent with the counter state.
        self._active = 0
        self._submitted = 0
        self._counters = threading.Lock()
        self._closed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def max_workers(self) -> int:
        return self._max_workers

    @property
    def max_queue_depth(self) -> int:
        return self._max_queue_depth

    @property
    def is_closed(self) -> bool:
        return self._closed

    def queue_depth(self) -> int:
        with self._counters:
            return max(0, self._submitted - self._active)

    def active_workers(self) -> int:
        with self._counters:
            return self._active

    async def submit(
        self,
        fn: Callable[..., R],
        *args: Any,
        timeout: float,
        **kwargs: Any,
    ) -> R:
        """Submit ``fn(*args, **kwargs)`` for execution; wait up to ``timeout``.

        The current ContextVar context is captured and restored inside the
        worker thread.

        Raises ``ExecutorRejected`` if the executor is closed or the wait
        queue is at capacity. Raises ``ExecutorTimeout`` if the worker
        doesn't finish in time. The exception raised by ``fn`` itself
        propagates unchanged.
        """
        ctx = contextvars.copy_context()
        bound = functools.partial(fn, *args, **kwargs)

        # Admission and submitted-counter increment happen under the same
        # lock so that the cap check and the increment are atomic — no
        # race in which two submitters both see "queue not full" and then
        # both increment past the cap.
        with self._counters:
            if self._closed:
                executor_rejections.labels(pool=self._name, reason="closed").inc()
                raise ExecutorRejected(f"executor {self._name!r} is shut down")

            if self._max_queue_depth > 0:
                queued = max(0, self._submitted - self._active)
                if queued >= self._max_queue_depth:
                    executor_rejections.labels(pool=self._name, reason="queue_full").inc()
                    raise ExecutorRejected(
                        f"executor {self._name!r} queue full: {queued} waiting "
                        f"(cap {self._max_queue_depth})"
                    )

            self._submitted += 1
            depth = max(0, self._submitted - self._active)
            executor_queue_depth.labels(pool=self._name).set(depth)

        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        future = loop.run_in_executor(self._executor, self._run_in_worker, ctx, bound)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            elapsed = time.perf_counter() - started
            executor_task_duration.labels(pool=self._name, outcome="submitter_timeout").observe(
                elapsed
            )
            log.warning(
                "executor task exceeded deadline pool=%s timeout=%.2fs elapsed=%.3fs",
                self._name,
                timeout,
                elapsed,
            )
            raise ExecutorTimeout(
                f"task in pool {self._name!r} exceeded {timeout:.2f}s deadline"
            ) from exc

    def _run_in_worker(
        self,
        ctx: contextvars.Context,
        fn: Callable[[], R],
    ) -> R:
        """Run inside the worker thread: restore context, instrument duration.

        Counter increments here are paired with the ``submit``-side
        increment; the ``finally`` block decrements both ``_active`` and
        ``_submitted`` so a submitter timeout cannot leak counter state.
        """
        with self._counters:
            self._active += 1
            depth = max(0, self._submitted - self._active)
            executor_queue_depth.labels(pool=self._name).set(depth)
        executor_active_workers.labels(pool=self._name).inc()

        started = time.perf_counter()
        outcome = "ok"
        try:
            return ctx.run(fn)
        except Exception:
            outcome = "error"
            raise
        finally:
            elapsed = time.perf_counter() - started
            executor_task_duration.labels(pool=self._name, outcome=outcome).observe(elapsed)
            with self._counters:
                self._active -= 1
                self._submitted -= 1
                depth = max(0, self._submitted - self._active)
                executor_queue_depth.labels(pool=self._name).set(depth)
            executor_active_workers.labels(pool=self._name).dec()

    async def aclose(self) -> None:
        """Stop accepting new tasks and wait for in-flight work to drain.

        Idempotent. ``ThreadPoolExecutor.shutdown(wait=True)`` is run on the
        default executor so the event loop stays responsive while workers
        finish. Multi-pool deployments that close concurrently share that
        default executor — if pool count approaches ``min(32, cpu_count+4)``,
        shutdowns serialize and drain time grows linearly. Stagger if needed.
        """
        self._closed = True
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._executor.shutdown, True)
