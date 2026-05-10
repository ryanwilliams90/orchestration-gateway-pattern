"""
Bounded executor — the async/sync boundary.

Async handlers submit a synchronous callable (a workflow run) and `await` its
completion. The submission is bounded by a fixed pool size, with admission
control enforced explicitly so that saturation manifests as observable queue
depth rather than as creeping latency on the event loop.

Three properties this module exists to enforce:

1. Concurrency is capped by the pool size. There is no implicit ceiling from
   asyncio scheduling; the cap is the number you configured.

2. Timeout is enforced from the *submitter's* side via `asyncio.wait_for`. The
   underlying thread continues to run after timeout — Python provides no
   portable mechanism to cancel arbitrary blocking code — but the submitter
   stops waiting and counter bookkeeping is preserved (the worker's `finally`
   block decrements counters and emits the duration metric with the actual
   elapsed time, not the timeout threshold).

3. ContextVars (request id, correlation id) propagate into the worker thread.
   `loop.run_in_executor` does not copy context by default; this module does.

Concurrency note on the gauge updates: counter increments and the
corresponding `gauge.set()` calls are performed under the same lock acquisition
so that gauge values remain consistent with the counter state. Without this,
two threads could each compute a depth value under their own lock acquisition
and then race to call `set()`, with the older value winning the race.
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
    """Raised when a submitted task exceeds its per-call deadline.

    The underlying thread is *not* cancelled. The caller should treat the
    executor slot as occupied until the work eventually completes; admission
    control is the right place to defend against runaway tasks, not cancellation.
    """


class ExecutorRejected(RuntimeError):
    """Raised when admission is refused (e.g. pool is shutting down)."""


class BoundedExecutor:
    """
    Thread-pool executor wrapped for async submission, with explicit metrics
    instrumentation and ContextVar propagation.

    The pool size is a deployment-time decision driven by per-workflow memory
    footprint, the upstream provider's concurrency ceiling, and the pod's CPU
    request — not request volume alone. See the case study for reasoning.
    """

    def __init__(self, *, name: str, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._name = name
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"gateway-{name}",
        )
        # `_active` and `_submitted` are mutated from worker threads and the
        # event-loop thread. `threading.Lock` guards both the counters and the
        # gauge.set() call that derives from them, so the gauge value always
        # reflects the locked-in counter state.
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

    def queue_depth(self) -> int:
        # Submitted-but-not-yet-running. Holding the lock for the read keeps
        # the counter pair consistent with what an observer would see.
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
        """
        Submit a synchronous callable for execution on the worker pool.

        The current ContextVar context is captured and restored inside the
        worker thread, so request ids and other correlation context flow
        through without explicit threading.
        """
        if self._closed:
            executor_rejections.labels(pool=self._name, reason="closed").inc()
            raise ExecutorRejected(f"executor {self._name!r} is shut down")

        ctx = contextvars.copy_context()
        bound = functools.partial(fn, *args, **kwargs)

        with self._counters:
            self._submitted += 1
            depth = max(0, self._submitted - self._active)
            executor_queue_depth.labels(pool=self._name).set(depth)

        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        future = loop.run_in_executor(self._executor, _runner, ctx, bound, self)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            # The thread keeps running; we stop waiting. The worker's `finally`
            # block in `_runner` will emit the actual elapsed duration when the
            # work eventually completes — so we record only the submitter-side
            # timeout event here, with the wall-clock elapsed at the moment of
            # giving up. This avoids double-counting and keeps the histogram
            # honest about *when the submitter abandoned the wait*.
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

    async def aclose(self) -> None:
        """Stop accepting new tasks and wait for in-flight work to drain.

        Calling ``aclose`` more than once is safe — the underlying
        ``ThreadPoolExecutor.shutdown(wait=True)`` is idempotent and the
        ``_closed`` flag is monotonic.
        """
        self._closed = True
        # `shutdown(wait=True)` is blocking; run it on the default executor so
        # the event loop stays responsive while workers finish.
        #
        # Caveat for multi-pool deployments: every BoundedExecutor that closes
        # concurrently submits a blocking shutdown call to the *same* default
        # executor (an unconfigured ThreadPoolExecutor with size
        # `min(32, cpu_count + 4)`). If the number of named pools approaches
        # that bound, shutdowns serialize behind it and graceful drain time
        # grows linearly with pool count. Production deployments with many
        # pools should stagger close() calls or supply a dedicated executor
        # for shutdown work.
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._executor.shutdown, True)


def _runner(
    ctx: contextvars.Context,
    fn: Callable[[], R],
    pool: BoundedExecutor,
) -> R:
    """
    Runs inside the worker thread. Restores context, instruments duration.

    This is intentionally a module-level free function (rather than a method
    on ``BoundedExecutor``) so that ``loop.run_in_executor`` can dispatch it
    without tripping over instance-bound-method serialization edge cases.
    The ``pool`` argument carries the executor reference explicitly; the
    private-attribute access (``pool._counters`` etc.) is the cost of that
    decoupling and is contained to this single function.
    """
    with pool._counters:
        pool._active += 1
        depth = max(0, pool._submitted - pool._active)
        executor_queue_depth.labels(pool=pool._name).set(depth)
    executor_active_workers.labels(pool=pool._name).inc()

    started = time.perf_counter()
    outcome = "ok"
    try:
        return ctx.run(fn)
    except Exception:
        outcome = "error"
        raise
    finally:
        elapsed = time.perf_counter() - started
        # Duration is observed with the *actual* elapsed time, even if the
        # submitter has already given up via timeout. The submitter recorded
        # its own observation under the `submitter_timeout` outcome label; this
        # one records what the worker actually did. Together they tell the
        # full story of timed-out-but-still-running work.
        executor_task_duration.labels(pool=pool._name, outcome=outcome).observe(elapsed)
        with pool._counters:
            pool._active -= 1
            pool._submitted -= 1
            depth = max(0, pool._submitted - pool._active)
            executor_queue_depth.labels(pool=pool._name).set(depth)
        executor_active_workers.labels(pool=pool._name).dec()
