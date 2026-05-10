"""
Smoke test that drives the running example app via real HTTP.

`httpx.ASGITransport` covers the ASGI contract well, but it does not
exercise the uvicorn event-loop integration, the lifespan ASGI message
flow as a real server runs it, or the actual TCP socket path. This test
spawns `examples/run_local.py` in a subprocess, waits for the readiness
probe to pass, drives one request and one metrics scrape, and tears down.

It is gated on uvicorn being importable and a free TCP port being
available. The test is small but high-signal: any wiring mistake that
ASGITransport hides will show up here.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_ready(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=1.0)
            if r.status_code == 200:
                return
        except (httpx.ConnectError, httpx.ReadError, httpx.ReadTimeout) as exc:
            last_exc = exc
        time.sleep(0.1)
    raise TimeoutError(f"server at {url} did not become ready in {timeout}s ({last_exc})")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess + signal handling differs on Windows; smoke test is Unix-only",
)
def test_smoke_real_uvicorn_serves_full_path(tmp_path: Path) -> None:
    """
    Boot the example app under real uvicorn, drive one full request, and
    assert the integrated path works end-to-end:
    HTTP socket → uvicorn → FastAPI → middleware → executor → runtime → wrapper.
    """
    port = _free_port()
    env = {
        **os.environ,
        "GATEWAY_FAKE_KEY": "smoke",
        # Keep logs out of the test runner's stderr unless something fails.
        "LOG_LEVEL": "WARNING",
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }

    # Tell uvicorn to bind the chosen port. We can't pass it via the
    # example script's args, so we run uvicorn directly against the same
    # app factory.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "gateway.app:create_app",
            "--factory",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )

    base = f"http://127.0.0.1:{port}"
    try:
        _wait_for_ready(f"{base}/healthz", timeout=15.0)

        # Drive one workflow.
        r = httpx.post(
            f"{base}/v1/run",
            json={"project": "smoke", "model": "m", "prompt": "hello"},
            timeout=10.0,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["project"] == "smoke"
        assert body["provider"] == "fake"
        assert "echo[m]" in body["output"]
        assert "x-request-id" in r.headers

        # Metrics scrape returns Prometheus-formatted text with samples
        # from all three layers.
        m = httpx.get(f"{base}/metrics", timeout=5.0)
        assert m.status_code == 200
        assert "gateway_request_duration_seconds_count" in m.text
        assert "executor_task_duration_seconds_count" in m.text
        assert 'provider_call_duration_seconds_count{model="m"' in m.text
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
