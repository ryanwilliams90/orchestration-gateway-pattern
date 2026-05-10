"""
Tests for the workflow runtime stub.

The stub is intentionally narrow — the boundary, not the workflow logic, is
the point of this module. The tests pin two contracts:

- WorkflowRequest validates non-empty fields at construction. This makes
  defensive validation layered (API + runtime) rather than relying on the
  caller to validate.
- WorkflowRuntime.run() round-trips the provider response into a
  WorkflowResult shape, including provider name forwarding.
"""

from __future__ import annotations

import pytest

from gateway.provider import FakeProvider, ProviderWrapper
from gateway.runtime import WorkflowRequest, WorkflowResult, WorkflowRuntime


def test_workflow_request_rejects_empty_project() -> None:
    with pytest.raises(ValueError, match="project"):
        WorkflowRequest(project="", model="m", prompt="p")


def test_workflow_request_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        WorkflowRequest(project="demo", model="", prompt="p")


def test_workflow_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        WorkflowRequest(project="demo", model="m", prompt="")


def test_workflow_request_is_frozen() -> None:
    req = WorkflowRequest(project="demo", model="m", prompt="p")
    with pytest.raises(AttributeError):
        req.project = "other"  # type: ignore[misc]


def test_runtime_run_returns_normalized_result() -> None:
    """Round-trip the provider response into a WorkflowResult."""
    runtime = WorkflowRuntime(ProviderWrapper(FakeProvider(latency_seconds=0.0)))
    result = runtime.run(WorkflowRequest(project="demo", model="m", prompt="hello"))
    assert isinstance(result, WorkflowResult)
    assert result.project == "demo"
    assert result.model == "m"
    assert result.provider == "fake"
    assert result.output.startswith("echo[m]")


def test_runtime_forwards_provider_name() -> None:
    """
    Whatever provider name the wrapper exposes is what shows up on the
    result. This pins the audit-trail property: a downstream consumer
    can always tell which provider produced a given result.
    """

    class CustomProvider:
        name = "anthropic-bedrock"

        def invoke(self, model: str, prompt: str) -> object:  # type: ignore[override]
            from gateway.provider import ProviderResponse

            return ProviderResponse(
                text="custom-output",
                model=model,
                input_tokens=1,
                output_tokens=1,
                latency_seconds=0.0,
            )

    runtime = WorkflowRuntime(ProviderWrapper(CustomProvider()))  # type: ignore[arg-type]
    result = runtime.run(WorkflowRequest(project="demo", model="m", prompt="hi"))
    assert result.provider == "anthropic-bedrock"
