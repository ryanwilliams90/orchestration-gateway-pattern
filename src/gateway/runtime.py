"""
Workflow runtime stub.

In production this layer is the agent framework's supported entrypoint
(`crewai run`, equivalent). It is synchronous, long-running, and assumes a
specific CWD and initialization order. Bypassing the framework's lifecycle
produces silent drift — see the case study for specifics.

This stub stands in for that runtime in tests and the example app. It calls
through the provider wrapper so the full async-handler → executor →
synchronous-runtime → provider-wrapper path is exercised end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from gateway.provider import ProviderResponse, ProviderWrapper

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkflowRequest:
    project: str
    prompt: str
    model: str


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    project: str
    output: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


class WorkflowRuntime:
    """
    Synchronous workflow runtime. Orchestrates one or more provider calls
    behind a stable result shape.

    Intentionally narrow: the point of this module is the *boundary*, not the
    workflow logic. Real workflows live behind framework lifecycle entries.
    """

    def __init__(self, provider: ProviderWrapper) -> None:
        self._provider = provider

    def run(self, request: WorkflowRequest) -> WorkflowResult:
        log.info(
            "runtime invocation project=%s model=%s prompt_len=%d",
            request.project,
            request.model,
            len(request.prompt),
        )
        response: ProviderResponse = self._provider.complete(
            model=request.model,
            prompt=request.prompt,
        )
        return WorkflowResult(
            project=request.project,
            output=response.text,
            provider=self._provider.provider_name,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
