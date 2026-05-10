# Security

This repository is a reference implementation and architecture artifact, not a production system.

The code demonstrates an async/sync executor-boundary pattern with admission control, ContextVar propagation, and structured observability. It is not intended to be deployed as-is for production traffic. Real deployments need credential rotation tied to lifespan, hardened secret loading, durable rejection metrics, request-level auth, and many other concerns that are out of scope here.

If you find a security issue in the demonstration code itself — a path that bypasses admission control, a credential handling bug, a logic flaw in claims the prototype makes about its boundary properties — please report it privately to ryan90@gmail.com.

## Scope

In scope:

- Admission-control bypasses (paths that reach the executor without going through `BoundedExecutor.submit`).
- Credentials or secrets leaking into logs, traces, or response bodies.
- Verification mismatches between what the gateway claims about itself and what the code does.

Out of scope:

- The `FakeProvider` and `WorkflowRuntime` stub.
- Missing production features (credential rotation, hardened secret store, request-level auth) — documented in the README as out-of-scope for a pattern reference.
