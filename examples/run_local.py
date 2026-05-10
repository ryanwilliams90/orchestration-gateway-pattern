"""
Run the gateway locally with the fake provider.

    GATEWAY_FAKE_KEY=anything python examples/run_local.py

Then in another shell:

    curl -s localhost:8080/healthz
    curl -s localhost:8080/metrics | head
    curl -s -X POST localhost:8080/v1/run \\
        -H 'content-type: application/json' \\
        -d '{"project":"demo","model":"claude-test","prompt":"hello"}'
"""

from __future__ import annotations

import uvicorn

from gateway.app import GatewayConfig, configure_logging, create_app


def main() -> None:
    configure_logging()
    app = create_app(
        config=GatewayConfig(
            pool_name="workflows",
            pool_size=4,
            workflow_timeout_seconds=30.0,
            secret_env_prefix="GATEWAY_FAKE",
        ),
    )
    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104  (intentional)


if __name__ == "__main__":
    main()
