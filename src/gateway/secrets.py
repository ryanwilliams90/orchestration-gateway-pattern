"""
Lifespan-scoped secret loading.

The case study's central claim about secrets is that they are loaded once, at
service startup, from a Kubernetes-managed source. The runtime never reaches
into the environment, the filesystem, or a secret manager on the request
path.

This module implements that contract. Two sources are supported, in order:

1. A directory of files (the standard Kubernetes mounted-secret shape).
2. An environment-variable shim, intended for local development.

If neither is present, startup fails. "Running but misconfigured" is not a
state the design allows.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Secrets:
    """Immutable bag of resolved secrets, frozen at lifespan startup."""

    values: Mapping[str, str]

    def require(self, key: str) -> str:
        try:
            return self.values[key]
        except KeyError as exc:
            raise SecretMissing(key) from exc

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.values.get(key, default)


class SecretMissing(KeyError):
    """Raised when required state is missing at startup or first-use."""


class SecretSourceUnavailable(RuntimeError):
    """Raised when no configured source produced a usable result."""


def load_secrets(
    *,
    mount_path: str | os.PathLike[str] | None = None,
    env_prefix: str | None = None,
) -> Secrets:
    """
    Load secrets at lifespan startup.

    Parameters
    ----------
    mount_path
        Directory of files where each filename is a key and contents are the
        value. Matches the Kubernetes `secret` volume mount layout.
    env_prefix
        If set, also reads `{env_prefix}_*` environment variables. Lowercased
        and stripped of the prefix to form keys. Useful for local development.

    Both sources may be supplied; the file mount wins on key collision.
    """
    values: dict[str, str] = {}

    if env_prefix:
        prefix = f"{env_prefix.upper()}_"
        for raw_key, raw_value in os.environ.items():
            if raw_key.startswith(prefix):
                key = raw_key.removeprefix(prefix).lower()
                values[key] = raw_value

    if mount_path is not None:
        path = Path(mount_path)
        if not path.is_dir():
            raise SecretSourceUnavailable(
                f"secret mount path {path!s} does not exist or is not a directory"
            )
        for entry in path.iterdir():
            # Kubernetes mounts produce dotfile temp links during atomic
            # updates; skip them rather than treating them as keys.
            if not entry.is_file() or entry.name.startswith(".."):
                continue
            values[entry.name] = entry.read_text(encoding="utf-8").rstrip("\n")

    if not values:
        raise SecretSourceUnavailable(
            "no secrets resolved from configured sources; refusing to start"
        )

    log.info("loaded %d secrets at lifespan startup", len(values))
    return Secrets(values=dict(values))
