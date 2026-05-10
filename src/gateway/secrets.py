"""
Lifespan-scoped secret loading.

Secrets are resolved once, at service startup, from a Kubernetes-managed
source. The runtime never reaches into the environment, filesystem, or a
secret manager on the request path; that's both a latency property (no
per-request variance) and a security property (one auditable code path
through which credentials enter the process).

Sources, in order:

- A directory of files — the standard Kubernetes mounted-secret shape.
- An environment-variable shim, intended for local development.

If neither produces anything, startup fails. Running but misconfigured is
not a state this allows.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Secrets:
    """Immutable bag of resolved secrets, frozen at lifespan startup.

    The ``values`` mapping is wrapped in ``MappingProxyType`` so it cannot be
    mutated through the dataclass — neither by the caller that constructed it
    nor by any code that obtained a reference to it later. The dataclass's
    ``frozen=True`` only prevents reassigning the attribute; the proxy is
    what prevents mutating the underlying dict.
    """

    values: Mapping[str, str]

    def require(self, key: str) -> str:
        try:
            return self.values[key]
        except KeyError as exc:
            raise SecretMissing(key) from exc

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.values.get(key, default)


class SecretMissing(KeyError):
    """Raised when a required key is absent."""


class SecretSourceUnavailable(RuntimeError):
    """Raised when no configured source produced a usable result."""


def load_secrets(
    *,
    mount_path: str | os.PathLike[str] | None = None,
    env_prefix: str | None = None,
) -> Secrets:
    """Resolve secrets at lifespan startup.

    ``mount_path`` is a directory where each filename is a key and contents
    are the value (the Kubernetes ``secret`` volume layout). ``env_prefix``
    matches ``{prefix}_*`` environment variables, lowercased and stripped of
    the prefix to form keys.

    Both sources may be supplied. The file mount wins on key collision.
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
            # Kubernetes mounts use ``..data`` symlinks and ``..<timestamp>``
            # directories during atomic secret rotation. Skip those so the
            # rotation artifacts aren't mistaken for secret keys.
            if not entry.is_file() or entry.name.startswith(".."):
                continue
            # ``rstrip("\r\n")`` in case files were committed with CRLF
            # endings; secrets shouldn't contain trailing whitespace anyway.
            values[entry.name] = entry.read_text(encoding="utf-8").rstrip("\r\n")

    if not values:
        raise SecretSourceUnavailable(
            "no secrets resolved from configured sources; refusing to start"
        )

    log.info("loaded %d secrets at lifespan startup", len(values))
    return Secrets(values=MappingProxyType(dict(values)))
