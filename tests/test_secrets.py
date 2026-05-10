"""
Tests for the lifespan-scoped secret loader.

These pin the documented contracts:

- Loads from a Kubernetes-style mount path (one file per key, contents are
  the value, trailing newlines stripped).
- Loads from environment variables matching a prefix, lowercasing the
  remaining portion to form keys.
- Mount-path source wins on key collision with env source.
- Loader refuses to start when no source produced anything (the design
  goal: "running but misconfigured" is structurally impossible).
- Skips Kubernetes atomic-update dotfiles (`..data`, `..2024_01_01_*`)
  that appear in mounts during secret rotation.
- Raises `SecretSourceUnavailable` when a mount path is configured but
  doesn't exist or isn't a directory.
- `Secrets.require()` raises `SecretMissing` for absent keys; `Secrets.get()`
  returns the supplied default.
- `Secrets` instances are immutable at the dataclass level.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.secrets import (
    SecretMissing,
    Secrets,
    SecretSourceUnavailable,
    load_secrets,
)

# ----- Mount-path loading --------------------------------------------------


def test_loads_from_kubernetes_style_mount(tmp_path: Path) -> None:
    (tmp_path / "api_token").write_text("s3cret")
    (tmp_path / "model_id").write_text("anthropic.claude-3\n")  # trailing newline

    secrets = load_secrets(mount_path=tmp_path)
    assert secrets.require("api_token") == "s3cret"
    assert secrets.require("model_id") == "anthropic.claude-3"


def test_skips_kubernetes_atomic_update_dotfiles(tmp_path: Path) -> None:
    """
    Kubernetes secret mounts use a `..data` symlink and timestamped
    directories (`..2024_01_01_12_00_00.000000`) during rotation. These
    must not be picked up as keys; if they were, the loader would expose
    `..data` as a "secret" containing every secret concatenated.

    This is the contract that prevents secret rotation from mid-flight
    corrupting the loaded view.
    """
    # Real keys.
    (tmp_path / "api_token").write_text("s3cret")
    (tmp_path / "model_id").write_text("claude-3")

    # Kubernetes rotation artifacts — must be ignored.
    (tmp_path / "..data").write_text("rotation symlink target")
    (tmp_path / "..2024_01_01_12_00_00").write_text("rotation directory marker")

    secrets = load_secrets(mount_path=tmp_path)
    assert set(secrets.values.keys()) == {"api_token", "model_id"}


def test_mount_path_must_exist(tmp_path: Path) -> None:
    """A configured-but-missing mount path is a deploy-time misconfig, not a no-op."""
    nonexistent = tmp_path / "does-not-exist"
    with pytest.raises(SecretSourceUnavailable, match="does not exist"):
        load_secrets(mount_path=nonexistent)


def test_mount_path_must_be_a_directory(tmp_path: Path) -> None:
    """A file (not a directory) at the mount path is also a misconfig."""
    file_not_dir = tmp_path / "not-a-dir"
    file_not_dir.write_text("oops")
    with pytest.raises(SecretSourceUnavailable):
        load_secrets(mount_path=file_not_dir)


# ----- Env-var loading -----------------------------------------------------


def test_loads_from_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOO_BAR", "1")
    monkeypatch.setenv("FOO_BAZ_QUX", "2")
    monkeypatch.setenv("OTHER_VAR", "ignored")

    secrets = load_secrets(env_prefix="FOO")
    assert secrets.require("bar") == "1"
    assert secrets.require("baz_qux") == "2"
    with pytest.raises(SecretMissing):
        secrets.require("other_var")


def test_env_prefix_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Common typo: lowercase prefix passed in should still match uppercase env."""
    monkeypatch.setenv("MIXED_TOKEN", "value")
    secrets = load_secrets(env_prefix="mixed")
    assert secrets.require("token") == "value"


# ----- Source precedence and refusal ---------------------------------------


def test_mount_overrides_env_on_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_KEY", "from-env")
    (tmp_path / "key").write_text("from-mount")

    secrets = load_secrets(mount_path=tmp_path, env_prefix="X")
    assert secrets.require("key") == "from-mount"


def test_no_sources_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Empty resolution at lifespan startup is a hard failure. A handler
    cannot reach for a secret that wasn't loaded; the design goal is for
    misconfiguration to fail the readiness probe, not the request path.
    """
    monkeypatch.delenv("GATEWAY_TEST_KEY", raising=False)
    with pytest.raises(SecretSourceUnavailable):
        load_secrets(env_prefix="DEFINITELY_NOT_SET")


# ----- Secrets accessor contract -------------------------------------------


def test_require_raises_secret_missing_on_absent_key() -> None:
    """`require()` is the strict accessor; missing keys raise the typed exception."""
    s = Secrets(values={"present": "yes"})
    with pytest.raises(SecretMissing, match="absent"):
        s.require("absent")


def test_get_returns_default_on_absent_key() -> None:
    """`get()` is the soft accessor; the default is returned, not raised."""
    s = Secrets(values={"present": "yes"})
    assert s.get("absent") is None
    assert s.get("absent", "fallback") == "fallback"
    assert s.get("present") == "yes"


def test_secrets_dataclass_is_frozen() -> None:
    """`Secrets` is a frozen dataclass; reassigning `values` raises."""
    s = Secrets(values={"a": "b"})
    with pytest.raises(AttributeError):
        s.values = {"c": "d"}  # type: ignore[misc]
