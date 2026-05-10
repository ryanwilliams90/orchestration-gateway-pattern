from __future__ import annotations

from pathlib import Path

import pytest

from gateway.secrets import (
    SecretMissing,
    Secrets,
    SecretSourceUnavailable,
    load_secrets,
)


def test_loads_from_kubernetes_style_mount(tmp_path: Path) -> None:
    (tmp_path / "api_token").write_text("s3cret")
    (tmp_path / "model_id").write_text("anthropic.claude-3\n")  # trailing newline

    secrets = load_secrets(mount_path=tmp_path)
    assert secrets.require("api_token") == "s3cret"
    assert secrets.require("model_id") == "anthropic.claude-3"


def test_loads_from_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOO_BAR", "1")
    monkeypatch.setenv("FOO_BAZ_QUX", "2")
    monkeypatch.setenv("OTHER_VAR", "ignored")

    secrets = load_secrets(env_prefix="FOO")
    assert secrets.require("bar") == "1"
    assert secrets.require("baz_qux") == "2"
    with pytest.raises(SecretMissing):
        secrets.require("other_var")


def test_mount_overrides_env_on_collision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("X_KEY", "from-env")
    (tmp_path / "key").write_text("from-mount")

    secrets = load_secrets(mount_path=tmp_path, env_prefix="X")
    assert secrets.require("key") == "from-mount"


def test_no_sources_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    # No env vars matching, no mount path.
    monkeypatch.delenv("GATEWAY_TEST_KEY", raising=False)
    with pytest.raises(SecretSourceUnavailable):
        load_secrets(env_prefix="DEFINITELY_NOT_SET")


def test_secrets_are_immutable() -> None:
    s = Secrets(values={"a": "b"})
    with pytest.raises(AttributeError):
        s.values = {"c": "d"}  # type: ignore[misc]
