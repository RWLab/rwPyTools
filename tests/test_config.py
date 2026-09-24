"""Configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rwpytools import ClientConfig
from rwpytools.errors import ConfigError


def test_from_env_uses_explicit_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RWPYTOOLS_API_KEY", "env-key")
    monkeypatch.setenv("RWPYTOOLS_API_BASE_URL", "https://from-env.example.com")
    cfg = ClientConfig.from_env(cache_dir=tmp_path)
    assert cfg.api_key == "env-key"
    assert cfg.api_base_url == "https://from-env.example.com"
    assert cfg.cache_dir == tmp_path


def test_from_env_explicit_arg_wins(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RWPYTOOLS_API_KEY", "env-key")
    cfg = ClientConfig.from_env(api_key="arg-key", cache_dir=tmp_path)
    assert cfg.api_key == "arg-key"


def test_missing_api_key_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RWPYTOOLS_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        ClientConfig.from_env(cache_dir=tmp_path)


def test_invalid_url_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ClientConfig(
            api_base_url="not-absolute",
            api_key="k",
            cache_dir=tmp_path,
        )


def test_negative_cache_max_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ClientConfig(
            api_base_url="https://x.test",
            api_key="k",
            cache_dir=tmp_path,
            cache_max_bytes=-1,
        )


def test_expires_in_bounds(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ClientConfig(
            api_base_url="https://x.test",
            api_key="k",
            cache_dir=tmp_path,
            default_expires_in=0,
        )
    with pytest.raises(ConfigError):
        ClientConfig(
            api_base_url="https://x.test",
            api_key="k",
            cache_dir=tmp_path,
            default_expires_in=24 * 3600 + 1,
        )


def test_int_env_invalid_value(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RWPYTOOLS_API_KEY", "k")
    monkeypatch.setenv("RWPYTOOLS_CACHE_MAX_BYTES", "not-a-number")
    with pytest.raises(ConfigError):
        ClientConfig.from_env(cache_dir=tmp_path)
