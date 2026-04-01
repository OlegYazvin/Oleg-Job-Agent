from pathlib import Path

import pytest

from job_agent.config import load_settings


def _write_env(path: Path, content: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".env").write_text(content, encoding="utf-8")


def test_load_settings_allows_ollama_without_openai_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("USE_OPENAI_FALLBACK", "false")
    _write_env(
        tmp_path,
        "\n".join(
            [
                "OPENAI_API_KEY=",
                "LLM_PROVIDER=ollama",
                "USE_OPENAI_FALLBACK=false",
            ]
        ),
    )
    settings = load_settings(tmp_path)
    assert settings.llm_provider == "ollama"
    assert settings.use_openai_fallback is False


def test_load_settings_requires_openai_key_when_openai_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("USE_OPENAI_FALLBACK", "false")
    _write_env(
        tmp_path,
        "\n".join(
            [
                "OPENAI_API_KEY=",
                "LLM_PROVIDER=openai",
            ]
        ),
    )
    with pytest.raises(ValueError):
        load_settings(tmp_path)


def test_load_settings_requires_openai_key_when_fallback_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("USE_OPENAI_FALLBACK", "true")
    _write_env(
        tmp_path,
        "\n".join(
            [
                "OPENAI_API_KEY=",
                "LLM_PROVIDER=ollama",
                "USE_OPENAI_FALLBACK=true",
            ]
        ),
    )
    with pytest.raises(ValueError):
        load_settings(tmp_path)


def test_load_settings_resolves_firefox_extension_profile_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("USE_OPENAI_FALLBACK", "false")
    _write_env(
        tmp_path,
        "\n".join(
            [
                "OPENAI_API_KEY=",
                "LLM_PROVIDER=ollama",
                "USE_OPENAI_FALLBACK=false",
                "FIREFOX_EXTENSION_PROFILE_DIR=profiles/default-release",
            ]
        ),
    )

    settings = load_settings(tmp_path)

    assert settings.firefox_extension_profile_dir == tmp_path / "profiles/default-release"


def test_load_settings_prefers_project_env_over_inherited_firefox_profile_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("USE_OPENAI_FALLBACK", "false")
    monkeypatch.setenv("FIREFOX_EXTENSION_PROFILE_DIR", "/tmp/inherited-profile")
    _write_env(
        tmp_path,
        "\n".join(
            [
                "OPENAI_API_KEY=",
                "LLM_PROVIDER=ollama",
                "USE_OPENAI_FALLBACK=false",
                "FIREFOX_EXTENSION_PROFILE_DIR=profiles/default-release",
            ]
        ),
    )

    settings = load_settings(tmp_path)

    assert settings.firefox_extension_profile_dir == tmp_path / "profiles/default-release"
