from pathlib import Path

import job_agent.firefox_extension_host as host_module


def test_geckodriver_download_url_supports_macos_arm(monkeypatch) -> None:
    monkeypatch.setattr(host_module.sys, "platform", "darwin", raising=False)
    monkeypatch.setattr(host_module.platform, "machine", lambda: "arm64")

    url = host_module._geckodriver_download_url()

    assert "macos-aarch64" in url
    assert url.endswith(".tar.gz")


def test_geckodriver_download_url_supports_windows(monkeypatch) -> None:
    monkeypatch.setattr(host_module.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(host_module.platform, "machine", lambda: "AMD64")

    url = host_module._geckodriver_download_url()

    assert "win64" in url
    assert url.endswith(".zip")


def test_geckodriver_binary_path_uses_exe_on_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(host_module.sys, "platform", "win32", raising=False)
    path = host_module._geckodriver_binary_path(tmp_path)
    assert path.name == "geckodriver.exe"
