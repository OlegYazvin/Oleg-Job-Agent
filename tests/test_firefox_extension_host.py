import json
from pathlib import Path
from types import SimpleNamespace

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


def test_read_state_returns_none_and_clears_stale_pid(monkeypatch, tmp_path: Path) -> None:
    state_path = host_module.host_state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"pid": 4242, "profile_dir": str(tmp_path / "profile")}), encoding="utf-8")
    monkeypatch.setattr(host_module, "_pid_is_running", lambda pid: False)

    assert host_module.read_state(tmp_path) is None
    assert not state_path.exists()


def test_read_state_returns_payload_for_live_pid(monkeypatch, tmp_path: Path) -> None:
    state_path = host_module.host_state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"pid": 4242, "profile_dir": str(tmp_path / "profile")}), encoding="utf-8")
    monkeypatch.setattr(host_module, "_pid_is_running", lambda pid: True)

    payload = host_module.read_state(tmp_path)

    assert payload is not None
    assert payload["pid"] == 4242


def test_add_linkedin_cookies_omits_http_only_field(monkeypatch) -> None:
    added_payloads: list[dict[str, object]] = []

    class _FakeDriver:
        def get(self, url: str) -> None:
            return None

        def add_cookie(self, payload: dict[str, object]) -> None:
            added_payloads.append(payload)

    monkeypatch.setattr(
        "job_agent.linkedin._load_linkedin_auth_cookies",
        lambda settings: [
            {
                "name": "li_at",
                "value": "abc123",
                "domain": ".linkedin.com",
                "path": "/",
                "expires": 2000000000,
                "secure": True,
                "httpOnly": True,
                "sameSite": "Lax",
            }
        ],
    )

    host_module._add_linkedin_cookies(_FakeDriver(), object())  # type: ignore[arg-type]

    assert len(added_payloads) == 1
    assert added_payloads[0]["name"] == "li_at"
    assert added_payloads[0]["sameSite"] == "Lax"
    assert "httpOnly" not in added_payloads[0]


def test_firefox_extension_is_installed_in_profile_checks_extensions_json(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "extensions.json").write_text(
        json.dumps(
            {
                "addons": [
                    {
                        "id": host_module.FIREFOX_EXTENSION_ID,
                        "active": True,
                        "visible": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert host_module.firefox_extension_is_installed_in_profile(profile_dir) is True


def test_firefox_extension_is_installed_in_profile_accepts_live_temporary_addon(
    monkeypatch, tmp_path: Path
) -> None:
    profile_dir = tmp_path / "profile"
    (profile_dir / "weave").mkdir(parents=True)
    (profile_dir / "prefs.js").write_text(
        '\n'.join(
            [
                'user_pref("devtools.aboutdebugging.tmpExtDirPath", "/tmp/firefox_extension");',
                f'user_pref("extensions.webextensions.uuids", "{{\\"{host_module.FIREFOX_EXTENSION_ID}\\":\\"uuid-123\\"}}");',
            ]
        ),
        encoding="utf-8",
    )
    (profile_dir / "weave" / "addonsreconciler.json").write_text(
        json.dumps(
            {
                "addons": {
                    host_module.FIREFOX_EXTENSION_ID: {
                        "enabled": True,
                        "installed": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(host_module, "_firefox_process_running_for_profile", lambda path: True)

    assert host_module.firefox_extension_is_installed_in_profile(profile_dir) is True


def test_inspect_configured_firefox_extension_profile_reports_ready(monkeypatch, tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    monkeypatch.setattr(host_module, "firefox_extension_is_installed_in_profile", lambda profile: True)
    monkeypatch.setattr(host_module, "firefox_profile_has_linkedin_auth", lambda profile: True)
    monkeypatch.setattr(host_module, "_firefox_process_running_for_profile", lambda profile: True)
    monkeypatch.setattr(host_module, "_temporary_extension_marker_present", lambda profile: True)

    payload = host_module.inspect_configured_firefox_extension_profile(
        SimpleNamespace(firefox_extension_profile_dir=profile_dir)
    )

    assert payload["configured"] is True
    assert payload["ready"] is True
    assert payload["path"] == str(profile_dir.resolve())
    assert payload["process_running"] is True
    assert payload["temporary_extension_loaded"] is True


def test_open_url_in_firefox_profile_prefers_live_running_instance_for_temporary_addon(
    monkeypatch, tmp_path: Path
) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    launched: list[list[str]] = []

    monkeypatch.setattr(host_module.shutil, "which", lambda binary: "/usr/bin/firefox" if binary == "firefox" else None)
    monkeypatch.setattr(host_module, "_default_firefox_binary", lambda: "/usr/bin/firefox")
    monkeypatch.setattr(host_module, "_firefox_process_running_for_profile", lambda profile: True)
    monkeypatch.setattr(host_module, "_temporary_extension_marker_present", lambda profile: True)
    monkeypatch.setattr(
        host_module.subprocess,
        "Popen",
        lambda command, stdout=None, stderr=None: launched.append(command) or SimpleNamespace(),
    )

    opened = host_module.open_url_in_firefox_profile(profile_dir, "https://www.linkedin.com/feed/")

    assert opened is True
    assert launched[0] == ["/usr/bin/firefox", "--new-tab", "https://www.linkedin.com/feed/"]


def test_deploy_host_background_prefers_configured_profile(monkeypatch, tmp_path: Path) -> None:
    profile_dir = tmp_path / "real-profile"
    profile_dir.mkdir()
    monkeypatch.setattr(
        host_module,
        "load_settings",
        lambda require_openai=False: SimpleNamespace(project_root=tmp_path, firefox_extension_profile_dir=profile_dir),
    )
    monkeypatch.setattr(
        host_module,
        "inspect_configured_firefox_extension_profile",
        lambda settings: {
            "configured": True,
            "path": str(profile_dir),
            "exists": True,
            "extension_installed": True,
            "linkedin_authenticated": True,
            "ready": True,
            "extension_id": host_module.FIREFOX_EXTENSION_ID,
        },
    )
    opened: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        host_module,
        "open_url_in_firefox_profile",
        lambda profile, url: opened.append((profile, url)) or True,
    )

    payload = host_module.deploy_host_background()

    assert payload["mode"] == "configured_profile"
    assert payload["started"] is True
    assert opened == [(profile_dir, "https://www.linkedin.com/feed/")]
