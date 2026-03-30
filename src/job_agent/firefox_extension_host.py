from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.request import urlopen
import zipfile

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

from .config import Settings, load_settings


GECKODRIVER_VERSION = "0.36.0"
HOST_HEARTBEAT_INTERVAL_SECONDS = 5.0
HOST_STARTUP_TIMEOUT_SECONDS = 20.0
HOST_DRIVER_START_RETRIES = 3
FIREFOX_EXTENSION_ID = "job-agent-linkedin-bridge@local"


def host_root(project_root: Path) -> Path:
    return project_root / ".secrets" / "firefox-extension-host"


def host_state_path(project_root: Path) -> Path:
    return host_root(project_root) / "state.json"


def host_profile_dir(project_root: Path) -> Path:
    return host_root(project_root) / "profile"


def host_log_path(project_root: Path) -> Path:
    return project_root / "output" / "firefox_extension_host.log"


def host_command_queue_path(project_root: Path) -> Path:
    return host_root(project_root) / "commands.jsonl"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _geckodriver_binary_path(project_root: Path) -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return project_root / ".tools" / f"geckodriver{suffix}"


def _geckodriver_download_url() -> str:
    machine = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if machine in {"aarch64", "arm64"}:
            arch = "linux-aarch64"
        elif machine in {"x86_64", "amd64"}:
            arch = "linux64"
        else:
            raise RuntimeError(f"Unsupported machine architecture for geckodriver auto-download: {machine}")
        extension = "tar.gz"
    elif sys.platform == "darwin":
        if machine in {"aarch64", "arm64"}:
            arch = "macos-aarch64"
        elif machine in {"x86_64", "amd64"}:
            arch = "macos"
        else:
            raise RuntimeError(f"Unsupported machine architecture for geckodriver auto-download: {machine}")
        extension = "tar.gz"
    elif sys.platform.startswith("win"):
        if machine not in {"x86_64", "amd64"}:
            raise RuntimeError(f"Unsupported Windows architecture for geckodriver auto-download: {machine}")
        arch = "win64"
        extension = "zip"
    else:
        raise RuntimeError(f"Automatic geckodriver download is not implemented for platform: {sys.platform}")
    return (
        "https://github.com/mozilla/geckodriver/releases/download/"
        f"v{GECKODRIVER_VERSION}/geckodriver-v{GECKODRIVER_VERSION}-{arch}.{extension}"
    )


def ensure_geckodriver(project_root: Path) -> Path:
    binary_path = _geckodriver_binary_path(project_root)
    if binary_path.exists():
        return binary_path

    binary_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="job-agent-geckodriver-") as temp_root:
        archive_name = "geckodriver.zip" if sys.platform.startswith("win") else "geckodriver.tar.gz"
        archive_path = Path(temp_root) / archive_name
        with urlopen(_geckodriver_download_url()) as response:
            archive_path.write_bytes(response.read())
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(binary_path.parent)
        else:
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(binary_path.parent)
    binary_path.chmod(0o755)
    return binary_path


def _default_firefox_binary() -> str | None:
    candidates = [shutil.which("firefox"), shutil.which("firefox-esr")]
    if sys.platform == "darwin":
        candidates.extend(
            [
                "/Applications/Firefox.app/Contents/MacOS/firefox",
                str(Path.home() / "Applications/Firefox.app/Contents/MacOS/firefox"),
            ]
        )
    elif sys.platform.startswith("win"):
        candidates.extend(
            [
                r"C:\Program Files\Mozilla Firefox\firefox.exe",
                r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
            ]
        )
    else:
        candidates.append("/usr/bin/firefox")

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _extensions_json_path(profile_dir: Path) -> Path:
    return profile_dir / "extensions.json"


def _prefs_js_path(profile_dir: Path) -> Path:
    return profile_dir / "prefs.js"


def _addons_reconciler_path(profile_dir: Path) -> Path:
    return profile_dir / "weave" / "addonsreconciler.json"


def _firefox_process_running_for_profile(profile_dir: Path) -> bool:
    try:
        output = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    profile_arg = str(profile_dir)
    for line in output.splitlines():
        candidate = line.strip()
        if "firefox" not in candidate.lower():
            continue
        if profile_arg in candidate:
            return True
    return False


def _temporary_extension_marker_present(profile_dir: Path) -> bool:
    prefs_js_path = _prefs_js_path(profile_dir)
    if prefs_js_path.exists():
        try:
            prefs_payload = prefs_js_path.read_text(encoding="utf-8")
        except OSError:
            prefs_payload = ""
        if FIREFOX_EXTENSION_ID in prefs_payload and "devtools.aboutdebugging.tmpExtDirPath" in prefs_payload:
            return True

    addons_reconciler_path = _addons_reconciler_path(profile_dir)
    if not addons_reconciler_path.exists():
        return False
    try:
        payload = json.loads(addons_reconciler_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    addon = payload.get("addons", {}).get(FIREFOX_EXTENSION_ID)
    if not isinstance(addon, dict):
        return False
    return bool(addon.get("installed", False) and addon.get("enabled", False))


def firefox_extension_is_installed_in_profile(profile_dir: Path | None) -> bool:
    if profile_dir is None or not profile_dir.exists():
        return False

    extensions_dir = profile_dir / "extensions"
    direct_candidates = [
        extensions_dir / FIREFOX_EXTENSION_ID,
        extensions_dir / f"{FIREFOX_EXTENSION_ID}.xpi",
        profile_dir / "browser-extension-data" / FIREFOX_EXTENSION_ID,
    ]
    if any(candidate.exists() for candidate in direct_candidates):
        return True

    extensions_json_path = _extensions_json_path(profile_dir)
    if extensions_json_path.exists():
        try:
            payload = json.loads(extensions_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for addon in payload.get("addons", []):
            if str(addon.get("id") or "").strip() != FIREFOX_EXTENSION_ID:
                continue
            if addon.get("visible", True) is False:
                continue
            if addon.get("active", True) is False and not addon.get("temporarilyInstalled", False):
                continue
            return True
    return _firefox_process_running_for_profile(profile_dir) and _temporary_extension_marker_present(profile_dir)


def firefox_profile_has_linkedin_auth(profile_dir: Path | None) -> bool:
    if profile_dir is None or not profile_dir.exists():
        return False
    from .linkedin import _load_firefox_linkedin_cookies, linkedin_cookies_authenticate

    try:
        cookies = _load_firefox_linkedin_cookies(profile_dir)
    except Exception:
        return False
    cookie_names = {str(cookie.get("name") or "") for cookie in cookies}
    if "li_at" not in cookie_names or "JSESSIONID" not in cookie_names:
        return False
    return linkedin_cookies_authenticate(cookies)


def inspect_firefox_extension_profile(profile_dir: Path | None) -> dict[str, object]:
    resolved_profile_dir = profile_dir.expanduser().resolve() if profile_dir is not None else None
    exists = bool(resolved_profile_dir and resolved_profile_dir.exists())
    process_running = bool(resolved_profile_dir and _firefox_process_running_for_profile(resolved_profile_dir))
    temporary_extension_loaded = bool(
        resolved_profile_dir and process_running and _temporary_extension_marker_present(resolved_profile_dir)
    )
    extension_installed = firefox_extension_is_installed_in_profile(resolved_profile_dir)
    linkedin_authenticated = firefox_profile_has_linkedin_auth(resolved_profile_dir)
    return {
        "path": str(resolved_profile_dir) if resolved_profile_dir is not None else None,
        "exists": exists,
        "process_running": process_running,
        "temporary_extension_loaded": temporary_extension_loaded,
        "extension_installed": extension_installed,
        "linkedin_authenticated": linkedin_authenticated,
        "ready": bool(exists and extension_installed and linkedin_authenticated),
        "extension_id": FIREFOX_EXTENSION_ID,
    }


def inspect_configured_firefox_extension_profile(settings: Settings) -> dict[str, object]:
    summary = inspect_firefox_extension_profile(settings.firefox_extension_profile_dir)
    from .linkedin import linkedin_storage_state_is_authenticated

    storage_state_path = getattr(settings, "linkedin_storage_state", None)
    summary["storage_state_authenticated"] = (
        linkedin_storage_state_is_authenticated(storage_state_path)
        if isinstance(storage_state_path, Path)
        else False
    )
    summary["configured"] = settings.firefox_extension_profile_dir is not None
    return summary


def _sync_configured_profile_from_storage_state(settings: Settings) -> bool:
    from .linkedin import sync_linkedin_cookies_to_firefox_profile

    profile_dir = settings.firefox_extension_profile_dir
    if profile_dir is None:
        return False
    resolved_profile = profile_dir.expanduser().resolve()
    if not resolved_profile.exists():
        return False
    return sync_linkedin_cookies_to_firefox_profile(resolved_profile, settings.linkedin_storage_state)


def _resolve_extension_host_profile(settings: Settings) -> tuple[Path, bool]:
    configured_profile = settings.firefox_extension_profile_dir
    if configured_profile is not None:
        resolved_profile = configured_profile.expanduser().resolve()
        if (
            resolved_profile.exists()
            and firefox_profile_has_linkedin_auth(resolved_profile)
            and not _firefox_process_running_for_profile(resolved_profile)
        ):
            return resolved_profile, True
    profile_dir = host_profile_dir(settings.project_root)
    return profile_dir, False


def open_url_in_firefox_profile(profile_dir: Path, url: str) -> bool:
    resolved_profile_dir = profile_dir.expanduser().resolve()
    firefox_binary = shutil.which("firefox") or _default_firefox_binary()
    firefox_esr_binary = shutil.which("firefox-esr")
    flatpak_binary = shutil.which("flatpak")
    prefer_live_instance = (
        _firefox_process_running_for_profile(resolved_profile_dir)
        and _temporary_extension_marker_present(resolved_profile_dir)
    )

    commands: list[list[str]] = []
    if prefer_live_instance:
        if firefox_binary:
            commands.append([firefox_binary, "--new-tab", url])
        if firefox_esr_binary:
            commands.append([firefox_esr_binary, "--new-tab", url])
        if flatpak_binary:
            commands.append([flatpak_binary, "run", "org.mozilla.firefox", "--new-tab", url])
    if firefox_binary:
        commands.append([firefox_binary, "--profile", str(resolved_profile_dir), "--new-tab", url])
    if firefox_esr_binary:
        commands.append([firefox_esr_binary, "--profile", str(resolved_profile_dir), "--new-tab", url])
    if flatpak_binary:
        commands.append(
            [
                flatpak_binary,
                "run",
                "org.mozilla.firefox",
                "--profile",
                str(resolved_profile_dir),
                "--new-tab",
                url,
            ]
        )

    for command in commands:
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except OSError:
            continue
    return False


def _write_state_file(project_root: Path, payload: dict[str, object]) -> None:
    path = host_state_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_state_file(project_root: Path) -> None:
    path = host_state_path(project_root)
    if path.exists():
        path.unlink()


def enqueue_open_url(project_root: Path, url: str) -> None:
    queue_path = host_command_queue_path(project_root)
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"action": "open_url", "url": url}) + "\n")


def _add_linkedin_cookies(driver: webdriver.Firefox, settings: Settings) -> None:
    from .linkedin import _load_linkedin_auth_cookies

    cookies = _load_linkedin_auth_cookies(settings)
    if not cookies:
        return
    driver.get("https://www.linkedin.com/")
    for cookie in cookies:
        try:
            payload = {
                "name": str(cookie["name"]),
                "value": str(cookie["value"]),
                "path": str(cookie.get("path") or "/"),
                "secure": bool(cookie.get("secure", False)),
            }
            domain = str(cookie.get("domain") or "").lstrip(".")
            if domain:
                payload["domain"] = domain
            expires = int(cookie.get("expires") or -1)
            if expires > 0:
                payload["expiry"] = expires
            same_site = cookie.get("sameSite")
            if same_site in {"Lax", "Strict", "None"}:
                payload["sameSite"] = same_site
            driver.add_cookie(payload)
        except Exception:
            continue


def _build_driver_with_retries(
    settings: Settings,
    *,
    profile_dir: Path,
    attempts: int = HOST_DRIVER_START_RETRIES,
) -> webdriver.Firefox:
    last_error: Exception | None = None
    for attempt_number in range(1, max(1, attempts) + 1):
        try:
            return _build_driver(settings, profile_dir=profile_dir)
        except Exception as exc:
            last_error = exc
            print(
                f"[firefox-extension-host] driver startup attempt {attempt_number}/{attempts} failed: {exc}",
                flush=True,
            )
            if attempt_number < attempts:
                time.sleep(min(5, attempt_number * 2))
    assert last_error is not None
    raise last_error


def _restart_driver(driver: webdriver.Firefox | None, settings: Settings, *, profile_dir: Path) -> webdriver.Firefox:
    if driver is not None:
        try:
            driver.quit()
        except Exception:
            pass
    rebuilt_driver = _build_driver_with_retries(settings, profile_dir=profile_dir)
    _prime_driver(rebuilt_driver, settings)
    return rebuilt_driver


def _tail_log_excerpt(path: Path, *, lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def _build_driver(settings: Settings, *, profile_dir: Path) -> webdriver.Firefox:
    profile_dir.mkdir(parents=True, exist_ok=True)

    options = Options()
    firefox_binary = _default_firefox_binary()
    if firefox_binary is None:
        raise RuntimeError(
            "Firefox executable was not found. Install Firefox or set it on PATH before deploying the extension host."
        )
    options.binary_location = firefox_binary
    options.add_argument("-profile")
    options.add_argument(str(profile_dir))
    options.add_argument("--new-instance")

    service = Service(str(ensure_geckodriver(settings.project_root)))
    driver = webdriver.Firefox(service=service, options=options)
    driver.install_addon(str((settings.project_root / "firefox_extension").resolve()), temporary=True)
    _add_linkedin_cookies(driver, settings)
    driver.set_page_load_timeout(20)
    return driver


def _linkedin_session_is_authenticated(driver: webdriver.Firefox) -> bool:
    try:
        current_url = str(driver.current_url or "")
    except Exception:
        return False
    if "linkedin.com/uas/login" in current_url or "linkedin.com/login" in current_url:
        return False
    return "linkedin.com" in current_url


def _prime_driver(driver: webdriver.Firefox, settings: Settings) -> bool:
    try:
        driver.get("https://www.linkedin.com/feed/")
    except Exception:
        try:
            driver.get("https://www.linkedin.com/")
        except Exception:
            return False
    if _linkedin_session_is_authenticated(driver):
        return True
    _add_linkedin_cookies(driver, settings)
    try:
        driver.get("https://www.linkedin.com/feed/")
    except Exception:
        try:
            driver.get("https://www.linkedin.com/")
        except Exception:
            return False
    return _linkedin_session_is_authenticated(driver)


def run_host() -> None:
    settings = load_settings(require_openai=False)
    root = host_root(settings.project_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = host_log_path(settings.project_root)
    command_queue_path = host_command_queue_path(settings.project_root)
    profile_dir, using_configured_profile = _resolve_extension_host_profile(settings)
    if command_queue_path.exists():
        command_queue_path.unlink()

    state_payload = {
        "pid": os.getpid(),
        "profile_dir": str(profile_dir),
        "using_configured_profile": using_configured_profile,
        "state_path": str(host_state_path(settings.project_root)),
        "log_path": str(log_path),
        "bridge_capture_mode": settings.linkedin_capture_mode,
        "deployed_at": _utc_timestamp(),
        "status": "starting",
        "last_heartbeat_at": _utc_timestamp(),
        "restart_count": 0,
    }
    _write_state_file(settings.project_root, state_payload)

    driver = _build_driver_with_retries(settings, profile_dir=profile_dir)
    linked_in_authenticated = _prime_driver(driver, settings)
    state_payload["linkedin_authenticated"] = linked_in_authenticated
    state_payload["status"] = "running" if linked_in_authenticated else "login_required"
    if not linked_in_authenticated:
        state_payload["last_error"] = "LinkedIn session is not authenticated in the Firefox extension host profile."
    state_payload["ready_at"] = _utc_timestamp()
    state_payload["last_heartbeat_at"] = _utc_timestamp()
    _write_state_file(settings.project_root, state_payload)

    should_stop = False
    next_heartbeat_at = time.monotonic() + HOST_HEARTBEAT_INTERVAL_SECONDS

    def handle_signal(signum, frame) -> None:  # type: ignore[unused-ignore]
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while not should_stop:
            if time.monotonic() >= next_heartbeat_at:
                state_payload["last_heartbeat_at"] = _utc_timestamp()
                _write_state_file(settings.project_root, state_payload)
                next_heartbeat_at = time.monotonic() + HOST_HEARTBEAT_INTERVAL_SECONDS
            if command_queue_path.exists():
                commands = [line.strip() for line in command_queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                command_queue_path.write_text("", encoding="utf-8")
                for raw_command in commands:
                    try:
                        payload = json.loads(raw_command)
                    except json.JSONDecodeError:
                        continue
                    if payload.get("action") != "open_url":
                        continue
                    url = str(payload.get("url") or "").strip()
                    if not url:
                        continue
                    try:
                        linked_in_authenticated = _prime_driver(driver, settings)
                        state_payload["linkedin_authenticated"] = linked_in_authenticated
                        if not linked_in_authenticated:
                            state_payload["status"] = "login_required"
                            state_payload["last_error"] = (
                                "LinkedIn session is not authenticated in the Firefox extension host profile."
                            )
                            state_payload["last_heartbeat_at"] = _utc_timestamp()
                            _write_state_file(settings.project_root, state_payload)
                            continue
                        state_payload["status"] = "running"
                        state_payload.pop("last_error", None)
                        driver.switch_to.new_window("tab")
                        driver.get(url)
                        state_payload["last_command_url"] = url
                        state_payload["last_command_at"] = _utc_timestamp()
                        state_payload["last_heartbeat_at"] = _utc_timestamp()
                        _write_state_file(settings.project_root, state_payload)
                    except Exception as exc:
                        print(f"[firefox-extension-host] open_url failed for {url}: {exc}", flush=True)
                        state_payload["status"] = "restarting"
                        state_payload["last_error"] = str(exc)[:300]
                        state_payload["last_restart_at"] = _utc_timestamp()
                        state_payload["restart_count"] = int(state_payload.get("restart_count") or 0) + 1
                        _write_state_file(settings.project_root, state_payload)
                        try:
                            driver = _restart_driver(driver, settings, profile_dir=profile_dir)
                            linked_in_authenticated = _prime_driver(driver, settings)
                            state_payload["linkedin_authenticated"] = linked_in_authenticated
                            if not linked_in_authenticated:
                                state_payload["status"] = "login_required"
                                state_payload["last_error"] = (
                                    "LinkedIn session is not authenticated in the Firefox extension host profile."
                                )
                                _write_state_file(settings.project_root, state_payload)
                                continue
                            driver.switch_to.new_window("tab")
                            driver.get(url)
                            state_payload["status"] = "running"
                            state_payload.pop("last_error", None)
                            state_payload["last_command_url"] = url
                            state_payload["last_command_at"] = _utc_timestamp()
                            state_payload["last_heartbeat_at"] = _utc_timestamp()
                            _write_state_file(settings.project_root, state_payload)
                        except Exception as restart_exc:
                            print(
                                f"[firefox-extension-host] driver restart failed while opening {url}: {restart_exc}",
                                flush=True,
                            )
                            state_payload["status"] = "degraded"
                            state_payload["last_error"] = str(restart_exc)[:300]
                            _write_state_file(settings.project_root, state_payload)
                            continue
            time.sleep(1)
    finally:
        try:
            driver.quit()
        finally:
            _clear_state_file(settings.project_root)
            if command_queue_path.exists():
                command_queue_path.unlink()


def deploy_host_background() -> dict[str, object]:
    settings = load_settings(require_openai=False)
    configured_profile = inspect_configured_firefox_extension_profile(settings)
    if configured_profile.get("configured"):
        extension_installed = bool(configured_profile.get("extension_installed"))
        linkedin_authenticated = bool(configured_profile.get("linkedin_authenticated"))
        storage_state_authenticated = bool(configured_profile.get("storage_state_authenticated"))
        process_running = bool(configured_profile.get("process_running"))
        if not linkedin_authenticated and storage_state_authenticated and not process_running:
            if _sync_configured_profile_from_storage_state(settings):
                configured_profile = inspect_configured_firefox_extension_profile(settings)
                extension_installed = bool(configured_profile.get("extension_installed"))
                linkedin_authenticated = bool(configured_profile.get("linkedin_authenticated"))
                storage_state_authenticated = bool(configured_profile.get("storage_state_authenticated"))
        launch_urls: list[str]
        if not extension_installed:
            if linkedin_authenticated and not process_running:
                host_payload = deploy_dedicated_host_background(settings)
                host_state = host_payload.get("state") if isinstance(host_payload.get("state"), dict) else host_payload
                if bool(host_payload.get("started") or host_payload.get("already_running")) and isinstance(host_state, dict):
                    if bool(host_state.get("linkedin_authenticated")):
                        return {
                            **host_payload,
                            "mode": "configured_profile_host",
                            "profile": configured_profile,
                            "next_action": "Configured Firefox profile is now running under the extension host.",
                        }
                launch_urls = ["about:debugging#/runtime/this-firefox"]
                next_action = (
                    "Automatic extension-host startup against the configured Firefox profile failed. "
                    "Open that profile and use 'Load Temporary Add-on...' on firefox_extension/manifest.json."
                )
            else:
                launch_urls = []
                if not linkedin_authenticated:
                    launch_urls.append("https://www.linkedin.com/feed/")
                launch_urls.append("about:debugging#/runtime/this-firefox")
                next_action = (
                    "Open the real Firefox profile, sign into LinkedIn, and use "
                    "'Load Temporary Add-on...' on firefox_extension/manifest.json."
                )
        elif not linkedin_authenticated:
            launch_urls = ["https://www.linkedin.com/feed/"]
            if storage_state_authenticated and not process_running:
                next_action = (
                    "Saved LinkedIn storage state looks valid, but the configured Firefox profile still did not "
                    "validate after cookie sync. Open LinkedIn in that profile and refresh the session."
                )
            else:
                next_action = "Sign into LinkedIn in the configured Firefox profile."
        else:
            launch_urls = ["https://www.linkedin.com/feed/"]
            next_action = "Configured Firefox profile is ready for extension capture."
        launch_started = False
        profile_path = settings.firefox_extension_profile_dir
        if profile_path is not None:
            for launch_url in launch_urls:
                if open_url_in_firefox_profile(profile_path, launch_url):
                    launch_started = True
        return {
            "started": launch_started,
            "mode": "configured_profile",
            "launch_url": launch_urls[0],
            "launch_urls": launch_urls,
            "next_action": next_action,
            "profile": configured_profile,
        }

    return deploy_dedicated_host_background(settings)


def deploy_dedicated_host_background(settings: Settings | None = None) -> dict[str, object]:
    settings = settings or load_settings(require_openai=False)

    state = read_state(settings.project_root)
    if state is not None:
        return {"already_running": True, **state}

    log_path = host_log_path(settings.project_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== deploy_firefox_extension {time.strftime('%Y-%m-%dT%H:%M:%S%z')} ===\n")
        process = subprocess.Popen(
            [sys.executable, "-m", "job_agent.firefox_extension_host", "run"],
            cwd=str(settings.project_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + HOST_STARTUP_TIMEOUT_SECONDS
    state = None
    while time.monotonic() < deadline:
        state = read_state(settings.project_root)
        if state is not None and state.get("ready_at"):
            return {
                "started": True,
                "pid": process.pid,
                "log_path": str(log_path),
                **state,
            }
        if process.poll() is not None:
            break
        time.sleep(0.5)
    return {
        "started": False,
        "pid": process.pid,
        "log_path": str(log_path),
        "exit_code": process.poll(),
        "state": state,
        "log_tail": _tail_log_excerpt(log_path),
    }


def read_state(project_root: Path) -> dict[str, object] | None:
    path = host_state_path(project_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    try:
        pid = int(payload.get("pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid and not _pid_is_running(pid):
        _clear_state_file(project_root)
        return None
    return payload


def remove_host(delete_profile: bool = True) -> dict[str, object]:
    settings = load_settings(require_openai=False)
    state = read_state(settings.project_root)
    if state is not None:
        pid = int(state.get("pid") or 0)
        if pid > 0:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        _clear_state_file(settings.project_root)

    if delete_profile:
        shutil.rmtree(host_root(settings.project_root), ignore_errors=True)
    return {
        "removed": True,
        "profile_dir": str(host_profile_dir(settings.project_root)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the dedicated Firefox extension host.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("deploy")
    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("--keep-profile", action="store_true")
    subparsers.add_parser("status")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        run_host()
    elif args.command == "deploy":
        print(json.dumps(deploy_host_background(), indent=2))
    elif args.command == "remove":
        print(json.dumps(remove_host(delete_profile=not args.keep_profile), indent=2))
    elif args.command == "status":
        settings = load_settings(require_openai=False)
        print(json.dumps(read_state(settings.project_root) or {"running": False}, indent=2))
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
