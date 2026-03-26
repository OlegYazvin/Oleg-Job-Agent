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


def _add_linkedin_cookies(driver: webdriver.Firefox) -> None:
    from .linkedin import _load_firefox_linkedin_cookies

    cookies = _load_firefox_linkedin_cookies()
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
            driver.add_cookie(payload)
        except Exception:
            continue


def _build_driver(settings: Settings) -> webdriver.Firefox:
    profile_dir = host_profile_dir(settings.project_root)
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
    _add_linkedin_cookies(driver)
    driver.set_page_load_timeout(20)
    return driver


def _prime_driver(driver: webdriver.Firefox) -> None:
    try:
        driver.get("https://www.linkedin.com/feed/")
    except Exception:
        try:
            driver.get("https://www.linkedin.com/")
        except Exception:
            pass


def run_host() -> None:
    settings = load_settings(require_openai=False)
    root = host_root(settings.project_root)
    root.mkdir(parents=True, exist_ok=True)
    log_path = host_log_path(settings.project_root)
    command_queue_path = host_command_queue_path(settings.project_root)
    if command_queue_path.exists():
        command_queue_path.unlink()

    driver = _build_driver(settings)
    _write_state_file(
        settings.project_root,
        {
            "pid": os.getpid(),
            "profile_dir": str(host_profile_dir(settings.project_root)),
            "state_path": str(host_state_path(settings.project_root)),
            "log_path": str(log_path),
            "bridge_capture_mode": settings.linkedin_capture_mode,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )
    _prime_driver(driver)

    should_stop = False

    def handle_signal(signum, frame) -> None:  # type: ignore[unused-ignore]
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while not should_stop:
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
                        driver.switch_to.new_window("tab")
                        driver.get(url)
                    except Exception:
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
    state = read_state(settings.project_root)
    if state is not None:
        return {"already_running": True, **state}

    log_path = host_log_path(settings.project_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "job_agent.firefox_extension_host", "run"],
            cwd=str(settings.project_root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    time.sleep(3)
    state = read_state(settings.project_root)
    return {
        "started": True,
        "pid": process.pid,
        "log_path": str(log_path),
        **(state or {}),
    }


def read_state(project_root: Path) -> dict[str, object] | None:
    path = host_state_path(project_root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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
