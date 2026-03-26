from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

import httpx

from .config import Settings


_RUNTIME_LOCKS: dict[str, asyncio.Lock] = {}
_REQUEST_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _runtime_lock(settings: Settings) -> asyncio.Lock:
    key = f"{settings.project_root}:{settings.ollama_base_url}"
    lock = _RUNTIME_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _RUNTIME_LOCKS[key] = lock
    return lock


def get_ollama_request_semaphore(limit: int) -> asyncio.Semaphore:
    normalized_limit = max(1, int(limit))
    semaphore = _REQUEST_SEMAPHORES.get(normalized_limit)
    if semaphore is None:
        semaphore = asyncio.Semaphore(normalized_limit)
        _REQUEST_SEMAPHORES[normalized_limit] = semaphore
    return semaphore


def _pid_path(settings: Settings) -> Path:
    return settings.ollama_runtime_dir / "serve.pid"


def _event_log_path(settings: Settings) -> Path:
    return settings.output_dir / "ollama-events.jsonl"


def _version_url(settings: Settings) -> str:
    return f"{settings.ollama_base_url.rstrip('/')}/api/version"


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _managed_pid(settings: Settings) -> int | None:
    path = _pid_path(settings)
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None
    return pid if pid > 0 else None


def _write_managed_pid(settings: Settings, pid: int) -> None:
    settings.ollama_runtime_dir.mkdir(parents=True, exist_ok=True)
    _pid_path(settings).write_text(str(pid), encoding="utf-8")


def _tail_file_lines(path: Path, *, limit: int = 20) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return list(deque((line.rstrip() for line in handle), maxlen=max(1, limit)))
    except Exception:
        return []


def _filter_diagnostic_lines(output: str, *, pid: int | None = None, limit: int = 20) -> list[str]:
    if not output.strip():
        return []
    pid_text = str(pid) if pid is not None else ""
    matched: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if (
            "ollama" in lowered
            or "oom" in lowered
            or "out of memory" in lowered
            or "killed process" in lowered
            or (pid_text and pid_text in line)
        ):
            matched.append(line)
    return matched[-limit:]


def _run_diagnostic_command(command: list[str], *, pid: int | None = None) -> list[str]:
    executable = shutil.which(command[0])
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return [f"{command[0]} probe failed: {type(exc).__name__}: {exc}"]

    lines = _filter_diagnostic_lines(result.stdout, pid=pid)
    if lines:
        return lines

    stderr = result.stderr.strip()
    if stderr:
        return [stderr.splitlines()[-1]]
    return []


def _linux_kill_diagnostics(settings: Settings, *, pid: int | None = None) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    ollama_log_tail = _tail_file_lines(settings.ollama_log_path, limit=25)
    if ollama_log_tail:
        diagnostics["ollama_log_tail"] = ollama_log_tail

    journal_lines = _run_diagnostic_command(["journalctl", "--user", "-u", "ollama.service", "-n", "50", "--no-pager"], pid=pid)
    if journal_lines:
        diagnostics["journalctl"] = journal_lines

    dmesg_lines = _run_diagnostic_command(["dmesg", "--ctime", "--color=never"], pid=pid)
    if dmesg_lines:
        diagnostics["dmesg"] = dmesg_lines

    return diagnostics


def record_ollama_event(settings: Settings, event_type: str, **details: object) -> None:
    payload = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event_type,
        **details,
    }
    try:
        path = _event_log_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, default=str))
            handle.write("\n")
    except Exception:
        return


def _clear_managed_pid(settings: Settings) -> None:
    path = _pid_path(settings)
    if path.exists():
        path.unlink()


def _is_ollama_healthy_sync(settings: Settings) -> bool:
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(_version_url(settings))
        return response.is_success
    except Exception:
        return False


def _wait_for_ollama_health_sync(settings: Settings, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        if _is_ollama_healthy_sync(settings):
            return True
        time.sleep(1)
    return _is_ollama_healthy_sync(settings)


def _stop_managed_ollama_sync(settings: Settings) -> None:
    pid = _managed_pid(settings)
    if pid is None:
        return
    if not _pid_is_alive(pid):
        record_ollama_event(
            settings,
            "managed_process_already_dead",
            pid=pid,
            diagnostics=_linux_kill_diagnostics(settings, pid=pid),
        )
        _clear_managed_pid(settings)
        return
    try:
        os.kill(pid, signal.SIGTERM)
        record_ollama_event(settings, "managed_process_stop_requested", pid=pid, signal="SIGTERM")
    except ProcessLookupError:
        _clear_managed_pid(settings)
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _pid_is_alive(pid):
            _clear_managed_pid(settings)
            return
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
        record_ollama_event(settings, "managed_process_force_killed", pid=pid, signal="SIGKILL")
    except ProcessLookupError:
        pass
    _clear_managed_pid(settings)


def _try_manage_with_systemd(settings: Settings, *, restart: bool) -> bool:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False
    command = [systemctl, "--user", "restart" if restart else "start", "ollama.service"]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _start_managed_ollama_sync(settings: Settings) -> None:
    executable_parts = shlex.split(settings.ollama_command.strip()) if settings.ollama_command.strip() else ["ollama"]
    if not executable_parts:
        executable_parts = ["ollama"]
    command = [*executable_parts, "serve"]
    settings.ollama_log_path.parent.mkdir(parents=True, exist_ok=True)
    with settings.ollama_log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    _write_managed_pid(settings, process.pid)
    record_ollama_event(settings, "managed_process_started", pid=process.pid, command=command)


def _ensure_ollama_server_sync(settings: Settings, *, force_restart: bool) -> None:
    settings.ollama_runtime_dir.mkdir(parents=True, exist_ok=True)
    if not force_restart and _is_ollama_healthy_sync(settings):
        return

    managed_pid = _managed_pid(settings)
    if managed_pid is not None and not _pid_is_alive(managed_pid):
        record_ollama_event(
            settings,
            "managed_process_missing",
            pid=managed_pid,
            diagnostics=_linux_kill_diagnostics(settings, pid=managed_pid),
        )
        _clear_managed_pid(settings)
        managed_pid = None

    if force_restart:
        record_ollama_event(settings, "restart_requested", pid=managed_pid)
        _stop_managed_ollama_sync(settings)

    used_systemd = _try_manage_with_systemd(settings, restart=force_restart)
    if used_systemd:
        record_ollama_event(settings, "systemd_control_requested", action="restart" if force_restart else "start")
    if not used_systemd and not _is_ollama_healthy_sync(settings):
        managed_pid = _managed_pid(settings)
        if managed_pid is None or not _pid_is_alive(managed_pid):
            _start_managed_ollama_sync(settings)

    if not _wait_for_ollama_health_sync(settings, settings.ollama_start_timeout_seconds):
        record_ollama_event(
            settings,
            "health_check_timeout",
            version_url=_version_url(settings),
            start_timeout_seconds=settings.ollama_start_timeout_seconds,
            diagnostics=_linux_kill_diagnostics(settings, pid=_managed_pid(settings)),
        )
        raise RuntimeError(
            f"Timed out waiting for Ollama to become healthy at {_version_url(settings)} "
            f"after {settings.ollama_start_timeout_seconds} seconds."
        )


async def ensure_ollama_server(settings: Settings, *, force_restart: bool = False) -> None:
    async with _runtime_lock(settings):
        await asyncio.to_thread(_ensure_ollama_server_sync, settings, force_restart=force_restart)
