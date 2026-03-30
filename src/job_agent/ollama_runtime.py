from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import statistics
import subprocess
import time
from typing import Any

import httpx

from .config import Settings
from .models import OllamaRunSummary, OllamaTuningProfile


_RUNTIME_LOCKS: dict[str, asyncio.Lock] = {}
_REQUEST_SEMAPHORES: dict[int, asyncio.Semaphore] = {}
_QUALITY_EVENT_TYPES = {
    "lead_refinement_outcome",
    "lead_refinement_rejection",
    "drafting_outcome",
    "drafting_template_fallback",
    "drafting_openai_fallback",
    "ollama_degraded_skip",
}


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
    return settings.ollama_event_log_path


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


def _read_ollama_events(settings: Settings) -> list[dict[str, Any]]:
    path = _event_log_path(settings)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    entries.append(payload)
    except Exception:
        return []
    return entries


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    position = max(0.0, min(1.0, percentile)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    if lower == upper:
        return round(ordered[lower], 3)
    fraction = position - lower
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(interpolated, 3)


def _current_profile_from_settings(settings: Settings) -> OllamaTuningProfile:
    return OllamaTuningProfile(
        model=settings.ollama_model,
        keep_alive=settings.ollama_keep_alive,
        num_ctx=settings.ollama_num_ctx,
        num_batch=settings.ollama_num_batch,
        num_predict=settings.ollama_num_predict,
        degraded=settings.ollama_degraded_for_run,
        degraded_reason=settings.ollama_degraded_reason,
    )


def load_ollama_tuning_profile(settings: Settings) -> OllamaTuningProfile:
    path = settings.ollama_tuning_profile_path
    if not path.exists():
        return _current_profile_from_settings(settings)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _current_profile_from_settings(settings)
    try:
        return OllamaTuningProfile.model_validate(payload)
    except Exception:
        return _current_profile_from_settings(settings)


def save_ollama_tuning_profile(settings: Settings, profile: OllamaTuningProfile) -> None:
    path = settings.ollama_tuning_profile_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.model_dump(mode="json"), indent=2), encoding="utf-8")


def _recent_request_events(settings: Settings, *, limit: int = 10) -> list[dict[str, Any]]:
    entries = _read_ollama_events(settings)
    filtered = [
        entry
        for entry in entries
        if entry.get("event") in {"request_success", "request_failure", "request_outer_timeout"}
        and entry.get("caller") != "manual_probe"
        and "probe" not in str(entry.get("prompt_category") or "").lower()
    ]
    return filtered[-max(1, limit) :]


def _event_timestamp(entry: dict[str, Any]) -> datetime | None:
    raw_value = entry.get("timestamp")
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def _recent_request_history_is_stale(events: list[dict[str, Any]], *, max_age: timedelta = timedelta(hours=2)) -> bool:
    if not events:
        return False
    latest_event_at = max((timestamp for timestamp in (_event_timestamp(entry) for entry in events) if timestamp), default=None)
    if latest_event_at is None:
        return False
    return datetime.now(UTC) - latest_event_at > max_age


def _warm_success_median_seconds(events: list[dict[str, Any]]) -> float | None:
    warm_successes = [
        float(entry["wall_duration_seconds"])
        for entry in events
        if entry.get("event") == "request_success"
        and entry.get("cold_start") is False
        and isinstance(entry.get("wall_duration_seconds"), (int, float))
    ]
    if len(warm_successes) < 5:
        return None
    return round(statistics.median(warm_successes[-5:]), 3)


def _available_ollama_model_names(settings: Settings) -> set[str]:
    tags_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(tags_url, timeout=2.0)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return set()
    models = payload.get("models")
    if not isinstance(models, list):
        return set()
    names: set[str] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        for key in ("name", "model"):
            value = str(item.get(key) or "").strip()
            if value:
                names.add(value)
    return names


def _probe_ollama_profile_sync(settings: Settings, profile: OllamaTuningProfile) -> tuple[bool, str | None, float]:
    timeout_seconds = max(15.0, min(45.0, settings.ollama_timeout_seconds / 2))
    payload = {
        "model": profile.model,
        "prompt": "Reply with OK only.",
        "stream": False,
        "keep_alive": profile.keep_alive,
        "options": {
            "temperature": 0,
            "num_ctx": min(profile.num_ctx, 256),
            "num_batch": 1,
            "num_predict": min(profile.num_predict, 32),
        },
    }
    started_at = time.monotonic()
    try:
        _ensure_ollama_server_sync(settings, force_restart=False)
        response = httpx.post(f"{settings.ollama_base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        raw_text = str(response.json().get("response") or "").strip().lower()
        if not raw_text.startswith("ok"):
            raise RuntimeError("probe returned an unexpected response")
        return True, None, round(time.monotonic() - started_at, 3)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", round(time.monotonic() - started_at, 3)


def _recovery_probe_candidates(
    settings: Settings,
    profile: OllamaTuningProfile,
    available_models: set[str],
) -> list[OllamaTuningProfile]:
    candidates: list[OllamaTuningProfile] = []
    seen_models: set[str] = set()

    def maybe_add(model: str, *, prefer_ctx: int, prefer_predict: int) -> None:
        if model in seen_models:
            return
        if available_models and model not in available_models:
            return
        candidates.append(
            profile.model_copy(
                update={
                    "model": model,
                    "num_ctx": min(profile.num_ctx, prefer_ctx),
                    "num_batch": 1,
                    "num_predict": min(profile.num_predict, prefer_predict),
                    "degraded": False,
                    "degraded_reason": None,
                }
            )
        )
        seen_models.add(model)

    maybe_add(profile.model, prefer_ctx=512, prefer_predict=128)
    maybe_add(settings.ollama_degraded_model, prefer_ctx=512, prefer_predict=128)
    maybe_add(settings.ollama_model, prefer_ctx=768, prefer_predict=128)
    return candidates


def _step_down_profile(profile: OllamaTuningProfile) -> tuple[OllamaTuningProfile, str]:
    if profile.num_batch > 1:
        next_batch = max(1, profile.num_batch // 2)
        return profile.model_copy(update={"num_batch": next_batch}), f"Reduced num_batch to {next_batch}."
    if profile.num_predict > 128:
        next_predict = max(128, profile.num_predict // 2)
        return profile.model_copy(update={"num_predict": next_predict}), f"Reduced num_predict to {next_predict}."
    if profile.num_ctx > 512:
        next_ctx = max(512, profile.num_ctx // 2)
        return profile.model_copy(update={"num_ctx": next_ctx}), f"Reduced num_ctx to {next_ctx}."
    return profile, ""


def auto_tune_ollama_settings(settings: Settings, *, run_id: str | None = None) -> tuple[Settings, OllamaTuningProfile]:
    if settings.llm_provider != "ollama" or not settings.ollama_enable_auto_tune:
        profile = _current_profile_from_settings(settings)
        return settings, profile

    profile = load_ollama_tuning_profile(settings)
    recent_events = _recent_request_events(settings, limit=10)
    if _recent_request_history_is_stale(recent_events):
        recent_events = []
    available_models = _available_ollama_model_names(settings)
    update_reason: str | None = None
    single_model_mode = settings.ollama_degraded_model == settings.ollama_model

    if single_model_mode and (
        profile.model != settings.ollama_model
        or (profile.degraded and not recent_events)
    ):
        profile = profile.model_copy(
            update={
                "model": settings.ollama_model,
                "keep_alive": settings.ollama_keep_alive,
                "num_ctx": settings.ollama_num_ctx,
                "num_batch": settings.ollama_num_batch,
                "num_predict": settings.ollama_num_predict,
                "degraded": False,
                "degraded_reason": None,
            }
        )
        update_reason = f"Reset Ollama profile to single-model mode on {settings.ollama_model}."

    if profile.degraded or (available_models and profile.model not in available_models):
        for candidate in _recovery_probe_candidates(settings, profile, available_models):
            success, error_message, probe_duration = _probe_ollama_profile_sync(settings, candidate)
            record_ollama_event(
                settings,
                "probe_success" if success else "probe_failure",
                run_id=run_id,
                model=candidate.model,
                num_ctx=candidate.num_ctx,
                num_batch=candidate.num_batch,
                num_predict=candidate.num_predict,
                keep_alive=candidate.keep_alive,
                wall_duration_seconds=probe_duration,
                error_message=error_message,
            )
            if success:
                profile = candidate
                update_reason = f"Recovered Ollama via readiness probe using {candidate.model}."
                break

    request_event_count = len(recent_events)
    if request_event_count:
        failure_count = sum(1 for entry in recent_events if entry.get("event") in {"request_failure", "request_outer_timeout"})
        outer_timeout_count = sum(1 for entry in recent_events if entry.get("event") == "request_outer_timeout")
        failure_ratio = failure_count / request_event_count
        warm_median = _warm_success_median_seconds(recent_events)
        degraded_model_available = (
            not available_models or settings.ollama_degraded_model in available_models
        )

        if warm_median is not None and warm_median > 60 and profile.model != settings.ollama_degraded_model:
            if degraded_model_available:
                profile = profile.model_copy(
                    update={
                        "model": settings.ollama_degraded_model,
                        "num_ctx": min(profile.num_ctx, 1024),
                        "num_batch": min(profile.num_batch, 2),
                        "num_predict": min(profile.num_predict, 192),
                        "degraded": False,
                        "degraded_reason": None,
                    }
                )
                update_reason = (
                    f"Warm-call median was {warm_median}s over the last 5 successes; switched to {settings.ollama_degraded_model}."
                )
            else:
                update_reason = (
                    f"Warm-call median was {warm_median}s, but configured degraded model "
                    f"{settings.ollama_degraded_model} is not installed; keeping {profile.model}."
                )
        elif failure_ratio > 0.2 or outer_timeout_count >= 2:
            stepped_profile, step_reason = _step_down_profile(profile)
            if step_reason:
                profile = stepped_profile
                update_reason = (
                    f"Failure ratio {round(failure_ratio, 3)} over the last {request_event_count} requests triggered tuning. "
                    f"{step_reason}"
                )
            elif profile.model != settings.ollama_degraded_model and degraded_model_available:
                profile = profile.model_copy(
                    update={
                        "model": settings.ollama_degraded_model,
                        "num_ctx": min(profile.num_ctx, 1024),
                        "num_batch": min(profile.num_batch, 2),
                        "num_predict": min(profile.num_predict, 192),
                        "degraded": False,
                        "degraded_reason": None,
                    }
                )
                update_reason = (
                    f"Failure ratio {round(failure_ratio, 3)} persisted after knob reductions; switched to {settings.ollama_degraded_model}."
                )
            else:
                degraded_reason = (
                    f"Ollama stayed unstable on {profile.model} after auto-tuning; optional Ollama steps will be skipped for this run."
                )
                if profile.model != settings.ollama_degraded_model and not degraded_model_available:
                    degraded_reason = (
                        f"Ollama stayed unstable on {profile.model}, and configured degraded model "
                        f"{settings.ollama_degraded_model} is not installed; optional Ollama steps will be skipped for this run."
                    )
                profile = profile.model_copy(
                    update={
                        "degraded": True,
                        "degraded_reason": degraded_reason,
                    }
                )
                update_reason = profile.degraded_reason

    if update_reason:
        record_ollama_event(
            settings,
            "auto_tune_update",
            run_id=run_id,
            model=profile.model,
            num_ctx=profile.num_ctx,
            num_batch=profile.num_batch,
            num_predict=profile.num_predict,
            keep_alive=profile.keep_alive,
            degraded=profile.degraded,
            degraded_reason=profile.degraded_reason,
            reason=update_reason,
        )

    profile = profile.model_copy(
        update={
            "last_updated_at": datetime.now(UTC),
            "based_on_event_count": len(_read_ollama_events(settings)),
        }
    )
    save_ollama_tuning_profile(settings, profile)
    tuned_settings = replace(
        settings,
        ollama_model=profile.model,
        ollama_keep_alive=profile.keep_alive,
        ollama_num_ctx=profile.num_ctx,
        ollama_num_batch=profile.num_batch,
        ollama_num_predict=profile.num_predict,
        ollama_degraded_for_run=profile.degraded,
        ollama_degraded_reason=profile.degraded_reason,
    )
    return tuned_settings, profile


def build_ollama_run_summary(
    settings: Settings,
    *,
    run_id: str,
    tuning_profile: OllamaTuningProfile | None = None,
    generated_at: datetime | None = None,
) -> OllamaRunSummary:
    entries = [entry for entry in _read_ollama_events(settings) if entry.get("run_id") == run_id]
    request_entries = [
        entry
        for entry in entries
        if entry.get("event") in {"request_success", "request_failure", "request_outer_timeout"}
    ]
    success_entries = [entry for entry in request_entries if entry.get("event") == "request_success"]
    failure_entries = [entry for entry in request_entries if entry.get("event") != "request_success"]
    wall_times = [
        float(entry["wall_duration_seconds"])
        for entry in success_entries
        if isinstance(entry.get("wall_duration_seconds"), (int, float))
    ]
    warm_wall_times = [
        float(entry["wall_duration_seconds"])
        for entry in success_entries
        if entry.get("cold_start") is False and isinstance(entry.get("wall_duration_seconds"), (int, float))
    ]
    failure_breakdown = Counter(
        str(entry.get("error_type") or entry.get("event") or "unknown")
        for entry in failure_entries
    )
    caller_breakdown = Counter(str(entry.get("caller") or "unknown") for entry in request_entries)
    prompt_category_breakdown = Counter(str(entry.get("prompt_category") or "unknown") for entry in request_entries)
    quality_counters: Counter[str] = Counter()
    for entry in entries:
        if entry.get("event") not in _QUALITY_EVENT_TYPES:
            continue
        for key, value in entry.items():
            if key in {"timestamp", "event", "run_id", "caller", "prompt_category", "prompt_hash"}:
                continue
            if isinstance(value, bool):
                quality_counters[key] += int(value)
            elif isinstance(value, (int, float)):
                quality_counters[key] += value
    request_count = len(request_entries)
    summary = OllamaRunSummary(
        run_id=run_id,
        generated_at=generated_at or datetime.now(UTC),
        tuning_profile=tuning_profile or load_ollama_tuning_profile(settings),
        request_count=request_count,
        success_count=len(success_entries),
        failure_count=len(failure_entries),
        outer_timeout_count=sum(1 for entry in request_entries if entry.get("event") == "request_outer_timeout"),
        success_rate=round(len(success_entries) / request_count, 3) if request_count else 0.0,
        warm_hit_rate=round(len(warm_wall_times) / len(success_entries), 3) if success_entries else 0.0,
        median_wall_duration_seconds=round(statistics.median(wall_times), 3) if wall_times else None,
        p95_wall_duration_seconds=_quantile(wall_times, 0.95),
        median_warm_wall_duration_seconds=round(statistics.median(warm_wall_times), 3) if warm_wall_times else None,
        p95_warm_wall_duration_seconds=_quantile(warm_wall_times, 0.95),
        failure_breakdown=dict(failure_breakdown),
        caller_breakdown=dict(caller_breakdown),
        prompt_category_breakdown=dict(prompt_category_breakdown),
        quality_counters={key: round(float(value), 3) for key, value in quality_counters.items()},
    )
    return summary


def save_ollama_run_summary(settings: Settings, summary: OllamaRunSummary) -> Path:
    path = settings.ollama_summary_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def load_latest_ollama_summary(settings: Settings) -> OllamaRunSummary | None:
    path = settings.ollama_summary_path
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        return OllamaRunSummary.model_validate(payload)
    except Exception:
        return None


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
