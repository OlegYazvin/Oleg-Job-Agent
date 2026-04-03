from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any
from uuid import uuid4

try:
    import tkinter as tk
    from tkinter import scrolledtext
except Exception:  # pragma: no cover - tkinter availability is platform-specific
    tk = None
    scrolledtext = None


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StatusReporter:
    def __init__(self, path: Path, *, event_limit: int = 60) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._events: deque[dict[str, Any]] = deque(maxlen=event_limit)
        self._state: dict[str, Any] = {
            "run_id": uuid4().hex,
            "pid": os.getpid(),
            "started_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "stage": "starting",
            "message": "Preparing workflow.",
            "done": False,
            "failed": False,
            "stale": False,
            "metrics": {},
            "recent_events": [],
        }
        self._write_state()

    @property
    def run_id(self) -> str:
        return str(self._state["run_id"])

    @property
    def started_at(self) -> str:
        return str(self._state["started_at"])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def emit(self, stage: str, message: str, **metrics: Any) -> None:
        with self._lock:
            self._state["updated_at"] = _utc_now_iso()
            self._state["stage"] = stage
            self._state["message"] = message
            if metrics:
                self._state["metrics"].update(metrics)
            self._events.append(
                {
                    "time": self._state["updated_at"],
                    "stage": stage,
                    "message": message,
                }
            )
            self._state["recent_events"] = list(self._events)
            self._write_state()

    def heartbeat(self, message: str = "Still running.") -> None:
        with self._lock:
            heartbeat_count = int(self._state["metrics"].get("heartbeat_count", 0)) + 1
        self.emit("heartbeat", message, heartbeat_count=heartbeat_count)

    def complete(self, message: str, **metrics: Any) -> None:
        with self._lock:
            self._state["done"] = True
            self._state["failed"] = False
        self.emit("completed", message, **metrics)

    def fail(self, message: str, **metrics: Any) -> None:
        with self._lock:
            self._state["done"] = True
            self._state["failed"] = True
        self.emit("failed", message, **metrics)

    def _write_state(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")


def can_launch_progress_gui() -> bool:
    if tk is None:
        return False
    return bool(os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def spawn_progress_gui(status_path: Path) -> subprocess.Popen[bytes] | None:
    if not can_launch_progress_gui():
        return None
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "job_agent.dashboard"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except Exception:
        return None


def _read_status_payload(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return {
            "stage": "waiting",
            "message": "Waiting for workflow status file.",
            "updated_at": "",
            "metrics": {},
            "recent_events": [],
            "done": False,
            "failed": False,
        }
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "stage": "reading",
            "message": "Reading workflow status.",
            "updated_at": "",
            "metrics": {},
            "recent_events": [],
            "done": False,
            "failed": False,
        }
    auto_loop_state_path = status_path.parent / "auto-loop-state.json"
    if auto_loop_state_path.exists():
        try:
            auto_loop_payload = json.loads(auto_loop_state_path.read_text(encoding="utf-8"))
        except Exception:
            auto_loop_payload = None
        if isinstance(auto_loop_payload, dict):
            payload_metrics = dict(payload.get("metrics", {}) or {})
            if auto_loop_payload.get("enabled"):
                payload_metrics.setdefault("auto_loop_enabled", True)
                if auto_loop_payload.get("status") is not None:
                    payload_metrics["auto_loop_status"] = auto_loop_payload.get("status")
                if auto_loop_payload.get("current_iteration") is not None:
                    payload_metrics["auto_loop_iteration"] = auto_loop_payload.get("current_iteration")
                if auto_loop_payload.get("completed_attempts") is not None:
                    payload_metrics["auto_loop_completed_attempts"] = auto_loop_payload.get("completed_attempts")
                if auto_loop_payload.get("target_attempts") is not None:
                    payload_metrics["auto_loop_target_attempts"] = auto_loop_payload.get("target_attempts")
                if auto_loop_payload.get("codex_session_id") is not None:
                    payload_metrics["codex_session_id"] = auto_loop_payload.get("codex_session_id")
                payload["metrics"] = payload_metrics
    if _status_payload_is_stale(payload):
        payload = dict(payload)
        payload["stage"] = "stale"
        payload["message"] = "Previous workflow appears to have exited without marking status complete."
        payload["done"] = True
        payload["failed"] = True
        payload["stale"] = True
    return payload


def _status_payload_is_stale(payload: dict[str, Any]) -> bool:
    if payload.get("done"):
        return False
    pid = payload.get("pid")
    if not isinstance(pid, int):
        return False
    return not _pid_is_alive(pid)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _format_status_stage(stage: str, payload: dict[str, Any]) -> str:
    metrics = dict(payload.get("metrics", {}) or {})
    suffix = ""
    loop_iteration = metrics.get("auto_loop_iteration")
    loop_target = metrics.get("auto_loop_target_attempts")
    if loop_iteration is not None and loop_target is not None:
        suffix = f" | Loop {loop_iteration}/{loop_target}"
    elif loop_iteration is not None:
        suffix = f" | Loop {loop_iteration}"
    status_suffix = ""
    if payload.get("done"):
        status_suffix = " (failed)" if payload.get("failed") else " (complete)"
    return f"{stage}{suffix}{status_suffix}"


def run_status_viewer(status_path: Path) -> None:  # pragma: no cover - GUI-only
    if tk is None or scrolledtext is None:
        raise RuntimeError("Tkinter is not available on this system.")

    root = tk.Tk()
    root.title("Job Agent Status")
    root.geometry("760x440")
    root.minsize(640, 360)

    stage_var = tk.StringVar(value="Starting")
    message_var = tk.StringVar(value="Preparing workflow status viewer.")
    updated_var = tk.StringVar(value="")
    metrics_var = tk.StringVar(value="")

    frame = tk.Frame(root, padx=16, pady=16)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Stage", anchor="w", font=("TkDefaultFont", 10, "bold")).pack(fill="x")
    tk.Label(frame, textvariable=stage_var, anchor="w").pack(fill="x", pady=(0, 8))

    tk.Label(frame, text="Current Update", anchor="w", font=("TkDefaultFont", 10, "bold")).pack(fill="x")
    tk.Label(frame, textvariable=message_var, anchor="w", justify="left", wraplength=700).pack(fill="x", pady=(0, 8))

    tk.Label(frame, text="Last Updated", anchor="w", font=("TkDefaultFont", 10, "bold")).pack(fill="x")
    tk.Label(frame, textvariable=updated_var, anchor="w").pack(fill="x", pady=(0, 8))

    tk.Label(frame, text="Metrics", anchor="w", font=("TkDefaultFont", 10, "bold")).pack(fill="x")
    tk.Label(frame, textvariable=metrics_var, anchor="w", justify="left", wraplength=700).pack(fill="x", pady=(0, 8))

    tk.Label(frame, text="Recent Events", anchor="w", font=("TkDefaultFont", 10, "bold")).pack(fill="x")
    event_box = scrolledtext.ScrolledText(frame, height=12, wrap="word", state="disabled")
    event_box.pack(fill="both", expand=True)

    def refresh() -> None:
        payload = _read_status_payload(status_path)
        stage = payload.get("stage", "unknown")
        stage_var.set(_format_status_stage(str(stage), payload))
        message_var.set(payload.get("message", ""))
        updated_var.set(payload.get("updated_at", ""))

        metrics = payload.get("metrics", {})
        metrics_lines: list[str] = []
        if "auto_loop_iteration" in metrics and "auto_loop_target_attempts" in metrics:
            metrics_lines.append(
                f"loop_progress: {metrics.get('auto_loop_iteration')}/{metrics.get('auto_loop_target_attempts')}"
            )
        metrics_lines.extend(f"{key}: {value}" for key, value in sorted(metrics.items()))
        metrics_var.set("\n".join(metrics_lines) if metrics_lines else "No metrics yet.")

        recent_events = payload.get("recent_events", [])
        rendered_events = "\n".join(
            f"[{event.get('time', '')}] {event.get('stage', '')}: {event.get('message', '')}"
            for event in recent_events
        )
        event_box.configure(state="normal")
        event_box.delete("1.0", tk.END)
        event_box.insert("1.0", rendered_events or "No events yet.")
        event_box.configure(state="disabled")

        root.after(1000, refresh)

    refresh()
    root.mainloop()


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - GUI-only
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        raise SystemExit("usage: python -m job_agent.status /path/to/live-status.json")
    run_status_viewer(Path(args[0]))
    return 0


if __name__ == "__main__":  # pragma: no cover - GUI-only
    raise SystemExit(main())
