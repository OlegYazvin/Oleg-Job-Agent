from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

from .config import load_settings
from .history import load_job_history_entries, load_run_history_entries
from .status import _pid_is_alive, _read_status_payload, can_launch_progress_gui

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox, scrolledtext, ttk
except Exception:  # pragma: no cover - GUI-only
    tk = None
    tkfont = None
    messagebox = None
    scrolledtext = None
    ttk = None


_WINDOW_BG = "#eff1f5"
_CARD_BG = "#fcfcfc"
_TEXT_PRIMARY = "#232629"
_TEXT_MUTED = "#5a616d"
_ACCENT = "#3daee9"
_ACCENT_DARK = "#1d99f3"
_BORDER = "#cfd8dc"
_SUCCESS = "#2ecc71"
_ERROR = "#da4453"
_WARNING = "#fdbc4b"


def _pick_font_family(candidates: list[str], fallback: str) -> str:
    if tkfont is None:
        return fallback
    available = set(tkfont.families())
    for candidate in candidates:
        if candidate in available:
            return candidate
    return fallback


def _format_timestamp(value: str | None) -> str:
    if not value:
        return ""
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def _status_badge_colors(status: str) -> tuple[str, str]:
    normalized = status.lower()
    if normalized in {"completed", "complete"}:
        return _SUCCESS, "#ffffff"
    if normalized in {"failed", "stale"}:
        return _ERROR, "#ffffff"
    if normalized in {"heartbeat", "search", "linkedin", "drafting", "reporting", "starting"}:
        return _ACCENT_DARK, "#ffffff"
    return _WARNING, _TEXT_PRIMARY


def _open_path(path: str) -> None:
    if not path:
        return
    subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _active_run_from_payload(payload: dict[str, Any]) -> bool:
    if payload.get("done"):
        return False
    pid = payload.get("pid")
    return isinstance(pid, int) and _pid_is_alive(pid)


def _build_runs_for_display(live_status: dict[str, Any], run_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(run_history)
    if live_status and _active_run_from_payload(live_status):
        live_entry = {
            "run_id": live_status.get("run_id"),
            "status": live_status.get("stage", "running"),
            "started_at": live_status.get("started_at"),
            "ended_at": live_status.get("updated_at"),
            "message": live_status.get("message"),
            "jobs_found_by_search": live_status.get("metrics", {}).get("jobs_found_by_search", 0),
            "jobs_kept_after_validation": live_status.get("metrics", {}).get("jobs_kept_after_validation", 0),
            "jobs_with_any_messages": live_status.get("metrics", {}).get("jobs_with_any_messages", 0),
            "message_docx_path": None,
            "summary_docx_path": None,
            "live": True,
        }
        rows = [row for row in rows if row.get("run_id") != live_entry["run_id"]]
        rows.insert(0, live_entry)
    return rows


def run_dashboard() -> None:  # pragma: no cover - GUI-only
    if tk is None or ttk is None or scrolledtext is None or messagebox is None:
        raise RuntimeError("Tkinter is not available on this system.")

    settings = load_settings(require_openai=False)
    root = tk.Tk()
    root.title("Job Agent Control Center")
    root.geometry("1320x860")
    root.minsize(1080, 720)
    root.configure(bg=_WINDOW_BG)

    sans_font = _pick_font_family(
        ["Noto Sans", "Inter", "Cantarell", "Segoe UI", "Ubuntu", "Arial"],
        "TkDefaultFont",
    )
    mono_font = _pick_font_family(
        ["JetBrains Mono", "Fira Code", "Hack", "DejaVu Sans Mono"],
        "TkFixedFont",
    )

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(".", background=_WINDOW_BG, foreground=_TEXT_PRIMARY, font=(sans_font, 10))
    style.configure("Card.TFrame", background=_CARD_BG, relief="flat")
    style.configure("CardTitle.TLabel", background=_CARD_BG, foreground=_TEXT_PRIMARY, font=(sans_font, 12, "bold"))
    style.configure("Body.TLabel", background=_CARD_BG, foreground=_TEXT_PRIMARY, font=(sans_font, 10))
    style.configure("Muted.TLabel", background=_CARD_BG, foreground=_TEXT_MUTED, font=(sans_font, 10))
    style.configure("Header.TLabel", background=_WINDOW_BG, foreground=_TEXT_PRIMARY, font=(sans_font, 22, "bold"))
    style.configure("Subheader.TLabel", background=_WINDOW_BG, foreground=_TEXT_MUTED, font=(sans_font, 10))
    style.configure("Primary.TButton", font=(sans_font, 10, "bold"))
    style.map("Primary.TButton", background=[("active", _ACCENT_DARK), ("!disabled", _ACCENT)])
    style.configure("Treeview", font=(sans_font, 10), rowheight=28, fieldbackground=_CARD_BG, background=_CARD_BG)
    style.configure("Treeview.Heading", font=(sans_font, 10, "bold"))

    outer = ttk.Frame(root, padding=18, style="Card.TFrame")
    outer.pack(fill="both", expand=True)
    outer.configure(style="TFrame")

    header_frame = ttk.Frame(outer)
    header_frame.pack(fill="x", pady=(0, 12))

    title_frame = ttk.Frame(header_frame)
    title_frame.pack(side="left", fill="x", expand=True)
    ttk.Label(title_frame, text="Job Agent Control Center", style="Header.TLabel").pack(anchor="w")
    ttk.Label(
        title_frame,
        text="Breeze-inspired desktop panel for recurring AI PM job discovery and outreach drafting.",
        style="Subheader.TLabel",
    ).pack(anchor="w", pady=(4, 0))

    button_frame = ttk.Frame(header_frame)
    button_frame.pack(side="right")

    status_badge = tk.Label(
        header_frame,
        text="Idle",
        bg=_WARNING,
        fg=_TEXT_PRIMARY,
        padx=12,
        pady=6,
        font=(sans_font, 10, "bold"),
    )
    status_badge.pack(side="right", padx=(0, 16))

    content = ttk.Panedwindow(outer, orient="horizontal")
    content.pack(fill="both", expand=True)

    left_panel = ttk.Frame(content, style="Card.TFrame", padding=14)
    right_panel = ttk.Frame(content, style="Card.TFrame", padding=14)
    content.add(left_panel, weight=3)
    content.add(right_panel, weight=2)

    ttk.Label(left_panel, text="Run History", style="CardTitle.TLabel").pack(anchor="w")
    ttk.Label(
        left_panel,
        text="Every completed or failed run is retained here, with date-stamped artifacts and clear counts.",
        style="Muted.TLabel",
    ).pack(anchor="w", pady=(2, 10))

    tree = ttk.Treeview(
        left_panel,
        columns=("time", "status", "found", "kept", "messages"),
        show="headings",
        height=14,
    )
    tree.heading("time", text="Time")
    tree.heading("status", text="Status")
    tree.heading("found", text="Found")
    tree.heading("kept", text="Kept")
    tree.heading("messages", text="Msgs")
    tree.column("time", width=220, anchor="w")
    tree.column("status", width=120, anchor="center")
    tree.column("found", width=90, anchor="center")
    tree.column("kept", width=90, anchor="center")
    tree.column("messages", width=90, anchor="center")
    tree.pack(fill="both", expand=False)

    run_details = scrolledtext.ScrolledText(
        left_panel,
        height=13,
        wrap="word",
        bg="#ffffff",
        fg=_TEXT_PRIMARY,
        font=(mono_font, 10),
        relief="flat",
        borderwidth=1,
    )
    run_details.pack(fill="both", expand=True, pady=(12, 0))

    current_card = ttk.Frame(right_panel, style="Card.TFrame")
    current_card.pack(fill="x")
    ttk.Label(current_card, text="Current Run", style="CardTitle.TLabel").pack(anchor="w")

    current_stage_var = tk.StringVar(value="Idle")
    current_message_var = tk.StringVar(value="No active workflow.")
    current_updated_var = tk.StringVar(value="")
    current_metrics_var = tk.StringVar(value="")
    history_counts_var = tk.StringVar(value="")

    ttk.Label(current_card, textvariable=current_stage_var, style="Body.TLabel").pack(anchor="w", pady=(8, 2))
    ttk.Label(current_card, textvariable=current_message_var, style="Muted.TLabel", wraplength=420, justify="left").pack(
        anchor="w"
    )
    ttk.Label(current_card, textvariable=current_updated_var, style="Muted.TLabel").pack(anchor="w", pady=(6, 0))
    ttk.Label(current_card, textvariable=current_metrics_var, style="Body.TLabel", justify="left").pack(anchor="w", pady=(10, 0))
    ttk.Label(current_card, textvariable=history_counts_var, style="Muted.TLabel", justify="left").pack(anchor="w", pady=(8, 0))

    controls_card = ttk.Frame(right_panel, style="Card.TFrame")
    controls_card.pack(fill="x", pady=(18, 0))
    ttk.Label(controls_card, text="Controls", style="CardTitle.TLabel").pack(anchor="w")

    def start_run() -> None:
        live = _read_status_payload(settings.live_status_path)
        if _active_run_from_payload(live):
            messagebox.showinfo("Run In Progress", "A workflow run is already active.")
            return
        env = os.environ.copy()
        env["ENABLE_PROGRESS_GUI"] = "false"
        subprocess.Popen(
            [sys.executable, "-m", "job_agent.cli", "run", "--no-gui"],
            cwd=str(settings.project_root),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def kill_run() -> None:
        live = _read_status_payload(settings.live_status_path)
        pid = live.get("pid")
        if not isinstance(pid, int) or not _active_run_from_payload(live):
            messagebox.showinfo("No Active Run", "There is no running workflow to stop.")
            return
        if not messagebox.askyesno("Stop Workflow", "Send a termination signal to the current workflow run?"):
            return
        os.kill(pid, signal.SIGTERM)

    def open_selected_message_doc() -> None:
        selection = tree.selection()
        if not selection:
            return
        run_id = selection[0]
        for entry in current_runs:
            if str(entry.get("run_id")) == run_id and entry.get("message_docx_path"):
                _open_path(str(entry["message_docx_path"]))
                return

    def open_selected_summary_doc() -> None:
        selection = tree.selection()
        if not selection:
            return
        run_id = selection[0]
        for entry in current_runs:
            if str(entry.get("run_id")) == run_id and entry.get("summary_docx_path"):
                _open_path(str(entry["summary_docx_path"]))
                return

    button_row = ttk.Frame(controls_card)
    button_row.pack(fill="x", pady=(10, 0))
    start_button = ttk.Button(button_row, text="Start Run", style="Primary.TButton", command=start_run)
    start_button.pack(side="left")
    kill_button = ttk.Button(button_row, text="Kill Run", command=kill_run)
    kill_button.pack(side="left", padx=(10, 0))
    ttk.Button(button_row, text="Open Messages DOCX", command=open_selected_message_doc).pack(side="left", padx=(10, 0))
    ttk.Button(button_row, text="Open Summary DOCX", command=open_selected_summary_doc).pack(side="left", padx=(10, 0))
    ttk.Button(button_row, text="Open Output Folder", command=lambda: _open_path(str(settings.output_dir))).pack(
        side="left",
        padx=(10, 0),
    )

    events_card = ttk.Frame(right_panel, style="Card.TFrame")
    events_card.pack(fill="both", expand=True, pady=(18, 0))
    ttk.Label(events_card, text="Recent Events", style="CardTitle.TLabel").pack(anchor="w")
    events_box = scrolledtext.ScrolledText(
        events_card,
        height=20,
        wrap="word",
        bg="#ffffff",
        fg=_TEXT_PRIMARY,
        font=(mono_font, 10),
        relief="flat",
        borderwidth=1,
    )
    events_box.pack(fill="both", expand=True, pady=(10, 0))

    current_runs: list[dict[str, Any]] = []

    def render_selected_run(*args) -> None:
        selection = tree.selection()
        if not selection:
            return
        run_id = selection[0]
        selected = next((entry for entry in current_runs if str(entry.get("run_id")) == run_id), None)
        if selected is None:
            return
        detail_lines = [
            f"Run ID: {selected.get('run_id', '')}",
            f"Status: {selected.get('status', '')}",
            f"Started: {_format_timestamp(selected.get('started_at'))}",
            f"Ended: {_format_timestamp(selected.get('ended_at'))}",
            f"Jobs Found: {selected.get('jobs_found_by_search', 0)}",
            f"Jobs Kept: {selected.get('jobs_kept_after_validation', 0)}",
            f"Jobs With Messages: {selected.get('jobs_with_any_messages', 0)}",
            "",
            f"Message DOCX: {selected.get('message_docx_path') or 'n/a'}",
            f"Summary DOCX: {selected.get('summary_docx_path') or 'n/a'}",
            "",
            f"Message: {selected.get('message') or ''}",
        ]
        run_details.delete("1.0", tk.END)
        run_details.insert("1.0", "\n".join(detail_lines))

    tree.bind("<<TreeviewSelect>>", render_selected_run)

    def refresh() -> None:
        nonlocal current_runs
        live_status = _read_status_payload(settings.live_status_path)
        run_history = load_run_history_entries(settings.data_dir)
        job_history = load_job_history_entries(settings.data_dir)
        current_runs = _build_runs_for_display(live_status, run_history)

        current_stage = str(live_status.get("stage", "idle"))
        badge_bg, badge_fg = _status_badge_colors(current_stage if _active_run_from_payload(live_status) else "idle")
        status_badge.configure(text=current_stage.upper(), bg=badge_bg, fg=badge_fg)
        current_stage_var.set(f"Stage: {current_stage}")
        current_message_var.set(str(live_status.get("message") or "No active workflow."))
        current_updated_var.set(f"Last updated: {_format_timestamp(live_status.get('updated_at'))}")

        metrics = live_status.get("metrics", {})
        metrics_lines = [
            f"Target jobs: {metrics.get('target_job_count', 0)}",
            f"Found by search: {metrics.get('jobs_found_by_search', metrics.get('unique_leads_discovered', 0))}",
            f"Kept after validation: {metrics.get('jobs_kept_after_validation', metrics.get('qualifying_jobs', 0))}",
            f"Jobs with messages: {metrics.get('jobs_with_any_messages', 0)}",
            f"Current company: {metrics.get('current_company', '') or 'n/a'}",
            f"Current role: {metrics.get('current_role', '') or 'n/a'}",
        ]
        current_metrics_var.set("\n".join(metrics_lines))
        history_counts_var.set(
            f"Historical runs logged: {len(run_history)}\n"
            f"Unique jobs archived: {len(job_history)}"
        )

        existing_selection = tree.selection()
        existing_ids = set(tree.get_children())
        desired_ids = {str(entry.get("run_id")) for entry in current_runs if entry.get("run_id")}
        for item_id in existing_ids - desired_ids:
            tree.delete(item_id)
        for entry in current_runs:
            run_id = str(entry.get("run_id") or "")
            if not run_id:
                continue
            values = (
                _format_timestamp(entry.get("started_at") or entry.get("ended_at")),
                str(entry.get("status") or ""),
                int(entry.get("jobs_found_by_search") or 0),
                int(entry.get("jobs_kept_after_validation") or 0),
                int(entry.get("jobs_with_any_messages") or 0),
            )
            if tree.exists(run_id):
                tree.item(run_id, values=values)
            else:
                tree.insert("", "end", iid=run_id, values=values)

        if existing_selection and tree.exists(existing_selection[0]):
            tree.selection_set(existing_selection[0])
        elif current_runs:
            first_run_id = str(current_runs[0].get("run_id") or "")
            if first_run_id:
                tree.selection_set(first_run_id)
        render_selected_run()

        events = live_status.get("recent_events", [])
        events_box.delete("1.0", tk.END)
        events_box.insert(
            "1.0",
            "\n".join(
                f"[{_format_timestamp(event.get('time'))}] {event.get('stage', '').upper()}: {event.get('message', '')}"
                for event in events
            )
            or "No events yet.",
        )

        is_active = _active_run_from_payload(live_status)
        start_button.state(["disabled"] if is_active else ["!disabled"])
        kill_button.state(["!disabled"] if is_active else ["disabled"])
        root.after(1000, refresh)

    refresh()
    root.mainloop()


def main() -> int:  # pragma: no cover - GUI-only
    if not can_launch_progress_gui():
        raise SystemExit("A graphical display is not available for the Job Agent dashboard.")
    run_dashboard()
    return 0


if __name__ == "__main__":  # pragma: no cover - GUI-only
    raise SystemExit(main())
