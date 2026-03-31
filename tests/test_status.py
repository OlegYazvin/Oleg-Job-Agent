from pathlib import Path
import subprocess

import job_agent.status as status


def test_spawn_progress_gui_detaches_dashboard(monkeypatch) -> None:
    recorded: dict[str, object] = {}

    def fake_popen(args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return "process"

    monkeypatch.setattr(status, "can_launch_progress_gui", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    launched = status.spawn_progress_gui(Path("data/live-status.json"))

    assert launched == "process"
    assert recorded["args"] == [status.sys.executable, "-m", "job_agent.dashboard"]
    assert recorded["kwargs"]["stdin"] is subprocess.DEVNULL
    assert recorded["kwargs"]["stdout"] is subprocess.DEVNULL
    assert recorded["kwargs"]["stderr"] is subprocess.DEVNULL
    assert recorded["kwargs"]["start_new_session"] is True
    assert recorded["kwargs"]["close_fds"] is True
