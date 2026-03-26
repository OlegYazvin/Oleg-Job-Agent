import json
from pathlib import Path

from job_agent.status import StatusReporter


def test_status_reporter_writes_latest_state(tmp_path: Path) -> None:
    status_path = tmp_path / "live-status.json"
    reporter = StatusReporter(status_path)

    reporter.emit("search", "Searching for jobs.", qualifying_jobs=3)
    reporter.heartbeat("Still running.")
    reporter.complete("Done.", qualifying_jobs=5)

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["done"] is True
    assert payload["failed"] is False
    assert payload["stage"] == "completed"
    assert payload["message"] == "Done."
    assert payload["metrics"]["qualifying_jobs"] == 5
    assert payload["metrics"]["heartbeat_count"] == 1
    assert payload["recent_events"]
