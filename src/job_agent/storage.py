from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

from .history import record_successful_run
from .models import JobOutreachBundle, RunManifest, SearchDiagnostics


def save_json_snapshot(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def save_run_artifacts(
    data_dir: Path,
    bundles: list[JobOutreachBundle],
    manifest: RunManifest,
    live_outreach_payload: dict | None = None,
    search_diagnostics: SearchDiagnostics | None = None,
    status_payload: dict | None = None,
) -> None:
    timestamp = manifest.generated_at.strftime("%Y%m%d-%H%M%S")
    payload = {
        "manifest": manifest.model_dump(mode="json"),
        "bundles": [bundle.model_dump(mode="json") for bundle in bundles],
    }
    if live_outreach_payload is not None:
        payload["live_outreach"] = live_outreach_payload
        save_json_snapshot(data_dir / "live-outreach.json", live_outreach_payload)
    if search_diagnostics is not None:
        payload["search_diagnostics"] = search_diagnostics.model_dump(mode="json")
    save_json_snapshot(data_dir / f"run-{timestamp}.json", payload)
    record_successful_run(
        data_dir,
        run_id=manifest.run_id,
        manifest=manifest,
        bundles=bundles,
        status_payload=status_payload,
    )


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")
