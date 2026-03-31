from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

from .history import record_successful_run
from .models import JobOutreachBundle, RunManifest, SearchDiagnostics
from .scorecard import build_run_scorecard, save_run_scorecard


def save_json_snapshot(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def save_run_artifacts(
    data_dir: Path,
    bundles: list[JobOutreachBundle],
    manifest: RunManifest,
    live_outreach_payload: dict | None = None,
    near_miss_payload: dict | None = None,
    ollama_summary_payload: dict | None = None,
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
    if near_miss_payload is not None:
        payload["near_misses"] = near_miss_payload
        save_json_snapshot(data_dir / "near-misses-latest.json", near_miss_payload)
    if ollama_summary_payload is not None:
        payload["ollama_summary"] = ollama_summary_payload
        save_json_snapshot(data_dir / "ollama-summary-latest.json", ollama_summary_payload)
    if search_diagnostics is not None:
        payload["search_diagnostics"] = search_diagnostics.model_dump(mode="json")
        save_json_snapshot(
            data_dir / "false-negative-audit-latest.json",
            {"items": search_diagnostics.model_dump(mode="json").get("false_negative_audit", [])},
        )
    save_json_snapshot(data_dir / f"run-{timestamp}.json", payload)
    scorecard = build_run_scorecard(
        run_id=manifest.run_id,
        status="completed",
        manifest=manifest,
        search_diagnostics=search_diagnostics,
        near_miss_payload=near_miss_payload,
        ollama_summary_payload=ollama_summary_payload,
        status_payload=status_payload,
        generated_at=manifest.generated_at,
    )
    save_run_scorecard(data_dir, scorecard)
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
