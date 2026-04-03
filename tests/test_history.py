from datetime import UTC, datetime
import json
from pathlib import Path

from job_agent.history import (
    load_company_history_entries,
    load_company_watchlist_entries,
    load_previously_reported_company_keys,
    load_previously_reported_job_keys,
    record_company_watchlist,
    record_failed_run,
    record_successful_run,
)
from job_agent.models import JobOutreachBundle, JobPosting, RunManifest, SearchFailure


def _build_bundle() -> JobOutreachBundle:
    return JobOutreachBundle(
        job=JobPosting(
            company_name="Acme AI",
            role_title="Staff Product Manager, AI",
            direct_job_url="https://jobs.ashbyhq.com/acme/123",
            resolved_job_url="https://jobs.ashbyhq.com/acme/123",
            ats_platform="Ashby",
            location_text="Remote",
            is_fully_remote=True,
            posted_date_text="2026-03-25",
            posted_date_iso="2026-03-25",
            salary_text="$220,000",
            evidence_notes="Valid direct role.",
        )
    )


def test_record_successful_run_updates_run_and_job_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("job_agent.history._utc_now", lambda: datetime(2026, 3, 26, 0, 0, 0, tzinfo=UTC))
    bundle = _build_bundle()
    manifest = RunManifest(
        run_id="run-123",
        generated_at=datetime(2026, 3, 25, 14, 44, 53),
        message_docx_path=str(tmp_path / "output" / "linkedin_outreach_messages-20260325-144453.docx"),
        summary_docx_path=str(tmp_path / "output" / "job_summary-20260325-144453.docx"),
        jobs_found_by_search=50,
        jobs_kept_after_validation=1,
        jobs_with_any_messages=0,
    )

    record_successful_run(
        tmp_path,
        run_id="run-123",
        manifest=manifest,
        bundles=[bundle],
        status_payload={"started_at": "2026-03-25T19:00:00+00:00", "updated_at": "2026-03-25T19:44:53+00:00"},
    )

    run_history = json.loads((tmp_path / "run-history.json").read_text(encoding="utf-8"))
    job_history = json.loads((tmp_path / "job-history.json").read_text(encoding="utf-8"))
    company_history = json.loads((tmp_path / "company-history.json").read_text(encoding="utf-8"))

    assert run_history[0]["run_id"] == "run-123"
    assert run_history[0]["status"] == "completed"
    assert run_history[0]["message_docx_path"].endswith("linkedin_outreach_messages-20260325-144453.docx")
    assert "ashby:acme:123" in job_history
    assert "acmeai" in company_history
    reported_keys = load_previously_reported_job_keys(tmp_path)
    assert "ashby:acme:123" in reported_keys
    assert "https://jobs.ashbyhq.com/acme/123" in reported_keys
    assert load_previously_reported_company_keys(tmp_path) == {"acmeai"}
    assert load_company_history_entries(tmp_path)["acmeai"]["company_name"] == "Acme AI"


def test_record_failed_run_writes_run_history_entry(tmp_path: Path) -> None:
    record_failed_run(
        tmp_path,
        run_id="run-456",
        status_payload={
            "started_at": "2026-03-25T20:00:00+00:00",
            "updated_at": "2026-03-25T20:05:00+00:00",
            "metrics": {"jobs_found_by_search": 12, "jobs_kept_after_validation": 3},
        },
        failure_message="Workflow terminated before completion.",
    )

    run_history = json.loads((tmp_path / "run-history.json").read_text(encoding="utf-8"))
    assert run_history[0]["run_id"] == "run-456"
    assert run_history[0]["status"] == "failed"
    assert run_history[0]["jobs_found_by_search"] == 12


def test_record_failed_run_uses_live_progress_fallback_metrics(tmp_path: Path) -> None:
    record_failed_run(
        tmp_path,
        run_id="run-qualifying",
        status_payload={
            "started_at": "2026-03-30T18:33:36+00:00",
            "updated_at": "2026-03-30T18:56:21+00:00",
            "metrics": {
                "unique_leads_discovered": 213,
                "qualifying_jobs": 2,
            },
        },
        failure_message="Workflow terminated before completion.",
    )

    run_history = json.loads((tmp_path / "run-history.json").read_text(encoding="utf-8"))
    assert run_history[0]["jobs_found_by_search"] == 213
    assert run_history[0]["jobs_kept_after_validation"] == 2


def test_record_company_watchlist_tracks_promising_companies(tmp_path: Path) -> None:
    failure = SearchFailure(
        stage="validation",
        reason_code="stale_posting",
        detail="Posting was older than the configured freshness window.",
        company_name="Tiny AI",
        source_url="https://jobs.ashbyhq.com/tiny/123",
        direct_job_url="https://jobs.ashbyhq.com/tiny/123",
    )

    record_company_watchlist(
        tmp_path,
        generated_at=datetime(2026, 3, 26, 12, 0, 0),
        failures=[failure],
    )

    company_watchlist = json.loads((tmp_path / "company-watchlist.json").read_text(encoding="utf-8"))
    assert "tinyai" in company_watchlist
    assert company_watchlist["tinyai"]["priority_score"] > 0
    assert "jobs.ashbyhq.com" in company_watchlist["tinyai"]["source_hosts"]
    assert load_company_watchlist_entries(tmp_path)["tinyai"]["company_name"] == "Tiny AI"


def test_record_successful_run_normalizes_job_history_tracking_urls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("job_agent.history._utc_now", lambda: datetime(2026, 3, 27, 0, 0, 0, tzinfo=UTC))
    bundle = JobOutreachBundle(
        job=JobPosting(
            company_name="Dropbox",
            role_title="Staff Product Manager, AI Organization Workflows",
            direct_job_url="https://www.dropbox.jobs/en/jobs/7729233/staff-product-manager-ai-organization-workflows/?gh_src=c393b7f81us",
            resolved_job_url="https://www.dropbox.jobs/en/jobs/7729233/staff-product-manager-ai-organization-workflows/?gh_src=c393b7f81us",
            ats_platform="Dropbox",
            location_text="Remote",
            is_fully_remote=True,
            posted_date_text="2026-03-25",
            posted_date_iso="2026-03-25",
            salary_text="$250,000",
            evidence_notes="Valid direct role.",
        )
    )
    manifest = RunManifest(
        run_id="run-dropbox",
        generated_at=datetime(2026, 3, 26, 16, 0, 0),
        message_docx_path=str(tmp_path / "output" / "linkedin_outreach_messages-20260326-160000.docx"),
        summary_docx_path=str(tmp_path / "output" / "job_summary-20260326-160000.docx"),
        jobs_found_by_search=10,
        jobs_kept_after_validation=1,
        jobs_with_any_messages=0,
    )

    record_successful_run(tmp_path, run_id="run-dropbox", manifest=manifest, bundles=[bundle])

    job_history = json.loads((tmp_path / "job-history.json").read_text(encoding="utf-8"))
    normalized_key = "https://www.dropbox.jobs/en/jobs/7729233/staff-product-manager-ai-organization-workflows"
    assert normalized_key in job_history
    assert load_previously_reported_job_keys(tmp_path) == {normalized_key}


def test_load_previously_reported_job_keys_expires_old_entries(tmp_path: Path, monkeypatch) -> None:
    bundle = _build_bundle()
    manifest = RunManifest(
        run_id="run-old",
        generated_at=datetime(2026, 3, 20, 12, 0, 0),
        message_docx_path=str(tmp_path / "output" / "linkedin_outreach_messages-20260320-120000.docx"),
        summary_docx_path=str(tmp_path / "output" / "job_summary-20260320-120000.docx"),
        jobs_found_by_search=10,
        jobs_kept_after_validation=1,
        jobs_with_any_messages=0,
    )

    record_successful_run(tmp_path, run_id="run-old", manifest=manifest, bundles=[bundle])
    monkeypatch.setattr("job_agent.history._utc_now", lambda: datetime(2026, 4, 2, 0, 0, 0, tzinfo=UTC))

    assert load_previously_reported_job_keys(tmp_path) == set()
