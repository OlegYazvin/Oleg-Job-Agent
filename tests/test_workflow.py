import asyncio
import json
from pathlib import Path

import pytest

from job_agent.config import Settings
from job_agent.models import JobOutreachBundle, JobPosting, NearMissJob, OllamaTuningProfile, SearchDiagnostics
from job_agent.status import StatusReporter
from job_agent.workflow import run_daily_workflow


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        openai_api_key="test-key",
        linkedin_email=None,
        linkedin_password=None,
        linkedin_totp_secret=None,
        linkedin_li_at=None,
        linkedin_jsessionid=None,
        google_email=None,
        google_password=None,
        google_totp_secret=None,
        browser_executable_path=None,
        browser_channel=None,
        linkedin_profile_dir=tmp_path / ".secrets/profile",
        linkedin_storage_state=tmp_path / ".secrets/state.json",
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        headless=True,
        timezone="America/Chicago",
        search_country="US",
        search_city="Chicago",
        search_region="Illinois",
        min_base_salary_usd=200000,
        enable_principal_ai_pm_salary_presumption=True,
        company_discovery_enabled=True,
        posted_within_days=14,
        minimum_qualifying_jobs=5,
        target_job_count=10,
        max_adaptive_search_passes=3,
        max_search_rounds=3,
        search_round_query_limit=6,
        max_leads_per_query=10,
        max_leads_to_resolve_per_pass=80,
        reacquisition_attempt_cap=10,
        per_query_timeout_seconds=45,
        per_lead_timeout_seconds=30,
        workflow_timeout_seconds=3600,
        max_linkedin_results_per_company=25,
        max_linkedin_pages_per_company=3,
        daily_run_hour=8,
        daily_run_minute=0,
        status_heartbeat_seconds=120,
        enable_progress_gui=False,
        linkedin_manual_review_mode=True,
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:14b-instruct",
        use_openai_fallback=False,
        local_confidence_threshold=0.75,
    )


def test_run_daily_workflow_marks_status_failed_when_timeout_elapses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = build_settings(tmp_path)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    status = StatusReporter(settings.live_status_path)

    async def slow_find_matching_jobs(*args, **kwargs):
        await asyncio.sleep(0.05)
        return [], [], 0, None

    monkeypatch.setattr("job_agent.workflow.find_matching_jobs", slow_find_matching_jobs)

    with pytest.raises(TimeoutError):
        asyncio.run(run_daily_workflow(settings, status=status, timeout_seconds=0.01))

    payload = json.loads(settings.live_status_path.read_text(encoding="utf-8"))
    assert payload["done"] is True
    assert payload["failed"] is True
    assert payload["stage"] == "failed"
    assert "timed out" in payload["message"].lower()
    assert payload["metrics"]["workflow_timeout_seconds"] == 0.01
    assert (tmp_path / "data" / "run-scorecard-latest.json").exists()


def test_run_daily_workflow_writes_near_miss_and_ollama_summary_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = build_settings(tmp_path)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    status = StatusReporter(settings.live_status_path)
    job = JobPosting(
        company_name="Acme AI",
        role_title="Staff Product Manager, AI",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        resolved_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-27",
        posted_date_iso="2026-03-27",
        base_salary_min_usd=210000,
        base_salary_max_usd=240000,
        salary_text="$210,000 - $240,000",
        evidence_notes="Valid role.",
    )
    diagnostics = SearchDiagnostics(
        minimum_qualifying_jobs=5,
        near_misses=[
            NearMissJob(
                company_name="Close Co",
                role_title="Staff Product Manager, AI Platform",
                reason_code="missing_salary",
                detail="No salary was listed.",
                why_close="Strong ATS role with missing compensation.",
                source_url="https://jobs.ashbyhq.com/closeco/123",
                direct_job_url="https://jobs.ashbyhq.com/closeco/123",
            )
        ],
    )

    async def fake_find_matching_jobs(*args, **kwargs):
        return [job], [], 12, diagnostics

    async def fake_draft_outreach_bundle(*args, **kwargs):
        return JobOutreachBundle(job=job)

    profile = OllamaTuningProfile(
        model=settings.ollama_model,
        keep_alive=settings.ollama_keep_alive,
        num_ctx=settings.ollama_num_ctx,
        num_batch=settings.ollama_num_batch,
        num_predict=settings.ollama_num_predict,
    )

    monkeypatch.setattr("job_agent.workflow.find_matching_jobs", fake_find_matching_jobs)
    monkeypatch.setattr("job_agent.workflow.draft_outreach_bundle", fake_draft_outreach_bundle)
    monkeypatch.setattr("job_agent.workflow.auto_tune_ollama_settings", lambda configured, run_id=None: (configured, profile))

    bundles, manifest = asyncio.run(run_daily_workflow(settings, status=status, timeout_seconds=10))

    assert len(bundles) == 1
    assert manifest.near_miss_count == 1
    assert (tmp_path / "data" / "near-misses-latest.json").exists()
    assert (tmp_path / "data" / "ollama-summary-latest.json").exists()
    assert (tmp_path / "data" / "run-scorecard-latest.json").exists()
    assert (tmp_path / "data" / "run-scorecards.jsonl").exists()
    summary_payload = json.loads((tmp_path / "data" / "ollama-summary-latest.json").read_text(encoding="utf-8"))
    assert summary_payload["run_id"] == manifest.run_id


def test_run_daily_workflow_defers_ollama_prewarm_until_needed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = build_settings(tmp_path)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    status = StatusReporter(settings.live_status_path)
    diagnostics = SearchDiagnostics(minimum_qualifying_jobs=5)

    profile = OllamaTuningProfile(
        model=settings.ollama_model,
        keep_alive=settings.ollama_keep_alive,
        num_ctx=settings.ollama_num_ctx,
        num_batch=settings.ollama_num_batch,
        num_predict=settings.ollama_num_predict,
    )

    async def fake_find_matching_jobs(*args, **kwargs):
        return [], [], 0, diagnostics

    monkeypatch.setattr("job_agent.workflow.find_matching_jobs", fake_find_matching_jobs)
    monkeypatch.setattr("job_agent.workflow.auto_tune_ollama_settings", lambda configured, run_id=None: (configured, profile))

    bundles, manifest = asyncio.run(run_daily_workflow(settings, status=status, timeout_seconds=10))

    assert bundles == []
    assert manifest.jobs_kept_after_validation == 0
    summary_payload = json.loads((tmp_path / "data" / "ollama-summary-latest.json").read_text(encoding="utf-8"))
    assert summary_payload["tuning_profile"]["degraded"] is False
