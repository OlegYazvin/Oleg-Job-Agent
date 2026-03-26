import asyncio
import json
from pathlib import Path

import pytest

from job_agent.config import Settings
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
        posted_within_days=14,
        minimum_qualifying_jobs=5,
        target_job_count=10,
        max_adaptive_search_passes=3,
        max_search_rounds=3,
        search_round_query_limit=6,
        max_leads_per_query=10,
        max_leads_to_resolve_per_pass=80,
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
    status = StatusReporter(settings.live_status_path)

    async def slow_find_matching_jobs(*args, **kwargs):
        await asyncio.sleep(0.05)
        return [], 0, None

    monkeypatch.setattr("job_agent.workflow.find_matching_jobs", slow_find_matching_jobs)

    with pytest.raises(TimeoutError):
        asyncio.run(run_daily_workflow(settings, status=status, timeout_seconds=0.01))

    payload = json.loads(settings.live_status_path.read_text(encoding="utf-8"))
    assert payload["done"] is True
    assert payload["failed"] is True
    assert payload["stage"] == "failed"
    assert "timed out" in payload["message"].lower()
    assert payload["metrics"]["workflow_timeout_seconds"] == 0.01
