from pathlib import Path

from job_agent.config import Settings
from job_agent.scheduler import render_cron_line


def test_render_cron_line_contains_marker_and_hour() -> None:
    settings = Settings(
        project_root=Path("/tmp/job-agent"),
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
        linkedin_profile_dir=Path("/tmp/job-agent/.secrets/profile"),
        linkedin_storage_state=Path("/tmp/job-agent/.secrets/state.json"),
        output_dir=Path("/tmp/job-agent/output"),
        data_dir=Path("/tmp/job-agent/data"),
        headless=True,
        timezone="America/Chicago",
        search_country="US",
        search_city="Chicago",
        search_region="Illinois",
        min_base_salary_usd=200000,
        posted_within_days=7,
        minimum_qualifying_jobs=5,
        target_job_count=10,
        max_adaptive_search_passes=3,
        max_search_rounds=8,
        search_round_query_limit=6,
        max_leads_per_query=8,
        max_leads_to_resolve_per_pass=40,
        reacquisition_attempt_cap=10,
        per_query_timeout_seconds=45,
        per_lead_timeout_seconds=30,
        workflow_timeout_seconds=3600,
        max_linkedin_results_per_company=25,
        max_linkedin_pages_per_company=3,
        daily_run_hour=8,
        daily_run_minute=15,
        status_heartbeat_seconds=120,
        enable_progress_gui=False,
    )
    line = render_cron_line(settings)
    assert line.startswith("15 8 ")
    assert "PYTHONPATH=src .venv/bin/python -m job_agent.cli run" in line


def test_render_cron_line_uses_configured_ollama_command() -> None:
    settings = Settings(
        project_root=Path("/tmp/job-agent"),
        openai_api_key="",
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
        linkedin_profile_dir=Path("/tmp/job-agent/.secrets/profile"),
        linkedin_storage_state=Path("/tmp/job-agent/.secrets/state.json"),
        output_dir=Path("/tmp/job-agent/output"),
        data_dir=Path("/tmp/job-agent/data"),
        headless=True,
        timezone="America/Chicago",
        search_country="US",
        search_city="Chicago",
        search_region="Illinois",
        min_base_salary_usd=200000,
        posted_within_days=7,
        minimum_qualifying_jobs=5,
        target_job_count=10,
        max_adaptive_search_passes=3,
        max_search_rounds=8,
        search_round_query_limit=6,
        max_leads_per_query=8,
        max_leads_to_resolve_per_pass=40,
        reacquisition_attempt_cap=10,
        per_query_timeout_seconds=45,
        per_lead_timeout_seconds=30,
        workflow_timeout_seconds=3600,
        max_linkedin_results_per_company=25,
        max_linkedin_pages_per_company=3,
        daily_run_hour=8,
        daily_run_minute=15,
        status_heartbeat_seconds=120,
        enable_progress_gui=False,
        llm_provider="ollama",
        ollama_command="/custom/bin/ollama",
    )

    line = render_cron_line(settings)

    assert "nohup /custom/bin/ollama serve" in line
