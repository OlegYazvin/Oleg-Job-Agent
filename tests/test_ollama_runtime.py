from datetime import UTC, datetime
import json
from pathlib import Path

from job_agent.config import Settings
from job_agent.models import OllamaTuningProfile
from job_agent.ollama_runtime import auto_tune_ollama_settings, build_ollama_run_summary, record_ollama_event


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
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
        llm_provider="ollama",
        ollama_model="qwen2.5:7b-instruct",
        ollama_keep_alive="5m",
        ollama_num_ctx=1024,
        ollama_num_batch=4,
        ollama_num_predict=256,
        ollama_degraded_model="qwen2.5:3b-instruct",
    )


def test_auto_tune_ollama_settings_reduces_batch_after_failures(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    for index in range(8):
        record_ollama_event(
            settings,
            "request_success",
            run_id=f"run-{index}",
            caller="lead_refinement",
            prompt_category="lead_cleanup",
            model=settings.ollama_model,
            wall_duration_seconds=10.0,
            cold_start=False,
        )
    for index in range(3):
        record_ollama_event(
            settings,
            "request_failure",
            run_id=f"run-fail-{index}",
            caller="lead_refinement",
            prompt_category="lead_cleanup",
            model=settings.ollama_model,
            error_type="ReadTimeout",
        )

    tuned_settings, profile = auto_tune_ollama_settings(settings, run_id="run-current")

    assert tuned_settings.ollama_num_batch == 2
    assert profile.num_batch == 2
    saved_profile = json.loads(settings.ollama_tuning_profile_path.read_text(encoding="utf-8"))
    assert saved_profile["num_batch"] == 2


def test_auto_tune_ollama_settings_switches_to_smaller_model_when_warm_calls_are_slow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = build_settings(tmp_path)
    for index in range(5):
        record_ollama_event(
            settings,
            "request_success",
            run_id=f"slow-{index}",
            caller="drafting",
            prompt_category="first_order_messages",
            model=settings.ollama_model,
            wall_duration_seconds=75.0,
            cold_start=False,
        )

    monkeypatch.setattr(
        "job_agent.ollama_runtime._available_ollama_model_names",
        lambda _settings: {settings.ollama_model, settings.ollama_degraded_model},
    )

    tuned_settings, profile = auto_tune_ollama_settings(settings, run_id="run-slow")

    assert tuned_settings.ollama_model == "qwen2.5:3b-instruct"
    assert profile.model == "qwen2.5:3b-instruct"


def test_auto_tune_ollama_settings_keeps_current_model_when_degraded_model_is_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = build_settings(tmp_path)
    for index in range(5):
        record_ollama_event(
            settings,
            "request_success",
            run_id=f"slow-{index}",
            caller="drafting",
            prompt_category="first_order_messages",
            model=settings.ollama_model,
            wall_duration_seconds=75.0,
            cold_start=False,
        )

    monkeypatch.setattr(
        "job_agent.ollama_runtime._available_ollama_model_names",
        lambda _settings: {settings.ollama_model},
    )

    tuned_settings, profile = auto_tune_ollama_settings(settings, run_id="run-slow")

    assert tuned_settings.ollama_model == settings.ollama_model
    assert profile.model == settings.ollama_model
    assert profile.degraded is False


def test_build_ollama_run_summary_aggregates_quality_counters(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    profile = OllamaTuningProfile(
        model=settings.ollama_model,
        keep_alive=settings.ollama_keep_alive,
        num_ctx=settings.ollama_num_ctx,
        num_batch=settings.ollama_num_batch,
        num_predict=settings.ollama_num_predict,
    )
    record_ollama_event(
        settings,
        "request_success",
        run_id="run-123",
        caller="drafting",
        prompt_category="first_order_messages",
        model=settings.ollama_model,
        wall_duration_seconds=12.0,
        cold_start=False,
    )
    record_ollama_event(
        settings,
        "request_outer_timeout",
        run_id="run-123",
        caller="lead_refinement",
        prompt_category="lead_cleanup",
        model=settings.ollama_model,
    )
    record_ollama_event(
        settings,
        "drafting_outcome",
        run_id="run-123",
        caller="drafting",
        prompt_category="first_order_messages",
        draft_count=3,
        average_lint_score=92.0,
        duplicate_count=0,
        used_output_count=3,
    )
    record_ollama_event(
        settings,
        "lead_refinement_outcome",
        run_id="run-123",
        caller="lead_refinement",
        prompt_category="lead_cleanup",
        proposed_lead_count=8,
        returned_lead_count=3,
        merged_lead_count=4,
        used_output_count=3,
    )

    summary = build_ollama_run_summary(
        settings,
        run_id="run-123",
        tuning_profile=profile,
        generated_at=datetime(2026, 3, 28, tzinfo=UTC),
    )

    assert summary.request_count == 2
    assert summary.success_count == 1
    assert summary.outer_timeout_count == 1
    assert summary.caller_breakdown["drafting"] == 1
    assert summary.caller_breakdown["lead_refinement"] == 1
    assert summary.quality_counters["draft_count"] == 3.0
    assert summary.quality_counters["proposed_lead_count"] == 8.0
