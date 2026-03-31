import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess

from job_agent.auto_loop import (
    build_run_improvement_analysis,
    ensure_baseline_commit,
    invoke_codex_iteration,
    run_autonomous_loop,
)
from job_agent.config import Settings
from job_agent.models import (
    AutoLoopState,
    CodexIterationResult,
    RunDiscoveryMetrics,
    RunOllamaMetrics,
    RunOutcomeMetrics,
    RunScorecard,
    RunTimingMetrics,
    RunValidationMetrics,
    ValidationCommandResult,
)
from job_agent.scorecard import save_run_scorecard


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
        ollama_model="qwen2.5:7b-instruct",
        use_openai_fallback=False,
        local_confidence_threshold=0.75,
        codex_command="codex",
    )


def _make_scorecard(
    *,
    run_id: str,
    status: str,
    generated_at: datetime,
    validated_jobs_count: int = 0,
    jobs_with_messages_count: int = 0,
    fresh_new_leads_count: int = 0,
    raw_near_miss_count: int = 0,
    actionable_near_miss_count: int = 0,
    replayed_seed_leads_count: int = 0,
    query_timeout_count: int = 0,
    query_skipped_timeout_budget_count: int = 0,
    discovery_efficiency: float = 0.0,
    request_count: int = 0,
    useful_actions_per_request: float = 0.0,
) -> RunScorecard:
    return RunScorecard(
        run_id=run_id,
        generated_at=generated_at,
        status=status,
        outcome=RunOutcomeMetrics(
            validated_jobs_count=validated_jobs_count,
            jobs_with_messages_count=jobs_with_messages_count,
            unique_leads_discovered_count=fresh_new_leads_count + replayed_seed_leads_count,
            fresh_new_leads_count=fresh_new_leads_count,
            actionable_near_miss_count=actionable_near_miss_count,
            raw_near_miss_count=raw_near_miss_count,
        ),
        discovery=RunDiscoveryMetrics(
            unique_leads_discovered_count=fresh_new_leads_count + replayed_seed_leads_count,
            fresh_new_leads_count=fresh_new_leads_count,
            replayed_seed_leads_count=replayed_seed_leads_count,
            repeated_failed_leads_suppressed_count=0,
            executed_query_count=max(1, fresh_new_leads_count + query_timeout_count),
            query_timeout_count=query_timeout_count,
            query_skipped_timeout_budget_count=query_skipped_timeout_budget_count,
            zero_yield_pass_count=0,
            discovery_efficiency=discovery_efficiency,
        ),
        validation=RunValidationMetrics(
            validated_jobs_count=validated_jobs_count,
            validated_yield=0.0,
            jobs_with_messages_count=jobs_with_messages_count,
            message_coverage_rate=0.0,
            raw_near_miss_count=raw_near_miss_count,
            actionable_near_miss_count=actionable_near_miss_count,
            actionable_near_miss_yield=0.0,
            company_mismatch_count=0,
            not_specific_job_page_count=0,
            missing_salary_count=0,
            fetch_non_200_count=0,
            stale_posting_count=0,
            not_remote_count=0,
            false_negative_fixable_count=0,
            false_negative_near_miss_count=0,
            false_negative_correct_rejection_count=0,
        ),
        ollama=RunOllamaMetrics(
            model="qwen2.5:7b-instruct",
            degraded=False,
            request_count=request_count,
            success_count=request_count,
            failure_count=0,
            outer_timeout_count=0,
            warm_hit_rate=0.0,
            median_wall_duration_seconds=None,
            p95_wall_duration_seconds=None,
            useful_action_count=round(request_count * useful_actions_per_request, 3),
            useful_actions_per_request=useful_actions_per_request,
            quality_counters={},
        ),
        timing=RunTimingMetrics(
            started_at=generated_at.isoformat(),
            ended_at=generated_at.isoformat(),
            duration_seconds=0.0,
            time_to_first_validated_job_seconds=None,
        ),
    )


def test_build_run_improvement_analysis_prioritizes_failed_run_repair(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    older = _make_scorecard(
        run_id="run-older",
        status="completed",
        generated_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=7,
    )
    current = _make_scorecard(
        run_id="run-current",
        status="failed",
        generated_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=2,
        query_timeout_count=8,
    )
    save_run_scorecard(settings.data_dir, older)
    save_run_scorecard(settings.data_dir, current)

    analysis = build_run_improvement_analysis(
        settings,
        iteration_number=1,
        run_id="run-current",
        failure_message="Workflow failed: Selenium login redirect.",
    )

    assert analysis.selected_theme == "failure_repair"
    assert analysis.top_patterns[0].summary == "Workflow failed: Selenium login redirect."
    assert analysis.analyzed_run_ids[:2] == ["run-current", "run-older"]


def test_build_run_improvement_analysis_prioritizes_query_timeout_burden(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    older = _make_scorecard(
        run_id="run-older",
        status="completed",
        generated_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=8,
        query_timeout_count=2,
    )
    current = _make_scorecard(
        run_id="run-current",
        status="completed",
        generated_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=3,
        query_timeout_count=12,
        query_skipped_timeout_budget_count=4,
        request_count=0,
    )
    save_run_scorecard(settings.data_dir, older)
    save_run_scorecard(settings.data_dir, current)

    analysis = build_run_improvement_analysis(
        settings,
        iteration_number=2,
        run_id="run-current",
    )

    assert analysis.selected_theme == "query_timeout_burden"
    assert analysis.metric_deltas["query_timeout_delta"] > 0


def test_ensure_baseline_commit_commits_dirty_main(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Oleg Y"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "olegyazvin@gmail.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "README.md").write_text("hello world\n", encoding="utf-8")
    settings = build_settings(tmp_path)

    baseline_hash = ensure_baseline_commit(settings)

    head_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_output = subprocess.run(
        ["git", "status", "--short"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert baseline_hash == head_hash
    assert status_output == ""


def test_invoke_codex_iteration_captures_new_session_id(tmp_path: Path, monkeypatch) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    session_reads = [
        [],
        ["session-1"],
    ]

    monkeypatch.setattr("job_agent.auto_loop._read_codex_session_ids", lambda _settings: session_reads.pop(0))
    monkeypatch.setattr("job_agent.auto_loop._codex_command_parts", lambda _settings: ["codex"])

    def fake_run(command, **kwargs):
        last_message_path = Path(command[command.index("-o") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("Updated timeout suppression logic.", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = invoke_codex_iteration(
        settings,
        iteration_number=1,
        prompt="Implement a bounded fix.",
        selected_theme="query_timeout_burden",
        session_id=None,
    )

    assert result.status == "succeeded"
    assert result.session_id == "session-1"
    assert result.summary == "Updated timeout suppression logic."


def test_run_autonomous_loop_stops_on_validation_failure_after_failed_run(tmp_path: Path, monkeypatch) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    current = _make_scorecard(
        run_id="run-1",
        status="failed",
        generated_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=2,
        query_timeout_count=5,
    )
    save_run_scorecard(settings.data_dir, current)

    async def fake_run_workflow_attempt(_settings: Settings, *, timeout_seconds: int | None = None):
        return "run-1", "failed", "Workflow failed: mock failure"

    monkeypatch.setattr("job_agent.auto_loop._ensure_main_branch", lambda _settings: None)
    monkeypatch.setattr("job_agent.auto_loop.ensure_baseline_commit", lambda _settings, iteration_number=1: "baseline123")
    monkeypatch.setattr("job_agent.auto_loop._run_workflow_attempt", fake_run_workflow_attempt)
    monkeypatch.setattr(
        "job_agent.auto_loop.invoke_codex_iteration",
        lambda *_args, **_kwargs: CodexIterationResult(
            iteration_number=1,
            generated_at=datetime.now(UTC),
            status="succeeded",
            session_id="session-1",
            selected_theme="failure_repair",
            summary="Applied a repair.",
        ),
    )
    monkeypatch.setattr(
        "job_agent.auto_loop.run_validation_commands",
        lambda _settings, *, iteration_number: [
            ValidationCommandResult(
                command="PYTHONPATH=src .venv/bin/pytest -q",
                passed=False,
                exit_code=1,
                output_path=str(_settings.auto_loop_dir / f"iteration-{iteration_number:02d}" / "validation-1.log"),
            )
        ],
    )
    monkeypatch.setattr("job_agent.auto_loop._git_head", lambda _settings: "head123")

    state = asyncio.run(
        run_autonomous_loop(
            settings,
            attempts=2,
            show_gui=False,
            timeout_seconds=1,
        )
    )

    assert isinstance(state, AutoLoopState)
    assert state.status == "failed"
    assert state.completed_attempts == 1
    assert state.codex_session_id == "session-1"
    assert state.latest_validation_result == "validation_failed"
    assert len(state.iterations) == 1
    assert state.iterations[0].run_status == "failed"
    assert state.iterations[0].validation_passed is False
