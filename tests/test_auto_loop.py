import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess

from job_agent.auto_loop import (
    build_run_improvement_analysis,
    ensure_baseline_commit,
    invoke_codex_iteration,
    render_codex_prompt,
    run_autonomous_loop,
)
from job_agent.config import Settings
from job_agent.models import (
    AutoLoopState,
    CodexIterationResult,
    ImprovementPattern,
    RunDiscoveryMetrics,
    RunImprovementAnalysis,
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
    novel_validated_jobs_count: int | None = None,
    reacquired_validated_jobs_count: int = 0,
    total_current_validated_jobs_count: int | None = None,
    jobs_with_messages_count: int = 0,
    fresh_new_leads_count: int = 0,
    raw_near_miss_count: int = 0,
    actionable_near_miss_count: int = 0,
    replayed_seed_leads_count: int = 0,
    query_timeout_count: int = 0,
    query_skipped_timeout_budget_count: int = 0,
    discovery_efficiency: float = 0.0,
    missing_salary_count: int = 0,
    fetch_non_200_count: int = 0,
    not_remote_count: int = 0,
    stale_posting_count: int = 0,
    request_count: int = 0,
    success_count: int | None = None,
    failure_count: int | None = None,
    useful_actions_per_request: float = 0.0,
) -> RunScorecard:
    novel_validated_jobs_count = validated_jobs_count if novel_validated_jobs_count is None else novel_validated_jobs_count
    total_current_validated_jobs_count = (
        novel_validated_jobs_count + reacquired_validated_jobs_count
        if total_current_validated_jobs_count is None
        else total_current_validated_jobs_count
    )
    success_count = request_count if success_count is None else success_count
    failure_count = max(0, request_count - success_count) if failure_count is None else failure_count
    return RunScorecard(
        run_id=run_id,
        generated_at=generated_at,
        status=status,
        outcome=RunOutcomeMetrics(
            validated_jobs_count=novel_validated_jobs_count,
            novel_validated_jobs_count=novel_validated_jobs_count,
            reacquired_validated_jobs_count=reacquired_validated_jobs_count,
            total_current_validated_jobs_count=total_current_validated_jobs_count,
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
            validated_jobs_count=novel_validated_jobs_count,
            validated_yield=0.0,
            novel_validated_jobs_count=novel_validated_jobs_count,
            novel_validated_yield=0.0,
            reacquired_validated_jobs_count=reacquired_validated_jobs_count,
            total_current_validated_jobs_count=total_current_validated_jobs_count,
            reacquisition_attempt_count=0,
            reacquired_jobs_suppressed_count=0,
            reacquisition_yield=0.0,
            coverage_retention_rate=None,
            jobs_with_messages_count=jobs_with_messages_count,
            message_coverage_rate=0.0,
            raw_near_miss_count=raw_near_miss_count,
            actionable_near_miss_count=actionable_near_miss_count,
            actionable_near_miss_yield=0.0,
            company_mismatch_count=0,
            not_specific_job_page_count=0,
            missing_salary_count=missing_salary_count,
            fetch_non_200_count=fetch_non_200_count,
            stale_posting_count=stale_posting_count,
            not_remote_count=not_remote_count,
            false_negative_fixable_count=0,
            false_negative_near_miss_count=0,
            false_negative_correct_rejection_count=0,
        ),
        ollama=RunOllamaMetrics(
            model="qwen2.5:7b-instruct",
            degraded=False,
            request_count=request_count,
            success_count=success_count,
            failure_count=failure_count,
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


def test_build_run_improvement_analysis_does_not_prioritize_productive_timeout_scouting(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    current = _make_scorecard(
        run_id="run-current",
        status="completed",
        generated_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        validated_jobs_count=1,
        fresh_new_leads_count=12,
        query_timeout_count=8,
        query_skipped_timeout_budget_count=1,
    ).model_copy(
        update={
            "discovery": RunDiscoveryMetrics(
                unique_leads_discovered_count=12,
                fresh_new_leads_count=12,
                replayed_seed_leads_count=0,
                repeated_failed_leads_suppressed_count=0,
                executed_query_count=18,
                query_timeout_count=8,
                query_skipped_timeout_budget_count=1,
                zero_yield_pass_count=0,
                discovery_efficiency=0.667,
                new_companies_discovered_count=2,
                new_boards_discovered_count=1,
                official_board_leads_count=3,
                companies_with_ai_pm_leads_count=3,
                company_discovery_yield=0.5,
            )
        }
    )
    save_run_scorecard(settings.data_dir, current)

    analysis = build_run_improvement_analysis(
        settings,
        iteration_number=1,
        run_id="run-current",
    )

    assert analysis.selected_theme != "query_timeout_burden"
    assert all(pattern.key != "query_timeout_burden" for pattern in analysis.top_patterns)


def test_build_run_improvement_analysis_detects_plateau_breaker_after_repeated_ollama_idle_streak(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        scorecard = _make_scorecard(
            run_id=f"run-{index + 1}",
            status="completed",
            generated_at=datetime(2026, 3, 30, 12, index, tzinfo=UTC),
            fresh_new_leads_count=11,
            request_count=0,
        )
        save_run_scorecard(settings.data_dir, scorecard)

    auto_loop_state_payload = {
        "enabled": True,
        "status": "running",
        "target_attempts": 20,
        "completed_attempts": 5,
        "current_iteration": 6,
        "current_run_id": "run-5",
        "codex_session_id": "session-1",
        "baseline_commit_hash": "baseline123",
        "latest_commit_hash": "latest123",
        "latest_validation_result": "passed",
        "last_failure_summary": None,
        "started_at": "2026-03-30T12:00:00Z",
        "updated_at": "2026-03-30T12:05:00Z",
        "iterations": [
            {
                "iteration_number": idx + 1,
                "run_id": f"run-{idx + 1}",
                "run_status": "completed",
                "started_at": f"2026-03-30T12:0{idx}:00Z",
                "completed_at": f"2026-03-30T12:0{idx}:30Z",
                "selected_theme": "ollama_idle",
                "analysis_path": None,
                "prompt_path": None,
                "result_path": None,
                "codex_log_path": None,
                "codex_last_message_path": None,
                "commit_hash": f"commit-{idx + 1}",
                "validation_passed": True,
            }
            for idx in range(5)
        ],
    }
    settings.auto_loop_state_path.write_text(json.dumps(auto_loop_state_payload), encoding="utf-8")

    analysis = build_run_improvement_analysis(
        settings,
        iteration_number=6,
        run_id="run-5",
    )

    assert analysis.selected_theme == "plateau_breaker"
    assert analysis.recent_selected_themes[:3] == ["ollama_idle", "ollama_idle", "ollama_idle"]
    prompt = render_codex_prompt(analysis, iteration_number=6)
    assert "Recent selected themes:" in prompt
    assert "Recent theme streak: `ollama_idle` x5." in prompt
    assert "Do not spend this iteration on another micro-tweak in that same area" in prompt


def test_build_run_improvement_analysis_does_not_prioritize_ollama_without_recent_successes(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    older = _make_scorecard(
        run_id="run-older",
        status="completed",
        generated_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=4,
        request_count=2,
        success_count=0,
        failure_count=2,
    )
    current = _make_scorecard(
        run_id="run-current",
        status="completed",
        generated_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=4,
        request_count=2,
        success_count=0,
        failure_count=2,
    )
    save_run_scorecard(settings.data_dir, older)
    save_run_scorecard(settings.data_dir, current)

    analysis = build_run_improvement_analysis(
        settings,
        iteration_number=2,
        run_id="run-current",
    )

    assert analysis.selected_theme != "ollama_idle"
    assert all(pattern.key != "ollama_idle" for pattern in analysis.top_patterns)


def test_build_run_improvement_analysis_prioritizes_validation_hard_filter_burden(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    older = _make_scorecard(
        run_id="run-older",
        status="completed",
        generated_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=10,
    )
    current = _make_scorecard(
        run_id="run-current",
        status="completed",
        generated_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=11,
        not_remote_count=2,
        stale_posting_count=1,
        missing_salary_count=1,
        request_count=0,
    )
    save_run_scorecard(settings.data_dir, older)
    save_run_scorecard(settings.data_dir, current)

    analysis = build_run_improvement_analysis(
        settings,
        iteration_number=2,
        run_id="run-current",
    )

    assert analysis.selected_theme == "validation_hard_filter_burden"
    assert analysis.current_metrics["not_remote_count"] == 2
    assert analysis.current_metrics["stale_posting_count"] == 1
    assert analysis.current_metrics["missing_salary_count"] == 1


def test_build_run_improvement_analysis_detects_diversification_gap_when_coverage_exists(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    older = _make_scorecard(
        run_id="run-older",
        status="completed",
        generated_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        novel_validated_jobs_count=0,
        reacquired_validated_jobs_count=2,
        total_current_validated_jobs_count=2,
        fresh_new_leads_count=8,
    )
    current = _make_scorecard(
        run_id="run-current",
        status="completed",
        generated_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        novel_validated_jobs_count=0,
        reacquired_validated_jobs_count=2,
        total_current_validated_jobs_count=2,
        fresh_new_leads_count=7,
    )
    save_run_scorecard(settings.data_dir, older)
    save_run_scorecard(settings.data_dir, current)

    analysis = build_run_improvement_analysis(
        settings,
        iteration_number=2,
        run_id="run-current",
    )

    assert analysis.selected_theme == "diversification_gap"
    assert analysis.current_metrics["total_current_validated_jobs_count"] == 2
    assert analysis.current_metrics["novel_validated_jobs_count"] == 0


def test_build_run_improvement_analysis_detects_coverage_regression(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    older = _make_scorecard(
        run_id="run-older",
        status="completed",
        generated_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        novel_validated_jobs_count=1,
        reacquired_validated_jobs_count=3,
        total_current_validated_jobs_count=4,
        fresh_new_leads_count=8,
    )
    current = _make_scorecard(
        run_id="run-current",
        status="completed",
        generated_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        novel_validated_jobs_count=0,
        reacquired_validated_jobs_count=1,
        total_current_validated_jobs_count=1,
        fresh_new_leads_count=8,
    )
    save_run_scorecard(settings.data_dir, older)
    save_run_scorecard(settings.data_dir, current)

    analysis = build_run_improvement_analysis(
        settings,
        iteration_number=2,
        run_id="run-current",
    )

    assert any(pattern.key == "coverage_regression" for pattern in analysis.top_patterns)
    assert analysis.metric_deltas["total_current_validated_jobs_delta"] < 0


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

    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
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
    assert captured["command"][:6] == ["codex", "exec", "--full-auto", "--skip-git-repo-check", "--json", "-o"]
    assert captured["command"][-3:] == ["-C", str(settings.project_root), "-"]


def test_invoke_codex_iteration_resume_places_exec_options_before_subcommand(tmp_path: Path, monkeypatch) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("job_agent.auto_loop._read_codex_session_ids", lambda _settings: ["session-1"])
    monkeypatch.setattr("job_agent.auto_loop._codex_command_parts", lambda _settings: ["codex"])

    captured: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        last_message_path = Path(command[command.index("-o") + 1])
        last_message_path.parent.mkdir(parents=True, exist_ok=True)
        last_message_path.write_text("Retried timeout suppression logic.", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = invoke_codex_iteration(
        settings,
        iteration_number=2,
        prompt="Retry the bounded fix.",
        selected_theme="query_timeout_burden",
        session_id="session-1",
    )

    assert result.status == "succeeded"
    assert captured["command"][:6] == ["codex", "exec", "--full-auto", "--skip-git-repo-check", "--json", "-o"]
    assert "resume" in captured["command"]
    assert captured["command"][captured["command"].index("resume") + 1 : captured["command"].index("resume") + 3] == ["session-1", "-"]
    assert "-C" not in captured["command"][captured["command"].index("resume") :]


def test_run_autonomous_loop_continues_after_validation_failure(tmp_path: Path, monkeypatch) -> None:
    settings = build_settings(tmp_path)
    settings.auto_loop_max_workflow_reruns_per_iteration = 0
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    first = _make_scorecard(
        run_id="run-1",
        status="failed",
        generated_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=2,
        query_timeout_count=5,
    )
    second = _make_scorecard(
        run_id="run-2",
        status="completed",
        generated_at=datetime(2026, 3, 31, 13, 0, tzinfo=UTC),
        fresh_new_leads_count=6,
        query_timeout_count=2,
    )
    save_run_scorecard(settings.data_dir, first)
    save_run_scorecard(settings.data_dir, second)

    workflow_results = iter(
        [
            ("run-1", "failed", "Workflow failed: mock failure"),
            ("run-2", "completed", None),
        ]
    )

    async def fake_run_workflow_attempt(_settings: Settings, *, timeout_seconds: int | None = None):
        return next(workflow_results)

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
    validation_results = iter(
        [
            [
                ValidationCommandResult(
                    command="PYTHONPATH=src .venv/bin/pytest -q",
                    passed=False,
                    exit_code=1,
                    output_path=str(settings.auto_loop_dir / "iteration-01" / "validation-1.log"),
                )
            ],
            [
                ValidationCommandResult(
                    command="PYTHONPATH=src .venv/bin/pytest -q",
                    passed=False,
                    exit_code=1,
                    output_path=str(settings.auto_loop_dir / "iteration-01" / "validation-2.log"),
                )
            ],
            [
                ValidationCommandResult(
                    command="PYTHONPATH=src .venv/bin/pytest -q",
                    passed=False,
                    exit_code=1,
                    output_path=str(settings.auto_loop_dir / "iteration-01" / "validation-3.log"),
                )
            ],
            [
                ValidationCommandResult(
                    command="PYTHONPATH=src .venv/bin/pytest -q",
                    passed=True,
                    exit_code=0,
                    output_path=str(settings.auto_loop_dir / "iteration-02" / "validation-1.log"),
                )
            ],
        ]
    )
    monkeypatch.setattr(
        "job_agent.auto_loop.run_validation_commands",
        lambda _settings, *, iteration_number: next(validation_results),
    )
    monkeypatch.setattr("job_agent.auto_loop._git_head", lambda _settings: "head123")
    dirty_results = iter([True])
    monkeypatch.setattr("job_agent.auto_loop._working_tree_dirty", lambda _settings: next(dirty_results))
    monkeypatch.setattr("job_agent.auto_loop._commit_all_changes", lambda _settings, _message: "commit-2")

    state = asyncio.run(
        run_autonomous_loop(
            settings,
            attempts=2,
            show_gui=False,
            timeout_seconds=1,
        )
    )

    assert isinstance(state, AutoLoopState)
    assert state.status == "stopped"
    assert state.completed_attempts == 2
    assert state.codex_session_id == "session-1"
    assert state.latest_validation_result == "passed"
    assert len(state.iterations) == 2
    assert state.iterations[0].run_status == "failed"
    assert state.iterations[0].validation_passed is False
    assert state.iterations[1].run_status == "completed"
    assert state.iterations[1].validation_passed is True
    assert state.latest_commit_hash == "commit-2"


def test_build_run_improvement_analysis_can_prioritize_company_discovery_gap(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    current = _make_scorecard(
        run_id="run-current",
        status="completed",
        generated_at=datetime(2026, 4, 3, 12, 0, tzinfo=UTC),
        fresh_new_leads_count=4,
    ).model_copy(
        update={
            "discovery": RunDiscoveryMetrics(
                unique_leads_discovered_count=4,
                fresh_new_leads_count=4,
                replayed_seed_leads_count=0,
                repeated_failed_leads_suppressed_count=0,
                executed_query_count=6,
                query_timeout_count=0,
                query_skipped_timeout_budget_count=0,
                zero_yield_pass_count=0,
                discovery_efficiency=0.667,
                new_companies_discovered_count=0,
                new_boards_discovered_count=0,
                official_board_leads_count=0,
                companies_with_ai_pm_leads_count=0,
                company_discovery_yield=0.0,
            )
        }
    )
    save_run_scorecard(settings.data_dir, current)

    analysis = build_run_improvement_analysis(settings, iteration_number=1, run_id="run-current")

    assert analysis.selected_theme == "company_discovery_gap"


def test_run_autonomous_loop_records_workflow_rerun_evidence(tmp_path: Path, monkeypatch) -> None:
    settings = build_settings(tmp_path)
    settings.auto_loop_max_workflow_reruns_per_iteration = 1
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    workflow_results = iter(
        [
            ("run-1", "completed", None),
            ("run-1-rerun", "completed", None),
        ]
    )

    async def fake_run_workflow_attempt(_settings: Settings, *, timeout_seconds: int | None = None):
        return next(workflow_results)

    def fake_analysis(_settings: Settings, *, iteration_number: int, run_id: str | None, failure_message: str | None = None):
        return RunImprovementAnalysis(
            iteration_number=iteration_number,
            generated_at=datetime.now(UTC),
            target_run_id=run_id,
            analyzed_run_ids=[run_id] if run_id else [],
            recent_selected_themes=[],
            current_run_status="completed",
            current_metrics={"fresh_new_leads_count": 3, "query_timeout_count": 1, "total_current_validated_jobs_count": 0},
            metric_deltas={},
            top_patterns=[
                ImprovementPattern(
                    key="company_discovery_gap",
                    summary="Improve company discovery.",
                    severity_score=10.0,
                    evidence={},
                )
            ],
            selected_theme="company_discovery_gap",
            selected_summary="Improve company discovery.",
            acceptance_checks=[],
            artifact_paths={},
        )

    metric_snapshots = {
        "run-1": {
            "fresh_new_leads_count": 3,
            "query_timeout_count": 1,
            "total_current_validated_jobs_count": 0,
            "jobs_with_messages_count": 0,
            "actionable_near_miss_count": 0,
            "novel_validated_jobs_count": 0,
            "reacquired_validated_jobs_count": 0,
            "new_companies_discovered_count": 0,
            "new_boards_discovered_count": 0,
            "official_board_leads_count": 0,
            "principal_ai_pm_salary_presumption_count": 0,
        },
        "run-1-rerun": {
            "fresh_new_leads_count": 5,
            "query_timeout_count": 1,
            "total_current_validated_jobs_count": 0,
            "jobs_with_messages_count": 0,
            "actionable_near_miss_count": 0,
            "novel_validated_jobs_count": 0,
            "reacquired_validated_jobs_count": 0,
            "new_companies_discovered_count": 2,
            "new_boards_discovered_count": 1,
            "official_board_leads_count": 1,
            "principal_ai_pm_salary_presumption_count": 0,
        },
    }

    monkeypatch.setattr("job_agent.auto_loop._ensure_main_branch", lambda _settings: None)
    monkeypatch.setattr("job_agent.auto_loop.ensure_baseline_commit", lambda _settings, iteration_number=1: "baseline123")
    monkeypatch.setattr("job_agent.auto_loop._run_workflow_attempt", fake_run_workflow_attempt)
    monkeypatch.setattr("job_agent.auto_loop.build_run_improvement_analysis", fake_analysis)
    monkeypatch.setattr("job_agent.auto_loop._scorecard_metrics_for_run", lambda _settings, run_id: metric_snapshots.get(run_id or "", {}))
    monkeypatch.setattr(
        "job_agent.auto_loop.invoke_codex_iteration",
        lambda *_args, **_kwargs: CodexIterationResult(
            iteration_number=1,
            generated_at=datetime.now(UTC),
            status="succeeded",
            session_id="session-1",
            selected_theme="company_discovery_gap",
            summary="Expanded company discovery.",
        ),
    )
    monkeypatch.setattr(
        "job_agent.auto_loop.run_validation_commands",
        lambda _settings, *, iteration_number: [
            ValidationCommandResult(
                command="PYTHONPATH=src .venv/bin/pytest -q",
                passed=True,
                exit_code=0,
                output_path=str(settings.auto_loop_dir / "iteration-01" / "validation-1.log"),
            )
        ],
    )
    monkeypatch.setattr("job_agent.auto_loop._git_head", lambda _settings: "head123")
    monkeypatch.setattr("job_agent.auto_loop._working_tree_dirty", lambda _settings: True)
    monkeypatch.setattr("job_agent.auto_loop._commit_all_changes", lambda _settings, _message: "commit-1")

    state = asyncio.run(run_autonomous_loop(settings, attempts=1, show_gui=False, timeout_seconds=1))

    assert state.status == "stopped"
    result_path = Path(state.iterations[0].result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["workflow_rerun_count"] == 1
    assert payload["workflow_rerun_run_ids"] == ["run-1-rerun"]
    assert payload["metric_comparison"]["after_fresh_new_leads_count"] == 5
