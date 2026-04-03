from job_agent.dashboard import (
    _compute_dashboard_minimum_size,
    _format_scorecard_summary,
    _format_live_metrics_lines,
    _scorecard_detail_lines,
)


def test_compute_dashboard_minimum_size_respects_defaults() -> None:
    assert _compute_dashboard_minimum_size(900, 600) == (1080, 720)


def test_compute_dashboard_minimum_size_expands_for_required_layout() -> None:
    assert _compute_dashboard_minimum_size(1420, 830) == (1420, 830)


def test_format_scorecard_summary_renders_primary_metrics() -> None:
    rendered = _format_scorecard_summary(
        {
            "outcome": {
                "validated_jobs_count": 2,
                "novel_validated_jobs_count": 2,
                "reacquired_validated_jobs_count": 1,
                "total_current_validated_jobs_count": 3,
                "jobs_with_messages_count": 1,
                "fresh_new_leads_count": 14,
                "actionable_near_miss_count": 3,
            },
            "discovery": {
                "query_timeout_count": 4,
                "new_companies_discovered_count": 3,
                "new_boards_discovered_count": 5,
                "frontier_tasks_consumed_count": 6,
                "frontier_backlog_count": 8,
            },
            "validation": {"validated_yield": 0.143},
        }
    )

    assert "Latest scorecard" in rendered
    assert "Novel validated jobs: 2" in rendered
    assert "Current validated coverage: 3" in rendered
    assert "Reacquired jobs: 1" in rendered
    assert "Jobs with messages: 1" in rendered
    assert "Fresh leads: 14" in rendered
    assert "Actionable near-misses: 3" in rendered
    assert "New companies: 3" in rendered
    assert "New boards: 5" in rendered
    assert "Frontier consumed: 6" in rendered
    assert "Backlog: 8" in rendered


def test_scorecard_detail_lines_include_discovery_and_ollama_context() -> None:
    lines = _scorecard_detail_lines(
        {
            "outcome": {
                "novel_validated_jobs_count": 1,
                "reacquired_validated_jobs_count": 2,
                "total_current_validated_jobs_count": 3,
                "fresh_new_leads_count": 9,
                "actionable_near_miss_count": 2,
                "validated_jobs_with_inferred_salary_count": 1,
                "principal_ai_pm_salary_presumption_count": 1,
            },
            "discovery": {
                "replayed_seed_leads_count": 4,
                "reacquisition_attempt_count": 2,
                "reacquired_jobs_suppressed_count": 1,
                "repeated_failed_leads_suppressed_count": 7,
                "query_timeout_count": 3,
                "discovery_efficiency": 1.5,
                "new_companies_discovered_count": 2,
                "new_boards_discovered_count": 3,
                "official_board_leads_count": 1,
                "company_discovery_yield": 1.0,
                "company_concentration_top_10_share": 0.8,
                "frontier_tasks_consumed_count": 6,
                "frontier_backlog_count": 8,
                "official_board_crawl_success_rate": 0.5,
                "new_company_to_fresh_lead_yield": 4.5,
                "source_adapter_yields": {"ashby": 2, "directory_source": 4},
            },
            "validation": {
                "message_coverage_rate": 0.5,
                "novel_validated_yield": 0.111,
                "reacquisition_yield": 1.0,
                "official_roles_missed_count": 1,
            },
            "ollama": {"request_count": 2, "useful_actions_per_request": 0.5},
            "timing": {"duration_seconds": 720.0},
        }
    )

    assert "Novel validated jobs: 1" in lines
    assert "Current validated coverage: 3" in lines
    assert "Reacquired validated jobs: 2" in lines
    assert "Fresh new leads: 9" in lines
    assert "Validated jobs with inferred salary: 1" in lines
    assert "Principal AI PM salary presumptions: 1" in lines
    assert "New companies discovered: 2" in lines
    assert "Official roles missed: 1" in lines
    assert "Frontier tasks consumed: 6" in lines
    assert "Official-board crawl success rate: 0.5" in lines
    assert "Replay seeds: 4" in lines
    assert "Reacquisition attempts: 2" in lines
    assert "Ollama requests: 2" in lines


def test_format_live_metrics_lines_includes_loop_progress() -> None:
    lines = _format_live_metrics_lines(
        {
            "auto_loop_iteration": 7,
            "auto_loop_target_attempts": 20,
            "auto_loop_status": "running",
            "target_job_count": 10,
            "unique_leads_discovered": 12,
            "qualifying_jobs": 2,
        }
    )

    assert lines[0] == "Loop: 7/20 (running)"
    assert "Found by search: 12" in lines
    assert "Kept after validation: 2" in lines
