from datetime import datetime
import json
from pathlib import Path

from job_agent.models import (
    FalseNegativeAuditEntry,
    NearMissJob,
    OllamaTuningProfile,
    RunManifest,
    SearchDiagnostics,
    SearchFailure,
    SearchPassSummary,
)
from job_agent.scorecard import (
    build_run_scorecard,
    load_latest_run_scorecard,
    load_run_scorecard_entries,
    save_failed_run_scorecard,
    save_run_scorecard,
)


def test_build_run_scorecard_splits_fresh_leads_and_actionable_near_misses() -> None:
    manifest = RunManifest(
        run_id="run-scorecard",
        generated_at=datetime(2026, 3, 31, 1, 0, 0),
        message_docx_path="/tmp/messages.docx",
        summary_docx_path="/tmp/summary.docx",
        jobs_found_by_search=25,
        jobs_kept_after_validation=2,
        jobs_with_any_messages=1,
    )
    diagnostics = SearchDiagnostics(
        run_id="run-scorecard",
        minimum_qualifying_jobs=5,
        unique_leads_discovered=25,
        seed_replayed_lead_count=10,
        new_companies_discovered_count=3,
        new_boards_discovered_count=4,
        official_board_leads_count=2,
        companies_with_ai_pm_leads_count=2,
        official_roles_missed_count=1,
        failures=[
            SearchFailure(stage="discovery", reason_code="query_timeout", detail="timed out"),
            SearchFailure(stage="filter", reason_code="repeated_failed_lead", detail="replayed dead lead"),
            SearchFailure(stage="validation", reason_code="missing_salary", detail="no salary"),
            SearchFailure(stage="validation", reason_code="company_mismatch", detail="wrong company"),
        ],
        passes=[
            SearchPassSummary(attempt_number=1, unique_leads_discovered=25, qualifying_jobs=2, query_count=8),
            SearchPassSummary(attempt_number=2, unique_leads_discovered=0, qualifying_jobs=2, query_count=6),
        ],
        near_misses=[
            NearMissJob(
                company_name="Databricks",
                role_title="Staff Product Manager, AI Platform",
                reason_code="missing_salary",
                detail="No salary listed.",
                why_close="Strong company page.",
                source_url="https://builtin.com/job/123",
                direct_job_url="https://www.databricks.com/company/careers/product/staff-product-manager-ai-platform-1",
                source_quality_score=10,
            ),
            NearMissJob(
                company_name="Jobs",
                role_title="Senior Product Manager, AI-Driven CX",
                reason_code="missing_salary",
                detail="Mirror page only.",
                why_close="Weak mirror page.",
                source_url="https://www.mediabistro.com/jobs/123",
                direct_job_url="https://www.mediabistro.com/jobs/123",
                source_quality_score=10,
            ),
        ],
        false_negative_audit=[
            FalseNegativeAuditEntry(
                reason_code="resolution_missing",
                verdict="fixable",
                detail="Could be recovered.",
                notes="Retry direct resolution.",
            ),
            FalseNegativeAuditEntry(
                reason_code="missing_salary",
                verdict="near_miss",
                detail="Comp missing.",
                notes="Worth review.",
            ),
        ],
    )
    tuning_profile = OllamaTuningProfile(
        model="qwen2.5:7b-instruct",
        keep_alive="15m",
        num_ctx=768,
        num_batch=1,
        num_predict=160,
    )
    scorecard = build_run_scorecard(
        run_id="run-scorecard",
        status="completed",
        manifest=manifest,
        search_diagnostics=diagnostics,
        ollama_summary_payload={
            "run_id": "run-scorecard",
            "tuning_profile": tuning_profile.model_dump(mode="json"),
            "request_count": 2,
            "success_count": 2,
            "failure_count": 0,
            "outer_timeout_count": 0,
            "warm_hit_rate": 1.0,
            "median_wall_duration_seconds": 4.2,
            "p95_wall_duration_seconds": 5.1,
            "quality_counters": {"kept_count": 1, "merged_count": 1},
        },
        status_payload={
            "started_at": "2026-03-31T01:00:00+00:00",
            "updated_at": "2026-03-31T01:30:00+00:00",
            "recent_events": [
                {
                    "time": "2026-03-31T01:10:00+00:00",
                    "stage": "search",
                    "message": "Accepted job 1/5: Acme | Staff Product Manager, AI",
                }
            ],
        },
        generated_at=datetime(2026, 3, 31, 1, 30, 0),
    )

    assert scorecard.outcome.fresh_new_leads_count == 15
    assert scorecard.outcome.novel_validated_jobs_count == 2
    assert scorecard.outcome.reacquired_validated_jobs_count == 0
    assert scorecard.outcome.total_current_validated_jobs_count == 2
    assert scorecard.outcome.actionable_near_miss_count == 1
    assert scorecard.discovery.executed_query_count == 14
    assert scorecard.discovery.zero_yield_pass_count == 1
    assert scorecard.discovery.new_companies_discovered_count == 3
    assert scorecard.discovery.new_boards_discovered_count == 4
    assert scorecard.discovery.official_board_leads_count == 2
    assert scorecard.discovery.company_discovery_yield == 0.667
    assert scorecard.validation.novel_validated_yield == scorecard.validation.validated_yield
    assert scorecard.validation.message_coverage_rate == 0.5
    assert scorecard.validation.official_roles_missed_count == 1
    assert scorecard.ollama.useful_action_count == 2.0
    assert scorecard.ollama.useful_actions_per_request == 1.0
    assert scorecard.timing.time_to_first_validated_job_seconds == 600.0


def test_build_run_scorecard_tracks_reacquired_validated_jobs_separately() -> None:
    manifest = RunManifest(
        run_id="run-reacquired",
        generated_at=datetime(2026, 3, 31, 1, 0, 0),
        message_docx_path="/tmp/messages.docx",
        summary_docx_path="/tmp/summary.docx",
        jobs_found_by_search=9,
        jobs_kept_after_validation=1,
        jobs_with_any_messages=1,
        novel_validated_jobs_count=1,
        reacquired_validated_jobs_count=2,
        total_current_validated_jobs_count=3,
    )
    diagnostics = SearchDiagnostics(
        run_id="run-reacquired",
        minimum_qualifying_jobs=5,
        unique_leads_discovered=9,
        seed_replayed_lead_count=2,
        reacquisition_attempt_count=2,
        reacquired_jobs_suppressed_count=1,
        failures=[],
        passes=[SearchPassSummary(attempt_number=1, unique_leads_discovered=9, qualifying_jobs=1, query_count=7)],
    )

    scorecard = build_run_scorecard(
        run_id="run-reacquired",
        status="completed",
        manifest=manifest,
        reacquired_jobs_payload={
            "run_id": "run-reacquired",
            "reacquired_validated_jobs_count": 2,
            "items": [
                {"company_name": "Acme AI", "role_title": "Staff Product Manager, AI"},
                {"company_name": "Bravo AI", "role_title": "Principal Product Manager, AI"},
            ],
        },
        search_diagnostics=diagnostics,
    )

    assert scorecard.outcome.validated_jobs_count == 1
    assert scorecard.outcome.novel_validated_jobs_count == 1
    assert scorecard.outcome.reacquired_validated_jobs_count == 2
    assert scorecard.outcome.total_current_validated_jobs_count == 3
    assert scorecard.discovery.reacquisition_attempt_count == 2
    assert scorecard.discovery.reacquired_jobs_suppressed_count == 1
    assert scorecard.outcome.fresh_new_leads_count == 4
    assert scorecard.validation.reacquisition_yield == 1.0


def test_build_run_scorecard_counts_inferred_salary_validated_jobs() -> None:
    manifest = RunManifest(
        run_id="run-inferred-salary",
        generated_at=datetime(2026, 4, 2, 1, 0, 0),
        message_docx_path="/tmp/messages.docx",
        summary_docx_path="/tmp/summary.docx",
        jobs_found_by_search=5,
        jobs_kept_after_validation=1,
        jobs_with_any_messages=0,
    )
    scorecard = build_run_scorecard(
        run_id="run-inferred-salary",
        status="completed",
        manifest=manifest,
        bundles=[
            {
                "job": {
                    "company_name": "ButterflyMX",
                    "role_title": "Principal Product Manager, AI",
                    "salary_inferred": True,
                    "salary_inference_kind": "salary_presumed_from_principal_ai_pm",
                }
            }
        ],
    )

    assert scorecard.outcome.validated_jobs_with_inferred_salary_count == 1
    assert scorecard.outcome.principal_ai_pm_salary_presumption_count == 1
    assert scorecard.validation.validated_jobs_with_inferred_salary_count == 1
    assert scorecard.validation.principal_ai_pm_salary_presumption_count == 1


def test_run_scorecard_history_bootstraps_from_run_artifacts(tmp_path: Path) -> None:
    (tmp_path / "run-history.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "run-boot",
                    "status": "completed",
                    "started_at": "2026-03-31T01:00:00+00:00",
                    "ended_at": "2026-03-31T01:20:00+00:00",
                    "jobs_found_by_search": 12,
                    "jobs_kept_after_validation": 1,
                    "jobs_with_any_messages": 0,
                    "message_docx_path": "/tmp/messages.docx",
                    "summary_docx_path": "/tmp/summary.docx",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (tmp_path / "run-20260331-012000.json").write_text(
        json.dumps(
            {
                "manifest": {
                    "run_id": "run-boot",
                    "generated_at": "2026-03-31T01:20:00+00:00",
                    "message_docx_path": "/tmp/messages.docx",
                    "summary_docx_path": "/tmp/summary.docx",
                    "jobs_found_by_search": 12,
                    "jobs_kept_after_validation": 1,
                    "jobs_with_any_messages": 0,
                },
                "search_diagnostics": {
                    "run_id": "run-boot",
                    "minimum_qualifying_jobs": 5,
                    "unique_leads_discovered": 12,
                    "seed_replayed_lead_count": 3,
                    "failures": [{"stage": "discovery", "reason_code": "query_timeout", "detail": "timed out"}],
                    "passes": [{"attempt_number": 1, "unique_leads_discovered": 12, "qualifying_jobs": 1, "query_count": 7}],
                    "near_misses": [],
                    "false_negative_audit": [],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    entries = load_run_scorecard_entries(tmp_path)

    assert len(entries) == 1
    assert entries[0].run_id == "run-boot"
    assert entries[0].outcome.fresh_new_leads_count == 9
    assert entries[0].discovery.query_timeout_count == 1
    assert load_latest_run_scorecard(tmp_path).run_id == "run-boot"


def test_save_failed_run_scorecard_uses_matching_latest_artifacts(tmp_path: Path) -> None:
    (tmp_path / "search-diagnostics-latest.json").write_text(
        json.dumps(
            {
                "run_id": "run-failed",
                "minimum_qualifying_jobs": 5,
                "unique_leads_discovered": 7,
                "seed_replayed_lead_count": 2,
                "failures": [{"stage": "discovery", "reason_code": "query_timeout", "detail": "timed out"}],
                "passes": [],
                "near_misses": [],
                "false_negative_audit": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    scorecard = save_failed_run_scorecard(
        tmp_path,
        run_id="run-failed",
        status_payload={
            "started_at": "2026-03-31T02:00:00+00:00",
            "updated_at": "2026-03-31T02:05:00+00:00",
            "metrics": {"unique_leads_discovered": 7, "qualifying_jobs": 1},
        },
        failure_message="Timed out.",
    )

    assert scorecard.status == "failed"
    assert scorecard.outcome.validated_jobs_count == 1
    assert scorecard.outcome.fresh_new_leads_count == 5
    assert load_latest_run_scorecard(tmp_path).run_id == "run-failed"


def test_save_run_scorecard_round_trips_history(tmp_path: Path) -> None:
    scorecard = build_run_scorecard(
        run_id="run-save",
        status="completed",
        manifest={
            "run_id": "run-save",
            "generated_at": "2026-03-31T03:00:00+00:00",
            "message_docx_path": "/tmp/messages.docx",
            "summary_docx_path": "/tmp/summary.docx",
            "jobs_found_by_search": 4,
            "jobs_kept_after_validation": 1,
            "jobs_with_any_messages": 1,
        },
        search_diagnostics={
            "run_id": "run-save",
            "minimum_qualifying_jobs": 5,
            "unique_leads_discovered": 4,
            "seed_replayed_lead_count": 1,
            "failures": [],
            "passes": [],
            "near_misses": [],
            "false_negative_audit": [],
        },
        generated_at=datetime(2026, 3, 31, 3, 0, 0),
    )

    save_run_scorecard(tmp_path, scorecard)
    entries = load_run_scorecard_entries(tmp_path, bootstrap=False)

    assert len(entries) == 1
    assert entries[0].run_id == "run-save"
    assert entries[0].outcome.fresh_new_leads_count == 3


def test_save_run_scorecard_handles_mixed_naive_and_aware_timestamps(tmp_path: Path) -> None:
    history_path = tmp_path / "run-scorecards.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "run_id": "run-naive",
                "generated_at": "2026-03-31T03:00:00",
                "status": "completed",
                "outcome": {"validated_jobs_count": 0},
                "discovery": {},
                "validation": {},
                "ollama": {},
                "timing": {
                    "started_at": "2026-03-31T03:00:00",
                    "ended_at": "2026-03-31T03:10:00",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    scorecard = build_run_scorecard(
        run_id="run-aware",
        status="completed",
        manifest={
            "run_id": "run-aware",
            "generated_at": "2026-03-31T04:00:00+00:00",
            "message_docx_path": "/tmp/messages.docx",
            "summary_docx_path": "/tmp/summary.docx",
            "jobs_found_by_search": 2,
            "jobs_kept_after_validation": 0,
            "jobs_with_any_messages": 0,
        },
        search_diagnostics={
            "run_id": "run-aware",
            "minimum_qualifying_jobs": 5,
            "unique_leads_discovered": 2,
            "seed_replayed_lead_count": 0,
            "failures": [],
            "passes": [],
            "near_misses": [],
            "false_negative_audit": [],
        },
        generated_at=datetime(2026, 3, 31, 4, 0, 0),
    )

    save_run_scorecard(tmp_path, scorecard)
    entries = load_run_scorecard_entries(tmp_path, bootstrap=False)

    assert [entry.run_id for entry in entries] == ["run-aware", "run-naive"]
