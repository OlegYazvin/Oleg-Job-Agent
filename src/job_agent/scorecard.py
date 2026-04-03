from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .history import load_run_history_entries
from .models import (
    JobOutreachBundle,
    OllamaRunSummary,
    RunDiscoveryMetrics,
    RunManifest,
    RunOllamaMetrics,
    RunOutcomeMetrics,
    RunScorecard,
    RunTimingMetrics,
    RunValidationMetrics,
    SearchDiagnostics,
)


RUN_SCORECARD_LATEST_FILENAME = "run-scorecard-latest.json"
RUN_SCORECARDS_HISTORY_FILENAME = "run-scorecards.jsonl"
ACTIONABLE_NEAR_MISS_REASON_CODES = {
    "missing_salary",
    "fetch_non_200",
    "resolution_missing",
    "resolution_blocked_url",
    "remote_unclear",
    "missing_posted_date",
    "validation_timeout",
    "resolution_timeout",
}
LOW_TRUST_NEAR_MISS_HOST_FRAGMENTS = (
    "mediabistro.com",
    "smartrecruiterscareers.com",
    "remotejobshive.com",
    "thatstartupjob.com",
    "remoterocketship.com",
    "jobgether.com",
    "tracxn.com",
)
_USEFUL_OLLAMA_COUNTER_KEYS = (
    "merged_count",
    "kept_count",
    "assisted_validated_jobs_count",
    "assisted_actionable_near_misses_count",
    "schema_valid_count",
    "used_suggestion_count",
)


def _scorecard_latest_path(data_dir: Path) -> Path:
    return data_dir / RUN_SCORECARD_LATEST_FILENAME


def _scorecard_history_path(data_dir: Path) -> Path:
    return data_dir / RUN_SCORECARDS_HISTORY_FILENAME


def _load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except Exception:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "\n".join(json.dumps(entry, sort_keys=True) for entry in entries)
    path.write_text(f"{rendered}\n" if rendered else "", encoding="utf-8")


def _parse_datetime(value: str | None) -> datetime | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _manifest_dict(manifest: RunManifest | Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {}
    if isinstance(manifest, RunManifest):
        return manifest.model_dump(mode="json")
    return dict(manifest)


def _diagnostics_dict(search_diagnostics: SearchDiagnostics | Mapping[str, Any] | None) -> dict[str, Any]:
    if search_diagnostics is None:
        return {}
    if isinstance(search_diagnostics, SearchDiagnostics):
        return search_diagnostics.model_dump(mode="json")
    return dict(search_diagnostics)


def _ollama_summary_dict(summary: OllamaRunSummary | Mapping[str, Any] | None) -> dict[str, Any]:
    if summary is None:
        return {}
    if isinstance(summary, OllamaRunSummary):
        return summary.model_dump(mode="json")
    return dict(summary)


def _reacquired_job_items(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, Mapping)]


def _bundle_jobs(bundles: list[JobOutreachBundle] | list[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    if not bundles:
        return []
    items: list[dict[str, Any]] = []
    for bundle in bundles:
        if isinstance(bundle, JobOutreachBundle):
            items.append(bundle.job.model_dump(mode="json"))
            continue
        if not isinstance(bundle, Mapping):
            continue
        job = bundle.get("job")
        if isinstance(job, Mapping):
            items.append(dict(job))
    return items


def _near_miss_items(
    near_miss_payload: Mapping[str, Any] | None,
    search_diagnostics: SearchDiagnostics | Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if isinstance(near_miss_payload, Mapping):
        items = near_miss_payload.get("items")
        if isinstance(items, list):
            return [dict(item) for item in items if isinstance(item, Mapping)]
    diagnostics = _diagnostics_dict(search_diagnostics)
    items = diagnostics.get("near_misses")
    if isinstance(items, list):
        return [dict(item) for item in items if isinstance(item, Mapping)]
    return []


def _failure_counts(search_diagnostics: SearchDiagnostics | Mapping[str, Any] | None) -> Counter[str]:
    diagnostics = _diagnostics_dict(search_diagnostics)
    counts: Counter[str] = Counter()
    for item in diagnostics.get("failures", []):
        if not isinstance(item, Mapping):
            continue
        reason_code = str(item.get("reason_code") or "").strip()
        if reason_code:
            counts[reason_code] += 1
    return counts


def _false_negative_counts(search_diagnostics: SearchDiagnostics | Mapping[str, Any] | None) -> Counter[str]:
    diagnostics = _diagnostics_dict(search_diagnostics)
    counts: Counter[str] = Counter()
    for item in diagnostics.get("false_negative_audit", []):
        if not isinstance(item, Mapping):
            continue
        verdict = str(item.get("verdict") or "").strip()
        if verdict:
            counts[verdict] += 1
    return counts


def _host_from_urls(*urls: str | None) -> str:
    for url in urls:
        normalized = str(url or "").strip()
        if not normalized.startswith(("http://", "https://")):
            continue
        return (urlparse(normalized).netloc or "").lower()
    return ""


def _is_actionable_near_miss(item: Mapping[str, Any]) -> bool:
    reason_code = str(item.get("reason_code") or "").strip()
    if reason_code not in ACTIONABLE_NEAR_MISS_REASON_CODES:
        return False
    source_quality_score = int(item.get("source_quality_score") or 0)
    if source_quality_score < 8:
        return False
    host = _host_from_urls(
        str(item.get("direct_job_url") or "").strip() or None,
        str(item.get("source_url") or "").strip() or None,
    )
    if not host:
        return False
    if any(fragment in host for fragment in LOW_TRUST_NEAR_MISS_HOST_FRAGMENTS):
        return False
    return True


def _time_to_first_validated_job_seconds(status_payload: Mapping[str, Any] | None) -> float | None:
    if not status_payload:
        return None
    started_at = _parse_datetime(str(status_payload.get("started_at") or "").strip() or None)
    if started_at is None:
        return None
    for event in list(status_payload.get("recent_events", []) or []):
        if not isinstance(event, Mapping):
            continue
        message = str(event.get("message") or "")
        if not message.startswith("Accepted job "):
            continue
        event_time = _parse_datetime(str(event.get("time") or "").strip() or None)
        if event_time is None:
            continue
        return round(max(0.0, (event_time - started_at).total_seconds()), 3)
    return None


def build_run_scorecard(
    *,
    run_id: str,
    status: str,
    manifest: RunManifest | Mapping[str, Any] | None = None,
    bundles: list[JobOutreachBundle] | list[Mapping[str, Any]] | None = None,
    reacquired_jobs_payload: Mapping[str, Any] | None = None,
    search_diagnostics: SearchDiagnostics | Mapping[str, Any] | None = None,
    near_miss_payload: Mapping[str, Any] | None = None,
    ollama_summary_payload: OllamaRunSummary | Mapping[str, Any] | None = None,
    status_payload: Mapping[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> RunScorecard:
    manifest_payload = _manifest_dict(manifest)
    diagnostics_payload = _diagnostics_dict(search_diagnostics)
    ollama_payload = _ollama_summary_dict(ollama_summary_payload)
    near_miss_items = _near_miss_items(near_miss_payload, search_diagnostics)
    bundle_jobs = _bundle_jobs(bundles)
    reacquired_items = _reacquired_job_items(reacquired_jobs_payload)
    failure_counts = _failure_counts(search_diagnostics)
    false_negative_counts = _false_negative_counts(search_diagnostics)

    generated_at = generated_at or _parse_datetime(str(manifest_payload.get("generated_at") or "").strip() or None) or datetime.now(UTC)
    unique_leads_discovered = int(
        diagnostics_payload.get("unique_leads_discovered")
        or (status_payload or {}).get("metrics", {}).get("unique_leads_discovered")
        or manifest_payload.get("jobs_found_by_search")
        or 0
    )
    replayed_seed_leads_count = int(diagnostics_payload.get("seed_replayed_lead_count") or 0)
    reacquisition_attempt_count = int(diagnostics_payload.get("reacquisition_attempt_count") or 0)
    reacquired_jobs_suppressed_count = int(diagnostics_payload.get("reacquired_jobs_suppressed_count") or 0)
    novel_validated_jobs_count = int(
        manifest_payload.get("novel_validated_jobs_count")
        or manifest_payload.get("jobs_kept_after_validation")
        or len(bundle_jobs)
        or (status_payload or {}).get("metrics", {}).get("jobs_kept_after_validation")
        or (status_payload or {}).get("metrics", {}).get("qualifying_jobs")
        or 0
    )
    reacquired_validated_jobs_count = int(
        manifest_payload.get("reacquired_validated_jobs_count")
        or len(reacquired_items)
        or 0
    )
    total_current_validated_jobs_count = int(
        manifest_payload.get("total_current_validated_jobs_count")
        or (novel_validated_jobs_count + reacquired_validated_jobs_count)
    )
    validated_jobs_count = novel_validated_jobs_count
    fresh_new_leads_count = max(
        0,
        unique_leads_discovered - replayed_seed_leads_count - reacquisition_attempt_count - reacquired_jobs_suppressed_count,
    )
    jobs_with_messages_count = int(
        manifest_payload.get("jobs_with_any_messages")
        or (status_payload or {}).get("metrics", {}).get("jobs_with_any_messages")
        or 0
    )
    validated_jobs_with_inferred_salary_count = sum(
        1
        for item in [*bundle_jobs, *reacquired_items]
        if bool(item.get("salary_inferred"))
    )
    principal_ai_pm_salary_presumption_count = sum(
        1
        for item in [*bundle_jobs, *reacquired_items]
        if str(item.get("salary_inference_kind") or "").strip() == "salary_presumed_from_principal_ai_pm"
    )
    actionable_near_miss_count = sum(1 for item in near_miss_items if _is_actionable_near_miss(item))
    raw_near_miss_count = len(near_miss_items)
    executed_query_count = sum(
        int(item.get("query_count") or 0)
        for item in diagnostics_payload.get("passes", [])
        if isinstance(item, Mapping)
    )
    zero_yield_pass_count = sum(
        1
        for item in diagnostics_payload.get("passes", [])
        if isinstance(item, Mapping) and int(item.get("unique_leads_discovered") or 0) == 0
    )
    company_lead_counts = {
        str(key): int(value or 0)
        for key, value in dict(diagnostics_payload.get("company_lead_counts") or {}).items()
        if str(key).strip()
    }
    source_adapter_yields = {
        str(key): int(value or 0)
        for key, value in dict(diagnostics_payload.get("source_adapter_yields") or {}).items()
        if str(key).strip()
    }
    total_company_leads = sum(company_lead_counts.values())
    company_concentration_top_10_share = (
        round(sum(sorted(company_lead_counts.values(), reverse=True)[:10]) / total_company_leads, 3)
        if total_company_leads
        else 0.0
    )
    new_companies_discovered_count = int(diagnostics_payload.get("new_companies_discovered_count") or 0)
    official_board_crawl_attempt_count = int(diagnostics_payload.get("official_board_crawl_attempt_count") or 0)
    official_board_crawl_success_count = int(diagnostics_payload.get("official_board_crawl_success_count") or 0)

    started_at = str((status_payload or {}).get("started_at") or "").strip() or None
    ended_at = (
        str((status_payload or {}).get("updated_at") or "").strip()
        or str(manifest_payload.get("generated_at") or "").strip()
        or None
    )
    duration_seconds = None
    started_dt = _parse_datetime(started_at)
    ended_dt = _parse_datetime(ended_at)
    if started_dt is not None and ended_dt is not None:
        duration_seconds = round(max(0.0, (ended_dt - started_dt).total_seconds()), 3)

    quality_counters = {
        str(key): round(float(value), 3)
        for key, value in dict(ollama_payload.get("quality_counters") or {}).items()
        if isinstance(value, (int, float))
    }
    useful_action_count = round(
        sum(float(quality_counters.get(key) or 0.0) for key in _USEFUL_OLLAMA_COUNTER_KEYS),
        3,
    )
    ollama_request_count = int(ollama_payload.get("request_count") or 0)

    return RunScorecard(
        run_id=run_id,
        generated_at=generated_at,
        status=status,
        outcome=RunOutcomeMetrics(
            validated_jobs_count=validated_jobs_count,
            novel_validated_jobs_count=novel_validated_jobs_count,
            reacquired_validated_jobs_count=reacquired_validated_jobs_count,
            total_current_validated_jobs_count=total_current_validated_jobs_count,
            validated_jobs_with_inferred_salary_count=validated_jobs_with_inferred_salary_count,
            principal_ai_pm_salary_presumption_count=principal_ai_pm_salary_presumption_count,
            jobs_with_messages_count=jobs_with_messages_count,
            unique_leads_discovered_count=unique_leads_discovered,
            fresh_new_leads_count=fresh_new_leads_count,
            actionable_near_miss_count=actionable_near_miss_count,
            raw_near_miss_count=raw_near_miss_count,
        ),
        discovery=RunDiscoveryMetrics(
            unique_leads_discovered_count=unique_leads_discovered,
            fresh_new_leads_count=fresh_new_leads_count,
            replayed_seed_leads_count=replayed_seed_leads_count,
            reacquisition_attempt_count=reacquisition_attempt_count,
            reacquired_jobs_suppressed_count=reacquired_jobs_suppressed_count,
            new_companies_discovered_count=new_companies_discovered_count,
            new_boards_discovered_count=int(diagnostics_payload.get("new_boards_discovered_count") or 0),
            official_board_leads_count=int(diagnostics_payload.get("official_board_leads_count") or 0),
            companies_with_ai_pm_leads_count=int(diagnostics_payload.get("companies_with_ai_pm_leads_count") or 0),
            repeated_failed_leads_suppressed_count=int(failure_counts.get("repeated_failed_lead") or 0),
            executed_query_count=executed_query_count,
            query_timeout_count=int(failure_counts.get("query_timeout") or 0),
            query_skipped_timeout_budget_count=int(failure_counts.get("query_skipped_timeout_budget") or 0),
            zero_yield_pass_count=zero_yield_pass_count,
            discovery_efficiency=round(fresh_new_leads_count / executed_query_count, 3) if executed_query_count else 0.0,
            company_discovery_yield=round(
                int(diagnostics_payload.get("companies_with_ai_pm_leads_count") or 0)
                / max(1, new_companies_discovered_count),
                3,
            ),
            company_concentration_top_10_share=company_concentration_top_10_share,
            frontier_tasks_consumed_count=int(diagnostics_payload.get("frontier_tasks_consumed_count") or 0),
            frontier_backlog_count=int(diagnostics_payload.get("frontier_backlog_count") or 0),
            official_board_crawl_success_rate=round(
                official_board_crawl_success_count / official_board_crawl_attempt_count,
                3,
            )
            if official_board_crawl_attempt_count
            else 0.0,
            new_company_to_fresh_lead_yield=round(fresh_new_leads_count / new_companies_discovered_count, 3)
            if new_companies_discovered_count
            else 0.0,
            source_adapter_yields=source_adapter_yields,
        ),
        validation=RunValidationMetrics(
            validated_jobs_count=validated_jobs_count,
            validated_yield=round(validated_jobs_count / fresh_new_leads_count, 3) if fresh_new_leads_count else 0.0,
            novel_validated_jobs_count=novel_validated_jobs_count,
            novel_validated_yield=round(novel_validated_jobs_count / fresh_new_leads_count, 3) if fresh_new_leads_count else 0.0,
            reacquired_validated_jobs_count=reacquired_validated_jobs_count,
            total_current_validated_jobs_count=total_current_validated_jobs_count,
            reacquisition_attempt_count=reacquisition_attempt_count,
            reacquired_jobs_suppressed_count=reacquired_jobs_suppressed_count,
            reacquisition_yield=round(reacquired_validated_jobs_count / reacquisition_attempt_count, 3) if reacquisition_attempt_count else 0.0,
            coverage_retention_rate=None,
            validated_jobs_with_inferred_salary_count=validated_jobs_with_inferred_salary_count,
            principal_ai_pm_salary_presumption_count=principal_ai_pm_salary_presumption_count,
            official_roles_missed_count=int(diagnostics_payload.get("official_roles_missed_count") or 0),
            jobs_with_messages_count=jobs_with_messages_count,
            message_coverage_rate=round(jobs_with_messages_count / validated_jobs_count, 3) if validated_jobs_count else 0.0,
            raw_near_miss_count=raw_near_miss_count,
            actionable_near_miss_count=actionable_near_miss_count,
            actionable_near_miss_yield=round(actionable_near_miss_count / fresh_new_leads_count, 3) if fresh_new_leads_count else 0.0,
            company_mismatch_count=int(failure_counts.get("company_mismatch") or 0),
            not_specific_job_page_count=int(failure_counts.get("not_specific_job_page") or 0),
            missing_salary_count=int(failure_counts.get("missing_salary") or 0),
            fetch_non_200_count=int(failure_counts.get("fetch_non_200") or 0),
            stale_posting_count=int(failure_counts.get("stale_posting") or 0),
            not_remote_count=int(failure_counts.get("not_remote") or 0),
            false_negative_fixable_count=int(false_negative_counts.get("fixable") or 0),
            false_negative_near_miss_count=int(false_negative_counts.get("near_miss") or 0),
            false_negative_correct_rejection_count=int(false_negative_counts.get("correct_rejection") or 0),
        ),
        ollama=RunOllamaMetrics(
            model=str(dict(ollama_payload.get("tuning_profile") or {}).get("model") or "") or None,
            degraded=bool(dict(ollama_payload.get("tuning_profile") or {}).get("degraded") or False),
            request_count=ollama_request_count,
            success_count=int(ollama_payload.get("success_count") or 0),
            failure_count=int(ollama_payload.get("failure_count") or 0),
            outer_timeout_count=int(ollama_payload.get("outer_timeout_count") or 0),
            warm_hit_rate=round(float(ollama_payload.get("warm_hit_rate") or 0.0), 3),
            median_wall_duration_seconds=(
                round(float(ollama_payload["median_wall_duration_seconds"]), 3)
                if isinstance(ollama_payload.get("median_wall_duration_seconds"), (int, float))
                else None
            ),
            p95_wall_duration_seconds=(
                round(float(ollama_payload["p95_wall_duration_seconds"]), 3)
                if isinstance(ollama_payload.get("p95_wall_duration_seconds"), (int, float))
                else None
            ),
            useful_action_count=useful_action_count,
            useful_actions_per_request=round(useful_action_count / ollama_request_count, 3) if ollama_request_count else 0.0,
            quality_counters=quality_counters,
        ),
        timing=RunTimingMetrics(
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            time_to_first_validated_job_seconds=_time_to_first_validated_job_seconds(status_payload),
        ),
        message_docx_path=str(manifest_payload.get("message_docx_path") or "") or None,
        summary_docx_path=str(manifest_payload.get("summary_docx_path") or "") or None,
        near_miss_docx_path=str(manifest_payload.get("near_miss_docx_path") or "") or None,
        near_miss_json_path=str(manifest_payload.get("near_miss_json_path") or "") or None,
        ollama_summary_json_path=str(manifest_payload.get("ollama_summary_json_path") or "") or None,
        company_discovery_json_path=str(manifest_payload.get("company_discovery_json_path") or "") or None,
        company_discovery_frontier_json_path=(
            str(manifest_payload.get("company_discovery_frontier_json_path") or "") or None
        ),
        company_discovery_crawl_history_json_path=(
            str(manifest_payload.get("company_discovery_crawl_history_json_path") or "") or None
        ),
        company_discovery_audit_json_path=(
            str(manifest_payload.get("company_discovery_audit_json_path") or "") or None
        ),
    )


def save_run_scorecard(data_dir: Path, scorecard: RunScorecard) -> None:
    latest_path = _scorecard_latest_path(data_dir)
    history_path = _scorecard_history_path(data_dir)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    previous_entries = [
        entry
        for entry in load_run_scorecard_entries(data_dir, bootstrap=False)
        if entry.run_id != scorecard.run_id
    ]
    if previous_entries:
        baseline = sum(entry.outcome.total_current_validated_jobs_count for entry in previous_entries[:20]) / min(len(previous_entries), 20)
        coverage_retention_rate = round(
            scorecard.outcome.total_current_validated_jobs_count / baseline,
            3,
        ) if baseline else None
        scorecard = scorecard.model_copy(
            update={
                "validation": scorecard.validation.model_copy(
                    update={"coverage_retention_rate": coverage_retention_rate}
                )
            }
        )
    payload = scorecard.model_dump(mode="json")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    entries_by_run_id = {
        entry.run_id: entry
        for entry in previous_entries
    }
    entries_by_run_id[scorecard.run_id] = scorecard
    ordered_entries = sorted(
        entries_by_run_id.values(),
        key=lambda item: (
            _parse_datetime(item.timing.ended_at or item.timing.started_at or item.generated_at.isoformat()) or datetime.min.replace(tzinfo=UTC)
        ),
        reverse=True,
    )
    _write_jsonl(history_path, [entry.model_dump(mode="json") for entry in ordered_entries])


def _bootstrap_run_scorecards(data_dir: Path) -> list[RunScorecard]:
    existing_entries = {
        entry.run_id: entry
        for entry in load_run_scorecard_entries(data_dir, bootstrap=False)
    }
    run_history = load_run_history_entries(data_dir)
    artifact_payloads: dict[str, dict[str, Any]] = {}
    for artifact_path in sorted(data_dir.glob("run-*.json")):
        payload = _load_json(artifact_path, default={})
        if not isinstance(payload, dict):
            continue
        manifest_payload = payload.get("manifest", payload)
        if not isinstance(manifest_payload, Mapping):
            continue
        run_id = str(manifest_payload.get("run_id") or "").strip()
        if run_id:
            artifact_payloads[run_id] = payload

    changed = False
    for entry in run_history:
        run_id = str(entry.get("run_id") or "").strip()
        if not run_id or run_id in existing_entries:
            continue
        artifact = artifact_payloads.get(run_id, {})
        manifest_payload = artifact.get("manifest") if isinstance(artifact.get("manifest"), Mapping) else None
        scorecard = build_run_scorecard(
            run_id=run_id,
            status=str(entry.get("status") or "completed"),
            manifest=manifest_payload,
            bundles=artifact.get("bundles") if isinstance(artifact.get("bundles"), list) else None,
            reacquired_jobs_payload=artifact.get("reacquired_jobs_payload") if isinstance(artifact.get("reacquired_jobs_payload"), Mapping) else None,
            search_diagnostics=artifact.get("search_diagnostics") if isinstance(artifact.get("search_diagnostics"), Mapping) else None,
            near_miss_payload=artifact.get("near_misses") if isinstance(artifact.get("near_misses"), Mapping) else None,
            ollama_summary_payload=artifact.get("ollama_summary") if isinstance(artifact.get("ollama_summary"), Mapping) else None,
            status_payload={
                "started_at": entry.get("started_at"),
                "updated_at": entry.get("ended_at"),
                "metrics": {
                    "jobs_found_by_search": entry.get("jobs_found_by_search"),
                    "jobs_kept_after_validation": entry.get("jobs_kept_after_validation"),
                    "jobs_with_any_messages": entry.get("jobs_with_any_messages"),
                },
            },
            generated_at=_parse_datetime(str(entry.get("ended_at") or "").strip() or None),
        )
        existing_entries[run_id] = scorecard
        changed = True
    ordered_entries = sorted(
        existing_entries.values(),
        key=lambda item: (
            _parse_datetime(item.timing.ended_at or item.timing.started_at or item.generated_at.isoformat()) or datetime.min.replace(tzinfo=UTC)
        ),
        reverse=True,
    )
    if changed:
        _write_jsonl(_scorecard_history_path(data_dir), [entry.model_dump(mode="json") for entry in ordered_entries])
        if ordered_entries:
            _scorecard_latest_path(data_dir).write_text(
                json.dumps(ordered_entries[0].model_dump(mode="json"), indent=2),
                encoding="utf-8",
            )
    return ordered_entries


def load_run_scorecard_entries(data_dir: Path, *, bootstrap: bool = True) -> list[RunScorecard]:
    if bootstrap:
        return _bootstrap_run_scorecards(data_dir)
    entries: list[RunScorecard] = []
    for payload in _read_jsonl(_scorecard_history_path(data_dir)):
        try:
            entries.append(RunScorecard.model_validate(payload))
        except Exception:
            continue
    return entries


def load_latest_run_scorecard(data_dir: Path) -> RunScorecard | None:
    payload = _load_json(_scorecard_latest_path(data_dir), default=None)
    if isinstance(payload, Mapping):
        try:
            return RunScorecard.model_validate(payload)
        except Exception:
            pass
    entries = load_run_scorecard_entries(data_dir)
    return entries[0] if entries else None


def save_failed_run_scorecard(
    data_dir: Path,
    *,
    run_id: str,
    status_payload: Mapping[str, Any] | None = None,
    failure_message: str | None = None,
) -> RunScorecard:
    search_diagnostics = _load_json(data_dir / "search-diagnostics-latest.json", default={})
    if not isinstance(search_diagnostics, Mapping) or str(search_diagnostics.get("run_id") or "").strip() != run_id:
        search_diagnostics = None
    near_miss_payload = _load_json(data_dir / "near-misses-latest.json", default={})
    if not isinstance(near_miss_payload, Mapping) or str(near_miss_payload.get("run_id") or "").strip() != run_id:
        near_miss_payload = None
    reacquired_jobs_payload = _load_json(data_dir / "reacquired-jobs-latest.json", default=None)
    if not isinstance(reacquired_jobs_payload, Mapping) or str(reacquired_jobs_payload.get("run_id") or "").strip() != run_id:
        reacquired_jobs_payload = None
    ollama_summary_payload = _load_json(data_dir / "ollama-summary-latest.json", default={})
    if not isinstance(ollama_summary_payload, Mapping) or str(ollama_summary_payload.get("run_id") or "").strip() != run_id:
        ollama_summary_payload = None
    scorecard = build_run_scorecard(
        run_id=run_id,
        status="failed",
        reacquired_jobs_payload=reacquired_jobs_payload,
        search_diagnostics=search_diagnostics,
        near_miss_payload=near_miss_payload,
        ollama_summary_payload=ollama_summary_payload,
        status_payload=status_payload,
        generated_at=_parse_datetime(str((status_payload or {}).get("updated_at") or "").strip() or None) or datetime.now(UTC),
    )
    save_run_scorecard(data_dir, scorecard)
    return scorecard
