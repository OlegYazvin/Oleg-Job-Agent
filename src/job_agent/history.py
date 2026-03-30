from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import JobOutreachBundle, RunManifest, SearchFailure


WATCHLIST_REASON_CODES = {
    "resolution_missing",
    "resolution_blocked_url",
    "fetch_non_200",
    "not_specific_job_page",
    "missing_salary",
    "salary_below_min",
    "salary_not_base",
    "not_remote",
    "remote_unclear",
    "stale_posting",
    "missing_posted_date",
    "company_mismatch",
    "validation_timeout",
    "resolution_timeout",
}
SMALL_COMPANY_FRIENDLY_HOST_FRAGMENTS = (
    "jobs.ashbyhq.com",
    "jobs.lever.co",
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.workable.com",
    "careers.bamboohr.com",
    "recruiting.paylocity.com",
    "jobs.dayforcehcm.com",
    "jobs.smartrecruiters.com",
)
ENTERPRISE_HEAVY_HOST_FRAGMENTS = (
    "myworkdayjobs.com",
    "careers.workday.com",
    "icims.com",
    "jobvite.com",
    "adp.com",
)
AGGREGATOR_HOST_FRAGMENTS = (
    "linkedin.com",
    "builtin.com",
    "glassdoor.com",
    "indeed.com",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_company_key(company_name: str | None) -> str:
    return "".join(character.lower() for character in str(company_name or "") if character.isalnum())


def _normalize_job_history_key(job_key: str | None) -> str:
    normalized = str(job_key or "").strip()
    if not normalized:
        return ""
    try:
        from .job_search import _job_history_primary_key
    except Exception:
        return normalized
    try:
        return _job_history_primary_key(normalized)
    except Exception:
        return normalized


def _board_identifier(url: str | None) -> str | None:
    normalized = str(url or "").strip()
    if not normalized:
        return None
    try:
        from .job_search import _extract_company_board_identifier
    except Exception:
        return None
    try:
        return _extract_company_board_identifier(normalized)
    except Exception:
        return None


def _normalize_job_history_entries(job_history: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], bool]:
    normalized_entries: dict[str, dict[str, Any]] = {}
    changed = False
    for raw_key, raw_entry in dict(job_history or {}).items():
        if not isinstance(raw_entry, Mapping):
            changed = True
            continue
        normalized_key = _normalize_job_history_key(raw_entry.get("job_key") or raw_key)
        if not normalized_key:
            changed = True
            continue
        entry = dict(raw_entry)
        entry["job_key"] = normalized_key
        entry.setdefault("canonical_job_key", normalized_key)
        if normalized_key in normalized_entries:
            merged_entry = dict(normalized_entries[normalized_key])
            merged_entry.update({key: value for key, value in entry.items() if value not in (None, "", [], {})})
            merged_entry["first_reported_at"] = (
                merged_entry.get("first_reported_at")
                or entry.get("first_reported_at")
            )
            merged_entry["last_reported_at"] = (
                entry.get("last_reported_at")
                or merged_entry.get("last_reported_at")
            )
            merged_entry["report_count"] = max(
                int(normalized_entries[normalized_key].get("report_count") or 0),
                int(entry.get("report_count") or 0),
            )
            entry = merged_entry
            changed = True
        if normalized_key != str(raw_key).strip() or str(raw_entry.get("job_key") or "").strip() != normalized_key:
            changed = True
        normalized_entries[normalized_key] = entry
    if len(normalized_entries) != len(job_history):
        changed = True
    return normalized_entries, changed


def _company_host(*urls: str | None) -> str:
    for url in urls:
        normalized = str(url or "").strip()
        if normalized.startswith(("http://", "https://")):
            return (urlparse(normalized).netloc or "").lower()
    return ""


def _watchlist_priority_delta(host: str, reason_code: str) -> int:
    score = 1
    if any(fragment in host for fragment in SMALL_COMPANY_FRIENDLY_HOST_FRAGMENTS):
        score += 4
    elif host and not any(fragment in host for fragment in ENTERPRISE_HEAVY_HOST_FRAGMENTS + AGGREGATOR_HOST_FRAGMENTS):
        score += 2
    if any(fragment in host for fragment in ENTERPRISE_HEAVY_HOST_FRAGMENTS):
        score -= 1
    if any(fragment in host for fragment in AGGREGATOR_HOST_FRAGMENTS):
        score -= 2
    if reason_code in {"stale_posting", "not_remote", "remote_unclear", "missing_salary", "salary_below_min"}:
        score += 2
    if reason_code in {"resolution_missing", "resolution_blocked_url", "not_specific_job_page", "fetch_non_200"}:
        score += 1
    return score


def _update_company_history_entry(
    company_history: dict[str, dict[str, Any]],
    *,
    company_name: str | None,
    role_title: str | None,
    generated_at: str,
    run_id: str | None,
    message_docx_path: str | None,
    summary_docx_path: str | None,
    board_identifier: str | None = None,
) -> None:
    company_key = _normalize_company_key(company_name)
    if not company_key:
        return
    entry = dict(company_history.get(company_key) or {})
    first_reported_at = entry.get("first_reported_at") or generated_at
    previous_count = int(entry.get("report_count") or 0)
    if str(entry.get("last_run_id") or "").strip() == str(run_id or "").strip():
        report_count = max(previous_count, 1)
    else:
        report_count = previous_count + 1
    role_titles = [str(title).strip() for title in entry.get("role_titles", []) if str(title).strip()]
    if role_title and role_title not in role_titles:
        role_titles.append(role_title)
    board_identifiers = [str(item).strip() for item in entry.get("board_identifiers", []) if str(item).strip()]
    if board_identifier and board_identifier not in board_identifiers:
        board_identifiers.append(board_identifier)
    entry.update(
        {
            "company_key": company_key,
            "company_name": company_name,
            "role_titles": role_titles[:8],
            "board_identifiers": board_identifiers[:8],
            "first_reported_at": first_reported_at,
            "last_reported_at": generated_at,
            "last_run_id": run_id,
            "report_count": report_count,
            "message_docx_path": message_docx_path,
            "summary_docx_path": summary_docx_path,
        }
    )
    company_history[company_key] = entry


def _update_company_watchlist_entry(
    company_watchlist: dict[str, dict[str, Any]],
    *,
    company_name: str | None,
    reason_code: str,
    generated_at: str,
    source_url: str | None,
    direct_job_url: str | None,
    detail: str | None,
) -> None:
    company_key = _normalize_company_key(company_name)
    if not company_key or reason_code not in WATCHLIST_REASON_CODES:
        return
    host = _company_host(direct_job_url, source_url)
    entry = dict(company_watchlist.get(company_key) or {})
    first_seen_at = entry.get("first_seen_at") or generated_at
    source_hosts = [str(item).strip() for item in entry.get("source_hosts", []) if str(item).strip()]
    if host and host not in source_hosts:
        source_hosts.append(host)
    source_urls = [str(item).strip() for item in entry.get("source_urls", []) if str(item).strip()]
    for url in (direct_job_url, source_url):
        normalized = str(url or "").strip()
        if normalized.startswith(("http://", "https://")) and normalized not in source_urls:
            source_urls.append(normalized)
    board_identifiers = [str(item).strip() for item in entry.get("board_identifiers", []) if str(item).strip()]
    for url in (direct_job_url, source_url):
        board_identifier = _board_identifier(url)
        if board_identifier and board_identifier not in board_identifiers:
            board_identifiers.append(board_identifier)
    reason_counts = {
        str(key): int(value)
        for key, value in dict(entry.get("recent_rejection_reasons") or {}).items()
        if str(key).strip()
    }
    reason_counts[reason_code] = reason_counts.get(reason_code, 0) + 1
    entry.update(
        {
            "company_key": company_key,
            "company_name": company_name,
            "first_seen_at": first_seen_at,
            "last_seen_at": generated_at,
            "watch_count": int(entry.get("watch_count") or 0) + 1,
            "priority_score": int(entry.get("priority_score") or 0) + _watchlist_priority_delta(host, reason_code),
            "last_reason_code": reason_code,
            "last_detail": (detail or "")[:240],
            "source_hosts": source_hosts[:8],
            "source_urls": source_urls[:6],
            "board_identifiers": board_identifiers[:8],
            "recent_rejection_reasons": reason_counts,
        }
    )
    company_watchlist[company_key] = entry


def _bootstrap_histories_from_run_artifacts(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    run_history = _load_json(data_dir / "run-history.json", default=[])
    if not isinstance(run_history, list):
        run_history = []
    job_history = _load_json(data_dir / "job-history.json", default={})
    if not isinstance(job_history, dict):
        job_history = {}
    job_history, job_history_changed = _normalize_job_history_entries(job_history)

    seen_run_ids = {str(entry.get("run_id") or "").strip() for entry in run_history}
    changed = job_history_changed

    for artifact_path in sorted(data_dir.glob("run-*.json")):
        payload = _load_json(artifact_path, default={})
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest", payload)
        if not isinstance(manifest, dict):
            continue
        run_id = str(manifest.get("run_id") or artifact_path.stem).strip()
        generated_at = str(
            manifest.get("generated_at")
            or datetime.fromtimestamp(artifact_path.stat().st_mtime, tz=UTC).isoformat(timespec="seconds")
        )
        if run_id and run_id not in seen_run_ids:
            run_history.insert(
                0,
                {
                    "run_id": run_id,
                    "status": "completed",
                    "started_at": None,
                    "ended_at": generated_at,
                    "message": "Imported from historical run artifact.",
                    "jobs_found_by_search": int(manifest.get("jobs_found_by_search") or 0),
                    "jobs_kept_after_validation": int(manifest.get("jobs_kept_after_validation") or 0),
                    "jobs_with_any_messages": int(manifest.get("jobs_with_any_messages") or 0),
                    "message_docx_path": manifest.get("message_docx_path"),
                    "summary_docx_path": manifest.get("summary_docx_path"),
                },
            )
            seen_run_ids.add(run_id)
            changed = True

        for bundle in payload.get("bundles", []):
            if not isinstance(bundle, dict):
                continue
            job_payload = bundle.get("job")
            if not isinstance(job_payload, dict):
                continue
            job_key = _normalize_job_history_key(job_payload.get("resolved_job_url") or job_payload.get("direct_job_url"))
            if not job_key:
                continue
            if job_key in job_history:
                continue
            job_history[job_key] = {
                "job_key": job_key,
                "normalized_job_url": _normalize_job_history_key(job_payload.get("resolved_job_url") or job_payload.get("direct_job_url")),
                "canonical_job_key": job_key,
                "company_name": job_payload.get("company_name"),
                "role_title": job_payload.get("role_title"),
                "posted_date_iso": job_payload.get("posted_date_iso"),
                "posted_date_text": job_payload.get("posted_date_text"),
                "salary_text": job_payload.get("salary_text"),
                "first_reported_at": generated_at,
                "last_reported_at": generated_at,
                "last_run_id": run_id,
                "report_count": 1,
                "message_docx_path": manifest.get("message_docx_path"),
                "summary_docx_path": manifest.get("summary_docx_path"),
            }
            changed = True

    if changed:
        _write_json(data_dir / "run-history.json", run_history)
        _write_json(data_dir / "job-history.json", job_history)
    return run_history, job_history


def _bootstrap_company_files_from_run_artifacts(data_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    company_history_path = data_dir / "company-history.json"
    company_watchlist_path = data_dir / "company-watchlist.json"
    company_history = _load_json(company_history_path, default={})
    if not isinstance(company_history, dict):
        company_history = {}
    company_watchlist = _load_json(company_watchlist_path, default={})
    if not isinstance(company_watchlist, dict):
        company_watchlist = {}

    changed = not company_history_path.exists() or not company_watchlist_path.exists()
    for artifact_path in sorted(data_dir.glob("run-*.json")):
        payload = _load_json(artifact_path, default={})
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest", payload)
        if not isinstance(manifest, dict):
            continue
        run_id = str(manifest.get("run_id") or artifact_path.stem).strip()
        generated_at = str(
            manifest.get("generated_at")
            or datetime.fromtimestamp(artifact_path.stat().st_mtime, tz=UTC).isoformat(timespec="seconds")
        )

        for bundle in payload.get("bundles", []):
            if not isinstance(bundle, dict):
                continue
            job_payload = bundle.get("job")
            if not isinstance(job_payload, dict):
                continue
            _update_company_history_entry(
                company_history,
                company_name=str(job_payload.get("company_name") or "").strip() or None,
                role_title=str(job_payload.get("role_title") or "").strip() or None,
                generated_at=generated_at,
                run_id=run_id,
                message_docx_path=manifest.get("message_docx_path"),
                summary_docx_path=manifest.get("summary_docx_path"),
                board_identifier=_board_identifier(job_payload.get("resolved_job_url") or job_payload.get("direct_job_url")),
            )
            changed = True

        diagnostics = payload.get("search_diagnostics", {})
        failures = diagnostics.get("failures") if isinstance(diagnostics, dict) else None
        if isinstance(failures, list):
            for raw_failure in failures:
                if not isinstance(raw_failure, dict):
                    continue
                try:
                    failure = SearchFailure.model_validate(raw_failure)
                except Exception:
                    continue
                _update_company_watchlist_entry(
                    company_watchlist,
                    company_name=failure.company_name,
                    reason_code=failure.reason_code,
                    generated_at=generated_at,
                    source_url=failure.source_url,
                    direct_job_url=failure.direct_job_url,
                    detail=failure.detail,
                )
                changed = True

    if changed:
        _write_json(company_history_path, company_history)
        _write_json(company_watchlist_path, company_watchlist)
    return company_history, company_watchlist


def load_run_history_entries(data_dir: Path) -> list[dict[str, Any]]:
    run_history, _ = _bootstrap_histories_from_run_artifacts(data_dir)
    return run_history


def load_job_history_entries(data_dir: Path) -> dict[str, dict[str, Any]]:
    _, job_history = _bootstrap_histories_from_run_artifacts(data_dir)
    return job_history


def load_company_history_entries(data_dir: Path) -> dict[str, dict[str, Any]]:
    company_history, _ = _bootstrap_company_files_from_run_artifacts(data_dir)
    return company_history


def load_company_watchlist_entries(data_dir: Path) -> dict[str, dict[str, Any]]:
    _, company_watchlist = _bootstrap_company_files_from_run_artifacts(data_dir)
    return company_watchlist


def load_previously_reported_job_keys(data_dir: Path) -> set[str]:
    keys: set[str] = set()
    for job_key, entry in load_job_history_entries(data_dir).items():
        normalized = str(job_key).strip()
        if normalized:
            keys.add(normalized)
        if isinstance(entry, Mapping):
            canonical_key = str(entry.get("canonical_job_key") or "").strip()
            normalized_url = str(entry.get("normalized_job_url") or "").strip()
            if canonical_key:
                keys.add(canonical_key)
            if normalized_url:
                keys.add(normalized_url)
    return keys


def load_previously_reported_company_keys(data_dir: Path) -> set[str]:
    return {
        str(company_key).strip()
        for company_key in load_company_history_entries(data_dir).keys()
        if str(company_key).strip()
    }


def _upsert_run_history_entry(data_dir: Path, entry: Mapping[str, Any]) -> None:
    entries = load_run_history_entries(data_dir)
    run_id = str(entry.get("run_id") or "").strip()
    if run_id:
        entries = [existing for existing in entries if str(existing.get("run_id") or "").strip() != run_id]
    entries.insert(0, dict(entry))
    _write_json(data_dir / "run-history.json", entries)


def record_successful_run(
    data_dir: Path,
    *,
    run_id: str,
    manifest: RunManifest,
    bundles: list[JobOutreachBundle],
    status_payload: Mapping[str, Any] | None = None,
) -> None:
    generated_at_iso = manifest.generated_at.isoformat()
    _upsert_run_history_entry(
        data_dir,
        {
            "run_id": run_id,
            "status": "completed",
            "started_at": status_payload.get("started_at") if status_payload else None,
            "ended_at": status_payload.get("updated_at") if status_payload else generated_at_iso,
            "message": "Workflow completed successfully.",
            "jobs_found_by_search": manifest.jobs_found_by_search,
            "jobs_kept_after_validation": manifest.jobs_kept_after_validation,
            "jobs_with_any_messages": manifest.jobs_with_any_messages,
            "message_docx_path": manifest.message_docx_path,
            "summary_docx_path": manifest.summary_docx_path,
        },
    )

    job_history = load_job_history_entries(data_dir)
    company_history = load_company_history_entries(data_dir)
    for bundle in bundles:
        job = bundle.job
        job_key = _normalize_job_history_key(job.resolved_job_url or job.direct_job_url)
        if not job_key:
            continue
        entry = dict(job_history.get(job_key) or {})
        try:
            from .job_search import _normalize_direct_job_url
            normalized_job_url = _normalize_direct_job_url(str(job.resolved_job_url or job.direct_job_url))
        except Exception:
            normalized_job_url = str(job.resolved_job_url or job.direct_job_url).strip()
        canonical_job_key = _normalize_job_history_key(normalized_job_url)
        first_reported_at = entry.get("first_reported_at") or generated_at_iso
        previous_count = int(entry.get("report_count") or 0)
        if str(entry.get("last_run_id") or "").strip() == run_id:
            report_count = max(previous_count, 1)
        else:
            report_count = previous_count + 1
        entry.update(
            {
                "job_key": job_key,
                "normalized_job_url": normalized_job_url,
                "canonical_job_key": canonical_job_key,
                "company_name": job.company_name,
                "role_title": job.role_title,
                "posted_date_iso": job.posted_date_iso,
                "posted_date_text": job.posted_date_text,
                "salary_text": job.salary_text,
                "first_reported_at": first_reported_at,
                "last_reported_at": generated_at_iso,
                "last_run_id": run_id,
                "report_count": report_count,
                "message_docx_path": manifest.message_docx_path,
                "summary_docx_path": manifest.summary_docx_path,
            }
        )
        job_history[job_key] = entry
        _update_company_history_entry(
            company_history,
            company_name=job.company_name,
            role_title=job.role_title,
            generated_at=generated_at_iso,
            run_id=run_id,
            message_docx_path=manifest.message_docx_path,
            summary_docx_path=manifest.summary_docx_path,
            board_identifier=_board_identifier(job.resolved_job_url or job.direct_job_url),
        )

    _write_json(data_dir / "job-history.json", job_history)
    _write_json(data_dir / "company-history.json", company_history)


def record_company_watchlist(
    data_dir: Path,
    *,
    generated_at: datetime,
    failures: list[SearchFailure] | None = None,
) -> None:
    company_watchlist = _load_json(data_dir / "company-watchlist.json", default={})
    if not isinstance(company_watchlist, dict):
        company_watchlist = {}
    generated_at_iso = generated_at.isoformat()
    for failure in failures or []:
        _update_company_watchlist_entry(
            company_watchlist,
            company_name=failure.company_name,
            reason_code=failure.reason_code,
            generated_at=generated_at_iso,
            source_url=failure.source_url,
            direct_job_url=failure.direct_job_url,
            detail=failure.detail,
        )
    _write_json(data_dir / "company-watchlist.json", company_watchlist)


def record_failed_run(
    data_dir: Path,
    *,
    run_id: str,
    status_payload: Mapping[str, Any] | None = None,
    failure_message: str | None = None,
) -> None:
    _upsert_run_history_entry(
        data_dir,
        {
            "run_id": run_id,
            "status": "failed",
            "started_at": status_payload.get("started_at") if status_payload else None,
            "ended_at": status_payload.get("updated_at") if status_payload else _utc_now_iso(),
            "message": failure_message or (status_payload.get("message") if status_payload else "Workflow failed."),
            "jobs_found_by_search": int((status_payload or {}).get("metrics", {}).get("jobs_found_by_search", 0) or 0),
            "jobs_kept_after_validation": int(
                (status_payload or {}).get("metrics", {}).get("jobs_kept_after_validation", 0) or 0
            ),
            "jobs_with_any_messages": int((status_payload or {}).get("metrics", {}).get("jobs_with_any_messages", 0) or 0),
            "message_docx_path": None,
            "summary_docx_path": None,
        },
    )
