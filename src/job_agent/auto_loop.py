from __future__ import annotations

import asyncio
from collections import deque
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any

from .config import Settings
from .models import (
    AutoLoopIteration,
    AutoLoopState,
    CodexIterationResult,
    ImprovementPattern,
    RunImprovementAnalysis,
    RunScorecard,
    ValidationCommandResult,
)
from .scorecard import load_run_scorecard_entries
from .status import StatusReporter, spawn_progress_gui
from .storage import save_json_snapshot
from .workflow import run_daily_workflow


AUTO_LOOP_STATUS_EVENT_LIMIT = 30
AUTO_LOOP_STATE_FILENAME = "auto-loop-state.json"
AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION = 3
VALIDATION_COMMANDS = [
    "PYTHONPATH=src .venv/bin/pytest -q",
]
THEME_COMMIT_SUMMARIES = {
    "failure_repair": "repair workflow failure handling",
    "query_timeout_burden": "reduce timeout-heavy discovery families",
    "low_fresh_discovery": "improve fresh ATS-first discovery",
    "validation_resolution_quality": "tighten validation and resolution quality",
    "validation_hard_filter_burden": "reduce hard-filter validation waste",
    "linkedin_message_gap": "restore LinkedIn message generation",
    "ollama_idle": "increase bounded Ollama utilization",
    "near_miss_noise": "improve near-miss quality filters",
    "salary_extraction_gap": "improve salary extraction",
    "plateau_breaker": "break repeated-theme plateau",
    "coverage_regression": "preserve reacquired validated coverage",
    "diversification_gap": "increase novel validated job discovery",
    "company_discovery_gap": "expand built-in company and board discovery",
    "official_board_coverage_gap": "close official board coverage gaps",
    "discovery_starvation": "unblock discovery growth and index expansion",
    "company_concentration_gap": "reduce overreliance on the same few companies",
    "directory_source_yield_gap": "improve directory and careers-source yields",
    "salary_presumption_opportunity": "apply principal AI PM salary presumption",
    "general_iterative_improvement": "apply bounded iterative improvement",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _read_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_auto_loop_state(settings: Settings) -> AutoLoopState | None:
    payload = _read_json(settings.auto_loop_state_path, default=None)
    if not isinstance(payload, dict):
        return None
    try:
        return AutoLoopState.model_validate(payload)
    except Exception:
        return None


def save_auto_loop_state(settings: Settings, state: AutoLoopState) -> None:
    state.updated_at = _utc_now()
    save_json_snapshot(settings.auto_loop_state_path, state.model_dump(mode="json"))


def _iteration_dir(settings: Settings, iteration_number: int) -> Path:
    return settings.auto_loop_dir / f"iteration-{iteration_number:02d}"


def _iteration_artifact_paths(settings: Settings, iteration_number: int) -> dict[str, Path]:
    base_dir = _iteration_dir(settings, iteration_number)
    return {
        "dir": base_dir,
        "analysis": base_dir / "analysis.json",
        "prompt": base_dir / "prompt.md",
        "result": base_dir / "result.json",
        "codex_log": base_dir / "codex.jsonl",
        "codex_last_message": base_dir / "codex-last-message.md",
        "validation": base_dir / "validation.log",
    }


def _append_recent_event(existing_payload: dict[str, Any], stage: str, message: str) -> list[dict[str, str]]:
    current_events = list(existing_payload.get("recent_events", []) or [])
    rendered = deque(
        (
            {
                "time": str(item.get("time") or ""),
                "stage": str(item.get("stage") or ""),
                "message": str(item.get("message") or ""),
            }
            for item in current_events
            if isinstance(item, dict)
        ),
        maxlen=AUTO_LOOP_STATUS_EVENT_LIMIT,
    )
    rendered.append(
        {
            "time": _utc_now().isoformat(timespec="seconds"),
            "stage": stage,
            "message": message,
        }
    )
    return list(rendered)


def write_auto_loop_status(
    settings: Settings,
    state: AutoLoopState,
    *,
    stage: str,
    message: str,
    done: bool = False,
    failed: bool = False,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    existing = _read_json(settings.live_status_path, default={})
    if not isinstance(existing, dict):
        existing = {}
    started_at = str(existing.get("started_at") or (state.started_at.isoformat(timespec="seconds") if state.started_at else _utc_now().isoformat(timespec="seconds")))
    payload = {
        "run_id": state.current_run_id or f"auto-loop-{state.current_iteration:02d}",
        "pid": os.getpid(),
        "started_at": started_at,
        "updated_at": _utc_now().isoformat(timespec="seconds"),
        "stage": stage,
        "message": message,
        "done": done,
        "failed": failed,
        "stale": False,
        "metrics": {
            "auto_loop_enabled": state.enabled,
            "auto_loop_status": state.status,
            "auto_loop_iteration": state.current_iteration,
            "auto_loop_completed_attempts": state.completed_attempts,
            "auto_loop_target_attempts": state.target_attempts,
            "codex_session_id": state.codex_session_id,
            **(extra_metrics or {}),
        },
        "recent_events": _append_recent_event(existing, stage, message),
    }
    save_json_snapshot(settings.live_status_path, payload)


def _git_command(settings: Settings, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=settings.project_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_stdout(settings: Settings, *args: str) -> str:
    result = _git_command(settings, *args)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _ensure_main_branch(settings: Settings) -> None:
    branch_name = _git_stdout(settings, "branch", "--show-current")
    if branch_name != "main":
        raise RuntimeError(f"Autonomous loop requires local main; found {branch_name!r}.")


def _git_head(settings: Settings) -> str:
    return _git_stdout(settings, "rev-parse", "HEAD")


def _working_tree_dirty(settings: Settings) -> bool:
    return bool(_git_stdout(settings, "status", "--short"))


def _commit_all_changes(settings: Settings, message: str) -> str:
    add_result = _git_command(settings, "add", "-A")
    if add_result.returncode != 0:
        raise RuntimeError(add_result.stderr.strip() or add_result.stdout.strip() or "git add -A failed.")
    commit_result = _git_command(settings, "commit", "-m", message)
    if commit_result.returncode != 0:
        raise RuntimeError(commit_result.stderr.strip() or commit_result.stdout.strip() or "git commit failed.")
    return _git_head(settings)


def ensure_baseline_commit(settings: Settings, *, iteration_number: int = 1) -> str:
    _ensure_main_branch(settings)
    if not _working_tree_dirty(settings):
        return _git_head(settings)
    return _commit_all_changes(settings, f"auto-loop: baseline before iteration {iteration_number}")


def _commit_message(iteration_number: int, selected_theme: str) -> str:
    summary = THEME_COMMIT_SUMMARIES.get(selected_theme, THEME_COMMIT_SUMMARIES["general_iterative_improvement"])
    return f"auto-loop: iteration {iteration_number} {summary}"


def _validation_commands() -> list[str]:
    return list(VALIDATION_COMMANDS)


def run_validation_commands(settings: Settings, *, iteration_number: int) -> list[ValidationCommandResult]:
    paths = _iteration_artifact_paths(settings, iteration_number)
    results: list[ValidationCommandResult] = []
    for index, command in enumerate(_validation_commands(), start=1):
        output_path = paths["dir"] / f"validation-{index}.log"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            ["bash", "-lc", command],
            cwd=settings.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        rendered_output = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
        output_path.write_text(rendered_output, encoding="utf-8")
        results.append(
            ValidationCommandResult(
                command=command,
                passed=completed.returncode == 0,
                exit_code=completed.returncode,
                output_path=str(output_path),
            )
        )
        if completed.returncode != 0:
            break
    return results


def _read_codex_session_ids(settings: Settings) -> list[str]:
    path = settings.codex_session_index_path
    if not path.exists():
        return []
    session_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except Exception:
            continue
        session_id = str(payload.get("id") or "").strip()
        if session_id:
            session_ids.append(session_id)
    return session_ids


def _detect_new_codex_session_id(before_ids: set[str], after_ids: list[str]) -> str | None:
    new_ids = [session_id for session_id in after_ids if session_id not in before_ids]
    if new_ids:
        return new_ids[-1]
    return after_ids[-1] if after_ids else None


def _codex_command_parts(settings: Settings) -> list[str]:
    parts = shlex.split(settings.codex_command.strip()) if settings.codex_command.strip() else ["codex"]
    if not parts:
        parts = ["codex"]
    executable = shutil.which(parts[0]) or parts[0]
    return [executable, *parts[1:]]


def invoke_codex_iteration(
    settings: Settings,
    *,
    iteration_number: int,
    prompt: str,
    selected_theme: str,
    session_id: str | None,
) -> CodexIterationResult:
    paths = _iteration_artifact_paths(settings, iteration_number)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["prompt"].write_text(prompt, encoding="utf-8")
    before_ids = set(_read_codex_session_ids(settings))
    command = _codex_command_parts(settings)
    command.extend(
        [
            "exec",
            "--full-auto",
            "--skip-git-repo-check",
            "--json",
            "-o",
            str(paths["codex_last_message"]),
        ]
    )
    if session_id:
        command.extend(["resume", session_id, "-"])
    else:
        command.extend(["-C", str(settings.project_root), "-"])
    with paths["codex_log"].open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=settings.project_root,
            input=prompt,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    resolved_session_id = session_id
    if resolved_session_id is None:
        resolved_session_id = _detect_new_codex_session_id(before_ids, _read_codex_session_ids(settings))
    if session_id is None and not resolved_session_id:
        return CodexIterationResult(
            iteration_number=iteration_number,
            generated_at=_utc_now(),
            status="session_failed",
            selected_theme=selected_theme,
            prompt_path=str(paths["prompt"]),
            log_path=str(paths["codex_log"]),
            last_message_path=str(paths["codex_last_message"]),
            exit_code=completed.returncode,
            summary="Codex run completed but no session id could be detected.",
        )
    last_message = ""
    if paths["codex_last_message"].exists():
        last_message = paths["codex_last_message"].read_text(encoding="utf-8").strip()
    return CodexIterationResult(
        iteration_number=iteration_number,
        generated_at=_utc_now(),
        status="succeeded" if completed.returncode == 0 else "failed",
        session_id=resolved_session_id,
        selected_theme=selected_theme,
        prompt_path=str(paths["prompt"]),
        log_path=str(paths["codex_log"]),
        last_message_path=str(paths["codex_last_message"]),
        exit_code=completed.returncode,
        summary=last_message[:4000] or None,
        validation_commands=_validation_commands(),
    )


def _render_codex_retry_prompt(
    base_prompt: str,
    *,
    attempt_number: int,
    codex_result: CodexIterationResult,
    validation_results: list[ValidationCommandResult] | None = None,
) -> str:
    validation_lines: list[str] = []
    for item in validation_results or []:
        status = "passed" if item.passed else "failed"
        rendered = f"- `{item.command}`: {status} (exit_code={item.exit_code})"
        if item.output_path:
            rendered += f" -> `{item.output_path}`"
        validation_lines.append(rendered)
    return "\n".join(
        [
            base_prompt.rstrip(),
            "",
            f"Retry context for autonomous repair attempt {attempt_number}:",
            f"- Previous Codex status: `{codex_result.status}`",
            f"- Previous exit code: `{codex_result.exit_code}`",
            f"- Previous summary: {codex_result.summary or 'No summary was captured.'}",
            "- Fix the failure from the previous attempt and rerun the required validation checks.",
            "- Preserve useful in-progress code changes unless they directly caused the failure.",
            "- Do not run the workflow again.",
            "- Do not commit, push, or sync.",
            "",
            "Previous validation results:",
            *(validation_lines or ["- No validation results were captured."]),
        ]
    ).strip() + "\n"


def _record_failed_iteration(
    settings: Settings,
    *,
    state: AutoLoopState,
    iteration: AutoLoopIteration,
    paths: dict[str, Path],
    codex_result: CodexIterationResult,
    failure_status: str,
    failure_summary: str,
) -> None:
    state.latest_validation_result = failure_status
    state.last_failure_summary = failure_summary
    codex_result.status = failure_status  # type: ignore[assignment]
    codex_result.summary = failure_summary
    save_json_snapshot(paths["result"], codex_result.model_dump(mode="json"))
    iteration.result_path = str(paths["result"])
    iteration.completed_at = _utc_now()
    state.iterations.append(iteration)
    save_auto_loop_state(settings, state)


def _find_run_artifact_path(settings: Settings, run_id: str | None) -> str | None:
    if not run_id:
        return None
    for path in sorted(settings.data_dir.glob("run-*.json"), reverse=True):
        payload = _read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict):
            continue
        if str(manifest.get("run_id") or "").strip() == run_id:
            return str(path)
    return None


def _analysis_artifact_paths(settings: Settings, run_id: str | None) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    run_artifact = _find_run_artifact_path(settings, run_id)
    if run_artifact:
        artifact_paths["run_artifact"] = run_artifact
    diagnostics_payload = _read_json(settings.data_dir / "search-diagnostics-latest.json", default={})
    if isinstance(diagnostics_payload, dict) and str(diagnostics_payload.get("run_id") or "").strip() == str(run_id or "").strip():
        artifact_paths["search_diagnostics"] = str(settings.data_dir / "search-diagnostics-latest.json")
        artifact_paths["false_negative_audit"] = str(settings.data_dir / "false-negative-audit-latest.json")
    near_miss_payload = _read_json(settings.data_dir / "near-misses-latest.json", default={})
    if isinstance(near_miss_payload, dict) and str(near_miss_payload.get("run_id") or "").strip() == str(run_id or "").strip():
        artifact_paths["near_misses"] = str(settings.data_dir / "near-misses-latest.json")
    ollama_payload = _read_json(settings.data_dir / "ollama-summary-latest.json", default={})
    if isinstance(ollama_payload, dict) and str(ollama_payload.get("run_id") or "").strip() == str(run_id or "").strip():
        artifact_paths["ollama_summary"] = str(settings.data_dir / "ollama-summary-latest.json")
    if settings.live_status_path.exists():
        artifact_paths["live_status"] = str(settings.live_status_path)
    if (settings.data_dir / "run-scorecard-latest.json").exists():
        artifact_paths["run_scorecard"] = str(settings.data_dir / "run-scorecard-latest.json")
    company_discovery_path = settings.data_dir / "company-discovery-index.json"
    if company_discovery_path.exists():
        artifact_paths["company_discovery_index"] = str(company_discovery_path)
    frontier_path = settings.data_dir / "company-discovery-frontier.json"
    if frontier_path.exists():
        artifact_paths["company_discovery_frontier"] = str(frontier_path)
    crawl_history_path = settings.data_dir / "company-discovery-crawl-history.json"
    if crawl_history_path.exists():
        artifact_paths["company_discovery_crawl_history"] = str(crawl_history_path)
    audit_path = settings.data_dir / "company-discovery-audit.json"
    if audit_path.exists():
        artifact_paths["company_discovery_audit"] = str(audit_path)
    return artifact_paths


def _scorecard_window(settings: Settings, run_id: str | None, *, limit: int = 20) -> list[RunScorecard]:
    entries = load_run_scorecard_entries(settings.data_dir)
    if not entries:
        return []
    if not run_id:
        return entries[:limit]
    for index, entry in enumerate(entries):
        if entry.run_id == run_id:
            return entries[index : index + limit]
    return entries[:limit]


def _average_metric(entries: list[RunScorecard], accessor) -> float:
    values = [float(accessor(entry)) for entry in entries]
    return round(sum(values) / len(values), 3) if values else 0.0


def _recent_selected_themes(settings: Settings, *, limit: int = 5) -> list[str]:
    state = load_auto_loop_state(settings)
    if state is None:
        return []
    themes = [
        str(iteration.selected_theme or "").strip()
        for iteration in reversed(state.iterations)
        if str(iteration.selected_theme or "").strip()
    ]
    return themes[:limit]


def _theme_streak(themes: list[str]) -> tuple[str | None, int]:
    if not themes:
        return None, 0
    current = themes[0]
    streak = 0
    for theme in themes:
        if theme != current:
            break
        streak += 1
    return current, streak


def _fresh_lead_range(entries: list[RunScorecard]) -> int:
    values = [int(entry.outcome.fresh_new_leads_count) for entry in entries]
    return max(values) - min(values) if values else 0


def _pattern(
    key: str,
    summary: str,
    severity_score: float,
    **evidence: float | int | str | None,
) -> ImprovementPattern:
    return ImprovementPattern(
        key=key,
        summary=summary,
        severity_score=round(float(severity_score), 3),
        evidence=evidence,
    )


def _build_patterns(
    window: list[RunScorecard],
    *,
    current_failure_message: str | None = None,
    recent_selected_themes: list[str] | None = None,
) -> list[ImprovementPattern]:
    if not window:
        return [_pattern("general_iterative_improvement", "No scorecard history was available; apply one bounded improvement batch.", 1.0)]
    current = window[0]
    previous_entries = window[1:]
    patterns: list[ImprovementPattern] = []
    recent_selected_themes = recent_selected_themes or []
    repeated_theme, repeated_theme_streak = _theme_streak(recent_selected_themes)
    recent_window = window[:5]

    if current.status == "failed":
        patterns.append(
            _pattern(
                "failure_repair",
                current_failure_message or "The latest run failed and should be repaired before quality tuning continues.",
                100.0,
                current_run_status=current.status,
            )
        )

    timeout_burden = current.discovery.query_timeout_count + current.discovery.query_skipped_timeout_budget_count
    timeout_discovery_yield = (
        current.outcome.fresh_new_leads_count
        + current.discovery.official_board_leads_count
        + current.discovery.new_companies_discovered_count * 2
        + current.discovery.new_boards_discovered_count * 2
    )
    timeout_yield_ratio = timeout_discovery_yield / max(timeout_burden, 1)
    if timeout_burden >= 5 and (
        timeout_yield_ratio < 0.9
        or (
            current.discovery.new_companies_discovered_count == 0
            and current.discovery.new_boards_discovered_count == 0
            and current.discovery.official_board_leads_count == 0
            and current.outcome.fresh_new_leads_count <= 6
        )
        or timeout_burden >= 10
    ):
        patterns.append(
            _pattern(
                "query_timeout_burden",
                "Discovery is still losing too much budget to timeout-heavy query families.",
                max(12.0, timeout_burden * 3.0 - timeout_discovery_yield * 1.5),
                query_timeout_count=current.discovery.query_timeout_count,
                skipped_timeout_budget=current.discovery.query_skipped_timeout_budget_count,
                discovery_efficiency=current.discovery.discovery_efficiency,
                timeout_discovery_yield=timeout_discovery_yield,
                timeout_yield_ratio=round(timeout_yield_ratio, 3),
                new_companies_discovered_count=current.discovery.new_companies_discovered_count,
                new_boards_discovered_count=current.discovery.new_boards_discovered_count,
                official_board_leads_count=current.discovery.official_board_leads_count,
            )
        )

    validation_hard_filter_burden = (
        current.validation.not_remote_count
        + current.validation.stale_posting_count
        + current.validation.missing_salary_count
    )
    if current.outcome.novel_validated_jobs_count == 0 and validation_hard_filter_burden >= 3:
        patterns.append(
            _pattern(
                "validation_hard_filter_burden",
                "Runs are consistently dying on remote, staleness, or salary hard filters after discovery; improve prevalidation or trusted-source evidence before another model-only tweak.",
                max(28.0, validation_hard_filter_burden * 7.0),
                not_remote_count=current.validation.not_remote_count,
                stale_posting_count=current.validation.stale_posting_count,
                missing_salary_count=current.validation.missing_salary_count,
            )
        )

    if current.outcome.fresh_new_leads_count <= 5 and current.outcome.total_current_validated_jobs_count == 0:
        patterns.append(
            _pattern(
                "low_fresh_discovery",
                "Fresh discovery is too low relative to attempt budget; ATS-first and board-focused discovery likely needs improvement.",
                30.0 - min(current.outcome.fresh_new_leads_count, 5) * 4.0,
                fresh_new_leads_count=current.outcome.fresh_new_leads_count,
                replayed_seed_leads_count=current.discovery.replayed_seed_leads_count,
            )
        )

    if (
        current.discovery.new_companies_discovered_count == 0
        and current.discovery.new_boards_discovered_count == 0
        and current.outcome.total_current_validated_jobs_count == 0
        and current.outcome.fresh_new_leads_count <= 6
    ):
        patterns.append(
            _pattern(
                "company_discovery_gap",
                "The workflow is not expanding its company/board universe enough; improve built-in company discovery instead of relying on replay seeds and known companies.",
                42.0,
                new_companies_discovered_count=current.discovery.new_companies_discovered_count,
                new_boards_discovered_count=current.discovery.new_boards_discovered_count,
                fresh_new_leads_count=current.outcome.fresh_new_leads_count,
            )
        )

    if (
        current.discovery.new_companies_discovered_count == 0
        and current.discovery.new_boards_discovered_count == 0
        and current.discovery.frontier_backlog_count <= 2
        and current.outcome.fresh_new_leads_count <= 6
    ):
        patterns.append(
            _pattern(
                "discovery_starvation",
                "The discovery frontier is not growing and fresh lead generation is starved; expand source adapters, frontier seeding, or board enumeration.",
                40.0,
                fresh_new_leads_count=current.outcome.fresh_new_leads_count,
                frontier_backlog_count=current.discovery.frontier_backlog_count,
                new_companies_discovered_count=current.discovery.new_companies_discovered_count,
                new_boards_discovered_count=current.discovery.new_boards_discovered_count,
            )
        )

    if (
        current.validation.official_roles_missed_count > 0
        or (
            current.discovery.official_board_leads_count > 0
            and current.outcome.total_current_validated_jobs_count == 0
        )
    ):
        patterns.append(
            _pattern(
                "official_board_coverage_gap",
                "Official ATS board discovery is surfacing roles that are not converting into current validated coverage; improve board ingestion and validation follow-through.",
                38.0,
                official_board_leads_count=current.discovery.official_board_leads_count,
                official_roles_missed_count=current.validation.official_roles_missed_count,
            )
        )

    if current.discovery.company_concentration_top_10_share >= 0.75 and current.outcome.fresh_new_leads_count <= 8:
        patterns.append(
            _pattern(
                "company_concentration_gap",
                "Discovery is concentrating too heavily on the same small set of companies; prioritize new-company expansion and less recursive sourcing.",
                36.0 + current.discovery.company_concentration_top_10_share * 10.0,
                company_concentration_top_10_share=current.discovery.company_concentration_top_10_share,
                fresh_new_leads_count=current.outcome.fresh_new_leads_count,
            )
        )

    directory_yield = float(current.discovery.source_adapter_yields.get("directory_source") or 0)
    if (
        current.discovery.new_companies_discovered_count == 0
        and directory_yield == 0
        and current.outcome.total_current_validated_jobs_count == 0
        and current.outcome.fresh_new_leads_count <= 6
    ):
        patterns.append(
            _pattern(
                "directory_source_yield_gap",
                "Directory and ecosystem sources are not producing new company frontier expansion; improve homepage extraction, careers crawling, or source coverage.",
                33.0,
                directory_source_yield=directory_yield,
                new_companies_discovered_count=current.discovery.new_companies_discovered_count,
            )
        )

    resolution_noise = (
        current.validation.company_mismatch_count
        + current.validation.not_specific_job_page_count
        + current.validation.fetch_non_200_count
    )
    if resolution_noise >= 3:
        patterns.append(
            _pattern(
                "validation_resolution_quality",
                "Validation is still spending too much effort on weak or mismatched direct URLs and host-specific resolution failures.",
                resolution_noise * 4.0,
                company_mismatch_count=current.validation.company_mismatch_count,
                not_specific_job_page_count=current.validation.not_specific_job_page_count,
                fetch_non_200_count=current.validation.fetch_non_200_count,
            )
        )

    if current.outcome.novel_validated_jobs_count > 0 and current.outcome.jobs_with_messages_count == 0:
        patterns.append(
            _pattern(
                "linkedin_message_gap",
                "The run produced validated jobs but no LinkedIn messages, so the LinkedIn capture/drafting path still needs work.",
                35.0,
                validated_jobs_count=current.outcome.novel_validated_jobs_count,
                jobs_with_messages_count=current.outcome.jobs_with_messages_count,
            )
        )

    previous_total_current_validated = _average_metric(
        previous_entries,
        lambda item: item.outcome.total_current_validated_jobs_count,
    )
    if previous_total_current_validated > 0 and current.outcome.total_current_validated_jobs_count < previous_total_current_validated:
        patterns.append(
            _pattern(
                "coverage_regression",
                "Current validated coverage has dropped run-over-run; preserve reacquired still-open jobs before focusing only on novelty.",
                34.0 + max(0.0, previous_total_current_validated - current.outcome.total_current_validated_jobs_count) * 4.0,
                total_current_validated_jobs_count=current.outcome.total_current_validated_jobs_count,
                previous_total_current_validated=previous_total_current_validated,
                reacquired_validated_jobs_count=current.outcome.reacquired_validated_jobs_count,
            )
        )

    if current.outcome.total_current_validated_jobs_count > 0 and current.outcome.novel_validated_jobs_count == 0:
        patterns.append(
            _pattern(
                "diversification_gap",
                "The system is still finding current valid jobs through coverage, but it is not adding novel validated jobs; prioritize diversification and discovery quality.",
                32.0,
                total_current_validated_jobs_count=current.outcome.total_current_validated_jobs_count,
                reacquired_validated_jobs_count=current.outcome.reacquired_validated_jobs_count,
                fresh_new_leads_count=current.outcome.fresh_new_leads_count,
            )
        )

    if (
        repeated_theme
        and repeated_theme_streak >= 2
        and len(recent_window) >= 4
        and all(entry.outcome.novel_validated_jobs_count == 0 for entry in recent_window)
        and all(entry.outcome.total_current_validated_jobs_count == 0 for entry in recent_window)
        and all(entry.outcome.actionable_near_miss_count == 0 for entry in recent_window)
        and _fresh_lead_range(recent_window) <= 1
    ):
        repeated_theme_summary = (
            "The loop has plateaued on repeated Ollama-focused fixes without improving validated jobs; switch this iteration to discovery, replay, or validation work instead of another Ollama-only tweak."
            if repeated_theme == "ollama_idle"
            else "The loop has plateaued on the same theme without improving validated jobs; switch to a different subsystem instead of another micro-tweak in the same area."
        )
        patterns.append(
            _pattern(
                "plateau_breaker",
                repeated_theme_summary,
                60.0,
                repeated_theme=repeated_theme,
                repeated_theme_streak=repeated_theme_streak,
                stagnant_run_count=len(recent_window),
                fresh_lead_range=_fresh_lead_range(recent_window),
                latest_fresh_new_leads=current.outcome.fresh_new_leads_count,
                latest_validated_jobs=current.outcome.novel_validated_jobs_count,
                latest_total_current_validated_jobs=current.outcome.total_current_validated_jobs_count,
            )
        )

    recent_ollama_requests = sum(entry.ollama.request_count for entry in window[:5])
    recent_ollama_successes = sum(entry.ollama.success_count for entry in window[:5])
    if recent_ollama_successes > 0:
        if current.ollama.request_count == 0 and window[:3] and all(entry.ollama.request_count == 0 for entry in window[:3]):
            patterns.append(
                _pattern(
                    "ollama_idle",
                    "Ollama has stayed idle across recent runs even though it has shown basic viability; the bounded sidecar path is no longer being exercised.",
                    20.0,
                    recent_request_count=recent_ollama_requests,
                    recent_success_count=recent_ollama_successes,
                    useful_actions_per_request=current.ollama.useful_actions_per_request,
                )
            )
        elif current.ollama.request_count > 0 and current.ollama.useful_actions_per_request < 0.2:
            patterns.append(
                _pattern(
                    "ollama_idle",
                    "Ollama is being invoked, but its useful actions per request remain too low.",
                    16.0,
                    request_count=current.ollama.request_count,
                    success_count=current.ollama.success_count,
                    useful_actions_per_request=current.ollama.useful_actions_per_request,
                )
            )

    if current.outcome.raw_near_miss_count > 0 and current.outcome.actionable_near_miss_count == 0:
        patterns.append(
            _pattern(
                "near_miss_noise",
                "Near-misses are being captured, but none are actionable, which suggests weak source quality or validation fallback issues.",
                16.0,
                raw_near_miss_count=current.outcome.raw_near_miss_count,
                actionable_near_miss_count=current.outcome.actionable_near_miss_count,
            )
        )

    if current.validation.missing_salary_count >= 3:
        patterns.append(
            _pattern(
                "salary_extraction_gap",
                "Salary extraction is still a major blocker for otherwise plausible candidates.",
                current.validation.missing_salary_count * 3.0,
                missing_salary_count=current.validation.missing_salary_count,
            )
        )

    if (
        current.validation.missing_salary_count >= 1
        and current.validation.principal_ai_pm_salary_presumption_count == 0
    ):
        patterns.append(
            _pattern(
                "salary_presumption_opportunity",
                "Principal AI PM roles are still being lost to missing salary; apply or strengthen the principal-level salary presumption path.",
                26.0 + current.validation.missing_salary_count,
                missing_salary_count=current.validation.missing_salary_count,
                principal_ai_pm_salary_presumption_count=current.validation.principal_ai_pm_salary_presumption_count,
            )
        )

    if not patterns:
        validated_avg = _average_metric(previous_entries or window, lambda item: item.outcome.novel_validated_jobs_count)
        patterns.append(
            _pattern(
                "general_iterative_improvement",
                "No single failure family dominates; apply one bounded improvement batch that most directly increases validated jobs or fresh leads.",
                1.0,
                validated_jobs_average=validated_avg,
                total_current_validated_jobs_count=current.outcome.total_current_validated_jobs_count,
            )
        )
    return sorted(patterns, key=lambda item: item.severity_score, reverse=True)


def _metric_deltas(window: list[RunScorecard]) -> dict[str, float]:
    if not window:
        return {}
    current = window[0]
    previous_entries = window[1:] or window
    previous_validated = _average_metric(previous_entries, lambda item: item.outcome.novel_validated_jobs_count)
    previous_total_current_validated = _average_metric(
        previous_entries,
        lambda item: item.outcome.total_current_validated_jobs_count,
    )
    previous_fresh = _average_metric(previous_entries, lambda item: item.outcome.fresh_new_leads_count)
    previous_timeouts = _average_metric(previous_entries, lambda item: item.discovery.query_timeout_count)
    previous_messages = _average_metric(previous_entries, lambda item: item.outcome.jobs_with_messages_count)
    previous_ollama = _average_metric(previous_entries, lambda item: item.ollama.request_count)
    previous_new_companies = _average_metric(previous_entries, lambda item: item.discovery.new_companies_discovered_count)
    previous_new_boards = _average_metric(previous_entries, lambda item: item.discovery.new_boards_discovered_count)
    previous_official_board_leads = _average_metric(previous_entries, lambda item: item.discovery.official_board_leads_count)
    return {
        "validated_jobs_delta": round(current.outcome.novel_validated_jobs_count - previous_validated, 3),
        "total_current_validated_jobs_delta": round(
            current.outcome.total_current_validated_jobs_count - previous_total_current_validated,
            3,
        ),
        "fresh_new_leads_delta": round(current.outcome.fresh_new_leads_count - previous_fresh, 3),
        "query_timeout_delta": round(current.discovery.query_timeout_count - previous_timeouts, 3),
        "jobs_with_messages_delta": round(current.outcome.jobs_with_messages_count - previous_messages, 3),
        "ollama_request_delta": round(current.ollama.request_count - previous_ollama, 3),
        "new_companies_discovered_delta": round(
            current.discovery.new_companies_discovered_count - previous_new_companies,
            3,
        ),
        "new_boards_discovered_delta": round(current.discovery.new_boards_discovered_count - previous_new_boards, 3),
        "official_board_leads_delta": round(
            current.discovery.official_board_leads_count - previous_official_board_leads,
            3,
        ),
    }


def _scorecard_metrics_for_run(settings: Settings, run_id: str | None) -> dict[str, float | int | None]:
    if not run_id:
        return {}
    for entry in _scorecard_window(settings, run_id, limit=1):
        if entry.run_id != run_id:
            continue
        return {
            "novel_validated_jobs_count": entry.outcome.novel_validated_jobs_count,
            "reacquired_validated_jobs_count": entry.outcome.reacquired_validated_jobs_count,
            "total_current_validated_jobs_count": entry.outcome.total_current_validated_jobs_count,
            "fresh_new_leads_count": entry.outcome.fresh_new_leads_count,
            "jobs_with_messages_count": entry.outcome.jobs_with_messages_count,
            "actionable_near_miss_count": entry.outcome.actionable_near_miss_count,
            "query_timeout_count": entry.discovery.query_timeout_count,
            "new_companies_discovered_count": entry.discovery.new_companies_discovered_count,
            "new_boards_discovered_count": entry.discovery.new_boards_discovered_count,
            "official_board_leads_count": entry.discovery.official_board_leads_count,
            "frontier_tasks_consumed_count": entry.discovery.frontier_tasks_consumed_count,
            "frontier_backlog_count": entry.discovery.frontier_backlog_count,
            "official_board_crawl_success_rate": entry.discovery.official_board_crawl_success_rate,
            "company_concentration_top_10_share": entry.discovery.company_concentration_top_10_share,
            "new_company_to_fresh_lead_yield": entry.discovery.new_company_to_fresh_lead_yield,
            "principal_ai_pm_salary_presumption_count": entry.validation.principal_ai_pm_salary_presumption_count,
        }
    return {}


def _metric_comparison(
    before_metrics: dict[str, float | int | None],
    after_metrics: dict[str, float | int | None],
) -> dict[str, float | int | str | None]:
    comparison: dict[str, float | int | str | None] = {}
    for key in sorted(set(before_metrics) | set(after_metrics)):
        before = before_metrics.get(key)
        after = after_metrics.get(key)
        comparison[f"before_{key}"] = before
        comparison[f"after_{key}"] = after
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            comparison[f"delta_{key}"] = round(float(after) - float(before), 3)
    return comparison


def _rerun_metrics_are_acceptable(
    before_metrics: dict[str, float | int | None],
    after_metrics: dict[str, float | int | None],
) -> bool:
    if not after_metrics:
        return False
    before_total = float(before_metrics.get("total_current_validated_jobs_count") or 0)
    after_total = float(after_metrics.get("total_current_validated_jobs_count") or 0)
    before_fresh = float(before_metrics.get("fresh_new_leads_count") or 0)
    after_fresh = float(after_metrics.get("fresh_new_leads_count") or 0)
    before_messages = float(before_metrics.get("jobs_with_messages_count") or 0)
    after_messages = float(after_metrics.get("jobs_with_messages_count") or 0)
    before_timeouts = float(before_metrics.get("query_timeout_count") or 0)
    after_timeouts = float(after_metrics.get("query_timeout_count") or 0)
    before_new_companies = float(before_metrics.get("new_companies_discovered_count") or 0)
    after_new_companies = float(after_metrics.get("new_companies_discovered_count") or 0)
    before_new_boards = float(before_metrics.get("new_boards_discovered_count") or 0)
    after_new_boards = float(after_metrics.get("new_boards_discovered_count") or 0)
    before_official_board_leads = float(before_metrics.get("official_board_leads_count") or 0)
    after_official_board_leads = float(after_metrics.get("official_board_leads_count") or 0)
    discovery_progress = (
        after_new_companies > before_new_companies
        or after_new_boards > before_new_boards
        or after_official_board_leads > before_official_board_leads
    )
    return (
        after_total >= before_total
        and after_messages >= before_messages
        and (after_fresh >= max(0.0, before_fresh - 2.0) or discovery_progress)
        and after_timeouts <= before_timeouts + 3.0
    )


def build_run_improvement_analysis(
    settings: Settings,
    *,
    iteration_number: int,
    run_id: str | None,
    failure_message: str | None = None,
) -> RunImprovementAnalysis:
    window = _scorecard_window(settings, run_id, limit=20)
    current = window[0] if window else None
    recent_selected_themes = _recent_selected_themes(settings, limit=5)
    patterns = _build_patterns(
        window,
        current_failure_message=failure_message,
        recent_selected_themes=recent_selected_themes,
    )
    selected = patterns[0]
    current_metrics = {}
    if current is not None:
        current_metrics = {
            "validated_jobs_count": current.outcome.novel_validated_jobs_count,
            "novel_validated_jobs_count": current.outcome.novel_validated_jobs_count,
            "reacquired_validated_jobs_count": current.outcome.reacquired_validated_jobs_count,
            "total_current_validated_jobs_count": current.outcome.total_current_validated_jobs_count,
            "jobs_with_messages_count": current.outcome.jobs_with_messages_count,
            "fresh_new_leads_count": current.outcome.fresh_new_leads_count,
            "actionable_near_miss_count": current.outcome.actionable_near_miss_count,
            "validated_jobs_with_inferred_salary_count": current.outcome.validated_jobs_with_inferred_salary_count,
            "principal_ai_pm_salary_presumption_count": current.outcome.principal_ai_pm_salary_presumption_count,
            "query_timeout_count": current.discovery.query_timeout_count,
            "query_skipped_timeout_budget_count": current.discovery.query_skipped_timeout_budget_count,
            "replayed_seed_leads_count": current.discovery.replayed_seed_leads_count,
            "reacquisition_attempt_count": current.discovery.reacquisition_attempt_count,
            "reacquired_jobs_suppressed_count": current.discovery.reacquired_jobs_suppressed_count,
            "new_companies_discovered_count": current.discovery.new_companies_discovered_count,
            "new_boards_discovered_count": current.discovery.new_boards_discovered_count,
            "official_board_leads_count": current.discovery.official_board_leads_count,
            "companies_with_ai_pm_leads_count": current.discovery.companies_with_ai_pm_leads_count,
            "company_discovery_yield": current.discovery.company_discovery_yield,
                "company_concentration_top_10_share": current.discovery.company_concentration_top_10_share,
                "frontier_tasks_consumed_count": current.discovery.frontier_tasks_consumed_count,
                "frontier_backlog_count": current.discovery.frontier_backlog_count,
                "official_board_crawl_success_rate": current.discovery.official_board_crawl_success_rate,
                "new_company_to_fresh_lead_yield": current.discovery.new_company_to_fresh_lead_yield,
                "source_adapter_yields": json.dumps(current.discovery.source_adapter_yields, sort_keys=True),
                "discovery_efficiency": current.discovery.discovery_efficiency,
            "validated_yield": current.validation.validated_yield,
            "novel_validated_yield": current.validation.novel_validated_yield,
            "reacquisition_yield": current.validation.reacquisition_yield,
            "coverage_retention_rate": current.validation.coverage_retention_rate,
            "missing_salary_count": current.validation.missing_salary_count,
            "fetch_non_200_count": current.validation.fetch_non_200_count,
            "not_remote_count": current.validation.not_remote_count,
            "stale_posting_count": current.validation.stale_posting_count,
            "official_roles_missed_count": current.validation.official_roles_missed_count,
            "ollama_request_count": current.ollama.request_count,
            "ollama_useful_actions_per_request": current.ollama.useful_actions_per_request,
        }
    acceptance_checks = [
        "Run targeted tests for the changed area and then `PYTHONPATH=src .venv/bin/pytest -q` successfully.",
        "Leave the repository in a clean, committable state.",
        "The controller will rerun the workflow for evidence after validation; you do not need to run it yourself.",
        "Do not commit or sync; the controller handles commits.",
    ]
    return RunImprovementAnalysis(
        iteration_number=iteration_number,
        generated_at=_utc_now(),
        target_run_id=run_id,
        analyzed_run_ids=[entry.run_id for entry in window],
        recent_selected_themes=recent_selected_themes,
        current_run_status=current.status if current is not None else "unknown",
        current_metrics=current_metrics,
        metric_deltas=_metric_deltas(window),
        top_patterns=patterns[:5],
        selected_theme=selected.key,
        selected_summary=selected.summary,
        acceptance_checks=acceptance_checks,
        artifact_paths=_analysis_artifact_paths(settings, run_id),
    )


def render_codex_prompt(
    analysis: RunImprovementAnalysis,
    *,
    iteration_number: int,
) -> str:
    def _render_pattern_line(pattern: ImprovementPattern) -> str:
        if not pattern.evidence:
            return f"- `{pattern.key}` ({pattern.severity_score}): {pattern.summary}"
        evidence_text = ", ".join(f"{key}={value}" for key, value in sorted(pattern.evidence.items()))
        return f"- `{pattern.key}` ({pattern.severity_score}): {pattern.summary} Evidence: {evidence_text}"

    top_pattern_lines = [
        _render_pattern_line(pattern)
        for pattern in analysis.top_patterns
    ]
    artifact_lines = [
        f"- `{name}`: `{path}`"
        for name, path in sorted(analysis.artifact_paths.items())
    ]
    delta_lines = [
        f"- `{key}`: {value}"
        for key, value in sorted(analysis.metric_deltas.items())
    ]
    metric_lines = [
        f"- `{key}`: {value}"
        for key, value in sorted(analysis.current_metrics.items())
    ]
    acceptance_lines = [f"- {item}" for item in analysis.acceptance_checks]
    recent_theme_lines = [f"- `{theme}`" for theme in analysis.recent_selected_themes]
    repeated_theme, repeated_theme_streak = _theme_streak(analysis.recent_selected_themes)
    plateau_guardrails: list[str] = []
    if repeated_theme and repeated_theme_streak >= 2:
        plateau_guardrails = [
            f"- Recent theme streak: `{repeated_theme}` x{repeated_theme_streak}.",
            "- Do not spend this iteration on another micro-tweak in that same area unless you can directly change a primary metric trend.",
            "- Prefer a different subsystem with measurable upside, such as discovery, replay quality, validation, or message generation.",
        ]
    return "\n".join(
        [
            f"# Autonomous Loop Iteration {iteration_number:02d}",
            "",
            "You are working inside an autonomous improvement loop for the Job Agent repository.",
            "",
            f"Selected theme: `{analysis.selected_theme}`",
            f"Why: {analysis.selected_summary}",
            "",
            "Current run metrics:",
            *(metric_lines or ["- No current run metrics were available."]),
            "",
            "Metric deltas versus recent history:",
            *(delta_lines or ["- No metric deltas were available."]),
            "",
            "Recent selected themes:",
            *(recent_theme_lines or ["- No recent theme history was available."]),
            "",
            "Top learned patterns from the latest run plus the previous 19 runs:",
            *(top_pattern_lines or ["- No ranked patterns were available."]),
            "",
            *(["Plateau guardrails:", *plateau_guardrails, ""] if plateau_guardrails else []),
            "Relevant artifacts:",
            *(artifact_lines or ["- No artifact paths were available."]),
            "",
            "Task:",
            f"Implement one primary improvement batch for `{analysis.selected_theme}`.",
            "You may include up to two adjacent fixes if they are part of the same failure frontier.",
            "Prefer the smallest coherent change set that can materially improve the selected theme without forcing another plateau micro-tweak.",
            "",
            "Required behavior:",
            "- Inspect the repository and relevant artifacts as needed.",
            "- Make the code changes yourself.",
            "- It is allowed to add or expand discovery lanes, persistent indexes/frontiers, source adapters, or instrumentation when that is the highest-yield path.",
            "- Run the required validation check(s).",
            "- Summarize what changed, what tests ran, and whether they passed.",
            "- The controller may rerun the workflow after your changes to measure before/after behavior; you do not need to run it yourself.",
            "- Do not commit, push, or sync.",
            "",
            "Acceptance checks:",
            *acceptance_lines,
        ]
    ).strip() + "\n"


async def _run_workflow_attempt(
    settings: Settings,
    *,
    timeout_seconds: int | None = None,
) -> tuple[str, str, str | None]:
    status = StatusReporter(settings.live_status_path)
    try:
        await run_daily_workflow(
            settings,
            status=status,
            timeout_seconds=timeout_seconds,
        )
        return status.run_id, "completed", None
    except Exception as exc:
        return status.run_id, "failed", str(exc)


async def run_autonomous_loop(
    settings: Settings,
    *,
    attempts: int,
    show_gui: bool,
    timeout_seconds: int | None = None,
) -> AutoLoopState:
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    _ensure_main_branch(settings)
    if show_gui and settings.enable_progress_gui:
        spawn_progress_gui(settings.live_status_path)

    baseline_commit_hash = ensure_baseline_commit(settings, iteration_number=1)
    state = AutoLoopState(
        enabled=True,
        status="running",
        target_attempts=attempts,
        completed_attempts=0,
        current_iteration=0,
        current_run_id=None,
        codex_session_id=None,
        baseline_commit_hash=baseline_commit_hash,
        latest_commit_hash=baseline_commit_hash,
        latest_validation_result=None,
        last_failure_summary=None,
        started_at=_utc_now(),
        updated_at=_utc_now(),
    )
    save_auto_loop_state(settings, state)
    write_auto_loop_status(
        settings,
        state,
        stage="autonomous",
        message="Autonomous loop initialized and baseline commit created.",
        extra_metrics={"baseline_commit_hash": baseline_commit_hash},
    )

    for iteration_number in range(1, attempts + 1):
        state.current_iteration = iteration_number
        state.status = "running"
        save_auto_loop_state(settings, state)
        iteration = AutoLoopIteration(
            iteration_number=iteration_number,
            started_at=_utc_now(),
        )
        paths = _iteration_artifact_paths(settings, iteration_number)
        paths["dir"].mkdir(parents=True, exist_ok=True)

        write_auto_loop_status(
            settings,
            state,
            stage="autonomous",
            message=f"Starting workflow attempt {iteration_number}/{attempts}.",
        )
        run_id, run_status, failure_message = await _run_workflow_attempt(
            settings,
            timeout_seconds=timeout_seconds,
        )
        state.current_run_id = run_id
        state.completed_attempts = iteration_number
        state.last_failure_summary = failure_message
        iteration.run_id = run_id
        iteration.run_status = run_status

        state.status = "analysis"
        save_auto_loop_state(settings, state)
        write_auto_loop_status(
            settings,
            state,
            stage="analysis",
            message=f"Analyzing run {run_id} for iteration {iteration_number}.",
        )
        analysis = build_run_improvement_analysis(
            settings,
            iteration_number=iteration_number,
            run_id=run_id,
            failure_message=failure_message,
        )
        save_json_snapshot(paths["analysis"], analysis.model_dump(mode="json"))
        prompt = render_codex_prompt(analysis, iteration_number=iteration_number)
        paths["prompt"].write_text(prompt, encoding="utf-8")
        iteration.analysis_path = str(paths["analysis"])
        iteration.prompt_path = str(paths["prompt"])
        iteration.selected_theme = analysis.selected_theme
        baseline_metrics = _scorecard_metrics_for_run(settings, run_id) or dict(analysis.current_metrics)

        base_prompt = prompt
        retry_prompt = prompt
        codex_succeeded = False
        codex_result: CodexIterationResult | None = None
        for codex_attempt in range(1, AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION + 1):
            state.status = "waiting_for_codex"
            save_auto_loop_state(settings, state)
            write_auto_loop_status(
                settings,
                state,
                stage="codex",
                message=(
                    f"Running Codex improvement pass for iteration {iteration_number}: "
                    f"{analysis.selected_theme} (attempt {codex_attempt}/{AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION})."
                ),
                extra_metrics={
                    "selected_theme": analysis.selected_theme,
                    "codex_attempt": codex_attempt,
                    "codex_attempt_limit": AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION,
                },
            )
            head_before_codex = _git_head(settings)
            codex_result = invoke_codex_iteration(
                settings,
                iteration_number=iteration_number,
                prompt=retry_prompt,
                selected_theme=analysis.selected_theme,
                session_id=state.codex_session_id,
            )
            if codex_result.session_id:
                state.codex_session_id = codex_result.session_id
            iteration.codex_log_path = codex_result.log_path
            iteration.codex_last_message_path = codex_result.last_message_path
            if codex_result.status not in {"succeeded"}:
                failure_summary = codex_result.summary or "Codex iteration failed."
                if codex_attempt < AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION:
                    retry_prompt = _render_codex_retry_prompt(
                        base_prompt,
                        attempt_number=codex_attempt + 1,
                        codex_result=codex_result,
                    )
                    state.last_failure_summary = failure_summary
                    save_auto_loop_state(settings, state)
                    write_auto_loop_status(
                        settings,
                        state,
                        stage="codex",
                        message=(
                            f"Codex attempt {codex_attempt} failed: {failure_summary} "
                            f"Retrying repair attempt {codex_attempt + 1}/{AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION}."
                        ),
                        extra_metrics={"selected_theme": analysis.selected_theme},
                    )
                    continue
                _record_failed_iteration(
                    settings,
                    state=state,
                    iteration=iteration,
                    paths=paths,
                    codex_result=codex_result,
                    failure_status=codex_result.status,
                    failure_summary=failure_summary,
                )
                write_auto_loop_status(
                    settings,
                    state,
                    stage="codex",
                    message=(
                        f"Iteration {iteration_number} exhausted Codex repair attempts after status "
                        f"`{codex_result.status}`. Continuing to the next workflow attempt."
                    ),
                    done=False,
                    failed=False,
                    extra_metrics={"selected_theme": analysis.selected_theme},
                )
                break
            if _git_head(settings) != head_before_codex:
                failure_summary = "Codex changed Git history unexpectedly; the controller owns commit sequencing."
                codex_result.status = "session_failed"
                if codex_attempt < AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION:
                    retry_prompt = _render_codex_retry_prompt(
                        base_prompt,
                        attempt_number=codex_attempt + 1,
                        codex_result=codex_result,
                    )
                    state.last_failure_summary = failure_summary
                    save_auto_loop_state(settings, state)
                    write_auto_loop_status(
                        settings,
                        state,
                        stage="codex",
                        message=(
                            f"Codex changed Git history unexpectedly on attempt {codex_attempt}. "
                            f"Retrying repair attempt {codex_attempt + 1}/{AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION}."
                        ),
                    )
                    continue
                _record_failed_iteration(
                    settings,
                    state=state,
                    iteration=iteration,
                    paths=paths,
                    codex_result=codex_result,
                    failure_status="session_failed",
                    failure_summary=failure_summary,
                )
                write_auto_loop_status(
                    settings,
                    state,
                    stage="codex",
                    message=f"Iteration {iteration_number} exhausted repair attempts after unexpected Git-history changes.",
                    done=False,
                    failed=False,
                )
                break

            state.status = "validating"
            save_auto_loop_state(settings, state)
            write_auto_loop_status(
                settings,
                state,
                stage="validation",
                message=f"Running validation for iteration {iteration_number}.",
            )
            validation_results = run_validation_commands(settings, iteration_number=iteration_number)
            codex_result.validation_results = validation_results
            validation_passed = all(item.passed for item in validation_results)
            iteration.validation_passed = validation_passed
            if not validation_passed:
                failure_summary = "Validation failed after the Codex improvement pass."
                codex_result.status = "validation_failed"
                if codex_attempt < AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION:
                    retry_prompt = _render_codex_retry_prompt(
                        base_prompt,
                        attempt_number=codex_attempt + 1,
                        codex_result=codex_result,
                        validation_results=validation_results,
                    )
                    state.last_failure_summary = failure_summary
                    save_auto_loop_state(settings, state)
                    write_auto_loop_status(
                        settings,
                        state,
                        stage="validation",
                        message=(
                            f"Validation failed for Codex attempt {codex_attempt}. "
                            f"Retrying repair attempt {codex_attempt + 1}/{AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION}."
                        ),
                    )
                    continue
                _record_failed_iteration(
                    settings,
                    state=state,
                    iteration=iteration,
                    paths=paths,
                    codex_result=codex_result,
                    failure_status="validation_failed",
                    failure_summary=failure_summary,
                )
                write_auto_loop_status(
                    settings,
                    state,
                    stage="validation",
                    message=(
                        f"Iteration {iteration_number} exhausted validation repair attempts. "
                        "Continuing to the next workflow attempt."
                    ),
                    done=False,
                    failed=False,
                )
                break

            if not _working_tree_dirty(settings):
                failure_summary = "Codex reported success but left no changes to commit."
                codex_result.status = "no_changes"
                if codex_attempt < AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION:
                    retry_prompt = _render_codex_retry_prompt(
                        base_prompt,
                        attempt_number=codex_attempt + 1,
                        codex_result=codex_result,
                        validation_results=validation_results,
                    )
                    state.last_failure_summary = failure_summary
                    save_auto_loop_state(settings, state)
                    write_auto_loop_status(
                        settings,
                        state,
                        stage="codex",
                        message=(
                            f"Codex left no committable changes on attempt {codex_attempt}. "
                            f"Retrying repair attempt {codex_attempt + 1}/{AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION}."
                        ),
                    )
                    continue
                _record_failed_iteration(
                    settings,
                    state=state,
                    iteration=iteration,
                    paths=paths,
                    codex_result=codex_result,
                    failure_status="no_changes",
                    failure_summary=failure_summary,
                )
                write_auto_loop_status(
                    settings,
                    state,
                    stage="codex",
                    message=f"Iteration {iteration_number} produced no committable changes after all repair attempts.",
                    done=False,
                    failed=False,
                )
                break

            rerun_run_ids: list[str] = []
            metric_comparison: dict[str, float | int | str | None] = {}
            rerun_limit = max(0, settings.auto_loop_max_workflow_reruns_per_iteration)
            rerun_metrics_ok = rerun_limit == 0
            rerun_failure_summary: str | None = None
            if rerun_limit > 0:
                for rerun_index in range(1, rerun_limit + 1):
                    write_auto_loop_status(
                        settings,
                        state,
                        stage="rerun",
                        message=(
                            f"Running workflow evidence rerun {rerun_index}/{rerun_limit} "
                            f"for iteration {iteration_number}."
                        ),
                    )
                    rerun_run_id, rerun_status, rerun_failure = await _run_workflow_attempt(
                        settings,
                        timeout_seconds=timeout_seconds,
                    )
                    rerun_run_ids.append(rerun_run_id)
                    if rerun_status != "completed":
                        rerun_failure_summary = rerun_failure or "Workflow evidence rerun failed."
                        break
                    after_metrics = _scorecard_metrics_for_run(settings, rerun_run_id)
                    metric_comparison = _metric_comparison(baseline_metrics, after_metrics)
                    if _rerun_metrics_are_acceptable(baseline_metrics, after_metrics):
                        rerun_metrics_ok = True
                        break
                codex_result.workflow_rerun_run_ids = rerun_run_ids
                codex_result.workflow_rerun_count = len(rerun_run_ids)
                codex_result.metric_comparison = metric_comparison
                if rerun_failure_summary is not None or not rerun_metrics_ok:
                    failure_summary = rerun_failure_summary or "Workflow evidence reruns regressed the targeted metrics."
                    codex_result.status = "rerun_failed" if rerun_failure_summary else "rerun_regressed"
                    codex_result.summary = failure_summary
                    if codex_attempt < AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION:
                        retry_prompt = _render_codex_retry_prompt(
                            base_prompt,
                            attempt_number=codex_attempt + 1,
                            codex_result=codex_result,
                            validation_results=validation_results,
                        )
                        state.last_failure_summary = failure_summary
                        save_auto_loop_state(settings, state)
                        write_auto_loop_status(
                            settings,
                            state,
                            stage="rerun",
                            message=(
                                f"Workflow rerun evidence was not good enough after Codex attempt {codex_attempt}. "
                                f"Retrying repair attempt {codex_attempt + 1}/{AUTO_LOOP_CODEX_MAX_ATTEMPTS_PER_ITERATION}."
                            ),
                        )
                        continue
                    _record_failed_iteration(
                        settings,
                        state=state,
                        iteration=iteration,
                        paths=paths,
                        codex_result=codex_result,
                        failure_status=codex_result.status,
                        failure_summary=failure_summary,
                    )
                    write_auto_loop_status(
                        settings,
                        state,
                        stage="rerun",
                        message=(
                            f"Iteration {iteration_number} failed workflow evidence reruns. "
                            "Continuing to the next workflow attempt."
                        ),
                        done=False,
                        failed=False,
                    )
                    break

            try:
                commit_hash = _commit_all_changes(settings, _commit_message(iteration_number, analysis.selected_theme))
            except Exception as exc:
                failure_summary = f"Commit failed after validation: {exc}"
                codex_result.status = "commit_failed"
                codex_result.summary = str(exc)
                _record_failed_iteration(
                    settings,
                    state=state,
                    iteration=iteration,
                    paths=paths,
                    codex_result=codex_result,
                    failure_status="commit_failed",
                    failure_summary=failure_summary,
                )
                write_auto_loop_status(
                    settings,
                    state,
                    stage="validation",
                    message=(
                        f"Iteration {iteration_number} could not commit validated changes: {exc}. "
                        "Continuing to the next workflow attempt."
                    ),
                    done=False,
                    failed=False,
                )
                break

            codex_result.commit_hash = commit_hash
            codex_result.commit_message = _commit_message(iteration_number, analysis.selected_theme)
            state.latest_commit_hash = commit_hash
            state.latest_validation_result = "passed"
            state.last_failure_summary = None
            save_json_snapshot(paths["result"], codex_result.model_dump(mode="json"))
            iteration.result_path = str(paths["result"])
            iteration.commit_hash = commit_hash
            iteration.completed_at = _utc_now()
            state.iterations.append(iteration)
            save_auto_loop_state(settings, state)
            codex_succeeded = True
            break

        if not codex_succeeded:
            state.status = "running"
            save_auto_loop_state(settings, state)
            continue

    state.status = "stopped"
    save_auto_loop_state(settings, state)
    write_auto_loop_status(
        settings,
        state,
        stage="completed",
        message=f"Autonomous loop stopped after {attempts} attempts.",
        done=True,
        failed=False,
    )
    return state
