from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import UTC, datetime
from uuid import uuid4

from .company_discovery import (
    company_discovery_audit_path,
    company_discovery_crawl_history_path,
    company_discovery_frontier_path,
    load_company_discovery_audit,
)
from .config import Settings
from .drafting import draft_outreach_bundle
from .history import record_failed_run
from .job_search import find_matching_jobs
from .linkedin_extension_bridge import LinkedInExtensionBridge
from .linkedin import LinkedInClient
from .models import JobOutreachBundle, RunManifest
from .ollama_runtime import (
    auto_tune_ollama_settings,
    build_ollama_run_summary,
    save_ollama_run_summary,
)
from .reports import (
    build_live_outreach_payload,
    build_manifest,
    build_message_document,
    build_near_miss_document,
    build_near_miss_payload,
    build_reacquired_jobs_payload,
    build_summary_document,
)
from .scorecard import save_failed_run_scorecard
from .status import StatusReporter
from .storage import save_run_artifacts


async def _heartbeat(status: StatusReporter, interval_seconds: int) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        status.heartbeat("Workflow still running.")


async def _run_daily_workflow_body(
    settings: Settings,
    *,
    run_id: str,
    status: StatusReporter | None = None,
) -> tuple[list[JobOutreachBundle], RunManifest]:
    tuning_profile = None
    if settings.llm_provider == "ollama":
        settings, tuning_profile = auto_tune_ollama_settings(settings, run_id=run_id)
        if status:
            status.emit(
                "starting",
                "Applied Ollama tuning profile for this run.",
                ollama_model=settings.ollama_model,
                ollama_num_ctx=settings.ollama_num_ctx,
                ollama_num_batch=settings.ollama_num_batch,
                ollama_num_predict=settings.ollama_num_predict,
                ollama_degraded_for_run=settings.ollama_degraded_for_run,
                ollama_degraded_reason=settings.ollama_degraded_reason,
            )
        if settings.ollama_degraded_for_run:
            if status:
                status.emit(
                    "starting",
                    "Optional Ollama steps are already degraded for this run.",
                    ollama_model=settings.ollama_model,
                    ollama_degraded_reason=settings.ollama_degraded_reason,
                )
        else:
            if status:
                status.emit(
                    "starting",
                    "Deferred Ollama prewarm until the first actual local-model task for this run.",
                    ollama_model=settings.ollama_model,
                    ollama_degraded_for_run=False,
                )

    jobs, reacquired_jobs, jobs_found_by_search, search_diagnostics = await find_matching_jobs(
        settings,
        status=status,
        run_id=run_id,
    )
    linkedin = LinkedInClient(settings)
    bundles: list[JobOutreachBundle] = []

    async def process_jobs(linkedin_context=None, extension_bridge: LinkedInExtensionBridge | None = None) -> None:
        for index, job in enumerate(jobs, start=1):
            if settings.linkedin_manual_review_mode:
                manual_links = linkedin.build_manual_review_links(job.company_name, job.role_title)
                manual_notes = [
                    "Open the 1st- and 2nd-degree search links while logged into LinkedIn.",
                    "In LinkedIn, apply the Current Company filter to the exact company before selecting contacts.",
                    "Pick only contacts that currently work at the company and are relevant to hiring or PM/AI leadership.",
                    "After selecting contacts, rerun outreach drafting against those contacts if you want personalized messages.",
                ]
                if status:
                    status.emit(
                        "linkedin",
                        f"Prepared manual LinkedIn review links for {job.company_name} ({index}/{len(jobs)}).",
                        qualifying_jobs=len(jobs),
                        current_company=job.company_name,
                        current_role=job.role_title,
                        manual_link_count=len(manual_links),
                    )
                bundles.append(
                    JobOutreachBundle(
                        job=job,
                        manual_review_links=manual_links,
                        manual_review_notes=manual_notes,
                    )
                )
                continue

            if status:
                status.emit(
                    "linkedin",
                    f"Discovering LinkedIn contacts for {job.company_name} ({index}/{len(jobs)}).",
                    qualifying_jobs=len(jobs),
                    current_company=job.company_name,
                    current_role=job.role_title,
                )
            try:
                discovery = await linkedin.discover_company_contacts(
                    job.company_name,
                    role_title=job.role_title,
                    context=linkedin_context,
                    extension_bridge=extension_bridge,
                )
            except Exception as exc:
                print(f"LinkedIn discovery failed for {job.company_name}: {exc}")
                discovery = None
                if status:
                    status.emit(
                        "linkedin",
                        f"LinkedIn discovery failed for {job.company_name}: {exc}",
                        qualifying_jobs=len(jobs),
                        current_company=job.company_name,
                    )

            if discovery is None:
                bundle = await draft_outreach_bundle(settings, job, [], [], run_id=run_id)
                bundles.append(bundle)
                continue

            if status:
                status.emit(
                    "drafting",
                    f"Drafting outreach for {job.company_name}.",
                    first_order_contacts=len(discovery.first_order_contacts),
                    second_order_contacts=len(discovery.second_order_contacts),
                    current_company=job.company_name,
                    current_role=job.role_title,
                )
            bundle = await draft_outreach_bundle(
                settings,
                job,
                discovery.first_order_contacts,
                discovery.second_order_contacts,
                run_id=run_id,
            )
            bundles.append(bundle)

    if settings.linkedin_manual_review_mode:
        await process_jobs()
    elif settings.linkedin_capture_mode == "firefox_extension":
        async with LinkedInExtensionBridge(settings) as extension_bridge:
            await process_jobs(extension_bridge=extension_bridge)
    else:
        async with linkedin.context() as linkedin_context:
            await process_jobs(linkedin_context)

    if status:
        status.emit("reporting", "Building Word documents and saving run artifacts.", qualifying_jobs=len(jobs))
    generated_at = datetime.now(UTC)
    discovery_summary = {
        "new_companies_discovered_count": search_diagnostics.new_companies_discovered_count,
        "new_boards_discovered_count": search_diagnostics.new_boards_discovered_count,
        "official_board_leads_count": search_diagnostics.official_board_leads_count,
        "companies_with_ai_pm_leads_count": search_diagnostics.companies_with_ai_pm_leads_count,
        "frontier_tasks_consumed_count": search_diagnostics.frontier_tasks_consumed_count,
        "frontier_backlog_count": search_diagnostics.frontier_backlog_count,
        "source_adapter_yields": dict(search_diagnostics.source_adapter_yields),
    }
    official_board_audit = [
        item
        for item in load_company_discovery_audit(settings.data_dir)
        if str(item.get("run_id") or "").strip() == run_id
    ]
    message_docx_path = build_message_document(bundles, settings.output_dir, generated_at=generated_at)
    summary_docx_path = build_summary_document(
        bundles,
        settings.output_dir,
        reacquired_jobs=reacquired_jobs,
        discovery_summary=discovery_summary,
        official_board_audit=official_board_audit,
        generated_at=generated_at,
    )
    near_miss_docx_path = build_near_miss_document(
        search_diagnostics.near_misses,
        settings.output_dir,
        generated_at=generated_at,
    )
    reacquired_jobs_payload = build_reacquired_jobs_payload(
        reacquired_jobs,
        run_id=run_id,
        generated_at=generated_at,
    )
    near_miss_payload = build_near_miss_payload(
        search_diagnostics.near_misses,
        run_id=run_id,
        generated_at=generated_at,
    )
    ollama_summary_path = None
    ollama_summary_payload = None
    if settings.llm_provider == "ollama" and tuning_profile is not None:
        ollama_summary = build_ollama_run_summary(
            settings,
            run_id=run_id,
            tuning_profile=tuning_profile,
            generated_at=generated_at,
        )
        ollama_summary_path = save_ollama_run_summary(settings, ollama_summary)
        ollama_summary_payload = ollama_summary.model_dump(mode="json")
    manifest = build_manifest(
        run_id=run_id,
        bundles=bundles,
        reacquired_jobs=reacquired_jobs,
        jobs_found_by_search=jobs_found_by_search,
        message_docx_path=message_docx_path,
        summary_docx_path=summary_docx_path,
        reacquired_jobs_json_path=settings.data_dir / "reacquired-jobs-latest.json",
        company_discovery_json_path=settings.data_dir / "company-discovery-index.json",
        company_discovery_frontier_json_path=company_discovery_frontier_path(settings.data_dir),
        company_discovery_crawl_history_json_path=company_discovery_crawl_history_path(settings.data_dir),
        company_discovery_audit_json_path=company_discovery_audit_path(settings.data_dir),
        near_misses=search_diagnostics.near_misses,
        near_miss_docx_path=near_miss_docx_path,
        near_miss_json_path=settings.data_dir / "near-misses-latest.json",
        ollama_summary_json_path=ollama_summary_path,
        generated_at=generated_at,
    )
    live_outreach_payload = build_live_outreach_payload(
        bundles,
        run_id=run_id,
        generated_at=generated_at,
    )
    save_run_artifacts(
        settings.data_dir,
        bundles,
        reacquired_jobs,
        manifest,
        live_outreach_payload=live_outreach_payload,
        reacquired_jobs_payload=reacquired_jobs_payload,
        near_miss_payload=near_miss_payload,
        ollama_summary_payload=ollama_summary_payload,
        search_diagnostics=search_diagnostics,
        status_payload=status.snapshot() if status else None,
    )
    if not settings.linkedin_manual_review_mode and settings.linkedin_capture_mode == "firefox_extension":
        await asyncio.sleep(3)
    if status:
        status.complete(
            "Workflow completed successfully.",
            jobs_found_by_search=jobs_found_by_search,
            jobs_kept_after_validation=manifest.jobs_kept_after_validation,
            jobs_with_any_messages=manifest.jobs_with_any_messages,
            near_miss_count=manifest.near_miss_count,
            ollama_degraded_for_run=settings.ollama_degraded_for_run,
        )
    return bundles, manifest


async def run_daily_workflow(
    settings: Settings,
    *,
    status: StatusReporter | None = None,
    timeout_seconds: int | None = None,
) -> tuple[list[JobOutreachBundle], RunManifest]:
    heartbeat_task: asyncio.Task[None] | None = None
    run_id = status.run_id if status is not None else uuid4().hex
    if status:
        status.emit(
            "starting",
            "Starting the job search workflow.",
            target_job_count=settings.target_job_count,
            min_base_salary_usd=settings.min_base_salary_usd,
            posted_within_days=settings.posted_within_days,
        )
        heartbeat_task = asyncio.create_task(_heartbeat(status, settings.status_heartbeat_seconds))

    effective_timeout = settings.workflow_timeout_seconds if timeout_seconds is None else max(0, timeout_seconds)
    try:
        if effective_timeout > 0:
            async with asyncio.timeout(effective_timeout):
                return await _run_daily_workflow_body(settings, run_id=run_id, status=status)
        return await _run_daily_workflow_body(settings, run_id=run_id, status=status)
    except TimeoutError:
        if status:
            status.fail(
                f"Workflow timed out after {effective_timeout} seconds.",
                workflow_timeout_seconds=effective_timeout,
            )
        record_failed_run(
            settings.data_dir,
            run_id=run_id,
            status_payload=status.snapshot() if status else None,
            failure_message=f"Workflow timed out after {effective_timeout} seconds.",
        )
        save_failed_run_scorecard(
            settings.data_dir,
            run_id=run_id,
            status_payload=status.snapshot() if status else None,
            failure_message=f"Workflow timed out after {effective_timeout} seconds.",
        )
        raise
    except asyncio.CancelledError:
        if status:
            status.fail("Workflow terminated before completion.")
        record_failed_run(
            settings.data_dir,
            run_id=run_id,
            status_payload=status.snapshot() if status else None,
            failure_message="Workflow terminated before completion.",
        )
        save_failed_run_scorecard(
            settings.data_dir,
            run_id=run_id,
            status_payload=status.snapshot() if status else None,
            failure_message="Workflow terminated before completion.",
        )
        raise
    except Exception as exc:
        if status:
            status.fail(f"Workflow failed: {exc}")
        record_failed_run(
            settings.data_dir,
            run_id=run_id,
            status_payload=status.snapshot() if status else None,
            failure_message=f"Workflow failed: {exc}",
        )
        save_failed_run_scorecard(
            settings.data_dir,
            run_id=run_id,
            status_payload=status.snapshot() if status else None,
            failure_message=f"Workflow failed: {exc}",
        )
        raise
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
