from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import signal
import shutil

from .auto_loop import run_autonomous_loop
from .config import load_settings
from .dashboard import run_dashboard
from .firefox_extension_host import (
    deploy_host_background,
    inspect_configured_firefox_extension_profile,
    remove_host,
)
from .linkedin import LinkedInClient, describe_browser_choice
from .ollama_runtime import load_latest_ollama_summary, load_ollama_tuning_profile
from .scheduler import install_user_cron, render_cron_line
from .status import StatusReporter, spawn_progress_gui
from .workflow import run_daily_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agentic job search and LinkedIn drafting workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("bootstrap-linkedin-session", help="Open a browser to establish a LinkedIn session.")
    subparsers.add_parser(
        "bootstrap-linkedin-google-session",
        help="Open a browser so you can manually sign in to LinkedIn with Google.",
    )
    run_parser = subparsers.add_parser("run", help="Run the daily workflow once.")
    run_parser.add_argument("--no-gui", action="store_true", help="Do not open the live progress window.")
    run_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Hard-stop the workflow after this many seconds. Overrides WORKFLOW_TIMEOUT_SECONDS for this run.",
    )
    auto_loop_parser = subparsers.add_parser(
        "autonomous-loop",
        help="Run the autonomous 20-run Codex improvement loop.",
    )
    auto_loop_parser.add_argument(
        "--attempts",
        type=int,
        default=20,
        help="Maximum number of workflow attempts to run before stopping.",
    )
    auto_loop_parser.add_argument("--no-gui", action="store_true", help="Do not open the live progress window.")
    auto_loop_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Hard-stop each workflow attempt after this many seconds.",
    )
    subparsers.add_parser("doctor", help="Show the current setup and authentication readiness.")
    subparsers.add_parser("install-cron", help="Install or update a daily user crontab entry.")
    subparsers.add_parser("dashboard", help="Open the desktop dashboard with run controls and history.")
    subparsers.add_parser(
        "deploy-firefox-extension",
        help="Launch a dedicated Firefox instance with the local LinkedIn bridge extension loaded.",
    )
    subparsers.add_parser(
        "remove-firefox-extension",
        help="Stop the dedicated Firefox extension instance and remove its profile files.",
    )
    return parser


async def _run_bootstrap() -> None:
    settings = load_settings(require_openai=False)
    client = LinkedInClient(settings)
    await client.bootstrap_session()
    print(f"Saved LinkedIn browser state to {settings.linkedin_storage_state}")


async def _run_bootstrap_google() -> None:
    settings = load_settings(require_openai=False)
    client = LinkedInClient(settings)
    await client.bootstrap_google_session()
    print(f"Saved LinkedIn browser state to {settings.linkedin_storage_state}")


async def _run_workflow(*, show_gui: bool, timeout_seconds: int | None = None) -> None:
    settings = load_settings()
    status = StatusReporter(settings.live_status_path)
    if show_gui and settings.enable_progress_gui:
        spawn_progress_gui(settings.live_status_path)

    workflow_task = asyncio.create_task(
        run_daily_workflow(
            settings,
            status=status,
            timeout_seconds=timeout_seconds,
        )
    )
    loop = asyncio.get_running_loop()
    termination_signal: dict[str, str | None] = {"name": None}

    def _request_shutdown(signal_name: str) -> None:
        if termination_signal["name"] is None:
            termination_signal["name"] = signal_name
            workflow_task.cancel()

    registered_signals: list[int] = []
    for signal_value in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_value, _request_shutdown, signal.Signals(signal_value).name)
            registered_signals.append(signal_value)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - platform dependent
            continue

    try:
        bundles, manifest = await workflow_task
    except asyncio.CancelledError:
        raise SystemExit(
            f"Workflow terminated by {termination_signal['name'] or 'an external signal'}."
        ) from None
    finally:
        for signal_value in registered_signals:
            loop.remove_signal_handler(signal_value)

    print(json.dumps(manifest.model_dump(mode="json"), indent=2))
    print(f"Generated {len(bundles)} job bundles.")


async def _run_autonomous_loop(*, show_gui: bool, attempts: int, timeout_seconds: int | None = None) -> None:
    settings = load_settings()
    state = await run_autonomous_loop(
        settings,
        attempts=attempts,
        show_gui=show_gui,
        timeout_seconds=timeout_seconds,
    )
    print(json.dumps(state.model_dump(mode="json"), indent=2))


def _run_doctor() -> None:
    settings = load_settings(require_openai=False)
    firefox_extension_profile = inspect_configured_firefox_extension_profile(settings)
    ollama_profile = load_ollama_tuning_profile(settings)
    ollama_summary = load_latest_ollama_summary(settings)
    ollama_command_parts = shlex.split(settings.ollama_command.strip()) if settings.ollama_command.strip() else ["ollama"]
    ollama_executable = ollama_command_parts[0] if ollama_command_parts else "ollama"
    payload = {
        "project_root": str(settings.project_root),
        "openai_key_present": bool(settings.openai_api_key),
        "linkedin_auth_methods": {
            "email_password": bool(settings.linkedin_email and settings.linkedin_password),
            "totp_secret": bool(settings.linkedin_totp_secret),
            "cookies": bool(settings.linkedin_li_at and settings.linkedin_jsessionid),
            "storage_state_exists": settings.linkedin_storage_state.exists(),
            "google_sign_in": bool(settings.google_email and settings.google_password),
            "google_totp_secret": bool(settings.google_totp_secret),
        },
        "headless": settings.headless,
        "enable_progress_gui": settings.enable_progress_gui,
        "linkedin_manual_review_mode": settings.linkedin_manual_review_mode,
        "llm_provider": settings.llm_provider,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_command": settings.ollama_command,
        "ollama_command_resolved": shutil.which(ollama_executable),
        "ollama_model": settings.ollama_model,
        "ollama_tuning_profile": ollama_profile.model_dump(mode="json"),
        "ollama_keep_alive": settings.ollama_keep_alive,
        "ollama_num_ctx": settings.ollama_num_ctx,
        "ollama_num_batch": settings.ollama_num_batch,
        "ollama_num_predict": settings.ollama_num_predict,
        "ollama_max_concurrent_requests": settings.ollama_max_concurrent_requests,
        "ollama_restart_on_failure": settings.ollama_restart_on_failure,
        "ollama_log_path": str(settings.ollama_log_path),
        "ollama_event_log_path": str(settings.ollama_event_log_path),
        "ollama_summary_path": str(settings.ollama_summary_path),
        "ollama_latest_summary": ollama_summary.model_dump(mode="json") if ollama_summary is not None else None,
        "use_openai_fallback": settings.use_openai_fallback,
        "local_confidence_threshold": settings.local_confidence_threshold,
        "linkedin_capture_mode": settings.linkedin_capture_mode,
        "linkedin_extension_bridge": (
            f"http://{settings.linkedin_extension_bridge_host}:{settings.linkedin_extension_bridge_port}"
        ),
        "linkedin_extension_capture_timeout_seconds": settings.linkedin_extension_capture_timeout_seconds,
        "linkedin_extension_history_timeout_seconds": settings.linkedin_extension_history_timeout_seconds,
        "linkedin_extension_auto_open_search_tabs": settings.linkedin_extension_auto_open_search_tabs,
        "firefox_extension_profile": firefox_extension_profile,
        "cron_available": shutil.which("crontab") is not None,
        "recommended_cron_line": render_cron_line(settings),
        "recommended_auth_bootstrap": "job-agent bootstrap-linkedin-google-session",
        "browser_choice": describe_browser_choice(settings),
        "live_status_path": str(settings.live_status_path),
        "run_history_path": str(settings.run_history_path),
        "job_history_path": str(settings.job_history_path),
        "minimum_qualifying_jobs": settings.minimum_qualifying_jobs,
        "target_job_count": settings.target_job_count,
        "max_adaptive_search_passes": settings.max_adaptive_search_passes,
        "max_search_rounds": settings.max_search_rounds,
        "max_leads_to_resolve_per_pass": settings.max_leads_to_resolve_per_pass,
        "per_query_timeout_seconds": settings.per_query_timeout_seconds,
        "per_lead_timeout_seconds": settings.per_lead_timeout_seconds,
        "workflow_timeout_seconds": settings.workflow_timeout_seconds,
    }
    print(json.dumps(payload, indent=2))


def _run_install_cron() -> None:
    settings = load_settings()
    line = install_user_cron(settings)
    print(line)


def _run_dashboard() -> None:
    run_dashboard()


def _run_deploy_firefox_extension() -> None:
    print(json.dumps(deploy_host_background(), indent=2))


def _run_remove_firefox_extension() -> None:
    print(json.dumps(remove_host(delete_profile=True), indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "bootstrap-linkedin-session":
        asyncio.run(_run_bootstrap())
    elif args.command == "bootstrap-linkedin-google-session":
        asyncio.run(_run_bootstrap_google())
    elif args.command == "run":
        asyncio.run(_run_workflow(show_gui=not args.no_gui, timeout_seconds=args.timeout_seconds))
    elif args.command == "autonomous-loop":
        asyncio.run(
            _run_autonomous_loop(
                show_gui=not args.no_gui,
                attempts=args.attempts,
                timeout_seconds=args.timeout_seconds,
            )
        )
    elif args.command == "doctor":
        _run_doctor()
    elif args.command == "install-cron":
        _run_install_cron()
    elif args.command == "dashboard":
        _run_dashboard()
    elif args.command == "deploy-firefox-extension":
        _run_deploy_firefox_extension()
    elif args.command == "remove-firefox-extension":
        _run_remove_firefox_extension()
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
