from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

from .criteria import DEFAULT_ROLE_SEARCH_PROFILE

DEFAULT_SEARCH_QUERIES = list(DEFAULT_ROLE_SEARCH_PROFILE.default_search_queries)


@dataclass(slots=True)
class Settings:
    project_root: Path
    openai_api_key: str
    linkedin_email: str | None
    linkedin_password: str | None
    linkedin_totp_secret: str | None
    linkedin_li_at: str | None
    linkedin_jsessionid: str | None
    google_email: str | None
    google_password: str | None
    google_totp_secret: str | None
    browser_executable_path: str | None
    browser_channel: str | None
    linkedin_profile_dir: Path
    linkedin_storage_state: Path
    output_dir: Path
    data_dir: Path
    headless: bool
    timezone: str
    search_country: str
    search_city: str
    search_region: str
    min_base_salary_usd: int
    enable_principal_ai_pm_salary_presumption: bool
    company_discovery_enabled: bool
    posted_within_days: int
    minimum_qualifying_jobs: int
    target_job_count: int
    max_adaptive_search_passes: int
    max_search_rounds: int
    search_round_query_limit: int
    max_leads_per_query: int
    max_leads_to_resolve_per_pass: int
    reacquisition_attempt_cap: int
    per_query_timeout_seconds: int
    per_lead_timeout_seconds: int
    workflow_timeout_seconds: int
    max_linkedin_results_per_company: int
    max_linkedin_pages_per_company: int
    daily_run_hour: int
    daily_run_minute: int
    status_heartbeat_seconds: int
    enable_progress_gui: bool
    linkedin_manual_review_mode: bool = True
    llm_provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b-instruct"
    use_openai_fallback: bool = False
    local_confidence_threshold: float = 0.75
    linkedin_capture_mode: str = "playwright"
    linkedin_extension_bridge_host: str = "127.0.0.1"
    linkedin_extension_bridge_port: int = 8765
    linkedin_extension_capture_timeout_seconds: int = 120
    linkedin_extension_history_timeout_seconds: int = 30
    linkedin_extension_auto_open_search_tabs: bool = True
    search_queries: list[str] = field(default_factory=lambda: list(DEFAULT_SEARCH_QUERIES))
    ollama_timeout_seconds: float = 120.0
    ollama_keep_alive: str = "5m"
    ollama_num_ctx: int = 1024
    ollama_num_batch: int = 4
    ollama_num_predict: int = 256
    ollama_max_concurrent_requests: int = 1
    ollama_max_retries: int = 3
    ollama_restart_on_failure: bool = True
    ollama_start_timeout_seconds: int = 45
    ollama_retry_backoff_seconds: float = 3.0
    ollama_command: str = "ollama"
    codex_command: str = "codex"
    firefox_extension_profile_dir: Path | None = None
    ollama_degraded_model: str = "qwen2.5:7b-instruct"
    ollama_enable_auto_tune: bool = True
    ollama_degraded_for_run: bool = False
    ollama_degraded_reason: str | None = None
    auto_loop_max_workflow_reruns_per_iteration: int = 2
    company_discovery_indexer_enabled: bool = True
    company_discovery_frontier_budget_per_run: int = 12
    company_discovery_board_crawl_budget_per_run: int = 12
    company_discovery_directory_crawl_budget_per_run: int = 8
    company_discovery_source_max_trust: int = 10

    @property
    def user_location(self) -> dict[str, object]:
        return {
            "type": "approximate",
            "country": self.search_country,
            "city": self.search_city,
            "region": self.search_region,
            "timezone": self.timezone,
        }

    @property
    def live_status_path(self) -> Path:
        return self.data_dir / "live-status.json"

    @property
    def run_history_path(self) -> Path:
        return self.data_dir / "run-history.json"

    @property
    def job_history_path(self) -> Path:
        return self.data_dir / "job-history.json"

    @property
    def ollama_runtime_dir(self) -> Path:
        return self.project_root / ".secrets" / "ollama-runtime"

    @property
    def ollama_log_path(self) -> Path:
        return self.output_dir / "ollama.log"

    @property
    def ollama_event_log_path(self) -> Path:
        return self.output_dir / "ollama-events.jsonl"

    @property
    def ollama_tuning_profile_path(self) -> Path:
        return self.data_dir / "ollama-profile.json"

    @property
    def ollama_summary_path(self) -> Path:
        return self.data_dir / "ollama-summary-latest.json"

    @property
    def auto_loop_state_path(self) -> Path:
        return self.data_dir / "auto-loop-state.json"

    @property
    def auto_loop_dir(self) -> Path:
        return self.data_dir / "auto-loop"

    @property
    def codex_home_dir(self) -> Path:
        raw_home = os.getenv("CODEX_HOME")
        if raw_home:
            return Path(raw_home).expanduser()
        return Path.home() / ".codex"

    @property
    def codex_session_index_path(self) -> Path:
        return self.codex_home_dir / "session_index.jsonl"


def load_settings(project_root: Path | None = None, *, require_openai: bool = True) -> Settings:
    root = project_root or Path.cwd()
    # Project-local .env values should take precedence over inherited shell values
    # so one machine-wide export does not leak into another repo or test fixture.
    load_dotenv(root / ".env", override=True)

    def resolve_config_path(raw_value: str) -> Path:
        candidate = Path(raw_value).expanduser()
        return candidate if candidate.is_absolute() else root / candidate

    def resolve_optional_config_path(raw_value: str | None) -> Path | None:
        if not raw_value:
            return None
        return resolve_config_path(raw_value)

    output_dir = root / "output"
    data_dir = root / "data"
    linkedin_profile_dir = resolve_config_path(os.getenv("LINKEDIN_PROFILE_DIR", ".secrets/linkedin-profile"))
    linkedin_storage_state = resolve_config_path(os.getenv("LINKEDIN_STORAGE_STATE", ".secrets/linkedin-state.json"))
    firefox_extension_profile_dir = resolve_optional_config_path(os.getenv("FIREFOX_EXTENSION_PROFILE_DIR"))

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    linkedin_profile_dir.parent.mkdir(parents=True, exist_ok=True)
    linkedin_storage_state.parent.mkdir(parents=True, exist_ok=True)

    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    use_openai_fallback = os.getenv("USE_OPENAI_FALLBACK", "false").lower() == "true"

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_required = require_openai and (llm_provider == "openai" or use_openai_fallback)
    if openai_required and not api_key:
        raise ValueError("OPENAI_API_KEY is required.")

    return Settings(
        project_root=root,
        openai_api_key=api_key,
        linkedin_email=os.getenv("LINKEDIN_EMAIL") or None,
        linkedin_password=os.getenv("LINKEDIN_PASSWORD") or None,
        linkedin_totp_secret=os.getenv("LINKEDIN_TOTP_SECRET") or None,
        linkedin_li_at=os.getenv("LINKEDIN_LI_AT") or None,
        linkedin_jsessionid=os.getenv("LINKEDIN_JSESSIONID") or None,
        google_email=os.getenv("GOOGLE_EMAIL") or None,
        google_password=os.getenv("GOOGLE_PASSWORD") or None,
        google_totp_secret=os.getenv("GOOGLE_TOTP_SECRET") or None,
        browser_executable_path=os.getenv("BROWSER_EXECUTABLE_PATH") or None,
        browser_channel=os.getenv("BROWSER_CHANNEL") or None,
        linkedin_profile_dir=linkedin_profile_dir,
        linkedin_storage_state=linkedin_storage_state,
        output_dir=output_dir,
        data_dir=data_dir,
        headless=os.getenv("HEADLESS", "true").lower() == "true",
        timezone=os.getenv("JOB_SEARCH_TIMEZONE", "America/Chicago"),
        search_country=os.getenv("JOB_SEARCH_COUNTRY", "US"),
        search_city=os.getenv("JOB_SEARCH_CITY", "Chicago"),
        search_region=os.getenv("JOB_SEARCH_REGION", "Illinois"),
        min_base_salary_usd=int(os.getenv("MIN_BASE_SALARY_USD", "200000")),
        enable_principal_ai_pm_salary_presumption=os.getenv(
            "ENABLE_PRINCIPAL_AI_PM_SALARY_PRESUMPTION",
            "true",
        ).lower()
        == "true",
        company_discovery_enabled=os.getenv("COMPANY_DISCOVERY_ENABLED", "true").lower() == "true",
        posted_within_days=int(os.getenv("POSTED_WITHIN_DAYS", "14")),
        minimum_qualifying_jobs=int(os.getenv("MINIMUM_QUALIFYING_JOBS", "5")),
        target_job_count=int(os.getenv("TARGET_JOB_COUNT", "10")),
        max_adaptive_search_passes=int(os.getenv("MAX_ADAPTIVE_SEARCH_PASSES", "3")),
        max_search_rounds=int(os.getenv("MAX_SEARCH_ROUNDS", "3")),
        search_round_query_limit=int(os.getenv("SEARCH_ROUND_QUERY_LIMIT", "6")),
        max_leads_per_query=int(os.getenv("MAX_LEADS_PER_QUERY", "6")),
        max_leads_to_resolve_per_pass=int(os.getenv("MAX_LEADS_TO_RESOLVE_PER_PASS", "60")),
        reacquisition_attempt_cap=int(os.getenv("REACQUISITION_ATTEMPT_CAP", "10")),
        per_query_timeout_seconds=int(os.getenv("PER_QUERY_TIMEOUT_SECONDS", "35")),
        per_lead_timeout_seconds=int(os.getenv("PER_LEAD_TIMEOUT_SECONDS", "25")),
        workflow_timeout_seconds=max(0, int(os.getenv("WORKFLOW_TIMEOUT_SECONDS", "3600"))),
        max_linkedin_results_per_company=int(os.getenv("MAX_LINKEDIN_RESULTS_PER_COMPANY", "25")),
        max_linkedin_pages_per_company=int(os.getenv("MAX_LINKEDIN_PAGES_PER_COMPANY", "3")),
        daily_run_hour=int(os.getenv("DAILY_RUN_HOUR", "8")),
        daily_run_minute=int(os.getenv("DAILY_RUN_MINUTE", "0")),
        status_heartbeat_seconds=int(os.getenv("STATUS_HEARTBEAT_SECONDS", "120")),
        enable_progress_gui=os.getenv("ENABLE_PROGRESS_GUI", "true").lower() == "true",
        linkedin_manual_review_mode=os.getenv("LINKEDIN_MANUAL_REVIEW_MODE", "true").lower() == "true",
        llm_provider=llm_provider,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        use_openai_fallback=use_openai_fallback,
        local_confidence_threshold=float(os.getenv("LOCAL_CONFIDENCE_THRESHOLD", "0.75")),
        linkedin_capture_mode=os.getenv("LINKEDIN_CAPTURE_MODE", "playwright").strip().lower(),
        linkedin_extension_bridge_host=os.getenv("LINKEDIN_EXTENSION_BRIDGE_HOST", "127.0.0.1"),
        linkedin_extension_bridge_port=int(os.getenv("LINKEDIN_EXTENSION_BRIDGE_PORT", "8765")),
        linkedin_extension_capture_timeout_seconds=int(
            os.getenv("LINKEDIN_EXTENSION_CAPTURE_TIMEOUT_SECONDS", "120")
        ),
        linkedin_extension_history_timeout_seconds=int(
            os.getenv("LINKEDIN_EXTENSION_HISTORY_TIMEOUT_SECONDS", "30")
        ),
        linkedin_extension_auto_open_search_tabs=os.getenv(
            "LINKEDIN_EXTENSION_AUTO_OPEN_SEARCH_TABS",
            "true",
        ).lower()
        == "true",
        ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "5m"),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "1024")),
        ollama_num_batch=int(os.getenv("OLLAMA_NUM_BATCH", "4")),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
        ollama_max_concurrent_requests=int(os.getenv("OLLAMA_MAX_CONCURRENT_REQUESTS", "1")),
        ollama_max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
        ollama_restart_on_failure=os.getenv("OLLAMA_RESTART_ON_FAILURE", "true").lower() == "true",
        ollama_start_timeout_seconds=int(os.getenv("OLLAMA_START_TIMEOUT_SECONDS", "45")),
        ollama_retry_backoff_seconds=float(os.getenv("OLLAMA_RETRY_BACKOFF_SECONDS", "3")),
        ollama_command=os.getenv("OLLAMA_COMMAND", "ollama"),
        codex_command=os.getenv("CODEX_COMMAND", "codex"),
        firefox_extension_profile_dir=firefox_extension_profile_dir,
        ollama_degraded_model=os.getenv("OLLAMA_DEGRADED_MODEL", "qwen2.5:7b-instruct"),
        ollama_enable_auto_tune=os.getenv("OLLAMA_ENABLE_AUTO_TUNE", "true").lower() == "true",
        auto_loop_max_workflow_reruns_per_iteration=max(
            0,
            int(os.getenv("AUTO_LOOP_MAX_WORKFLOW_RERUNS_PER_ITERATION", "2")),
        ),
        company_discovery_indexer_enabled=os.getenv("COMPANY_DISCOVERY_INDEXER_ENABLED", "true").lower() == "true",
        company_discovery_frontier_budget_per_run=max(
            0,
            int(os.getenv("COMPANY_DISCOVERY_FRONTIER_BUDGET_PER_RUN", "12")),
        ),
        company_discovery_board_crawl_budget_per_run=max(
            0,
            int(os.getenv("COMPANY_DISCOVERY_BOARD_CRAWL_BUDGET_PER_RUN", "12")),
        ),
        company_discovery_directory_crawl_budget_per_run=max(
            0,
            int(os.getenv("COMPANY_DISCOVERY_DIRECTORY_CRAWL_BUDGET_PER_RUN", "8")),
        ),
        company_discovery_source_max_trust=max(
            0,
            int(os.getenv("COMPANY_DISCOVERY_SOURCE_MAX_TRUST", "10")),
        ),
    )
