from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class JobLead(BaseModel):
    company_name: str
    role_title: str
    source_url: str = Field(description="Source page where this role was discovered.")
    source_type: Literal["direct_ats", "linkedin", "builtin", "glassdoor", "company_site", "other"]
    direct_job_url: str | None = Field(
        default=None,
        description="Direct ATS or company careers URL when known. Null if discovery only found an aggregator page.",
    )
    location_hint: str | None = None
    posted_date_hint: str | None = None
    is_remote_hint: bool | None = None
    base_salary_min_usd_hint: int | None = None
    base_salary_max_usd_hint: int | None = None
    salary_text_hint: str | None = None
    source_query: str | None = None
    evidence_notes: str = Field(description="Short explanation of why this lead looks promising.")
    refined_by_ollama: bool = False
    source_quality_score_hint: int | None = None

    @field_validator("source_url", "direct_job_url")
    @classmethod
    def validate_lead_urls(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class JobLeadSearchResult(BaseModel):
    leads: list[JobLead]


class SearchFailure(BaseModel):
    stage: Literal["discovery", "resolution", "validation", "filter"]
    reason_code: str
    detail: str
    company_name: str | None = None
    role_title: str | None = None
    source_query: str | None = None
    source_url: str | None = None
    direct_job_url: str | None = None
    posted_date_text: str | None = None
    salary_text: str | None = None
    is_remote: bool | None = None
    lead_refined_by_ollama: bool | None = None
    source_quality_score: int | None = None
    attempt_number: int | None = None
    round_number: int | None = None


class NearMissJob(BaseModel):
    company_name: str
    role_title: str
    reason_code: str
    detail: str
    why_close: str
    source_url: str | None = None
    direct_job_url: str | None = None
    source_type: str | None = None
    ats_platform: str | None = None
    posted_date_text: str | None = None
    salary_text: str | None = None
    is_remote: bool | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    validation_evidence: list[str] = Field(default_factory=list)
    close_score: int = 0
    source_quality_score: int = 0
    attempt_number: int | None = None
    round_number: int | None = None

    @field_validator("source_url", "direct_job_url")
    @classmethod
    def validate_near_miss_urls(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class FalseNegativeAuditEntry(BaseModel):
    reason_code: str
    verdict: Literal["correct_rejection", "fixable", "near_miss"]
    company_name: str | None = None
    role_title: str | None = None
    detail: str
    notes: str
    source_url: str | None = None
    direct_job_url: str | None = None
    salary_text: str | None = None
    posted_date_text: str | None = None
    is_remote: bool | None = None
    attempt_number: int | None = None
    round_number: int | None = None

    @field_validator("source_url", "direct_job_url")
    @classmethod
    def validate_audit_urls(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class SearchPassSummary(BaseModel):
    attempt_number: int
    unique_leads_discovered: int
    qualifying_jobs: int
    failure_counts: dict[str, int] = Field(default_factory=dict)
    accepted_job_urls: list[str] = Field(default_factory=list)
    query_count: int


class SearchDiagnostics(BaseModel):
    run_id: str | None = None
    minimum_qualifying_jobs: int
    unique_leads_discovered: int = 0
    seed_replayed_lead_count: int = 0
    failures: list[SearchFailure] = Field(default_factory=list)
    passes: list[SearchPassSummary] = Field(default_factory=list)
    near_misses: list[NearMissJob] = Field(default_factory=list)
    false_negative_audit: list[FalseNegativeAuditEntry] = Field(default_factory=list)


class DirectJobResolution(BaseModel):
    accepted: bool
    direct_job_url: str | None = None
    ats_platform: str | None = None
    evidence_notes: str = Field(description="Short explanation of how the direct job URL was resolved.")
    rejection_reason: str | None = None

    @field_validator("direct_job_url")
    @classmethod
    def validate_direct_job_url(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class JobPosting(BaseModel):
    company_name: str = Field(description="Company name.")
    role_title: str = Field(description="Job title.")
    direct_job_url: str = Field(description="Direct ATS job URL, never an aggregator.")
    resolved_job_url: str | None = Field(
        default=None,
        description="Resolved final ATS job URL after redirects.",
    )
    ats_platform: str = Field(description="ATS vendor or direct careers platform.")
    location_text: str = Field(description="Location text from the posting.")
    is_fully_remote: bool | None = Field(default=None, description="True only if the role is fully remote.")
    posted_date_text: str = Field(description="Human-readable posted date text from the source.")
    posted_date_iso: str | None = Field(
        default=None,
        description="ISO date when available. Leave null if unavailable.",
    )
    base_salary_min_usd: int | None = Field(default=None)
    base_salary_max_usd: int | None = Field(default=None)
    salary_text: str | None = Field(default=None)
    salary_inferred: bool = Field(default=False)
    salary_inference_reason: str | None = Field(default=None)
    inferred_experience_years_min: int | None = Field(default=None)
    source_query: str | None = Field(default=None)
    job_page_title: str | None = Field(default=None)
    evidence_notes: str = Field(description="Short explanation of how the filters were satisfied.")
    validation_evidence: list[str] = Field(default_factory=list)
    lead_refined_by_ollama: bool = Field(default=False)
    source_quality_score: int | None = Field(default=None)

    @field_validator("direct_job_url", "resolved_job_url")
    @classmethod
    def validate_job_urls(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class JobSearchResult(BaseModel):
    jobs: list[JobPosting]


class LinkedInContact(BaseModel):
    name: str
    profile_url: str
    headline: str | None = None
    company_text: str | None = None
    connection_degree: Literal["1st", "2nd"]
    mutual_connection_names: list[str] = Field(default_factory=list)
    connected_first_order_names: list[str] = Field(default_factory=list)
    connected_first_order_profile_urls: dict[str, str] = Field(default_factory=dict)
    connected_first_order_message_histories: dict[str, list[str]] = Field(default_factory=dict)
    message_history: list[str] = Field(default_factory=list)

    @field_validator("profile_url")
    @classmethod
    def validate_profile_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("Profile URL must start with http:// or https://")
        return value


class ManualReviewLink(BaseModel):
    label: str
    url: str

    @field_validator("url")
    @classmethod
    def validate_manual_review_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class FirstOrderMessage(BaseModel):
    contact_name: str
    contact_profile_url: str
    subject_context: str
    message_body: str

    @field_validator("contact_profile_url")
    @classmethod
    def validate_contact_profile_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("Profile URL must start with http:// or https://")
        return value


class SecondOrderIntroMessage(BaseModel):
    first_order_contact_name: str
    first_order_contact_profile_url: str | None = None
    second_order_contact_names: list[str] = Field(default_factory=list)
    second_order_contact_profile_urls: list[str] = Field(default_factory=list)
    message_body: str

    @field_validator("first_order_contact_profile_url")
    @classmethod
    def validate_intro_profile_urls(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.startswith(("http://", "https://")):
            raise ValueError("Profile URL must start with http:// or https://")
        return value

    @field_validator("second_order_contact_profile_urls")
    @classmethod
    def validate_intro_target_profile_urls(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.startswith(("http://", "https://")):
                raise ValueError("Profile URL must start with http:// or https://")
        return value


class JobOutreachBundle(BaseModel):
    job: JobPosting
    first_order_contacts: list[LinkedInContact] = Field(default_factory=list)
    second_order_contacts: list[LinkedInContact] = Field(default_factory=list)
    first_order_messages: list[FirstOrderMessage] = Field(default_factory=list)
    second_order_messages: list[SecondOrderIntroMessage] = Field(default_factory=list)
    manual_review_links: list[ManualReviewLink] = Field(default_factory=list)
    manual_review_notes: list[str] = Field(default_factory=list)


class JobSummaryRow(BaseModel):
    company_name: str
    role_title: str
    direct_job_url: str
    posted_date_text: str
    salary_text: str | None = None
    second_order_message_count: int

    @field_validator("direct_job_url")
    @classmethod
    def validate_summary_job_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return value


class RunManifest(BaseModel):
    run_id: str
    generated_at: datetime
    message_docx_path: str
    summary_docx_path: str
    jobs_found_by_search: int
    jobs_kept_after_validation: int
    jobs_with_any_messages: int
    near_miss_docx_path: str | None = None
    near_miss_json_path: str | None = None
    ollama_summary_json_path: str | None = None
    near_miss_count: int = 0


class OllamaTuningProfile(BaseModel):
    model: str
    keep_alive: str
    num_ctx: int
    num_batch: int
    num_predict: int
    degraded: bool = False
    degraded_reason: str | None = None
    last_updated_at: datetime | None = None
    based_on_event_count: int = 0


class OllamaRunSummary(BaseModel):
    run_id: str
    generated_at: datetime
    tuning_profile: OllamaTuningProfile
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    outer_timeout_count: int = 0
    success_rate: float = 0.0
    warm_hit_rate: float = 0.0
    median_wall_duration_seconds: float | None = None
    p95_wall_duration_seconds: float | None = None
    median_warm_wall_duration_seconds: float | None = None
    p95_warm_wall_duration_seconds: float | None = None
    failure_breakdown: dict[str, int] = Field(default_factory=dict)
    caller_breakdown: dict[str, int] = Field(default_factory=dict)
    prompt_category_breakdown: dict[str, int] = Field(default_factory=dict)
    quality_counters: dict[str, float] = Field(default_factory=dict)


class RunOutcomeMetrics(BaseModel):
    validated_jobs_count: int = 0
    jobs_with_messages_count: int = 0
    unique_leads_discovered_count: int = 0
    fresh_new_leads_count: int = 0
    actionable_near_miss_count: int = 0
    raw_near_miss_count: int = 0


class RunDiscoveryMetrics(BaseModel):
    unique_leads_discovered_count: int = 0
    fresh_new_leads_count: int = 0
    replayed_seed_leads_count: int = 0
    repeated_failed_leads_suppressed_count: int = 0
    executed_query_count: int = 0
    query_timeout_count: int = 0
    query_skipped_timeout_budget_count: int = 0
    zero_yield_pass_count: int = 0
    discovery_efficiency: float = 0.0


class RunValidationMetrics(BaseModel):
    validated_jobs_count: int = 0
    validated_yield: float = 0.0
    jobs_with_messages_count: int = 0
    message_coverage_rate: float = 0.0
    raw_near_miss_count: int = 0
    actionable_near_miss_count: int = 0
    actionable_near_miss_yield: float = 0.0
    company_mismatch_count: int = 0
    not_specific_job_page_count: int = 0
    missing_salary_count: int = 0
    fetch_non_200_count: int = 0
    stale_posting_count: int = 0
    not_remote_count: int = 0
    false_negative_fixable_count: int = 0
    false_negative_near_miss_count: int = 0
    false_negative_correct_rejection_count: int = 0


class RunOllamaMetrics(BaseModel):
    model: str | None = None
    degraded: bool = False
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    outer_timeout_count: int = 0
    warm_hit_rate: float = 0.0
    median_wall_duration_seconds: float | None = None
    p95_wall_duration_seconds: float | None = None
    useful_action_count: float = 0.0
    useful_actions_per_request: float = 0.0
    quality_counters: dict[str, float] = Field(default_factory=dict)


class RunTimingMetrics(BaseModel):
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    time_to_first_validated_job_seconds: float | None = None


class RunScorecard(BaseModel):
    run_id: str
    generated_at: datetime
    status: str
    outcome: RunOutcomeMetrics = Field(default_factory=RunOutcomeMetrics)
    discovery: RunDiscoveryMetrics = Field(default_factory=RunDiscoveryMetrics)
    validation: RunValidationMetrics = Field(default_factory=RunValidationMetrics)
    ollama: RunOllamaMetrics = Field(default_factory=RunOllamaMetrics)
    timing: RunTimingMetrics = Field(default_factory=RunTimingMetrics)
    message_docx_path: str | None = None
    summary_docx_path: str | None = None
    near_miss_docx_path: str | None = None
    near_miss_json_path: str | None = None
    ollama_summary_json_path: str | None = None


class ImprovementPattern(BaseModel):
    key: str
    summary: str
    severity_score: float = 0.0
    evidence: dict[str, float | int | str | None] = Field(default_factory=dict)


class RunImprovementAnalysis(BaseModel):
    iteration_number: int
    generated_at: datetime
    target_run_id: str | None = None
    analyzed_run_ids: list[str] = Field(default_factory=list)
    recent_selected_themes: list[str] = Field(default_factory=list)
    current_run_status: str
    current_metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    metric_deltas: dict[str, float] = Field(default_factory=dict)
    top_patterns: list[ImprovementPattern] = Field(default_factory=list)
    selected_theme: str
    selected_summary: str
    acceptance_checks: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)


class ValidationCommandResult(BaseModel):
    command: str
    passed: bool
    exit_code: int
    output_path: str | None = None


class CodexIterationResult(BaseModel):
    iteration_number: int
    generated_at: datetime
    status: Literal["succeeded", "failed", "validation_failed", "no_changes", "commit_failed", "session_failed"]
    session_id: str | None = None
    selected_theme: str | None = None
    prompt_path: str | None = None
    log_path: str | None = None
    last_message_path: str | None = None
    exit_code: int | None = None
    summary: str | None = None
    validation_commands: list[str] = Field(default_factory=list)
    validation_results: list[ValidationCommandResult] = Field(default_factory=list)
    commit_hash: str | None = None
    commit_message: str | None = None


class AutoLoopIteration(BaseModel):
    iteration_number: int
    run_id: str | None = None
    run_status: Literal["completed", "failed", "unknown"] = "unknown"
    started_at: datetime
    completed_at: datetime | None = None
    selected_theme: str | None = None
    analysis_path: str | None = None
    prompt_path: str | None = None
    result_path: str | None = None
    codex_log_path: str | None = None
    codex_last_message_path: str | None = None
    commit_hash: str | None = None
    validation_passed: bool = False


class AutoLoopState(BaseModel):
    enabled: bool = False
    status: Literal["idle", "running", "analysis", "waiting_for_codex", "validating", "stopped", "failed"] = "idle"
    target_attempts: int = 0
    completed_attempts: int = 0
    current_iteration: int = 0
    current_run_id: str | None = None
    codex_session_id: str | None = None
    baseline_commit_hash: str | None = None
    latest_commit_hash: str | None = None
    latest_validation_result: str | None = None
    last_failure_summary: str | None = None
    started_at: datetime | None = None
    updated_at: datetime | None = None
    iterations: list[AutoLoopIteration] = Field(default_factory=list)
