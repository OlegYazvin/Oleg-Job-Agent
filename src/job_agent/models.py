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
    attempt_number: int | None = None
    round_number: int | None = None


class SearchPassSummary(BaseModel):
    attempt_number: int
    unique_leads_discovered: int
    qualifying_jobs: int
    failure_counts: dict[str, int] = Field(default_factory=dict)
    accepted_job_urls: list[str] = Field(default_factory=list)
    query_count: int


class SearchDiagnostics(BaseModel):
    minimum_qualifying_jobs: int
    unique_leads_discovered: int = 0
    failures: list[SearchFailure] = Field(default_factory=list)
    passes: list[SearchPassSummary] = Field(default_factory=list)


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
