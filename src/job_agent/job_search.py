from __future__ import annotations

import asyncio
import base64
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from html import unescape
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from urllib.parse import parse_qs, parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse

from agents import Agent, Runner, WebSearchTool
from bs4 import BeautifulSoup
import httpx

from .company_discovery import (
    KNOWN_BOARD_HOST_FRAGMENTS,
    LOW_TRUST_SCOUT_HOST_FRAGMENTS,
    append_company_discovery_audit_entry,
    board_identifier_from_url,
    board_url_ats_type,
    company_discovery_audit_path,
    company_discovery_crawl_history_path,
    company_discovery_frontier_path,
    company_discovery_index_path,
    default_careers_candidate_urls,
    extract_careers_page_urls,
    extract_directory_company_tasks,
    extract_company_homepage_urls,
    extract_embedded_board_urls,
    frontier_task_key,
    infer_careers_root,
    is_company_discovery_seed_url,
    is_low_value_company_discovery_entry,
    load_company_discovery_audit,
    load_company_discovery_crawl_history,
    load_company_discovery_entries,
    load_company_discovery_frontier,
    record_crawl_result,
    save_company_discovery_audit,
    save_company_discovery_crawl_history,
    save_company_discovery_entries,
    save_company_discovery_frontier,
    select_frontier_tasks,
    select_directory_company_tasks,
    source_directory_seed_tasks,
    trust_score_for_url,
    update_frontier_task_state,
    upsert_frontier_task,
    upsert_company_discovery_entry,
    workday_board_root_url,
)
from .config import Settings
from .criteria import DEFAULT_ROLE_SEARCH_PROFILE
from .history import (
    load_company_watchlist_entries,
    load_previously_reported_company_keys,
    load_validated_job_history_index,
)
from .job_pages import USER_AGENT, JobPageSnapshot, fetch_job_page
from .llm_provider import LLMProviderError, OllamaStructuredProvider
from .models import (
    DiscoveryFrontierSuggestionResult,
    DirectJobResolution,
    FalseNegativeAuditEntry,
    JobLead,
    JobLeadSearchResult,
    JobPosting,
    NearMissJob,
    SearchDiagnostics,
    SearchFailure,
    SearchPassSummary,
)
from .ollama_runtime import prewarm_ollama_model, record_ollama_event
from .status import StatusReporter
from .storage import save_json_snapshot


ALLOWED_JOB_HOST_FRAGMENTS = (
    "greenhouse.io",
    "job-boards.greenhouse.io",
    "boards.greenhouse.io",
    "jobs.lever.co",
    "lever.co",
    "ashbyhq.com",
    "recruitee.com",
    "careers.tellent.com",
    "comeet.com",
    "jobscore.com",
    "myworkdayjobs.com",
    "jobvite.com",
    "smartrecruiters.com",
    "workable.com",
    "icims.com",
    "bamboohr.com",
    "dayforcehcm.com",
    "recruiting.paylocity.com",
    "adp.com",
    "paylocity.com",
    "ultipro.com",
    "portal.dynamicsats.com",
    "ats.rippling.com",
)

BLOCKED_JOB_HOST_FRAGMENTS = (
    "linkedin.com",
    "indeed.com",
    "ziprecruiter.com",
    "glassdoor.com",
    "builtin.com",
    "builtinchicago",
    "wellfound.com",
    "monster.com",
    "careerbuilder.com",
    "dice.com",
    "simplyhired.com",
    "talent.com",
    "jooble.org",
    "adzuna.com",
    "remoteok.com",
    "otta.com",
    "startup.jobs",
    "employbl.com",
    "weloveproduct.co",
    "jobsora.com",
    "jobgether.com",
    "themuse.com",
    "welcometothejungle.com",
    "joinhandshake.com",
    "remoterocketship.com",
    "getro.com",
    "fitt.co",
    "jobright.ai",
    "grabjobs.co",
    "disabledperson.com",
    "apna.co",
    "publiremote.com",
    "tealhq.com",
    "ladders.com",
    "empllo.com",
    "thepmrepo.com",
    "thehomebase.ai",
    "thesaraslist.com",
    "jobs.generalcatalyst.com",
)
LINKEDIN_RESOLUTION_BLOCKED_HOST_FRAGMENTS = (
    "joinleland.com",
    "remotejobshive.com",
    "thatstartupjob.com",
    "tangerinefeed.net",
    "cari.me.uk",
    "tracxn.com",
)
LOW_TRUST_REACQUISITION_HOST_FRAGMENTS = (
    "mediabistro.com",
    "smartrecruiterscareers.com",
    "remotejobshive.com",
    "thatstartupjob.com",
    "remoterocketship.com",
    "jobgether.com",
    "tracxn.com",
)

JOB_PATH_HINTS = (
    "/careers/",
    "/career/",
    "/jobs/",
    "/job/",
    "/openings/",
    "/positions/",
    "/apply/",
    "/requisitions/",
    "/recruiting/",
)
COMPANY_JOB_QUERY_HINTS = (
    "gh_jid",
    "gh_src",
    "jid",
    "jobid",
    "job_id",
    "reqid",
    "req_id",
    "requisitionid",
    "requisition_id",
    "posting",
)
GENERIC_COMPANY_CAREERS_SEGMENTS = {
    "jobs",
    "job",
    "careers",
    "career",
    "apply",
    "applications",
    "openings",
    "positions",
    "join-us",
    "joinus",
    "search",
    "job-search",
    "search-jobs",
}
GENERIC_CAREERS_TAIL_SEGMENTS = {
    "us",
    "eu",
    "uk",
    "ca",
    "en",
    "de",
    "fr",
    "es",
    "apac",
    "emea",
    "locations",
    "culture",
    "people",
    "team",
    "stores",
    "retail",
    "restaurants",
}
GENERIC_HOST_SUBDOMAIN_PREFIXES = {
    "www",
    "jobs",
    "job",
    "careers",
    "career",
    "apply",
    "join",
    "workat",
    "recruiting",
}
REDIRECT_QUERY_PARAM_NAMES = {
    "url",
    "u",
    "target",
    "target_url",
    "targeturl",
    "redirect",
    "redirect_url",
    "redirecturl",
    "dest",
    "destination",
    "next",
    "continue",
    "joburl",
    "job_url",
}

DISCOVERY_SOURCE_PRIORITY = {
    "direct_ats": 0,
    "company_site": 1,
    "linkedin": 2,
    "builtin": 3,
    "glassdoor": 4,
    "other": 5,
}

RECENCY_REASON_CODES = {"stale_posting", "missing_posted_date"}
SALARY_REASON_CODES = {"missing_salary", "salary_below_min", "salary_not_base"}
REMOTE_REASON_CODES = {"not_remote", "remote_unclear"}
RESOLUTION_REASON_CODES = {"resolution_missing", "resolution_blocked_url", "not_specific_job_page", "fetch_non_200"}
REPLAYABLE_FAILURE_REASON_CODES = {
    "resolution_missing",
    "resolution_blocked_url",
    "fetch_non_200",
    "missing_salary",
    "missing_posted_date",
    "remote_unclear",
    "validation_timeout",
    "resolution_timeout",
}
FAILED_LEAD_TRACKED_REASON_CODES = {
    "stale_posting",
    "missing_salary",
    "not_remote",
    "remote_unclear",
    "salary_below_min",
    "salary_not_base",
    "company_mismatch",
    "direct_url_not_allowed",
    "not_specific_job_page",
    "fetch_non_200",
    "resolution_missing",
    "resolution_blocked_url",
    "not_ai_product_manager",
    "already_reported",
}
FAILED_LEAD_IMMEDIATE_SUPPRESS_REASON_CODES = {
    "direct_url_not_allowed",
    "not_specific_job_page",
}
FAILED_LEAD_REPEAT_SUPPRESS_THRESHOLDS = {
    "company_mismatch": 2,
    "stale_posting": 2,
    "missing_salary": 1,
    "not_remote": 2,
    "remote_unclear": 3,
    "salary_below_min": 2,
    "salary_not_base": 2,
    "not_ai_product_manager": 2,
    "fetch_non_200": 3,
    "resolution_missing": 3,
    "resolution_blocked_url": 3,
}
ADAPTIVE_FOCUS_REASON_CODES = {
    "resolution_missing",
    "resolution_blocked_url",
    "fetch_non_200",
    "not_specific_job_page",
    "missing_posted_date",
    "missing_salary",
    "salary_below_min",
    "salary_not_base",
    "remote_unclear",
}
FOCUSABLE_REASON_CODES = {
    "resolution_missing",
    "resolution_blocked_url",
    "fetch_non_200",
    "not_specific_job_page",
    "missing_salary",
    "remote_unclear",
}
TRUSTED_DIRECT_LEAD_SUPPRESS_REASON_CODES = {
    "stale_posting",
    "not_remote",
    "salary_below_min",
}
BROAD_REMOTE_OVERRIDE_MARKERS = (
    "100% remote",
    "fully remote",
    "remote - united states",
    "remote, united states",
    "united states remote",
    "remote us",
    "remote - us",
    "remote - usa",
    "remote, usa",
    "work from home",
)
BROAD_GENERIC_QUERY_TIMEOUT_SKIP_THRESHOLD = 6
COMPANY_FOCUSED_QUERY_TIMEOUT_SKIP_THRESHOLD = 2
TIMEOUT_SENSITIVE_QUERY_MARKERS = (
    "site:workatastartup.com/jobs",
    "site:getro.com/companies",
    "site:ycombinator.com/companies",
    "site:wellfound.com/jobs",
)
TIMEOUT_SENSITIVE_QUERY_SKIP_THRESHOLD = 2
MAX_ROUND_LEADS_PER_COMPANY = 3
MAX_LOW_TRUST_DIRECT_RESOLUTION_LEADS = 2

LOCAL_OLLAMA_SEMAPHORE = asyncio.Semaphore(1)
BUILTIN_PRIMARY_BASE_URL = "https://builtin.com"
BUILTIN_REGIONAL_BASE_URLS = (
    "https://www.builtinnyc.com",
    "https://www.builtinsf.com",
    "https://www.builtinseattle.com",
    "https://www.builtinchicago.org",
    "https://www.builtinla.com",
)
BUILTIN_CATEGORY_PAGE_COUNT = 3
BUILTIN_HTML_CACHE: dict[str, str] = {}
BUILTIN_LEAD_CACHE: dict[str, JobLead | None] = {}
GREENHOUSE_BOARD_JOBS_CACHE: dict[str, list[dict[str, object]]] = {}
ASHBY_BOARD_JOBS_CACHE: dict[str, list[dict[str, object]]] = {}
LEVER_BOARD_JOBS_CACHE: dict[str, list[dict[str, object]]] = {}
SMARTRECRUITERS_BOARD_JOBS_CACHE: dict[str, list[dict[str, object]]] = {}
SMARTRECRUITERS_POSTING_DETAILS_CACHE: dict[str, dict[str, object] | None] = {}
WORKDAY_BOARD_JOBS_CACHE: dict[str, list[dict[str, object]]] = {}
SUPPORTED_OFFICIAL_BOARD_PREFIXES = {"greenhouse", "ashby", "lever", "smartrecruiters", "workday"}
SEARCH_ENGINE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
}
GOOGLE_HTTP_SEARCH_AVAILABLE: bool | None = None
LINKEDIN_GUEST_SEARCH_BASE_URL = "https://www.linkedin.com/jobs/search/"
CAREERS_HUB_HINTS = (
    "/careers",
    "/career",
    "/jobs",
    "/job",
    "/openings",
    "/positions",
    "/join-us",
    "/joinus",
    "/company/careers",
    "/work-with-us",
)
GENERIC_CAREERS_INFO_HINTS = (
    "recruitment-fraud",
    "recruitmentfraud",
    "fraud",
    "how-we-hire",
    "how_we_hire",
    "being-a-",
    "being a ",
    "career-listing",
    "careerlisting",
    "join?domain=",
    "join-us",
    "joinus",
    "talent-community",
    "talentcommunity",
    "candidate/login",
)
COMPANY_SITE_JUNK_HOST_FRAGMENTS = (
    "reddit.com",
    "stackexchange.com",
    "stackoverflow.com",
    "mountainproject.com",
    "quora.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "pinterest.com",
    "jobicy.com",
)
COMPANY_RESOLUTION_URL_CACHE: dict[str, list[str]] = {}
LOCAL_SEARCH_JOB_BOARD_DOMAIN_BATCHES = (
    ("linkedin.com/jobs/view", "glassdoor.com/Job", "builtin.com/jobs"),
    ("wellfound.com/jobs", "workatastartup.com/jobs", "getro.com/companies"),
    ("monster.com/jobs", "startup.jobs", "jobgether.com"),
    ("welcometothejungle.com/en/companies", "joinhandshake.com/jobs", "ycombinator.com/companies"),
    ("dynamitejobs.com/company", "dailyremote.com/remote-job", "remote.io/remote-product-jobs"),
    ("flexhired.com/jobs", "remoteai.io/roles", "indeed.com/viewjob"),
)
LOCAL_SEARCH_DIRECT_ATS_DOMAIN_BATCHES = (
    ("job-boards.greenhouse.io", "boards.greenhouse.io", "jobs.lever.co"),
    ("jobs.ashbyhq.com", "myworkdayjobs.com", "jobs.smartrecruiters.com"),
    ("jobs.recruitee.com", "comeet.com/jobs", "jobscore.com"),
    ("jobs.jobvite.com", "jobs.workable.com", "jobs.icims.com"),
    ("careers.bamboohr.com", "jobs.dayforcehcm.com", "recruiting.paylocity.com"),
    ("careers.adp.com", "careers.workday.com"),
)
LOCAL_SEARCH_DOMAIN_BATCH_SPAN = 2
LOCAL_SEARCH_ATS_DOMAIN_QUERY_LIMIT = 3
LOCAL_SEARCH_BOARD_DOMAIN_QUERY_LIMIT = 2
LOCAL_SEARCH_TOTAL_QUERY_LIMIT = 14
SEARCH_QUERY_RESULT_CACHE: dict[tuple[str, int], list[tuple[str, str, str]]] = {}
FORCED_OLLAMA_REFINEMENT_ATTEMPTS: set[tuple[str, int]] = set()
FORCED_OLLAMA_ROUND_REFINEMENT_ATTEMPTS: set[tuple[str, int]] = set()
FORCED_OLLAMA_SEED_REFINEMENT_RUNS: set[str] = set()
LAZY_OLLAMA_PREWARM_RUNS: set[str] = set()
SEED_LEADS_FILENAME = "seed-leads.json"
FAILED_LEAD_HISTORY_FILENAME = "failed-lead-history.json"
QUERY_FAMILY_HISTORY_FILENAME = "query-family-history.json"
NOVEL_COMPANY_TARGET_RATIO = 0.90
LOW_TRUST_REPLAY_SOURCE_HOST_FRAGMENTS = (
    "mediabistro.com",
    "smartrecruiterscareers.com",
    "jobgether.com",
    "thatstartupjob.com",
    "remotejobshive.com",
    "tracxn.com",
    "remoterocketship.com",
)
QUERY_FAMILY_COOLDOWN_ELIGIBLE = {
    "broad_generic",
    "generic_discovery",
    "structured_ats",
    "startup_getro",
    "startup_generic",
    "startup_wellfound",
    "startup_workatastartup",
    "startup_ycombinator",
}
QUERY_FAMILY_COOLDOWN_MAX_AGE = timedelta(hours=48)
TRACKING_QUERY_PARAM_NAMES = {
    "gh_src",
    "gh_sid",
    "gh_oid",
    "gh_bid",
    "gh_aid",
    "gh_cid",
    "gh_u",
    "gclid",
    "fbclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
    "ref",
    "referrer",
    "source",
    "src",
    "trk",
}
SMALL_COMPANY_SCOUT_DOMAINS = (
    "jobs.ashbyhq.com",
    "jobs.lever.co",
    "jobs.recruitee.com",
    "careers.tellent.com",
    "comeet.com/jobs",
    "jobscore.com",
    "jobs.workable.com",
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.jobvite.com",
    "jobs.gem.com",
    "teamtailor.com/jobs",
    "careers.bamboohr.com",
    "recruiting.paylocity.com",
    "jobs.dayforcehcm.com",
    "jobs.smartrecruiters.com",
    "wellfound.com/jobs",
    "startup.jobs",
    "welcometothejungle.com/en/companies",
    "workatastartup.com/jobs",
    "ycombinator.com/companies",
    "dynamitejobs.com/company",
    "dailyremote.com/remote-job",
    "remoteai.io/roles",
    "jobgether.com",
    "flexhired.com/jobs",
)
PORTFOLIO_BOARD_SCOUT_DOMAINS = (
    "workatastartup.com/jobs",
    "wellfound.com/jobs",
    "getro.com/companies",
    "ycombinator.com/companies",
)
SMALL_COMPANY_SCOUT_TOPICS = (
    "AI",
    "machine learning",
    "generative AI",
    "agentic AI",
    "applied AI",
    "applied intelligence",
    "vertical AI",
    "LLM",
    "AI platform",
    "ML platform",
    "AI agents",
    "conversational AI",
    "voice AI",
    "AI workflow automation",
    "AI infrastructure",
    "data products",
)
SMALL_COMPANY_LOCAL_SCOUT_MODIFIERS = (
    "startup",
    "\"series a\"",
    "\"series b\"",
    "\"series c\"",
    "\"seed stage\"",
    "\"venture backed\"",
    "\"vc backed\"",
    "\"portfolio company\"",
    "\"early stage\"",
    "\"growth stage\"",
    "\"founding team\"",
    "\"small team\"",
    "\"high growth\"",
)
FOCUS_COMPANY_DISCOVERY_ONLY_HOST_FRAGMENTS = (
    "builtin",
    "linkedin.com",
    "glassdoor.com",
    "wellfound.com",
    "workatastartup.com",
    "getro.com",
    "ycombinator.com",
    "startup.jobs",
    "welcometothejungle.com",
    "joinhandshake.com",
    "dynamitejobs.com",
    "dailyremote.com",
    "remote.io",
    "remoteai.io",
    "jobgether.com",
    "flexhired.com",
    "monster.com",
)
ENTERPRISE_ATS_HOST_FRAGMENTS = (
    "myworkdayjobs.com",
    "careers.workday.com",
    "careers.adp.com",
    "jobs.icims.com",
)
NON_ACTIONABLE_WATCHLIST_REASON_CODES = {
    "already_reported",
    "company_mismatch",
    "direct_url_not_allowed",
    "fetch_non_200",
    "missing_salary",
    "not_remote",
    "not_specific_job_page",
    "remote_unclear",
    "resolution_missing",
    "salary_below_min",
    "stale_posting",
}
NEAR_MISS_REASON_CODES = {
    "remote_unclear",
    "missing_salary",
    "salary_below_min",
    "stale_posting",
    "fetch_non_200",
    "resolution_missing",
}
FALSE_NEGATIVE_AUDIT_REASON_CODES = {
    "not_ai_product_manager",
    "remote_unclear",
    "fetch_non_200",
    "resolution_missing",
    "salary_below_min",
    "missing_salary",
}
LOW_SIGNAL_ROLE_TERMS = {
    "strategist",
    "strategy",
    "operations",
    "ops",
    "consultant",
    "consulting",
    "program manager",
    "project manager",
    "partnerships",
    "marketing",
    "growth",
    "customer success",
    "solutions architect",
    "sales",
    "business development",
}
CLOSE_MISS_SALARY_BUFFER_USD = 25000
NEAR_MISS_STALE_BUFFER_DAYS = 7


class SearchQueryTimeoutError(asyncio.TimeoutError):
    def __init__(self, query: str) -> None:
        super().__init__(f"Timed out while searching query: {query}")
        self.query = query


class SearchQueryExecutionError(RuntimeError):
    def __init__(self, query: str, cause: Exception) -> None:
        super().__init__(str(cause))
        self.query = query
        self.cause = cause


class OpenAIQuotaExceededError(RuntimeError):
    pass


class LocalSearchBackendBlockedError(RuntimeError):
    pass


@dataclass(slots=True)
class SearchTuning:
    attempt_number: int
    prioritize_recency: bool = False
    prioritize_salary: bool = False
    prioritize_remote: bool = False
    focus_companies: list[str] = field(default_factory=list)
    focus_roles: list[str] = field(default_factory=list)


def _looks_like_company_job_page(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    if any(fragment in host for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
        return False
    if any(fragment in host for fragment in COMPANY_SITE_JUNK_HOST_FRAGMENTS):
        return False
    if any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
        return _looks_like_direct_ats_job_path(host, path)
    if any(token in query for token in COMPANY_JOB_QUERY_HINTS):
        return True
    if any(hint in f"{path}/" for hint in JOB_PATH_HINTS):
        return _has_company_job_detail_signal(parsed)
    if host.startswith(("jobs.", "careers.", "apply.", "join.", "workat.", "recruiting.")):
        return _has_company_job_detail_signal(parsed)
    return False


def _unwrap_direct_job_url(url: str) -> str:
    candidate = str(url or "").strip()
    if not candidate:
        return ""

    seen: set[str] = set()
    while candidate and candidate not in seen:
        seen.add(candidate)

        decoded = _decode_search_result_url(candidate)
        if decoded != candidate:
            candidate = decoded
            continue

        try:
            parsed = urlparse(candidate)
        except ValueError:
            break
        next_candidate: str | None = None
        for key, value in parse_qsl(parsed.query, keep_blank_values=True):
            if key.lower() not in REDIRECT_QUERY_PARAM_NAMES:
                continue
            unwrapped = unquote(unescape(value)).strip()
            if unwrapped.startswith(("http://", "https://")) and unwrapped != candidate:
                next_candidate = unwrapped
                break

        if next_candidate is None and (parsed.netloc or "").lower() in {"ad.doubleclick.net", "clickserve.dartsearch.net"}:
            decoded_query = unquote(unescape(parsed.query))
            match = re.search(r"https?://[^\s\"'<>]+", decoded_query)
            if match:
                next_candidate = match.group(0)

        if next_candidate is None:
            decoded_candidate = unquote(unescape(candidate))
            match = re.search(r"https?://[^\s\"'<>]+", decoded_candidate)
            if match and match.group(0) != candidate:
                next_candidate = match.group(0)

        if next_candidate is None:
            break
        candidate = next_candidate

    return candidate


def _normalize_direct_job_url(url: str) -> str:
    unwrapped_url = _unwrap_direct_job_url(url)
    if not unwrapped_url:
        return ""
    try:
        parsed = urlparse(unwrapped_url)
    except ValueError:
        return ""
    host = (parsed.netloc or "").lower()
    path = parsed.path.rstrip("/")
    query = _strip_tracking_query_params(parsed.query)

    if "jobs.lever.co" in host and "/apply" in path:
        path = path.split("/apply", 1)[0]
        query = ""
    elif "ashbyhq.com" in host:
        segments = _path_segments(path)
        if len(segments) >= 2:
            path = "/" + "/".join(segments[:2])
        elif path.endswith("/application"):
            path = path[: -len("/application")]
        query = ""
    elif "greenhouse.io" in host and path.endswith("/application"):
        path = path[: -len("/application")]
        query = ""
    elif "jobvite.com" in host and path.endswith("/apply"):
        path = path[: -len("/apply")]
        query = ""
    elif ("recruitee.com" in host or "careers.tellent.com" in host) and path.endswith("/apply"):
        path = path[: -len("/apply")]
        query = ""
    elif any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
        query = ""

    if not path:
        path = "/"

    try:
        return urlunparse(parsed._replace(path=path, query=query, fragment=""))
    except ValueError:
        return ""


def _direct_job_url_matches_expected_company(url: str, expected_company_name: str | None) -> bool:
    if not expected_company_name or _is_weak_company_hint(expected_company_name):
        return True
    company_hint = _company_hint_from_url(url)
    if _is_weak_company_hint(company_hint):
        return True
    return _company_names_match(expected_company_name, company_hint)


def _is_linkedin_resolution_blocked_host(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(fragment in host for fragment in LINKEDIN_RESOLUTION_BLOCKED_HOST_FRAGMENTS)


def _candidate_direct_job_url_is_trustworthy(url: str, lead: JobLead) -> bool:
    normalized = _normalize_direct_job_url(url)
    if not normalized:
        return False
    if lead.source_type == "linkedin" and _is_linkedin_resolution_blocked_host(normalized):
        return False
    if not _is_allowed_direct_job_url(normalized) or _looks_like_generic_job_url(normalized):
        return False
    return _direct_job_url_matches_expected_company(normalized, lead.company_name)


def _lead_direct_job_url_precheck_failure(
    lead: JobLead,
    *,
    attempt_number: int,
    round_number: int,
) -> SearchFailure | None:
    normalized_url = _normalize_direct_job_url(lead.direct_job_url or "")
    if not normalized_url:
        return None
    if lead.source_type == "linkedin" and _is_linkedin_resolution_blocked_host(normalized_url):
        return _make_failure(
            stage="resolution",
            reason_code="resolution_blocked_url",
            detail="Lead direct URL pointed at a blocked LinkedIn resolution host instead of a real ATS or company careers page.",
            lead=lead,
            direct_job_url=normalized_url,
            attempt_number=attempt_number,
            round_number=round_number,
        )
    if _looks_like_generic_job_url(normalized_url):
        return _make_failure(
            stage="resolution",
            reason_code="not_specific_job_page",
            detail="Lead direct URL pointed at a generic careers page rather than a specific job posting.",
            lead=lead,
            direct_job_url=normalized_url,
            attempt_number=attempt_number,
            round_number=round_number,
        )
    if not _direct_job_url_matches_expected_company(normalized_url, lead.company_name):
        company_hint = _company_hint_from_url(normalized_url)
        if not _is_weak_company_hint(company_hint):
            return _make_failure(
                stage="resolution",
                reason_code="company_mismatch",
                detail=(
                    f"Lead direct URL company hint '{company_hint}' did not match expected company "
                    f"'{lead.company_name}'."
                ),
                lead=lead,
                direct_job_url=normalized_url,
                attempt_number=attempt_number,
                round_number=round_number,
            )
    return None


def _extract_company_board_identifier(url: str | None) -> str | None:
    normalized = _normalize_direct_job_url(str(url or ""))
    if not normalized.startswith(("http://", "https://")):
        return None
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").lower()
    segments = _path_segments(parsed.path)
    query = parse_qs(parsed.query)

    if "greenhouse.io" in host and segments:
        board_token = segments[0]
        if board_token:
            return f"greenhouse:{board_token.lower()}"
    if "jobs.lever.co" in host and segments:
        return f"lever:{segments[0].lower()}"
    if "ashbyhq.com" in host and segments:
        return f"ashby:{segments[0].lower()}"
    if "myworkdayjobs.com" in host:
        host_prefix = host.split(".wd", 1)[0]
        if host_prefix:
            return f"workday:{host_prefix.lower()}"
    if "smartrecruiters.com" in host and segments:
        return f"smartrecruiters:{segments[0].lower()}"
    if "recruitee.com" in host or "careers.tellent.com" in host:
        if len(segments) >= 2 and segments[0] == "o":
            return f"recruitee:{segments[1].lower()}"
    if "jobscore.com" in host:
        host_prefix = host.split(".jobscore.com", 1)[0]
        if host_prefix and host_prefix != host:
            return f"jobscore:{host_prefix.lower()}"
    for field in ("gh_jid", "gh_job_id"):
        values = query.get(field)
        if values and values[0].strip():
            return f"{host}:{values[0].strip()}"
    return None


def _canonical_job_key(url: str | None) -> str | None:
    normalized = _normalize_direct_job_url(str(url or ""))
    if not normalized.startswith(("http://", "https://")):
        return None
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").lower()
    segments = _path_segments(parsed.path)
    query = parse_qs(parsed.query)
    board_identifier = _extract_company_board_identifier(normalized)

    if "greenhouse.io" in host:
        job_id = None
        gh_values = query.get("gh_jid") or query.get("gh_job_id")
        if gh_values and gh_values[0].strip():
            job_id = gh_values[0].strip()
        elif "jobs" in segments:
            try:
                job_id = segments[segments.index("jobs") + 1]
            except (ValueError, IndexError):
                job_id = None
        if board_identifier and job_id:
            return f"{board_identifier}:{job_id}"

    if "jobs.lever.co" in host and len(segments) >= 2:
        return f"lever:{segments[0].lower()}:{segments[1].lower()}"

    if "ashbyhq.com" in host and len(segments) >= 2:
        return f"ashby:{segments[0].lower()}:{segments[1].lower()}"

    if "myworkdayjobs.com" in host:
        gh_values = query.get("gh_jid") or query.get("gh_job_id")
        if gh_values and gh_values[0].strip():
            return f"{board_identifier or host}:{gh_values[0].strip()}"
        if segments:
            tail = segments[-1]
            if "_" in tail:
                job_id = tail.rsplit("_", 1)[-1]
                if job_id:
                    return f"{board_identifier or host}:{job_id}"

    if "smartrecruiters.com" in host and len(segments) >= 2:
        return f"smartrecruiters:{segments[0].lower()}:{segments[1].lower()}"

    if ("recruitee.com" in host or "careers.tellent.com" in host) and len(segments) >= 2 and segments[0] == "o":
        return f"recruitee:{segments[1].lower()}"

    if "jobscore.com" in host and segments:
        if "jobs" in segments:
            return f"{board_identifier or host}:{segments[-1].lower()}"
        if "job" in segments:
            return f"{board_identifier or host}:{segments[-1].lower()}"

    return normalized


def _job_history_key_candidates(url: str | None) -> set[str]:
    normalized = _normalize_direct_job_url(str(url or ""))
    candidates = {normalized} if normalized else set()
    canonical = _canonical_job_key(url)
    if canonical:
        candidates.add(canonical)
    return {candidate for candidate in candidates if candidate}


def _job_history_primary_key(url: str | None) -> str:
    canonical = _canonical_job_key(url)
    if canonical:
        return canonical
    return _normalize_direct_job_url(str(url or ""))


def _validated_job_history_entry_for_url(
    url: str | None,
    validated_job_history_index: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    for candidate in sorted(_job_history_key_candidates(url)):
        entry = validated_job_history_index.get(candidate)
        if isinstance(entry, dict):
            return entry
    return None


def _reacquisition_history_metadata(entry: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(entry, dict):
        return {}
    return {
        "canonical_job_key": str(entry.get("canonical_job_key") or entry.get("job_key") or "").strip() or None,
        "first_reported_at": str(entry.get("first_reported_at") or "").strip() or None,
        "last_reported_at": str(entry.get("last_reported_at") or "").strip() or None,
        "report_count": max(1, int(entry.get("report_count") or 0) + 1),
    }


def _lead_is_reacquisition_eligible(
    lead: JobLead,
    settings: Settings,
    *,
    direct_job_url: str | None,
) -> bool:
    normalized_direct_url = _normalize_direct_job_url(direct_job_url or "")
    if not normalized_direct_url or not _is_allowed_direct_job_url(normalized_direct_url):
        return False
    direct_host = (urlparse(normalized_direct_url).netloc or "").lower()
    source_host = (urlparse(str(lead.source_url)).netloc or "").lower()
    if not direct_host:
        return False
    if any(fragment in direct_host for fragment in LOW_TRUST_REACQUISITION_HOST_FRAGMENTS):
        return False
    if (
        source_host
        and any(fragment in source_host for fragment in LOW_TRUST_REACQUISITION_HOST_FRAGMENTS)
        and lead.source_type not in {"direct_ats", "company_site"}
    ):
        return False
    if lead.source_type in {"direct_ats", "company_site"}:
        return True
    if _lead_has_trusted_source_fallback_evidence(lead, settings):
        return True
    return (lead.source_quality_score_hint or 0) >= 8


def _failed_lead_history_key_candidates(
    company_name: str | None,
    role_title: str | None,
    *,
    source_url: str | None = None,
    direct_job_url: str | None = None,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for url in (direct_job_url, source_url):
        for candidate in sorted(_job_history_key_candidates(url)):
            key = f"url:{candidate}"
            if key not in seen:
                seen.add(key)
                candidates.append(key)
    normalized_role_key = ""
    if str(company_name or "").strip() and str(role_title or "").strip():
        normalized_role_key = _normalize_job_key(str(company_name), str(role_title))
    if normalized_role_key:
        role_key = f"role:{normalized_role_key}"
        if role_key not in seen:
            candidates.append(role_key)
    return candidates


def _record_failed_lead_history_entry(
    history: dict[str, dict[str, object]],
    failure: SearchFailure,
    *,
    generated_at: str,
) -> None:
    if failure.reason_code not in FAILED_LEAD_TRACKED_REASON_CODES:
        return
    keys = _failed_lead_history_key_candidates(
        failure.company_name,
        failure.role_title,
        source_url=failure.source_url,
        direct_job_url=failure.direct_job_url,
    )
    if not keys:
        return
    for key in keys:
        entry = dict(history.get(key) or {})
        reason_counts = {
            str(reason): int(count)
            for reason, count in dict(entry.get("recent_rejection_reasons") or {}).items()
            if str(reason).strip()
        }
        reason_counts[failure.reason_code] = reason_counts.get(failure.reason_code, 0) + 1
        source_urls = [str(item).strip() for item in entry.get("source_urls", []) if str(item).strip()]
        direct_job_urls = [str(item).strip() for item in entry.get("direct_job_urls", []) if str(item).strip()]
        if failure.source_url and failure.source_url not in source_urls:
            source_urls.append(failure.source_url)
        if failure.direct_job_url and failure.direct_job_url not in direct_job_urls:
            direct_job_urls.append(failure.direct_job_url)
        entry.update(
            {
                "lead_key": key,
                "company_name": failure.company_name,
                "role_title": failure.role_title,
                "first_seen_at": entry.get("first_seen_at") or generated_at,
                "last_seen_at": generated_at,
                "watch_count": int(entry.get("watch_count") or 0) + 1,
                "last_reason_code": failure.reason_code,
                "last_detail": failure.detail[:240],
                "source_urls": source_urls[:6],
                "direct_job_urls": direct_job_urls[:6],
                "recent_rejection_reasons": reason_counts,
            }
        )
        history[key] = entry


def _load_failed_lead_history(settings: Settings) -> dict[str, dict[str, object]]:
    history: dict[str, dict[str, object]] = {}
    for path in sorted(settings.data_dir.glob("run-*.json"), reverse=True)[:30]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        generated_at = str((payload.get("manifest") or {}).get("generated_at") or path.stem)
        diagnostics = payload.get("search_diagnostics")
        failures = diagnostics.get("failures") if isinstance(diagnostics, dict) else None
        if not isinstance(failures, list):
            continue
        for raw_failure in failures:
            if not isinstance(raw_failure, dict):
                continue
            try:
                failure = SearchFailure.model_validate(raw_failure)
            except Exception:
                continue
            _record_failed_lead_history_entry(history, failure, generated_at=generated_at)
    save_json_snapshot(settings.data_dir / FAILED_LEAD_HISTORY_FILENAME, history)
    return history


def _failed_lead_history_skip_reason(
    lead: JobLead,
    settings: Settings,
    failed_lead_history: dict[str, dict[str, object]],
) -> tuple[str, str] | None:
    matched_entry: dict[str, object] | None = None
    matched_key = ""
    for key in _failed_lead_history_key_candidates(
        lead.company_name,
        lead.role_title,
        source_url=lead.source_url,
        direct_job_url=lead.direct_job_url,
    ):
        entry = failed_lead_history.get(key)
        if not entry:
            continue
        if matched_entry is None or int(entry.get("watch_count") or 0) > int(matched_entry.get("watch_count") or 0):
            matched_entry = entry
            matched_key = key
    if matched_entry is None:
        return None
    reason_counts = {
        str(reason): int(count)
        for reason, count in dict(matched_entry.get("recent_rejection_reasons") or {}).items()
        if str(reason).strip()
    }
    current_url = lead.direct_job_url or lead.source_url
    current_company_hint = _company_hint_from_url(current_url or "") if current_url else ""
    if (
        matched_key.startswith("url:")
        and reason_counts.get("stale_posting", 0) >= 1
        and not _lead_has_override_hints_for_reason(lead, settings, "stale_posting")
    ):
        return (
            "stale_posting",
            f"Lead matched prior failed lead history ({matched_key}) with an exact stale URL and no fresher date evidence.",
        )
    for exact_url_reason_code in ("not_remote", "remote_unclear"):
        if (
            matched_key.startswith("url:")
            and reason_counts.get(exact_url_reason_code, 0) >= 1
            and not _lead_has_override_hints_for_reason(lead, settings, exact_url_reason_code)
        ):
            return (
                exact_url_reason_code,
                (
                    f"Lead matched prior failed lead history ({matched_key}) with an exact URL that already "
                    f"resolved to {exact_url_reason_code} and no stronger remote evidence."
                ),
            )
    for reason_code in FAILED_LEAD_IMMEDIATE_SUPPRESS_REASON_CODES:
        if reason_counts.get(reason_code, 0) >= 1:
            if reason_code == "company_mismatch" and _is_weak_company_hint(current_company_hint):
                continue
            return (
                reason_code,
                f"Lead matched prior failed lead history ({matched_key}) with persistent reason {reason_code}.",
            )
    for reason_code, threshold in FAILED_LEAD_REPEAT_SUPPRESS_THRESHOLDS.items():
        if reason_counts.get(reason_code, 0) >= threshold:
            if _lead_has_override_hints_for_reason(lead, settings, reason_code):
                continue
            return (
                reason_code,
                (
                    f"Lead matched prior failed lead history ({matched_key}) and had repeated "
                    f"{reason_code} outcomes ({reason_counts.get(reason_code, 0)}x)."
                ),
            )
    return None


def _strip_tracking_query_params(query: str) -> str:
    if not query:
        return ""
    kept_params = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        lowered_key = key.lower()
        if lowered_key.startswith("utm_") or lowered_key in TRACKING_QUERY_PARAM_NAMES:
            continue
        kept_params.append((key, value))
    return urlencode(kept_params, doseq=True)


def _describe_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _path_segments(path: str) -> list[str]:
    return [segment for segment in path.strip("/").split("/") if segment]


def _has_company_job_detail_signal(parsed_url) -> bool:
    full = f"{(parsed_url.netloc or '').lower()}{(parsed_url.path or '').lower()}?{(parsed_url.query or '').lower()}"
    if any(hint in full for hint in GENERIC_CAREERS_INFO_HINTS):
        return False
    query = (parsed_url.query or "").lower()
    if any(token in query for token in COMPANY_JOB_QUERY_HINTS):
        return True

    segments = _path_segments((parsed_url.path or "").lower())
    if not segments:
        return False
    if len(segments) <= 2 and segments[-1] in GENERIC_CAREERS_TAIL_SEGMENTS:
        return False
    if (
        segments[-1] in GENERIC_CAREERS_TAIL_SEGMENTS
        and len(segments) <= 3
        and any(segment in GENERIC_COMPANY_CAREERS_SEGMENTS for segment in segments[:-1])
    ):
        return False
    if len(segments) >= 2 and all(segment in GENERIC_CAREERS_TAIL_SEGMENTS for segment in segments[-2:]):
        return False

    last_segment = segments[-1]
    if last_segment not in GENERIC_COMPANY_CAREERS_SEGMENTS:
        return any(segment in GENERIC_COMPANY_CAREERS_SEGMENTS for segment in segments[:-1]) or len(segments) >= 2
    return False


def _looks_like_direct_ats_job_path(host: str, path: str) -> bool:
    normalized_path = (path or "").rstrip("/").lower()
    segments = _path_segments(normalized_path)

    if "jobs.lever.co" in host:
        return len(segments) >= 2 and segments[0] != "job-seeker-support"

    if "ashbyhq.com" in host:
        return len(segments) >= 2 and segments[1] not in {"jobs", "job-board", "careers"}

    if "greenhouse.io" in host:
        return "jobs" in segments and len(segments) >= 3

    if "smartrecruiters.com" in host:
        if len(segments) < 2:
            return False
        posting_slug = segments[1]
        return bool(re.search(r"\d{6,}", posting_slug))

    if "ats.rippling.com" in host:
        return len(segments) >= 3 and segments[1] == "jobs"

    if "recruitee.com" in host or "careers.tellent.com" in host:
        return len(segments) >= 2 and segments[0] == "o"

    if "comeet.com" in host:
        return "/jobs/" in f"{normalized_path}/"

    if "jobscore.com" in host:
        return "/job/" in f"{normalized_path}/" or "/jobs/" in f"{normalized_path}/"

    if "myworkdayjobs.com" in host:
        return "/job/" in f"{normalized_path}/"

    if "jobvite.com" in host:
        return "/job/" in f"{normalized_path}/"

    if "workable.com" in host:
        return "/j/" in f"{normalized_path}/" or "/jobs/" in f"{normalized_path}/"

    if "icims.com" in host:
        return "/jobs/" in f"{normalized_path}/"

    if "portal.dynamicsats.com" in host:
        return "/joblisting/details/" in f"{normalized_path}/"

    if any(fragment in host for fragment in ("bamboohr.com", "dayforcehcm.com", "paylocity.com", "adp.com", "ultipro.com")):
        return any(hint in f"{normalized_path}/" for hint in JOB_PATH_HINTS)

    return False


def _looks_like_generic_job_url(url: str) -> bool:
    raw_lower = url.lower()
    raw_query = (urlparse(url).query or "").lower()
    if "error=true" in raw_query or "error=true" in raw_lower:
        return True
    if any(hint in raw_lower for hint in GENERIC_CAREERS_INFO_HINTS):
        return True

    normalized_url = _normalize_direct_job_url(url)
    parsed = urlparse(normalized_url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/").lower()

    if "lever.co" in host and "job-seeker-support" in path:
        return True
    if "greenhouse.io" in host and "embed/job_board" in path:
        return True

    if any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
        return not _looks_like_direct_ats_job_path(host, path)

    if host.startswith(("careers.", "jobs.", "apply.", "join.", "workat.", "recruiting.")):
        if not _has_company_job_detail_signal(parsed):
            return True
    if any(hint in f"{path}/" for hint in JOB_PATH_HINTS) and not _has_company_job_detail_signal(parsed):
        return True
    return path in {"", "/", "/jobs", "/job", "/careers", "/career", "/openings", "/positions", "/global/en"}


def _is_allowed_direct_job_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if any(fragment in host for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
        return False
    if any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
        return True
    return _looks_like_company_job_page(url)


def _looks_like_careers_hub_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/").lower()
    full = f"{host}{path}?{(parsed.query or '').lower()}"
    if not host or any(fragment in host for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
        return False
    if any(fragment in host for fragment in COMPANY_SITE_JUNK_HOST_FRAGMENTS):
        return False
    if any(hint in full for hint in GENERIC_CAREERS_INFO_HINTS):
        return False
    if any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
        return _looks_like_generic_job_url(url)
    if host.startswith(("careers.", "jobs.", "apply.", "join.", "workat.", "recruiting.")):
        return not _has_company_job_detail_signal(parsed)
    return any(hint in f"{path}/" for hint in CAREERS_HUB_HINTS) and not _has_company_job_detail_signal(parsed)


def _looks_like_company_homepage_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/").lower()
    if not host or any(fragment in host for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
        return False
    if any(fragment in host for fragment in COMPANY_SITE_JUNK_HOST_FRAGMENTS):
        return False
    if any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
        return False
    return path in {"", "/", "/company", "/about", "/about-us"}


def _today_for_timezone(timezone_name: str | None) -> date:
    if timezone_name:
        try:
            return datetime.now(ZoneInfo(timezone_name)).date()
        except ZoneInfoNotFoundError:
            pass
    return datetime.now(UTC).date()


def _parse_absolute_posted_date_text(posted_date_text: str | None) -> date | None:
    if not posted_date_text:
        return None
    normalized = posted_date_text.strip()
    if not normalized:
        return None

    iso_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
    if iso_match:
        try:
            return date.fromisoformat(iso_match.group(0))
        except ValueError:
            return None

    month_match = re.search(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
        r"january|february|march|april|june|july|august|september|october|november|december)"
        r"\s+\d{1,2},?\s+\d{4}\b",
        normalized,
        re.I,
    )
    if not month_match:
        return None

    for date_format in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(month_match.group(0), date_format).date()
        except ValueError:
            continue
    return None


def _is_recent_enough(
    posted_date_iso: str | None,
    posted_date_text: str,
    max_age_days: int,
    *,
    timezone_name: str | None = None,
) -> bool:
    today = _today_for_timezone(timezone_name)
    if posted_date_iso:
        try:
            posted_on = date.fromisoformat(posted_date_iso)
            return posted_on >= today - timedelta(days=max_age_days)
        except ValueError:
            pass

    absolute_posted_on = _parse_absolute_posted_date_text(posted_date_text)
    if absolute_posted_on is not None:
        return absolute_posted_on >= today - timedelta(days=max_age_days)

    lowered = posted_date_text.lower()
    if "today" in lowered or "yesterday" in lowered:
        return True
    if "this week" in lowered:
        return True

    day_match = re.search(r"(\d+)\s*(day|days)\b", lowered)
    if day_match:
        return int(day_match.group(1)) <= max_age_days

    week_match = re.search(r"(\d+)\s*(week|weeks|wk|wks)\b", lowered)
    if week_match:
        return int(week_match.group(1)) * 7 <= max_age_days

    compact_week_match = re.search(r"(\d+)\s*w\b", lowered)
    if compact_week_match:
        return int(compact_week_match.group(1)) * 7 <= max_age_days

    return False


SENIOR_TITLE_TOKENS = DEFAULT_ROLE_SEARCH_PROFILE.senior_title_tokens
TITLE_ONLY_SALARY_INFERENCE_TOKENS = DEFAULT_ROLE_SEARCH_PROFILE.title_only_salary_inference_tokens
AI_SIGNAL_TOKENS = DEFAULT_ROLE_SEARCH_PROFILE.ai_signal_tokens
AI_STRONG_CONTEXT_TOKENS = DEFAULT_ROLE_SEARCH_PROFILE.ai_strong_context_tokens
AI_OWNERSHIP_TOKENS = DEFAULT_ROLE_SEARCH_PROFILE.ai_ownership_tokens
AI_LOW_SIGNAL_PATTERNS = DEFAULT_ROLE_SEARCH_PROFILE.ai_low_signal_patterns
NON_US_MARKET_HINT_PATTERNS = DEFAULT_ROLE_SEARCH_PROFILE.non_us_market_hint_patterns


def _contains_ai_signal(text: str) -> bool:
    lowered = text.lower()
    return any(
        (
            token in lowered
            if " " in token or "/" in token or "-" in token
            else re.search(rf"\b{re.escape(token)}\b", lowered)
        )
        for token in AI_SIGNAL_TOKENS
    )


def _sentence_split(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part for part in parts if part] or [normalized]


def _is_low_signal_ai_sentence(text: str) -> bool:
    lowered = text.lower()
    if any(pattern in lowered for pattern in AI_LOW_SIGNAL_PATTERNS):
        return True
    return bool(re.search(r"\b(?:is|are)\s+an?\s+ai[- ]powered\b", lowered))


def _has_strong_ai_context(text: str) -> bool:
    for sentence in _sentence_split(text):
        lowered = sentence.lower()
        if not _contains_ai_signal(lowered):
            continue
        if _is_low_signal_ai_sentence(lowered):
            continue
        has_strong_scope = any(token in lowered for token in AI_STRONG_CONTEXT_TOKENS)
        has_ownership = any(token in lowered for token in AI_OWNERSHIP_TOKENS)
        if has_strong_scope or has_ownership:
            return True
    return False


def _is_ai_related_product_manager_text(text: str) -> bool:
    normalized = " ".join(text.split())
    lowered = normalized.lower()
    product_ok = "product manager" in lowered or (re.search(r"\bproduct\b", lowered) and re.search(r"\bmanager\b", lowered))
    if not product_ok:
        return False
    if len(normalized) <= 160 and _contains_ai_signal(normalized) and not _is_low_signal_ai_sentence(normalized):
        return True
    return _has_strong_ai_context(normalized)


def _is_ai_related_product_manager(job: JobPosting) -> bool:
    for primary_text in (job.role_title, job.job_page_title or ""):
        if primary_text and _is_ai_related_product_manager_text(primary_text):
            return True
    haystack = " ".join(
        part
        for part in (
            job.role_title,
            job.job_page_title or "",
            " ".join(job.validation_evidence),
        )
        if part
    )
    return _is_ai_related_product_manager_text(haystack)


def _lead_is_ai_related_product_manager(lead: JobLead) -> bool:
    if _is_ai_related_product_manager_text(lead.role_title):
        return True
    if "product manager" not in lead.role_title.lower():
        return False
    strong_evidence_source = lead.source_type in {"builtin", "direct_ats", "company_site"} or bool(lead.direct_job_url)
    if not strong_evidence_source:
        return False
    haystack = " ".join(
        part
        for part in (
            lead.role_title,
            lead.evidence_notes,
            lead.location_hint or "",
        )
        if part
    )
    return _is_ai_related_product_manager_text(haystack)


def _company_token_candidates(company_name: str) -> list[str]:
    tokens = [token for token in re.findall(r"[a-z0-9]+", company_name.lower()) if len(token) >= 3]
    compact = "".join(ch for ch in company_name.lower() if ch.isalnum())
    if len(compact) >= 4:
        tokens.append(compact)
    return list(dict.fromkeys(tokens))


def _role_token_candidates(role_title: str) -> list[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "into",
        "from",
        "at",
        "of",
        "us",
    }
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", role_title.lower().replace("ai/ml", "ai ml").replace("data/ml", "data ml"))
        if len(token) >= 2 and token not in stopwords
    ]
    return list(dict.fromkeys(tokens))


def _role_match_score(role_title: str, candidate_text: str) -> int:
    haystack = candidate_text.lower().replace("ai/ml", "ai ml").replace("data/ml", "data ml")
    tokens = _role_token_candidates(role_title)
    if not tokens:
        return 0
    matches = 0
    for token in tokens:
        if re.search(rf"\b{re.escape(token)}\b", haystack):
            matches += 1

    score = matches * 2
    if "product manager" in role_title.lower():
        if "product manager" in haystack:
            score += 8
        elif re.search(r"\bproduct\b", haystack) and re.search(r"\bmanager\b", haystack):
            score += 4
        else:
            score -= 8
    if "ai" in role_title.lower() and not re.search(r"\bai\b", haystack) and "machine learning" not in haystack and not re.search(r"\bml\b", haystack):
        score -= 2
    if "ml" in role_title.lower() and not re.search(r"\bml\b", haystack) and "machine learning" not in haystack:
        score -= 2
    return score


def _role_titles_align(expected_role_title: str | None, observed_role_title: str | None) -> bool:
    expected = " ".join(str(expected_role_title or "").split())
    observed = " ".join(str(observed_role_title or "").split())
    if not expected or not observed:
        return False
    if expected.lower() == observed.lower():
        return True

    expected_is_ai_pm = _is_ai_related_product_manager_text(expected)
    observed_is_ai_pm = _is_ai_related_product_manager_text(observed)
    if expected_is_ai_pm and not observed_is_ai_pm:
        return False

    forward_score = _role_match_score(expected, observed)
    reverse_score = _role_match_score(observed, expected)
    if min(forward_score, reverse_score) >= 14:
        return True

    expected_tokens = set(_role_token_candidates(expected))
    observed_tokens = set(_role_token_candidates(observed))
    overlap = expected_tokens.intersection(observed_tokens)
    return len(overlap) >= 4 and "product" in overlap and "manager" in overlap


def _extract_experience_years_floor(text: str) -> int | None:
    if not text:
        return None
    floors: list[int] = []

    for match in re.finditer(r"\b(?P<years>\d{1,2})\s*\+\s*(?:years?|yrs?)\b", text, re.I):
        floors.append(int(match.group("years")))
    for match in re.finditer(
        r"\b(?:at\s+least|minimum(?:\s+of)?|min\.?|over)\s*(?P<years>\d{1,2})\s*(?:years?|yrs?)\b",
        text,
        re.I,
    ):
        floors.append(int(match.group("years")))
    for match in re.finditer(
        r"\b(?P<min>\d{1,2})\s*(?:-|to|–|—)\s*(?P<max>\d{1,2})\s*(?:years?|yrs?)\b",
        text,
        re.I,
    ):
        floors.append(int(match.group("min")))
    for match in re.finditer(
        r"\b(?P<years>\d{1,2})\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)\b",
        text,
        re.I,
    ):
        floors.append(int(match.group("years")))

    if not floors:
        return None
    return max(floors)


def _has_senior_title_signal(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in SENIOR_TITLE_TOKENS)


def _seniority_signal_score(*parts: str | None) -> int:
    text = " ".join(part for part in parts if part)
    if not text:
        return 0
    score = 0
    years_floor = _extract_experience_years_floor(text)
    if years_floor is not None:
        if years_floor >= 9:
            score += 2
        elif years_floor >= 7:
            score += 1
    if _has_senior_title_signal(text):
            score += 1
    return score


def _is_principal_ai_pm_title_text(text: str | None) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized or "principal" not in normalized:
        return False
    if "product manager" not in normalized and "technical product manager" not in normalized:
        return False
    return any(
        token in normalized
        for token in (
            " ai",
            "ai ",
            "machine learning",
            "ml ",
            " llm",
            "llm ",
            "genai",
            "generative ai",
            "agentic",
            "applied ai",
            "ai/ml",
            "ai platform",
            "ml platform",
        )
    )


def _job_looks_us_remote_without_geo_limit(job: JobPosting, snapshot: JobPageSnapshot) -> bool:
    if job.is_fully_remote is not True:
        return False
    if _job_has_geo_limited_remote_restriction(job, snapshot):
        return False
    context = " ".join(
        part
        for part in (
            job.location_text,
            snapshot.location_text,
            snapshot.text_excerpt,
            job.evidence_notes,
        )
        if part
    ).lower()
    has_non_us_market_hint = any(
        re.search(rf"\b{re.escape(pattern)}\b", context) for pattern in NON_US_MARKET_HINT_PATTERNS
    )
    has_us_market_hint = bool(
        re.search(
            r"\b(united states|u\.s\.|usa|us-based|us remote|remote\s*[-,]?\s*(?:us|usa|united states)|within the united states)\b",
            context,
        )
    )
    if re.search(
        r"\b(?:within|in|for|across)?\s*(?:the )?(?:united states|u\.s\.|usa|us)\s+only\b",
        context,
    ):
        return False
    return not has_non_us_market_hint or has_us_market_hint


def _job_supports_principal_ai_pm_salary_presumption(
    job: JobPosting,
    snapshot: JobPageSnapshot,
    settings: Settings,
) -> bool:
    if not settings.enable_principal_ai_pm_salary_presumption:
        return False
    title_context = " ".join(
        part
        for part in (
            job.role_title,
            job.job_page_title,
            snapshot.role_title,
            snapshot.page_title,
        )
        if part
    )
    if not _is_principal_ai_pm_title_text(title_context):
        return False
    if not _is_ai_related_product_manager(job):
        return False
    if not _job_looks_us_remote_without_geo_limit(job, snapshot):
        return False
    job_url = str(job.resolved_job_url or job.direct_job_url)
    if not _is_allowed_direct_job_url(job_url):
        return False
    if _lead_source_quality_score(
        JobLead(
            company_name=job.company_name,
            role_title=job.role_title,
            source_url=job_url,
            source_type="direct_ats" if any(fragment in job_url for fragment in ALLOWED_JOB_HOST_FRAGMENTS) else "company_site",
            direct_job_url=job_url,
            location_hint=job.location_text,
            posted_date_hint=job.posted_date_text,
            is_remote_hint=job.is_fully_remote,
            base_salary_min_usd_hint=job.base_salary_min_usd,
            base_salary_max_usd_hint=job.base_salary_max_usd,
            salary_text_hint=job.salary_text,
            evidence_notes=job.evidence_notes,
            source_query=job.source_query,
            source_quality_score_hint=job.source_quality_score,
        ),
        settings,
    ) < 7:
        return False
    if not _is_recent_enough(
        job.posted_date_iso,
        job.posted_date_text or "",
        settings.posted_within_days,
        timezone_name=settings.timezone,
    ):
        return False
    return True


def _infer_salary_from_experience(
    job: JobPosting,
    snapshot: JobPageSnapshot,
    settings: Settings,
) -> tuple[bool, str | None, int | None, str | None]:
    if _salary_values(job):
        return False, None, None, None
    years_floor = _extract_experience_years_floor(
        " ".join(
            part
            for part in (
                job.role_title,
                job.job_page_title or "",
                job.location_text,
                job.evidence_notes,
                snapshot.location_text,
                snapshot.text_excerpt,
            )
            if part
        )
    )
    if job.is_fully_remote is True and _job_has_geo_limited_remote_restriction(job, snapshot):
        return False, None, years_floor, None
    if _job_supports_principal_ai_pm_salary_presumption(job, snapshot, settings):
        reason = (
            f"Presumed likely >= ${settings.min_base_salary_usd:,} base because this is a fresh, US-remote principal AI product role on an official source."
        )
        return True, reason, years_floor, "salary_presumed_from_principal_ai_pm"
    context_text = " ".join(
        part
        for part in (
            job.role_title,
            job.job_page_title or "",
            job.location_text,
            job.evidence_notes,
            snapshot.location_text,
            snapshot.text_excerpt,
        )
        if part
    )
    lowered_context = context_text.lower()
    has_us_market_hint = bool(
        re.search(
            r"\b(united states|u\.s\.|usa|us-based|remote\s*[-,]?\s*(?:us|usa|united states)|within the united states)\b",
            lowered_context,
        )
    )
    has_non_us_market_hint = any(
        re.search(rf"\b{re.escape(pattern)}\b", lowered_context) for pattern in NON_US_MARKET_HINT_PATTERNS
    )
    title_context = " ".join(
        part
        for part in (
            job.role_title,
            job.job_page_title or "",
            snapshot.role_title or "",
            snapshot.page_title or "",
        )
        if part
    ).lower()
    if years_floor is None or years_floor < 7:
        has_high_salary_title_signal = any(token in title_context for token in TITLE_ONLY_SALARY_INFERENCE_TOKENS)
        if not (has_us_market_hint and has_high_salary_title_signal):
            return False, None, years_floor, None
        reason = (
            f"Inferred likely >= ${settings.min_base_salary_usd:,} base because this appears to be a US-remote "
            f"{job.role_title} role, which is typically compensated at this level."
        )
        return True, reason, years_floor, "experience_title_market_inference"
    if has_non_us_market_hint and not has_us_market_hint:
        return False, None, years_floor, None
    reason = (
        f"Inferred likely >= ${settings.min_base_salary_usd:,} base because the posting appears to require "
        f"at least {years_floor} years of experience."
    )
    return True, reason, years_floor, "experience_years_inference"


def _apply_salary_inference(job: JobPosting, snapshot: JobPageSnapshot, settings: Settings) -> JobPosting:
    inferred_ok, inference_reason, years_floor, inference_kind = _infer_salary_from_experience(job, snapshot, settings)
    if not inferred_ok:
        return job
    merged_notes = " ".join(part for part in (job.evidence_notes, inference_reason) if part).strip()
    inferred_salary_text = job.salary_text or inference_reason
    return job.model_copy(
        update={
            "salary_inferred": True,
            "salary_inference_reason": inference_reason,
            "salary_inference_kind": inference_kind,
            "inferred_experience_years_min": years_floor,
            "salary_text": inferred_salary_text,
            "evidence_notes": merged_notes,
        }
    )


def _salary_values(job: JobPosting) -> list[int]:
    return [value for value in (job.base_salary_min_usd, job.base_salary_max_usd) if value is not None]


def _salary_meets_minimum(job: JobPosting, settings: Settings) -> bool:
    salary_values = _salary_values(job)
    if salary_values:
        return max(salary_values) >= settings.min_base_salary_usd
    return bool(job.salary_inferred)


def _matches_filters(job: JobPosting, settings: Settings) -> bool:
    salary_ok = _salary_meets_minimum(job, settings)
    job_url = str(job.resolved_job_url or job.direct_job_url)
    return (
        _is_allowed_direct_job_url(job_url)
        and job.is_fully_remote
        and salary_ok
        and _is_recent_enough(
            job.posted_date_iso,
            job.posted_date_text,
            settings.posted_within_days,
            timezone_name=settings.timezone,
        )
        and _is_ai_related_product_manager(job)
    )


def _normalize_job_key(company_name: str, role_title: str) -> str:
    normalized_company = "".join(ch for ch in company_name.lower() if ch.isalnum())
    normalized_role = "".join(ch for ch in role_title.lower() if ch.isalnum())
    return f"{normalized_company}:{normalized_role}"


def _company_names_match(expected: str, observed: str) -> bool:
    def acronym_candidates(value: str) -> set[str]:
        candidates: set[str] = set()
        caps = "".join(ch for ch in value if ch.isupper())
        if len(caps) >= 2:
            candidates.add(caps)
        camel_parts = re.findall(r"[A-Z][a-z0-9]*", value)
        if len(camel_parts) >= 2:
            candidates.add("".join(part[0] for part in camel_parts).upper())
        word_parts = re.findall(r"[A-Za-z][A-Za-z0-9]*", value)
        if len(word_parts) >= 2:
            candidates.add("".join(part[0] for part in word_parts).upper())
        compact = "".join(ch for ch in value if ch.isalnum())
        if 2 <= len(compact) <= 5 and compact.isalpha():
            candidates.add(compact.upper())
        return {candidate for candidate in candidates if len(candidate) >= 2}

    expected_key = "".join(ch for ch in expected.lower() if ch.isalnum())
    observed_key = "".join(ch for ch in observed.lower() if ch.isalnum())
    if not expected_key or not observed_key:
        return False
    if expected_key == observed_key:
        return True
    if expected_key in observed_key or observed_key in expected_key:
        return True
    expected_acronyms = acronym_candidates(expected)
    observed_acronyms = acronym_candidates(observed)
    if expected_acronyms.intersection(observed_acronyms):
        return True
    for left in expected_acronyms:
        for right in observed_acronyms:
            if left.startswith(right) or right.startswith(left):
                return True
    expected_tokens = {token for token in re.findall(r"[a-z0-9]+", expected.lower()) if len(token) >= 3}
    observed_tokens = {token for token in re.findall(r"[a-z0-9]+", observed.lower()) if len(token) >= 3}
    return bool(expected_tokens and observed_tokens and expected_tokens.intersection(observed_tokens))


def _is_weak_company_hint(value: str | None) -> bool:
    if not value:
        return True
    compact = "".join(ch for ch in value.lower() if ch.isalnum())
    if not compact:
        return True
    weak_tokens = {
        "enus",
        "ext",
        "external",
        "company",
        "content",
        "corporate",
        "corporateinformation",
        "jobs",
        "job",
        "careers",
        "career",
        "apply",
        "wl",
        "wlcareers",
        "hs",
    }
    if compact in weak_tokens or len(compact) <= 3:
        return True
    tokens = [token for token in re.findall(r"[a-z0-9]+", value.lower()) if token]
    return bool(tokens) and all(token in weak_tokens or len(token) <= 2 for token in tokens)


def _normalize_source_type(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    if "linkedin.com" in host:
        return "linkedin"
    if "builtin" in host or "builtinchicago" in host:
        return "builtin"
    if "glassdoor.com" in host:
        return "glassdoor"
    if _is_allowed_direct_job_url(url):
        return "direct_ats" if any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS) else "company_site"
    return "other"


def _is_supported_discovery_source_url(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if "linkedin.com" in host:
        return "/jobs/view" in path
    if "indeed.com" in host:
        return "/viewjob" in path or "/rc/clk" in path
    if "glassdoor.com" in host:
        return "/job/" in path or "joblisting.htm" in path
    if "builtin" in host:
        return "/job/" in path
    if "wellfound.com" in host:
        return "/jobs/" in path
    if "workatastartup.com" in host:
        return "/jobs/" in path
    if "getro.com" in host:
        return "/companies/" in path and "/jobs/" in path
    if "ziprecruiter.com" in host:
        return "/jobs/" in path
    if "themuse.com" in host:
        return "/jobs/" in path
    if "monster.com" in host:
        return "/job-openings/" in path or "/jobs/search/" not in path and "/jobs/" in path
    if "startup.jobs" in host:
        return bool(re.search(r"/[^/]+/[^/]+", path))
    if "jobgether.com" in host:
        return "/offer/" in path or "/jobs/" in path
    if "welcometothejungle.com" in host:
        return "/jobs" in path
    if "joinhandshake.com" in host:
        return "/jobs/" in path
    if "ycombinator.com" in host:
        return "/companies/" in path and "/jobs/" in path
    if "dynamitejobs.com" in host:
        return "/company/" in path and "/remote-job/" in path
    if "dailyremote.com" in host:
        return "/remote-job/" in path or "/job/" in path
    if "remote.io" in host:
        return "/remote-product-jobs/" in path or "/remote-jobs/" in path
    if "flexhired.com" in host:
        return "/jobs/" in path
    if "remoteai.io" in host:
        return "/roles/" in path
    return False


def _is_low_trust_replay_source_url(url: str | None) -> bool:
    if not url:
        return False
    host = (urlparse(url).netloc or "").lower()
    return any(fragment in host for fragment in LOW_TRUST_REPLAY_SOURCE_HOST_FRAGMENTS)


def _lead_is_replay_source_trustworthy(lead: JobLead) -> bool:
    if _is_low_trust_replay_source_url(lead.source_url):
        return False
    if lead.direct_job_url and _is_low_trust_replay_source_url(lead.direct_job_url):
        return False
    if _normalize_company_key(lead.company_name) == "jobgether":
        return False
    return True


def _hint_is_recent(posted_date_hint: str | None, settings: Settings) -> bool | None:
    if not posted_date_hint:
        return None
    lowered = posted_date_hint.lower()
    if "day" in lowered or "today" in lowered or "yesterday" in lowered or "week" in lowered:
        return _is_recent_enough(
            None,
            posted_date_hint,
            settings.posted_within_days,
            timezone_name=settings.timezone,
        )
    posted_on = _parse_absolute_posted_date_text(posted_date_hint)
    if posted_on is None:
        return None
    return posted_on >= _today_for_timezone(settings.timezone) - timedelta(days=settings.posted_within_days)


def _small_company_host_priority(hosts: list[str]) -> int:
    normalized_hosts = [host.lower() for host in hosts if host]
    if any(any(fragment in host for fragment in SMALL_COMPANY_SCOUT_DOMAINS) for host in normalized_hosts):
        return 0
    if any(any(fragment in host for fragment in ENTERPRISE_ATS_HOST_FRAGMENTS) for host in normalized_hosts):
        return 2
    return 1


def _lead_has_portfolio_board_source(lead: JobLead) -> bool:
    source_url = (lead.source_url or "").lower()
    return any(domain in source_url for domain in PORTFOLIO_BOARD_SCOUT_DOMAINS)


def _focus_company_host_supports_search_queries(host: str | None) -> bool:
    normalized_host = str(host or "").strip().lower()
    if not normalized_host:
        return False
    return not any(fragment in normalized_host for fragment in FOCUS_COMPANY_DISCOVERY_ONLY_HOST_FRAGMENTS)


def _focus_company_url_supports_search_queries(url: str | None) -> bool:
    normalized_url = _normalize_direct_job_url(str(url or "").strip())
    if not normalized_url:
        return False
    if (
        _is_allowed_direct_job_url(normalized_url)
        or _looks_like_careers_hub_url(normalized_url)
        or _looks_like_company_homepage_url(normalized_url)
    ):
        return True
    return _focus_company_host_supports_search_queries(urlparse(normalized_url).netloc or "")


def _focus_company_entry_supports_search_queries(entry: Mapping[str, object] | None) -> bool:
    if not entry:
        return True
    board_identifiers = [str(item).strip() for item in entry.get("board_identifiers") or [] if str(item).strip()]
    if board_identifiers or int(entry.get("official_board_lead_count") or 0) > 0:
        return True
    careers_roots = [str(item).strip() for item in entry.get("careers_roots") or [] if str(item).strip()]
    if any(_focus_company_url_supports_search_queries(url) for url in careers_roots):
        return True
    source_hosts = [str(item).strip().lower() for item in entry.get("source_hosts") or [] if str(item).strip()]
    if not source_hosts and not careers_roots:
        return True
    return any(_focus_company_host_supports_search_queries(host) for host in source_hosts)


def _site_hints_from_focus_company_entry(entry: Mapping[str, object] | None) -> list[str]:
    if not entry:
        return []
    hints: list[str] = []
    seen: set[str] = set()

    def _add_hint(value: str | None) -> None:
        normalized_value = " ".join(str(value or "").split())
        if not normalized_value or normalized_value in seen:
            return
        seen.add(normalized_value)
        hints.append(normalized_value)

    for hint in _site_hints_from_board_identifiers(dict(entry)):
        _add_hint(hint)

    for raw_url in entry.get("careers_roots") or []:
        normalized_url = _normalize_direct_job_url(str(raw_url or "").strip())
        if not normalized_url:
            continue
        host = (urlparse(normalized_url).netloc or "").strip().lower()
        if not _focus_company_host_supports_search_queries(host):
            continue
        _add_hint(f"site:{host}")

    for raw_host in entry.get("source_hosts") or []:
        host = str(raw_host or "").strip().lower()
        if not _focus_company_host_supports_search_queries(host):
            continue
        _add_hint(f"site:{host}")

    return hints


def _lead_has_strong_validation_hints(lead: JobLead, settings: Settings) -> bool:
    recent_hint = _hint_is_recent(lead.posted_date_hint, settings)
    salary_min, salary_max, _salary_text = _hydrate_salary_hint_values(
        lead.base_salary_min_usd_hint,
        lead.base_salary_max_usd_hint,
        lead.salary_text_hint,
        lead.evidence_notes,
    )
    salary_values = [value for value in (salary_min, salary_max) if value is not None]
    return (
        lead.is_remote_hint is True
        and recent_hint is True
        and bool(salary_values and max(salary_values) >= settings.min_base_salary_usd)
    )


def _lead_has_explicit_broad_remote_evidence(lead: JobLead) -> bool:
    combined_remote_hint = _join_remote_restriction_context(
        lead.location_hint,
        lead.evidence_notes,
        _unwrap_direct_job_url(lead.direct_job_url or ""),
    ).lower()
    explicit_remote_markers = (
        *BROAD_REMOTE_OVERRIDE_MARKERS,
        "location: remote",
        "this role is remote",
        "this position is remote",
        "remote position",
        "remote role",
    )
    return any(marker in combined_remote_hint for marker in explicit_remote_markers)


def _should_skip_low_trust_direct_remote_lead(lead: JobLead) -> bool:
    if lead.source_type not in {"builtin", "linkedin", "other"}:
        return False
    if not lead.direct_job_url or lead.is_remote_hint is not True:
        return False
    combined_remote_hint = _join_remote_restriction_context(
        lead.location_hint,
        lead.evidence_notes,
        _unwrap_direct_job_url(lead.direct_job_url or ""),
    )
    if _extract_geo_limited_remote_region(combined_remote_hint):
        return True
    return not _lead_has_explicit_broad_remote_evidence(lead)


def _lead_host_is_js_blank_prone(lead: JobLead) -> bool:
    host = (urlparse(lead.direct_job_url or lead.source_url).netloc or "").lower()
    return any(fragment in host for fragment in ("myworkdayjobs.com", "careers.workday.com", "getro.com"))


def _lead_priority(lead: JobLead, settings: Settings) -> tuple[int, int, int, int, int, int, int, str, str]:
    direct_bonus = 0 if lead.direct_job_url else 1
    source_priority = DISCOVERY_SOURCE_PRIORITY.get(lead.source_type, 9)
    if source_priority >= DISCOVERY_SOURCE_PRIORITY["other"] and _is_supported_discovery_source_url(lead.source_url):
        source_priority = DISCOVERY_SOURCE_PRIORITY["glassdoor"]

    recent_hint = _hint_is_recent(lead.posted_date_hint, settings)
    if recent_hint is True:
        recency_penalty = 0
    elif recent_hint is None:
        recency_penalty = 1
    else:
        recency_penalty = 3

    salary_values = [value for value in (lead.base_salary_min_usd_hint, lead.base_salary_max_usd_hint) if value is not None]
    seniority_score = _seniority_signal_score(
        lead.role_title,
        lead.evidence_notes,
        lead.salary_text_hint,
    )
    if salary_values and max(salary_values) >= settings.min_base_salary_usd:
        salary_penalty = 0
    elif salary_values:
        salary_penalty = 4
    else:
        if seniority_score >= 2:
            salary_penalty = 0
        elif seniority_score == 1:
            salary_penalty = 1
        else:
            salary_penalty = 2

    seniority_penalty = 0 if seniority_score > 0 else 1
    remote_penalty = 0 if lead.is_remote_hint else 1
    focus_penalty = 0 if lead.source_type in {"direct_ats", "company_site"} else 1
    structured_penalty = 0 if _lead_has_strong_validation_hints(lead, settings) else 1
    portfolio_penalty = 0 if _lead_has_portfolio_board_source(lead) else 1
    js_blank_penalty = 2 if _lead_host_is_js_blank_prone(lead) and not _lead_has_trusted_source_fallback_evidence(lead, settings) else 0
    host_candidates = [
        urlparse(lead.direct_job_url or "").netloc.lower(),
        urlparse(lead.source_url).netloc.lower(),
    ]
    company_scale_penalty = _small_company_host_priority(host_candidates)
    return (
        recency_penalty,
        salary_penalty,
        structured_penalty,
        seniority_penalty,
        remote_penalty,
        source_priority,
        company_scale_penalty,
        direct_bonus + focus_penalty + portfolio_penalty + js_blank_penalty,
        lead.company_name.lower(),
        lead.role_title.lower(),
    )


def _role_title_is_low_signal(role_title: str, evidence_notes: str = "") -> bool:
    lowered = " ".join(part for part in (role_title, evidence_notes) if part).lower()
    if not lowered.strip():
        return True
    if "product manager" not in lowered:
        return True
    if "strategy" in lowered or "strategist" in lowered:
        return True
    return any(term in lowered for term in LOW_SIGNAL_ROLE_TERMS if term not in {"strategy", "strategist"})


def _lead_source_quality_score(
    lead: JobLead,
    settings: Settings,
    watchlist_entry: dict[str, object] | None = None,
) -> int:
    score = 0
    if lead.direct_job_url:
        score += 4
    if lead.source_type in {"direct_ats", "company_site"}:
        score += 4
    elif lead.source_type == "builtin":
        score += 2
    elif lead.source_type == "linkedin":
        score += 1
    if _is_supported_discovery_source_url(lead.source_url):
        score += 2
    if _lead_has_portfolio_board_source(lead):
        score += 3
    host_candidates = [
        urlparse(lead.direct_job_url or "").netloc.lower(),
        urlparse(lead.source_url).netloc.lower(),
    ]
    if _company_looks_small_from_hosts(host_candidates):
        score += 3
    recent_hint = _hint_is_recent(lead.posted_date_hint, settings)
    if recent_hint is True:
        score += 2
    elif recent_hint is False:
        score -= 2
    if lead.is_remote_hint is True:
        score += 2
    elif lead.is_remote_hint is False:
        score -= 3
    if lead.base_salary_min_usd_hint or lead.base_salary_max_usd_hint or lead.salary_text_hint:
        score += 2
    if _lead_has_strong_validation_hints(lead, settings):
        score += 3
    if _lead_host_is_js_blank_prone(lead) and not _lead_has_trusted_source_fallback_evidence(lead, settings):
        score -= 3
    if _role_title_is_low_signal(lead.role_title, lead.evidence_notes):
        score -= 5
    if watchlist_entry:
        recycle_penalty, saturation_penalty = _watchlist_focus_penalties(watchlist_entry)
        score -= min(6, recycle_penalty // 2)
        score -= min(4, saturation_penalty // 3)
    return score


def _annotate_and_filter_resolution_leads(
    leads: list[JobLead],
    settings: Settings,
    company_watchlist: dict[str, dict[str, object]],
) -> list[JobLead]:
    annotated: list[JobLead] = []
    for lead in leads:
        watchlist_entry = company_watchlist.get(_normalize_company_key(lead.company_name), {})
        if _watchlist_entry_should_skip_resolution(lead, settings, watchlist_entry):
            continue
        if _should_skip_low_trust_direct_remote_lead(lead):
            continue
        source_quality_score = _lead_source_quality_score(lead, settings, watchlist_entry)
        if source_quality_score <= 0:
            continue
        if source_quality_score <= 2 and _role_title_is_low_signal(lead.role_title, lead.evidence_notes):
            continue
        annotated.append(lead.model_copy(update={"source_quality_score_hint": source_quality_score}))
    annotated.sort(
        key=lambda lead: (
            -(lead.source_quality_score_hint or 0),
            *_lead_priority(lead, settings),
        )
    )
    capped: list[JobLead] = []
    low_trust_direct_count = 0
    for lead in annotated:
        if lead.source_type in {"builtin", "linkedin", "other"} and lead.direct_job_url:
            if low_trust_direct_count >= MAX_LOW_TRUST_DIRECT_RESOLUTION_LEADS:
                continue
            low_trust_direct_count += 1
        capped.append(lead)
    return capped


def _normalize_company_key(company_name: str | None) -> str:
    return "".join(character.lower() for character in str(company_name or "") if character.isalnum())


def _focus_company_name_is_timeout_safe(company_name: str | None) -> bool:
    company_key = _normalize_company_key(company_name)
    if not company_key:
        return False
    return not company_key.startswith("builtin")


def _sanitize_focus_companies(companies: list[str] | None) -> list[str]:
    sanitized: list[str] = []
    seen: set[str] = set()
    for company in companies or []:
        normalized_company = re.sub(r"\s+", " ", str(company or "").strip())
        company_key = _normalize_company_key(normalized_company)
        if not normalized_company or company_key in seen or not _focus_company_name_is_timeout_safe(normalized_company):
            continue
        seen.add(company_key)
        sanitized.append(normalized_company)
    return sanitized


def _company_looks_small_from_hosts(hosts: list[str]) -> bool:
    normalized_hosts = [host.lower() for host in hosts if host]
    return any(any(fragment in host for fragment in SMALL_COMPANY_SCOUT_DOMAINS) for host in normalized_hosts)


def _company_discovery_entry_saturation_score(entry: Mapping[str, object]) -> int:
    official_board_lead_count = max(0, int(entry.get("official_board_lead_count") or 0))
    recent_fresh_role_count = max(0, int(entry.get("recent_fresh_role_count") or 0))
    board_crawl_success_count = max(0, int(entry.get("board_crawl_success_count") or 0))
    return official_board_lead_count + recent_fresh_role_count + board_crawl_success_count * 4


REPORTED_BOARD_RECRAWL_SATURATION_THRESHOLD = 12


def _company_discovery_entry_seed_priority(entry: Mapping[str, object]) -> tuple[int, int, int, int, int, str]:
    source_hosts = [str(item).strip().lower() for item in entry.get("source_hosts") or [] if str(item).strip()]
    board_identifiers = [str(item).strip() for item in entry.get("board_identifiers") or [] if str(item).strip()]
    careers_roots = [str(item).strip() for item in entry.get("careers_roots") or [] if str(item).strip()]
    official_board_lead_count = max(0, int(entry.get("official_board_lead_count") or 0))
    saturation_score = _company_discovery_entry_saturation_score(entry)
    if board_identifiers and official_board_lead_count == 0:
        exploration_stage = 0
    elif careers_roots and official_board_lead_count == 0:
        exploration_stage = 1
    else:
        exploration_stage = 2
    return (
        exploration_stage,
        0 if _company_looks_small_from_hosts(source_hosts) else 1,
        saturation_score,
        -int(entry.get("source_trust") or 0),
        -len(board_identifiers),
        str(entry.get("company_name") or "").lower(),
    )


def _company_discovery_entry_frontier_priority(
    entry: Mapping[str, object],
    *,
    company_key: str,
    previously_reported_company_keys: set[str],
) -> int:
    official_board_lead_count = max(0, int(entry.get("official_board_lead_count") or 0))
    novelty_boost = 2 if company_key and company_key not in previously_reported_company_keys else 0
    exploration_boost = 2 if official_board_lead_count == 0 else 0
    saturation_penalty = min(5, _company_discovery_entry_saturation_score(entry))
    return max(4, min(12, 7 + novelty_boost + exploration_boost - saturation_penalty))


def _frontier_has_pending_novel_company_expansion(
    frontier: Sequence[Mapping[str, object]],
    *,
    previously_reported_company_keys: set[str],
) -> bool:
    for task in frontier:
        if str(task.get("status") or "pending") != "pending":
            continue
        task_type = str(task.get("task_type") or "").strip()
        if task_type in {"directory_source", "portfolio_source"}:
            return True
        if task_type not in {"company_page", "careers_root"}:
            continue
        company_key = _normalize_company_key(str(task.get("company_key") or task.get("company_name") or ""))
        if not company_key or company_key not in previously_reported_company_keys:
            return True
    return False


def _should_defer_reported_saturated_board_task(
    task: Mapping[str, object],
    *,
    entries: Mapping[str, Mapping[str, object]],
    previously_reported_company_keys: set[str],
) -> bool:
    company_key = _normalize_company_key(str(task.get("company_key") or task.get("company_name") or ""))
    if not company_key:
        return False
    if company_key not in previously_reported_company_keys:
        return False
    entry = entries.get(company_key)
    if not isinstance(entry, Mapping):
        return False
    return _company_discovery_entry_saturation_score(entry) >= REPORTED_BOARD_RECRAWL_SATURATION_THRESHOLD


def _watchlist_reason_count(entry: dict[str, object], reason_code: str) -> int:
    reasons = entry.get("recent_rejection_reasons")
    if not isinstance(reasons, dict):
        return 0
    value = reasons.get(reason_code)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _watchlist_focus_penalties(entry: dict[str, object]) -> tuple[int, int]:
    watch_count = max(0, int(entry.get("watch_count") or 0))
    reason_counts = {
        reason_code: _watchlist_reason_count(entry, reason_code) for reason_code in NON_ACTIONABLE_WATCHLIST_REASON_CODES
    }
    stale_count = reason_counts["stale_posting"]
    already_reported_count = reason_counts["already_reported"]
    not_remote_count = reason_counts["not_remote"] + reason_counts["remote_unclear"]
    hard_failure_count = max(reason_counts.values(), default=0)
    recycle_penalty = (
        min(10, stale_count // 3)
        + min(6, already_reported_count // 2)
        + min(4, not_remote_count // 4)
        + min(8, hard_failure_count // 6)
    )
    saturation_penalty = min(12, watch_count // 25)
    return recycle_penalty, saturation_penalty


def _watchlist_entry_is_focusable(entry: dict[str, object]) -> bool:
    watch_count = max(0, int(entry.get("watch_count") or 0))
    last_reason_code = str(entry.get("last_reason_code") or "")
    reason_counts = entry.get("recent_rejection_reasons")
    max_reason_count = 0
    if isinstance(reason_counts, dict):
        for value in reason_counts.values():
            try:
                max_reason_count = max(max_reason_count, int(value))
            except (TypeError, ValueError):
                continue
    if max_reason_count >= 20:
        return False
    if last_reason_code in NON_ACTIONABLE_WATCHLIST_REASON_CODES and max_reason_count >= 6:
        return False
    if watch_count >= 30 and max_reason_count >= 4:
        return False
    return True


def _lead_has_strong_override_hints(lead: JobLead, settings: Settings) -> bool:
    if not _lead_is_ai_related_product_manager(lead):
        return False
    if lead.is_remote_hint is not True:
        return False
    combined_location_hint = _join_remote_restriction_context(lead.location_hint, lead.evidence_notes)
    if _extract_geo_limited_remote_region(combined_location_hint):
        return False
    if lead.posted_date_hint and _hint_is_recent(lead.posted_date_hint, settings) is False:
        return False
    salary_min, salary_max, _salary_text = _hydrate_salary_hint_values(
        lead.base_salary_min_usd_hint,
        lead.base_salary_max_usd_hint,
        lead.salary_text_hint,
        lead.evidence_notes,
    )
    salary_values = [value for value in (salary_min, salary_max) if value is not None]
    salary_ok = not salary_values or max(salary_values) >= settings.min_base_salary_usd
    return salary_ok and (
        bool(lead.direct_job_url)
        or lead.source_type in {"direct_ats", "company_site", "builtin"}
        or bool(salary_values)
    )


def _lead_has_override_hints_for_reason(lead: JobLead, settings: Settings, reason_code: str) -> bool:
    if not _lead_has_strong_override_hints(lead, settings):
        return False
    if reason_code == "stale_posting":
        return _hint_is_recent(lead.posted_date_hint, settings) is True
    if reason_code in {"not_remote", "remote_unclear"}:
        combined_location_hint = _join_remote_restriction_context(
            lead.role_title,
            lead.location_hint,
            lead.evidence_notes,
            _unwrap_direct_job_url(lead.direct_job_url or ""),
        )
        lowered_hints = combined_location_hint.lower()
        if any(marker in lowered_hints for marker in BROAD_REMOTE_OVERRIDE_MARKERS):
            return True
        direct_job_url = _normalize_direct_job_url(lead.direct_job_url or "")
        return (
            lead.source_type in {"direct_ats", "company_site"}
            and bool(direct_job_url)
            and not _direct_job_url_has_specific_location_hint(direct_job_url)
        )
    if reason_code != "missing_salary":
        return True
    salary_min, salary_max, _salary_text = _hydrate_salary_hint_values(
        lead.base_salary_min_usd_hint,
        lead.base_salary_max_usd_hint,
        lead.salary_text_hint,
        lead.evidence_notes,
    )
    salary_values = [value for value in (salary_min, salary_max) if value is not None]
    return bool(salary_values and max(salary_values) >= settings.min_base_salary_usd)


def _watchlist_entry_should_skip_resolution(
    lead: JobLead,
    settings: Settings,
    watchlist_entry: dict[str, object],
) -> bool:
    if not watchlist_entry:
        return False
    stale_count = _watchlist_reason_count(watchlist_entry, "stale_posting")
    not_remote_count = _watchlist_reason_count(watchlist_entry, "not_remote") + _watchlist_reason_count(
        watchlist_entry,
        "remote_unclear",
    )
    if stale_count >= 4 and not _lead_has_override_hints_for_reason(lead, settings, "stale_posting"):
        return True
    if not_remote_count >= 4 and not _lead_has_override_hints_for_reason(lead, settings, "not_remote"):
        return True
    return False


def _merged_focus_company_entries(
    settings: Settings,
    *,
    company_discovery_entries: dict[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    merged = {
        key: dict(value)
        for key, value in load_company_watchlist_entries(settings.data_dir).items()
    }
    discovery_entries = company_discovery_entries or load_company_discovery_entries(settings.data_dir)
    for key, raw_entry in discovery_entries.items():
        existing = dict(merged.get(key) or {})
        board_identifiers = [
            str(item)
            for item in [*(existing.get("board_identifiers") or []), *(raw_entry.get("board_identifiers") or [])]
            if str(item).strip()
        ]
        source_hosts = [
            str(item)
            for item in [*(existing.get("source_hosts") or []), *(raw_entry.get("source_hosts") or [])]
            if str(item).strip()
        ]
        merged[key] = {
            **existing,
            **raw_entry,
            "company_name": str(raw_entry.get("company_name") or existing.get("company_name") or key),
            "board_identifiers": list(dict.fromkeys(board_identifiers))[:16],
            "source_hosts": list(dict.fromkeys(source_hosts))[:8],
            "priority_score": max(int(existing.get("priority_score") or 0), int(raw_entry.get("source_trust") or 0)),
        }
    return merged


def _select_watchlist_focus_companies(
    settings: Settings,
    known_company_keys: set[str],
    *,
    company_discovery_entries: dict[str, dict[str, object]] | None = None,
    limit: int = 10,
) -> list[str]:
    focus_entries = _merged_focus_company_entries(settings, company_discovery_entries=company_discovery_entries)
    ranked_entries = sorted(
        focus_entries.values(),
        key=lambda entry: (
            0 if _normalize_company_key(entry.get("company_name")) not in known_company_keys else 1,
            0 if _company_looks_small_from_hosts([str(item) for item in entry.get("source_hosts", [])]) else 1,
            -int(entry.get("official_board_lead_count") or 0),
            -int(entry.get("ai_pm_candidate_count") or 0),
            *_watchlist_focus_penalties(entry),
            -int(entry.get("priority_score") or 0),
            str(entry.get("company_name") or "").lower(),
        ),
    )
    companies: list[str] = []
    seen_company_keys: set[str] = set()
    for entry in ranked_entries:
        company_name = str(entry.get("company_name") or "").strip()
        company_key = _normalize_company_key(company_name)
        normalized_name = company_name.lower()
        if not company_name or not company_key or company_key in known_company_keys or company_key in seen_company_keys:
            continue
        if not _focus_company_name_is_timeout_safe(company_name):
            continue
        if any(pattern in normalized_name for pattern in ("hiring for client", "confidential", "stealth startup")):
            continue
        if not _watchlist_entry_is_focusable(entry):
            continue
        if not _focus_company_entry_supports_search_queries(entry):
            continue
        companies.append(company_name)
        seen_company_keys.add(company_key)
        if len(companies) >= limit:
            break
    return _sanitize_focus_companies(companies)[:limit]


def _apply_company_novelty_quota(
    leads: list[JobLead],
    known_company_keys: set[str],
    *,
    min_novelty_ratio: float = NOVEL_COMPANY_TARGET_RATIO,
    limit: int | None = None,
) -> list[JobLead]:
    if not leads:
        return []

    novel_leads = [lead for lead in leads if _normalize_company_key(lead.company_name) not in known_company_keys]
    known_leads = [lead for lead in leads if _normalize_company_key(lead.company_name) in known_company_keys]
    if not novel_leads or min_novelty_ratio <= 0:
        ordered = [*novel_leads, *known_leads]
        return ordered[:limit] if limit is not None else ordered

    novel_block_size = max(
        1,
        int((min_novelty_ratio / max(1e-6, (1 - min_novelty_ratio))) + 0.999),
    )
    ordered: list[JobLead] = []
    while novel_leads or known_leads:
        for _ in range(novel_block_size):
            if not novel_leads:
                break
            ordered.append(novel_leads.pop(0))
            if limit is not None and len(ordered) >= limit:
                return ordered
        if known_leads:
            ordered.append(known_leads.pop(0))
            if limit is not None and len(ordered) >= limit:
                return ordered
        if not known_leads and novel_leads:
            ordered.extend(novel_leads)
            break
        if not novel_leads and known_leads:
            ordered.extend(known_leads)
            break

    return ordered[:limit] if limit is not None else ordered


def _build_small_company_scout_queries(settings: Settings, tuning: SearchTuning) -> list[str]:
    recency_terms = _current_recency_terms(settings.timezone)
    salary_term = f"\"${settings.min_base_salary_usd:,}\""
    domain_window = _attempt_query_window(
        list(SMALL_COMPANY_SCOUT_DOMAINS),
        tuning.attempt_number,
        max(4, min(6, settings.search_round_query_limit)),
    )
    topic_pool = [topic for topic in SMALL_COMPANY_SCOUT_TOPICS if topic != "voice AI" or tuning.attempt_number >= 4]
    topic_window = _attempt_query_window(
        topic_pool,
        tuning.attempt_number,
        max(3, min(4, settings.search_round_query_limit)),
    )
    queries: list[str] = []
    for domain in domain_window:
        for topic in topic_window:
            queries.append(f'site:{domain} "product manager" "{topic}" remote {recency_terms[0]}')
            queries.append(f'site:{domain} "senior product manager" "{topic}" remote')
            queries.append(f'site:{domain} "staff product manager" "{topic}" remote "early stage"')
            queries.append(f'site:{domain} "principal product manager" "{topic}" remote "portfolio company"')
            queries.append(f'site:{domain} "{topic}" "product manager" remote "venture backed"')
            if tuning.prioritize_salary:
                queries.append(f'site:{domain} "product manager" "{topic}" remote {salary_term}')
    return _dedupe_queries(queries)[: max(settings.search_round_query_limit * 2, 10)]


def _build_portfolio_company_scout_queries(settings: Settings, tuning: SearchTuning) -> list[str]:
    recency_terms = _current_recency_terms(settings.timezone)
    topic_pool = [topic for topic in SMALL_COMPANY_SCOUT_TOPICS if topic != "voice AI" or tuning.attempt_number >= 4]
    topic_window = _attempt_query_window(
        topic_pool,
        tuning.attempt_number,
        max(3, min(5, settings.search_round_query_limit)),
    )
    preferred_domains = (
        ["workatastartup.com/jobs", "wellfound.com/jobs", "ycombinator.com/companies"]
        if tuning.attempt_number <= 1
        else ["ycombinator.com/companies", "workatastartup.com/jobs", "wellfound.com/jobs"]
        if tuning.attempt_number == 2
        else ["ycombinator.com/companies", "workatastartup.com/jobs", "getro.com/companies"]
    )
    domain_window = preferred_domains[: max(2, min(3, settings.search_round_query_limit))]
    queries: list[str] = []
    for domain in domain_window:
        for topic in topic_window:
            queries.append(f'site:{domain} "product manager" "{topic}" remote startup')
            queries.append(f'site:{domain} "senior product manager" "{topic}" remote')
            queries.append(f'site:{domain} "staff product manager" "{topic}" remote "portfolio company"')
            queries.append(f'site:{domain} "principal product manager" "{topic}" remote {recency_terms[0]}')
    return _dedupe_queries(queries)[: max(settings.search_round_query_limit * 2, 8)]


def _default_focus_role_terms() -> list[str]:
    return [
        "\"AI Product Manager\"",
        "\"Senior Product Manager, AI\"",
        "\"Principal Product Manager, AI\"",
        "\"Group Product Manager, AI\"",
        "\"Staff Product Manager, AI\"",
        "\"Senior Product Manager, Machine Learning\"",
        "\"Principal Product Manager, Machine Learning\"",
    ]


def _build_focus_company_queries(
    settings: Settings,
    tuning: SearchTuning,
    *,
    include_site_domains: bool,
) -> list[str]:
    focus_roles = tuning.focus_roles or _default_focus_role_terms()
    focus_companies = _sanitize_focus_companies(tuning.focus_companies)
    if not focus_companies or not focus_roles:
        return []
    focus_entries = _merged_focus_company_entries(settings)

    recency_terms = _current_recency_terms(settings.timezone)
    salary_terms = [f"\"${settings.min_base_salary_usd:,}\"", *_salary_disclosure_terms()]
    domain_window = _attempt_query_window(
        list(SMALL_COMPANY_SCOUT_DOMAINS),
        tuning.attempt_number,
        max(4, min(8, settings.search_round_query_limit + 2)),
    )

    company_groups: list[list[str]] = []
    for company in focus_companies[:12]:
        entry = focus_entries.get(_normalize_company_key(company))
        if entry and not _focus_company_entry_supports_search_queries(entry):
            continue
        structured_site_hints = (
            _site_hints_from_focus_company_entry(entry)
            if entry and _watchlist_entry_is_focusable(entry)
            else []
        )
        prefer_structured_site_hints = bool(structured_site_hints) and not include_site_domains
        company_term = f"\"{company}\""
        company_queries: list[str] = []
        for role_term in focus_roles[:6]:
            if not prefer_structured_site_hints:
                company_queries.append(f"{company_term} {role_term} remote")
                if include_site_domains:
                    company_queries.append(f"{company_term} careers {role_term} remote")
                    company_queries.append(
                        f"{company_term} {role_term} remote {_today_for_timezone(settings.timezone).strftime('%Y')}"
                    )
                    if tuning.prioritize_recency:
                        company_queries.append(f"{company_term} {role_term} remote {recency_terms[0]}")
                    if tuning.prioritize_salary:
                        company_queries.append(f"{company_term} {role_term} remote {salary_terms[0]}")
            if include_site_domains:
                for domain in domain_window:
                    company_queries.append(f"{company_term} {role_term} remote site:{domain}")
                    if tuning.prioritize_recency:
                        company_queries.append(f"{company_term} {role_term} remote site:{domain} {recency_terms[0]}")
                    if tuning.prioritize_salary:
                        company_queries.append(f"{company_term} {role_term} remote site:{domain} {salary_terms[1]}")
        company_groups.append(
            _dedupe_queries(company_queries)[: max(6, settings.search_round_query_limit + (2 if include_site_domains else 0))]
        )

    return _dedupe_queries(_interleave_query_groups(company_groups))


def _site_hints_from_board_identifiers(entry: dict[str, object]) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for raw_identifier in entry.get("board_identifiers", []) if isinstance(entry.get("board_identifiers"), list) else []:
        identifier = str(raw_identifier or "").strip().lower()
        if not identifier or ":" not in identifier:
            continue
        prefix, token = identifier.split(":", 1)
        token = token.strip()
        candidates: list[str] = []
        if prefix == "greenhouse" and token:
            candidates = [
                f"site:job-boards.greenhouse.io/{token}",
                f"site:boards.greenhouse.io/{token}",
            ]
        elif prefix == "lever" and token:
            candidates = [f"site:jobs.lever.co/{token}"]
        elif prefix == "ashby" and token:
            candidates = [f"site:jobs.ashbyhq.com/{token}"]
        elif prefix == "smartrecruiters" and token:
            candidates = [f"site:jobs.smartrecruiters.com/{token}"]
        elif prefix == "recruitee" and token:
            candidates = [
                f"site:jobs.recruitee.com/{token}",
                f"site:careers.tellent.com/o/{token}",
            ]
        elif prefix == "jobscore" and token:
            candidates = [f"site:{token}.jobscore.com"]
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                hints.append(candidate)
    return hints


def _build_watchlist_board_focus_queries(settings: Settings, tuning: SearchTuning) -> list[str]:
    focus_roles = tuning.focus_roles or _default_focus_role_terms()
    focus_companies = _sanitize_focus_companies(tuning.focus_companies)
    if not focus_companies or not focus_roles:
        return []
    watchlist = _merged_focus_company_entries(settings)
    recency_terms = _current_recency_terms(settings.timezone)
    salary_terms = [f"\"${settings.min_base_salary_usd:,}\"", *_salary_disclosure_terms()]
    queries: list[str] = []
    for company in focus_companies[:8]:
        entry = watchlist.get(_normalize_company_key(company))
        if not entry or not _watchlist_entry_is_focusable(entry):
            continue
        site_hints = _site_hints_from_focus_company_entry(entry)
        if not site_hints:
            continue
        company_term = f"\"{company}\""
        for role_term in focus_roles[:4]:
            for site_hint in site_hints[:3]:
                queries.append(f"{company_term} {role_term} remote {site_hint}")
                if tuning.prioritize_recency:
                    queries.append(f"{company_term} {role_term} remote {site_hint} {recency_terms[0]}")
                if tuning.prioritize_salary:
                    queries.append(f"{company_term} {role_term} remote {site_hint} {salary_terms[0]}")
    return _dedupe_queries(queries)[: max(settings.search_round_query_limit * 2, 10)]


def _build_company_discovery_seed_queries(settings: Settings, tuning: SearchTuning) -> list[str]:
    role_terms = [
        "\"Principal Product Manager\" AI remote",
        "\"Principal Product Manager\" \"machine learning\" remote",
        "\"Principal Product Manager\" GenAI remote",
        "\"Principal Product Manager\" \"agentic AI\" remote",
        "\"Principal Product Manager\" \"applied AI\" remote",
        "\"Principal Product Manager\" LLM remote",
        "\"Staff Product Manager\" AI remote",
        "\"Staff Product Manager\" \"machine learning platform\" remote",
        "\"Group Product Manager\" AI remote",
        "\"Lead Product Manager\" AI remote",
        "\"Senior Product Manager\" AI remote",
        "\"Technical Product Manager\" AI remote",
        "\"AI Product Manager\" remote",
        "\"AI platform product manager\" remote",
        "\"agentic AI\" \"product manager\" remote",
        "\"applied AI\" \"product manager\" remote",
        "\"model platform\" \"product manager\" remote",
    ]
    ats_domains = [
        "jobs.ashbyhq.com",
        "job-boards.greenhouse.io",
        "boards.greenhouse.io",
        "jobs.lever.co",
        "myworkdayjobs.com",
        "careers.workday.com",
        "jobs.smartrecruiters.com",
        "jobs.icims.com",
        "jobs.workable.com",
        "jobs.jobvite.com",
        "jobs.recruitee.com",
        "careers.tellent.com",
        "ats.rippling.com",
    ]
    directory_domains = [
        "www.ycombinator.com",
        "www.workatastartup.com",
        "wellfound.com",
        "www.builtin.com",
    ]
    recency_terms = _current_recency_terms(settings.timezone)
    queries: list[str] = []
    for role_term in role_terms:
        for domain in ats_domains:
            queries.append(f"{role_term} site:{domain}")
            queries.append(f"{role_term} site:{domain} {recency_terms[0]}")
        for domain in directory_domains:
            queries.append(f"{role_term} site:{domain}")
            queries.append(f"{role_term} site:{domain} {recency_terms[0]}")
    return _dedupe_queries(queries)[: max(settings.search_round_query_limit * 4, 28)]


def _build_local_small_company_scout_queries(settings: Settings, tuning: SearchTuning) -> list[str]:
    recency_terms = _current_recency_terms(settings.timezone)
    salary_term = f"\"${settings.min_base_salary_usd:,}\""
    topic_window = _attempt_query_window(
        list(SMALL_COMPANY_SCOUT_TOPICS),
        tuning.attempt_number,
        max(4, min(6, settings.search_round_query_limit)),
    )
    modifier_window = _attempt_query_window(
        list(SMALL_COMPANY_LOCAL_SCOUT_MODIFIERS),
        tuning.attempt_number,
        max(3, min(4, settings.search_round_query_limit)),
    )
    queries: list[str] = []
    for topic in topic_window:
        for modifier in modifier_window:
            queries.append(f'"product manager" "{topic}" remote {modifier}')
            queries.append(f'"senior product manager" "{topic}" remote {modifier}')
        queries.append(f'"product manager" "{topic}" remote "company careers"')
        queries.append(f'"principal product manager" "{topic}" remote "portfolio company"')
        queries.append(f'"staff product manager" "{topic}" remote "early stage"')
        queries.append(f'"staff product manager" "{topic}" remote startup')
        queries.append(f'"product manager" "{topic}" remote startup {recency_terms[0]}')
        if tuning.prioritize_salary:
            queries.append(f'"product manager" "{topic}" remote startup {salary_term}')
    return _dedupe_queries(queries)[: max(settings.search_round_query_limit * 2, 12)]


def _build_local_targeted_attempt_queries(settings: Settings, tuning: SearchTuning) -> list[str]:
    role_pool = [
        "AI product manager",
        "machine learning product manager",
        "generative AI product manager",
        "AI/ML product manager",
        "senior product manager AI",
        "staff product manager AI",
        "principal product manager AI",
        "group product manager AI",
        "lead product manager AI",
        "AI platform product manager",
        "agentic AI product manager",
        "technical product manager AI",
    ]
    recency_terms = _current_recency_terms(settings.timezone)
    salary_term = f"\"${settings.min_base_salary_usd:,}\""
    ats_domains = [
        "jobs.lever.co",
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "jobs.ashbyhq.com",
        "myworkdayjobs.com",
        "jobs.smartrecruiters.com",
        "jobs.icims.com",
        "careers.workday.com",
    ]
    role_batch = _role_query_batch_for_attempt(
        role_pool,
        tuning.attempt_number,
        batch_size=max(4, settings.search_round_query_limit),
    )
    queries: list[str] = []
    for role_query in role_batch:
        for domain in ats_domains:
            queries.append(f"site:{domain} {role_query} remote")
            queries.append(f"site:{domain} {role_query} remote {recency_terms[0]}")
            if tuning.prioritize_salary:
                queries.append(f"site:{domain} {role_query} remote {salary_term}")
        if tuning.attempt_number > 1:
            queries.append(f"{role_query} remote")
            queries.append(f"{role_query} remote careers")
            queries.append(f"{role_query} remote {recency_terms[0]}")
            if tuning.prioritize_salary:
                queries.append(f"{role_query} remote {salary_term}")
    return _dedupe_queries(queries)


def _base_role_queries() -> list[str]:
    return [
        "\"principal product manager\" AI remote",
        "\"staff product manager\" AI remote",
        "\"group product manager\" AI remote",
        "\"senior product manager\" AI remote",
        "\"lead product manager\" AI remote",
        "\"principal product manager\" \"machine learning\" remote",
        "\"staff product manager\" \"machine learning\" remote",
        "\"group product manager\" \"machine learning\" remote",
        "\"senior product manager\" \"machine learning\" remote",
        "\"principal product manager\" \"LLM\" remote",
        "\"staff product manager\" \"agentic\" remote",
        "\"AI product manager\" remote",
        "\"AI Product Manager\" remote",
        "\"AI/ML product manager\" remote",
        "\"machine learning product manager\" remote",
        "\"generative AI\" \"product manager\" remote",
        "\"LLM\" \"product manager\" remote",
        "\"AI platform product manager\" remote",
        "\"agentic AI\" \"product manager\" remote",
        "\"product manager\" \"AI\" remote",
        "\"product manager\" \"ML\" remote",
        "\"product manager\" \"machine learning\" remote",
        "\"product manager\" \"agentic\" remote",
        "\"product manager\" \"applied AI\" remote",
        "\"ML product manager\" remote",
        "\"AI senior product manager\" remote",
        "\"AI principal product manager\" remote",
        "\"AI staff product manager\" remote",
        "\"AI group product manager\" remote",
        "\"product manager\" \"LLM\" remote",
        "\"product manager\" \"genai\" remote",
        "\"product manager\" \"NLP\" remote",
        "\"product manager\" \"computer vision\" remote",
        "\"product manager\" \"intelligent automation\" remote",
        "\"lead product manager\" machine learning remote",
        "\"platform product manager\" machine learning remote",
        "\"technical product manager\" AI remote",
        "\"technical product manager\" machine learning remote",
        "\"product manager\" \"data science\" remote",
        "\"product manager\" \"model platform\" remote",
        "\"product manager\" \"ai platform\" remote",
        "\"product manager\" \"conversational AI\" remote",
        "\"product manager\" chatbot remote",
        "\"product manager\" \"chat & knowledge\" remote",
        "\"product manager\" \"ML platform\" remote",
        "\"product manager\" \"recommendation systems\" remote",
        "\"product manager\" perception ML remote",
    ]


def _current_recency_terms(timezone_name: str | None = None) -> list[str]:
    now = datetime.now(UTC)
    if timezone_name:
        try:
            now = datetime.now(ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError:
            pass
    current_month_year = now.strftime("%B %Y")
    current_month = now.strftime("%B")
    current_year = now.strftime("%Y")
    return [
        "\"posted this week\"",
        "\"posted in the last week\"",
        "\"posted 1 day ago\"",
        "\"posted 2 days ago\"",
        "\"posted 3 days ago\"",
        f"\"{current_month_year}\"",
        f"\"{current_month} {current_year}\"",
        current_year,
    ]


def _salary_disclosure_terms() -> list[str]:
    return [
        "\"salary\"",
        "\"base salary\"",
        "\"pay range\"",
        "\"compensation\"",
        "\"annual salary\"",
    ]


def _salary_disclosure_regions() -> list[str]:
    return [
        "\"California\"",
        "\"Colorado\"",
        "\"New York\"",
        "\"Washington\"",
    ]


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join(query.split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _interleave_query_groups(groups: list[list[str]]) -> list[str]:
    queues = [deque(group) for group in groups if group]
    ordered: list[str] = []
    while queues:
        next_queues: list[deque[str]] = []
        for queue in queues:
            ordered.append(queue.popleft())
            if queue:
                next_queues.append(queue)
        queues = next_queues
    return ordered


def _role_query_batch_for_attempt(role_queries: list[str], attempt_number: int, batch_size: int) -> list[str]:
    if not role_queries:
        return []
    size = min(batch_size, len(role_queries))
    start = ((attempt_number - 1) * size) % len(role_queries)
    rotated = role_queries[start:] + role_queries[:start]
    return rotated[:size]


def _attempt_query_window(role_queries: list[str], attempt_number: int, window_size: int) -> list[str]:
    if not role_queries or window_size <= 0:
        return []
    start = (attempt_number - 1) * window_size
    if start >= len(role_queries):
        start %= len(role_queries)
    rotated = role_queries[start:] + role_queries[:start]
    return rotated[: min(window_size, len(role_queries))]


def _build_local_role_queries() -> list[str]:
    core_queries = [
        "AI product manager",
        "machine learning product manager",
        "AI/ML product manager",
        "generative AI product manager",
        "genAI product manager",
        "applied AI product manager",
        "applied intelligence product manager",
        "LLM product manager",
        "AI platform product manager",
        "machine learning platform product manager",
        "AI agents product manager",
        "product manager AI agents",
        "agentic AI product manager",
        "agentic systems product manager",
        "AI assistant product manager",
        "AI foundations product manager",
        "AI systems product manager",
        "AI features product manager",
        "AI solutions product manager",
        "AI factory product manager",
        "AI control plane product manager",
        "AI guardrails product manager",
        "voice AI product manager",
        "conversational AI product manager",
        "chatbot product manager",
        "chat and knowledge product manager",
        "LLM operations product manager",
        "ML platform product manager",
        "recommendation systems product manager",
        "perception ML product manager",
        "technical product manager AI",
        "technical product manager machine learning",
        "technical product manager AI platform",
        "technical product manager AI agents",
    ]
    seniorities = ["senior", "staff", "principal", "group", "lead"]
    topics = [
        "AI",
        "machine learning",
        "AI/ML",
        "generative AI",
        "applied AI",
        "applied intelligence",
        "LLM",
        "AI platform",
        "AI agents",
        "agentic AI",
        "AI foundations",
        "AI systems",
        "AI features",
        "AI solutions",
        "AI factory",
        "AI control plane",
        "AI guardrails",
        "conversational AI",
        "chatbot",
        "chat and knowledge",
        "LLM operations",
        "ML platform",
        "recommendation systems",
        "perception ML",
    ]
    seniority_groups: list[list[str]] = []
    for seniority in seniorities:
        group: list[str] = []
        for topic in topics:
            group.append(f"{seniority} product manager {topic}")
        group.append(f"{seniority} technical product manager AI")
        group.append(f"{seniority} technical product manager machine learning")
        seniority_groups.append(group)

    topic_first_queries: list[str] = []
    for topic in (
        "AI",
        "machine learning",
        "AI/ML",
        "LLM",
        "AI platform",
        "AI agents",
        "applied AI",
        "AI foundations",
        "AI systems",
    ):
        topic_first_queries.append(f"{topic} senior product manager")
        topic_first_queries.append(f"{topic} principal product manager")
        topic_first_queries.append(f"{topic} staff product manager")

    specialist_queries = [
        "product manager generative AI",
        "product manager applied AI",
        "product manager applied intelligence",
        "product manager LLM",
        "product manager machine learning platform",
        "product manager AI foundations",
        "product manager AI systems",
        'product manager "computer vision"',
        'product manager "data science" "machine learning"',
        'product manager "modeling" "machine learning"',
        'product manager "recommendation systems"',
        'product manager "deep learning"',
        'product manager "semantic search"',
        'product manager "model serving"',
        'product manager "embeddings"',
        "senior product manager machine learning",
        "staff product manager machine learning",
        "principal product manager machine learning",
        "group product manager machine learning",
        "lead product manager machine learning",
        "senior product manager conversational AI",
        "staff product manager conversational AI",
        "principal product manager conversational AI",
        "senior product manager chatbot",
        "principal product manager ML platform",
        "staff product manager ML platform",
    ]

    ordered_groups = [core_queries, *seniority_groups, topic_first_queries, specialist_queries]
    return _dedupe_queries(_interleave_query_groups(ordered_groups))


def _build_targeted_attempt_queries(settings: Settings, tuning: SearchTuning) -> list[str]:
    role_pool = [
        "\"AI product manager\" remote",
        "\"machine learning product manager\" remote",
        "\"generative AI\" \"product manager\" remote",
        "\"AI/ML product manager\" remote",
        "\"senior product manager\" AI remote",
        "\"staff product manager\" AI remote",
        "\"principal product manager\" AI remote",
        "\"group product manager\" AI remote",
        "\"lead product manager\" AI remote",
        "\"AI platform product manager\" remote",
        "\"agentic AI\" \"product manager\" remote",
        "\"technical product manager\" AI remote",
    ]
    role_batch = _role_query_batch_for_attempt(
        role_pool,
        tuning.attempt_number,
        batch_size=max(4, settings.search_round_query_limit),
    )
    ats_domains = [
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "jobs.lever.co",
        "jobs.ashbyhq.com",
        "jobs.recruitee.com",
        "careers.tellent.com",
        "jobscore.com",
        "myworkdayjobs.com",
        "jobs.smartrecruiters.com",
        "jobs.jobvite.com",
    ]
    queries: list[str] = []
    for index, role_query in enumerate(role_batch):
        primary_domain = ats_domains[index % len(ats_domains)]
        secondary_domain = ats_domains[(index + 2) % len(ats_domains)]
        queries.append(f"{role_query} site:{primary_domain} \"posted this week\"")
        queries.append(f"{role_query} site:{secondary_domain} \"${settings.min_base_salary_usd:,}\"")
        queries.append(f"{role_query} site:{primary_domain}")

    return _dedupe_queries(queries)


def _build_search_query_bank(settings: Settings, tuning: SearchTuning | None = None) -> list[str]:
    tuning = tuning or SearchTuning(attempt_number=1)
    focus_companies = _sanitize_focus_companies(tuning.focus_companies)
    focus_entries = _merged_focus_company_entries(settings) if focus_companies else {}
    eligible_focus_companies = [
        company
        for company in focus_companies
        if _focus_company_entry_supports_search_queries(focus_entries.get(_normalize_company_key(company)))
    ]
    role_queries = _base_role_queries()
    scout_queries = _build_small_company_scout_queries(settings, tuning)
    portfolio_scout_queries = _build_portfolio_company_scout_queries(settings, tuning)
    board_focus_queries = _build_watchlist_board_focus_queries(settings, tuning)
    ats_domains = [
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "jobs.lever.co",
        "jobs.ashbyhq.com",
        "jobs.recruitee.com",
        "careers.tellent.com/o",
        "jobscore.com",
        "comeet.com/jobs",
        "myworkdayjobs.com",
        "jobs.smartrecruiters.com",
        "jobs.jobvite.com",
        "jobs.workable.com",
        "jobs.icims.com",
        "careers.bamboohr.com",
        "recruiting.paylocity.com",
        "jobs.dayforcehcm.com",
        "careers.adp.com",
        "careers.workday.com",
    ]
    discovery_domains = [
        "linkedin.com/jobs/view",
        "builtin.com/jobs",
        "builtinnyc.com/jobs",
        "builtinsf.com/jobs",
        "builtinseattle.com/jobs",
        "builtinla.com/jobs",
        "builtinchicago.org/jobs",
        "builtinchicago.com/jobs",
        "glassdoor.com/Job",
    ]
    seed_queries = [
        query
        for query in settings.search_queries
        if tuning.attempt_number > 1 or not _query_targets_generic_discovery_domain(query)
    ]
    focus_queries: list[str] = []
    direct_queries: list[str] = []
    salary_queries: list[str] = []
    recency_queries: list[str] = []
    discovery_queries: list[str] = []
    broad_queries: list[str] = []

    recency_terms = _current_recency_terms(settings.timezone)
    salary_terms = [f"\"${settings.min_base_salary_usd:,}\"", *_salary_disclosure_terms()]
    salary_regions = _salary_disclosure_regions()

    for role_query in role_queries:
        if tuning.attempt_number > 1:
            broad_queries.append(role_query)
            broad_queries.append(f"{role_query} \"United States\"")

        direct_queries.append(role_query)
        direct_queries.append(role_query.replace(" remote", ' "Remote - US"', 1))

        for domain in ats_domains:
            direct_queries.append(f"{role_query} site:{domain}")
            recency_queries.append(f"{role_query} site:{domain} {recency_terms[0]}")
            salary_queries.append(f"{role_query} site:{domain} {salary_terms[0]}")

        for domain in discovery_domains:
            discovery_queries.append(f"site:{domain} {role_query} {recency_terms[0]}")
            if tuning.attempt_number > 1:
                discovery_queries.append(f"site:{domain} {role_query}")

        for recency_term in recency_terms[:4]:
            recency_queries.append(f"{role_query} {recency_term}")

        for salary_term in salary_terms[:2]:
            salary_queries.append(f"{role_query} {salary_term}")
        if tuning.prioritize_salary or tuning.attempt_number > 1:
            for salary_term in salary_terms[2:]:
                salary_queries.append(f"{role_query} {salary_term}")
            for salary_region in salary_regions:
                salary_queries.append(f"{role_query} {salary_region} \"salary\"")
                salary_queries.append(f"{role_query} {salary_region} \"base salary\"")

    focus_queries.extend(_build_focus_company_queries(settings, tuning, include_site_domains=tuning.attempt_number > 1))
    for company in eligible_focus_companies[:10]:
        company_term = f"\"{company}\""
        for role_term in (tuning.focus_roles or _default_focus_role_terms())[:6]:
            for domain in discovery_domains:
                focus_queries.append(f"site:{domain} {company_term} {role_term} remote")

    query_groups = [
        direct_queries,
        recency_queries,
        salary_queries,
        seed_queries,
    ]
    if tuning.attempt_number > 1:
        query_groups.append(discovery_queries)
    if tuning.attempt_number > 2 and broad_queries:
        query_groups.append(broad_queries)

    queries = [
        *focus_queries,
        *board_focus_queries,
        *portfolio_scout_queries,
        *scout_queries,
        *_interleave_query_groups(query_groups),
    ]
    return _dedupe_queries(queries)


def _filter_query_bank_for_cross_run_cooldowns(
    queries: list[str],
    *,
    query_family_history: dict[str, dict[str, object]] | None = None,
    run_id: str | None = None,
) -> list[str]:
    if not query_family_history:
        return queries
    return [
        query
        for query in queries
        if _query_family_cooldown_reason(
            query,
            query_family_history=query_family_history,
            run_id=run_id,
        )
        is None
    ]


def _build_query_rounds(
    settings: Settings,
    tuning: SearchTuning | None = None,
    *,
    query_family_history: dict[str, dict[str, object]] | None = None,
    run_id: str | None = None,
) -> list[list[str]]:
    if settings.llm_provider == "ollama":
        return _build_local_query_rounds(
            settings,
            tuning=tuning,
            query_family_history=query_family_history,
            run_id=run_id,
        )

    tuning = tuning or SearchTuning(attempt_number=1)
    targeted_queries = _build_targeted_attempt_queries(settings, tuning)
    fallback_queries = _build_search_query_bank(settings, tuning=tuning)
    query_bank = _dedupe_queries([*targeted_queries, *fallback_queries])
    query_bank = _filter_query_bank_for_cross_run_cooldowns(
        query_bank,
        query_family_history=query_family_history,
        run_id=run_id,
    )
    max_queries = settings.max_search_rounds * settings.search_round_query_limit
    limited_bank = query_bank[:max_queries]
    return [
        limited_bank[index : index + settings.search_round_query_limit]
        for index in range(0, len(limited_bank), settings.search_round_query_limit)
    ]


def _build_local_query_rounds(
    settings: Settings,
    tuning: SearchTuning | None = None,
    *,
    query_family_history: dict[str, dict[str, object]] | None = None,
    run_id: str | None = None,
) -> list[list[str]]:
    tuning = tuning or SearchTuning(attempt_number=1)
    local_role_queries = _build_local_role_queries()
    scout_queries = _build_local_small_company_scout_queries(settings, tuning)
    portfolio_scout_queries = _build_portfolio_company_scout_queries(settings, tuning)
    board_focus_queries = _build_watchlist_board_focus_queries(settings, tuning)
    per_attempt_budget = settings.max_search_rounds * settings.search_round_query_limit

    focused_roles = _dedupe_queries((tuning.focus_roles or [])[:6])
    focus_company_queries = _build_focus_company_queries(
        settings,
        tuning,
        include_site_domains=tuning.attempt_number > 1,
    )
    targeted_queries = _build_local_targeted_attempt_queries(settings, tuning)[:per_attempt_budget]
    structured_targeted_queries = [query for query in targeted_queries if _query_uses_structured_source_hint(query)]
    generic_targeted_queries = [query for query in targeted_queries if query not in structured_targeted_queries]
    local_window_budget = per_attempt_budget if tuning.attempt_number == 1 else max(2, settings.search_round_query_limit)
    local_window = _attempt_query_window(local_role_queries, tuning.attempt_number, local_window_budget)
    if tuning.attempt_number == 1:
        local_window = []
    else:
        local_window = [query for query in local_window if not _query_is_broad_generic(query)]
    query_groups = [
        focus_company_queries,
        board_focus_queries,
        structured_targeted_queries,
        portfolio_scout_queries,
    ]
    if tuning.attempt_number == 1:
        query_groups.extend([scout_queries, generic_targeted_queries])
    else:
        query_groups.extend([scout_queries, generic_targeted_queries, local_window])
    query_bank = [
        *focused_roles,
        *_interleave_query_groups(query_groups),
    ]
    query_bank = _dedupe_queries(query_bank)
    query_bank = _filter_query_bank_for_cross_run_cooldowns(
        query_bank,
        query_family_history=query_family_history,
        run_id=run_id,
    )
    limited_bank = query_bank[:per_attempt_budget]
    return [
        limited_bank[index : index + settings.search_round_query_limit]
        for index in range(0, len(limited_bank), settings.search_round_query_limit)
    ]


def _query_uses_structured_source_hint(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "site:",
            "workatastartup.com",
            "ycombinator.com/companies",
            "getro.com",
            "builtin.com",
            "greenhouse",
            "ashby",
            "lever",
            "smartrecruiters",
            "recruitee",
            "tellent",
            "jobscore",
            "workday",
            "icims",
        )
    )


def _query_family_history_path(settings: Settings) -> Path:
    return settings.data_dir / QUERY_FAMILY_HISTORY_FILENAME


def _load_query_family_history(settings: Settings) -> dict[str, dict[str, object]]:
    path = _query_family_history_path(settings)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw_families = payload.get("families") if isinstance(payload, dict) else None
    if not isinstance(raw_families, dict):
        return {}
    families: dict[str, dict[str, object]] = {}
    for family, metrics in raw_families.items():
        if isinstance(family, str) and isinstance(metrics, dict):
            families[family] = dict(metrics)
    return families


def _save_query_family_history(settings: Settings, families: dict[str, dict[str, object]]) -> None:
    path = _query_family_history_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "families": families,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _query_family_key(query: str) -> str:
    lowered = " ".join(query.lower().split())
    if not lowered:
        return "other"
    if "site:getro.com/companies" in lowered:
        return "startup_getro"
    if "site:workatastartup.com/jobs" in lowered:
        return "startup_workatastartup"
    if "site:ycombinator.com/companies" in lowered:
        return "startup_ycombinator"
    if "site:wellfound.com/jobs" in lowered:
        return "startup_wellfound"
    if any(marker in lowered for marker in ("site:linkedin.com/jobs/view", "site:builtin", "site:glassdoor.com/job")):
        return "generic_discovery"
    if any(
        marker in lowered
        for marker in (
            "site:job-boards.greenhouse.io",
            "site:boards.greenhouse.io",
            "site:jobs.lever.co",
            "site:jobs.ashbyhq.com",
            "site:jobs.recruitee.com",
            "site:careers.tellent.com/o",
            "site:jobscore.com",
            "site:myworkdayjobs.com",
            "site:jobs.smartrecruiters.com",
            "site:jobs.jobvite.com",
            "site:jobs.workable.com",
            "site:jobs.icims.com",
            "site:careers.workday.com",
        )
    ):
        return "structured_ats"
    if _query_targets_startup_ecosystem(lowered):
        return "startup_generic"
    if _query_is_broad_generic(lowered):
        return "broad_generic"
    if "site:" in lowered:
        return "site_scoped_other"
    return "company_focus"


def _query_family_is_timeout_cooldown_eligible(family: str) -> bool:
    return family in QUERY_FAMILY_COOLDOWN_ELIGIBLE


def _query_family_cooldown_is_stale(metrics: dict[str, object]) -> bool:
    raw_timestamp = str(metrics.get("last_updated_at") or "").strip()
    if not raw_timestamp:
        return False
    try:
        updated_at = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return datetime.now(UTC) - updated_at.astimezone(UTC) > QUERY_FAMILY_COOLDOWN_MAX_AGE


def _query_family_cooldown_reason(
    query: str,
    *,
    query_family_history: dict[str, dict[str, object]] | None = None,
    run_id: str | None = None,
) -> str | None:
    if not query_family_history:
        return None
    family = _query_family_key(query)
    if not _query_family_is_timeout_cooldown_eligible(family):
        return None
    metrics = query_family_history.get(family) or {}
    if _query_family_cooldown_is_stale(metrics):
        return None
    timeout_streak = int(metrics.get("consecutive_timeout_heavy_runs") or 0)
    zero_yield_streak = int(metrics.get("consecutive_zero_yield_runs") or 0)
    if run_id and str(metrics.get("last_run_id") or "").strip() == run_id and timeout_streak >= 1:
        return (
            f"The {family.replace('_', ' ')} query family is cooling down for the rest of this run "
            "after an earlier timeout-heavy pass."
        )
    if timeout_streak >= 2:
        return (
            f"The {family.replace('_', ' ')} query family is cooling down after "
            f"{timeout_streak} timeout-heavy runs with no meaningful yield."
        )
    if zero_yield_streak >= 2:
        return (
            f"The {family.replace('_', ' ')} query family is cooling down after "
            f"{zero_yield_streak} zero-yield runs."
        )
    return None


def _query_family_metrics_template() -> dict[str, int]:
    return {
        "executed_queries": 0,
        "timeout_count": 0,
        "zero_yield_queries": 0,
        "fresh_lead_count": 0,
        "validated_job_count": 0,
        "new_company_count": 0,
        "new_board_count": 0,
        "official_board_lead_count": 0,
        "frontier_expansion_count": 0,
    }


def _update_query_family_run_metrics(
    query_family_metrics: dict[str, dict[str, int]],
    query: str,
    *,
    count_execution: bool = True,
    timed_out: bool = False,
    fresh_leads: int = 0,
    validated_jobs: int = 0,
    new_companies: int = 0,
    new_boards: int = 0,
    official_board_leads: int = 0,
    frontier_expansion: int = 0,
) -> None:
    family = _query_family_key(query)
    metrics = query_family_metrics.setdefault(family, _query_family_metrics_template())
    if count_execution:
        metrics["executed_queries"] += 1
    if timed_out:
        metrics["timeout_count"] += 1
    if count_execution and fresh_leads <= 0:
        metrics["zero_yield_queries"] += 1
    elif fresh_leads > 0:
        metrics["fresh_lead_count"] += fresh_leads
    if validated_jobs > 0:
        metrics["validated_job_count"] += validated_jobs
    if new_companies > 0:
        metrics["new_company_count"] += new_companies
    if new_boards > 0:
        metrics["new_board_count"] += new_boards
    if official_board_leads > 0:
        metrics["official_board_lead_count"] += official_board_leads
    if frontier_expansion > 0:
        metrics["frontier_expansion_count"] += frontier_expansion


def _update_query_family_metric_sets(
    metric_sets: tuple[dict[str, dict[str, int]], ...],
    query: str,
    *,
    count_execution: bool = True,
    timed_out: bool = False,
    fresh_leads: int = 0,
    validated_jobs: int = 0,
    new_companies: int = 0,
    new_boards: int = 0,
    official_board_leads: int = 0,
    frontier_expansion: int = 0,
) -> None:
    for metrics in metric_sets:
        _update_query_family_run_metrics(
            metrics,
            query,
            count_execution=count_execution,
            timed_out=timed_out,
            fresh_leads=fresh_leads,
            validated_jobs=validated_jobs,
            new_companies=new_companies,
            new_boards=new_boards,
            official_board_leads=official_board_leads,
            frontier_expansion=frontier_expansion,
        )


def _query_family_has_meaningful_discovery_yield(metrics: dict[str, object] | None) -> bool:
    if not metrics:
        return False
    return (
        int(metrics.get("validated_job_count") or 0) > 0
        or int(metrics.get("official_board_lead_count") or 0) > 0
        or int(metrics.get("new_company_count") or 0) > 0
        or int(metrics.get("new_board_count") or 0) > 0
        or int(metrics.get("frontier_expansion_count") or 0) >= 2
        or int(metrics.get("fresh_lead_count") or 0) >= 2
    )


def _query_family_run_is_zero_yield(run_metrics: dict[str, int]) -> bool:
    executed_queries = int(run_metrics.get("executed_queries") or 0)
    return executed_queries > 0 and not _query_family_has_meaningful_discovery_yield(run_metrics)


def _query_family_run_is_timeout_heavy(run_metrics: dict[str, int]) -> bool:
    executed_queries = int(run_metrics.get("executed_queries") or 0)
    timeout_count = int(run_metrics.get("timeout_count") or 0)
    return (
        _query_family_run_is_zero_yield(run_metrics)
        and executed_queries > 0
        and timeout_count >= max(2, executed_queries // 2)
    )


def _merge_query_family_history(
    query_family_history: dict[str, dict[str, object]],
    *,
    run_id: str | None,
    query_family_metrics: dict[str, dict[str, int]],
) -> dict[str, dict[str, object]]:
    if not query_family_metrics:
        return query_family_history
    history = {family: dict(metrics) for family, metrics in query_family_history.items()}
    updated_at = datetime.now(UTC).isoformat(timespec="seconds")
    for family, run_metrics in query_family_metrics.items():
        previous = dict(history.get(family) or {})
        zero_yield_run = _query_family_run_is_zero_yield(run_metrics)
        timeout_heavy_run = _query_family_run_is_timeout_heavy(run_metrics)
        history[family] = {
            "run_count": int(previous.get("run_count") or 0) + 1,
            "executed_queries_total": int(previous.get("executed_queries_total") or 0)
            + int(run_metrics.get("executed_queries") or 0),
            "timeout_count_total": int(previous.get("timeout_count_total") or 0)
            + int(run_metrics.get("timeout_count") or 0),
            "zero_yield_queries_total": int(previous.get("zero_yield_queries_total") or 0)
            + int(run_metrics.get("zero_yield_queries") or 0),
            "fresh_lead_count_total": int(previous.get("fresh_lead_count_total") or 0)
            + int(run_metrics.get("fresh_lead_count") or 0),
            "validated_job_count_total": int(previous.get("validated_job_count_total") or 0)
            + int(run_metrics.get("validated_job_count") or 0),
            "new_company_count_total": int(previous.get("new_company_count_total") or 0)
            + int(run_metrics.get("new_company_count") or 0),
            "new_board_count_total": int(previous.get("new_board_count_total") or 0)
            + int(run_metrics.get("new_board_count") or 0),
            "official_board_lead_count_total": int(previous.get("official_board_lead_count_total") or 0)
            + int(run_metrics.get("official_board_lead_count") or 0),
            "frontier_expansion_count_total": int(previous.get("frontier_expansion_count_total") or 0)
            + int(run_metrics.get("frontier_expansion_count") or 0),
            "consecutive_zero_yield_runs": int(previous.get("consecutive_zero_yield_runs") or 0) + 1
            if zero_yield_run
            else 0,
            "consecutive_timeout_heavy_runs": int(previous.get("consecutive_timeout_heavy_runs") or 0) + 1
            if timeout_heavy_run
            else 0,
            "last_run_id": run_id,
            "last_updated_at": updated_at,
        }
    return history


def _persist_query_family_history(
    settings: Settings,
    *,
    run_id: str | None,
    query_family_metrics: dict[str, dict[str, int]],
) -> None:
    if not query_family_metrics:
        return
    history = _merge_query_family_history(
        _load_query_family_history(settings),
        run_id=run_id,
        query_family_metrics=query_family_metrics,
    )
    _save_query_family_history(settings, history)


def _query_timeout_sensitive_marker(query: str) -> str | None:
    lowered = query.lower()
    for marker in TIMEOUT_SENSITIVE_QUERY_MARKERS:
        if marker in lowered:
            return marker
    return None


def _query_targets_startup_ecosystem(query: str) -> bool:
    lowered = query.lower()
    if _query_timeout_sensitive_marker(lowered):
        return True
    return any(
        token in lowered
        for token in (
            "startup",
            "series a",
            "series b",
            "series c",
            "seed stage",
            "venture backed",
            "vc backed",
            "portfolio company",
            "early stage",
            "growth stage",
            "founding team",
            "vertical ai",
            "voice ai",
            "conversational ai",
            "applied intelligence",
        )
    )


def _query_targets_generic_discovery_domain(query: str) -> bool:
    lowered = query.lower()
    return any(
        marker in lowered
        for marker in (
            "site:linkedin.com/jobs/view",
            "site:builtin.com/jobs",
            "site:builtinnyc.com/jobs",
            "site:builtinsf.com/jobs",
            "site:builtinseattle.com/jobs",
            "site:builtinla.com/jobs",
            "site:builtinchicago.org/jobs",
            "site:builtinchicago.com/jobs",
            "site:glassdoor.com/job",
        )
    )


def _query_is_broad_generic(query: str) -> bool:
    lowered = query.lower()
    if _query_uses_structured_source_hint(query):
        return False
    if "company careers" in lowered:
        return True
    if any(token in lowered for token in ("series a", "series b", "seed stage", "portfolio company", "venture backed", "vertical ai", "early stage")):
        return False
    quoted_phrases = re.findall(r'"[^"]+"', query)
    if len(quoted_phrases) >= 2:
        return False
    return True


def _query_timeout_seconds_for_query(settings: Settings, query: str) -> int:
    base_timeout = max(10, settings.per_query_timeout_seconds)
    if settings.llm_provider != "ollama":
        return base_timeout
    if _query_timeout_sensitive_marker(query):
        return min(max(base_timeout - 12, 18), 22)
    if _query_is_broad_generic(query):
        return min(max(base_timeout - 15, 15), 25)
    if _query_uses_structured_source_hint(query):
        return min(max(base_timeout, 30), 45)
    return min(max(base_timeout - 5, 20), 35)


def _timed_out_queries(diagnostics: SearchDiagnostics, *, attempt_number: int | None = None) -> set[str]:
    queries: set[str] = set()
    for failure in diagnostics.failures:
        if failure.reason_code != "query_timeout" or not failure.source_query:
            continue
        if attempt_number is not None and failure.attempt_number != attempt_number:
            continue
        normalized_query = " ".join(failure.source_query.split())
        if normalized_query:
            queries.add(normalized_query)
    return queries


def _broad_generic_query_timeout_count(diagnostics: SearchDiagnostics, *, attempt_number: int) -> int:
    return sum(
        1
        for failure in diagnostics.failures
        if failure.reason_code == "query_timeout"
        and failure.attempt_number == attempt_number
        and failure.source_query
        and _query_is_broad_generic(failure.source_query)
    )


def _timeout_sensitive_query_timeout_count(
    diagnostics: SearchDiagnostics,
    *,
    attempt_number: int,
    marker: str,
) -> int:
    return sum(
        1
        for failure in diagnostics.failures
        if failure.reason_code == "query_timeout"
        and failure.attempt_number == attempt_number
        and failure.source_query
        and marker in failure.source_query.lower()
    )


def _company_focused_query_marker(query: str) -> str | None:
    phrases = [" ".join(match.lower().split()) for match in re.findall(r'"([^"]+)"', query)]
    for phrase in phrases:
        if len(phrase) < 3:
            continue
        if len(phrase.split()) > 4:
            continue
        if any(
            token in phrase
            for token in (
                "product manager",
                "program manager",
                "group product",
                "staff product",
                "senior product",
                "principal product",
                "technical product",
                "careers",
                "jobs",
                "remote",
                "posted this week",
                "series a",
                "series b",
                "series c",
                "seed stage",
                "growth stage",
                "early stage",
                "venture backed",
                "vc backed",
                "portfolio company",
                "founding team",
                "startup",
                "vertical ai",
                "voice ai",
                "conversational ai",
                "applied intelligence",
            )
        ):
            continue
        if phrase in {"ai", "genai", "voice ai", "artificial intelligence"}:
            continue
        return phrase
    return None


def _company_focused_query_timeout_count(
    diagnostics: SearchDiagnostics,
    *,
    attempt_number: int,
    marker: str,
) -> int:
    return sum(
        1
        for failure in diagnostics.failures
        if failure.reason_code == "query_timeout"
        and failure.attempt_number == attempt_number
        and failure.source_query
        and _company_focused_query_marker(failure.source_query) == marker
    )


def _company_focused_query_is_timeout_prone_variant(query: str) -> bool:
    lowered = " ".join(query.lower().split())
    if not lowered:
        return False
    if " careers " in lowered:
        return False
    if re.search(r"\b20\d{2}\b", lowered):
        return True
    if "$" in query:
        return True
    return any(
        token in lowered
        for token in (
            "salary",
            "base salary",
            "compensation",
            "annual salary",
            "posted this week",
            "this week",
            "past week",
            "today",
        )
    )


def _query_is_company_named_open_web_query(query: str) -> bool:
    normalized_query = " ".join(query.lower().split())
    if not normalized_query or "site:" in normalized_query:
        return False
    return _company_focused_query_marker(query) is not None


def _query_timeout_skip_reason(
    diagnostics: SearchDiagnostics,
    query: str,
    *,
    attempt_number: int,
    query_family_history: dict[str, dict[str, object]] | None = None,
    attempt_query_family_metrics: dict[str, dict[str, int]] | None = None,
    run_id: str | None = None,
) -> str | None:
    normalized_query = " ".join(query.split())
    if not normalized_query:
        return None
    family = _query_family_key(normalized_query)
    current_family_metrics = (attempt_query_family_metrics or {}).get(family) or {}
    family_has_productive_yield = _query_family_has_meaningful_discovery_yield(current_family_metrics)
    family_cooldown_reason = _query_family_cooldown_reason(
        normalized_query,
        query_family_history=query_family_history,
        run_id=run_id,
    )
    if family_cooldown_reason is not None:
        return family_cooldown_reason
    if normalized_query in _timed_out_queries(diagnostics):
        return "The same discovery query already timed out earlier in this run."
    lowered_query = normalized_query.lower()
    company_marker = _company_focused_query_marker(normalized_query)
    if company_marker is not None:
        company_timeout_count = _company_focused_query_timeout_count(
            diagnostics,
            attempt_number=attempt_number,
            marker=company_marker,
        )
        if company_timeout_count >= 1 and " careers " in lowered_query:
            return (
                "The company careers query variant is being skipped for "
                f"{company_marker} after an earlier timeout in this pass."
            )
        if company_timeout_count >= COMPANY_FOCUSED_QUERY_TIMEOUT_SKIP_THRESHOLD:
            return (
                "The company-specific timeout circuit breaker is open for "
                f"{company_marker} after {company_timeout_count} timeouts in this pass."
            )
        if company_timeout_count >= 1 and _company_focused_query_is_timeout_prone_variant(normalized_query):
            return (
                "The timeout-prone company query variant is being skipped for "
                f"{company_marker} after an earlier timeout in this pass."
            )
        if (
            company_timeout_count >= 1
            and _query_is_company_named_open_web_query(normalized_query)
            and not family_has_productive_yield
        ):
            if attempt_number >= 2:
                return (
                    "Late-pass company-specific open-web discovery queries are being suppressed for "
                    f"{company_marker} after an earlier timeout in this pass."
                )
            return (
                "Same-pass company-specific open-web discovery queries are being suppressed for "
                f"{company_marker} after an earlier timeout in this pass."
            )
    if attempt_number >= 3 and "site:getro.com/companies" in lowered_query:
        return "Late-pass Getro company-board queries are being suppressed after repeated low-yield timeout behavior."
    if attempt_number >= 3 and '"voice ai"' in lowered_query:
        return "Late-pass voice-AI scout queries are being suppressed because they have been low-yield timeout sinks."
    timeout_sensitive_marker = _query_timeout_sensitive_marker(normalized_query)
    if timeout_sensitive_marker is not None:
        timeout_sensitive_threshold = TIMEOUT_SENSITIVE_QUERY_SKIP_THRESHOLD + (2 if family_has_productive_yield else 0)
        timeout_sensitive_count = _timeout_sensitive_query_timeout_count(
            diagnostics,
            attempt_number=attempt_number,
            marker=timeout_sensitive_marker,
        )
        if timeout_sensitive_count >= timeout_sensitive_threshold:
            return (
                "The startup-board timeout circuit breaker is open for "
                f"{timeout_sensitive_marker} after {timeout_sensitive_count} timeouts in this pass."
            )
    if not _query_is_broad_generic(normalized_query):
        return None
    broad_timeout_count = _broad_generic_query_timeout_count(diagnostics, attempt_number=attempt_number)
    broad_timeout_threshold = BROAD_GENERIC_QUERY_TIMEOUT_SKIP_THRESHOLD + (3 if family_has_productive_yield else 0)
    if broad_timeout_count >= broad_timeout_threshold:
        return (
            "The broad-query timeout circuit breaker is open after "
            f"{broad_timeout_count} broad discovery query timeouts in this pass."
        )
    return None


def _failure_counts_for_attempt(diagnostics: SearchDiagnostics, attempt_number: int) -> Counter[str]:
    return Counter(failure.reason_code for failure in diagnostics.failures if failure.attempt_number == attempt_number)


def _timeout_budget_failure_count(diagnostics: SearchDiagnostics, attempt_number: int) -> int:
    counts = _failure_counts_for_attempt(diagnostics, attempt_number)
    return counts.get("query_timeout", 0) + counts.get("query_skipped_timeout_budget", 0)


def _should_abort_dead_attempt_round(
    diagnostics: SearchDiagnostics,
    *,
    attempt_number: int,
    consecutive_zero_yield_rounds: int,
    attempt_discovery_gain: int,
) -> bool:
    return (
        attempt_number >= 2
        and consecutive_zero_yield_rounds >= 2
        and attempt_discovery_gain == 0
        and _timeout_budget_failure_count(diagnostics, attempt_number) >= 4
    )


def _should_stop_after_dead_attempt(
    diagnostics: SearchDiagnostics,
    *,
    attempt_number: int,
    attempt_discovery_gain: int,
    resolved_leads_this_attempt: int,
) -> bool:
    return (
        attempt_number >= 2
        and attempt_discovery_gain == 0
        and resolved_leads_this_attempt == 0
        and _timeout_budget_failure_count(diagnostics, attempt_number) >= 4
    )


def _top_failure_summary(diagnostics: SearchDiagnostics, attempt_number: int, *, limit: int = 4) -> str:
    counts = _failure_counts_for_attempt(diagnostics, attempt_number)
    if not counts:
        return "no failures recorded"
    return ", ".join(f"{reason}: {count}" for reason, count in counts.most_common(limit))


def _select_focus_companies(settings: Settings, diagnostics: SearchDiagnostics, attempt_number: int) -> list[str]:
    candidates: Counter[str] = Counter()
    for failure in diagnostics.failures:
        if failure.attempt_number != attempt_number:
            continue
        if failure.reason_code not in FOCUSABLE_REASON_CODES:
            continue
        if failure.posted_date_text and _hint_is_recent(failure.posted_date_text, settings) is False:
            continue
        if failure.is_remote is False:
            continue
        if not _failure_supports_adaptive_focus(failure):
            continue
        if failure.company_name and _focus_company_name_is_timeout_safe(failure.company_name):
            candidates[failure.company_name] += 1
    return _sanitize_focus_companies([company for company, _ in candidates.most_common(8)])


def _normalize_role_title_to_focus_queries(role_title: str) -> list[str]:
    normalized = role_title.lower()
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[^a-z0-9+/ ]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if "product manager" not in normalized and "ml product manager" not in normalized:
        return []

    seniority = ""
    for label in ("principal", "staff", "senior", "group", "lead"):
        if re.search(rf"\b{label}\b", normalized):
            seniority = label
            break

    base_role = "technical product manager" if "technical product manager" in normalized else "product manager"

    topic_rules = [
        ("agentic AI", ("agentic ai",)),
        ("AI agents", ("ai agents", "agents")),
        ("AI assistant", ("ai assistant", "assistant")),
        ("applied AI", ("applied ai",)),
        ("generative AI", ("generative ai", "genai")),
        ("LLM", ("llm",)),
        ("AI/ML", ("ai/ml",)),
        ("machine learning", ("machine learning", " ml ")),
        ("AI platform", ("ai platform", "platform")),
        ("AI", (" ai ",)),
    ]
    padded = f" {normalized} "
    topics: list[str] = []
    for label, needles in topic_rules:
        if any(needle in padded for needle in needles):
            topics.append(label)

    if not topics and "ml product manager" in normalized:
        topics.append("machine learning")

    if not topics:
        return []

    queries = [
        " ".join(part for part in (seniority, base_role, topic) if part).strip()
        for topic in topics[:2]
    ]
    if "AI" not in topics:
        queries.append(" ".join(part for part in (seniority, base_role, "AI") if part).strip())
    return _dedupe_queries(queries)


def _failure_supports_adaptive_focus(failure: SearchFailure) -> bool:
    role_title = (failure.role_title or "").strip()
    if not role_title:
        return False
    if _normalize_role_title_to_focus_queries(role_title):
        return True
    return _is_ai_related_product_manager_text(role_title)


def _select_focus_roles(settings: Settings, diagnostics: SearchDiagnostics, attempt_number: int) -> list[str]:
    candidates: Counter[str] = Counter()
    for failure in diagnostics.failures:
        if failure.attempt_number != attempt_number:
            continue
        if failure.reason_code not in FOCUSABLE_REASON_CODES:
            continue
        if failure.posted_date_text and _hint_is_recent(failure.posted_date_text, settings) is False:
            continue
        if failure.is_remote is False:
            continue
        if not _failure_supports_adaptive_focus(failure):
            continue
        if failure.role_title:
            for query in _normalize_role_title_to_focus_queries(failure.role_title):
                candidates[query] += 1
    return [role for role, _ in candidates.most_common(6)]


def _derive_next_tuning(settings: Settings, diagnostics: SearchDiagnostics, attempt_number: int) -> SearchTuning:
    counts = _failure_counts_for_attempt(diagnostics, attempt_number)
    return SearchTuning(
        attempt_number=attempt_number + 1,
        prioritize_recency=sum(count for code, count in counts.items() if code in RECENCY_REASON_CODES) >= 3,
        prioritize_salary=sum(count for code, count in counts.items() if code in SALARY_REASON_CODES) >= 3,
        prioritize_remote=sum(count for code, count in counts.items() if code in REMOTE_REASON_CODES) >= 2,
        focus_companies=_select_focus_companies(settings, diagnostics, attempt_number),
        focus_roles=_select_focus_roles(settings, diagnostics, attempt_number),
    )


def build_job_discovery_agent(settings: Settings, tuning: SearchTuning) -> Agent:
    instructions = f"""
You discover strong leads for AI-related Product Manager roles.

Rules:
- Prefer direct ATS or company careers URLs.
- If a lead comes from LinkedIn/Built In/Glassdoor, return it only when you can also provide a matching direct_job_url.
- Skip generic jobs index pages and ambiguous company pages.
- Focus on US-remote roles posted in the last {settings.posted_within_days} days.
- Prefer roles likely to meet >= ${settings.min_base_salary_usd:,} base.
- If compensation is missing, senior roles with clear 7+ years experience requirements are strong leads.
- Never invent company names, titles, or URLs.
- Return at most {settings.max_leads_per_query} leads.

Pass priorities:
- recency: {tuning.prioritize_recency}
- salary visibility: {tuning.prioritize_salary}
- explicit remote evidence: {tuning.prioritize_remote}

source_type must be one of: direct_ats, linkedin, builtin, glassdoor, company_site, other
""".strip()

    return Agent(
        name=f"Job Discovery Agent Pass {tuning.attempt_number}",
        model="gpt-5.1",
        instructions=instructions,
        tools=[
            WebSearchTool(
                user_location=settings.user_location,
                search_context_size="medium",
            )
        ],
        output_type=JobLeadSearchResult,
    )


def build_direct_job_resolution_agent(settings: Settings) -> Agent:
    instructions = """
Resolve each lead to the exact direct ATS or company careers posting URL.

Rules:
- Return only the specific job posting URL, not a jobs index.
- Never return aggregator URLs (LinkedIn Jobs, Built In, Glassdoor, Indeed, ZipRecruiter, etc.).
- Accept only when company and role clearly match; otherwise reject.
- Keep evidence_notes brief and concrete.
""".strip()

    return Agent(
        name="Direct Job Resolution Agent",
        model="gpt-5.1",
        instructions=instructions,
        tools=[
            WebSearchTool(
                user_location=settings.user_location,
                search_context_size="medium",
            )
        ],
        output_type=DirectJobResolution,
    )


def _normalize_and_filter_discovery_leads(leads: list[JobLead], query: str) -> list[JobLead]:
    normalized_leads: list[JobLead] = []
    expected_company_marker = (
        _company_focused_query_marker(query) if _query_is_company_named_open_web_query(query) else None
    )
    for lead in leads:
        normalized_direct = _normalize_direct_job_url(lead.direct_job_url) if lead.direct_job_url else None
        source_page_type = _normalize_source_type(lead.source_url)
        normalized = lead.model_copy(
            update={
                "source_query": query,
                "direct_job_url": normalized_direct,
                "source_type": _normalize_source_type(normalized_direct or lead.source_url),
            }
        )

        if normalized.direct_job_url and not _is_allowed_direct_job_url(normalized.direct_job_url):
            normalized = normalized.model_copy(update={"direct_job_url": None})

        if not normalized.direct_job_url and not _is_supported_discovery_source_url(normalized.source_url):
            continue
        if expected_company_marker and source_page_type in {"builtin", "linkedin", "glassdoor", "other"}:
            continue
        if "product manager" not in normalized.role_title.lower():
            continue
        if not _lead_is_ai_related_product_manager(normalized):
            continue
        if expected_company_marker:
            company_hints = [
                normalized.company_name,
                _company_hint_from_url(normalized.direct_job_url or ""),
                _company_hint_from_url(normalized.source_url),
            ]
            if not any(
                hint and _company_names_match(expected_company_marker, hint)
                for hint in company_hints
            ):
                continue

        normalized_leads.append(normalized)
    return normalized_leads


def _parse_money_token(value: str | None) -> int | None:
    if not value:
        return None
    raw = value.strip().lower().replace(",", "").replace("$", "")
    multiplier = 1
    if raw.endswith("k"):
        multiplier = 1_000
        raw = raw[:-1]
    elif raw.endswith("m"):
        multiplier = 1_000_000
        raw = raw[:-1]
    try:
        return int(float(raw) * multiplier)
    except ValueError:
        return None


def _money_token_looks_salary(value: str | None) -> bool:
    if not value:
        return False
    raw = value.strip().lower()
    if "$" in raw or raw.endswith(("k", "m")):
        return True
    parsed = _parse_money_token(raw)
    return parsed is not None and parsed >= 30_000


def _extract_salary_hint(text: str) -> tuple[int | None, int | None, str | None]:
    range_pattern = re.compile(
        r"(?P<min>\$?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?)\s*(?:-|to)\s*(?P<max>\$?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?)",
    )
    for range_match in range_pattern.finditer(text):
        if not _salary_match_uses_supported_currency(text, range_match.start("min"), range_match.end("max")):
            continue
        min_token = range_match.group("min")
        max_token = range_match.group("max")
        if _money_token_looks_salary(min_token) and _money_token_looks_salary(max_token):
            min_value = _parse_money_token(min_token)
            max_value = _parse_money_token(max_token)
            return min_value, max_value, range_match.group(0).replace("  ", " ").strip()

    single_pattern = re.compile(
        r"(?:base salary|salary|compensation|pay range)[^$]{0,80}(?P<value>\$?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?)",
        re.I,
    )
    for single_match in single_pattern.finditer(text):
        if not _salary_match_uses_supported_currency(text, single_match.start("value"), single_match.end("value")):
            continue
        token = single_match.group("value")
        if _money_token_looks_salary(token):
            value = _parse_money_token(token)
            return value, value, token.strip()
    return None, None, None


def _hydrate_salary_hint_values(
    min_value: int | None,
    max_value: int | None,
    salary_text: str | None,
    *fallback_texts: str | None,
) -> tuple[int | None, int | None, str | None]:
    if min_value is not None or max_value is not None:
        return min_value, max_value, salary_text
    for candidate_text in (salary_text, *fallback_texts):
        parsed_min, parsed_max, parsed_text = _extract_salary_hint(candidate_text or "")
        if parsed_min is not None or parsed_max is not None:
            return parsed_min, parsed_max, salary_text or parsed_text
    return min_value, max_value, salary_text


NON_USD_CURRENCY_MARKERS = (
    "ca$",
    "c$",
    "cad",
    "aud",
    "a$",
    "nzd",
    "nz$",
    "eur",
    "€",
    "gbp",
    "£",
    "chf",
    "jpy",
    "¥",
    "inr",
    "₹",
)


def _salary_match_uses_supported_currency(text: str, start: int, end: int) -> bool:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start > 0 and re.match(r"[A-Za-z0-9$€£¥₹]", text[start - 1]):
        return False
    if end < len(text) and re.match(r"[A-Za-z0-9]", text[end]):
        return False
    snippet = text[max(0, start - 8) : min(len(text), end + 8)].lower()
    return not any(marker in snippet for marker in NON_USD_CURRENCY_MARKERS)


def _normalize_remote_region_hint(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^a-z0-9 ,/&-]", " ", value.lower())
    normalized = re.sub(r"\b(?:the|state of|area|metro|metropolitan)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,-/")
    if not normalized:
        return None
    if normalized in {"united states", "united states of america", "usa", "us", "u s", "worldwide", "global"}:
        return None
    if normalized in {"north america", "north american"}:
        return None
    return normalized


def _join_remote_restriction_context(*parts: str | None) -> str:
    return ". ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def _extract_geo_limited_remote_region(text: str | None) -> str | None:
    normalized_text = " ".join(str(text or "").split()).lower()
    if "remote" not in normalized_text:
        return None
    patterns = (
        r"\bonly considering candidates who reside in (?P<region>[a-z][a-z ,/&-]{1,80}?)(?:[.;)]|$)",
        r"\bmust (?:reside|live|be based|be located|work) in (?P<region>[a-z][a-z ,/&-]{1,80}?)(?:[.;)]|$)",
        r"\bremote (?:within|from|in) (?P<region>[a-z][a-z ,/&-]{1,80}?)(?:[.;)]|$)",
        r"\b(?:100%\s*)?remote\s*[-\u2013\u2014/,]\s*(?P<region>[a-z][a-z ,/&-]{1,80}?)(?:\bonly\b|[.;)]|$)",
        r"\bcandidates must be located in (?P<region>[a-z][a-z ,/&-]{1,80}?)(?:[.;)]|$)",
        r"\b(?P<region>california|new york|washington|texas|illinois|massachusetts|florida|san francisco(?: bay)?|new york city|los angeles|seattle|boston|chicago)\s+only\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized_text)
        if not match:
            continue
        region = _normalize_remote_region_hint(match.group("region"))
        if region:
            return region
    return None


def _extract_posted_hint(text: str) -> str | None:
    match = re.search(r"(today|yesterday|\d+\s+(?:day|days|week|weeks)\s+ago|posted this week)", text, re.I)
    if match:
        return match.group(1)
    absolute_match = re.search(
        r"\b(?P<value>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s+\d{4})\b",
        text,
        re.I,
    )
    if absolute_match:
        raw = absolute_match.group("value")
        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                return datetime.strptime(raw, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _decode_search_result_url(url: str) -> str:
    parsed = urlparse(url)
    if "google.com" in parsed.netloc and parsed.path == "/url":
        encoded = parse_qs(parsed.query).get("q") or parse_qs(parsed.query).get("url")
        if encoded:
            return unquote(encoded[0])
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        encoded = parse_qs(parsed.query).get("uddg")
        if encoded:
            return unquote(encoded[0])
    if "bing.com" in parsed.netloc and parsed.path.startswith("/ck/a"):
        encoded = parse_qs(parsed.query).get("u")
        if encoded:
            raw = encoded[0]
            if raw.startswith("a1"):
                token = raw[2:]
                token += "=" * (-len(token) % 4)
                try:
                    return base64.b64decode(token).decode()
                except Exception:
                    return url
    if "search.yahoo.com" in parsed.netloc or "r.search.yahoo.com" in parsed.netloc:
        match = re.search(r"/RU=([^/]+)/RK=", parsed.path)
        if match:
            return unquote(match.group(1))
    return url


def _company_hint_from_url(url: str) -> str:
    def host_company_hint(host: str) -> str | None:
        labels = [label for label in host.lower().split(".") if label]
        if not labels:
            return None
        ats_vendor_labels = {
            "icims",
            "greenhouse",
            "lever",
            "ashbyhq",
            "smartrecruiters",
            "recruitee",
            "tellent",
            "jobscore",
            "jobvite",
            "workable",
            "bamboohr",
            "dayforcehcm",
            "paylocity",
            "adp",
            "myworkdayjobs",
            "workday",
        }

        def clean_label(label: str) -> str:
            cleaned = re.sub(
                r"^(?:[a-z]{2,3})?(?:careers|career|jobs|job|apply|join|workat|recruiting)[-_]?",
                "",
                label,
            )
            cleaned = re.sub(r"^(?:www)[-_]?", "", cleaned)
            return cleaned.strip("-_")

        candidate_labels = list(labels[:-1]) if len(labels) > 1 else list(labels)
        for label in reversed(candidate_labels):
            cleaned = clean_label(label)
            if not cleaned or cleaned in GENERIC_HOST_SUBDOMAIN_PREFIXES or cleaned in ats_vendor_labels:
                continue
            normalized = cleaned.replace("-", " ").replace("_", " ").strip()
            if normalized:
                return normalized.title()

        if len(labels) >= 3 and len(labels[-1]) == 2 and len(labels[-2]) <= 3:
            candidate = labels[-3]
        elif len(labels) >= 2:
            candidate = labels[-2]
        else:
            candidate = labels[0]
        normalized = clean_label(candidate).replace("-", " ").replace("_", " ").strip()
        return normalized.title() if normalized else None

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path_segments = [segment for segment in parsed.path.split("/") if segment]
    if "builtin" in host and len(path_segments) >= 2 and path_segments[0] == "company":
        return path_segments[1].replace("-", " ").title()
    if "dynamitejobs.com" in host and len(path_segments) >= 2 and path_segments[0] == "company":
        return path_segments[1].replace("-", " ").title()
    if "ycombinator.com" in host and len(path_segments) >= 3 and path_segments[0] == "companies":
        return path_segments[1].replace("-", " ").title()
    if "workatastartup.com" in host and len(path_segments) >= 2 and path_segments[0] in {"company", "companies"}:
        return path_segments[1].replace("-", " ").title()
    if "wellfound.com" in host and len(path_segments) >= 2 and path_segments[0] in {"company", "companies", "organization", "organizations"}:
        return path_segments[1].replace("-", " ").title()
    if "remoterocketship.com" in host and len(path_segments) >= 2 and path_segments[0] == "company":
        return path_segments[1].replace("-", " ").title()
    if "getro.com" in host and len(path_segments) >= 2 and path_segments[0] == "companies":
        return path_segments[1].replace("-", " ").title()
    if "jobscore.com" in host:
        if len(path_segments) >= 2 and path_segments[0] == "careers":
            return path_segments[1].replace("-", " ").title()
        host_prefix = host.split(".jobscore.com", 1)[0]
        if host_prefix and host_prefix != host:
            return host_prefix.replace("-", " ").title()
    if "careers.tellent.com" in host and len(path_segments) >= 2 and path_segments[0] == "o":
        return path_segments[1].replace("-", " ").title()
    if "myworkdayjobs.com" in host:
        host_prefix = host.split(".wd", 1)[0]
        host_candidate = host_prefix.replace("-", " ").title() if host_prefix else ""
        if (
            len(path_segments) >= 2
            and re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", path_segments[0], re.I)
            and path_segments[1].lower() not in {"job", "jobs", "ext"}
        ):
            candidate = path_segments[1].replace("_", " ").replace("-", " ").title()
            if (
                not _is_weak_company_hint(candidate)
                and not re.search(r"(careers?|jobs?)$", path_segments[1], re.I)
            ):
                return candidate
        if host_candidate:
            return host_candidate
    if "ats.rippling.com" in host and path_segments:
        candidate = path_segments[0].replace("_", " ").replace("-", " ").title()
        if not _is_weak_company_hint(candidate):
            return candidate
    if "portal.dynamicsats.com" in host:
        return ""
    if "jobs.lever.co" in host and path_segments:
        return path_segments[0].replace("-", " ").title()
    if "ashbyhq.com" in host and path_segments:
        return path_segments[0].replace("-", " ").title()
    if "greenhouse.io" in host and path_segments:
        return path_segments[0].replace("-", " ").title()
    host_hint = host_company_hint(host)
    if not _is_weak_company_hint(host_hint):
        return str(host_hint)
    if path_segments:
        return path_segments[0].replace("-", " ").title()
    return (host.split(".")[0] if host else "Unknown").replace("-", " ").title()


def _url_has_strong_expected_company_hint(url: str, expected_company_name: str | None) -> bool:
    if not expected_company_name or _is_weak_company_hint(expected_company_name):
        return False
    company_hint = _company_hint_from_url(url)
    if _is_weak_company_hint(company_hint):
        return False
    return _company_names_match(expected_company_name, company_hint)


def _extract_role_company_from_title(title: str, url: str) -> tuple[str, str]:
    cleaned = unescape(title).strip()
    cleaned = re.sub(r"^\s*Job Application for\s*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s*\|\s*LinkedIn.*$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s*\|\s*Glassdoor.*$", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"^\S+\s+\S+\s+›\s+", "", cleaned).strip()

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path_segments = [segment for segment in parsed.path.split("/") if segment]
    if "dynamitejobs.com" in host and len(path_segments) >= 4 and path_segments[0] == "company":
        company = path_segments[1].replace("-", " ").title()
        role = path_segments[3].replace("-", " ").title()
        return company, role

    breadcrumb_parts = [part.strip() for part in re.split(r"\s+›\s+", cleaned) if part.strip()]
    if len(breadcrumb_parts) >= 2:
        company_hint = _company_hint_from_url(url)
        for part in reversed(breadcrumb_parts):
            lowered = part.lower()
            if "product manager" in lowered:
                role = re.sub(r"^[^A-Za-z0-9]*(?:jobs?\s*\|\s*)?", "", part).strip()
                role = re.sub(r"^[a-f0-9-]{8,}\s+", "", role, flags=re.I).strip()
                return company_hint, role

    greenhouse_match = re.search(r"(?P<role>.+?)\s+at\s+(?P<company>.+)$", cleaned, re.I)
    if "job application for" in title.lower() and greenhouse_match:
        return greenhouse_match.group("company").strip(), greenhouse_match.group("role").strip()

    jobscore_match = re.search(
        r"Share the\s+(?P<role>.+?)\s+open at\s+(?P<company>.+?)\s+in\s+(?P<location>.+?)(?:, powered by JobScore|[.])",
        cleaned,
        re.I,
    )
    if jobscore_match:
        return jobscore_match.group("company").strip(), jobscore_match.group("role").strip()

    ats_at_match = re.search(r"(?P<role>[A-Za-z0-9,&()/'’.\- ]*product manager[^@|]*)\s+@\s+(?P<company>[^|]+)$", cleaned, re.I)
    if ats_at_match:
        return ats_at_match.group("company").strip(), ats_at_match.group("role").strip()

    separators = [" - ", " | ", " at "]
    for separator in separators:
        if separator not in cleaned:
            continue
        left, right = cleaned.split(separator, 1)
        left = left.strip()
        right = right.strip()
        left_is_role = "product manager" in left.lower()
        right_is_role = "product manager" in right.lower()
        if left_is_role and not right_is_role:
            return right, left
        if right_is_role and not left_is_role:
            return left, right
        return left, right

    company = _company_hint_from_url(url)
    return company, cleaned


def _lead_confidence(lead: JobLead) -> float:
    score = 0.0
    if lead.direct_job_url and _is_allowed_direct_job_url(lead.direct_job_url):
        score += 0.45
    elif (
        not lead.direct_job_url
        and lead.source_type in {"direct_ats", "company_site"}
        and _is_allowed_direct_job_url(lead.source_url)
        and _lead_is_replay_source_trustworthy(lead)
    ):
        score += 0.45
    if lead.source_type in {"direct_ats", "company_site"}:
        score += 0.2
    elif _is_supported_discovery_source_url(lead.source_url):
        score += 0.1
    if "product manager" in lead.role_title.lower():
        score += 0.15
    if _lead_is_ai_related_product_manager(lead):
        score += 0.1
    if lead.is_remote_hint:
        score += 0.05
    if lead.posted_date_hint:
        score += 0.05
    return max(0.0, min(1.0, score))


def _lead_needs_local_cleanup(lead: JobLead) -> bool:
    if lead.source_type in {"linkedin", "builtin", "other"}:
        return True
    if not lead.direct_job_url:
        return True
    normalized_direct_url = _normalize_direct_job_url(lead.direct_job_url)
    if not _candidate_direct_job_url_is_trustworthy(normalized_direct_url, lead):
        return True
    return _looks_like_generic_job_url(normalized_direct_url)


def _ollama_inline_refinement_enabled(settings: Settings) -> bool:
    return settings.llm_provider == "ollama" and settings.ollama_inline_lead_refinement_enabled


def _should_force_ollama_refinement_sample(
    settings: Settings,
    *,
    sample_size: int,
    average_confidence: float,
    cleanup_signal_count: int,
    low_trust_source_count: int,
    trustworthy_direct_url_count: int,
    query: str | None = None,
) -> bool:
    if not _ollama_inline_refinement_enabled(settings) or sample_size < 2:
        return False
    if cleanup_signal_count <= 0:
        return (
            (
                sample_size >= 5
                and _is_clean_high_confidence_direct_bundle(
                    candidate_pool_count=sample_size,
                    average_confidence=average_confidence,
                    cleanup_signal_count=cleanup_signal_count,
                    low_trust_source_count=low_trust_source_count,
                    trustworthy_direct_url_count=trustworthy_direct_url_count,
                    min_candidate_pool_count=5,
                    min_trustworthy_direct_url_count=sample_size,
                )
            )
            or (
                query is not None
                and sample_size >= 5
                and _is_trusted_company_careers_bundle(
                    query=query,
                    candidate_pool_count=sample_size,
                    average_confidence=average_confidence,
                    cleanup_signal_count=cleanup_signal_count,
                    low_trust_source_count=low_trust_source_count,
                    trustworthy_direct_url_count=trustworthy_direct_url_count,
                    min_candidate_pool_count=5,
                )
            )
        )
    if cleanup_signal_count > 0 or low_trust_source_count > 0:
        return True
    if trustworthy_direct_url_count < sample_size:
        return True
    return average_confidence < min(settings.local_confidence_threshold + 0.1, 0.9)


def _is_trusted_company_careers_bundle(
    *,
    query: str,
    candidate_pool_count: int,
    average_confidence: float,
    cleanup_signal_count: int,
    low_trust_source_count: int,
    trustworthy_direct_url_count: int,
    min_candidate_pool_count: int = 10,
) -> bool:
    return (
        candidate_pool_count >= min_candidate_pool_count
        and cleanup_signal_count == 0
        and low_trust_source_count == 0
        and trustworthy_direct_url_count >= 5
        and average_confidence >= 0.9
        and "company careers" in query.lower()
    )


def _is_clean_high_confidence_direct_bundle(
    *,
    candidate_pool_count: int,
    average_confidence: float,
    cleanup_signal_count: int,
    low_trust_source_count: int,
    trustworthy_direct_url_count: int,
    min_candidate_pool_count: int = 10,
    min_trustworthy_direct_url_count: int = 5,
) -> bool:
    return (
        candidate_pool_count >= min_candidate_pool_count
        and cleanup_signal_count == 0
        and low_trust_source_count == 0
        and trustworthy_direct_url_count >= min_trustworthy_direct_url_count
        and average_confidence >= 0.9
    )


def _ollama_refinement_mode_for_local_leads(
    settings: Settings,
    *,
    query: str,
    candidate_pool_count: int,
    average_confidence: float,
    cleanup_signal_count: int,
    low_trust_source_count: int,
    trustworthy_direct_url_count: int,
) -> str | None:
    single_high_confidence_direct_candidate = (
        candidate_pool_count == 1
        and cleanup_signal_count == 0
        and low_trust_source_count == 0
        and trustworthy_direct_url_count >= 1
        and average_confidence >= 0.9
    )
    if not _ollama_inline_refinement_enabled(settings) or (
        candidate_pool_count < 3 and not single_high_confidence_direct_candidate
    ):
        return None
    if average_confidence < min(settings.local_confidence_threshold + 0.1, 0.9):
        return "low_confidence"
    if trustworthy_direct_url_count == 0:
        return "no_trustworthy_direct_urls"
    if single_high_confidence_direct_candidate:
        return "trusted_direct_bundle"
    if (
        _query_targets_startup_ecosystem(query)
        and candidate_pool_count >= 5
        and cleanup_signal_count >= 1
        and low_trust_source_count >= 1
        and average_confidence < 0.99
    ):
        return "startup_board_bundle"
    if (
        candidate_pool_count >= 10
        and cleanup_signal_count == 0
        and low_trust_source_count == 0
        and (
            (
                trustworthy_direct_url_count >= 4
                and average_confidence >= 0.9
                and (
                    _query_is_broad_generic(query)
                    or _is_trusted_company_careers_bundle(
                        query=query,
                        candidate_pool_count=candidate_pool_count,
                        average_confidence=average_confidence,
                        cleanup_signal_count=cleanup_signal_count,
                        low_trust_source_count=low_trust_source_count,
                        trustworthy_direct_url_count=trustworthy_direct_url_count,
                        min_candidate_pool_count=10,
                    )
                )
            )
            or (
                trustworthy_direct_url_count >= 5
                and average_confidence >= 0.86
                and _query_targets_startup_ecosystem(query)
                and _query_uses_structured_source_hint(query)
            )
        )
    ):
        return "trusted_direct_bundle"
    if cleanup_signal_count >= 2 or low_trust_source_count >= 3:
        return "high_noise"
    if (
        3 <= candidate_pool_count <= 8
        and cleanup_signal_count >= 1
        and low_trust_source_count >= 1
        and average_confidence < 0.985
    ):
        return "borderline_bundle"
    if (
        candidate_pool_count >= 8
        and cleanup_signal_count >= 1
        and low_trust_source_count >= 1
        and average_confidence < 0.97
        and _query_is_broad_generic(query)
    ):
        return "broad_generic"
    return None


def _should_refine_local_leads_with_ollama(
    settings: Settings,
    *,
    query: str,
    candidate_pool_count: int,
    average_confidence: float,
    cleanup_signal_count: int,
    low_trust_source_count: int,
    trustworthy_direct_url_count: int,
) -> bool:
    return _ollama_refinement_mode_for_local_leads(
        settings,
        query=query,
        candidate_pool_count=candidate_pool_count,
        average_confidence=average_confidence,
        cleanup_signal_count=cleanup_signal_count,
        low_trust_source_count=low_trust_source_count,
        trustworthy_direct_url_count=trustworthy_direct_url_count,
    ) is not None


def _is_duckduckgo_anomaly_page(status_code: int, html: str) -> bool:
    lowered = html.lower()
    return status_code >= 400 or "anomaly-modal" in lowered or "bots use duckduckgo too" in lowered


def _is_google_interstitial_page(status_code: int, html: str) -> bool:
    lowered = html.lower()
    return (
        status_code >= 400
        or "httpservice/retry/enablejs" in lowered
        or ("<title>google search</title>" in lowered and "please click" in lowered and "redirected" in lowered)
    )


def _extract_startpage_search_results(html: str, *, max_results: int) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str, str]] = []
    for node in soup.select(".result"):
        link = node.select_one("a.result-link")
        if link is None:
            continue
        href = link.get("href")
        if not href:
            continue
        snippet = ""
        snippet_candidates = [p.get_text(" ", strip=True) for p in node.select("p")]
        snippet_candidates = [candidate for candidate in snippet_candidates if len(candidate) >= 24]
        if snippet_candidates:
            snippet = max(snippet_candidates, key=len)
        title = link.get_text(" ", strip=True)
        if not title:
            continue
        results.append((_decode_search_result_url(href), title, snippet))
        if len(results) >= max_results:
            break
    return results


def _extract_yahoo_search_results(html: str, *, max_results: int) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str, str]] = []
    for node in soup.select(".algo"):
        link = node.select_one(".compTitle a")
        if link is None:
            continue
        href = link.get("href")
        if not href:
            continue
        title = link.get_text(" ", strip=True)
        if not title:
            continue
        snippet_el = node.select_one(".compText p")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append((_decode_search_result_url(href), title, snippet))
        if len(results) >= max_results:
            break
    return results


def _extract_mojeek_search_results(html: str, *, max_results: int) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str, str]] = []
    for node in soup.select(".results-standard li"):
        link = node.select_one("h2 a.title")
        if link is None:
            continue
        href = link.get("href")
        if not href:
            continue
        title = link.get_text(" ", strip=True)
        if not title:
            continue
        snippet_el = node.select_one("p.s")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append((_decode_search_result_url(href), title, snippet))
        if len(results) >= max_results:
            break
    return results


def _extract_google_search_results(html: str, *, max_results: int) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()
    for link in soup.select('a[href^="/url?"], a[href^="http"]'):
        href = link.get("href")
        if not href:
            continue
        absolute_href = urljoin("https://www.google.com", href)
        decoded = _decode_search_result_url(absolute_href)
        if not decoded.startswith(("http://", "https://")):
            continue
        host = (urlparse(decoded).netloc or "").lower()
        if not host or "google.com" in host:
            continue
        if decoded in seen_urls:
            continue
        title = link.get_text(" ", strip=True)
        if not title:
            continue
        seen_urls.add(decoded)
        results.append((decoded, title, ""))
        if len(results) >= max_results:
            break
    return results


async def _google_search(query: str, *, max_results: int) -> list[tuple[str, str, str]]:
    global GOOGLE_HTTP_SEARCH_AVAILABLE
    if GOOGLE_HTTP_SEARCH_AVAILABLE is False:
        raise LocalSearchBackendBlockedError("Google returned a JS interstitial on this machine and was disabled for this run.")

    search_url = "https://www.google.com/search"
    params = {"q": query, "hl": "en", "gl": "us", "num": max(10, max_results)}
    async with httpx.AsyncClient(
        timeout=10.0,
        follow_redirects=True,
        headers=SEARCH_ENGINE_HEADERS,
    ) as client:
        response = await client.get(search_url, params=params)

    if _is_google_interstitial_page(response.status_code, response.text):
        GOOGLE_HTTP_SEARCH_AVAILABLE = False
        raise LocalSearchBackendBlockedError("Google returned an enable-JS interstitial instead of results.")

    response.raise_for_status()
    GOOGLE_HTTP_SEARCH_AVAILABLE = True
    return _extract_google_search_results(response.text, max_results=max_results)


async def _duckduckgo_search(query: str, *, max_results: int) -> list[tuple[str, str, str]]:
    search_url = "https://duckduckgo.com/html/"
    params = {"q": query}
    async with httpx.AsyncClient(
        timeout=8.0,
        follow_redirects=True,
        headers=SEARCH_ENGINE_HEADERS,
    ) as client:
        response = await client.get(search_url, params=params)

    if _is_duckduckgo_anomaly_page(response.status_code, response.text):
        raise LocalSearchBackendBlockedError("DuckDuckGo returned an anti-bot challenge page instead of results.")

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[tuple[str, str, str]] = []
    for node in soup.select(".result"):
        link = node.select_one("a.result__a")
        snippet_el = node.select_one(".result__snippet")
        if link is None:
            continue
        href = link.get("href")
        if not href:
            continue
        decoded = _decode_search_result_url(href)
        title = link.get_text(" ", strip=True)
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append((decoded, title, snippet))
        if len(results) >= max_results:
            break
    return results


async def _bing_search(query: str, *, max_results: int) -> list[tuple[str, str, str]]:
    search_url = "https://www.bing.com/search"
    params = {"q": query, "setlang": "en-US"}
    async with httpx.AsyncClient(
        timeout=12.0,
        follow_redirects=True,
        headers=SEARCH_ENGINE_HEADERS,
    ) as client:
        response = await client.get(search_url, params=params)

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    results: list[tuple[str, str, str]] = []
    for node in soup.select("li.b_algo"):
        link = node.select_one("h2 a")
        snippet_el = node.select_one(".b_caption p")
        if link is None:
            continue
        href = link.get("href")
        if not href:
            continue
        decoded = _decode_search_result_url(href)
        title = link.get_text(" ", strip=True)
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        results.append((decoded, title, snippet))
        if len(results) >= max_results:
            break
    return results


async def _yahoo_search(query: str, *, max_results: int) -> list[tuple[str, str, str]]:
    search_url = "https://search.yahoo.com/search"
    params = {"p": query}
    async with httpx.AsyncClient(
        timeout=12.0,
        follow_redirects=True,
        headers=SEARCH_ENGINE_HEADERS,
    ) as client:
        response = await client.get(search_url, params=params)

    response.raise_for_status()
    return _extract_yahoo_search_results(response.text, max_results=max_results)


async def _startpage_search(query: str, *, max_results: int) -> list[tuple[str, str, str]]:
    search_url = "https://www.startpage.com/sp/search"
    params = {"query": query}
    async with httpx.AsyncClient(
        timeout=12.0,
        follow_redirects=True,
        headers=SEARCH_ENGINE_HEADERS,
    ) as client:
        response = await client.get(search_url, params=params)

    response.raise_for_status()
    return _extract_startpage_search_results(response.text, max_results=max_results)


async def _mojeek_search(query: str, *, max_results: int) -> list[tuple[str, str, str]]:
    search_url = "https://www.mojeek.com/search"
    params = {"q": query}
    async with httpx.AsyncClient(
        timeout=12.0,
        follow_redirects=True,
        headers=SEARCH_ENGINE_HEADERS,
    ) as client:
        response = await client.get(search_url, params=params)

    response.raise_for_status()
    return _extract_mojeek_search_results(response.text, max_results=max_results)


async def _search_query_across_backends(query: str, *, max_results: int) -> list[tuple[str, str, str]]:
    cache_key = (query, max_results)
    cached = SEARCH_QUERY_RESULT_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    async def _run_backend_group(backends: list) -> tuple[list[tuple[str, str, str]], Exception | None]:
        merged_results: list[tuple[str, str, str]] = []
        seen_urls: set[str] = set()
        last_error: Exception | None = None
        tasks = [
            asyncio.create_task(backend(query, max_results=max(6, min(max_results, 8))))
            for backend in backends
        ]
        try:
            for task in asyncio.as_completed(tasks):
                try:
                    search_results = await task
                except Exception as exc:
                    last_error = exc
                    continue

                for url, title, snippet in search_results:
                    normalized_url = _normalize_direct_job_url(url)
                    if normalized_url in seen_urls:
                        continue
                    seen_urls.add(normalized_url)
                    merged_results.append((normalized_url, title, snippet))
                    if len(merged_results) >= max_results:
                        return merged_results, last_error
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        return merged_results, last_error

    primary_backends = [_bing_search, _yahoo_search, _startpage_search]
    if GOOGLE_HTTP_SEARCH_AVAILABLE is not False:
        primary_backends.insert(0, _google_search)
    secondary_backends = [_duckduckgo_search] if "site:" in query.lower() else [_mojeek_search, _duckduckgo_search]

    merged_results, last_error = await _run_backend_group(primary_backends)
    if len(merged_results) < max_results and secondary_backends:
        secondary_results, secondary_error = await _run_backend_group(secondary_backends)
        seen_urls = {url for url, _, _ in merged_results}
        for url, title, snippet in secondary_results:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            merged_results.append((url, title, snippet))
            if len(merged_results) >= max_results:
                break
        if secondary_error is not None:
            last_error = secondary_error

    if not merged_results and last_error is not None:
        raise last_error

    SEARCH_QUERY_RESULT_CACHE[cache_key] = list(merged_results)
    return merged_results


def _score_company_site_search_result(url: str, title: str, lead: JobLead) -> int:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    full = f"{host}{path}?{(parsed.query or '').lower()}"
    if not host:
        return -100
    if any(fragment in host for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
        return -100
    if any(fragment in host for fragment in COMPANY_SITE_JUNK_HOST_FRAGMENTS):
        return -100
    if any(hint in full for hint in GENERIC_CAREERS_INFO_HINTS):
        return -50

    haystack = f"{host} {path} {title.lower()}"
    score = 0
    company_matches = 0
    for token in _company_token_candidates(lead.company_name):
        if token in haystack:
            score += 3
            company_matches += 1

    careers_hub = _looks_like_careers_hub_url(url)
    company_homepage = _looks_like_company_homepage_url(url)
    if careers_hub:
        score += 5
    elif company_homepage and company_matches >= 1:
        score += 4
    direct_ats = any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS)
    if direct_ats:
        score += 4 if _looks_like_generic_job_url(url) else 8
    if any(hint in haystack for hint in ("career", "careers", "jobs", "join us", "join-us")):
        score += 2
    role_score = _role_match_score(lead.role_title, haystack)
    if direct_ats:
        score += role_score
    else:
        score += max(role_score, 0)
    if lead.role_title.lower() in haystack:
        score += 3
    if direct_ats and role_score <= 0:
        score -= 12
    return score


def _extract_followup_resolution_urls(base_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    base_host = (urlparse(base_url).netloc or "").lower()
    candidates: dict[str, int] = {}

    def maybe_add(raw_url: str | None, anchor_text: str = "") -> None:
        if not raw_url:
            return
        cleaned_url = unescape(raw_url.strip()).replace("\\/", "/")
        cleaned_url = cleaned_url.split('"', 1)[0].split("'", 1)[0]
        if any(token in cleaned_url for token in ("&quot", "&gt", "&lt", "\\", "<", ">")):
            return
        absolute_url = _normalize_direct_job_url(urljoin(base_url, cleaned_url))
        parsed = urlparse(absolute_url)
        host = (parsed.netloc or "").lower()
        if not host:
            return
        if any(fragment in host for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
            return
        if any(fragment in host for fragment in COMPANY_SITE_JUNK_HOST_FRAGMENTS):
            return
        full = f"{host}{parsed.path.lower()}?{(parsed.query or '').lower()}"
        if any(hint in full or hint in anchor_text.lower() for hint in GENERIC_CAREERS_INFO_HINTS):
            return

        score = 0
        haystack = f"{absolute_url} {anchor_text}".lower()
        if any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
            if not _looks_like_generic_job_url(absolute_url):
                return
            score += 8
        elif _looks_like_careers_hub_url(absolute_url):
            score += 5
        elif host == base_host and any(hint in f"{parsed.path.lower()}/" for hint in CAREERS_HUB_HINTS):
            score += 4
        else:
            return

        if any(token in haystack for token in ("career", "careers", "jobs", "join us", "join-us", "work with us")):
            score += 2
        previous = candidates.get(absolute_url)
        if previous is None or score > previous:
            candidates[absolute_url] = score

    for link in soup.select("a[href]"):
        maybe_add(link.get("href"), link.get_text(" ", strip=True))

    for raw_url in re.findall(r"https?://[^\s\"'<>]+", html):
        maybe_add(raw_url)

    for board_url in extract_embedded_board_urls(base_url, html):
        maybe_add(board_url, "Embedded ATS board")

    for careers_url in extract_careers_page_urls(base_url, html):
        maybe_add(careers_url, "Careers")

    for homepage_url in extract_company_homepage_urls(base_url, html):
        maybe_add(homepage_url, "Company website")
        for careers_candidate in default_careers_candidate_urls(homepage_url):
            maybe_add(careers_candidate, "Careers")

    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    return [url for url, _ in ranked[:8]]


async def _search_company_resolution_candidates(lead: JobLead) -> list[str]:
    company_key = lead.company_name.strip().lower()
    cached = COMPANY_RESOLUTION_URL_CACHE.get(company_key)
    if cached is not None:
        return cached

    ats_domains = list(
        dict.fromkeys(domain for batch in LOCAL_SEARCH_DIRECT_ATS_DOMAIN_BATCHES for domain in batch)
    )[:6]
    normalized_role_queries = _normalize_role_title_to_focus_queries(lead.role_title)
    role_variants = _dedupe_queries([lead.role_title, *normalized_role_queries])

    direct_ats_queries: list[str] = []
    for variant in role_variants[:3]:
        for domain in ats_domains[:4]:
            direct_ats_queries.append(f'site:{domain} "{lead.company_name}" "{variant}"')
    if "product manager" in lead.role_title.lower():
        for domain in ats_domains[:3]:
            direct_ats_queries.append(f'site:{domain} "{lead.company_name}" "product manager"')
        for variant in normalized_role_queries[:2]:
            for domain in ats_domains[:2]:
                direct_ats_queries.append(f'site:{domain} "{lead.company_name}" {variant}')

    queries = [
        *direct_ats_queries,
        f"\"{lead.company_name}\" careers",
        f"\"{lead.company_name}\" jobs",
        f"\"{lead.company_name}\" \"{lead.role_title}\"",
    ]
    ranked_results: dict[str, int] = {}
    async def collect_queries(query_list: list[str], *, batch_size: int, stop_after_direct_hits: int) -> int:
        direct_hits = 0
        unique_queries = list(dict.fromkeys(query_list))
        for start in range(0, len(unique_queries), batch_size):
            batch = unique_queries[start : start + batch_size]
            tasks = {asyncio.create_task(_search_query_across_backends(query, max_results=8)): query for query in batch}
            try:
                for task in asyncio.as_completed(tasks):
                    try:
                        results = await task
                    except Exception:
                        continue
                    for url, title, _snippet in results:
                        score = _score_company_site_search_result(url, title, lead)
                        if score <= 0:
                            continue
                        existing = ranked_results.get(url)
                        if existing is None or score > existing:
                            ranked_results[url] = score
                        if any(fragment in (urlparse(url).netloc or "").lower() for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
                            direct_hits += 1
                    if direct_hits >= stop_after_direct_hits:
                        return direct_hits
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        return direct_hits

    fast_queries = direct_ats_queries[:6]
    remaining_queries = [query for query in queries if query not in fast_queries]
    direct_hits = await collect_queries(fast_queries, batch_size=4, stop_after_direct_hits=2)
    if direct_hits < 2:
        direct_hits += await collect_queries(remaining_queries, batch_size=2, stop_after_direct_hits=4)

    if not ranked_results:
        fallback_queries = [
            f"\"{lead.company_name}\"",
            lead.company_name,
        ]
        await collect_queries(fallback_queries, batch_size=2, stop_after_direct_hits=1)

    resolved = [url for url, _ in sorted(ranked_results.items(), key=lambda item: item[1], reverse=True)[:6]]
    COMPANY_RESOLUTION_URL_CACHE[company_key] = resolved
    return resolved


def _board_resolution_candidate_score(expected_lead: JobLead, candidate_lead: JobLead) -> int:
    score = min(
        _role_match_score(expected_lead.role_title, candidate_lead.role_title),
        _role_match_score(candidate_lead.role_title, expected_lead.role_title),
    )
    if _role_titles_align(expected_lead.role_title, candidate_lead.role_title):
        score += 16
    if _company_names_match(expected_lead.company_name, candidate_lead.company_name):
        score += 4
    if expected_lead.is_remote_hint is True and candidate_lead.is_remote_hint is True:
        score += 2
    if expected_lead.is_remote_hint is True and candidate_lead.is_remote_hint is False:
        score -= 4
    if candidate_lead.direct_job_url and _candidate_direct_job_url_is_trustworthy(candidate_lead.direct_job_url, expected_lead):
        score += 3
    return score


async def _resolve_supported_board_job_url_from_lead(lead: JobLead) -> str | None:
    board_candidates: list[tuple[str, str]] = []
    seen_board_keys: set[tuple[str, str]] = set()
    for raw_url in (lead.direct_job_url, lead.source_url):
        normalized_url = _normalize_direct_job_url(raw_url or "")
        if not normalized_url:
            continue
        board_identifier = str(_extract_company_board_identifier(normalized_url) or board_identifier_from_url(normalized_url) or "").strip()
        if not board_identifier:
            continue
        board_root = str(infer_careers_root(normalized_url) or normalized_url).strip()
        board_key = (board_identifier, board_root)
        if board_key in seen_board_keys:
            continue
        seen_board_keys.add(board_key)
        board_candidates.append(board_key)

    best_url: str | None = None
    best_score = 0
    for board_identifier, board_root in board_candidates:
        prefix, _, token = board_identifier.partition(":")
        board_leads: list[JobLead] = []
        try:
            if prefix == "greenhouse" and token:
                board_leads = [
                    candidate
                    for candidate in (
                        _greenhouse_board_job_to_lead(token, lead.company_name, job)
                        for job in await _fetch_greenhouse_board_jobs(token)
                    )
                    if candidate is not None
                ]
            elif prefix == "ashby" and token:
                board_leads = [
                    candidate
                    for candidate in (
                        _ashby_board_job_to_lead(token, lead.company_name, job)
                        for job in await _fetch_ashby_board_jobs(token)
                    )
                    if candidate is not None
                ]
            elif prefix == "lever" and token:
                board_leads = [
                    candidate
                    for candidate in (
                        _lever_board_job_to_lead(token, lead.company_name, job)
                        for job in await _fetch_lever_board_jobs(token)
                    )
                    if candidate is not None
                ]
            elif prefix == "smartrecruiters" and token:
                board_leads = await _smartrecruiters_board_jobs_to_leads(token, lead.company_name)
            elif prefix == "workday":
                board_leads = [
                    candidate
                    for candidate in (
                        _workday_board_job_to_lead(board_root, lead.company_name, job)
                        for job in await _fetch_workday_board_jobs(board_root)
                    )
                    if candidate is not None
                ]
        except Exception:
            continue

        for candidate_lead in board_leads:
            candidate_url = _normalize_direct_job_url(candidate_lead.direct_job_url or "")
            if not candidate_url.startswith(("http://", "https://")):
                continue
            score = _board_resolution_candidate_score(lead, candidate_lead)
            if score > best_score:
                best_score = score
                best_url = candidate_url

    if best_score < 14:
        return None
    return best_url


def _extract_greenhouse_board_token(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "greenhouse.io" not in host:
        return None
    segments = _path_segments(parsed.path)
    if not segments:
        return None
    first_segment = segments[0]
    if first_segment in {"embed", "job_board"}:
        return None
    return first_segment


async def _fetch_greenhouse_board_jobs(board_token: str) -> list[dict[str, object]]:
    cached = GREENHOUSE_BOARD_JOBS_CACHE.get(board_token)
    if cached is not None:
        return cached

    api_url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
        ) as client:
            response = await client.get(api_url)
    except httpx.RequestError:
        GREENHOUSE_BOARD_JOBS_CACHE[board_token] = []
        return []
    if response.status_code != 200:
        GREENHOUSE_BOARD_JOBS_CACHE[board_token] = []
        return []
    try:
        payload = response.json()
    except json.JSONDecodeError:
        GREENHOUSE_BOARD_JOBS_CACHE[board_token] = []
        return []
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list):
        GREENHOUSE_BOARD_JOBS_CACHE[board_token] = []
        return []
    normalized_jobs = [job for job in jobs if isinstance(job, dict)]
    GREENHOUSE_BOARD_JOBS_CACHE[board_token] = normalized_jobs
    return normalized_jobs


def _extract_ashby_board_token(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "ashbyhq.com" not in host:
        return None
    segments = _path_segments(parsed.path)
    if not segments:
        return None
    return segments[0].lower()


def _extract_lever_board_token(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "jobs.lever.co" not in host:
        return None
    segments = _path_segments(parsed.path)
    if not segments:
        return None
    return segments[0].lower()


async def _fetch_ashby_board_jobs(board_token: str) -> list[dict[str, object]]:
    cached = ASHBY_BOARD_JOBS_CACHE.get(board_token)
    if cached is not None:
        return cached

    api_url = f"https://api.ashbyhq.com/posting-api/job-board/{board_token}?includeCompensation=true"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
        ) as client:
            response = await client.get(api_url)
    except httpx.RequestError:
        ASHBY_BOARD_JOBS_CACHE[board_token] = []
        return []
    if response.status_code != 200:
        ASHBY_BOARD_JOBS_CACHE[board_token] = []
        return []
    try:
        payload = response.json()
    except json.JSONDecodeError:
        ASHBY_BOARD_JOBS_CACHE[board_token] = []
        return []
    jobs = payload.get("jobs") if isinstance(payload, dict) else None
    if not isinstance(jobs, list):
        ASHBY_BOARD_JOBS_CACHE[board_token] = []
        return []
    normalized_jobs = [job for job in jobs if isinstance(job, dict)]
    ASHBY_BOARD_JOBS_CACHE[board_token] = normalized_jobs
    return normalized_jobs


async def _fetch_lever_board_jobs(board_token: str) -> list[dict[str, object]]:
    cached = LEVER_BOARD_JOBS_CACHE.get(board_token)
    if cached is not None:
        return cached

    api_url = f"https://api.lever.co/v0/postings/{board_token}?mode=json"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
        ) as client:
            response = await client.get(api_url)
    except httpx.RequestError:
        LEVER_BOARD_JOBS_CACHE[board_token] = []
        return []
    if response.status_code != 200:
        LEVER_BOARD_JOBS_CACHE[board_token] = []
        return []
    try:
        payload = response.json()
    except json.JSONDecodeError:
        LEVER_BOARD_JOBS_CACHE[board_token] = []
        return []
    if not isinstance(payload, list):
        LEVER_BOARD_JOBS_CACHE[board_token] = []
        return []
    jobs = [job for job in payload if isinstance(job, dict)]
    LEVER_BOARD_JOBS_CACHE[board_token] = jobs
    return jobs


def _workday_board_context(board_url: str | None) -> dict[str, str] | None:
    board_root_url = workday_board_root_url(board_url)
    if not board_root_url:
        return None
    parsed = urlparse(board_root_url)
    host = (parsed.netloc or "").lower()
    if "myworkdayjobs.com" not in host:
        return None
    segments = _path_segments(parsed.path)
    if not segments:
        return None
    tenant = host.split(".wd", 1)[0]
    site = segments[-1]
    if not tenant or not site:
        return None
    return {
        "board_root_url": board_root_url,
        "api_url": f"{parsed.scheme}://{host}/wday/cxs/{tenant}/{site}/jobs",
        "host_root_url": f"{parsed.scheme}://{host}",
        "site": site,
    }


def _repair_workday_board_task_url(
    task_url: str,
    *,
    company_key: str | None,
    entries: Mapping[str, dict[str, object]],
) -> str:
    candidates = [task_url]
    if company_key:
        entry = entries.get(company_key)
        if isinstance(entry, Mapping):
            candidates.extend(str(item).strip() for item in entry.get("board_urls") or [])
            candidates.extend(str(item).strip() for item in entry.get("careers_roots") or [])
    for candidate in candidates:
        repaired = workday_board_root_url(candidate)
        if repaired:
            return repaired
    return task_url


async def _fetch_workday_board_jobs(board_url: str) -> list[dict[str, object]]:
    context = _workday_board_context(board_url)
    if context is None:
        return []
    cache_key = context["board_root_url"]
    cached = WORKDAY_BOARD_JOBS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    jobs: list[dict[str, object]] = []
    offset = 0
    limit = 20
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=20.0,
        ) as client:
            while offset < 200:
                response = await client.post(
                    context["api_url"],
                    json={
                        "appliedFacets": {},
                        "limit": limit,
                        "offset": offset,
                        "searchText": "",
                    },
                )
                if response.status_code != 200:
                    WORKDAY_BOARD_JOBS_CACHE[cache_key] = []
                    return []
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    WORKDAY_BOARD_JOBS_CACHE[cache_key] = []
                    return []
                if not isinstance(payload, dict):
                    WORKDAY_BOARD_JOBS_CACHE[cache_key] = []
                    return []
                page_jobs = payload.get("jobPostings")
                if not isinstance(page_jobs, list):
                    WORKDAY_BOARD_JOBS_CACHE[cache_key] = []
                    return []
                normalized_jobs = [job for job in page_jobs if isinstance(job, dict)]
                jobs.extend(normalized_jobs)
                total = int(payload.get("total") or 0)
                if len(normalized_jobs) < limit or (total and offset + limit >= total):
                    break
                offset += limit
    except httpx.RequestError:
        WORKDAY_BOARD_JOBS_CACHE[cache_key] = []
        return []

    WORKDAY_BOARD_JOBS_CACHE[cache_key] = jobs
    return jobs


def _workday_job_text_value(job: dict[str, object], *keys: str) -> str:
    for key in keys:
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _workday_board_job_to_lead(
    board_url: str,
    company_name: str,
    job: dict[str, object],
) -> JobLead | None:
    context = _workday_board_context(board_url)
    if context is None:
        return None
    title = _workday_job_text_value(job, "title", "name")
    if not title or not _looks_like_product_manager_title(title):
        return None

    bullet_fields = [str(item).strip() for item in job.get("bulletFields") or [] if str(item).strip()]
    location_text = _workday_job_text_value(job, "locationsText", "location")
    remote_type = _workday_job_text_value(job, "remoteType", "workplaceType")
    description_text = _workday_job_text_value(
        job,
        "description",
        "jobDescription",
        "externalDescription",
        "shortDescription",
    )
    content_text = " ".join(part for part in (title, description_text, location_text, remote_type, " ".join(bullet_fields)) if part)
    if not _is_ai_related_product_manager_text(content_text):
        return None

    external_path = _workday_job_text_value(job, "externalPath", "externalUrl", "jobUrl")
    direct_job_url = _normalize_direct_job_url(urljoin(f"{context['host_root_url']}/", external_path))
    if not direct_job_url.startswith(("http://", "https://")):
        return None

    remote_haystack = " ".join(part for part in (location_text, remote_type, description_text, " ".join(bullet_fields)) if part).lower()
    if any(token in remote_haystack for token in ("hybrid", "on-site", "onsite", "in office")):
        is_remote_hint: bool | None = False
    elif "remote" in remote_haystack:
        is_remote_hint = True
    else:
        is_remote_hint = None

    posted_date_hint = _extract_posted_hint(
        " ".join(
            part
            for part in (
                _workday_job_text_value(job, "postedOn", "postedDate"),
                " ".join(bullet_fields),
            )
            if part
        )
    )
    location_hint = _lead_location_hint_with_remote_restriction(location_text, title, description_text, remote_type, " ".join(bullet_fields))
    remote_restriction_note = _remote_restriction_note(title, location_text, description_text, remote_type, " ".join(bullet_fields))
    evidence_notes = " ".join(
        part
        for part in (
            "Discovered via official Workday board enumeration.",
            remote_restriction_note,
        )
        if part
    )
    return JobLead(
        company_name=company_name,
        role_title=title,
        source_url=context["board_root_url"],
        source_type="direct_ats",
        direct_job_url=direct_job_url,
        location_hint=location_hint,
        posted_date_hint=posted_date_hint,
        is_remote_hint=is_remote_hint,
        source_query=f"company_discovery:workday:{context['site'].lower()}",
        evidence_notes=evidence_notes,
        source_quality_score_hint=10,
    )


def _extract_smartrecruiters_board_token(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    if "jobs.smartrecruiters.com" not in host and "careers.smartrecruiters.com" not in host:
        return None
    segments = _path_segments(parsed.path)
    if not segments:
        return None
    return segments[0].lower()


async def _fetch_smartrecruiters_board_jobs(board_token: str) -> list[dict[str, object]]:
    cached = SMARTRECRUITERS_BOARD_JOBS_CACHE.get(board_token)
    if cached is not None:
        return cached

    jobs: list[dict[str, object]] = []
    offset = 0
    limit = 100
    api_url = f"https://api.smartrecruiters.com/v1/companies/{board_token}/postings"

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
        ) as client:
            while True:
                response = await client.get(api_url, params={"offset": offset, "limit": limit})
                if response.status_code != 200:
                    SMARTRECRUITERS_BOARD_JOBS_CACHE[board_token] = []
                    return []
                try:
                    payload = response.json()
                except json.JSONDecodeError:
                    SMARTRECRUITERS_BOARD_JOBS_CACHE[board_token] = []
                    return []
                if not isinstance(payload, dict):
                    SMARTRECRUITERS_BOARD_JOBS_CACHE[board_token] = []
                    return []
                content = payload.get("content")
                if not isinstance(content, list):
                    SMARTRECRUITERS_BOARD_JOBS_CACHE[board_token] = []
                    return []
                normalized_jobs = [job for job in content if isinstance(job, dict)]
                jobs.extend(normalized_jobs)
                total_found = payload.get("totalFound")
                if len(normalized_jobs) < limit:
                    break
                if isinstance(total_found, int) and offset + len(normalized_jobs) >= total_found:
                    break
                offset += len(normalized_jobs)
                if offset >= 500:
                    break
    except httpx.RequestError:
        SMARTRECRUITERS_BOARD_JOBS_CACHE[board_token] = []
        return []

    SMARTRECRUITERS_BOARD_JOBS_CACHE[board_token] = jobs
    return jobs


async def _fetch_smartrecruiters_posting_detail(
    board_token: str,
    posting_id: str,
) -> dict[str, object] | None:
    cache_key = f"{board_token}:{posting_id}"
    if cache_key in SMARTRECRUITERS_POSTING_DETAILS_CACHE:
        return SMARTRECRUITERS_POSTING_DETAILS_CACHE[cache_key]

    api_url = f"https://api.smartrecruiters.com/v1/companies/{board_token}/postings/{posting_id}"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
        ) as client:
            response = await client.get(api_url)
    except httpx.RequestError:
        SMARTRECRUITERS_POSTING_DETAILS_CACHE[cache_key] = None
        return None
    if response.status_code != 200:
        SMARTRECRUITERS_POSTING_DETAILS_CACHE[cache_key] = None
        return None
    try:
        payload = response.json()
    except json.JSONDecodeError:
        SMARTRECRUITERS_POSTING_DETAILS_CACHE[cache_key] = None
        return None
    if not isinstance(payload, dict):
        SMARTRECRUITERS_POSTING_DETAILS_CACHE[cache_key] = None
        return None
    SMARTRECRUITERS_POSTING_DETAILS_CACHE[cache_key] = payload
    return payload


def _slugify_url_title(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return normalized[:96]


def _looks_like_product_manager_title(title: str) -> bool:
    lowered = str(title or "").strip().lower()
    return "product manager" in lowered or bool(re.search(r"\bproduct\b", lowered) and re.search(r"\bmanager\b", lowered))


def _smartrecruiters_company_identifier(
    board_token: str,
    job: dict[str, object],
    detail: dict[str, object] | None = None,
) -> str:
    for payload in (detail, job):
        if not isinstance(payload, dict):
            continue
        company = payload.get("company")
        if not isinstance(company, dict):
            continue
        identifier = str(company.get("identifier") or "").strip()
        if identifier:
            return identifier
    return board_token


def _smartrecruiters_location_text(
    *payloads: dict[str, object] | None,
) -> str:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        location = payload.get("location")
        if not isinstance(location, dict):
            continue
        parts = [
            str(location.get("city") or "").strip(),
            str(location.get("region") or "").strip(),
            str(location.get("country") or "").strip().upper(),
        ]
        location_text = ", ".join(part for part in parts if part)
        if location.get("remote") is True:
            return f"{location_text} Remote".strip() if location_text else "Remote"
        if location_text:
            return location_text
    return ""


def _smartrecruiters_job_description_text(detail: dict[str, object] | None) -> str:
    if not isinstance(detail, dict):
        return ""
    job_ad = detail.get("jobAd")
    if not isinstance(job_ad, dict):
        return ""
    sections = job_ad.get("sections")
    if not isinstance(sections, dict):
        return ""
    parts: list[str] = []
    for value in sections.values():
        if not isinstance(value, dict):
            continue
        title = str(value.get("title") or "").strip()
        text = str(value.get("text") or "").strip()
        if title:
            parts.append(title)
        if text:
            parts.append(text)
    return " ".join(parts)


def _smartrecruiters_public_job_url(
    company_identifier: str,
    job: dict[str, object],
) -> str:
    posting_id = str(job.get("id") or job.get("uuid") or "").strip()
    if not posting_id:
        return ""
    title = str(job.get("name") or job.get("title") or "").strip()
    slug = _slugify_url_title(title)
    path = f"/{company_identifier}/{posting_id}"
    if slug:
        path = f"{path}-{slug}"
    return _normalize_direct_job_url(f"https://jobs.smartrecruiters.com{path}")


async def _smartrecruiters_board_jobs_to_leads(
    board_token: str,
    company_name: str,
) -> list[JobLead]:
    leads: list[JobLead] = []
    for job in await _fetch_smartrecruiters_board_jobs(board_token):
        title = str(job.get("name") or job.get("title") or "").strip()
        if not title or not _looks_like_product_manager_title(title):
            continue
        posting_id = str(job.get("id") or job.get("uuid") or "").strip()
        detail = await _fetch_smartrecruiters_posting_detail(board_token, posting_id) if posting_id else None
        description_text = _smartrecruiters_job_description_text(detail)
        location_text = _smartrecruiters_location_text(detail, job)
        content_text = " ".join(part for part in (title, description_text, location_text) if part)
        if not _is_ai_related_product_manager_text(content_text):
            continue
        company_identifier = _smartrecruiters_company_identifier(board_token, job, detail)
        direct_job_url = _smartrecruiters_public_job_url(company_identifier, job)
        if not direct_job_url.startswith(("http://", "https://")):
            continue
        location_hint = _lead_location_hint_with_remote_restriction(location_text, title, description_text)
        remote_restriction_note = _remote_restriction_note(title, location_text, description_text)
        location = detail.get("location") if isinstance(detail, dict) and isinstance(detail.get("location"), dict) else job.get("location")
        is_remote_hint = True if isinstance(location, dict) and location.get("remote") is True else ("remote" in location_text.lower() if location_text else None)
        evidence_notes = " ".join(
            part
            for part in (
                "Discovered via official SmartRecruiters board enumeration.",
                remote_restriction_note,
            )
            if part
        )
        leads.append(
            JobLead(
                company_name=company_name,
                role_title=title,
                source_url=f"https://jobs.smartrecruiters.com/{company_identifier}",
                source_type="direct_ats",
                direct_job_url=direct_job_url,
                location_hint=location_hint,
                posted_date_hint=str(job.get("releasedDate") or "").strip()[:10] or None,
                is_remote_hint=is_remote_hint,
                source_query=f"company_discovery:smartrecruiters:{board_token}",
                evidence_notes=evidence_notes,
                source_quality_score_hint=10,
            )
        )
    return leads


def _greenhouse_board_job_match_score(lead: JobLead, job: dict[str, object]) -> int:
    title = str(job.get("title") or "").strip()
    absolute_url = str(job.get("absolute_url") or "").strip()
    company_hint = _company_hint_from_url(absolute_url)
    score = _role_match_score(lead.role_title, title)
    if title.lower() == lead.role_title.lower():
        score += 8
    if "product manager" in title.lower():
        score += 4
    if _is_ai_related_product_manager_text(" ".join(part for part in (title, lead.evidence_notes) if part)):
        score += 3
    if company_hint and _company_names_match(lead.company_name, company_hint):
        score += 2
    return score


async def _resolve_greenhouse_board_job_url_from_lead(lead: JobLead) -> str | None:
    board_token = _extract_greenhouse_board_token(lead.direct_job_url or lead.source_url)
    if not board_token:
        return None
    jobs = await _fetch_greenhouse_board_jobs(board_token)
    if not jobs:
        return None

    best_url: str | None = None
    best_score = 0
    for job in jobs:
        absolute_url = str(job.get("absolute_url") or "").strip()
        if not absolute_url.startswith(("http://", "https://")):
            continue
        score = _greenhouse_board_job_match_score(lead, job)
        if score > best_score:
            best_score = score
            best_url = absolute_url
    if best_score <= 0:
        return None
    return _normalize_direct_job_url(best_url)


def _title_case_company_name(company_name: str) -> str:
    words = re.split(r"[\s_-]+", company_name.strip())
    return " ".join(word[:1].upper() + word[1:] for word in words if word)


def _greenhouse_board_job_to_lead(
    board_token: str,
    company_name: str,
    job: dict[str, object],
) -> JobLead | None:
    title = str(job.get("title") or "").strip()
    absolute_url = _normalize_direct_job_url(str(job.get("absolute_url") or "").strip())
    if not title or not absolute_url.startswith(("http://", "https://")):
        return None
    location_text = ""
    location = job.get("location")
    if isinstance(location, dict):
        location_text = str(location.get("name") or "").strip()
    content_text = " ".join(part for part in (title, location_text) if part)
    if not _is_ai_related_product_manager_text(content_text):
        return None
    posted_date_hint = str(job.get("updated_at") or job.get("created_at") or "").strip()[:10]
    is_remote_hint = "remote" in location_text.lower()
    compensation_text = str(job.get("salary") or "").strip() or None
    location_hint = _lead_location_hint_with_remote_restriction(location_text, title)
    remote_restriction_note = _remote_restriction_note(title, location_text)
    evidence_notes = " ".join(
        part
        for part in (
            "Discovered via official Greenhouse board enumeration.",
            remote_restriction_note,
        )
        if part
    )
    return JobLead(
        company_name=company_name,
        role_title=title,
        source_url=f"https://job-boards.greenhouse.io/{board_token}",
        source_type="direct_ats",
        direct_job_url=absolute_url,
        location_hint=location_hint,
        posted_date_hint=posted_date_hint or None,
        is_remote_hint=is_remote_hint or None,
        salary_text_hint=compensation_text,
        source_query=f"company_discovery:greenhouse:{board_token}",
        evidence_notes=evidence_notes,
        source_quality_score_hint=10,
    )


def _ashby_board_job_to_lead(
    board_token: str,
    company_name: str,
    job: dict[str, object],
) -> JobLead | None:
    title = str(job.get("title") or "").strip()
    direct_job_url = _normalize_direct_job_url(str(job.get("jobUrl") or "").strip())
    if not title or not direct_job_url.startswith(("http://", "https://")):
        return None
    description_text = str(job.get("descriptionPlain") or "").strip()
    location_text = str(job.get("location") or "").strip()
    if not _is_ai_related_product_manager_text(" ".join(part for part in (title, description_text) if part)):
        return None
    salary_text_hint = _extract_ashby_compensation_summary(job)
    is_remote = job.get("isRemote")
    is_remote_hint = True if is_remote is True else ("remote" in location_text.lower() if location_text else None)
    published_at = str(job.get("publishedAt") or "").strip()
    location_hint = _lead_location_hint_with_remote_restriction(location_text, title, description_text)
    remote_restriction_note = _remote_restriction_note(title, location_text, description_text)
    evidence_notes = " ".join(
        part
        for part in (
            "Discovered via official Ashby board enumeration.",
            remote_restriction_note,
        )
        if part
    )
    return JobLead(
        company_name=company_name,
        role_title=title,
        source_url=f"https://jobs.ashbyhq.com/{board_token}",
        source_type="direct_ats",
        direct_job_url=direct_job_url,
        location_hint=location_hint,
        posted_date_hint=published_at[:10] or None,
        is_remote_hint=is_remote_hint,
        salary_text_hint=salary_text_hint,
        source_query=f"company_discovery:ashby:{board_token}",
        evidence_notes=evidence_notes,
        source_quality_score_hint=10,
    )


def _extract_ashby_compensation_summary(job: dict[str, object]) -> str | None:
    candidate_nodes: list[dict[str, object]] = [job]
    for key in ("compensation", "compensationData"):
        value = job.get(key)
        if isinstance(value, dict):
            candidate_nodes.append(value)
    summary_keys = (
        "scrapeableCompensationSalarySummary",
        "compensationTierSummary",
        "compensationSummary",
        "salarySummary",
        "payRangeSummary",
        "displayCompensation",
        "summary",
    )
    for node in candidate_nodes:
        for key in summary_keys:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for node in candidate_nodes:
        for key in ("tiers", "compensationTiers"):
            value = node.get(key)
            if not isinstance(value, list):
                continue
            for item in value:
                if not isinstance(item, dict):
                    continue
                for summary_key in summary_keys:
                    summary = item.get(summary_key)
                    if isinstance(summary, str) and summary.strip():
                        return summary.strip()
    return None


def _lever_board_job_to_lead(
    board_token: str,
    company_name: str,
    job: dict[str, object],
) -> JobLead | None:
    title = str(job.get("text") or job.get("title") or "").strip()
    direct_job_url = _normalize_direct_job_url(str(job.get("hostedUrl") or job.get("applyUrl") or "").strip())
    if not title or not direct_job_url.startswith(("http://", "https://")):
        return None
    categories = job.get("categories") if isinstance(job.get("categories"), dict) else {}
    location_text = str(categories.get("location") or "").strip()
    description_text = str(job.get("descriptionPlain") or job.get("description") or "").strip()
    if not _is_ai_related_product_manager_text(" ".join(part for part in (title, description_text) if part)):
        return None
    workplace_type = str(categories.get("commitment") or "").strip()
    salary_text = None
    salary_range = job.get("salaryRange") if isinstance(job.get("salaryRange"), dict) else {}
    if salary_range:
        interval = str(salary_range.get("interval") or "").strip()
        min_salary = salary_range.get("min")
        max_salary = salary_range.get("max")
        currency = str(salary_range.get("currency") or "USD").strip()
        if isinstance(min_salary, (int, float)) and isinstance(max_salary, (int, float)):
            salary_text = f"{currency} {int(min_salary):,} - {int(max_salary):,} {interval}".strip()
    is_remote_hint = "remote" in " ".join([location_text, workplace_type]).lower()
    created_at = str(job.get("createdAt") or "").strip()
    location_hint = _lead_location_hint_with_remote_restriction(location_text, title, description_text, workplace_type)
    remote_restriction_note = _remote_restriction_note(title, location_text, description_text, workplace_type)
    evidence_notes = " ".join(
        part
        for part in (
            "Discovered via official Lever board enumeration.",
            remote_restriction_note,
        )
        if part
    )
    return JobLead(
        company_name=company_name,
        role_title=title,
        source_url=f"https://jobs.lever.co/{board_token}",
        source_type="direct_ats",
        direct_job_url=direct_job_url,
        location_hint=location_hint,
        posted_date_hint=created_at[:10] or None,
        is_remote_hint=is_remote_hint or None,
        salary_text_hint=salary_text,
        source_query=f"company_discovery:lever:{board_token}",
        evidence_notes=evidence_notes,
        source_quality_score_hint=10,
    )


def _company_discovery_trust_score(lead: JobLead) -> int:
    if lead.source_type == "direct_ats":
        return 10
    if lead.source_type == "company_site":
        return 8
    if lead.direct_job_url and _is_allowed_direct_job_url(lead.direct_job_url):
        return 8
    return 5


def _increment_metric_count(bucket: dict[str, int], key: str | None, delta: int = 1) -> None:
    normalized_key = str(key or "").strip()
    if not normalized_key or delta == 0:
        return
    bucket[normalized_key] = int(bucket.get(normalized_key) or 0) + int(delta)


def _reactivate_repairable_board_frontier_tasks(
    frontier: list[dict[str, object]],
    *,
    entries: Mapping[str, dict[str, object]],
) -> None:
    _repair_company_scoped_board_frontier_tasks(frontier, entries=entries)

    merged_board_tasks: dict[str, dict[str, object]] = {}
    merged_board_order: list[str] = []
    passthrough_tasks: list[dict[str, object]] = []

    for task in frontier:
        if str(task.get("task_type") or "") != "board_url":
            passthrough_tasks.append(task)
            continue
        board_identifier = str(task.get("board_identifier") or "").strip()
        if board_identifier.lower() in {"none", "null"}:
            board_identifier = ""
        if not board_identifier:
            board_identifier = str(board_identifier_from_url(str(task.get("url") or "").strip()) or "").strip()
            if board_identifier:
                task["board_identifier"] = board_identifier
        last_error = str(task.get("last_error") or "").strip()
        prefix, _, _ = board_identifier.partition(":")
        if prefix in SUPPORTED_OFFICIAL_BOARD_PREFIXES and last_error in {"unsupported_adapter", "missing_board_identifier"}:
            task["next_retry_at"] = None
            task["last_error"] = None
            task["status"] = "pending"
        task_key = str(task.get("task_key") or "").strip()
        if not task_key:
            passthrough_tasks.append(task)
            continue
        existing = merged_board_tasks.get(task_key)
        if existing is None:
            merged_board_tasks[task_key] = dict(task)
            merged_board_order.append(task_key)
            continue
        existing["priority"] = max(int(existing.get("priority") or 0), int(task.get("priority") or 0))
        existing["source_trust"] = max(int(existing.get("source_trust") or 0), int(task.get("source_trust") or 0))
        existing["attempts"] = min(int(existing.get("attempts") or 0), int(task.get("attempts") or 0))
        for field in ("company_name", "company_key", "board_identifier", "source_kind", "discovered_from"):
            if not existing.get(field) and task.get(field):
                existing[field] = task[field]
        existing_completed = str(existing.get("status") or "").strip() == "completed"
        task_completed = str(task.get("status") or "").strip() == "completed"
        if existing_completed or task_completed:
            existing["status"] = "completed"
            existing["next_retry_at"] = None
            existing["last_error"] = None
            continue
        if not str(existing.get("last_error") or "").strip() or not str(task.get("last_error") or "").strip():
            existing["last_error"] = None
        if not str(existing.get("next_retry_at") or "").strip() or not str(task.get("next_retry_at") or "").strip():
            existing["next_retry_at"] = None
        existing["status"] = "pending"

    frontier[:] = [*passthrough_tasks, *(merged_board_tasks[key] for key in merged_board_order)]


def _frontier_task_kwargs(task: dict[str, object]) -> dict[str, object]:
    return {
        "task_type": str(task.get("task_type") or "").strip(),
        "url": str(task.get("url") or "").strip(),
        "company_name": str(task.get("company_name") or "").strip() or None,
        "company_key": str(task.get("company_key") or "").strip() or None,
        "board_identifier": str(task.get("board_identifier") or "").strip() or None,
        "source_kind": str(task.get("source_kind") or "").strip() or None,
        "source_trust": int(task.get("source_trust") or 0),
        "priority": int(task.get("priority") or 0),
        "discovered_from": str(task.get("discovered_from") or "").strip() or None,
    }


def _company_name_for_frontier_task(
    task: dict[str, object],
    entries: dict[str, dict[str, object]],
) -> str:
    company_name = str(task.get("company_name") or "").strip()
    if company_name:
        return company_name
    company_key = str(task.get("company_key") or "").strip()
    if company_key and company_key in entries:
        entry_name = str(entries[company_key].get("company_name") or "").strip()
        if entry_name:
            return entry_name
    url = str(task.get("url") or "").strip()
    if url:
        inferred = _company_hint_from_url(url)
        if inferred:
            return inferred
    if company_key:
        return _title_case_company_name(company_key)
    return "Unknown Company"


def _infer_company_scoped_board_identifier(
    url: str | None,
    *,
    company_name: str | None,
    company_key: str | None,
    entries: Mapping[str, dict[str, object]],
) -> str | None:
    normalized_url = str(url or "").strip()
    if not normalized_url.startswith(("http://", "https://")):
        return None
    candidate_company_key = str(company_key or _normalize_company_key(company_name)).strip()
    if not candidate_company_key:
        return None
    entry = entries.get(candidate_company_key)
    if not isinstance(entry, Mapping):
        return None

    ats_types = {str(value or "").strip().lower() for value in entry.get("ats_types") or [] if str(value or "").strip()}
    if "smartrecruiters" not in ats_types:
        return None

    parsed = urlparse(normalized_url)
    host = (parsed.netloc or "").lower()
    segments = _path_segments(parsed.path)
    source_hosts = {str(value or "").strip().lower() for value in entry.get("source_hosts") or [] if str(value or "").strip()}

    if "jobs.smartrecruiters.com" in host:
        return f"smartrecruiters:{candidate_company_key}"
    if host in source_hosts and segments and segments[0] == "jobs":
        return f"smartrecruiters:{candidate_company_key}"
    return None


def _repair_company_scoped_board_frontier_tasks(
    frontier: list[dict[str, object]],
    *,
    entries: Mapping[str, dict[str, object]],
) -> None:
    for task in frontier:
        if str(task.get("task_type") or "").strip() != "board_url":
            continue
        task_url = str(task.get("url") or "").strip()
        if not task_url:
            continue
        company_name = str(task.get("company_name") or "").strip() or None
        company_key = str(task.get("company_key") or "").strip() or None
        raw_board_identifier = str(task.get("board_identifier") or "").strip()
        normalized_board_identifier = (
            None if not raw_board_identifier or raw_board_identifier.lower() in {"none", "null"} else raw_board_identifier
        )
        repaired_board_identifier = str(normalized_board_identifier or board_identifier_from_url(task_url) or "").strip()
        if not repaired_board_identifier:
            repaired_board_identifier = str(
                _infer_company_scoped_board_identifier(
                    task_url,
                    company_name=company_name,
                    company_key=company_key,
                    entries=entries,
                )
                or ""
            ).strip()
        if not repaired_board_identifier:
            continue
        prefix, _, token = repaired_board_identifier.partition(":")
        canonical_task_url = task_url
        if prefix == "smartrecruiters" and token:
            canonical_task_url = f"https://jobs.smartrecruiters.com/{token}"
        elif prefix == "workday":
            canonical_task_url = _repair_workday_board_task_url(
                task_url,
                company_key=company_key,
                entries=entries,
            )
            repaired_board_identifier = str(board_identifier_from_url(canonical_task_url) or repaired_board_identifier).strip()
        else:
            continue

        task["board_identifier"] = repaired_board_identifier
        task["url"] = canonical_task_url
        task["task_key"] = frontier_task_key(
            "board_url",
            canonical_task_url,
            board_identifier=repaired_board_identifier,
        )
        task["status"] = "pending"
        task["next_retry_at"] = None
        task["last_error"] = None


def _company_discovery_board_urls_from_sources(*urls: str | None) -> list[str]:
    board_urls: list[str] = []
    for raw_url in urls:
        normalized_url = str(raw_url or "").strip()
        if not board_identifier_from_url(normalized_url):
            continue
        canonical_board_url = str(infer_careers_root(normalized_url) or normalized_url).strip()
        if board_identifier_from_url(canonical_board_url):
            board_urls.append(canonical_board_url)
        else:
            board_urls.append(normalized_url)
    return _dedupe_queries(board_urls)


def _upsert_company_discovery_from_lead(
    entries: dict[str, dict[str, object]],
    lead: JobLead,
    *,
    run_id: str | None,
    ai_pm_candidate_delta: int = 0,
    official_board_lead_delta: int = 0,
) -> tuple[bool, int]:
    board_urls = _company_discovery_board_urls_from_sources(lead.direct_job_url, lead.source_url)
    return upsert_company_discovery_entry(
        entries,
        company_name=lead.company_name,
        source_url=lead.source_url,
        careers_root=infer_careers_root(lead.direct_job_url or lead.source_url),
        board_urls=board_urls,
        board_identifiers=[value for value in (_extract_company_board_identifier(lead.direct_job_url), _extract_company_board_identifier(lead.source_url)) if value],
        ats_types=[value for value in (board_url_ats_type(lead.direct_job_url), board_url_ats_type(lead.source_url)) if value],
        source_trust=_company_discovery_trust_score(lead),
        run_id=run_id,
        ai_pm_candidate_delta=ai_pm_candidate_delta,
        official_board_lead_delta=official_board_lead_delta,
    )


def _upsert_company_discovery_from_validated_job(
    entries: dict[str, dict[str, object]],
    job: JobPosting,
    *,
    run_id: str | None,
) -> tuple[bool, int]:
    job_url = str(job.resolved_job_url or job.direct_job_url or "").strip()
    if not job_url:
        return False, 0
    board_identifier = _extract_company_board_identifier(job_url)
    board_urls = _company_discovery_board_urls_from_sources(job_url) if board_identifier else []
    return upsert_company_discovery_entry(
        entries,
        company_name=job.company_name,
        source_url=job_url,
        careers_root=infer_careers_root(job_url),
        board_urls=board_urls,
        board_identifiers=[board_identifier] if board_identifier else [],
        ats_types=[value for value in (board_url_ats_type(job_url), job.ats_platform) if value],
        source_trust=10 if board_identifier else 8,
        run_id=run_id,
        ai_pm_candidate_delta=1 if _is_ai_related_product_manager(job) else 0,
        official_board_lead_delta=1 if board_identifier else 0,
    )


async def _fetch_page_html(url: str) -> str | None:
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
        ) as client:
            response = await client.get(url)
    except httpx.RequestError:
        return None
    if response.status_code != 200:
        return None
    return response.text


def _ollama_discovery_sidecar_enabled(settings: Settings) -> bool:
    return (
        settings.llm_provider == "ollama"
        and settings.ollama_sidecar_discovery_enabled
        and settings.ollama_sidecar_max_requests_per_run > 0
    )


def _extract_discovery_sidecar_link_candidates(page_url: str, html: str, *, limit: int = 10) -> list[dict[str, object]]:
    soup = BeautifulSoup(html or "", "html.parser")
    page_host = (urlparse(page_url).netloc or "").lower()
    ranked: list[tuple[int, dict[str, object]]] = []
    seen_urls: set[str] = set()

    def maybe_add(raw_url: str | None, link_text: str = "", tag_name: str = "a") -> None:
        normalized = str(raw_url or "").strip()
        if not normalized:
            return
        resolved = _normalize_direct_job_url(urljoin(page_url, normalized))
        if not resolved.startswith(("http://", "https://")) or resolved in seen_urls:
            return
        parsed = urlparse(resolved)
        host = (parsed.netloc or "").lower()
        if not host:
            return
        lower_text = " ".join(link_text.lower().split())
        lower_path = (parsed.path or "").lower()
        lower_url = resolved.lower()
        score = 0
        if any(fragment in host for fragment in KNOWN_BOARD_HOST_FRAGMENTS):
            score += 8
        if host == page_host:
            score += 3
        if any(token in lower_text for token in ("career", "careers", "jobs", "join us", "join-us", "hiring")):
            score += 5
        if any(token in lower_path for token in ("career", "careers", "jobs", "join-us", "joinus")):
            score += 4
        if any(fragment in host for fragment in LOW_TRUST_SCOUT_HOST_FRAGMENTS):
            score -= 8
        if "mailto:" in lower_url or "javascript:" in lower_url:
            score -= 10
        if score <= 0:
            return
        seen_urls.add(resolved)
        ranked.append(
            (
                score,
                {
                    "url": resolved,
                    "link_text": link_text[:120],
                    "tag_name": tag_name,
                    "same_host": host == page_host,
                    "looks_like_board_host": any(fragment in host for fragment in KNOWN_BOARD_HOST_FRAGMENTS),
                },
            )
        )

    for tag in soup.find_all(["a", "iframe", "script"]):
        maybe_add(tag.get("href") or tag.get("src"), tag.get_text(" ", strip=True), tag.name)

    ranked.sort(key=lambda item: (-item[0], str(item[1].get("url") or "")))
    return [payload for _score, payload in ranked[:limit]]


async def _suggest_frontier_urls_with_ollama_sidecar(
    settings: Settings,
    *,
    page_url: str,
    company_name: str,
    link_candidates: list[dict[str, object]],
    run_id: str | None,
) -> list[dict[str, object]]:
    if not _ollama_discovery_sidecar_enabled(settings) or settings.ollama_degraded_for_run:
        return []
    if len(link_candidates) < 2:
        return []

    provider_settings = replace(
        settings,
        ollama_timeout_seconds=max(8.0, min(settings.ollama_sidecar_timeout_seconds, settings.ollama_timeout_seconds)),
        ollama_num_ctx=min(settings.ollama_num_ctx, 384),
        ollama_num_batch=1,
        ollama_num_predict=min(settings.ollama_num_predict, 96),
    )
    provider = OllamaStructuredProvider(provider_settings)
    candidate_urls = {str(item.get("url") or "").strip() for item in link_candidates}
    prompt = f"""
Page URL: {page_url}
Company: {company_name}

Pick only exact URLs from the candidate list below that look like likely official careers roots or official ATS board pages for this company.
Prefer official board hosts and company careers pages.
Do not invent or rewrite URLs.
Return at most 4 suggestions.

Candidates:
{json.dumps(link_candidates, indent=2)}
""".strip()
    try:
        output = await provider.generate_structured(
            system_prompt=(
                "You triage discovery links for a job-hunting pipeline. "
                "Return only exact candidate URLs that are likely official careers roots or official ATS board pages."
            ),
            user_prompt=prompt,
            schema=DiscoveryFrontierSuggestionResult,
            run_id=run_id,
            caller="discovery_sidecar",
            prompt_category="discovery_frontier_triage",
        )
    except LLMProviderError:
        return []

    suggestions: list[dict[str, object]] = []
    for item in output.suggestions[:4]:
        normalized_url = _normalize_direct_job_url(item.url)
        if normalized_url not in candidate_urls:
            continue
        suggestions.append(
            {
                "url": normalized_url,
                "task_type": item.task_type,
                "priority_boost": max(0, min(int(item.priority_boost or 0), 4)),
                "reason": (item.reason or "").strip() or None,
            }
        )
    record_ollama_event(
        settings,
        "discovery_sidecar_outcome",
        run_id=run_id,
        caller="discovery_sidecar",
        prompt_category="discovery_frontier_triage",
        page_url=page_url,
        candidate_link_count=len(link_candidates),
        suggested_count=len(output.suggestions),
        used_suggestion_count=len(suggestions),
        priority_boost_count=sum(int(item.get("priority_boost") or 0) for item in suggestions),
    )
    return suggestions


async def _collect_company_discovery_seed_leads(
    settings: Settings,
    *,
    discovery_agent: Agent | None,
    run_id: str | None,
    previously_reported_company_keys: set[str] | None = None,
) -> tuple[list[JobLead], dict[str, object]]:
    entries = load_company_discovery_entries(settings.data_dir)
    previously_reported_company_keys = set(
        previously_reported_company_keys or load_previously_reported_company_keys(settings.data_dir)
    )
    frontier = load_company_discovery_frontier(settings.data_dir)
    _reactivate_repairable_board_frontier_tasks(frontier, entries=entries)
    crawl_history = load_company_discovery_crawl_history(settings.data_dir)
    audit_entries = load_company_discovery_audit(settings.data_dir)
    new_company_keys: set[str] = set()
    new_board_identifiers: set[str] = set()
    company_keys_with_ai_pm_leads: set[str] = set()
    official_board_leads_count = 0
    official_roles_missed_count = 0
    frontier_tasks_consumed_count = 0
    official_board_crawl_attempt_count = 0
    official_board_crawl_success_count = 0
    source_adapter_yields: Counter[str] = Counter()
    ollama_sidecar_requests_remaining = max(0, settings.ollama_sidecar_max_requests_per_run)

    query_leads: list[JobLead] = []
    official_board_leads: list[JobLead] = []

    def _queue_frontier_url(
        *,
        url: str | None,
        company_name: str | None,
        discovered_from: str | None,
        priority: int,
        preferred_task_type: str | None = None,
    ) -> bool:
        normalized_url = str(url or "").strip()
        if not normalized_url.startswith(("http://", "https://")):
            return False
        source_trust = trust_score_for_url(normalized_url, explicit_task_type=preferred_task_type)
        if source_trust <= 0 or source_trust > settings.company_discovery_source_max_trust:
            return False
        company_key = _normalize_company_key(company_name)
        board_identifier = board_identifier_from_url(normalized_url) or _infer_company_scoped_board_identifier(
            normalized_url,
            company_name=company_name,
            company_key=company_key,
            entries=entries,
        )
        if board_identifier:
            return upsert_frontier_task(
                frontier,
                task_type="board_url",
                url=normalized_url,
                company_name=company_name,
                company_key=company_key,
                board_identifier=board_identifier,
                source_kind="board_url",
                source_trust=source_trust,
                priority=max(0, int(priority)),
                discovered_from=discovered_from,
            )
            
        task_type = preferred_task_type or ("careers_root" if "/careers" in normalized_url.lower() or normalized_url.lower().endswith("/jobs") else "company_page")
        if task_type in {"company_page", "careers_root"} and not is_company_discovery_seed_url(
            normalized_url,
            preferred_task_type=task_type,
        ):
            return False
        if task_type in {"company_page", "careers_root"} and company_name:
            candidate_company_hint = _company_hint_from_url(normalized_url)
            if (
                not _is_weak_company_hint(candidate_company_hint)
                and not _company_names_match(company_name, candidate_company_hint)
            ):
                return False
        return upsert_frontier_task(
            frontier,
            task_type=task_type,
            url=normalized_url,
            company_name=company_name,
            company_key=company_key,
            source_kind=task_type,
            source_trust=source_trust,
            priority=priority,
            discovered_from=discovered_from,
        )

    async def _maybe_apply_ollama_discovery_sidecar(
        *,
        task_type: str,
        task_url: str,
        company_name: str,
        html: str,
    ) -> int:
        nonlocal ollama_sidecar_requests_remaining
        if ollama_sidecar_requests_remaining <= 0 or not _ollama_discovery_sidecar_enabled(settings):
            return 0
        if task_type not in {"company_page", "careers_root"}:
            return 0
        link_candidates = _extract_discovery_sidecar_link_candidates(task_url, html)
        if len(link_candidates) < 2:
            return 0
        ollama_sidecar_requests_remaining -= 1
        suggestions = await _suggest_frontier_urls_with_ollama_sidecar(
            settings,
            page_url=task_url,
            company_name=company_name,
            link_candidates=link_candidates,
            run_id=run_id,
        )
        applied = 0
        for suggestion in suggestions:
            preferred_task_type = str(suggestion.get("task_type") or "").strip() or None
            priority_boost = int(suggestion.get("priority_boost") or 0)
            if _queue_frontier_url(
                url=str(suggestion.get("url") or "").strip(),
                company_name=company_name,
                discovered_from=f"ollama_sidecar:{task_url}",
                priority=10 + priority_boost if preferred_task_type == "board_url" else 8 + priority_boost,
                preferred_task_type=preferred_task_type,
            ):
                applied += 1
        if applied > 0:
            _increment_metric_count(source_adapter_yields, "ollama_sidecar", applied)
        return applied

    def _record_company_discovery_lead(lead: JobLead, *, source_adapter: str) -> None:
        was_new_company, new_board_count = _upsert_company_discovery_from_lead(
            entries,
            lead,
            run_id=run_id,
            ai_pm_candidate_delta=1,
        )
        company_key = _normalize_company_key(lead.company_name)
        if was_new_company and company_key:
            new_company_keys.add(company_key)
        if new_board_count:
            for identifier in (
                board_identifier_from_url(lead.direct_job_url),
                board_identifier_from_url(lead.source_url),
            ):
                if identifier:
                    new_board_identifiers.add(identifier)
        if company_key:
            company_keys_with_ai_pm_leads.add(company_key)
        _increment_metric_count(source_adapter_yields, source_adapter)
        _queue_frontier_url(
            url=lead.source_url,
            company_name=lead.company_name,
            discovered_from=lead.source_query or source_adapter,
            priority=7,
            preferred_task_type="company_page",
        )
        _queue_frontier_url(
            url=lead.direct_job_url,
            company_name=lead.company_name,
            discovered_from=lead.source_query or source_adapter,
            priority=10,
            preferred_task_type="board_url",
        )
        for careers_candidate in default_careers_candidate_urls(lead.source_url):
            _queue_frontier_url(
                url=careers_candidate,
                company_name=lead.company_name,
                discovered_from=lead.source_query or source_adapter,
                priority=6,
                preferred_task_type="careers_root",
            )

    async def _crawl_board_frontier(*, budget: int, phase_label: str) -> None:
        nonlocal official_board_leads_count
        nonlocal official_roles_missed_count
        nonlocal frontier_tasks_consumed_count
        nonlocal official_board_crawl_attempt_count
        nonlocal official_board_crawl_success_count

        board_tasks = select_frontier_tasks(frontier, budget=len(frontier), task_types={"board_url"})
        if _frontier_has_pending_novel_company_expansion(
            frontier,
            previously_reported_company_keys=previously_reported_company_keys,
        ):
            board_tasks = [
                task
                for task in board_tasks
                if not _should_defer_reported_saturated_board_task(
                    task,
                    entries=entries,
                    previously_reported_company_keys=previously_reported_company_keys,
                )
            ]

        for task in board_tasks[:budget]:
            task_key = str(task.get("task_key") or "")
            task_url = str(task.get("url") or "").strip()
            board_identifier = str(task.get("board_identifier") or board_identifier_from_url(task_url) or "").strip()
            if not board_identifier:
                inferred_company_name = str(task.get("company_name") or "").strip() or None
                inferred_company_key = str(task.get("company_key") or "").strip() or None
                inferred_board_identifier = _infer_company_scoped_board_identifier(
                    task_url,
                    company_name=inferred_company_name,
                    company_key=inferred_company_key,
                    entries=entries,
                )
                if inferred_board_identifier:
                    task["board_identifier"] = inferred_board_identifier
                    board_identifier = inferred_board_identifier
                    prefix, _, token = board_identifier.partition(":")
                    if prefix == "smartrecruiters" and token:
                        task_url = f"https://jobs.smartrecruiters.com/{token}"
                        task["url"] = task_url
                    task["task_key"] = frontier_task_key("board_url", task_url, board_identifier=board_identifier or None)
                    task_key = str(task["task_key"])
            if not task_url or not board_identifier:
                update_frontier_task_state(frontier, task_key=task_key, success=False, error="missing_board_identifier")
                record_crawl_result(
                    crawl_history,
                    target_type="board_url",
                    url=task_url,
                    company_key=str(task.get("company_key") or "").strip() or None,
                    board_identifier=board_identifier or None,
                    success=False,
                    error="missing_board_identifier",
                )
                official_roles_missed_count += 1
                frontier_tasks_consumed_count += 1
                continue

            prefix, _, token = board_identifier.partition(":")
            company_name = _company_name_for_frontier_task(task, entries)
            company_key = _normalize_company_key(company_name)
            leads_for_task: list[JobLead] = []
            if prefix == "workday":
                repaired_task_url = _repair_workday_board_task_url(
                    task_url,
                    company_key=company_key or None,
                    entries=entries,
                )
                repaired_board_identifier = str(board_identifier_from_url(repaired_task_url) or board_identifier).strip()
                if repaired_task_url != task_url or repaired_board_identifier != board_identifier:
                    task["url"] = repaired_task_url
                    task["board_identifier"] = repaired_board_identifier
                    task["task_key"] = frontier_task_key("board_url", repaired_task_url, board_identifier=repaired_board_identifier or None)
                    task_key = str(task["task_key"])
                    task_url = repaired_task_url
                    board_identifier = repaired_board_identifier
                    prefix, _, token = board_identifier.partition(":")
                if _workday_board_context(task_url) is None:
                    update_frontier_task_state(frontier, task_key=task_key, success=False, error="missing_board_identifier")
                    record_crawl_result(
                        crawl_history,
                        target_type="board_url",
                        url=task_url,
                        company_key=company_key or None,
                        board_identifier=board_identifier,
                        success=False,
                        error="missing_board_identifier",
                    )
                    official_roles_missed_count += 1
                    frontier_tasks_consumed_count += 1
                    continue
            official_board_crawl_attempt_count += 1
            frontier_tasks_consumed_count += 1

            try:
                if prefix == "greenhouse" and token:
                    leads_for_task = [
                        lead
                        for lead in (
                            _greenhouse_board_job_to_lead(token, company_name, job)
                            for job in await _fetch_greenhouse_board_jobs(token)
                        )
                        if lead is not None
                    ]
                elif prefix == "ashby" and token:
                    leads_for_task = [
                        lead
                        for lead in (
                            _ashby_board_job_to_lead(token, company_name, job)
                            for job in await _fetch_ashby_board_jobs(token)
                        )
                        if lead is not None
                    ]
                elif prefix == "lever" and token:
                    leads_for_task = [
                        lead
                        for lead in (
                            _lever_board_job_to_lead(token, company_name, job)
                            for job in await _fetch_lever_board_jobs(token)
                        )
                        if lead is not None
                    ]
                elif prefix == "smartrecruiters" and token:
                    leads_for_task = await _smartrecruiters_board_jobs_to_leads(token, company_name)
                elif prefix == "workday":
                    leads_for_task = [
                        lead
                        for lead in (
                            _workday_board_job_to_lead(task_url, company_name, job)
                            for job in await _fetch_workday_board_jobs(task_url)
                        )
                        if lead is not None
                    ]
                else:
                    append_company_discovery_audit_entry(
                        audit_entries,
                        {
                            "run_id": run_id,
                            "phase": phase_label,
                            "status": "unsupported_adapter",
                            "company_name": company_name,
                            "board_identifier": board_identifier,
                            "board_url": task_url,
                        },
                    )
                    update_frontier_task_state(frontier, task_key=task_key, success=False, error="unsupported_adapter")
                    record_crawl_result(
                        crawl_history,
                        target_type="board_url",
                        url=task_url,
                        company_key=company_key or None,
                        board_identifier=board_identifier,
                        success=False,
                        error="unsupported_adapter",
                    )
                    official_roles_missed_count += 1
                    continue
            except Exception as exc:
                update_frontier_task_state(frontier, task_key=task_key, success=False, error=str(exc))
                record_crawl_result(
                    crawl_history,
                    target_type="board_url",
                    url=task_url,
                    company_key=company_key or None,
                    board_identifier=board_identifier,
                    success=False,
                    error=str(exc),
                )
                append_company_discovery_audit_entry(
                    audit_entries,
                    {
                        "run_id": run_id,
                        "phase": phase_label,
                        "status": "crawl_failed",
                        "company_name": company_name,
                        "board_identifier": board_identifier,
                        "board_url": task_url,
                        "error": str(exc),
                    },
                )
                official_roles_missed_count += 1
                continue

            filtered_leads_for_task: list[JobLead] = []
            for lead in leads_for_task:
                suppression_failure = _trusted_direct_lead_precheck_failure(lead, settings)
                if suppression_failure is not None:
                    append_company_discovery_audit_entry(
                        audit_entries,
                        {
                            "run_id": run_id,
                            "phase": phase_label,
                            "status": "suppressed_out_of_scope",
                            "company_name": lead.company_name,
                            "role_title": lead.role_title,
                            "board_identifier": board_identifier,
                            "board_url": task_url,
                            "direct_job_url": lead.direct_job_url,
                            "reason_code": suppression_failure.reason_code,
                            "detail": suppression_failure.detail,
                        },
                    )
                    continue
                filtered_leads_for_task.append(lead)
            leads_for_task = filtered_leads_for_task

            official_board_crawl_success_count += 1
            record_crawl_result(
                crawl_history,
                target_type="board_url",
                url=task_url,
                company_key=company_key or None,
                board_identifier=board_identifier,
                success=True,
                fresh_role_count=len(leads_for_task),
            )
            update_frontier_task_state(frontier, task_key=task_key, success=True)
            was_new_company, new_board_count = upsert_company_discovery_entry(
                entries,
                company_name=company_name,
                source_url=task_url,
                careers_root=infer_careers_root(task_url),
                board_urls=[task_url],
                board_identifiers=[board_identifier],
                ats_types=[board_url_ats_type(task_url)] if board_url_ats_type(task_url) else [],
                source_trust=max(int(task.get("source_trust") or 0), 9),
                run_id=run_id,
                ai_pm_candidate_delta=len(leads_for_task),
                official_board_lead_delta=len(leads_for_task),
                source_kind="board_url",
                board_crawl_succeeded=True,
                fresh_role_delta=len(leads_for_task),
            )
            if was_new_company and company_key:
                new_company_keys.add(company_key)
            if new_board_count:
                new_board_identifiers.add(board_identifier)
            if company_key and leads_for_task:
                company_keys_with_ai_pm_leads.add(company_key)
            _increment_metric_count(source_adapter_yields, prefix, len(leads_for_task))

            for lead in leads_for_task:
                official_board_leads.append(lead)
                official_board_leads_count += 1
                append_company_discovery_audit_entry(
                    audit_entries,
                    {
                        "run_id": run_id,
                        "phase": phase_label,
                        "status": "surfaced",
                        "company_name": lead.company_name,
                        "role_title": lead.role_title,
                        "board_identifier": board_identifier,
                        "board_url": task_url,
                        "direct_job_url": lead.direct_job_url,
                    },
                )

    if settings.company_discovery_indexer_enabled:
        for seeded_task in source_directory_seed_tasks():
            upsert_frontier_task(frontier, reactivate_completed=True, **_frontier_task_kwargs(seeded_task))

        ranked_entries = [
            (company_key, entry)
            for company_key, entry in sorted(
                entries.items(),
                key=lambda item: (
                    1 if item[0] in previously_reported_company_keys else 0,
                    *_company_discovery_entry_seed_priority(item[1]),
                ),
            )
            if not is_low_value_company_discovery_entry(entry)
        ]
        for company_key, entry in ranked_entries[: max(60, settings.company_discovery_frontier_budget_per_run * 4)]:
            company_name = str(entry.get("company_name") or "").strip() or _title_case_company_name(company_key)
            priority = _company_discovery_entry_frontier_priority(
                entry,
                company_key=company_key,
                previously_reported_company_keys=previously_reported_company_keys,
            )
            for board_url in [str(item) for item in entry.get("board_urls") or [] if str(item).strip()][:4]:
                _queue_frontier_url(
                    url=board_url,
                    company_name=company_name,
                    discovered_from="company_discovery_index",
                    priority=priority,
                    preferred_task_type="board_url",
                )
            for careers_root in [str(item) for item in entry.get("careers_roots") or [] if str(item).strip()][:3]:
                _queue_frontier_url(
                    url=careers_root,
                    company_name=company_name,
                    discovered_from="company_discovery_index",
                    priority=max(4, priority - 1),
                    preferred_task_type="careers_root",
                )

        directory_budget = max(0, settings.company_discovery_directory_crawl_budget_per_run)
        for task in select_frontier_tasks(frontier, budget=directory_budget, task_types={"directory_source", "portfolio_source"}):
            task_key = str(task.get("task_key") or "")
            task_url = str(task.get("url") or "").strip()
            frontier_tasks_consumed_count += 1
            html = await _fetch_page_html(task_url)
            if not html:
                update_frontier_task_state(frontier, task_key=task_key, success=False, error="fetch_failed")
                record_crawl_result(
                    crawl_history,
                    target_type=str(task.get("task_type") or "directory_source"),
                    url=task_url,
                    success=False,
                    error="fetch_failed",
                )
                continue
            discovered_count = 0
            known_company_keys = {
                str(company_key).strip()
                for company_key in (
                    *entries.keys(),
                    *previously_reported_company_keys,
                    *(task.get("company_key") for task in frontier),
                )
                if str(company_key).strip()
            }
            directory_company_candidates = select_directory_company_tasks(
                extract_directory_company_tasks(task_url, html, limit=None),
                known_company_keys=known_company_keys,
                attempt_count=int(task.get("attempts") or 0),
            )
            for candidate in directory_company_candidates:
                if _queue_frontier_url(
                    url=str(candidate.get("url") or "").strip(),
                    company_name=str(candidate.get("company_name") or "").strip() or None,
                    discovered_from=task_url,
                    priority=9 if str(candidate.get("task_type") or "") == "careers_root" else 8,
                    preferred_task_type=str(candidate.get("task_type") or "").strip() or "company_page",
                ):
                    discovered_count += 1
            company_urls = extract_company_homepage_urls(task_url, html)
            for homepage in company_urls:
                if _queue_frontier_url(
                    url=homepage,
                    company_name=None,
                    discovered_from=task_url,
                    priority=6,
                    preferred_task_type="company_page",
                ):
                    discovered_count += 1
            _increment_metric_count(source_adapter_yields, str(task.get("task_type") or "directory_source"), discovered_count)
            update_frontier_task_state(frontier, task_key=task_key, success=True)
            record_crawl_result(
                crawl_history,
                target_type=str(task.get("task_type") or "directory_source"),
                url=task_url,
                success=True,
            )

        frontier_budget = max(0, settings.company_discovery_frontier_budget_per_run)
        for task in select_frontier_tasks(frontier, budget=frontier_budget, task_types={"company_page", "careers_root"}):
            task_key = str(task.get("task_key") or "")
            task_url = str(task.get("url") or "").strip()
            task_type = str(task.get("task_type") or "company_page")
            company_name = _company_name_for_frontier_task(task, entries)
            frontier_tasks_consumed_count += 1
            html = await _fetch_page_html(task_url)
            if not html:
                update_frontier_task_state(frontier, task_key=task_key, success=False, error="fetch_failed")
                record_crawl_result(
                    crawl_history,
                    target_type=task_type,
                    url=task_url,
                    company_key=_normalize_company_key(company_name) or None,
                    success=False,
                    error="fetch_failed",
                )
                continue

            default_careers_candidates = set(default_careers_candidate_urls(task_url)) if task_type == "company_page" else set()
            extracted_careers = extract_careers_page_urls(task_url, html)
            if task_type == "company_page":
                extracted_careers = _dedupe_queries([*extracted_careers, *default_careers_candidate_urls(task_url)])
            board_urls = extract_embedded_board_urls(task_url, html)
            discovered_count = 0
            needs_sidecar_help = not board_urls and (
                len(extracted_careers) <= 1
                or (
                    task_type == "company_page"
                    and extracted_careers
                    and all(url in default_careers_candidates for url in extracted_careers)
                )
            )
            if needs_sidecar_help:
                discovered_count += await _maybe_apply_ollama_discovery_sidecar(
                    task_type=task_type,
                    task_url=task_url,
                    company_name=company_name,
                    html=html,
                )
            if task_type == "careers_root":
                was_new_company, new_board_count = upsert_company_discovery_entry(
                    entries,
                    company_name=company_name,
                    source_url=task_url,
                    careers_root=task_url,
                    board_urls=board_urls,
                    ats_types=[value for value in (board_url_ats_type(url) for url in board_urls) if value],
                    source_trust=max(int(task.get("source_trust") or 0), 7),
                    run_id=run_id,
                    source_kind="careers_root",
                )
                if was_new_company:
                    new_company_keys.add(_normalize_company_key(company_name))
                if new_board_count:
                    for board_url in board_urls:
                        identifier = board_identifier_from_url(board_url)
                        if identifier:
                            new_board_identifiers.add(identifier)
            for careers_url in extracted_careers[:8]:
                _queue_frontier_url(
                    url=careers_url,
                    company_name=company_name,
                    discovered_from=task_url,
                    priority=8 if task_type == "company_page" else 7,
                    preferred_task_type="careers_root",
                )
                discovered_count += 1
            for board_url in board_urls[:8]:
                _queue_frontier_url(
                    url=board_url,
                    company_name=company_name,
                    discovered_from=task_url,
                    priority=10,
                    preferred_task_type="board_url",
                )
                discovered_count += 1
            _increment_metric_count(source_adapter_yields, task_type, discovered_count)
            update_frontier_task_state(frontier, task_key=task_key, success=True)
            record_crawl_result(
                crawl_history,
                target_type=task_type,
                url=task_url,
                company_key=_normalize_company_key(company_name) or None,
                success=True,
            )

        await _crawl_board_frontier(
            budget=max(0, settings.company_discovery_board_crawl_budget_per_run),
            phase_label="frontier_board_crawl",
        )

    if settings.company_discovery_enabled:
        query_budget = max(12, settings.search_round_query_limit * 3)
        for query in _build_company_discovery_seed_queries(settings, SearchTuning(attempt_number=1))[:query_budget]:
            try:
                _, discovered = await _search_query_with_context(
                    discovery_agent,
                    settings,
                    query,
                    _query_timeout_seconds_for_query(settings, query),
                    attempt_number=1,
                    run_id=run_id,
                )
            except Exception:
                continue
            fresh_for_query = 0
            for lead in discovered:
                _record_company_discovery_lead(lead, source_adapter="role_first_search")
                query_leads.append(lead)
                fresh_for_query += 1

        query_followup_budget = min(max(0, settings.company_discovery_frontier_budget_per_run), 12)
        for task in select_frontier_tasks(frontier, budget=query_followup_budget, task_types={"company_page", "careers_root"}):
            task_key = str(task.get("task_key") or "")
            task_url = str(task.get("url") or "").strip()
            task_type = str(task.get("task_type") or "company_page")
            company_name = _company_name_for_frontier_task(task, entries)
            frontier_tasks_consumed_count += 1
            html = await _fetch_page_html(task_url)
            if not html:
                update_frontier_task_state(frontier, task_key=task_key, success=False, error="fetch_failed")
                record_crawl_result(
                    crawl_history,
                    target_type=task_type,
                    url=task_url,
                    company_key=_normalize_company_key(company_name) or None,
                    success=False,
                    error="fetch_failed",
                )
                continue

            default_careers_candidates = set(default_careers_candidate_urls(task_url)) if task_type == "company_page" else set()
            extracted_careers = extract_careers_page_urls(task_url, html)
            if task_type == "company_page":
                extracted_careers = _dedupe_queries([*extracted_careers, *default_careers_candidate_urls(task_url)])
            board_urls = extract_embedded_board_urls(task_url, html)
            discovered_count = 0
            needs_sidecar_help = not board_urls and (
                len(extracted_careers) <= 1
                or (
                    task_type == "company_page"
                    and extracted_careers
                    and all(url in default_careers_candidates for url in extracted_careers)
                )
            )
            if needs_sidecar_help:
                discovered_count += await _maybe_apply_ollama_discovery_sidecar(
                    task_type=task_type,
                    task_url=task_url,
                    company_name=company_name,
                    html=html,
                )
            if task_type == "careers_root":
                was_new_company, new_board_count = upsert_company_discovery_entry(
                    entries,
                    company_name=company_name,
                    source_url=task_url,
                    careers_root=task_url,
                    board_urls=board_urls,
                    ats_types=[value for value in (board_url_ats_type(url) for url in board_urls) if value],
                    source_trust=max(int(task.get("source_trust") or 0), 7),
                    run_id=run_id,
                    source_kind="careers_root",
                )
                if was_new_company:
                    new_company_keys.add(_normalize_company_key(company_name))
                if new_board_count:
                    for board_url in board_urls:
                        identifier = board_identifier_from_url(board_url)
                        if identifier:
                            new_board_identifiers.add(identifier)
            for careers_url in extracted_careers[:8]:
                _queue_frontier_url(
                    url=careers_url,
                    company_name=company_name,
                    discovered_from=task_url,
                    priority=8 if task_type == "company_page" else 7,
                    preferred_task_type="careers_root",
                )
                discovered_count += 1
            for board_url in board_urls[:8]:
                _queue_frontier_url(
                    url=board_url,
                    company_name=company_name,
                    discovered_from=task_url,
                    priority=10,
                    preferred_task_type="board_url",
                )
                discovered_count += 1
            _increment_metric_count(source_adapter_yields, f"{task_type}_followup", discovered_count)
            update_frontier_task_state(frontier, task_key=task_key, success=True)
            record_crawl_result(
                crawl_history,
                target_type=task_type,
                url=task_url,
                company_key=_normalize_company_key(company_name) or None,
                success=True,
            )

        remaining_board_budget = max(
            0,
            settings.company_discovery_board_crawl_budget_per_run - official_board_crawl_attempt_count,
        )
        if remaining_board_budget:
            await _crawl_board_frontier(
                budget=remaining_board_budget,
                phase_label="query_discovered_board_crawl",
            )

    save_company_discovery_entries(settings.data_dir, entries)
    save_company_discovery_frontier(settings.data_dir, frontier)
    save_company_discovery_crawl_history(settings.data_dir, crawl_history)
    save_company_discovery_audit(settings.data_dir, audit_entries)
    leads = _dedupe_round_leads([*query_leads, *official_board_leads], settings)
    metrics = {
        "new_companies_discovered_count": len(new_company_keys),
        "new_boards_discovered_count": len(new_board_identifiers),
        "official_board_leads_count": official_board_leads_count,
        "companies_with_ai_pm_leads_count": len(company_keys_with_ai_pm_leads),
        "official_roles_missed_count": official_roles_missed_count,
        "frontier_tasks_consumed_count": frontier_tasks_consumed_count,
        "frontier_backlog_count": sum(1 for task in frontier if str(task.get("status") or "pending") == "pending"),
        "official_board_crawl_attempt_count": official_board_crawl_attempt_count,
        "official_board_crawl_success_count": official_board_crawl_success_count,
        "source_adapter_yields": dict(source_adapter_yields),
    }
    return leads, metrics


def _jsonld_graph_nodes(payload: object) -> list[dict[str, object]]:
    nodes: list[dict[str, object]] = []
    if isinstance(payload, dict):
        graph = payload.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                if isinstance(item, dict):
                    nodes.append(item)
        else:
            nodes.append(payload)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                nodes.append(item)
    return nodes


def _extract_builtin_search_items(html: str) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    items: list[tuple[str, str, str]] = []

    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue

        for node in _jsonld_graph_nodes(payload):
            if node.get("@type") != "ItemList":
                continue
            for item in node.get("itemListElement", []):
                if not isinstance(item, dict):
                    continue
                url = item.get("url")
                if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    continue
                title = item.get("name")
                description = item.get("description")
                items.append((url, str(title or "").strip(), str(description or "").strip()))
    return items


def _decode_embedded_absolute_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = unescape(value).strip()
    if not candidate:
        return None
    if "\\" in candidate:
        try:
            decoded = json.loads(f'"{candidate}"')
        except json.JSONDecodeError:
            decoded = candidate.replace("\\/", "/")
    else:
        decoded = candidate
    normalized = str(decoded).strip()
    if normalized.startswith("//"):
        normalized = f"https:{normalized}"
    if normalized.startswith(("http://", "https://")):
        return normalized
    return None


def _extract_builtin_apply_url_candidates_from_payload(payload: object) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    target_fields = {
        "howtoapply",
        "applyurl",
        "applicationurl",
        "hostedurl",
        "joburl",
        "externalurl",
    }

    def maybe_add(value: object) -> None:
        candidate = _decode_embedded_absolute_url(value)
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    def collect(node: object) -> None:
        if isinstance(node, dict):
            for raw_key, value in node.items():
                key = str(raw_key or "").strip().lower()
                if key in target_fields:
                    if key == "howtoapply" and isinstance(value, dict):
                        for nested_key in ("url", "@id", "href", "applyUrl", "applicationUrl", "hostedUrl", "jobUrl"):
                            maybe_add(value.get(nested_key))
                    else:
                        maybe_add(value)
                collect(value)
            return
        if isinstance(node, list):
            for item in node:
                collect(item)

    collect(payload)
    return candidates


def _extract_builtin_apply_url(html: str) -> str | None:
    key_pattern = r'"(?:howToApply|applyUrl|applicationUrl|hostedUrl|jobUrl|externalUrl)"'
    string_field_pattern = re.compile(rf"{key_pattern}\s*:\s*\"(?P<url>[^\"]+)\"", re.I)

    for match in string_field_pattern.finditer(html):
        candidate = _decode_embedded_absolute_url(match.group("url"))
        if candidate:
            return candidate

    soup = BeautifulSoup(html, "html.parser")
    for script in soup.select("script"):
        raw = script.get_text(strip=True)
        if not raw or not re.search(key_pattern, raw, re.I):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates = _extract_builtin_apply_url_candidates_from_payload(payload)
        if candidates:
            return candidates[0]

    return None


def _extract_builtin_company_followup_urls(html: str) -> list[str]:
    payload = _extract_builtin_jobposting_payload(html)
    if not isinstance(payload, dict):
        return []
    hiring_org = payload.get("hiringOrganization")
    if not isinstance(hiring_org, dict):
        return []

    candidates: list[str] = []
    seen: set[str] = set()

    def maybe_add(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                maybe_add(item)
            return
        candidate = _decode_embedded_absolute_url(value)
        if not candidate:
            return
        parsed = urlparse(candidate)
        host = (parsed.netloc or "").lower()
        if not host:
            return
        if any(fragment in host for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
            return
        if any(fragment in host for fragment in COMPANY_SITE_JUNK_HOST_FRAGMENTS):
            return
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    for key in ("sameAs", "url", "careerSiteUrl", "careersUrl", "careerUrl", "jobUrl"):
        maybe_add(hiring_org.get(key))

    return candidates


def _extract_structured_company_followup_urls(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []
    seen: set[str] = set()

    def maybe_add(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                maybe_add(item)
            return
        candidate = _decode_embedded_absolute_url(value)
        if not candidate:
            return
        parsed = urlparse(candidate)
        host = (parsed.netloc or "").lower()
        if not host:
            return
        if any(fragment in host for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
            return
        if any(fragment in host for fragment in COMPANY_SITE_JUNK_HOST_FRAGMENTS):
            return
        if candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    url_keys = ("sameAs", "url", "careerSiteUrl", "careersUrl", "careerUrl", "jobUrl", "@id")
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for node in _jsonld_graph_nodes(payload):
            if not isinstance(node, dict):
                continue
            nodes_to_scan: list[dict[str, object]] = [node]
            hiring_org = node.get("hiringOrganization")
            if isinstance(hiring_org, dict):
                nodes_to_scan.append(hiring_org)
            for candidate_node in nodes_to_scan:
                for key in url_keys:
                    maybe_add(candidate_node.get(key))

    return candidates


def _extract_builtin_jobposting_payload(html: str) -> dict[str, object] | None:
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.get_text(strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for node in _jsonld_graph_nodes(payload):
            if node.get("@type") == "JobPosting":
                return node
    return None


def _extract_builtin_company(jobposting: dict[str, object]) -> str | None:
    hiring_org = jobposting.get("hiringOrganization")
    if isinstance(hiring_org, dict):
        name = hiring_org.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return None


def _extract_builtin_location(jobposting: dict[str, object]) -> str | None:
    location = jobposting.get("jobLocation")
    if isinstance(location, list):
        parts: list[str] = []
        for item in location:
            if not isinstance(item, dict):
                continue
            address = item.get("address")
            if not isinstance(address, dict):
                continue
            for value in (address.get("addressLocality"), address.get("addressRegion"), address.get("addressCountry")):
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
        return ", ".join(dict.fromkeys(parts)) if parts else None
    if isinstance(location, dict):
        address = location.get("address")
        if isinstance(address, dict):
            parts: list[str] = []
            for value in (address.get("addressLocality"), address.get("addressRegion"), address.get("addressCountry")):
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
            return ", ".join(parts) if parts else None
    return None


def _extract_builtin_remote_hint(
    location_text: str | None,
    description_text: str,
    *,
    source_is_remote_listing: bool = False,
) -> bool | None:
    location_haystack = str(location_text or "").lower()
    description_haystack = str(description_text or "").lower()
    haystack = " ".join(part for part in (location_haystack, description_haystack) if part)
    if "hybrid" in haystack or "in-office" in haystack or "onsite" in haystack or "on-site" in haystack:
        return False
    if "remote" in location_haystack or "work from home" in location_haystack:
        return True
    strong_description_markers = (
        *BROAD_REMOTE_OVERRIDE_MARKERS,
        "location: remote",
        "this role is remote",
        "this position is remote",
        "remote position",
        "remote role",
    )
    if any(marker in description_haystack for marker in strong_description_markers):
        return True
    if source_is_remote_listing:
        return None
    return None


def _remote_restriction_note(*texts: str | None) -> str | None:
    region = _extract_geo_limited_remote_region(_join_remote_restriction_context(*texts))
    if not region:
        return None
    return f"Remote restriction: {region.title()} only."


def _lead_location_hint_with_remote_restriction(location_text: str | None, *texts: str | None) -> str | None:
    normalized_location = str(location_text or "").strip()
    remote_restriction_note = _remote_restriction_note(normalized_location, *texts)
    if not remote_restriction_note:
        return normalized_location or None
    restriction_suffix = remote_restriction_note.split(": ", 1)[1].rstrip(".")
    if not normalized_location:
        return f"Remote - {restriction_suffix}"
    if restriction_suffix.lower() in normalized_location.lower():
        return normalized_location
    if "remote" in normalized_location.lower():
        return f"{normalized_location} - {restriction_suffix}"
    return f"{normalized_location}. {remote_restriction_note}"


def _trusted_direct_lead_precheck_failure(lead: JobLead, settings: Settings) -> SearchFailure | None:
    if lead.source_type not in {"direct_ats", "company_site"}:
        return None
    precheck_failure = _precheck_lead_hints(
        lead,
        settings,
        attempt_number=0,
        round_number=0,
    )
    if precheck_failure is None or precheck_failure.reason_code not in TRUSTED_DIRECT_LEAD_SUPPRESS_REASON_CODES:
        return None
    return precheck_failure


def _stable_text_index(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return sum(ord(character) for character in text) % modulo


def _builtin_search_base_urls(query: str) -> list[str]:
    return _dedupe_queries([BUILTIN_PRIMARY_BASE_URL, *BUILTIN_REGIONAL_BASE_URLS])


def _builtin_category_paths_for_query(query: str) -> list[str]:
    normalized = query.lower()
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    paths: list[str] = []

    if any(token in normalized for token in ("ai", "llm", "agentic", "agent", "generative", "genai", "applied ai")):
        paths.append("/jobs/remote/product/artificial-intelligence")
        paths.append("/jobs/remote/artificial-intelligence")
    if "machine learning" in normalized or "ml" in compact:
        paths.append("/jobs/remote/product/machine-learning")
        paths.append("/jobs/remote/machine-learning")
    if "product manager" in normalized:
        paths.append("/jobs/remote/product")

    return _dedupe_queries(paths)


def _builtin_category_urls_for_query(query: str) -> list[str]:
    urls: list[str] = []
    for base_url in _builtin_search_base_urls(query):
        for path in _builtin_category_paths_for_query(query):
            urls.append(f"{base_url}{path}")
    return _dedupe_queries(urls)


def _builtin_paginated_category_urls(url: str, page_count: int) -> list[str]:
    effective_page_count = max(1, page_count)
    urls = [url]
    for page_number in range(2, effective_page_count + 1):
        urls.append(f"{url}?page={page_number}")
    return urls


def _builtin_listing_looks_relevant(title: str, description: str, query: str) -> bool:
    haystack = " ".join(part for part in (title, description, query) if part).lower()
    if "product manager" not in haystack:
        return False
    return _is_ai_related_product_manager_text(haystack)


def _chunk_queries(queries: list[str], chunk_size: int) -> list[list[str]]:
    effective_size = max(1, chunk_size)
    return [queries[index : index + effective_size] for index in range(0, len(queries), effective_size)]


async def _fetch_builtin_page(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, str] | None = None,
) -> str:
    cache_key = str(httpx.URL(url, params=params))
    cached = BUILTIN_HTML_CACHE.get(cache_key)
    if cached is not None:
        return cached

    response = await client.get(url, params=params)
    response.raise_for_status()
    BUILTIN_HTML_CACHE[cache_key] = response.text
    return response.text


async def _fetch_builtin_page_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, str] | None = None,
) -> str:
    last_error: httpx.HTTPError | None = None
    for attempt in range(2):
        try:
            return await _fetch_builtin_page(client, url, params=params)
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt == 0:
                await asyncio.sleep(1.0)
                continue
    if last_error is not None:
        raise last_error
    raise httpx.HTTPError("Built In page fetch failed.")


async def _fetch_builtin_job_lead(
    client: httpx.AsyncClient,
    detail_url: str,
    query: str,
    description_hint: str,
    *,
    source_is_remote_listing: bool = False,
) -> JobLead | None:
    if detail_url in BUILTIN_LEAD_CACHE:
        cached_lead = BUILTIN_LEAD_CACHE[detail_url]
        if cached_lead is None:
            return None
        return cached_lead.model_copy(update={"source_query": query})

    html = await _fetch_builtin_page_with_retry(client, detail_url)
    payload = _extract_builtin_jobposting_payload(html)
    if not payload:
        BUILTIN_LEAD_CACHE[detail_url] = None
        return None

    company_name = _extract_builtin_company(payload) or _company_hint_from_url(detail_url)
    role_title = str(payload.get("title") or "").strip()
    if not role_title:
        company_name, role_title = _extract_role_company_from_title(detail_url, detail_url)

    description_text = ""
    description_html = payload.get("description")
    if isinstance(description_html, str):
        description_text = BeautifulSoup(description_html, "html.parser").get_text(" ", strip=True)
    description_text = description_text or description_hint

    direct_url = _extract_builtin_apply_url(html)
    salary_min, salary_max, salary_text = _extract_salary_hint(description_text)
    date_posted = payload.get("datePosted")
    posted_hint = date_posted.strip() if isinstance(date_posted, str) and date_posted.strip() else _extract_posted_hint(description_text)
    location_text = _extract_builtin_location(payload)
    is_remote = _extract_builtin_remote_hint(
        location_text,
        description_text,
        source_is_remote_listing=source_is_remote_listing,
    )
    remote_restriction_note = _remote_restriction_note(location_text, description_text)
    location_hint = location_text or ("Remote" if is_remote else None)
    if remote_restriction_note and is_remote:
        location_hint = f"Remote - {remote_restriction_note.split(': ', 1)[1].rstrip('.')}"

    normalized_direct_url = _normalize_direct_job_url(direct_url) if direct_url else None
    if normalized_direct_url:
        provisional_lead = JobLead(
            company_name=company_name,
            role_title=role_title or description_hint or "Product Manager",
            source_url=detail_url,
            source_type="builtin",
            direct_job_url=normalized_direct_url,
            location_hint=location_hint,
            posted_date_hint=posted_hint,
            is_remote_hint=is_remote,
            base_salary_min_usd_hint=salary_min,
            base_salary_max_usd_hint=salary_max,
            salary_text_hint=salary_text,
            source_query=query,
            evidence_notes="",
        )
        if not _candidate_direct_job_url_is_trustworthy(normalized_direct_url, provisional_lead):
            normalized_direct_url = None

    lead = JobLead(
        company_name=company_name,
        role_title=role_title or description_hint or "Product Manager",
        source_url=detail_url,
        source_type="builtin",
        direct_job_url=normalized_direct_url,
        location_hint=location_hint,
        posted_date_hint=posted_hint,
        is_remote_hint=is_remote,
        base_salary_min_usd_hint=salary_min,
        base_salary_max_usd_hint=salary_max,
        salary_text_hint=salary_text,
        source_query=query,
        evidence_notes=(
            " ".join(
                part
                for part in (
                    remote_restriction_note,
                    description_text or description_hint or f"Built In result for '{query}'.",
                )
                if part
            )
        )[:500],
    )
    BUILTIN_LEAD_CACHE[detail_url] = lead.model_copy(update={"source_query": ""})
    return lead


def _builtin_search_terms_for_query(query: str) -> list[str]:
    normalized = query.lower()
    normalized = re.sub(r"site:[^\s]+", " ", normalized)
    normalized = re.sub(r'"\$[\d,]+"', " ", normalized)
    normalized = normalized.replace('"posted this week"', " ")
    normalized = normalized.replace('"posted in the last week"', " ")
    normalized = normalized.replace("remote", " ")
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("&", " ")
    normalized = normalized.replace('"', " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    candidates: list[str] = []
    if normalized:
        candidates.append(normalized)

    normalized_compact = normalized.replace(" ", "")
    broad_terms = {
        "aiproductmanager": "AI product manager",
        "machinelearningproductmanager": "machine learning product manager",
        "generativeaiproductmanager": "generative AI product manager",
        "aimlproductmanager": "machine learning product manager",
        "seniorproductmanagerai": "senior product manager AI",
        "staffproductmanagerai": "staff product manager AI",
        "principalproductmanagerai": "principal product manager AI",
        "groupproductmanagerai": "group product manager AI",
        "leadproductmanagerai": "lead product manager AI",
        "technicalproductmanagerai": "technical product manager AI",
        "aiplatformproductmanager": "AI platform product manager",
        "agenticaiproductmanager": "agentic AI product manager",
        "llmproductmanager": "LLM product manager",
    }
    for needle, fallback in broad_terms.items():
        if needle in normalized_compact:
            candidates.append(fallback)

    seniority = None
    for label in ("principal", "staff", "senior", "group", "lead", "technical"):
        if re.search(rf"\b{label}\b", normalized):
            seniority = label
            break

    if "machine learning" in normalized or "ml" in normalized_compact:
        candidates.append("machine learning product manager")
        candidates.append("AI product manager")
        if seniority:
            candidates.append(f"{seniority} product manager machine learning")
            candidates.append(f"{seniority} product manager AI")

    if "agent" in normalized:
        candidates.append("agentic AI product manager")
        candidates.append("AI product manager")
        if seniority:
            candidates.append(f"{seniority} product manager AI")

    if "llm" in normalized:
        candidates.append("LLM product manager")
        candidates.append("AI product manager")

    return _dedupe_queries(candidates)[:2]


def _extract_linkedin_guest_search_leads(html: str, query: str) -> list[JobLead]:
    soup = BeautifulSoup(html, "html.parser")
    leads: list[JobLead] = []

    for card in soup.select("div.base-search-card"):
        title_el = card.select_one("h3.base-search-card__title")
        company_el = card.select_one("h4.base-search-card__subtitle a, h4.base-search-card__subtitle")
        link_el = card.select_one("a.base-card__full-link")
        location_el = card.select_one(".job-search-card__location")
        time_el = card.select_one("time.job-search-card__listdate, time.job-search-card__listdate--new, time")

        role_title = title_el.get_text(" ", strip=True) if title_el else ""
        company_name = company_el.get_text(" ", strip=True) if company_el else ""
        source_url = _normalize_direct_job_url(link_el.get("href", "").strip()) if link_el else ""
        location_text = location_el.get_text(" ", strip=True) if location_el else None
        posted_hint = None
        if time_el:
            posted_hint = (time_el.get("datetime") or time_el.get_text(" ", strip=True) or "").strip() or None
        if not role_title or not company_name or not source_url.startswith(("http://", "https://")):
            continue

        remote_hint = True
        if location_text and any(token in location_text.lower() for token in ("hybrid", "on-site", "onsite", "in office")):
            remote_hint = False

        evidence_parts = [
            f"LinkedIn guest search result for '{query}'.",
            f"Location: {location_text}" if location_text else "",
            f"Posted: {posted_hint}" if posted_hint else "",
        ]
        leads.append(
            JobLead(
                company_name=company_name,
                role_title=role_title,
                source_url=source_url,
                source_type="linkedin",
                direct_job_url=None,
                location_hint=location_text,
                posted_date_hint=posted_hint,
                is_remote_hint=remote_hint,
                source_query=query,
                evidence_notes=" ".join(part for part in evidence_parts if part).strip(),
            )
        )
    return leads


async def _linkedin_guest_search(query: str, settings: Settings) -> list[JobLead]:
    leads: list[JobLead] = []
    seen_urls: set[str] = set()
    max_candidates = max(settings.max_leads_per_query * 2, 20)
    recency_seconds = settings.posted_within_days * 24 * 60 * 60
    async with httpx.AsyncClient(
        timeout=12.0,
        follow_redirects=True,
        headers=SEARCH_ENGINE_HEADERS,
    ) as client:
        for start in (0, 25):
            response = await client.get(
                LINKEDIN_GUEST_SEARCH_BASE_URL,
                params={
                    "keywords": query,
                    "location": "United States",
                    "f_WT": "2",
                    "f_TPR": f"r{recency_seconds}",
                    "start": str(start),
                },
            )
            response.raise_for_status()
            for lead in _extract_linkedin_guest_search_leads(response.text, query):
                key = _lead_dedupe_key(lead)
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                leads.append(lead)
                if len(leads) >= max_candidates:
                    return leads
    return leads


async def _builtin_search(query: str, settings: Settings) -> list[JobLead]:
    search_terms = _builtin_search_terms_for_query(query)
    if not search_terms:
        return []

    leads: list[JobLead] = []
    seen_detail_urls: set[str] = set()
    max_candidates = max(settings.max_leads_per_query * 2, 12)
    successful_searches = 0
    last_error: Exception | None = None
    search_base_urls = _builtin_search_base_urls(query)
    category_urls = _builtin_category_urls_for_query(query)

    async with httpx.AsyncClient(
        timeout=20.0,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for search_term in search_terms:
            for base_url in search_base_urls:
                try:
                    page_html = await _fetch_builtin_page_with_retry(
                        client,
                        f"{base_url}/jobs",
                        params={"search": search_term, "location": "Remote"},
                    )
                    successful_searches += 1
                except httpx.HTTPError as exc:
                    last_error = exc
                    continue
                for detail_url, title, description in _extract_builtin_search_items(page_html):
                    if detail_url in seen_detail_urls or not _builtin_listing_looks_relevant(title, description, query):
                        continue
                    seen_detail_urls.add(detail_url)
                    try:
                        lead = await _fetch_builtin_job_lead(
                            client,
                            detail_url,
                            query,
                            description,
                            source_is_remote_listing=True,
                        )
                    except Exception:
                        continue
                    if lead is None:
                        continue
                    leads.append(lead)
                    if len(leads) >= max_candidates:
                        break
                if len(leads) >= max_candidates:
                    break
            if len(leads) >= max_candidates:
                break

        if len(leads) < max_candidates:
            for category_url in category_urls:
                for paginated_url in _builtin_paginated_category_urls(category_url, BUILTIN_CATEGORY_PAGE_COUNT):
                    try:
                        page_html = await _fetch_builtin_page_with_retry(client, paginated_url)
                        successful_searches += 1
                    except httpx.HTTPError as exc:
                        last_error = exc
                        continue
                    for detail_url, title, description in _extract_builtin_search_items(page_html):
                        if detail_url in seen_detail_urls or not _builtin_listing_looks_relevant(title, description, query):
                            continue
                        seen_detail_urls.add(detail_url)
                        try:
                            lead = await _fetch_builtin_job_lead(
                                client,
                                detail_url,
                                query,
                                description,
                                source_is_remote_listing=True,
                            )
                        except Exception:
                            continue
                        if lead is None:
                            continue
                        leads.append(lead)
                        if len(leads) >= max_candidates:
                            break
                    if len(leads) >= max_candidates:
                        break
                if len(leads) >= max_candidates:
                    break
    if successful_searches == 0 and last_error is not None:
        raise last_error
    return leads


def _deterministic_trim_local_leads(
    settings: Settings,
    query: str,
    leads: list[JobLead],
    *,
    limit: int | None = None,
) -> list[JobLead]:
    normalized = _normalize_and_filter_discovery_leads(leads, query)
    normalized.sort(key=lambda lead: _lead_priority(lead, settings))
    deduped: list[JobLead] = []
    seen_role_keys: set[str] = set()
    for lead in normalized:
        role_key = _normalize_job_key(lead.company_name, lead.role_title)
        if role_key in seen_role_keys:
            continue
        seen_role_keys.add(role_key)
        deduped.append(lead)
    effective_limit = settings.max_leads_per_query if limit is None else max(1, limit)
    return deduped[:effective_limit]


def _dedupe_round_leads(leads: list[JobLead], settings: Settings) -> list[JobLead]:
    best_by_role: dict[str, JobLead] = {}
    for lead in leads:
        role_key = _normalize_job_key(lead.company_name, lead.role_title)
        existing = best_by_role.get(role_key)
        if existing is None or _lead_priority(lead, settings) < _lead_priority(existing, settings):
            best_by_role[role_key] = lead
    deduped = sorted(best_by_role.values(), key=lambda lead: _lead_priority(lead, settings))
    balanced: list[JobLead] = []
    company_counts: Counter[str] = Counter()
    for lead in deduped:
        company_key = _normalize_company_key(lead.company_name)
        if company_key and company_counts[company_key] >= MAX_ROUND_LEADS_PER_COMPANY:
            continue
        if company_key:
            company_counts[company_key] += 1
        balanced.append(lead)
    return balanced


def _load_seed_leads_from_file(settings: Settings) -> list[JobLead]:
    path = settings.data_dir / SEED_LEADS_FILENAME
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    raw_leads = payload.get("leads") if isinstance(payload, dict) else payload
    if not isinstance(raw_leads, list):
        return []

    leads: list[JobLead] = []
    for item in raw_leads:
        if not isinstance(item, dict):
            continue
        try:
            lead = JobLead.model_validate(item)
        except Exception:
            continue
        if not lead.direct_job_url or not _is_allowed_direct_job_url(lead.direct_job_url):
            continue
        if not _lead_is_ai_related_product_manager(lead):
            continue
        leads.append(lead)
    return leads


def _seed_lead_from_failure(failure: SearchFailure) -> JobLead | None:
    if failure.reason_code not in REPLAYABLE_FAILURE_REASON_CODES:
        return None
    if not failure.company_name or not failure.role_title:
        return None
    direct_job_url = _normalize_direct_job_url(failure.direct_job_url or "")
    if direct_job_url and _is_low_trust_replay_source_url(direct_job_url):
        direct_job_url = ""
    source_url = str(failure.source_url or direct_job_url).strip()
    if _is_low_trust_replay_source_url(source_url):
        return None
    if not source_url.startswith(("http://", "https://")):
        if direct_job_url:
            source_url = direct_job_url
        else:
            return None
    if direct_job_url and (not _is_allowed_direct_job_url(direct_job_url) or _looks_like_generic_job_url(direct_job_url)):
        direct_job_url = ""
    if not direct_job_url and failure.reason_code not in RESOLUTION_REASON_CODES:
        return None
    salary_min, salary_max, salary_text = _hydrate_salary_hint_values(None, None, failure.salary_text, failure.detail)
    source_type = _normalize_source_type(direct_job_url or source_url)
    replay_remote_haystack = " ".join(
        part
        for part in (
            failure.role_title,
            source_url,
            direct_job_url,
            failure.detail or "",
        )
        if part
    ).lower()
    is_remote_hint = failure.is_remote
    if is_remote_hint is None:
        lowered_query = " ".join(str(failure.source_query or "").lower().split())
        if (
            source_type in {"direct_ats", "company_site"}
            and not any(token in replay_remote_haystack for token in ("hybrid", "on-site", "onsite", "in office"))
            and (" remote" in f" {lowered_query}" or "work from home" in lowered_query)
        ):
            is_remote_hint = True
    remote_restriction_note = _remote_restriction_note(
        failure.role_title,
        failure.detail,
        source_url,
        direct_job_url or None,
    )
    location_hint: str | None = None
    if is_remote_hint:
        location_hint = "Remote"
        if remote_restriction_note:
            location_hint = f"Remote - {remote_restriction_note.split(': ', 1)[1].rstrip('.')}"
    lead = JobLead(
        company_name=failure.company_name,
        role_title=failure.role_title,
        source_url=source_url,
        source_type=source_type,
        direct_job_url=direct_job_url or None,
        location_hint=location_hint,
        posted_date_hint=failure.posted_date_text,
        is_remote_hint=is_remote_hint,
        base_salary_min_usd_hint=salary_min,
        base_salary_max_usd_hint=salary_max,
        salary_text_hint=salary_text,
        source_query=failure.source_query,
        evidence_notes=(
            " ".join(
                part
                for part in (
                    (failure.detail or "Historical lead from prior diagnostics.")[:400],
                    remote_restriction_note,
                )
                if part
            )
        )[:400],
    )
    if not _lead_is_ai_related_product_manager(lead):
        return None
    if not _lead_is_replay_source_trustworthy(lead):
        return None
    return lead


def _seed_lead_from_job_payload(payload: dict[str, object]) -> JobLead | None:
    try:
        job = JobPosting.model_validate(payload)
    except Exception:
        return None
    direct_job_url = _normalize_direct_job_url(job.direct_job_url)
    if not _is_allowed_direct_job_url(direct_job_url):
        return None
    lead = JobLead(
        company_name=job.company_name,
        role_title=job.role_title,
        source_url=direct_job_url,
        source_type=_normalize_source_type(direct_job_url),
        direct_job_url=direct_job_url,
        location_hint=_lead_location_hint_with_remote_restriction(
            job.location_text or ("Remote" if job.is_fully_remote else None),
            job.role_title,
            job.job_page_title,
            job.evidence_notes,
        ),
        posted_date_hint=job.posted_date_iso or job.posted_date_text,
        is_remote_hint=job.is_fully_remote,
        base_salary_min_usd_hint=job.base_salary_min_usd,
        base_salary_max_usd_hint=job.base_salary_max_usd,
        salary_text_hint=job.salary_text,
        source_query=job.source_query,
        evidence_notes=(
            " ".join(
                part
                for part in (
                    (job.evidence_notes or "Historical accepted candidate.")[:400],
                    _remote_restriction_note(job.role_title, job.job_page_title, job.evidence_notes),
                )
                if part
            )
        )[:400],
    )
    if not _lead_is_ai_related_product_manager(lead):
        return None
    return lead


def _load_historical_seed_leads(settings: Settings) -> list[JobLead]:
    leads: list[JobLead] = []
    for path in sorted(settings.data_dir.glob("run-*.json"), reverse=True)[:20]:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        for bundle in payload.get("bundles", []):
            if not isinstance(bundle, dict):
                continue
            job_payload = bundle.get("job")
            if isinstance(job_payload, dict):
                lead = _seed_lead_from_job_payload(job_payload)
                if lead is not None:
                    leads.append(lead)
        diagnostics = payload.get("search_diagnostics", {})
        failures = diagnostics.get("failures") if isinstance(diagnostics, dict) else None
        if not isinstance(failures, list):
            continue
        for raw_failure in failures:
            if not isinstance(raw_failure, dict):
                continue
            try:
                failure = SearchFailure.model_validate(raw_failure)
            except Exception:
                continue
            lead = _seed_lead_from_failure(failure)
            if lead is not None:
                leads.append(lead)
    return leads


def _collect_replay_seed_leads(settings: Settings) -> list[JobLead]:
    file_seed_leads = _load_seed_leads_from_file(settings)
    curated_seed_keys = {_lead_dedupe_key(lead) for lead in file_seed_leads}
    deduped: dict[str, JobLead] = {}
    for lead in [*file_seed_leads, *_load_historical_seed_leads(settings)]:
        if not _lead_is_replay_source_trustworthy(lead):
            continue
        if _trusted_direct_lead_precheck_failure(lead, settings) is not None:
            continue
        key = _lead_dedupe_key(lead)
        existing = deduped.get(key)
        if existing is None or _lead_priority(lead, settings) < _lead_priority(existing, settings):
            deduped[key] = lead
    return sorted(
        deduped.values(),
        key=lambda lead: (
            0 if _lead_dedupe_key(lead) in curated_seed_keys else 1,
            _lead_priority(lead, settings),
        ),
    )


async def _replay_seed_leads(
    seed_leads: list[JobLead],
    *,
    settings: Settings,
    diagnostics: SearchDiagnostics,
    company_watchlist: dict[str, dict[str, object]],
    failed_lead_history: dict[str, dict[str, object]],
    jobs_by_url: dict[str, JobPosting],
    reacquired_jobs_by_url: dict[str, JobPosting],
    previously_reported_company_keys: set[str],
    validated_job_history_index: dict[str, dict[str, object]],
    reacquisition_attempted_keys: set[str],
    reacquisition_suppressed_keys: set[str],
    seen_lead_keys: set[str],
    total_unique_leads: int,
    resolved_leads_this_attempt: int,
    stop_goal: int,
    lead_timeout_seconds: int,
    resolution_agent: Agent | None,
    attempt_number: int,
    status: StatusReporter | None,
    run_id: str | None = None,
    track_as_seed_replay: bool = True,
    company_discovery_entries: dict[str, dict[str, object]] | None = None,
    status_label: str | None = None,
) -> tuple[int, int]:
    if not seed_leads:
        return total_unique_leads, resolved_leads_this_attempt
    if settings.company_discovery_enabled and company_discovery_entries is None:
        company_discovery_entries = load_company_discovery_entries(settings.data_dir)

    replay_seed_leads = _dedupe_round_leads(seed_leads, settings)
    replay_seed_leads = await _maybe_force_seed_lead_refinement_with_ollama(
        settings,
        replay_seed_leads,
        run_id=run_id,
    )

    fresh_seed_leads: list[JobLead] = []
    for lead in replay_seed_leads:
        failed_history_skip = _failed_lead_history_skip_reason(lead, settings, failed_lead_history)
        if failed_history_skip is not None:
            reason_code, detail = failed_history_skip
            _record_failure_live(
                settings,
                diagnostics,
                _make_failure(
                    stage="filter",
                    reason_code="repeated_failed_lead",
                    detail=detail,
                    lead=lead,
                    attempt_number=attempt_number,
                    round_number=0,
                ),
                unique_leads_discovered=total_unique_leads,
            )
            continue
        key = _lead_dedupe_key(lead)
        if key in seen_lead_keys:
            continue
        seen_lead_keys.add(key)
        total_unique_leads += 1
        _increment_metric_count(diagnostics.company_lead_counts, _normalize_company_key(lead.company_name))
        fresh_seed_leads.append(lead)

    fresh_seed_leads = _dedupe_round_leads(fresh_seed_leads, settings)
    fresh_seed_leads = _annotate_and_filter_resolution_leads(fresh_seed_leads, settings, company_watchlist)
    seed_replay_cap = min(
        len(fresh_seed_leads),
        max(settings.max_leads_per_query * 4, settings.max_leads_to_resolve_per_pass // 2),
    )
    fresh_seed_leads = _apply_company_novelty_quota(
        fresh_seed_leads,
        previously_reported_company_keys,
        min_novelty_ratio=NOVEL_COMPANY_TARGET_RATIO,
        limit=seed_replay_cap,
    )
    if 0 < len(fresh_seed_leads) < len(replay_seed_leads):
        fresh_seed_leads = await _maybe_force_seed_lead_refinement_with_ollama(
            settings,
            fresh_seed_leads,
            run_id=run_id,
        )
    if track_as_seed_replay:
        diagnostics.seed_replayed_lead_count += len(fresh_seed_leads)
    if status:
        status.emit(
            "search",
            status_label
            or f"Pass {attempt_number}, replaying {len(fresh_seed_leads)} seeded ATS candidates from prior findings.",
            attempt_number=attempt_number,
            round_number=0,
            unique_leads_discovered=total_unique_leads,
            qualifying_jobs=len(jobs_by_url),
        )

    for lead_index, lead in enumerate(fresh_seed_leads, start=1):
        if len(jobs_by_url) >= stop_goal:
            break
        if resolved_leads_this_attempt >= settings.max_leads_to_resolve_per_pass:
            break

        precheck_failure = _precheck_lead_hints(
            lead,
            settings,
            attempt_number=attempt_number,
            round_number=0,
        )
        if precheck_failure is not None:
            _record_failure_with_followups(
                settings,
                diagnostics,
                precheck_failure,
                unique_leads_discovered=total_unique_leads,
                lead=lead,
                audit_entry=_build_false_negative_audit_entry(lead, precheck_failure),
                run_id=run_id,
            )
            continue

        replay_lead = lead
        replay_resolution: DirectJobResolution | None = None

        if lead.direct_job_url and _is_allowed_direct_job_url(lead.direct_job_url):
            replay_resolution = DirectJobResolution(
                accepted=True,
                direct_job_url=lead.direct_job_url,
                ats_platform=urlparse(lead.direct_job_url).netloc or "Unknown",
                evidence_notes="Revalidated from replayed seed leads.",
            )

        if replay_resolution is not None:
            replay_lead = lead.model_copy(
                update={"direct_job_url": _normalize_direct_job_url(replay_resolution.direct_job_url or lead.direct_job_url)}
            )

        reacquisition_entry = (
            _validated_job_history_entry_for_url(replay_lead.direct_job_url, validated_job_history_index)
            if replay_lead.direct_job_url
            else None
        )
        reacquisition_key = str((reacquisition_entry or {}).get("canonical_job_key") or (reacquisition_entry or {}).get("job_key") or "").strip()
        if reacquisition_entry is not None:
            if not _lead_is_reacquisition_eligible(replay_lead, settings, direct_job_url=replay_lead.direct_job_url):
                if reacquisition_key and reacquisition_key not in reacquisition_suppressed_keys:
                    reacquisition_suppressed_keys.add(reacquisition_key)
                    diagnostics.reacquired_jobs_suppressed_count += 1
                _record_failure_live(
                    settings,
                    diagnostics,
                    _make_failure(
                        stage="filter",
                        reason_code="reacquisition_suppressed",
                        detail="Skipping a previously validated job because the repeat hit came from a low-trust or non-direct source.",
                        lead=replay_lead,
                        direct_job_url=replay_lead.direct_job_url,
                        attempt_number=attempt_number,
                        round_number=0,
                    ),
                    unique_leads_discovered=total_unique_leads,
                )
                continue
            if len(reacquisition_attempted_keys) >= settings.reacquisition_attempt_cap and (
                not reacquisition_key or reacquisition_key not in reacquisition_attempted_keys
            ):
                if reacquisition_key and reacquisition_key not in reacquisition_suppressed_keys:
                    reacquisition_suppressed_keys.add(reacquisition_key)
                    diagnostics.reacquired_jobs_suppressed_count += 1
                _record_failure_live(
                    settings,
                    diagnostics,
                    _make_failure(
                        stage="filter",
                        reason_code="reacquisition_suppressed",
                        detail=(
                            "Skipping a previously validated job because the per-run reacquisition cap "
                            f"({settings.reacquisition_attempt_cap}) was reached."
                        ),
                        lead=replay_lead,
                        direct_job_url=replay_lead.direct_job_url,
                        attempt_number=attempt_number,
                        round_number=0,
                    ),
                    unique_leads_discovered=total_unique_leads,
                )
                continue
            if reacquisition_key and reacquisition_key not in reacquisition_attempted_keys:
                reacquisition_attempted_keys.add(reacquisition_key)
                diagnostics.reacquisition_attempt_count += 1

        if status:
            status.emit(
                "search",
                f"Revalidating seeded lead {lead_index}/{len(fresh_seed_leads)}: {lead.company_name} | {lead.role_title}",
                attempt_number=attempt_number,
                round_number=0,
                lead_index=lead_index,
                round_lead_count=len(fresh_seed_leads),
                unique_leads_discovered=total_unique_leads,
                qualifying_jobs=len(jobs_by_url),
            )

        resolved_leads_this_attempt += 1
        if replay_resolution is None:
            try:
                replay_resolution = await asyncio.wait_for(
                    _resolve_lead_to_direct_job_url(resolution_agent, lead),
                    timeout=lead_timeout_seconds,
                )
            except asyncio.TimeoutError:
                failure = _make_failure(
                    stage="resolution",
                    reason_code="resolution_timeout",
                    detail=f"Resolution timed out after {lead_timeout_seconds}s.",
                    lead=lead,
                    attempt_number=attempt_number,
                    round_number=0,
                )
                _record_failure_with_followups(
                    settings,
                    diagnostics,
                    failure,
                    unique_leads_discovered=total_unique_leads,
                    lead=lead,
                    audit_entry=_build_false_negative_audit_entry(lead, failure),
                    run_id=run_id,
                )
                continue
            except Exception as exc:
                if _is_insufficient_quota_error(exc):
                    _record_failure_live(
                        settings,
                        diagnostics,
                        _make_failure(
                            stage="resolution",
                            reason_code="openai_insufficient_quota",
                            detail=f"Resolution failed because the OpenAI API quota was exhausted: {exc}",
                            lead=lead,
                            attempt_number=attempt_number,
                            round_number=0,
                        ),
                        unique_leads_discovered=total_unique_leads,
                    )
                    raise OpenAIQuotaExceededError(
                        "OpenAI API quota is exhausted. Add billing or increase quota, then rerun the workflow."
                    ) from exc
                failure = _make_failure(
                    stage="resolution",
                    reason_code="resolution_error",
                    detail=f"Resolution failed with an exception: {exc}",
                    lead=lead,
                    attempt_number=attempt_number,
                    round_number=0,
                )
                _record_failure_with_followups(
                    settings,
                    diagnostics,
                    failure,
                    unique_leads_discovered=total_unique_leads,
                    lead=lead,
                    audit_entry=_build_false_negative_audit_entry(lead, failure),
                    run_id=run_id,
                )
                continue

            resolved_direct_job_url = _normalize_direct_job_url(replay_resolution.direct_job_url if replay_resolution else "")
            if not resolved_direct_job_url or not _is_allowed_direct_job_url(resolved_direct_job_url):
                failure = _make_failure(
                    stage="resolution",
                    reason_code="resolution_missing",
                    detail="Seeded lead could not be resolved to a valid direct ATS URL.",
                    lead=lead,
                    attempt_number=attempt_number,
                    round_number=0,
                )
                candidate = _build_candidate_job(
                    lead,
                    DirectJobResolution(
                        accepted=True,
                        direct_job_url=lead.source_url,
                        ats_platform=urlparse(lead.source_url).netloc or "Unknown",
                        evidence_notes="Near-miss fallback from seed replay.",
                    ),
                )
                near_miss = _build_near_miss(lead, candidate, failure, settings)
                _record_failure_with_followups(
                    settings,
                    diagnostics,
                    failure,
                    unique_leads_discovered=total_unique_leads,
                    lead=lead,
                    candidate=candidate,
                    near_miss=near_miss,
                    audit_entry=_build_false_negative_audit_entry(lead, failure, candidate=candidate, near_miss=near_miss),
                    run_id=run_id,
                )
                continue

            replay_lead = lead.model_copy(update={"direct_job_url": resolved_direct_job_url})
            replay_resolution = replay_resolution.model_copy(update={"direct_job_url": resolved_direct_job_url})

        candidate = _build_candidate_job(replay_lead, replay_resolution)
        try:
            validated, validation_failure, near_miss, audit_entry = await asyncio.wait_for(
                _validate_candidate(
                    replay_lead,
                    candidate,
                    settings,
                    resolution_agent=resolution_agent,
                    attempt_number=attempt_number,
                    round_number=0,
                ),
                timeout=lead_timeout_seconds,
            )
        except asyncio.TimeoutError:
            failure = _make_failure(
                stage="validation",
                reason_code="validation_timeout",
                detail=f"Validation timed out after {lead_timeout_seconds}s.",
                lead=replay_lead,
                direct_job_url=str(candidate.direct_job_url),
                candidate=candidate,
                attempt_number=attempt_number,
                round_number=0,
            )
            _record_failure_with_followups(
                settings,
                diagnostics,
                failure,
                unique_leads_discovered=total_unique_leads,
                lead=replay_lead,
                candidate=candidate,
                audit_entry=_build_false_negative_audit_entry(replay_lead, failure, candidate=candidate),
                run_id=run_id,
            )
            continue
        except Exception as exc:
            failure = _make_failure(
                stage="validation",
                reason_code="validation_error",
                detail=f"Validation failed with an exception: {exc}",
                lead=replay_lead,
                direct_job_url=str(candidate.direct_job_url),
                candidate=candidate,
                attempt_number=attempt_number,
                round_number=0,
            )
            _record_failure_with_followups(
                settings,
                diagnostics,
                failure,
                unique_leads_discovered=total_unique_leads,
                lead=replay_lead,
                candidate=candidate,
                audit_entry=_build_false_negative_audit_entry(replay_lead, failure, candidate=candidate),
                run_id=run_id,
            )
            continue

        if validation_failure is not None:
            _record_failure_with_followups(
                settings,
                diagnostics,
                validation_failure,
                unique_leads_discovered=total_unique_leads,
                lead=replay_lead,
                candidate=candidate,
                near_miss=near_miss,
                audit_entry=audit_entry,
                run_id=run_id,
            )
            continue

        if validated is None:
            continue

        if validated.salary_inference_kind == "salary_presumed_from_principal_ai_pm":
            diagnostics.principal_ai_pm_salary_presumption_count += 1
        validated_key = _job_posting_dedupe_key(validated)
        post_validation_reacquisition_entry = reacquisition_entry or _validated_job_history_entry_for_url(
            validated.resolved_job_url or validated.direct_job_url,
            validated_job_history_index,
        )
        if post_validation_reacquisition_entry is not None and reacquisition_entry is None:
            post_key = str(
                post_validation_reacquisition_entry.get("canonical_job_key")
                or post_validation_reacquisition_entry.get("job_key")
                or ""
            ).strip()
            if len(reacquisition_attempted_keys) >= settings.reacquisition_attempt_cap and (
                not post_key or post_key not in reacquisition_attempted_keys
            ):
                if post_key and post_key not in reacquisition_suppressed_keys:
                    reacquisition_suppressed_keys.add(post_key)
                    diagnostics.reacquired_jobs_suppressed_count += 1
                _record_failure_live(
                    settings,
                    diagnostics,
                    _make_failure(
                        stage="filter",
                        reason_code="reacquisition_suppressed",
                        detail=(
                            "Validated a previously reported job, but skipped coverage credit because the per-run "
                            f"reacquisition cap ({settings.reacquisition_attempt_cap}) was reached."
                        ),
                        lead=lead,
                        direct_job_url=str(validated.direct_job_url),
                        candidate=validated,
                        attempt_number=attempt_number,
                        round_number=0,
                    ),
                    unique_leads_discovered=total_unique_leads,
                )
                continue
            if post_key and post_key not in reacquisition_attempted_keys:
                reacquisition_attempted_keys.add(post_key)
                diagnostics.reacquisition_attempt_count += 1
        if post_validation_reacquisition_entry is not None:
            metadata = _reacquisition_history_metadata(post_validation_reacquisition_entry)
            validated = validated.model_copy(update={"is_reacquired": True, **metadata})
            reacquired_jobs_by_url.setdefault(validated_key, validated)
        else:
            jobs_by_url.setdefault(validated_key, validated)
        if settings.company_discovery_enabled and company_discovery_entries is not None:
            _upsert_company_discovery_from_validated_job(
                company_discovery_entries,
                validated,
                run_id=run_id,
            )
            save_company_discovery_entries(settings.data_dir, company_discovery_entries)
        diagnostics.unique_leads_discovered = total_unique_leads
        _persist_search_diagnostics(settings, diagnostics)
        _persist_validated_jobs_checkpoint(
            settings,
            jobs_by_url,
            reacquired_jobs_by_url=reacquired_jobs_by_url,
            run_id=run_id,
            diagnostics=diagnostics,
        )
        if status:
            status.emit(
                "search",
                (
                    f"Reacquired still-open job {len(reacquired_jobs_by_url)}: {validated.company_name} | {validated.role_title}"
                    if validated.is_reacquired
                    else f"Accepted job {len(jobs_by_url)}/{stop_goal}: {validated.company_name} | {validated.role_title}"
                ),
                attempt_number=attempt_number,
                round_number=0,
                unique_leads_discovered=total_unique_leads,
                qualifying_jobs=len(jobs_by_url),
                jobs_kept_after_validation=len(jobs_by_url),
            )

    return total_unique_leads, resolved_leads_this_attempt


def _build_lead_from_search_result(url: str, title: str, snippet: str, query: str) -> JobLead | None:
    normalized_url = _normalize_direct_job_url(url)
    if not normalized_url.startswith(("http://", "https://")):
        return None
    company_name, role_title = _extract_role_company_from_title(title, normalized_url)
    salary_min, salary_max, salary_text = _extract_salary_hint(snippet)
    posted_hint = _extract_posted_hint(snippet)
    source_type = _normalize_source_type(normalized_url)
    direct_url = normalized_url if _is_allowed_direct_job_url(normalized_url) and not _looks_like_generic_job_url(normalized_url) else None
    remote_haystack = " ".join(part for part in (title, snippet, normalized_url) if part).lower()
    if any(token in remote_haystack for token in ("hybrid", "on-site", "onsite", "in office")):
        is_remote = False
    elif "remote" in remote_haystack or "work from home" in remote_haystack:
        is_remote = True
    else:
        is_remote = None
        lowered_query = " ".join(query.lower().split())
        if (
            source_type in {"direct_ats", "company_site"}
            and (" remote" in f" {lowered_query}" or "work from home" in lowered_query)
        ):
            is_remote = True
    remote_restriction_note = _remote_restriction_note(title, snippet)
    location_hint = None
    if is_remote:
        location_hint = "Remote"
        if remote_restriction_note:
            location_hint = f"Remote - {remote_restriction_note.split(': ', 1)[1].rstrip('.')}"

    lead = JobLead(
        company_name=company_name or "Unknown",
        role_title=role_title or title,
        source_url=normalized_url,
        source_type=source_type,
        direct_job_url=direct_url,
        location_hint=location_hint,
        posted_date_hint=posted_hint,
        is_remote_hint=is_remote,
        base_salary_min_usd_hint=salary_min,
        base_salary_max_usd_hint=salary_max,
        salary_text_hint=salary_text,
        source_query=query,
        evidence_notes=(
            " ".join(part for part in (remote_restriction_note, snippet[:400] if snippet else "") if part)
            or f"Search match from query '{query}'."
        ),
    )
    return lead


def _build_local_search_engine_queries(query: str) -> list[str]:
    normalized = " ".join(query.split())
    if not normalized:
        return []
    broad_generic_query = _query_is_broad_generic(normalized)
    startup_ecosystem_query = _query_targets_startup_ecosystem(normalized)

    board_batch_index = _stable_text_index(normalized.lower(), len(LOCAL_SEARCH_JOB_BOARD_DOMAIN_BATCHES))
    ats_batch_index = _stable_text_index(normalized.lower()[::-1], len(LOCAL_SEARCH_DIRECT_ATS_DOMAIN_BATCHES))

    board_domains: list[str] = []
    for offset in range(min(LOCAL_SEARCH_DOMAIN_BATCH_SPAN, len(LOCAL_SEARCH_JOB_BOARD_DOMAIN_BATCHES))):
        board_domains.extend(LOCAL_SEARCH_JOB_BOARD_DOMAIN_BATCHES[(board_batch_index + offset) % len(LOCAL_SEARCH_JOB_BOARD_DOMAIN_BATCHES)])
    ats_domains: list[str] = []
    for offset in range(min(LOCAL_SEARCH_DOMAIN_BATCH_SPAN, len(LOCAL_SEARCH_DIRECT_ATS_DOMAIN_BATCHES))):
        ats_domains.extend(LOCAL_SEARCH_DIRECT_ATS_DOMAIN_BATCHES[(ats_batch_index + offset) % len(LOCAL_SEARCH_DIRECT_ATS_DOMAIN_BATCHES)])

    ats_limit = LOCAL_SEARCH_ATS_DOMAIN_QUERY_LIMIT if broad_generic_query else LOCAL_SEARCH_ATS_DOMAIN_QUERY_LIMIT + 1
    board_limit = 1 if broad_generic_query else (LOCAL_SEARCH_BOARD_DOMAIN_QUERY_LIMIT if startup_ecosystem_query else 1)
    prioritized_ats_domains = list(dict.fromkeys(ats_domains))[:ats_limit]
    prioritized_board_domains = list(dict.fromkeys(board_domains))[:board_limit]

    queries: list[str] = []
    for domain in prioritized_ats_domains:
        queries.append(f"site:{domain} {normalized}")
        queries.append(f"site:{domain} {normalized} remote")

    for domain in prioritized_board_domains:
        queries.append(f"site:{domain} {normalized}")
        queries.append(f"site:{domain} {normalized} remote")

    broad_query_expansions = [
        f"{normalized} remote",
        f'{normalized} "$200,000"',
    ]
    startup_board_expansions = (
        [
            f'site:workatastartup.com/jobs {normalized} remote',
            f'site:getro.com/companies {normalized} remote',
            f'site:ycombinator.com/companies {normalized} remote',
        ]
        if startup_ecosystem_query
        else []
    )
    targeted_query_expansions = [
        f'"{normalized}" remote',
        *broad_query_expansions,
        *startup_board_expansions,
        f'{normalized} "posted this week"',
    ]
    queries.extend(broad_query_expansions if broad_generic_query else targeted_query_expansions)

    deduped = _dedupe_queries(queries)
    total_limit = 10 if broad_generic_query else LOCAL_SEARCH_TOTAL_QUERY_LIMIT
    return deduped[:total_limit]


async def _run_local_search_engine_queries(query: str, *, max_results_per_query: int) -> list[tuple[str, str, str]]:
    queries = _build_local_search_engine_queries(query)
    merged_results: list[tuple[str, str, str]] = []
    seen_urls: set[str] = set()
    useful_result_count = 0
    last_error: Exception | None = None

    for search_query in queries:
        try:
            search_results = await _search_query_across_backends(
                search_query,
                max_results=max(max_results_per_query, 14),
            )
        except Exception as exc:
            last_error = exc
            continue

        for url, title, snippet in search_results:
            normalized_url = _normalize_direct_job_url(url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            merged_results.append((normalized_url, title, snippet))
            if (
                _is_allowed_direct_job_url(normalized_url)
                or _is_supported_discovery_source_url(normalized_url)
                or _looks_like_company_job_page(normalized_url)
            ):
                useful_result_count += 1

        if useful_result_count >= max(max_results_per_query, 12):
            break

    if not merged_results and last_error is not None:
        raise last_error
    return merged_results


async def _ensure_lazy_ollama_prewarm(settings: Settings, *, run_id: str | None) -> bool:
    if settings.llm_provider != "ollama" or settings.ollama_degraded_for_run or run_id is None:
        return not settings.ollama_degraded_for_run
    if run_id in LAZY_OLLAMA_PREWARM_RUNS:
        return True
    LAZY_OLLAMA_PREWARM_RUNS.add(run_id)
    prewarm_ok, prewarm_error, prewarm_duration = await prewarm_ollama_model(settings, run_id=run_id)
    if prewarm_ok:
        record_ollama_event(
            settings,
            "lazy_prewarm_success",
            run_id=run_id,
            model=settings.ollama_model,
            wall_duration_seconds=prewarm_duration,
        )
        return True
    settings.ollama_degraded_for_run = True
    settings.ollama_degraded_reason = f"Ollama lazy prewarm failed before lead refinement: {prewarm_error or 'unknown error'}"
    record_ollama_event(
        settings,
        "lazy_prewarm_failure",
        run_id=run_id,
        model=settings.ollama_model,
        wall_duration_seconds=prewarm_duration,
        error_message=prewarm_error,
    )
    return False


async def _cleanup_local_leads_with_ollama(
    settings: Settings,
    query: str,
    leads: list[JobLead],
    *,
    run_id: str | None = None,
) -> list[JobLead]:
    if not leads or not _ollama_inline_refinement_enabled(settings):
        return leads
    if settings.ollama_degraded_for_run:
        record_ollama_event(
            settings,
            "ollama_degraded_skip",
            run_id=run_id,
            caller="lead_refinement",
            prompt_category="lead_cleanup",
            reason=settings.ollama_degraded_reason,
            skipped_count=len(leads),
        )
        return leads
    if run_id is not None and not await _ensure_lazy_ollama_prewarm(settings, run_id=run_id):
        record_ollama_event(
            settings,
            "ollama_degraded_skip",
            run_id=run_id,
            caller="lead_refinement",
            prompt_category="lead_cleanup",
            reason=settings.ollama_degraded_reason,
            skipped_count=len(leads),
        )
        return leads
    provider = OllamaStructuredProvider(settings)
    compact_candidates = [
        {
            "company_name": lead.company_name,
            "role_title": lead.role_title,
            "source_type": lead.source_type,
            "source_url": lead.source_url,
            "direct_job_url": lead.direct_job_url,
            "location_hint": lead.location_hint,
            "posted_date_hint": lead.posted_date_hint,
            "is_remote_hint": lead.is_remote_hint,
            "salary_text_hint": lead.salary_text_hint,
            "base_salary_min_usd_hint": lead.base_salary_min_usd_hint,
            "base_salary_max_usd_hint": lead.base_salary_max_usd_hint,
            "evidence_notes": lead.evidence_notes[:320],
        }
        for lead in leads
    ]
    max_returned_leads = min(len(leads), max(3, min(settings.max_leads_per_query, 5)))
    prompt = f"""
Search query: {query}

Clean and normalize these potential job leads.
Keep only high-confidence AI-related product manager roles.
Prefer leads whose company slug/URL matches the stated company.
Prefer direct ATS URLs when present.
Never invent fields or rewrite URLs.
Return at most {max_returned_leads} leads.

Candidates:
{json.dumps(compact_candidates, indent=2)}
""".strip()
    try:
        async with LOCAL_OLLAMA_SEMAPHORE:
            output = await provider.generate_structured(
                system_prompt="You clean job lead candidates into strict structured output.",
                user_prompt=prompt,
                schema=JobLeadSearchResult,
                run_id=run_id,
                caller="lead_refinement",
                prompt_category="lead_cleanup",
            )
        cleaned = [lead.model_copy(update={"refined_by_ollama": True}) for lead in output.leads]
        record_ollama_event(
            settings,
            "lead_refinement_outcome",
            run_id=run_id,
            caller="lead_refinement",
            prompt_category="lead_cleanup",
            query=query,
            proposed_lead_count=len(leads),
            returned_lead_count=len(cleaned),
            schema_valid_count=len(cleaned),
            used_output_count=len(cleaned),
            discarded_output_count=0,
        )
        return cleaned
    except LLMProviderError:
        record_ollama_event(
            settings,
            "lead_refinement_outcome",
            run_id=run_id,
            caller="lead_refinement",
            prompt_category="lead_cleanup",
            query=query,
            proposed_lead_count=len(leads),
            returned_lead_count=0,
            schema_valid_count=0,
            used_output_count=0,
            discarded_output_count=len(leads),
        )
        return leads


async def _refine_local_leads_with_ollama(
    settings: Settings,
    query: str,
    candidate_pool: list[JobLead],
    *,
    cleanup_limit: int,
    refinement_mode: str | None = None,
    pre_refinement_average_confidence: float | None = None,
    pre_refinement_cleanup_signal_count: int | None = None,
    pre_refinement_trustworthy_direct_url_count: int | None = None,
    run_id: str | None = None,
) -> list[JobLead]:
    if not candidate_pool or not _ollama_inline_refinement_enabled(settings) or settings.ollama_degraded_for_run:
        return candidate_pool
    forced_refinement_mode = refinement_mode in {"forced_sample", "forced_seed_triage"}
    allow_single_seed_triage_without_cleanup = (
        refinement_mode == "forced_seed_triage"
        and len(candidate_pool) == 1
        and (pre_refinement_average_confidence or 0.0) >= 0.9
        and (pre_refinement_trustworthy_direct_url_count or 0) >= 1
    )
    allow_single_replay_source_seed_triage_without_cleanup = (
        refinement_mode == "forced_seed_triage"
        and len(candidate_pool) == 1
        and (pre_refinement_average_confidence or 0.0) >= 0.9
        and _lead_is_replay_source_trustworthy(candidate_pool[0])
    )
    allow_clean_seed_bundle_triage_without_cleanup = (
        refinement_mode == "forced_seed_triage"
        and len(candidate_pool) >= 5
        and (pre_refinement_average_confidence or 0.0) >= 0.9
        and (pre_refinement_trustworthy_direct_url_count or 0) >= len(candidate_pool)
    )
    if (
        forced_refinement_mode
        and (pre_refinement_cleanup_signal_count or 0) <= 0
        and not allow_single_seed_triage_without_cleanup
        and not allow_single_replay_source_seed_triage_without_cleanup
        and not allow_clean_seed_bundle_triage_without_cleanup
    ):
        record_ollama_event(
            settings,
            "lead_refinement_skip",
            run_id=run_id,
            caller="lead_refinement",
            prompt_category="lead_cleanup",
            query=query,
            refinement_mode=refinement_mode,
            candidate_pool_count=len(candidate_pool),
            normalized_lead_count=len(candidate_pool),
            average_confidence=round(pre_refinement_average_confidence or 0.0, 3),
            cleanup_signal_count=pre_refinement_cleanup_signal_count or 0,
            trustworthy_direct_url_count=pre_refinement_trustworthy_direct_url_count or 0,
            skip_reason="forced_refinement_without_cleanup_signals",
        )
        return candidate_pool

    cleanup_candidates = _deterministic_trim_local_leads(
        settings,
        query,
        candidate_pool,
        limit=max(1, min(cleanup_limit, 5)),
    )
    if run_id is None:
        cleaned_leads = await _cleanup_local_leads_with_ollama(settings, query, cleanup_candidates)
    else:
        cleaned_leads = await _cleanup_local_leads_with_ollama(settings, query, cleanup_candidates, run_id=run_id)
    normalized_cleaned = _normalize_and_filter_discovery_leads(cleaned_leads, query)
    if not normalized_cleaned:
        record_ollama_event(
            settings,
            "lead_refinement_outcome",
            run_id=run_id,
            caller="lead_refinement",
            prompt_category="lead_merge",
            query=query,
            refinement_mode=refinement_mode or "standard",
            forced_borderline=bool(refinement_mode == "borderline_bundle"),
            proposed_lead_count=len(candidate_pool),
            returned_lead_count=0,
            merged_lead_count=len(candidate_pool),
            used_output_count=0,
            discarded_output_count=len(cleaned_leads),
            cleanup_limit=cleanup_limit,
            pre_average_confidence=round(pre_refinement_average_confidence or 0.0, 3),
            post_average_confidence=round(pre_refinement_average_confidence or 0.0, 3),
            pre_cleanup_signal_count=pre_refinement_cleanup_signal_count or 0,
            post_cleanup_signal_count=pre_refinement_cleanup_signal_count or 0,
            pre_trustworthy_direct_url_count=pre_refinement_trustworthy_direct_url_count or 0,
            post_trustworthy_direct_url_count=pre_refinement_trustworthy_direct_url_count or 0,
        )
        return candidate_pool
    merged = _merge_and_dedupe_leads(normalized_cleaned, candidate_pool)
    post_refinement_confidences = [_lead_confidence(lead) for lead in merged]
    post_refinement_average_confidence = (
        sum(post_refinement_confidences) / len(post_refinement_confidences) if post_refinement_confidences else 0.0
    )
    post_refinement_cleanup_signal_count = sum(1 for lead in merged[:5] if _lead_needs_local_cleanup(lead))
    post_refinement_trustworthy_direct_url_count = sum(
        1
        for lead in merged[:5]
        if lead.direct_job_url and _candidate_direct_job_url_is_trustworthy(lead.direct_job_url, lead)
    )
    record_ollama_event(
        settings,
        "lead_refinement_outcome",
        run_id=run_id,
        caller="lead_refinement",
        prompt_category="lead_merge",
        query=query,
        refinement_mode=refinement_mode or "standard",
        forced_borderline=bool(refinement_mode == "borderline_bundle"),
        proposed_lead_count=len(candidate_pool),
        returned_lead_count=len(normalized_cleaned),
        merged_lead_count=len(merged),
        used_output_count=len(normalized_cleaned),
        discarded_output_count=max(0, len(candidate_pool) - len(merged)),
        cleanup_limit=cleanup_limit,
        pre_average_confidence=round(pre_refinement_average_confidence or 0.0, 3),
        post_average_confidence=round(post_refinement_average_confidence, 3),
        pre_cleanup_signal_count=pre_refinement_cleanup_signal_count or 0,
        post_cleanup_signal_count=post_refinement_cleanup_signal_count,
        pre_trustworthy_direct_url_count=pre_refinement_trustworthy_direct_url_count or 0,
        post_trustworthy_direct_url_count=post_refinement_trustworthy_direct_url_count,
    )
    return merged


async def _maybe_force_round_lead_refinement_with_ollama(
    settings: Settings,
    round_leads: list[JobLead],
    *,
    attempt_number: int,
    round_number: int,
    run_id: str | None = None,
) -> list[JobLead]:
    if (
        not _ollama_inline_refinement_enabled(settings)
        or settings.ollama_degraded_for_run
        or run_id is None
        or len(round_leads) < 2
    ):
        return round_leads
    forced_key = (run_id, attempt_number)
    if forced_key in FORCED_OLLAMA_ROUND_REFINEMENT_ATTEMPTS:
        return round_leads
    FORCED_OLLAMA_ROUND_REFINEMENT_ATTEMPTS.add(forced_key)
    confidences = [_lead_confidence(lead) for lead in round_leads]
    average_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    cleanup_window = round_leads[:5]
    cleanup_signal_count = sum(1 for lead in cleanup_window if _lead_needs_local_cleanup(lead))
    low_trust_source_count = sum(1 for lead in cleanup_window if lead.source_type in {"linkedin", "builtin", "other"})
    trustworthy_direct_url_count = sum(
        1
        for lead in cleanup_window
        if lead.direct_job_url and _candidate_direct_job_url_is_trustworthy(lead.direct_job_url, lead)
    )
    if (
        len(cleanup_window) >= 5
        and cleanup_signal_count == 0
        and average_confidence >= 0.9
        and trustworthy_direct_url_count == len(cleanup_window)
    ):
        return await _refine_local_leads_with_ollama(
            settings,
            f"attempt {attempt_number} round {round_number} aggregate cleanup",
            round_leads,
            cleanup_limit=2,
            refinement_mode="forced_round_sample",
            pre_refinement_average_confidence=average_confidence,
            pre_refinement_cleanup_signal_count=cleanup_signal_count,
            pre_refinement_trustworthy_direct_url_count=trustworthy_direct_url_count,
            run_id=run_id,
        )
    # Skip forced round cleanup for already-clean bundles with only a single cleanup signal.
    # This keeps the aggregate sample path focused on noisier rounds.
    if (
        cleanup_signal_count <= 1
        and average_confidence >= 0.9
        and trustworthy_direct_url_count >= max(1, len(cleanup_window) - 1)
    ):
        return round_leads
    if not _should_force_ollama_refinement_sample(
        settings,
        sample_size=len(cleanup_window),
        average_confidence=average_confidence,
        cleanup_signal_count=cleanup_signal_count,
        low_trust_source_count=low_trust_source_count,
        trustworthy_direct_url_count=trustworthy_direct_url_count,
    ):
        return round_leads
    return await _refine_local_leads_with_ollama(
        settings,
        f"attempt {attempt_number} round {round_number} aggregate cleanup",
        round_leads,
        cleanup_limit=2,
        refinement_mode="forced_round_sample",
        pre_refinement_average_confidence=average_confidence,
        pre_refinement_cleanup_signal_count=cleanup_signal_count,
        pre_refinement_trustworthy_direct_url_count=trustworthy_direct_url_count,
        run_id=run_id,
    )


async def _maybe_force_seed_lead_refinement_with_ollama(
    settings: Settings,
    seed_leads: list[JobLead],
    *,
    run_id: str | None = None,
) -> list[JobLead]:
    if (
        not _ollama_inline_refinement_enabled(settings)
        or settings.ollama_degraded_for_run
        or run_id is None
        or not seed_leads
        or run_id in FORCED_OLLAMA_SEED_REFINEMENT_RUNS
    ):
        return seed_leads
    seed_window = seed_leads[:5]
    confidences = [_lead_confidence(lead) for lead in seed_window]
    average_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
    cleanup_signal_count = sum(1 for lead in seed_window if _lead_needs_local_cleanup(lead))
    low_trust_source_count = sum(1 for lead in seed_window if lead.source_type in {"linkedin", "builtin", "other"})
    trustworthy_direct_url_count = sum(
        1
        for lead in seed_window
        if lead.direct_job_url and _candidate_direct_job_url_is_trustworthy(lead.direct_job_url, lead)
    )
    if (
        len(seed_window) == 1
        and average_confidence >= 0.9
        and low_trust_source_count == 0
        and (
            trustworthy_direct_url_count >= 1
            or _lead_is_replay_source_trustworthy(seed_window[0])
        )
    ):
        FORCED_OLLAMA_SEED_REFINEMENT_RUNS.add(run_id)
        return await _refine_local_leads_with_ollama(
            settings,
            "seed replay triage",
            seed_leads,
            cleanup_limit=1,
            refinement_mode="forced_seed_triage",
            pre_refinement_average_confidence=average_confidence,
            pre_refinement_cleanup_signal_count=cleanup_signal_count,
            pre_refinement_trustworthy_direct_url_count=trustworthy_direct_url_count,
            run_id=run_id,
        )
    if (
        len(seed_window) >= 5
        and average_confidence >= 0.9
        and cleanup_signal_count <= 0
        and low_trust_source_count == 0
        and trustworthy_direct_url_count >= len(seed_window)
    ):
        FORCED_OLLAMA_SEED_REFINEMENT_RUNS.add(run_id)
        return await _refine_local_leads_with_ollama(
            settings,
            "seed replay triage",
            seed_leads,
            cleanup_limit=2,
            refinement_mode="forced_seed_triage",
            pre_refinement_average_confidence=average_confidence,
            pre_refinement_cleanup_signal_count=cleanup_signal_count,
            pre_refinement_trustworthy_direct_url_count=trustworthy_direct_url_count,
            run_id=run_id,
        )
    # Skip forced seed cleanup for already-clean replay bundles with only a single cleanup signal.
    # This keeps seed triage focused on noisier replay windows.
    if (
        cleanup_signal_count <= 1
        and average_confidence >= 0.9
        and trustworthy_direct_url_count >= max(1, len(seed_window) - 1)
    ):
        return seed_leads
    if cleanup_signal_count <= 0:
        return seed_leads
    if not _should_force_ollama_refinement_sample(
        settings,
        sample_size=len(seed_window),
        average_confidence=average_confidence,
        cleanup_signal_count=cleanup_signal_count,
        low_trust_source_count=low_trust_source_count,
        trustworthy_direct_url_count=trustworthy_direct_url_count,
    ):
        return seed_leads
    FORCED_OLLAMA_SEED_REFINEMENT_RUNS.add(run_id)
    return await _refine_local_leads_with_ollama(
        settings,
        "seed replay triage",
        seed_leads,
        cleanup_limit=2,
        refinement_mode="forced_seed_triage",
        pre_refinement_average_confidence=average_confidence,
        pre_refinement_cleanup_signal_count=cleanup_signal_count,
        pre_refinement_trustworthy_direct_url_count=trustworthy_direct_url_count,
        run_id=run_id,
    )


def _merge_and_dedupe_leads(*groups: list[JobLead]) -> list[JobLead]:
    merged: list[JobLead] = []
    seen: set[str] = set()
    for group in groups:
        for lead in group:
            key = _lead_dedupe_key(lead)
            if key in seen:
                continue
            seen.add(key)
            merged.append(lead)
    return merged


async def _search_single_query_openai(agent: Agent, query: str) -> list[JobLead]:
    prompt = f"""
Search seed: {query}
Find high-confidence AI-related Product Manager leads and return concise structured results only.
""".strip()
    result = await Runner.run(agent, prompt)
    return _normalize_and_filter_discovery_leads(result.final_output.leads, query)


async def _search_single_query_local(
    settings: Settings,
    query: str,
    *,
    attempt_number: int | None = None,
    run_id: str | None = None,
) -> tuple[list[JobLead], float]:
    async def _run_named(name: str, coro, *, timeout_seconds: float | None = None):
        try:
            if timeout_seconds is not None:
                return name, await asyncio.wait_for(coro, timeout=timeout_seconds), None
            return name, await coro, None
        except Exception as exc:
            return name, None, exc

    builtin_leads: list[JobLead] = []
    linkedin_leads: list[JobLead] = []
    search_results: list[tuple[str, str, str]] = []
    builtin_error: Exception | None = None
    linkedin_error: Exception | None = None
    search_error: Exception | None = None

    builtin_enabled = not _query_is_company_named_open_web_query(query)
    tasks: list[asyncio.Task[tuple[str, object | None, Exception | None]]] = []
    if builtin_enabled:
        tasks.append(
            asyncio.create_task(
                _run_named(
                    "builtin",
                    _builtin_search(query, settings),
                    timeout_seconds=min(max(settings.per_query_timeout_seconds / 2, 12), 20),
                )
            )
        )
    tasks.extend(
        [
            asyncio.create_task(
                _run_named(
                    "linkedin",
                    _linkedin_guest_search(query, settings),
                    timeout_seconds=min(max(settings.per_query_timeout_seconds / 3, 8), 15),
                )
            ),
            asyncio.create_task(
                _run_named(
                    "search",
                    _run_local_search_engine_queries(
                        query,
                        max_results_per_query=max(settings.max_leads_per_query * 2, 12),
                    ),
                    timeout_seconds=min(max(settings.per_query_timeout_seconds, 20), 35),
                )
            ),
        ]
    )

    for task in asyncio.as_completed(tasks):
        name, result, error = await task
        if name == "builtin":
            builtin_error = error
            builtin_leads = result or []
        elif name == "linkedin":
            linkedin_error = error
            linkedin_leads = result or []
        else:
            search_error = error
            search_results = result or []

    if (
        not search_results
        and linkedin_error is not None
        and search_error is not None
        and (not builtin_enabled or builtin_error is not None)
    ):
        failure_parts: list[str] = []
        if builtin_enabled and builtin_error is not None:
            failure_parts.append(f"Built In search failed ({_describe_exception(builtin_error)})")
        failure_parts.append(f"LinkedIn guest search failed ({_describe_exception(linkedin_error)})")
        failure_parts.append(f"search engine discovery failed ({_describe_exception(search_error)})")
        raise LocalSearchBackendBlockedError("; ".join(failure_parts)) from search_error

    leads: list[JobLead] = []
    for url, title, snippet in search_results:
        lead = _build_lead_from_search_result(url, title, snippet, query)
        if lead is None:
            continue
        leads.append(lead)

    builtin_leads = _normalize_and_filter_discovery_leads(builtin_leads, query)
    linkedin_leads = _normalize_and_filter_discovery_leads(linkedin_leads, query)
    leads = _normalize_and_filter_discovery_leads(leads, query)

    merged_leads = _merge_and_dedupe_leads(builtin_leads, linkedin_leads, leads)
    candidate_pool = _deterministic_trim_local_leads(
        settings,
        query,
        merged_leads,
        limit=max(settings.max_leads_per_query * 3, 18),
    )
    normalized = _deterministic_trim_local_leads(settings, query, candidate_pool)
    normalized_confidences = [_lead_confidence(lead) for lead in normalized]
    average_confidence = (sum(normalized_confidences) / len(normalized_confidences)) if normalized_confidences else 0.0
    cleanup_window = normalized[:5]
    cleanup_signal_count = sum(1 for lead in cleanup_window if _lead_needs_local_cleanup(lead))
    low_trust_source_count = sum(1 for lead in cleanup_window if lead.source_type in {"linkedin", "builtin", "other"})
    trustworthy_direct_url_count = sum(
        1
        for lead in cleanup_window
        if lead.direct_job_url and _candidate_direct_job_url_is_trustworthy(lead.direct_job_url, lead)
    )
    if (
        _ollama_inline_refinement_enabled(settings)
        and _is_clean_high_confidence_direct_bundle(
            candidate_pool_count=len(candidate_pool),
            average_confidence=average_confidence,
            cleanup_signal_count=cleanup_signal_count,
            low_trust_source_count=low_trust_source_count,
            trustworthy_direct_url_count=trustworthy_direct_url_count,
        )
    ):
        refined_pool = await _refine_local_leads_with_ollama(
            settings,
            query,
            candidate_pool,
            cleanup_limit=3,
            refinement_mode="trusted_direct_bundle",
            pre_refinement_average_confidence=average_confidence,
            pre_refinement_cleanup_signal_count=cleanup_signal_count,
            pre_refinement_trustworthy_direct_url_count=trustworthy_direct_url_count,
            run_id=run_id,
        )
        normalized = _deterministic_trim_local_leads(settings, query, refined_pool)
        normalized_confidences = [_lead_confidence(lead) for lead in normalized]
        average_confidence = (sum(normalized_confidences) / len(normalized_confidences)) if normalized_confidences else 0.0
        await asyncio.sleep(0.5)
        return normalized, average_confidence
    should_force_sample = _should_force_ollama_refinement_sample(
        settings,
        sample_size=len(cleanup_window),
        average_confidence=average_confidence,
        cleanup_signal_count=cleanup_signal_count,
        low_trust_source_count=low_trust_source_count,
        trustworthy_direct_url_count=trustworthy_direct_url_count,
        query=query,
    )

    refinement_mode = _ollama_refinement_mode_for_local_leads(
        settings,
        query=query,
        candidate_pool_count=len(candidate_pool),
        average_confidence=average_confidence,
        cleanup_signal_count=cleanup_signal_count,
        low_trust_source_count=low_trust_source_count,
        trustworthy_direct_url_count=trustworthy_direct_url_count,
    )
    if (
        refinement_mode is None
        and _ollama_inline_refinement_enabled(settings)
        and _is_trusted_company_careers_bundle(
            query=query,
            candidate_pool_count=len(candidate_pool),
            average_confidence=average_confidence,
            cleanup_signal_count=cleanup_signal_count,
            low_trust_source_count=low_trust_source_count,
            trustworthy_direct_url_count=trustworthy_direct_url_count,
        )
    ):
        refinement_mode = "trusted_direct_bundle"
    if (
        refinement_mode is None
        and _ollama_inline_refinement_enabled(settings)
        and run_id is not None
        and attempt_number is not None
        and len(candidate_pool) >= 2
        and should_force_sample
    ):
        forced_key = (run_id, attempt_number)
        if forced_key not in FORCED_OLLAMA_REFINEMENT_ATTEMPTS:
            FORCED_OLLAMA_REFINEMENT_ATTEMPTS.add(forced_key)
            refinement_mode = "forced_sample"
    if refinement_mode is not None:
        cleanup_limit = (
            2
            if refinement_mode == "forced_sample"
            else
            3
            if refinement_mode == "trusted_direct_bundle"
            else
            4
            if refinement_mode == "startup_board_bundle"
            else 3
            if refinement_mode == "borderline_bundle"
            else max(settings.max_leads_per_query, 8)
        )
        refined_pool = await _refine_local_leads_with_ollama(
            settings,
            query,
            candidate_pool,
            cleanup_limit=cleanup_limit,
            refinement_mode=refinement_mode,
            pre_refinement_average_confidence=average_confidence,
            pre_refinement_cleanup_signal_count=cleanup_signal_count,
            pre_refinement_trustworthy_direct_url_count=trustworthy_direct_url_count,
            run_id=run_id,
        )
        normalized = _deterministic_trim_local_leads(settings, query, refined_pool)
        normalized_confidences = [_lead_confidence(lead) for lead in normalized]
        average_confidence = (sum(normalized_confidences) / len(normalized_confidences)) if normalized_confidences else 0.0
    elif _ollama_inline_refinement_enabled(settings) and run_id is not None:
        if _is_trusted_company_careers_bundle(
            query=query,
            candidate_pool_count=len(candidate_pool),
            average_confidence=average_confidence,
            cleanup_signal_count=cleanup_signal_count,
            low_trust_source_count=low_trust_source_count,
            trustworthy_direct_url_count=trustworthy_direct_url_count,
        ):
            refined_pool = await _refine_local_leads_with_ollama(
                settings,
                query,
                candidate_pool,
                cleanup_limit=3,
                refinement_mode="trusted_direct_bundle",
                pre_refinement_average_confidence=average_confidence,
                pre_refinement_cleanup_signal_count=cleanup_signal_count,
                pre_refinement_trustworthy_direct_url_count=trustworthy_direct_url_count,
                run_id=run_id,
            )
            normalized = _deterministic_trim_local_leads(settings, query, refined_pool)
            normalized_confidences = [_lead_confidence(lead) for lead in normalized]
            average_confidence = (
                (sum(normalized_confidences) / len(normalized_confidences)) if normalized_confidences else 0.0
            )
        else:
            record_ollama_event(
                settings,
                "lead_refinement_skip",
                run_id=run_id,
                caller="lead_refinement",
                prompt_category="lead_cleanup",
                query=query,
                refinement_mode="skipped",
                candidate_pool_count=len(candidate_pool),
                normalized_lead_count=len(normalized),
                average_confidence=round(average_confidence, 3),
                cleanup_signal_count=cleanup_signal_count,
                low_trust_source_count=low_trust_source_count,
                trustworthy_direct_url_count=trustworthy_direct_url_count,
            )
    await asyncio.sleep(0.5)
    return normalized, average_confidence


async def _search_single_query(
    agent: Agent | None,
    settings: Settings,
    query: str,
    *,
    attempt_number: int | None = None,
    run_id: str | None = None,
) -> list[JobLead]:
    if settings.llm_provider == "ollama":
        local_leads, confidence = await _search_single_query_local(
            settings,
            query,
            attempt_number=attempt_number,
            run_id=run_id,
        )
        if local_leads and not any(lead.refined_by_ollama for lead in local_leads):
            local_confidences = [_lead_confidence(lead) for lead in local_leads]
            local_average_confidence = (
                (sum(local_confidences) / len(local_confidences)) if local_confidences else 0.0
            )
            cleanup_window = local_leads[:5]
            local_cleanup_signal_count = sum(1 for lead in cleanup_window if _lead_needs_local_cleanup(lead))
            local_low_trust_source_count = sum(
                1 for lead in cleanup_window if lead.source_type in {"linkedin", "builtin", "other"}
            )
            local_trustworthy_direct_url_count = sum(
                1
                for lead in cleanup_window
                if lead.direct_job_url and _candidate_direct_job_url_is_trustworthy(lead.direct_job_url, lead)
            )
            if _is_clean_high_confidence_direct_bundle(
                candidate_pool_count=len(local_leads),
                average_confidence=local_average_confidence,
                cleanup_signal_count=local_cleanup_signal_count,
                low_trust_source_count=local_low_trust_source_count,
                trustworthy_direct_url_count=local_trustworthy_direct_url_count,
            ) or (
                len(local_leads) == 1
                and local_cleanup_signal_count == 0
                and local_low_trust_source_count == 0
                and local_trustworthy_direct_url_count >= 1
                and local_average_confidence >= 0.9
            ):
                local_leads = await _refine_local_leads_with_ollama(
                    settings,
                    query,
                    local_leads,
                    cleanup_limit=3,
                    refinement_mode="trusted_direct_bundle",
                    pre_refinement_average_confidence=local_average_confidence,
                    pre_refinement_cleanup_signal_count=local_cleanup_signal_count,
                    pre_refinement_trustworthy_direct_url_count=local_trustworthy_direct_url_count,
                    run_id=run_id,
                )
                local_confidences = [_lead_confidence(lead) for lead in local_leads]
                confidence = (sum(local_confidences) / len(local_confidences)) if local_confidences else 0.0
        should_fallback = (
            settings.use_openai_fallback
            and agent is not None
            and (not local_leads or confidence < settings.local_confidence_threshold)
        )
        if should_fallback:
            fallback_leads = await _search_single_query_openai(agent, query)
            return _merge_and_dedupe_leads(local_leads, fallback_leads)
        return local_leads

    if agent is None:
        return []
    return await _search_single_query_openai(agent, query)


async def _search_query_with_context(
    agent: Agent | None,
    settings: Settings,
    query: str,
    timeout_seconds: int,
    *,
    attempt_number: int | None = None,
    run_id: str | None = None,
) -> tuple[str, list[JobLead]]:
    try:
        return query, await asyncio.wait_for(
            _search_single_query(agent, settings, query, attempt_number=attempt_number, run_id=run_id),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise SearchQueryTimeoutError(query) from exc
    except Exception as exc:
        raise SearchQueryExecutionError(query, exc) from exc


def _build_resolution_prompt(lead: JobLead) -> str:
    return f"""
Find the exact direct ATS/company-careers posting URL for this role.

Company: {lead.company_name}
Role: {lead.role_title}
Source URL: {lead.source_url}
Direct URL hint: {lead.direct_job_url or "none"}
Location hint: {lead.location_hint or "unknown"}
Posted hint: {lead.posted_date_hint or "unknown"}
Salary hint: {lead.salary_text_hint or "unknown"}
Notes: {lead.evidence_notes}
""".strip()


def _build_resolution_retry_prompt(lead: JobLead, bad_direct_url: str, failure_reason: str) -> str:
    return f"""
Previous candidate URL was invalid/generic.

Invalid URL: {bad_direct_url}
Failure reason: {failure_reason}

Find a different exact direct ATS/company-careers posting URL for:
- Company: {lead.company_name}
- Role: {lead.role_title}
- Source URL: {lead.source_url}
""".strip()


def _build_candidate_job(lead: JobLead, resolution: DirectJobResolution) -> JobPosting:
    direct_job_url = _normalize_direct_job_url(resolution.direct_job_url or lead.direct_job_url or lead.source_url)
    salary_min_hint, salary_max_hint, salary_text_hint = _hydrate_salary_hint_values(
        lead.base_salary_min_usd_hint,
        lead.base_salary_max_usd_hint,
        lead.salary_text_hint,
        lead.evidence_notes,
    )
    evidence_lines = [lead.evidence_notes, resolution.evidence_notes]
    if lead.posted_date_hint:
        evidence_lines.append(f"Posted hint: {lead.posted_date_hint}")
    if salary_text_hint:
        evidence_lines.append(f"Salary hint: {salary_text_hint}")
    if lead.is_remote_hint:
        evidence_lines.append("Remote hint from discovery source.")
    return JobPosting(
        company_name=lead.company_name,
        role_title=lead.role_title,
        direct_job_url=direct_job_url,
        resolved_job_url=direct_job_url,
        ats_platform=resolution.ats_platform or urlparse(direct_job_url).netloc or "Unknown",
        location_text=lead.location_hint or ("Remote" if lead.is_remote_hint else ""),
        is_fully_remote=lead.is_remote_hint,
        posted_date_text=lead.posted_date_hint or "",
        posted_date_iso=None,
        base_salary_min_usd=salary_min_hint,
        base_salary_max_usd=salary_max_hint,
        salary_text=salary_text_hint,
        source_query=lead.source_query,
        evidence_notes=" ".join(part for part in evidence_lines if part).strip(),
        validation_evidence=[],
        lead_refined_by_ollama=lead.refined_by_ollama,
        source_quality_score=lead.source_quality_score_hint,
    )


def _lead_dedupe_key(lead: JobLead) -> str:
    direct_or_source = _normalize_direct_job_url(lead.direct_job_url or lead.source_url)
    return f"{_normalize_job_key(lead.company_name, lead.role_title)}:{direct_or_source}"


def _job_posting_dedupe_key(job: JobPosting) -> str:
    return _job_history_primary_key(str(job.resolved_job_url or job.direct_job_url))


def _url_candidate_score(url: str, anchor_text: str, lead: JobLead) -> tuple[int, int]:
    haystack = f"{url} {anchor_text}".lower()
    parsed = urlparse(url)
    query = (parsed.query or "").lower()
    has_job_query_hint = any(token in query for token in COMPANY_JOB_QUERY_HINTS)
    company_matches = _direct_job_url_matches_expected_company(url, lead.company_name)
    score = 0
    if lead.source_type == "linkedin" and _is_linkedin_resolution_blocked_host(url):
        score -= 30
    if not company_matches:
        score -= 24
    else:
        score += 4
    for token in lead.company_name.lower().split():
        if token and token in haystack:
            score += 3
    role_score = _role_match_score(lead.role_title, haystack)
    if (has_job_query_hint or (_is_allowed_direct_job_url(url) and company_matches)) and role_score < 0:
        role_score = 0
    score += role_score
    if has_job_query_hint:
        score += 7
    elif _looks_like_careers_hub_url(url):
        score += 2
    if _is_allowed_direct_job_url(url):
        score += 5
        if role_score <= 0 and not has_job_query_hint:
            score -= 10
    path_bonus = 1 if any(hint in url.lower() for hint in JOB_PATH_HINTS) else 0
    return score, path_bonus


def _link_context_text(link) -> str:
    parts: list[str] = []
    link_text = link.get_text(" ", strip=True)
    if link_text:
        parts.append(link_text)
    for attr in ("aria-label", "title"):
        value = link.get(attr)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    current = link
    for _ in range(3):
        current = current.parent
        if current is None:
            break
        context_text = current.get_text(" ", strip=True)
        if context_text:
            parts.append(context_text[:400])

    return " ".join(part for part in parts if part)


async def _extract_direct_job_url_from_source(lead: JobLead) -> str | None:
    candidate_url = _normalize_direct_job_url(lead.direct_job_url or lead.source_url)
    if candidate_url and _candidate_direct_job_url_is_trustworthy(candidate_url, lead):
        return candidate_url

    board_resolved_url = await _resolve_supported_board_job_url_from_lead(lead)
    if board_resolved_url and _candidate_direct_job_url_is_trustworthy(board_resolved_url, lead):
        return board_resolved_url

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=15.0,
        ) as client:
            response = await client.get(lead.source_url)
    except Exception:
        return None

    resolved_source = _normalize_direct_job_url(str(response.url))
    if _is_allowed_direct_job_url(resolved_source) and not _looks_like_generic_job_url(resolved_source):
        return resolved_source

    embedded_board_urls = _dedupe_queries(
        [
            *(extract_embedded_board_urls(resolved_source, response.text) or []),
            *(url for url in (infer_careers_root(resolved_source),) if url),
        ]
    )
    for board_url in embedded_board_urls:
        board_resolution = await _resolve_supported_board_job_url_from_lead(
            lead.model_copy(
                update={
                    "source_url": board_url,
                    "source_type": "direct_ats",
                    "direct_job_url": None,
                }
            )
        )
        if board_resolution and _candidate_direct_job_url_is_trustworthy(board_resolution, lead):
            return board_resolution

    builtin_apply_url = None
    if _normalize_source_type(lead.source_url) == "builtin":
        builtin_apply_url = _extract_builtin_apply_url(response.text)
        if builtin_apply_url:
            normalized_builtin_apply_url = _normalize_direct_job_url(builtin_apply_url)
            parsed_builtin_apply = urlparse(normalized_builtin_apply_url)
            builtin_apply_query = (parsed_builtin_apply.query or "").lower()
            has_job_query_hint = any(token in builtin_apply_query for token in COMPANY_JOB_QUERY_HINTS)
            if _candidate_direct_job_url_is_trustworthy(normalized_builtin_apply_url, lead) and (
                not _looks_like_generic_job_url(normalized_builtin_apply_url) or has_job_query_hint
            ):
                return normalized_builtin_apply_url

    soup = BeautifulSoup(response.text, "html.parser")
    candidates: list[tuple[tuple[int, int], str]] = []
    if builtin_apply_url:
        normalized_builtin_apply_url = _normalize_direct_job_url(builtin_apply_url)
        if _is_allowed_direct_job_url(normalized_builtin_apply_url):
            candidates.append((_url_candidate_score(normalized_builtin_apply_url, "Apply", lead), normalized_builtin_apply_url))
    for link in soup.select("a[href]"):
        href = link.get("href")
        if not href:
            continue
        absolute_url = _normalize_direct_job_url(urljoin(resolved_source, href))
        if not _candidate_direct_job_url_is_trustworthy(absolute_url, lead):
            continue
        score = _url_candidate_score(absolute_url, _link_context_text(link), lead)
        candidates.append((score, absolute_url))
    for raw_url in re.findall(r"https?://[^\s\"'<>]+", response.text):
        normalized_raw_url = _normalize_direct_job_url(raw_url)
        if not _candidate_direct_job_url_is_trustworthy(normalized_raw_url, lead):
            continue
        candidates.append((_url_candidate_score(normalized_raw_url, "", lead), normalized_raw_url))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_url = candidates[0]
    if best_score[0] <= 0:
        return None
    return best_url


async def _extract_source_followup_resolution_urls(lead: JobLead) -> list[str]:
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=15.0,
        ) as client:
            response = await client.get(lead.source_url)
    except Exception:
        return []

    resolved_source = _normalize_direct_job_url(str(response.url))
    candidates: dict[str, tuple[int, int]] = {}
    directory_candidates: list[str] = []

    def maybe_add(url: str | None, anchor_text: str = "") -> None:
        if not url:
            return
        normalized_url = _normalize_direct_job_url(urljoin(resolved_source, url))
        if not normalized_url or normalized_url == resolved_source:
            return
        if any(fragment in (urlparse(normalized_url).netloc or "").lower() for fragment in BLOCKED_JOB_HOST_FRAGMENTS):
            return
        if lead.source_type == "linkedin" and _is_linkedin_resolution_blocked_host(normalized_url):
            return
        if not (
            _is_allowed_direct_job_url(normalized_url)
            or _looks_like_careers_hub_url(normalized_url)
            or _looks_like_company_homepage_url(normalized_url)
        ):
            return
        if _is_allowed_direct_job_url(normalized_url) and not _direct_job_url_matches_expected_company(
            normalized_url,
            lead.company_name,
        ):
            return
        score = _url_candidate_score(normalized_url, anchor_text, lead)
        previous = candidates.get(normalized_url)
        if previous is None or score > previous:
            candidates[normalized_url] = score

    def maybe_add_directory_seed(url: str | None) -> None:
        normalized_url = _normalize_direct_job_url(str(url or "").strip())
        if not normalized_url or normalized_url == resolved_source:
            return
        if normalized_url in directory_candidates:
            return
        if not is_company_discovery_seed_url(normalized_url):
            return
        directory_candidates.append(normalized_url)

    if _normalize_source_type(lead.source_url) == "builtin":
        maybe_add(_extract_builtin_apply_url(response.text), "Apply")
        for company_url in _extract_builtin_company_followup_urls(response.text):
            maybe_add(company_url, "Company website")
        for task in extract_directory_company_tasks(lead.source_url, response.text):
            task_url = str(task.get("url") or "").strip()
            task_company_name = str(task.get("company_name") or "").strip()
            if task_company_name and not _company_names_match(lead.company_name, task_company_name):
                continue
            maybe_add_directory_seed(task_url)
            for followup_url in default_careers_candidate_urls(task_url):
                maybe_add_directory_seed(followup_url)

    for company_url in _extract_structured_company_followup_urls(response.text):
        maybe_add(company_url, "Structured company URL")
        for followup_url in default_careers_candidate_urls(company_url):
            maybe_add(followup_url, "Careers")

    for board_url in extract_embedded_board_urls(resolved_source, response.text):
        maybe_add(board_url, "Embedded ATS board")

    for careers_url in extract_careers_page_urls(resolved_source, response.text):
        maybe_add(careers_url, "Careers")

    for homepage_url in extract_company_homepage_urls(resolved_source, response.text):
        maybe_add(homepage_url, "Company website")
        for followup_url in default_careers_candidate_urls(homepage_url):
            maybe_add(followup_url, "Careers")

    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.select("a[href]"):
        maybe_add(link.get("href"), _link_context_text(link))

    for raw_url in re.findall(r"https?://[^\s\"'<>]+", response.text):
        maybe_add(raw_url, "")

    ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
    ordered = [*directory_candidates, *[url for url, score in ranked if score[0] > 0]]
    return list(dict.fromkeys(ordered))[:8]


async def _resolve_lead_via_company_careers_pages(lead: JobLead) -> DirectJobResolution | None:
    initial_urls = await _extract_source_followup_resolution_urls(lead)
    if len(initial_urls) < 4:
        searched_urls = await _search_company_resolution_candidates(lead)
        for url in searched_urls:
            if url not in initial_urls:
                initial_urls.append(url)
    if not initial_urls:
        return None

    queue: deque[tuple[str, int]] = deque((url, 0) for url in initial_urls)
    visited: set[str] = set()
    best_direct_url: str | None = None
    best_direct_score: tuple[int, int] | None = None

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
        timeout=15.0,
    ) as client:
        while queue and len(visited) < 10:
            page_url, depth = queue.popleft()
            normalized_page_url = _normalize_direct_job_url(page_url)
            if normalized_page_url in visited:
                continue
            visited.add(normalized_page_url)

            page_lead = lead.model_copy(
                update={
                    "source_url": normalized_page_url,
                    "source_type": "company_site",
                    "direct_job_url": None,
                }
            )
            direct_url = await _extract_direct_job_url_from_source(page_lead)
            if direct_url:
                score = _url_candidate_score(direct_url, "", lead)
                if best_direct_score is None or score > best_direct_score:
                    best_direct_url = direct_url
                    best_direct_score = score
                    page_host = (urlparse(normalized_page_url).netloc or "").lower()
                    if depth >= 2 or any(fragment in page_host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
                        break

            if depth >= 2:
                continue

            try:
                response = await client.get(normalized_page_url)
            except Exception:
                continue
            if response.status_code != 200:
                continue

            followup_urls = _extract_followup_resolution_urls(str(response.url), response.text)
            for followup_url in followup_urls:
                if followup_url not in visited:
                    queue.append((followup_url, depth + 1))

    if not best_direct_url:
        return None
    return DirectJobResolution(
        accepted=True,
        direct_job_url=best_direct_url,
        ats_platform=urlparse(best_direct_url).netloc,
        evidence_notes="Resolved via company homepage/careers discovery and ATS board traversal.",
    )


async def _resolve_lead_to_direct_job_url(agent: Agent | None, lead: JobLead) -> DirectJobResolution | None:
    locally_resolved_url = await _extract_direct_job_url_from_source(lead)
    if locally_resolved_url:
        return DirectJobResolution(
            accepted=True,
            direct_job_url=locally_resolved_url,
            ats_platform=urlparse(locally_resolved_url).netloc,
            evidence_notes="Resolved locally from the discovered source page.",
        )

    company_site_resolution = await _resolve_lead_via_company_careers_pages(lead)
    if company_site_resolution:
        return company_site_resolution

    if agent is None:
        return None

    result = await Runner.run(agent, _build_resolution_prompt(lead))
    resolution = result.final_output
    if not resolution.accepted or not resolution.direct_job_url:
        return None
    normalized_url = _normalize_direct_job_url(resolution.direct_job_url)
    if not _is_allowed_direct_job_url(normalized_url) or _looks_like_generic_job_url(normalized_url):
        return None
    return resolution.model_copy(update={"direct_job_url": normalized_url})


async def _repair_direct_job_url(
    agent: Agent | None,
    lead: JobLead,
    bad_direct_url: str,
    failure_reason: str,
) -> str | None:
    retry_lead = lead.model_copy(update={"direct_job_url": None})
    local_retry = await _extract_direct_job_url_from_source(retry_lead)
    if local_retry and local_retry != bad_direct_url and _is_allowed_direct_job_url(local_retry):
        return local_retry

    company_site_retry = await _resolve_lead_via_company_careers_pages(retry_lead)
    if company_site_retry and company_site_retry.direct_job_url:
        normalized_company_retry = _normalize_direct_job_url(company_site_retry.direct_job_url)
        if (
            normalized_company_retry != bad_direct_url
            and _is_allowed_direct_job_url(normalized_company_retry)
            and not _looks_like_generic_job_url(normalized_company_retry)
        ):
            return normalized_company_retry

    if agent is None:
        return None

    result = await Runner.run(agent, _build_resolution_retry_prompt(lead, bad_direct_url, failure_reason))
    resolution = result.final_output
    if not resolution.accepted or not resolution.direct_job_url:
        return None
    normalized_url = _normalize_direct_job_url(resolution.direct_job_url)
    if normalized_url == bad_direct_url:
        return None
    if not _is_allowed_direct_job_url(normalized_url) or _looks_like_generic_job_url(normalized_url):
        return None
    return normalized_url


def _make_failure(
    *,
    stage: str,
    reason_code: str,
    detail: str,
    lead: JobLead | None = None,
    source_query: str | None = None,
    direct_job_url: str | None = None,
    candidate: JobPosting | None = None,
    attempt_number: int,
    round_number: int,
) -> SearchFailure:
    company_name = lead.company_name if lead else (candidate.company_name if candidate else None)
    role_title = lead.role_title if lead else (candidate.role_title if candidate else None)
    failure_source_query = source_query if source_query is not None else (lead.source_query if lead else (candidate.source_query if candidate else None))
    source_url = lead.source_url if lead else None
    posted_date_text = lead.posted_date_hint if lead else (candidate.posted_date_text if candidate else None)
    salary_text = lead.salary_text_hint if lead else (candidate.salary_text if candidate else None)
    is_remote = lead.is_remote_hint if lead else (candidate.is_fully_remote if candidate else None)
    return SearchFailure(
        stage=stage,  # type: ignore[arg-type]
        reason_code=reason_code,
        detail=detail,
        company_name=company_name,
        role_title=role_title,
        source_query=failure_source_query,
        source_url=source_url,
        direct_job_url=direct_job_url or (candidate.direct_job_url if candidate else None),
        posted_date_text=posted_date_text,
        salary_text=salary_text,
        is_remote=is_remote,
        lead_refined_by_ollama=lead.refined_by_ollama if lead else (candidate.lead_refined_by_ollama if candidate else None),
        source_quality_score=(
            lead.source_quality_score_hint if lead and lead.source_quality_score_hint is not None else candidate.source_quality_score if candidate else None
        ),
        attempt_number=attempt_number,
        round_number=round_number,
    )


def _is_close_salary_miss(job: JobPosting, settings: Settings) -> bool:
    salary_values = _salary_values(job)
    return bool(salary_values and max(salary_values) >= settings.min_base_salary_usd - CLOSE_MISS_SALARY_BUFFER_USD)


def _is_close_stale_miss(job: JobPosting, settings: Settings) -> bool:
    posted_text = job.posted_date_text or ""
    return (
        not _is_recent_enough(job.posted_date_iso, posted_text, settings.posted_within_days, timezone_name=settings.timezone)
        and _is_recent_enough(
            job.posted_date_iso,
            posted_text,
            settings.posted_within_days + NEAR_MISS_STALE_BUFFER_DAYS,
            timezone_name=settings.timezone,
        )
    )


def _near_miss_why_close(reason_code: str, job: JobPosting) -> str:
    if reason_code == "remote_unclear":
        return "The role otherwise looked strong, but the page did not state fully remote status clearly enough."
    if reason_code == "missing_salary":
        return "The role looked like a strong AI PM fit on a real company page, but compensation was not disclosed."
    if reason_code == "salary_below_min":
        return "The role looked strong and came close to the salary threshold, but the listed base range landed slightly below target."
    if reason_code == "stale_posting":
        return "The role looked strong and direct, but the posting date landed just outside the freshness window."
    if reason_code == "fetch_non_200":
        return "Discovery found a strong direct role, but the final page could not be fetched successfully during validation."
    if reason_code == "resolution_missing":
        return "Discovery surfaced a strong role, but the direct ATS page could not be resolved deterministically."
    return f"The role was close, but failed validation with {reason_code}."


def _build_near_miss(
    lead: JobLead,
    job: JobPosting,
    failure: SearchFailure,
    settings: Settings,
) -> NearMissJob | None:
    if failure.reason_code not in NEAR_MISS_REASON_CODES:
        return None
    if not _lead_is_ai_related_product_manager(lead) and not _is_ai_related_product_manager(job):
        return None
    if _role_title_is_low_signal(job.role_title, job.evidence_notes):
        return None
    direct_url = str(job.resolved_job_url or job.direct_job_url or "")
    if direct_url and (not _is_allowed_direct_job_url(direct_url) or _looks_like_generic_job_url(direct_url)):
        return None
    if failure.reason_code == "salary_below_min" and not _is_close_salary_miss(job, settings):
        return None
    if failure.reason_code == "stale_posting" and not _is_close_stale_miss(job, settings):
        return None
    source_quality_score = max(
        lead.source_quality_score_hint or 0,
        job.source_quality_score or 0,
        _lead_source_quality_score(lead, settings),
    )
    if source_quality_score < 3:
        return None
    supporting_evidence = list(
        dict.fromkeys(
            [
                lead.evidence_notes,
                job.evidence_notes,
                *(job.validation_evidence or []),
            ]
        )
    )[:8]
    return NearMissJob(
        company_name=job.company_name,
        role_title=job.role_title,
        reason_code=failure.reason_code,
        detail=failure.detail,
        why_close=_near_miss_why_close(failure.reason_code, job),
        source_url=lead.source_url,
        direct_job_url=direct_url or None,
        source_type=lead.source_type,
        ats_platform=job.ats_platform,
        posted_date_text=job.posted_date_text,
        salary_text=job.salary_text,
        is_remote=job.is_fully_remote,
        supporting_evidence=supporting_evidence,
        validation_evidence=list(job.validation_evidence)[:8],
        close_score=source_quality_score + (2 if failure.reason_code in {"remote_unclear", "missing_salary"} else 0),
        source_quality_score=source_quality_score,
        attempt_number=failure.attempt_number,
        round_number=failure.round_number,
    )


def _build_false_negative_audit_entry(
    lead: JobLead,
    failure: SearchFailure,
    *,
    candidate: JobPosting | None = None,
    near_miss: NearMissJob | None = None,
) -> FalseNegativeAuditEntry | None:
    if failure.reason_code not in FALSE_NEGATIVE_AUDIT_REASON_CODES:
        return None
    verdict = "correct_rejection"
    notes = failure.detail
    role_title = candidate.role_title if candidate else lead.role_title
    if near_miss is not None:
        verdict = "near_miss"
        notes = near_miss.why_close
    elif failure.reason_code in {"fetch_non_200", "resolution_missing"}:
        verdict = "fixable"
        notes = "A strong discovery hit was blocked by URL resolution or page fetch reliability, not by role quality."
    elif failure.reason_code == "remote_unclear":
        verdict = "fixable"
        notes = "Remote evidence may be recoverable with better parsing or a cleaner direct page fetch."
    elif failure.reason_code == "salary_below_min" and candidate is not None and _contains_ai_signal(candidate.role_title):
        verdict = "fixable"
        notes = "Salary evidence may be incomplete or split across page fragments; re-check this title in audits."
    elif failure.reason_code == "not_ai_product_manager":
        haystack = " ".join(
            part
            for part in (
                role_title,
                candidate.job_page_title if candidate else "",
                " ".join(candidate.validation_evidence) if candidate else lead.evidence_notes,
            )
            if part
        )
        if "product manager" in haystack.lower() and _contains_ai_signal(haystack):
            verdict = "fixable"
            notes = "The title still carries PM and AI signal; this rejection belongs in the false-negative audit set."
    return FalseNegativeAuditEntry(
        reason_code=failure.reason_code,
        verdict=verdict,  # type: ignore[arg-type]
        company_name=failure.company_name,
        role_title=role_title,
        detail=failure.detail,
        notes=notes,
        source_url=failure.source_url,
        direct_job_url=failure.direct_job_url,
        salary_text=failure.salary_text,
        posted_date_text=failure.posted_date_text,
        is_remote=failure.is_remote,
        attempt_number=failure.attempt_number,
        round_number=failure.round_number,
    )


def _record_false_negative_audit(diagnostics: SearchDiagnostics, audit_entry: FalseNegativeAuditEntry | None) -> None:
    if audit_entry is None:
        return
    key = (
        audit_entry.reason_code,
        _normalize_company_key(audit_entry.company_name),
        _normalize_company_key(audit_entry.role_title),
    )
    existing_keys = {
        (
            item.reason_code,
            _normalize_company_key(item.company_name),
            _normalize_company_key(item.role_title),
        )
        for item in diagnostics.false_negative_audit
    }
    if key in existing_keys:
        return
    diagnostics.false_negative_audit.append(audit_entry)


def _record_near_miss(diagnostics: SearchDiagnostics, near_miss: NearMissJob | None) -> None:
    if near_miss is None:
        return
    key = _normalize_job_key(near_miss.company_name, near_miss.role_title)
    existing_keys = {
        _normalize_job_key(item.company_name, item.role_title)
        for item in diagnostics.near_misses
    }
    if key in existing_keys:
        return
    diagnostics.near_misses.append(near_miss)


def _record_failure(diagnostics: SearchDiagnostics, failure: SearchFailure) -> None:
    diagnostics.failures.append(failure)


def _record_failure_live(
    settings: Settings,
    diagnostics: SearchDiagnostics,
    failure: SearchFailure,
    *,
    unique_leads_discovered: int | None = None,
) -> None:
    if unique_leads_discovered is not None:
        diagnostics.unique_leads_discovered = unique_leads_discovered
    _record_failure(diagnostics, failure)
    _persist_search_diagnostics(settings, diagnostics)


def _record_failure_with_followups(
    settings: Settings,
    diagnostics: SearchDiagnostics,
    failure: SearchFailure,
    *,
    unique_leads_discovered: int | None = None,
    lead: JobLead | None = None,
    candidate: JobPosting | None = None,
    near_miss: NearMissJob | None = None,
    audit_entry: FalseNegativeAuditEntry | None = None,
    run_id: str | None = None,
) -> None:
    _record_failure_live(
        settings,
        diagnostics,
        failure,
        unique_leads_discovered=unique_leads_discovered,
    )
    _record_near_miss(diagnostics, near_miss)
    _record_false_negative_audit(diagnostics, audit_entry)
    if lead is not None and lead.refined_by_ollama:
        record_ollama_event(
            settings,
            "lead_refinement_rejection",
            run_id=run_id,
            caller="lead_refinement",
            prompt_category="lead_cleanup",
            company_name=lead.company_name,
            role_title=lead.role_title,
            reason_code=failure.reason_code,
            rejected_count=1,
            near_miss_count=int(near_miss is not None),
        )
    _persist_search_diagnostics(settings, diagnostics)


def _persist_search_diagnostics(settings: Settings, diagnostics: SearchDiagnostics) -> None:
    save_json_snapshot(
        settings.data_dir / "search-diagnostics-latest.json",
        diagnostics.model_dump(mode="json"),
    )


def _persist_validated_jobs_checkpoint(
    settings: Settings,
    jobs_by_url: dict[str, JobPosting],
    *,
    reacquired_jobs_by_url: dict[str, JobPosting] | None = None,
    run_id: str | None,
    diagnostics: SearchDiagnostics,
) -> None:
    jobs = sorted(jobs_by_url.values(), key=lambda item: (item.company_name.lower(), item.role_title.lower()))
    reacquired_jobs = sorted(
        (reacquired_jobs_by_url or {}).values(),
        key=lambda item: (item.company_name.lower(), item.role_title.lower()),
    )
    payload = {
        "run_id": run_id,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "qualifying_job_count": len(jobs),
        "reacquired_validated_job_count": len(reacquired_jobs),
        "total_current_validated_job_count": len(jobs) + len(reacquired_jobs),
        "unique_leads_discovered": diagnostics.unique_leads_discovered,
        "jobs": [job.model_dump(mode="json") for job in jobs],
        "reacquired_jobs": [job.model_dump(mode="json") for job in reacquired_jobs],
    }
    save_json_snapshot(settings.data_dir / "validated-jobs-checkpoint-latest.json", payload)
    if run_id:
        save_json_snapshot(settings.data_dir / f"validated-jobs-checkpoint-{run_id}.json", payload)


def _precheck_lead_hints(
    lead: JobLead,
    settings: Settings,
    *,
    attempt_number: int,
    round_number: int,
) -> SearchFailure | None:
    direct_url_failure = _lead_direct_job_url_precheck_failure(
        lead,
        attempt_number=attempt_number,
        round_number=round_number,
    )
    if direct_url_failure is not None:
        return direct_url_failure
    if "product manager" not in lead.role_title.lower():
        return _make_failure(
            stage="filter",
            reason_code="not_product_manager_hint",
            detail="Discovery lead title did not clearly indicate a product manager role.",
            lead=lead,
            attempt_number=attempt_number,
            round_number=round_number,
        )
    if lead.posted_date_hint and _hint_is_recent(lead.posted_date_hint, settings) is False:
        return _make_failure(
            stage="filter",
            reason_code="stale_posting",
            detail=f"Lead hint date '{lead.posted_date_hint}' was already outside the freshness window.",
            lead=lead,
            attempt_number=attempt_number,
            round_number=round_number,
        )
    combined_location_hint = _join_remote_restriction_context(
        lead.role_title,
        lead.location_hint,
        lead.evidence_notes,
        _unwrap_direct_job_url(lead.direct_job_url or ""),
    )
    if lead.is_remote_hint is False or any(token in combined_location_hint.lower() for token in ("hybrid", "on-site", "onsite", "in office")):
        return _make_failure(
            stage="filter",
            reason_code="not_remote",
            detail="Lead hints showed the role was not clearly fully remote.",
            lead=lead,
            attempt_number=attempt_number,
            round_number=round_number,
        )
    if _extract_geo_limited_remote_region(combined_location_hint):
        return _make_failure(
            stage="filter",
            reason_code="not_remote",
            detail="Lead hints showed the role was remote but geographically restricted.",
            lead=lead,
            attempt_number=attempt_number,
            round_number=round_number,
        )
    if lead.direct_job_url and _direct_job_url_has_specific_location_hint(lead.direct_job_url):
        lowered_hints = combined_location_hint.lower()
        if not any(marker in lowered_hints for marker in BROAD_REMOTE_OVERRIDE_MARKERS):
            return _make_failure(
                stage="filter",
                reason_code="not_remote",
                detail="Lead direct URL encoded a specific location while discovery hints did not clearly establish a broadly remote role.",
                lead=lead,
                attempt_number=attempt_number,
                round_number=round_number,
            )
    salary_min, salary_max, _salary_text = _hydrate_salary_hint_values(
        lead.base_salary_min_usd_hint,
        lead.base_salary_max_usd_hint,
        lead.salary_text_hint,
        lead.evidence_notes,
    )
    salary_values = [value for value in (salary_min, salary_max) if value is not None]
    if salary_values and max(salary_values) < settings.min_base_salary_usd:
        return _make_failure(
            stage="filter",
            reason_code="salary_below_min",
            detail=f"Lead salary hints were below the configured minimum of ${settings.min_base_salary_usd:,}.",
            lead=lead,
            attempt_number=attempt_number,
            round_number=round_number,
        )
    return None


def _is_generic_job_board_page(snapshot: JobPageSnapshot) -> bool:
    if _looks_like_generic_job_url(snapshot.resolved_url):
        return True
    page_title = (snapshot.page_title or "").lower()
    role_title = (snapshot.role_title or "").lower()
    text_excerpt = snapshot.text_excerpt.lower()
    if "error=true" in snapshot.resolved_url.lower():
        return True
    if page_title.startswith("jobs at "):
        return True
    if role_title in {"jobs", "job", "current openings"}:
        return True
    if "current openings" in text_excerpt and "powered by greenhouse" in text_excerpt:
        return True
    return False


def _is_insufficient_quota_error(exc: Exception) -> bool:
    return "insufficient_quota" in str(exc)


def _snapshot_remote_signal_is_low_confidence(snapshot: JobPageSnapshot) -> bool:
    text = " ".join(
        part for part in (snapshot.text_excerpt, " ".join(snapshot.evidence_snippets), snapshot.page_title or "") if part
    ).lower()
    if not text.strip():
        return True
    weak_patterns = (
        "enable javascript to run this app",
        "job you are trying to apply for has been filled",
        "job has been filled",
        "we're sorry",
    )
    return any(pattern in text for pattern in weak_patterns)


def _host_requires_strict_remote_confirmation(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(fragment in host for fragment in ("myworkdayjobs.com", "careers.workday.com", "icims.com"))


def _snapshot_location_is_specific_non_remote(location_text: str | None) -> bool:
    lowered = (location_text or "").lower().strip()
    if not lowered:
        return False
    if any(token in lowered for token in ("remote", "virtual", "telecommute", "telecommuting", "work from home")):
        return False
    if any(token in lowered for token in ("hybrid", "on-site", "onsite", "in office", "office")):
        return True
    if re.search(r"\b[a-z0-9 .&'-]+,\s*[a-z]{2}\b", lowered):
        return True
    if re.search(r"\b[a-z0-9 .&'-]+,\s*[a-z ]+\b", lowered):
        return True
    return bool(re.search(r"\b\d{2,}\s+[a-z0-9 .'-]+", lowered))


def _snapshot_has_strong_remote_evidence(snapshot: JobPageSnapshot) -> bool:
    haystack = " ".join(
        part
        for part in (
            snapshot.location_text,
            snapshot.page_title or "",
            snapshot.text_excerpt[:2500],
            " ".join(snapshot.evidence_snippets[:6]),
        )
        if part
    ).lower()
    if not haystack.strip():
        return False
    if any(token in haystack for token in ("hybrid", "on-site", "onsite", "in office", "required in office")):
        return False
    return any(
        token in haystack
        for token in (
            "fully remote",
            "100% remote",
            "remote-first",
            "remote only",
            "remote-only",
            "remote role",
            "remote position",
            "remote opportunity",
            "remote - united states",
            "remote, united states",
            "united states remote",
            "work persona : remote",
            "work persona: remote",
            "work from home",
            "work from anywhere",
            "virtual role",
            "virtual position",
        )
    )


def _snapshot_has_host_specific_remote_conflict(job_url: str, snapshot: JobPageSnapshot) -> bool:
    if not _host_requires_strict_remote_confirmation(job_url):
        return False
    if _direct_job_url_has_specific_location_hint(job_url) and not _snapshot_has_strong_remote_evidence(snapshot):
        return True
    if _snapshot_location_is_specific_non_remote(snapshot.location_text) and not _snapshot_has_strong_remote_evidence(snapshot):
        return True
    return False


def _direct_job_url_has_specific_location_hint(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()
    if "myworkdayjobs.com" in host:
        match = re.search(r"/job/([^/]+)/", path)
        if not match:
            return False
        location_slug = match.group(1).strip("-")
        if not location_slug:
            return False
        if any(token in location_slug for token in ("remote", "virtual", "work-from-home", "work_from_home")):
            return False
        return location_slug.count("-") >= 2
    return False


def _snapshot_role_title_is_specific(role_title: str | None) -> bool:
    normalized = (role_title or "").strip().lower()
    if not normalized:
        return False
    if normalized in {
        "careers",
        "career",
        "jobs",
        "job",
        "job application",
        "home",
        "overview",
        "benefits",
        "sign in",
        "login",
        "current openings",
        "open positions",
        "opportunities",
    }:
        return False
    if normalized.startswith("careers at ") or normalized.startswith("jobs at "):
        return False
    return True


def _snapshot_looks_like_generic_brand_page(snapshot: JobPageSnapshot, company_name: str | None = None) -> bool:
    title_candidates = [str(snapshot.role_title or "").strip(), str(snapshot.page_title or "").strip()]
    normalized_candidates = [candidate.lower() for candidate in title_candidates if candidate]
    if not normalized_candidates:
        return False
    if any(
        candidate in {"work", "apply", "home", "overview", "careers", "jobs", "job application"}
        or "careers" in candidate
        or "job opportunities" in candidate
        for candidate in normalized_candidates
    ):
        return True
    company_key = _normalize_company_key(company_name)
    return bool(company_key) and any(_normalize_company_key(candidate) == company_key for candidate in title_candidates if candidate)


def _snapshot_is_non_specific_company_page(
    snapshot: JobPageSnapshot,
    *,
    expected_company_name: str | None = None,
    expected_role_title: str | None = None,
    allow_trusted_source_role_fallback: bool = False,
) -> bool:
    resolved_host = (urlparse(snapshot.resolved_url).netloc or "").lower()
    normalized_path = (urlparse(snapshot.resolved_url).path or "").rstrip("/").lower()
    generic_brand_page = _snapshot_looks_like_generic_brand_page(snapshot, expected_company_name or snapshot.company_name)
    if _is_generic_job_board_page(snapshot):
        if not allow_trusted_source_role_fallback:
            return True
        if normalized_path in {
            "",
            "/",
            "/home",
            "/intro",
            "/benefits",
            "/v2/global/en/home",
            "/en/about/careers/benefits",
        }:
            return True
    if normalized_path in {
        "",
        "/",
        "/home",
        "/intro",
        "/benefits",
        "/v2/global/en/home",
        "/en/about/careers/benefits",
    }:
        return True

    if _looks_like_company_homepage_url(snapshot.resolved_url) and not _snapshot_role_title_is_specific(snapshot.role_title):
        return True

    if not any(fragment in resolved_host for fragment in ALLOWED_JOB_HOST_FRAGMENTS):
        if (
            expected_role_title
            and not _snapshot_supports_expected_role(expected_role_title, snapshot)
            and not (allow_trusted_source_role_fallback and generic_brand_page)
        ):
            return True
        if (
            not any(
                _snapshot_role_title_is_specific(candidate)
                for candidate in (snapshot.role_title, snapshot.page_title)
            )
            and not (allow_trusted_source_role_fallback and generic_brand_page)
        ):
            return True

    if _looks_like_careers_hub_url(snapshot.resolved_url) and expected_role_title and not _snapshot_supports_expected_role(
        expected_role_title,
        snapshot,
    ):
        return not allow_trusted_source_role_fallback

    if generic_brand_page:
        if allow_trusted_source_role_fallback:
            return False
        if expected_role_title and not _snapshot_supports_expected_role(expected_role_title, snapshot):
            return True
        return _looks_like_generic_job_url(snapshot.resolved_url)

    generic_page_title = " ".join(str(snapshot.page_title or "").lower().split())
    if generic_page_title in {"careers", "career", "jobs", "job application", "benefits", "home"}:
        return True

    return False


def _trusted_source_fallback_quality_score(
    lead: JobLead,
    candidate: JobPosting | None,
    settings: Settings,
) -> int:
    scores = [
        lead.source_quality_score_hint,
        candidate.source_quality_score if candidate is not None else None,
        _lead_source_quality_score(lead, settings),
    ]
    return max(int(score) for score in scores if score is not None)


def _lead_has_trusted_source_fallback_evidence(
    lead: JobLead,
    settings: Settings,
    candidate: JobPosting | None = None,
) -> bool:
    if lead.source_type not in {"builtin", "company_site", "direct_ats"}:
        return False
    salary_min, salary_max, _salary_text = _hydrate_salary_hint_values(
        lead.base_salary_min_usd_hint if lead.base_salary_min_usd_hint is not None else (candidate.base_salary_min_usd if candidate else None),
        lead.base_salary_max_usd_hint if lead.base_salary_max_usd_hint is not None else (candidate.base_salary_max_usd if candidate else None),
        lead.salary_text_hint or (candidate.salary_text if candidate else None),
        " ".join(part for part in (lead.evidence_notes, candidate.evidence_notes if candidate else "") if part),
    )
    salary_values = [value for value in (salary_min, salary_max) if value is not None]
    if not salary_values or max(salary_values) < settings.min_base_salary_usd:
        return False
    remote_hint = lead.is_remote_hint if lead.is_remote_hint is not None else (candidate.is_fully_remote if candidate else None)
    if remote_hint is not True:
        return False
    geo_hint_text = _join_remote_restriction_context(
        lead.location_hint,
        candidate.location_text if candidate else "",
        lead.evidence_notes,
        candidate.evidence_notes if candidate else "",
    )
    if _extract_geo_limited_remote_region(geo_hint_text):
        return False
    if not _lead_is_ai_related_product_manager(lead) and not (candidate and _is_ai_related_product_manager(candidate)):
        return False
    posted_date_hint = lead.posted_date_hint or (candidate.posted_date_iso if candidate else None) or (candidate.posted_date_text if candidate else "")
    return _is_recent_enough(
        None,
        posted_date_hint or "",
        settings.posted_within_days,
        timezone_name=settings.timezone,
    )


def _trusted_source_fallback_direct_url(
    lead: JobLead,
    candidate: JobPosting,
    settings: Settings,
) -> str | None:
    if not _lead_has_trusted_source_fallback_evidence(lead, settings, candidate):
        return None
    candidate_direct_urls = [
        str(candidate.direct_job_url or ""),
        str(candidate.resolved_job_url or ""),
        str(lead.direct_job_url or ""),
    ]
    strong_company_hosted_direct_url = any(
        direct_url
        and _url_has_strong_expected_company_hint(direct_url, lead.company_name)
        and _looks_like_company_job_page(direct_url)
        for direct_url in candidate_direct_urls
    )
    if (
        not strong_company_hosted_direct_url
        and _trusted_source_fallback_quality_score(lead, candidate, settings) < 16
    ):
        return None
    for direct_url in candidate_direct_urls:
        if not direct_url:
            continue
        if not _url_has_strong_expected_company_hint(direct_url, lead.company_name):
            continue
        if _looks_like_company_job_page(direct_url) or _is_allowed_direct_job_url(direct_url):
            return direct_url
    return None


def _should_accept_trusted_source_fallback_on_fetch_failure(
    lead: JobLead,
    candidate: JobPosting,
    settings: Settings,
) -> bool:
    return _trusted_source_fallback_direct_url(lead, candidate, settings) is not None


def _build_trusted_source_fallback_job(
    lead: JobLead,
    candidate: JobPosting,
    settings: Settings,
    *,
    status_code: int,
) -> JobPosting:
    source_quality_score = _trusted_source_fallback_quality_score(lead, candidate, settings)
    direct_url = _trusted_source_fallback_direct_url(lead, candidate, settings) or str(candidate.direct_job_url)
    evidence_note = (
        f"Accepted via trusted source fallback after the direct page returned HTTP {status_code}; "
        "remote, salary, freshness, and AI PM signals were retained from the trusted discovery source."
    )
    validation_evidence = list(dict.fromkeys([*candidate.validation_evidence, evidence_note]))[:8]
    return candidate.model_copy(
        update={
            "direct_job_url": direct_url,
            "resolved_job_url": direct_url,
            "validation_evidence": validation_evidence,
            "evidence_notes": " ".join(part for part in (candidate.evidence_notes, evidence_note) if part).strip(),
            "source_quality_score": source_quality_score,
        }
    )


def _job_has_geo_limited_remote_restriction(
    job: JobPosting,
    snapshot: JobPageSnapshot,
) -> bool:
    combined_text = _join_remote_restriction_context(
        job.role_title,
        job.location_text,
        job.job_page_title,
        job.evidence_notes,
        snapshot.role_title,
        snapshot.page_title,
        snapshot.location_text,
        snapshot.text_excerpt[:1600],
        " ".join(snapshot.evidence_snippets[:4]),
    )
    return _extract_geo_limited_remote_region(combined_text) is not None


def _candidate_has_strong_structured_remote_hints(candidate: JobPosting) -> bool:
    salary_values = _salary_values(candidate)
    remote_hint_text = " ".join(
        part
        for part in (
            candidate.location_text,
            candidate.evidence_notes,
            " ".join(candidate.validation_evidence[:4]),
        )
        if part
    ).lower()
    return (
        candidate.is_fully_remote is True
        and (candidate.source_quality_score or 0) >= 8
        and bool(candidate.posted_date_text or candidate.posted_date_iso)
        and bool(salary_values)
        and "remote" in remote_hint_text
    )


def _snapshot_supports_expected_role(expected_role_title: str | None, snapshot: JobPageSnapshot) -> bool:
    expected = " ".join(str(expected_role_title or "").split())
    if not expected:
        return True
    observed_role_title = snapshot.role_title if _snapshot_role_title_is_specific(snapshot.role_title) else None
    if observed_role_title and _role_titles_align(expected, observed_role_title):
        return True

    snapshot_haystack = " ".join(
        part
        for part in (
            observed_role_title or "",
            snapshot.page_title or "",
            snapshot.text_excerpt[:1600],
            " ".join(snapshot.evidence_snippets[:4]),
        )
        if part
    )
    if _role_match_score(expected, snapshot_haystack) < 10:
        return False
    if _is_ai_related_product_manager_text(expected):
        return _is_ai_related_product_manager_text(snapshot_haystack)
    return True


def _should_replace_candidate_role_title(candidate_role_title: str, snapshot: JobPageSnapshot) -> bool:
    observed_role_title = snapshot.role_title if _snapshot_role_title_is_specific(snapshot.role_title) else None
    if not observed_role_title:
        return False
    if not candidate_role_title.strip():
        return True
    if _role_titles_align(candidate_role_title, observed_role_title):
        return True
    return not _is_ai_related_product_manager_text(candidate_role_title) and _is_ai_related_product_manager_text(
        observed_role_title
    )


def _merge_candidate_with_snapshot(candidate: JobPosting, snapshot: JobPageSnapshot) -> JobPosting:
    remote_value = snapshot.is_fully_remote if snapshot.is_fully_remote is not None else candidate.is_fully_remote
    if (
        snapshot.is_fully_remote is False
        and candidate.is_fully_remote is True
        and _snapshot_remote_signal_is_low_confidence(snapshot)
        and not _snapshot_has_host_specific_remote_conflict(str(snapshot.resolved_url or candidate.direct_job_url), snapshot)
    ):
        remote_value = True
    if (
        snapshot.is_fully_remote is False
        and _candidate_has_strong_structured_remote_hints(candidate)
        and (
            _snapshot_has_strong_remote_evidence(snapshot)
            or not _snapshot_location_is_specific_non_remote(snapshot.location_text)
        )
        and not _snapshot_location_is_specific_non_remote(snapshot.location_text)
        and not _job_has_geo_limited_remote_restriction(candidate, snapshot)
        and not _snapshot_has_host_specific_remote_conflict(str(snapshot.resolved_url or candidate.direct_job_url), snapshot)
    ):
        remote_value = True

    merged = candidate.model_copy(
        update={
            "company_name": snapshot.company_name or candidate.company_name,
            "role_title": snapshot.role_title if _should_replace_candidate_role_title(candidate.role_title, snapshot) else candidate.role_title,
            "direct_job_url": snapshot.resolved_url,
            "resolved_job_url": snapshot.resolved_url,
            "ats_platform": snapshot.ats_platform or candidate.ats_platform,
            "location_text": snapshot.location_text or candidate.location_text,
            "is_fully_remote": remote_value,
            "posted_date_text": snapshot.posted_date_text or candidate.posted_date_text,
            "posted_date_iso": snapshot.posted_date_iso or candidate.posted_date_iso,
            "base_salary_min_usd": (
                snapshot.base_salary_min_usd
                if snapshot.base_salary_min_usd is not None
                else candidate.base_salary_min_usd
            ),
            "base_salary_max_usd": (
                snapshot.base_salary_max_usd
                if snapshot.base_salary_max_usd is not None
                else candidate.base_salary_max_usd
            ),
            "salary_text": snapshot.salary_text or candidate.salary_text,
            "job_page_title": snapshot.page_title or candidate.job_page_title,
            "validation_evidence": list(dict.fromkeys([*snapshot.evidence_snippets, *candidate.validation_evidence]))[:8],
            "evidence_notes": " ".join(
                part for part in (candidate.evidence_notes, " ".join(snapshot.evidence_snippets)) if part
            ).strip(),
        }
    )
    return merged


def _salary_is_base_salary(job: JobPosting, snapshot: JobPageSnapshot) -> bool:
    haystack = " ".join(
        part
        for part in (
            job.salary_text or "",
            snapshot.text_excerpt[:3000],
            job.evidence_notes,
        )
        if part
    ).lower()
    if re.search(r"\bon[-\s]?target earnings?\b|\bote\b", haystack):
        if "base salary" not in haystack:
            return False
    return True


def _should_accept_company_hosted_missing_posted_date(
    job: JobPosting,
    snapshot: JobPageSnapshot,
    settings: Settings,
    *,
    expected_company_name: str | None = None,
) -> bool:
    job_url = str(job.resolved_job_url or job.direct_job_url or "")
    host = (urlparse(job_url).netloc or "").lower()
    title_context = " ".join(
        part
        for part in (
            job.role_title,
            job.job_page_title or "",
            snapshot.role_title or "",
            snapshot.page_title or "",
        )
        if part
    )
    supports_high_salary_presumption = (
        _has_senior_title_signal(title_context)
        and _job_looks_us_remote_without_geo_limit(job, snapshot)
        and _is_ai_related_product_manager(job)
        and (job.source_quality_score or 0) >= 4
    )
    return (
        bool(job_url)
        and not any(fragment in host for fragment in ALLOWED_JOB_HOST_FRAGMENTS)
        and _looks_like_company_job_page(job_url)
        and _url_has_strong_expected_company_hint(job_url, expected_company_name or job.company_name)
        and (job.source_quality_score or 0) >= 4
        and job.is_fully_remote is True
        and (_salary_meets_minimum(job, settings) or supports_high_salary_presumption)
        and _is_ai_related_product_manager(job)
    )


def _evaluate_merged_job(
    job: JobPosting,
    snapshot: JobPageSnapshot,
    settings: Settings,
    *,
    expected_company_name: str | None = None,
    expected_role_title: str | None = None,
    allow_trusted_source_role_fallback: bool = False,
) -> tuple[str | None, str]:
    job_url = str(job.resolved_job_url or job.direct_job_url)
    if not _is_allowed_direct_job_url(job_url):
        return "direct_url_not_allowed", "Resolved URL is not a direct ATS or company careers page."

    if expected_company_name and not _direct_job_url_matches_expected_company(job_url, expected_company_name):
        company_hint = _company_hint_from_url(job_url)
        if not _is_weak_company_hint(company_hint):
            return "company_mismatch", f"Resolved job URL company hint '{company_hint}' did not match expected company '{expected_company_name}'."

    if _snapshot_is_non_specific_company_page(
        snapshot,
        expected_company_name=expected_company_name,
        expected_role_title=expected_role_title,
        allow_trusted_source_role_fallback=allow_trusted_source_role_fallback,
    ):
        return "not_specific_job_page", "Resolved page is a board index or an invalid job page, not a specific posting."

    if expected_company_name and not _is_weak_company_hint(expected_company_name) and not _company_names_match(expected_company_name, job.company_name):
        return "company_mismatch", f"Resolved job company '{job.company_name}' did not match expected company '{expected_company_name}'."

    if (
        expected_role_title
        and _snapshot_role_title_is_specific(snapshot.role_title)
        and not _snapshot_supports_expected_role(expected_role_title, snapshot)
        and not allow_trusted_source_role_fallback
    ):
        observed_role_title = snapshot.role_title or snapshot.page_title or "unknown role"
        return (
            "resolution_missing",
            f"Resolved page title '{observed_role_title}' did not line up with expected role '{expected_role_title}'.",
        )

    if not _is_ai_related_product_manager(job):
        return "not_ai_product_manager", "Role did not look like an AI-related product manager position."

    if job.is_fully_remote and _snapshot_has_host_specific_remote_conflict(job_url, snapshot):
        return "not_remote", "Resolved page encoded a specific location or office signal without strong host-specific remote confirmation."

    if (
        job.is_fully_remote is True
        and _snapshot_remote_signal_is_low_confidence(snapshot)
        and _direct_job_url_has_specific_location_hint(job_url)
    ):
        return "not_remote", "Resolved job URL encoded a specific location while remote evidence only came from weak discovery hints."

    if job.is_fully_remote and _job_has_geo_limited_remote_restriction(job, snapshot):
        return "not_remote", "Role was remote but geographically restricted rather than broadly fully remote."

    if job.is_fully_remote is False:
        return "not_remote", "Role was not clearly fully remote."
    if job.is_fully_remote is not True:
        return "remote_unclear", "Remote evidence was ambiguous rather than clearly fully remote."

    if not job.posted_date_iso and not job.posted_date_text:
        if _should_accept_company_hosted_missing_posted_date(
            job,
            snapshot,
            settings,
            expected_company_name=expected_company_name,
        ):
            return None, None
        return "missing_posted_date", "No posted date was available from the direct page or trusted hints."

    if not _is_recent_enough(
        job.posted_date_iso,
        job.posted_date_text or "",
        settings.posted_within_days,
        timezone_name=settings.timezone,
    ):
        return "stale_posting", f"Posting date '{job.posted_date_text or job.posted_date_iso}' was older than {settings.posted_within_days} days."

    salary_values = _salary_values(job)
    if salary_values and max(salary_values) < settings.min_base_salary_usd:
        return "salary_below_min", f"Salary was below the configured minimum of ${settings.min_base_salary_usd:,}."

    if salary_values and not _salary_is_base_salary(job, snapshot) and not job.salary_inferred:
        return "salary_not_base", "Compensation looked like OTE or total compensation instead of base salary."

    if not salary_values and not job.salary_inferred:
        return "missing_salary", "No salary range was available from the direct page or trusted hints."

    return None, "accepted"


async def _validate_candidate(
    lead: JobLead,
    candidate: JobPosting,
    settings: Settings,
    *,
    resolution_agent: Agent | None,
    attempt_number: int,
    round_number: int,
) -> tuple[JobPosting | None, SearchFailure | None, NearMissJob | None, FalseNegativeAuditEntry | None]:
    snapshot = await fetch_job_page(str(candidate.direct_job_url))
    if snapshot.status_code != 200:
        repaired_url = await _repair_direct_job_url(
            resolution_agent,
            lead,
            str(candidate.direct_job_url),
            f"HTTP {snapshot.status_code}",
        )
        if repaired_url:
            candidate = candidate.model_copy(update={"direct_job_url": repaired_url, "resolved_job_url": repaired_url})
            snapshot = await fetch_job_page(repaired_url)

    if snapshot.status_code != 200:
        if _should_accept_trusted_source_fallback_on_fetch_failure(lead, candidate, settings):
            fallback_job = _build_trusted_source_fallback_job(
                lead,
                candidate,
                settings,
                status_code=snapshot.status_code,
            )
            return fallback_job, None, None, None
        failure = _make_failure(
            stage="validation",
            reason_code="fetch_non_200",
            detail=f"Direct job page returned HTTP {snapshot.status_code}.",
            lead=lead,
            direct_job_url=str(candidate.direct_job_url),
            candidate=candidate,
            attempt_number=attempt_number,
            round_number=round_number,
        )
        near_miss = None
        if (lead.direct_job_url or candidate.direct_job_url) and _lead_source_quality_score(lead, settings) >= 4:
            near_miss = NearMissJob(
                company_name=candidate.company_name,
                role_title=candidate.role_title,
                reason_code=failure.reason_code,
                detail=failure.detail,
                why_close=_near_miss_why_close(failure.reason_code, candidate),
                source_url=lead.source_url,
                direct_job_url=str(candidate.direct_job_url),
                source_type=lead.source_type,
                ats_platform=candidate.ats_platform,
                posted_date_text=candidate.posted_date_text,
                salary_text=candidate.salary_text,
                is_remote=candidate.is_fully_remote,
                supporting_evidence=[lead.evidence_notes],
                close_score=_lead_source_quality_score(lead, settings),
                source_quality_score=_lead_source_quality_score(lead, settings),
                attempt_number=attempt_number,
                round_number=round_number,
            )
        return None, failure, near_miss, _build_false_negative_audit_entry(
            lead,
            failure,
            candidate=candidate,
            near_miss=near_miss,
        )

    if _is_generic_job_board_page(snapshot):
        repaired_url = await _repair_direct_job_url(
            resolution_agent,
            lead,
            str(candidate.direct_job_url),
            "Resolved to a generic board index or invalid job page.",
        )
        if repaired_url:
            candidate = candidate.model_copy(update={"direct_job_url": repaired_url, "resolved_job_url": repaired_url})
            snapshot = await fetch_job_page(repaired_url)

    merged_job = _apply_salary_inference(_merge_candidate_with_snapshot(candidate, snapshot), snapshot, settings)
    merged_job = merged_job.model_copy(
        update={
            "lead_refined_by_ollama": lead.refined_by_ollama,
            "source_quality_score": lead.source_quality_score_hint or _lead_source_quality_score(lead, settings),
        }
    )
    allow_trusted_source_role_fallback = (
        _snapshot_looks_like_generic_brand_page(snapshot, lead.company_name)
        and _lead_has_trusted_source_fallback_evidence(lead, settings)
        and _url_has_strong_expected_company_hint(snapshot.resolved_url, lead.company_name)
    )
    reason_code, detail = _evaluate_merged_job(
        merged_job,
        snapshot,
        settings,
        expected_company_name=lead.company_name,
        expected_role_title=lead.role_title,
        allow_trusted_source_role_fallback=allow_trusted_source_role_fallback,
    )
    if reason_code in {"resolution_missing", "company_mismatch", "not_specific_job_page"}:
        repaired_url = await _repair_direct_job_url(
            resolution_agent,
            lead,
            str(merged_job.direct_job_url),
            detail,
        )
        if repaired_url:
            repaired_candidate = candidate.model_copy(update={"direct_job_url": repaired_url, "resolved_job_url": repaired_url})
            repaired_snapshot = await fetch_job_page(repaired_url)
            if repaired_snapshot.status_code == 200:
                repaired_merged_job = _apply_salary_inference(
                    _merge_candidate_with_snapshot(repaired_candidate, repaired_snapshot),
                    repaired_snapshot,
                    settings,
                )
                repaired_merged_job = repaired_merged_job.model_copy(
                    update={
                        "lead_refined_by_ollama": lead.refined_by_ollama,
                        "source_quality_score": lead.source_quality_score_hint or _lead_source_quality_score(lead, settings),
                    }
                )
                repaired_allow_trusted_source_role_fallback = (
                    _snapshot_looks_like_generic_brand_page(repaired_snapshot, lead.company_name)
                    and _lead_has_trusted_source_fallback_evidence(lead, settings)
                    and _url_has_strong_expected_company_hint(repaired_snapshot.resolved_url, lead.company_name)
                )
                repaired_reason_code, repaired_detail = _evaluate_merged_job(
                    repaired_merged_job,
                    repaired_snapshot,
                    settings,
                    expected_company_name=lead.company_name,
                    expected_role_title=lead.role_title,
                    allow_trusted_source_role_fallback=repaired_allow_trusted_source_role_fallback,
                )
                if not repaired_reason_code:
                    return repaired_merged_job, None, None, None
                merged_job = repaired_merged_job
                snapshot = repaired_snapshot
                reason_code, detail = repaired_reason_code, repaired_detail
    if reason_code:
        failure = _make_failure(
            stage="validation",
            reason_code=reason_code,
            detail=detail,
            lead=lead,
            direct_job_url=str(merged_job.direct_job_url),
            candidate=merged_job,
            attempt_number=attempt_number,
            round_number=round_number,
        )
        near_miss = _build_near_miss(lead, merged_job, failure, settings)
        return None, failure, near_miss, _build_false_negative_audit_entry(
            lead,
            failure,
            candidate=merged_job,
            near_miss=near_miss,
        )
    return merged_job, None, None, None


async def find_matching_jobs(
    settings: Settings,
    status: StatusReporter | None = None,
    *,
    run_id: str | None = None,
) -> tuple[list[JobPosting], list[JobPosting], int, SearchDiagnostics]:
    stop_goal = max(1, settings.minimum_qualifying_jobs, settings.target_job_count)
    diagnostics = SearchDiagnostics(run_id=run_id, minimum_qualifying_jobs=stop_goal)
    _persist_search_diagnostics(settings, diagnostics)
    jobs_by_url: dict[str, JobPosting] = {}
    reacquired_jobs_by_url: dict[str, JobPosting] = {}
    _persist_validated_jobs_checkpoint(
        settings,
        jobs_by_url,
        reacquired_jobs_by_url=reacquired_jobs_by_url,
        run_id=run_id,
        diagnostics=diagnostics,
    )
    validated_job_history_index = load_validated_job_history_index(settings.data_dir)
    previously_reported_company_keys = load_previously_reported_company_keys(settings.data_dir)
    company_watchlist = load_company_watchlist_entries(settings.data_dir)
    company_discovery_entries = load_company_discovery_entries(settings.data_dir) if settings.company_discovery_enabled else {}
    failed_lead_history = _load_failed_lead_history(settings)
    query_family_history = _load_query_family_history(settings)
    query_family_metrics: dict[str, dict[str, int]] = {}
    seen_lead_keys: set[str] = set()
    reacquisition_attempted_keys: set[str] = set()
    reacquisition_suppressed_keys: set[str] = set()
    total_unique_leads = 0

    openai_agents_enabled = settings.llm_provider == "openai" or settings.use_openai_fallback
    resolution_agent = build_direct_job_resolution_agent(settings) if openai_agents_enabled else None

    tuning = SearchTuning(
        attempt_number=1,
        prioritize_recency=True,
        prioritize_salary=settings.min_base_salary_usd >= 180000,
        prioritize_remote=True,
        focus_companies=_select_watchlist_focus_companies(
            settings,
            previously_reported_company_keys,
            company_discovery_entries=company_discovery_entries,
        ),
    )
    for attempt_number in range(1, settings.max_adaptive_search_passes + 1):
        if len(jobs_by_url) >= stop_goal:
            break

        if status:
            status.emit(
                "search",
                f"Adaptive search pass {attempt_number} started.",
                attempt_number=attempt_number,
                target_job_count=settings.target_job_count,
                minimum_qualifying_jobs=stop_goal,
                qualifying_jobs=len(jobs_by_url),
            )

        discovery_agent = build_job_discovery_agent(settings, tuning) if openai_agents_enabled else None
        attempt_query_family_metrics: dict[str, dict[str, int]] = {}
        query_rounds = _build_query_rounds(
            settings,
            tuning=tuning,
            query_family_history=query_family_history,
            run_id=run_id,
        )
        attempt_start_leads = total_unique_leads
        resolved_leads_this_attempt = 0
        lead_timeout_seconds = settings.per_lead_timeout_seconds
        if _ollama_inline_refinement_enabled(settings):
            lead_timeout_seconds = max(lead_timeout_seconds, 40)

        if attempt_number == 1 and len(jobs_by_url) < stop_goal:
            if settings.company_discovery_enabled:
                company_discovery_seed_leads, discovery_metrics = await _collect_company_discovery_seed_leads(
                    settings,
                    discovery_agent=discovery_agent,
                    run_id=run_id,
                    previously_reported_company_keys=previously_reported_company_keys,
                )
                diagnostics.new_companies_discovered_count += int(
                    discovery_metrics.get("new_companies_discovered_count") or 0
                )
                diagnostics.new_boards_discovered_count += int(
                    discovery_metrics.get("new_boards_discovered_count") or 0
                )
                diagnostics.official_board_leads_count += int(
                    discovery_metrics.get("official_board_leads_count") or 0
                )
                diagnostics.companies_with_ai_pm_leads_count += int(
                    discovery_metrics.get("companies_with_ai_pm_leads_count") or 0
                )
                diagnostics.official_roles_missed_count += int(
                    discovery_metrics.get("official_roles_missed_count") or 0
                )
                diagnostics.frontier_tasks_consumed_count += int(
                    discovery_metrics.get("frontier_tasks_consumed_count") or 0
                )
                diagnostics.frontier_backlog_count = int(
                    discovery_metrics.get("frontier_backlog_count") or diagnostics.frontier_backlog_count
                )
                diagnostics.official_board_crawl_attempt_count += int(
                    discovery_metrics.get("official_board_crawl_attempt_count") or 0
                )
                diagnostics.official_board_crawl_success_count += int(
                    discovery_metrics.get("official_board_crawl_success_count") or 0
                )
                for adapter_key, adapter_count in dict(discovery_metrics.get("source_adapter_yields") or {}).items():
                    _increment_metric_count(
                        diagnostics.source_adapter_yields,
                        str(adapter_key),
                        int(adapter_count or 0),
                    )
                company_discovery_entries = load_company_discovery_entries(settings.data_dir)
                _persist_search_diagnostics(settings, diagnostics)
                if company_discovery_seed_leads:
                    total_unique_leads, resolved_leads_this_attempt = await _replay_seed_leads(
                        company_discovery_seed_leads,
                        settings=settings,
                        diagnostics=diagnostics,
                        company_watchlist=company_watchlist,
                        failed_lead_history=failed_lead_history,
                        jobs_by_url=jobs_by_url,
                        reacquired_jobs_by_url=reacquired_jobs_by_url,
                        previously_reported_company_keys=previously_reported_company_keys,
                        validated_job_history_index=validated_job_history_index,
                        reacquisition_attempted_keys=reacquisition_attempted_keys,
                        reacquisition_suppressed_keys=reacquisition_suppressed_keys,
                        seen_lead_keys=seen_lead_keys,
                        total_unique_leads=total_unique_leads,
                        resolved_leads_this_attempt=resolved_leads_this_attempt,
                        stop_goal=stop_goal,
                        lead_timeout_seconds=lead_timeout_seconds,
                        resolution_agent=resolution_agent,
                        attempt_number=attempt_number,
                        status=status,
                        run_id=run_id,
                        track_as_seed_replay=False,
                        company_discovery_entries=company_discovery_entries,
                        status_label=(
                            f"Pass {attempt_number}, company discovery surfaced "
                            f"{len(company_discovery_seed_leads)} official ATS candidates."
                        ),
                    )
                    if len(jobs_by_url) >= stop_goal:
                        diagnostics.unique_leads_discovered = total_unique_leads
                        _persist_search_diagnostics(settings, diagnostics)
                        break

            total_unique_leads, resolved_leads_this_attempt = await _replay_seed_leads(
                _collect_replay_seed_leads(settings),
                settings=settings,
                diagnostics=diagnostics,
                company_watchlist=company_watchlist,
                failed_lead_history=failed_lead_history,
                jobs_by_url=jobs_by_url,
                reacquired_jobs_by_url=reacquired_jobs_by_url,
                previously_reported_company_keys=previously_reported_company_keys,
                validated_job_history_index=validated_job_history_index,
                reacquisition_attempted_keys=reacquisition_attempted_keys,
                reacquisition_suppressed_keys=reacquisition_suppressed_keys,
                seen_lead_keys=seen_lead_keys,
                total_unique_leads=total_unique_leads,
                resolved_leads_this_attempt=resolved_leads_this_attempt,
                stop_goal=stop_goal,
                lead_timeout_seconds=lead_timeout_seconds,
                resolution_agent=resolution_agent,
                attempt_number=attempt_number,
                status=status,
                run_id=run_id,
                company_discovery_entries=company_discovery_entries,
            )
            if len(jobs_by_url) >= stop_goal:
                diagnostics.unique_leads_discovered = total_unique_leads
                _persist_search_diagnostics(settings, diagnostics)
                break

        consecutive_zero_yield_rounds = 0
        for round_number, queries in enumerate(query_rounds, start=1):
            if len(jobs_by_url) >= stop_goal:
                break
            if resolved_leads_this_attempt >= settings.max_leads_to_resolve_per_pass:
                break

            if status:
                status.emit(
                    "search",
                    f"Pass {attempt_number}, round {round_number} started with {len(queries)} queries.",
                    attempt_number=attempt_number,
                    round_number=round_number,
                    qualifying_jobs=len(jobs_by_url),
                )

            round_leads: list[JobLead] = []
            query_batches = [queries]
            if settings.llm_provider == "ollama":
                query_batches = _chunk_queries(queries, 3)

            for query_batch in query_batches:
                runnable_queries: list[str] = []
                for query in query_batch:
                    skip_reason = _query_timeout_skip_reason(
                        diagnostics,
                        query,
                        attempt_number=attempt_number,
                        query_family_history=query_family_history,
                        attempt_query_family_metrics=attempt_query_family_metrics,
                        run_id=run_id,
                    )
                    if skip_reason is None:
                        runnable_queries.append(query)
                        continue
                    _record_failure_live(
                        settings,
                        diagnostics,
                        _make_failure(
                            stage="discovery",
                            reason_code="query_skipped_timeout_budget",
                            detail=skip_reason,
                            source_query=query,
                            attempt_number=attempt_number,
                            round_number=round_number,
                        ),
                        unique_leads_discovered=total_unique_leads,
                    )
                    if status:
                        status.emit(
                            "search",
                            f"Skipping a low-yield discovery query during pass {attempt_number}: {query}",
                            attempt_number=attempt_number,
                            round_number=round_number,
                            qualifying_jobs=len(jobs_by_url),
                        )

                if not runnable_queries:
                    continue

                tasks = [
                    asyncio.create_task(
                        _search_query_with_context(
                            discovery_agent,
                            settings,
                            query,
                            _query_timeout_seconds_for_query(settings, query),
                            attempt_number=attempt_number,
                            run_id=run_id,
                        )
                    )
                    for query in runnable_queries
                ]

                for task in asyncio.as_completed(tasks):
                    try:
                        query, results = await task
                    except SearchQueryTimeoutError as exc:
                        _update_query_family_metric_sets(
                            (query_family_metrics, attempt_query_family_metrics),
                            exc.query,
                            timed_out=True,
                        )
                        _record_failure_live(
                            settings,
                            diagnostics,
                            _make_failure(
                                stage="discovery",
                                reason_code="query_timeout",
                                detail=(
                                    "Discovery query timed out after "
                                    f"{_query_timeout_seconds_for_query(settings, exc.query)}s: {exc.query}"
                                ),
                                source_query=exc.query,
                                attempt_number=attempt_number,
                                round_number=round_number,
                            ),
                            unique_leads_discovered=total_unique_leads,
                        )
                        if status:
                            status.emit(
                                "search",
                                f"A discovery query timed out during pass {attempt_number}: {exc.query}",
                                attempt_number=attempt_number,
                                round_number=round_number,
                                qualifying_jobs=len(jobs_by_url),
                            )
                        continue
                    except SearchQueryExecutionError as exc:
                        _update_query_family_metric_sets((query_family_metrics, attempt_query_family_metrics), exc.query)
                        if _is_insufficient_quota_error(exc.cause):
                            _record_failure_live(
                                settings,
                                diagnostics,
                                _make_failure(
                                    stage="discovery",
                                    reason_code="openai_insufficient_quota",
                                    detail=f"Discovery query failed because the OpenAI API quota was exhausted: {exc.cause}",
                                    source_query=exc.query,
                                    attempt_number=attempt_number,
                                    round_number=round_number,
                                ),
                                unique_leads_discovered=total_unique_leads,
                            )
                            raise OpenAIQuotaExceededError(
                                "OpenAI API quota is exhausted. Add billing or increase quota, then rerun the workflow."
                            ) from exc
                        _record_failure_live(
                            settings,
                            diagnostics,
                            _make_failure(
                                stage="discovery",
                                reason_code="query_failed",
                                detail=f"Discovery query failed: {_describe_exception(exc.cause)}",
                                source_query=exc.query,
                                attempt_number=attempt_number,
                                round_number=round_number,
                            ),
                            unique_leads_discovered=total_unique_leads,
                        )
                        if status:
                            status.emit(
                                "search",
                                f"A discovery query failed during pass {attempt_number}: {exc.query}",
                                attempt_number=attempt_number,
                                round_number=round_number,
                                qualifying_jobs=len(jobs_by_url),
                            )
                        continue

                    fresh_leads = 0
                    new_companies = 0
                    for lead in results:
                        failed_history_skip = _failed_lead_history_skip_reason(lead, settings, failed_lead_history)
                        if failed_history_skip is not None:
                            reason_code, detail = failed_history_skip
                            _record_failure_live(
                                settings,
                                diagnostics,
                                _make_failure(
                                    stage="filter",
                                    reason_code="repeated_failed_lead",
                                    detail=detail,
                                    lead=lead,
                                    attempt_number=attempt_number,
                                    round_number=round_number,
                                ),
                                unique_leads_discovered=total_unique_leads,
                            )
                            continue
                        key = _lead_dedupe_key(lead)
                        if key in seen_lead_keys:
                            continue
                        seen_lead_keys.add(key)
                        total_unique_leads += 1
                        company_key = _normalize_company_key(lead.company_name)
                        if company_key and company_key not in diagnostics.company_lead_counts:
                            new_companies += 1
                        _increment_metric_count(diagnostics.company_lead_counts, company_key)
                        fresh_leads += 1
                        round_leads.append(lead)

                    if status:
                        status.emit(
                            "search",
                            f"Query finished: '{query}' produced {fresh_leads} new leads.",
                            attempt_number=attempt_number,
                            round_number=round_number,
                            query=query,
                            unique_leads_discovered=total_unique_leads,
                            qualifying_jobs=len(jobs_by_url),
                        )
                    _update_query_family_run_metrics(
                        query_family_metrics,
                        query,
                        fresh_leads=fresh_leads,
                        new_companies=new_companies,
                    )
                    _update_query_family_run_metrics(
                        attempt_query_family_metrics,
                        query,
                        fresh_leads=fresh_leads,
                        new_companies=new_companies,
                    )
                    diagnostics.unique_leads_discovered = total_unique_leads
                    _persist_search_diagnostics(settings, diagnostics)

            round_leads = _dedupe_round_leads(round_leads, settings)
            round_leads = _annotate_and_filter_resolution_leads(round_leads, settings, company_watchlist)
            remaining_resolution_budget = max(0, settings.max_leads_to_resolve_per_pass - resolved_leads_this_attempt)
            known_company_keys_for_round = previously_reported_company_keys.union(
                {_normalize_company_key(job.company_name) for job in jobs_by_url.values()}
            )
            round_leads = _apply_company_novelty_quota(
                round_leads,
                known_company_keys_for_round,
                min_novelty_ratio=NOVEL_COMPANY_TARGET_RATIO,
                limit=remaining_resolution_budget or None,
            )
            round_leads = await _maybe_force_round_lead_refinement_with_ollama(
                settings,
                round_leads,
                attempt_number=attempt_number,
                round_number=round_number,
                run_id=run_id,
            )
            if round_leads:
                consecutive_zero_yield_rounds = 0
            else:
                consecutive_zero_yield_rounds += 1
            if status:
                status.emit(
                    "search",
                    f"Pass {attempt_number}, round {round_number} yielded {len(round_leads)} unique leads to resolve.",
                    attempt_number=attempt_number,
                    round_number=round_number,
                    unique_leads_discovered=total_unique_leads,
                    qualifying_jobs=len(jobs_by_url),
                )

            if _should_abort_dead_attempt_round(
                diagnostics,
                attempt_number=attempt_number,
                consecutive_zero_yield_rounds=consecutive_zero_yield_rounds,
                attempt_discovery_gain=total_unique_leads - attempt_start_leads,
            ):
                if status:
                    status.emit(
                        "search",
                        (
                            f"Ending pass {attempt_number} early after {consecutive_zero_yield_rounds} "
                            "zero-yield rounds and repeated timeout-budget failures."
                        ),
                        attempt_number=attempt_number,
                        round_number=round_number,
                        unique_leads_discovered=total_unique_leads,
                        qualifying_jobs=len(jobs_by_url),
                    )
                break

            for lead_index, lead in enumerate(round_leads, start=1):
                if len(jobs_by_url) >= stop_goal:
                    break
                if resolved_leads_this_attempt >= settings.max_leads_to_resolve_per_pass:
                    break

                precheck_failure = _precheck_lead_hints(
                    lead,
                    settings,
                    attempt_number=attempt_number,
                    round_number=round_number,
                )
                if precheck_failure is not None:
                    _record_failure_with_followups(
                        settings,
                        diagnostics,
                        precheck_failure,
                        unique_leads_discovered=total_unique_leads,
                        lead=lead,
                        audit_entry=_build_false_negative_audit_entry(lead, precheck_failure),
                        run_id=run_id,
                    )
                    continue

                if status:
                    status.emit(
                        "search",
                        f"Resolving lead {lead_index}/{len(round_leads)}: {lead.company_name} | {lead.role_title}",
                        attempt_number=attempt_number,
                        round_number=round_number,
                        lead_index=lead_index,
                        round_lead_count=len(round_leads),
                        unique_leads_discovered=total_unique_leads,
                        qualifying_jobs=len(jobs_by_url),
                    )

                resolved_leads_this_attempt += 1
                try:
                    resolution = await asyncio.wait_for(
                        _resolve_lead_to_direct_job_url(resolution_agent, lead),
                        timeout=lead_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    failure = _make_failure(
                        stage="resolution",
                        reason_code="resolution_timeout",
                        detail=f"Resolution timed out after {lead_timeout_seconds}s.",
                        lead=lead,
                        attempt_number=attempt_number,
                        round_number=round_number,
                    )
                    _record_failure_with_followups(
                        settings,
                        diagnostics,
                        failure,
                        unique_leads_discovered=total_unique_leads,
                        lead=lead,
                        audit_entry=_build_false_negative_audit_entry(lead, failure),
                        run_id=run_id,
                    )
                    continue
                except Exception as exc:
                    if _is_insufficient_quota_error(exc):
                        _record_failure_live(
                            settings,
                            diagnostics,
                            _make_failure(
                                stage="resolution",
                                reason_code="openai_insufficient_quota",
                                detail=f"Resolution failed because the OpenAI API quota was exhausted: {exc}",
                                lead=lead,
                                attempt_number=attempt_number,
                                round_number=round_number,
                            ),
                            unique_leads_discovered=total_unique_leads,
                        )
                        raise OpenAIQuotaExceededError(
                            "OpenAI API quota is exhausted. Add billing or increase quota, then rerun the workflow."
                        ) from exc
                    failure = _make_failure(
                        stage="resolution",
                        reason_code="resolution_error",
                        detail=f"Resolution failed with an exception: {exc}",
                        lead=lead,
                        attempt_number=attempt_number,
                        round_number=round_number,
                    )
                    _record_failure_with_followups(
                        settings,
                        diagnostics,
                        failure,
                        unique_leads_discovered=total_unique_leads,
                        lead=lead,
                        audit_entry=_build_false_negative_audit_entry(lead, failure),
                        run_id=run_id,
                    )
                    continue

                if resolution is None:
                    failure = _make_failure(
                        stage="resolution",
                        reason_code="resolution_missing",
                        detail="Could not resolve the discovery lead to a direct ATS or company careers URL.",
                        lead=lead,
                        attempt_number=attempt_number,
                        round_number=round_number,
                    )
                    fallback_candidate = _build_candidate_job(
                        lead,
                        DirectJobResolution(
                            accepted=True,
                            direct_job_url=lead.source_url,
                            ats_platform=urlparse(lead.source_url).netloc or "Unknown",
                            evidence_notes="Resolution failed after strong discovery hit.",
                        ),
                    )
                    near_miss = _build_near_miss(lead, fallback_candidate, failure, settings)
                    _record_failure_with_followups(
                        settings,
                        diagnostics,
                        failure,
                        unique_leads_discovered=total_unique_leads,
                        lead=lead,
                        candidate=fallback_candidate,
                        near_miss=near_miss,
                        audit_entry=_build_false_negative_audit_entry(
                            lead,
                            failure,
                            candidate=fallback_candidate,
                            near_miss=near_miss,
                        ),
                        run_id=run_id,
                    )
                    continue

                if not _is_allowed_direct_job_url(resolution.direct_job_url or ""):
                    failure = _make_failure(
                        stage="resolution",
                        reason_code="resolution_blocked_url",
                        detail="Resolution returned a blocked or aggregator URL.",
                        lead=lead,
                        direct_job_url=resolution.direct_job_url,
                        attempt_number=attempt_number,
                        round_number=round_number,
                    )
                    _record_failure_with_followups(
                        settings,
                        diagnostics,
                        failure,
                        unique_leads_discovered=total_unique_leads,
                        lead=lead,
                        audit_entry=_build_false_negative_audit_entry(lead, failure),
                        run_id=run_id,
                    )
                    continue

                normalized_resolution_url = _normalize_direct_job_url(resolution.direct_job_url or "")
                reacquisition_entry = _validated_job_history_entry_for_url(
                    normalized_resolution_url,
                    validated_job_history_index,
                )
                reacquisition_key = str((reacquisition_entry or {}).get("canonical_job_key") or (reacquisition_entry or {}).get("job_key") or "").strip()
                if reacquisition_entry is not None:
                    if not _lead_is_reacquisition_eligible(lead, settings, direct_job_url=normalized_resolution_url):
                        if reacquisition_key and reacquisition_key not in reacquisition_suppressed_keys:
                            reacquisition_suppressed_keys.add(reacquisition_key)
                            diagnostics.reacquired_jobs_suppressed_count += 1
                        _record_failure_live(
                            settings,
                            diagnostics,
                            _make_failure(
                                stage="filter",
                                reason_code="reacquisition_suppressed",
                                detail="Skipping a previously validated job because the repeat hit came from a low-trust or non-direct source.",
                                lead=lead,
                                direct_job_url=normalized_resolution_url,
                                attempt_number=attempt_number,
                                round_number=round_number,
                            ),
                            unique_leads_discovered=total_unique_leads,
                        )
                        continue
                    if len(reacquisition_attempted_keys) >= settings.reacquisition_attempt_cap and (
                        not reacquisition_key or reacquisition_key not in reacquisition_attempted_keys
                    ):
                        if reacquisition_key and reacquisition_key not in reacquisition_suppressed_keys:
                            reacquisition_suppressed_keys.add(reacquisition_key)
                            diagnostics.reacquired_jobs_suppressed_count += 1
                        _record_failure_live(
                            settings,
                            diagnostics,
                            _make_failure(
                                stage="filter",
                                reason_code="reacquisition_suppressed",
                                detail=(
                                    "Skipping a previously validated job because the per-run reacquisition cap "
                                    f"({settings.reacquisition_attempt_cap}) was reached."
                                ),
                                lead=lead,
                                direct_job_url=normalized_resolution_url,
                                attempt_number=attempt_number,
                                round_number=round_number,
                            ),
                            unique_leads_discovered=total_unique_leads,
                        )
                        continue
                    if reacquisition_key and reacquisition_key not in reacquisition_attempted_keys:
                        reacquisition_attempted_keys.add(reacquisition_key)
                        diagnostics.reacquisition_attempt_count += 1

                candidate = _build_candidate_job(lead, resolution)
                try:
                    validated, validation_failure, near_miss, audit_entry = await asyncio.wait_for(
                        _validate_candidate(
                            lead,
                            candidate,
                            settings,
                            resolution_agent=resolution_agent,
                            attempt_number=attempt_number,
                            round_number=round_number,
                        ),
                        timeout=lead_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    failure = _make_failure(
                        stage="validation",
                        reason_code="validation_timeout",
                        detail=f"Validation timed out after {lead_timeout_seconds}s.",
                        lead=lead,
                        direct_job_url=str(candidate.direct_job_url),
                        candidate=candidate,
                        attempt_number=attempt_number,
                        round_number=round_number,
                    )
                    _record_failure_with_followups(
                        settings,
                        diagnostics,
                        failure,
                        unique_leads_discovered=total_unique_leads,
                        lead=lead,
                        candidate=candidate,
                        audit_entry=_build_false_negative_audit_entry(lead, failure, candidate=candidate),
                        run_id=run_id,
                    )
                    continue
                except Exception as exc:
                    if _is_insufficient_quota_error(exc):
                        _record_failure_live(
                            settings,
                            diagnostics,
                            _make_failure(
                                stage="validation",
                                reason_code="openai_insufficient_quota",
                                detail=f"Validation failed because the OpenAI API quota was exhausted: {exc}",
                                lead=lead,
                                direct_job_url=str(candidate.direct_job_url),
                                candidate=candidate,
                                attempt_number=attempt_number,
                                round_number=round_number,
                            ),
                            unique_leads_discovered=total_unique_leads,
                        )
                        raise OpenAIQuotaExceededError(
                            "OpenAI API quota is exhausted. Add billing or increase quota, then rerun the workflow."
                        ) from exc
                    failure = _make_failure(
                        stage="validation",
                        reason_code="validation_error",
                        detail=f"Validation failed with an exception: {exc}",
                        lead=lead,
                        direct_job_url=str(candidate.direct_job_url),
                        candidate=candidate,
                        attempt_number=attempt_number,
                        round_number=round_number,
                    )
                    _record_failure_with_followups(
                        settings,
                        diagnostics,
                        failure,
                        unique_leads_discovered=total_unique_leads,
                        lead=lead,
                        candidate=candidate,
                        audit_entry=_build_false_negative_audit_entry(lead, failure, candidate=candidate),
                        run_id=run_id,
                    )
                    continue

                if validation_failure is not None:
                    _record_failure_with_followups(
                        settings,
                        diagnostics,
                        validation_failure,
                        unique_leads_discovered=total_unique_leads,
                        lead=lead,
                        candidate=candidate,
                        near_miss=near_miss,
                        audit_entry=audit_entry,
                        run_id=run_id,
                    )
                    continue

                if validated is None:
                    continue

                if validated.salary_inference_kind == "salary_presumed_from_principal_ai_pm":
                    diagnostics.principal_ai_pm_salary_presumption_count += 1
                validated_key = _job_posting_dedupe_key(validated)
                post_validation_reacquisition_entry = reacquisition_entry or _validated_job_history_entry_for_url(
                    validated.resolved_job_url or validated.direct_job_url,
                    validated_job_history_index,
                )
                if post_validation_reacquisition_entry is not None and reacquisition_entry is None:
                    post_key = str(
                        post_validation_reacquisition_entry.get("canonical_job_key")
                        or post_validation_reacquisition_entry.get("job_key")
                        or ""
                    ).strip()
                    if len(reacquisition_attempted_keys) >= settings.reacquisition_attempt_cap and (
                        not post_key or post_key not in reacquisition_attempted_keys
                    ):
                        if post_key and post_key not in reacquisition_suppressed_keys:
                            reacquisition_suppressed_keys.add(post_key)
                            diagnostics.reacquired_jobs_suppressed_count += 1
                        _record_failure_live(
                            settings,
                            diagnostics,
                            _make_failure(
                                stage="filter",
                                reason_code="reacquisition_suppressed",
                                detail=(
                                    "Validated a previously reported job, but skipped coverage credit because the per-run "
                                    f"reacquisition cap ({settings.reacquisition_attempt_cap}) was reached."
                                ),
                                lead=lead,
                                direct_job_url=str(validated.direct_job_url),
                                candidate=validated,
                                attempt_number=attempt_number,
                                round_number=round_number,
                            ),
                            unique_leads_discovered=total_unique_leads,
                        )
                        continue
                    if post_key and post_key not in reacquisition_attempted_keys:
                        reacquisition_attempted_keys.add(post_key)
                        diagnostics.reacquisition_attempt_count += 1
                if post_validation_reacquisition_entry is not None:
                    metadata = _reacquisition_history_metadata(post_validation_reacquisition_entry)
                    validated = validated.model_copy(update={"is_reacquired": True, **metadata})
                accepted_before = len(jobs_by_url)
                if validated.is_reacquired:
                    reacquired_before = len(reacquired_jobs_by_url)
                    reacquired_jobs_by_url.setdefault(validated_key, validated)
                    if len(reacquired_jobs_by_url) > reacquired_before:
                        _update_query_family_metric_sets(
                            (query_family_metrics, attempt_query_family_metrics),
                            validated.source_query or lead.source_query or "",
                            count_execution=False,
                            validated_jobs=1,
                        )
                else:
                    jobs_by_url.setdefault(validated_key, validated)
                if len(jobs_by_url) > accepted_before:
                    _update_query_family_metric_sets(
                        (query_family_metrics, attempt_query_family_metrics),
                        validated.source_query or lead.source_query or "",
                        count_execution=False,
                        validated_jobs=1,
                    )
                if settings.company_discovery_enabled:
                    _upsert_company_discovery_from_validated_job(
                        company_discovery_entries,
                        validated,
                        run_id=run_id,
                    )
                    save_company_discovery_entries(settings.data_dir, company_discovery_entries)
                diagnostics.unique_leads_discovered = total_unique_leads
                _persist_search_diagnostics(settings, diagnostics)
                _persist_validated_jobs_checkpoint(
                    settings,
                    jobs_by_url,
                    reacquired_jobs_by_url=reacquired_jobs_by_url,
                    run_id=run_id,
                    diagnostics=diagnostics,
                )
                if status:
                    status.emit(
                        "search",
                        (
                            f"Reacquired still-open job {len(reacquired_jobs_by_url)}: {validated.company_name} | {validated.role_title}"
                            if validated.is_reacquired
                            else f"Accepted job {len(jobs_by_url)}/{stop_goal}: {validated.company_name} | {validated.role_title}"
                        ),
                        attempt_number=attempt_number,
                        round_number=round_number,
                        unique_leads_discovered=total_unique_leads,
                        qualifying_jobs=len(jobs_by_url),
                        jobs_kept_after_validation=len(jobs_by_url),
                    )

        diagnostics.unique_leads_discovered = total_unique_leads
        attempt_failures = _failure_counts_for_attempt(diagnostics, attempt_number)
        diagnostics.passes.append(
            SearchPassSummary(
                attempt_number=attempt_number,
                unique_leads_discovered=total_unique_leads - attempt_start_leads,
                qualifying_jobs=len(jobs_by_url),
                failure_counts=dict(attempt_failures),
                accepted_job_urls=list(jobs_by_url.keys()),
                query_count=sum(len(query_round) for query_round in query_rounds),
            )
        )
        _persist_search_diagnostics(settings, diagnostics)
        query_family_history = _merge_query_family_history(
            query_family_history,
            run_id=run_id,
            query_family_metrics=attempt_query_family_metrics,
        )

        if status:
            status.emit(
                "search",
                f"Pass {attempt_number} completed after resolving {resolved_leads_this_attempt} leads. Top rejection reasons: {_top_failure_summary(diagnostics, attempt_number)}",
                attempt_number=attempt_number,
                qualifying_jobs=len(jobs_by_url),
                unique_leads_discovered=total_unique_leads,
            )

        if _should_stop_after_dead_attempt(
            diagnostics,
            attempt_number=attempt_number,
            attempt_discovery_gain=total_unique_leads - attempt_start_leads,
            resolved_leads_this_attempt=resolved_leads_this_attempt,
        ):
            if status:
                status.emit(
                    "search",
                    (
                        f"Stopping adaptive search after pass {attempt_number} because it produced no new "
                        "leads and exhausted the timeout budget."
                    ),
                    attempt_number=attempt_number,
                    qualifying_jobs=len(jobs_by_url),
                    unique_leads_discovered=total_unique_leads,
                )
            break

        if len(jobs_by_url) >= stop_goal:
            break

        tuning = _derive_next_tuning(settings, diagnostics, attempt_number)

    jobs = list(jobs_by_url.values())
    jobs.sort(key=lambda item: (item.company_name.lower(), item.role_title.lower()))
    reacquired_jobs = list(reacquired_jobs_by_url.values())
    reacquired_jobs.sort(key=lambda item: (item.company_name.lower(), item.role_title.lower()))

    if status:
        if len(jobs) >= stop_goal:
            status.emit(
                "search",
                f"Search target reached with {len(jobs)} qualifying jobs.",
                unique_leads_discovered=total_unique_leads,
                qualifying_jobs=len(jobs),
            )
        else:
            status.emit(
                "search",
                f"Search finished with {len(jobs)} qualifying jobs after exhausting the configured adaptive passes.",
                unique_leads_discovered=total_unique_leads,
                qualifying_jobs=len(jobs),
                top_rejection_reasons=_top_failure_summary(diagnostics, diagnostics.passes[-1].attempt_number if diagnostics.passes else 1),
            )
    _persist_search_diagnostics(settings, diagnostics)
    _persist_query_family_history(
        settings,
        run_id=run_id,
        query_family_metrics=query_family_metrics,
    )
    return jobs, reacquired_jobs, total_unique_leads, diagnostics
