import asyncio
from datetime import date
import json
from pathlib import Path

from job_agent.config import Settings
from job_agent.job_search import (
    SearchTuning,
    _annotate_and_filter_resolution_leads,
    _apply_salary_inference,
    _apply_company_novelty_quota,
    _build_candidate_job,
    _build_lead_from_search_result,
    _build_near_miss,
    _build_portfolio_company_scout_queries,
    _build_small_company_scout_queries,
    _build_watchlist_board_focus_queries,
    _build_local_search_engine_queries,
    _builtin_category_urls_for_query,
    _builtin_paginated_category_urls,
    _builtin_search_base_urls,
    _extract_builtin_remote_hint,
    _builtin_search_terms_for_query,
    _build_local_query_rounds,
    _build_local_targeted_attempt_queries,
    _build_query_rounds,
    _build_search_query_bank,
    _chunk_queries,
    _company_names_match,
    _company_hint_from_url,
    _collect_replay_seed_leads,
    _dedupe_round_leads,
    _deterministic_trim_local_leads,
    _ensure_lazy_ollama_prewarm,
    _extract_direct_job_url_from_source,
    _extract_experience_years_floor,
    _extract_linkedin_guest_search_leads,
    _extract_mojeek_search_results,
    _extract_followup_resolution_urls,
    _extract_posted_hint,
    _extract_startpage_search_results,
    _extract_role_company_from_title,
    _extract_yahoo_search_results,
    _failed_lead_history_skip_reason,
    _is_allowed_direct_job_url,
    _extract_salary_hint,
    _is_ai_related_product_manager,
    _is_ai_related_product_manager_text,
    _lead_is_ai_related_product_manager,
    _lead_priority,
    _load_seed_leads_from_file,
    _job_posting_dedupe_key,
    _is_duckduckgo_anomaly_page,
    _is_google_interstitial_page,
    _is_recent_enough,
    _persist_validated_jobs_checkpoint,
    _is_supported_discovery_source_url,
    _looks_like_careers_hub_url,
    _looks_like_generic_job_url,
    _matches_filters,
    _decode_search_result_url,
    _evaluate_merged_job,
    _is_weak_company_hint,
    _merge_candidate_with_snapshot,
    _maybe_force_round_lead_refinement_with_ollama,
    _maybe_force_seed_lead_refinement_with_ollama,
    _normalize_and_filter_discovery_leads,
    _normalize_company_key,
    _normalize_role_title_to_focus_queries,
    _normalize_direct_job_url,
    _precheck_lead_hints,
    _query_family_key,
    _query_timeout_seconds_for_query,
    _query_timeout_skip_reason,
    _repair_direct_job_url,
    _refine_local_leads_with_ollama,
    _resolve_greenhouse_board_job_url_from_lead,
    _resolve_lead_via_company_careers_pages,
    _salary_is_base_salary,
    _search_single_query_local,
    _seed_lead_from_failure,
    _select_watchlist_focus_companies,
    _select_focus_roles,
    _should_force_ollama_refinement_sample,
    _should_abort_dead_attempt_round,
    _should_refine_local_leads_with_ollama,
    _should_stop_after_dead_attempt,
    _should_accept_trusted_source_fallback_on_fetch_failure,
    _url_has_strong_expected_company_hint,
    _url_candidate_score,
)
from job_agent.job_pages import JobPageSnapshot
from job_agent.models import DirectJobResolution, JobLead, JobPosting, SearchDiagnostics, SearchFailure


def build_settings() -> Settings:
    root = Path("/tmp/job-agent-tests")
    return Settings(
        project_root=root,
        openai_api_key="test-key",
        linkedin_email=None,
        linkedin_password=None,
        linkedin_totp_secret=None,
        linkedin_li_at=None,
        linkedin_jsessionid=None,
        google_email=None,
        google_password=None,
        google_totp_secret=None,
        browser_executable_path=None,
        browser_channel=None,
        linkedin_profile_dir=root / ".secrets/profile",
        linkedin_storage_state=root / ".secrets/state.json",
        output_dir=root / "output",
        data_dir=root / "data",
        headless=True,
        timezone="America/Chicago",
        search_country="US",
        search_city="Chicago",
        search_region="Illinois",
        min_base_salary_usd=200000,
        posted_within_days=7,
        minimum_qualifying_jobs=5,
        target_job_count=10,
        max_adaptive_search_passes=3,
        max_search_rounds=8,
        search_round_query_limit=6,
        max_leads_per_query=8,
        max_leads_to_resolve_per_pass=40,
        per_query_timeout_seconds=45,
        per_lead_timeout_seconds=30,
        workflow_timeout_seconds=3600,
        max_linkedin_results_per_company=25,
        max_linkedin_pages_per_company=3,
        daily_run_hour=8,
        daily_run_minute=0,
        status_heartbeat_seconds=120,
        enable_progress_gui=False,
    )


def test_allowed_direct_job_url_accepts_ats_hosts() -> None:
    assert _is_allowed_direct_job_url("https://boards.greenhouse.io/acme/jobs/123")
    assert _is_allowed_direct_job_url("https://jobs.lever.co/acme/123")
    assert _is_allowed_direct_job_url("https://jobs.recruitee.com/acme/o/staff-ai-product-manager")
    assert _is_allowed_direct_job_url("https://careers.tellent.com/o/staff-ai-product-manager")
    assert _is_allowed_direct_job_url("https://www.comeet.com/jobs/acme/00.005")
    assert _is_allowed_direct_job_url("https://acme.jobscore.com/job/staff-ai-product-manager")
    assert _is_allowed_direct_job_url(
        "https://portal.dynamicsats.com/JobListing/Details/be9c621a-ba9d-41b6-bc7b-917d59117a03/eed50803-efca-f011-bbd3-6045bdeb7e04"
    )
    assert _is_allowed_direct_job_url("https://careers.acme.com/jobs/123")
    assert _is_allowed_direct_job_url("https://navan.com/careers/openings?gh_jid=7660273&gh_src=5f7fcffe1")


def test_allowed_direct_job_url_rejects_aggregators() -> None:
    assert not _is_allowed_direct_job_url("https://www.linkedin.com/jobs/view/123")
    assert not _is_allowed_direct_job_url("https://www.indeed.com/viewjob?jk=123")


def test_generic_job_url_detection_allows_dynamicsats_detail_pages() -> None:
    assert not _looks_like_generic_job_url(
        "https://portal.dynamicsats.com/JobListing/Details/be9c621a-ba9d-41b6-bc7b-917d59117a03/eed50803-efca-f011-bbd3-6045bdeb7e04"
    )
    assert not _is_allowed_direct_job_url("https://www.glassdoor.com/Job/acme-ai-product-manager-jobs-SRCH_KO0,23.htm")
    assert not _is_allowed_direct_job_url("https://weloveproduct.co/jobs/principal-product-manager-excella-p8v7gy")
    assert not _is_allowed_direct_job_url("https://coinbase.getro.com/companies/spindl/jobs/71442891-senior-product-manager-trading-growth")
    assert not _is_allowed_direct_job_url("https://www.remoterocketship.com/company/humana/jobs/lead-product-manager-ivr-digital-iva-conversational-ai-united-states-remote/")


def test_normalize_direct_job_url_strips_apply_suffixes() -> None:
    assert _normalize_direct_job_url("https://jobs.lever.co/acme/123/apply?lever-source=feed") == "https://jobs.lever.co/acme/123"
    assert _normalize_direct_job_url("https://jobs.ashbyhq.com/acme/123/application?jbid=abc") == "https://jobs.ashbyhq.com/acme/123"
    assert _normalize_direct_job_url("https://jobs.ashbyhq.com/january/837101a2-6bc5-44e8-8f93-110638dcaca3?utm_source=5zgqMql0dg") == (
        "https://jobs.ashbyhq.com/january/837101a2-6bc5-44e8-8f93-110638dcaca3"
    )
    assert _normalize_direct_job_url("https://jobs.ashbyhq.com/clickup/292d27c5-956c-4291-a209-2420076e4bcb/2657e8034us") == "https://jobs.ashbyhq.com/clickup/292d27c5-956c-4291-a209-2420076e4bcb"
    assert (
        _normalize_direct_job_url("https://navan.com/careers/openings?gh_jid=7660273&gh_src=5f7fcffe1")
        == "https://navan.com/careers/openings?gh_jid=7660273"
    )
    assert (
        _normalize_direct_job_url(
            "https://www.dropbox.jobs/en/jobs/7729233/staff-product-manager-ai-organization-workflows/?gh_src=c393b7f81us"
        )
        == "https://www.dropbox.jobs/en/jobs/7729233/staff-product-manager-ai-organization-workflows"
    )
    assert (
        _normalize_direct_job_url(
            "https://ad.doubleclick.net/ddm/clk/606334822;414068816;h?https%3A%2F%2Fcareers.cargill.com%2Fen%2Fjob%2Fatlanta%2Fadvisor-product-manager-ai-and-data-science%2F47859%2F88787736608%3Futm_source%3Dbuiltin.com"
        )
        == "https://careers.cargill.com/en/job/atlanta/advisor-product-manager-ai-and-data-science/47859/88787736608"
    )


def test_decode_search_result_url_handles_bing_redirects() -> None:
    redirected = (
        "https://www.bing.com/ck/a?!&&p=abc&u="
        "a1aHR0cHM6Ly9qb2JzLmxldmVyLmNvL2xldmVsYWkvMjc1YjMyNzUtMTNhZi00NWQ0LWFjNGEtNTAxMDI4ZjQ0MmEx"
        "&ntb=1"
    )
    assert _decode_search_result_url(redirected) == "https://jobs.lever.co/levelai/275b3275-13af-45d4-ac4a-501028f442a1"


def test_decode_search_result_url_handles_google_redirects() -> None:
    redirected = "https://www.google.com/url?q=https%3A%2F%2Fjobs.lever.co%2Facme%2F123&sa=U&ved=2ah"
    assert _decode_search_result_url(redirected) == "https://jobs.lever.co/acme/123"


def test_decode_search_result_url_handles_yahoo_redirects() -> None:
    redirected = (
        "https://r.search.yahoo.com/_ylt=test/RV=2/RE=1775570248/RO=10/"
        "RU=https%3a%2f%2fwww.indeed.com%2fviewjob%3fjk%3d123456/RK=2/RS=test-"
    )
    assert _decode_search_result_url(redirected) == "https://www.indeed.com/viewjob?jk=123456"


def test_extract_role_company_from_breadcrumb_style_ats_title() -> None:
    company, role = _extract_role_company_from_title(
        "jobs.ashbyhq.com jobs.ashbyhq.com › hopper › 9a3d0809-326b-4ca5-ae60 Principal Product Manager - AI Travel (100% Remote - USA)",
        "https://jobs.ashbyhq.com/hopper/9a3d0809-326b-4ca5-ae60-bae9a835234c",
    )
    assert company == "Hopper"
    assert role == "Principal Product Manager - AI Travel (100% Remote - USA)"


def test_company_hint_from_workday_url_uses_company_slug_or_host() -> None:
    assert (
        _company_hint_from_url(
            "https://iqvia.wd1.myworkdayjobs.com/en-US/IQVIA/job/Senior-Product-Manager---Agentic-AI_R1531679"
        )
        == "Iqvia"
    )
    assert (
        _company_hint_from_url(
            "https://autodesk.wd1.myworkdayjobs.com/en-US/ext/job/Product-Manager--Agentic-AI_25WD94166-1"
        )
        == "Autodesk"
    )


def test_company_hint_from_company_hosted_jobs_subdomain_uses_host_company_name() -> None:
    assert _company_hint_from_url("https://jobs.dominos.com/us/jobs/supply-chain") == "Dominos"
    assert _company_hint_from_url("https://careers.caterpillar.com/en/jobs/123") == "Caterpillar"


def test_strong_expected_company_hint_rejects_company_hosted_mismatch() -> None:
    assert not _url_has_strong_expected_company_hint(
        "https://jobs.dominos.com/us/jobs/supply-chain",
        "Domino Data Lab",
    )
    assert _url_has_strong_expected_company_hint(
        "https://careers.caterpillar.com/en/jobs/12345",
        "Caterpillar",
    )


def test_weak_company_hints_do_not_trigger_mismatch_rejection(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 24))
    job = JobPosting(
        company_name="Headspace",
        role_title="Principal Product Manager, LLM Innovation",
        direct_job_url="https://job-boards.greenhouse.io/hs/jobs/7580489",
        resolved_job_url="https://job-boards.greenhouse.io/hs/jobs/7580489",
        ats_platform="Greenhouse",
        location_text="Remote - New York City, NY",
        is_fully_remote=True,
        posted_date_text="2026-03-20",
        posted_date_iso="2026-03-20",
        base_salary_min_usd=205200,
        base_salary_max_usd=287500,
        salary_text="$205,200-$287,500",
        evidence_notes="Remote AI PM role.",
        validation_evidence=["LLM Innovation"],
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="Greenhouse",
        status_code=200,
        page_title="Job Application for Principal Product Manager, LLM Innovation at Headspace",
        company_name="Headspace",
        role_title=job.role_title,
        location_text=job.location_text,
        is_fully_remote=True,
        posted_date_iso="2026-03-20",
        posted_date_text="2026-03-20",
        base_salary_min_usd=205200,
        base_salary_max_usd=287500,
        salary_text="$205,200-$287,500",
        text_excerpt="Principal Product Manager, LLM Innovation remote role.",
    )
    assert _is_weak_company_hint("Hs") is True
    reason, _detail = _evaluate_merged_job(job, snapshot, settings, expected_company_name="Hs")
    assert reason is None


def test_generic_job_url_detection_catches_board_indexes() -> None:
    assert _looks_like_generic_job_url("https://boards.greenhouse.io/embed/job_board?for=array")
    assert _looks_like_generic_job_url("https://careers.cisco.com/global/en")
    assert _looks_like_generic_job_url("https://careers.mastercard.com/us/en/apply")
    assert _looks_like_generic_job_url("https://jobs.dominos.com/us/jobs/stores")
    assert _looks_like_generic_job_url("https://webflow.com/made-in-webflow/careers")
    assert _looks_like_generic_job_url("https://www.coreweave.com/careers/eu")
    assert _looks_like_generic_job_url("https://www.coreweave.com/careers/notice-on-recruitment-fraud")
    assert not _looks_like_generic_job_url("https://boards.greenhouse.io/acme/jobs/123")
    assert not _looks_like_generic_job_url("https://jobs.lever.co/acme/12345678-1111-2222-3333-123456789abc")
    assert not _looks_like_generic_job_url("https://jobs.ashbyhq.com/acme/12345678-1111-2222-3333-123456789abc")
    assert not _looks_like_generic_job_url("https://jobs.smartrecruiters.com/Acme/744000123456789-principal-product-manager-ai")
    assert _looks_like_generic_job_url("https://job-boards.greenhouse.io/acme?error=true")


def test_build_near_miss_accepts_strong_close_salary_miss() -> None:
    settings = build_settings()
    lead = JobLead(
        company_name="Small AI Co",
        role_title="Staff Product Manager, AI Platform",
        source_url="https://jobs.ashbyhq.com/smallai/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.ashbyhq.com/smallai/123",
        is_remote_hint=True,
        posted_date_hint="2026-03-24",
        salary_text_hint="$185,000 - $195,000",
        evidence_notes="Direct ATS role at a smaller AI company.",
        source_quality_score_hint=6,
    )
    job = JobPosting(
        company_name="Small AI Co",
        role_title="Staff Product Manager, AI Platform",
        direct_job_url="https://jobs.ashbyhq.com/smallai/123",
        resolved_job_url="https://jobs.ashbyhq.com/smallai/123",
        ats_platform="Ashby",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-24",
        posted_date_iso="2026-03-24",
        base_salary_min_usd=185000,
        base_salary_max_usd=195000,
        salary_text="$185,000 - $195,000",
        evidence_notes="Strong AI PM scope on a direct ATS page.",
        validation_evidence=["Own the AI platform roadmap."],
        source_quality_score=6,
    )
    failure = SearchFailure(
        stage="validation",
        reason_code="salary_below_min",
        detail="Salary was below the configured minimum.",
        company_name=job.company_name,
        role_title=job.role_title,
        source_url=lead.source_url,
        direct_job_url=job.direct_job_url,
        posted_date_text=job.posted_date_text,
        salary_text=job.salary_text,
        is_remote=True,
        source_quality_score=6,
    )

    near_miss = _build_near_miss(lead, job, failure, settings)

    assert near_miss is not None
    assert near_miss.reason_code == "salary_below_min"
    assert near_miss.company_name == "Small AI Co"


def test_build_near_miss_rejects_low_signal_strategy_roles() -> None:
    settings = build_settings()
    lead = JobLead(
        company_name="Strategy AI",
        role_title="AI Strategy Product Manager",
        source_url="https://jobs.ashbyhq.com/strategyai/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.ashbyhq.com/strategyai/123",
        is_remote_hint=True,
        posted_date_hint="2026-03-24",
        evidence_notes="Strategy-heavy title on a direct ATS page.",
        source_quality_score_hint=6,
    )
    job = JobPosting(
        company_name="Strategy AI",
        role_title="AI Strategy Product Manager",
        direct_job_url="https://jobs.ashbyhq.com/strategyai/123",
        resolved_job_url="https://jobs.ashbyhq.com/strategyai/123",
        ats_platform="Ashby",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-24",
        posted_date_iso="2026-03-24",
        base_salary_min_usd=195000,
        base_salary_max_usd=205000,
        salary_text="$195,000 - $205,000",
        evidence_notes="Strategy-heavy role.",
        source_quality_score=6,
    )
    failure = SearchFailure(
        stage="validation",
        reason_code="remote_unclear",
        detail="Remote evidence was ambiguous.",
        company_name=job.company_name,
        role_title=job.role_title,
        source_url=lead.source_url,
        direct_job_url=job.direct_job_url,
        posted_date_text=job.posted_date_text,
        salary_text=job.salary_text,
        is_remote=None,
        source_quality_score=6,
    )

    assert _build_near_miss(lead, job, failure, settings) is None


def test_recent_text_parser_rejects_multi_week_postings() -> None:
    assert _is_recent_enough(None, "1 week ago", 7)
    assert _is_recent_enough(None, "Reposted 4 Days Ago", 7)
    assert not _is_recent_enough(None, "3 weeks ago", 7)
    assert not _is_recent_enough(None, "Posted 25 Days Ago", 7)


def test_recent_text_parser_accepts_absolute_iso_dates(monkeypatch) -> None:
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 25))
    assert _is_recent_enough(None, "2026-03-22", 14, timezone_name="America/Chicago")
    assert _is_recent_enough(None, "Mar 14, 2026", 14, timezone_name="America/Chicago")


def test_recent_iso_parser_uses_configured_timezone_for_boundary(monkeypatch) -> None:
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 24))
    assert _is_recent_enough("2026-03-10", "", 14, timezone_name="America/Chicago")


def test_build_local_query_rounds_include_startup_biased_queries() -> None:
    settings = build_settings()
    rounds = _build_local_query_rounds(settings, tuning=SearchTuning(attempt_number=1))
    flattened = [query for group in rounds for query in group]
    assert any("startup" in query.lower() or "careers" in query.lower() for query in flattened)
    assert any("ai product manager" in query.lower() for query in flattened)


def test_load_seed_leads_from_file_filters_invalid_and_non_ai_entries() -> None:
    settings = build_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    seed_path = settings.data_dir / "seed-leads.json"
    seed_path.write_text(
        json.dumps(
            {
                "leads": [
                    {
                        "company_name": "Interface AI",
                        "role_title": "Staff Product Manager - Integrations Strategy",
                        "source_url": "https://job-boards.greenhouse.io/interfaceai/jobs/4661597006",
                        "source_type": "direct_ats",
                        "direct_job_url": "https://job-boards.greenhouse.io/interfaceai/jobs/4661597006",
                        "is_remote_hint": True,
                        "posted_date_hint": "2026-03-10",
                        "salary_text_hint": "$190,000 - $250,000",
                        "evidence_notes": "AI-native banking platform role.",
                    },
                    {
                        "company_name": "Noise",
                        "role_title": "Senior Product Manager - Growth",
                        "source_url": "https://job-boards.greenhouse.io/noise/jobs/123",
                        "source_type": "direct_ats",
                        "direct_job_url": "https://job-boards.greenhouse.io/noise/jobs/123",
                        "evidence_notes": "Generic growth PM role.",
                    },
                    {
                        "company_name": "Bad",
                        "role_title": "Senior Product Manager, AI",
                        "source_url": "https://www.linkedin.com/jobs/view/123",
                        "source_type": "linkedin",
                        "direct_job_url": "https://www.linkedin.com/jobs/view/123",
                        "evidence_notes": "Aggregator URL should be rejected.",
                    },
                ]
            }
        )
    )

    leads = _load_seed_leads_from_file(settings)
    assert len(leads) == 1
    assert leads[0].company_name == "Interface AI"


def test_collect_replay_seed_leads_prioritizes_curated_file_seeds(tmp_path: Path) -> None:
    settings = build_settings()
    settings.data_dir = tmp_path
    seed_path = settings.data_dir / "seed-leads.json"
    seed_path.write_text(
        json.dumps(
            {
                "leads": [
                    {
                        "company_name": "Yelp",
                        "role_title": "Principal Product Manager - Applied ML (Remote - United States)",
                        "source_url": "https://uscareers-yelp.icims.com/jobs/13442/principal-product-manager---applied-ml/job",
                        "source_type": "direct_ats",
                        "direct_job_url": "https://uscareers-yelp.icims.com/jobs/13442/principal-product-manager---applied-ml/job",
                        "location_hint": "Remote - United States",
                        "posted_date_hint": "2026-03-14",
                        "is_remote_hint": True,
                        "evidence_notes": "Curated direct ATS seed.",
                    }
                ]
            }
        )
    )
    (settings.data_dir / "run-20260325-000000.json").write_text(
        json.dumps(
            {
                "bundles": [
                    {
                        "job": {
                            "company_name": "ACompany",
                            "role_title": "Principal Product Manager, AI",
                            "direct_job_url": "https://jobs.lever.co/acompany/123",
                            "ats_platform": "jobs.lever.co",
                            "location_text": "Remote",
                            "is_fully_remote": True,
                            "posted_date_text": "2026-03-20",
                            "posted_date_iso": "2026-03-20",
                            "base_salary_min_usd": 200000,
                            "base_salary_max_usd": 250000,
                            "salary_text": "$200,000 - $250,000",
                            "evidence_notes": "Historical accepted role.",
                            "validation_evidence": [],
                        }
                    }
                ],
                "search_diagnostics": {"failures": []},
            }
        )
    )

    leads = _collect_replay_seed_leads(settings)
    assert leads[0].company_name == "Yelp"


def test_collect_replay_seed_leads_ignores_non_dict_run_artifacts(tmp_path: Path) -> None:
    settings = build_settings()
    settings.data_dir = tmp_path
    (settings.data_dir / "run-20260325-000000.json").write_text(json.dumps([]))

    leads = _collect_replay_seed_leads(settings)
    assert leads == []


def test_extract_posted_hint_parses_absolute_month_dates() -> None:
    assert _extract_posted_hint("Mar 14, 2026 · Senior PM role") == "2026-03-14"


def test_resolve_greenhouse_board_job_url_from_generic_board(monkeypatch) -> None:
    lead = JobLead(
        company_name="SmarterDx",
        role_title="Senior Product Manager, AI/ML Model Development",
        source_url="https://job-boards.greenhouse.io/smarterdx/jobs",
        source_type="other",
        evidence_notes="Search result listed the AI/ML model development role.",
    )

    async def fake_fetch(board_token: str):
        assert board_token == "smarterdx"
        return [
            {
                "title": "Senior Product Manager, AI/ML Model Development",
                "absolute_url": "https://job-boards.greenhouse.io/smarterdx/jobs/5004750007",
            },
            {
                "title": "Senior Product Manager, Data",
                "absolute_url": "https://job-boards.greenhouse.io/smarterdx/jobs/5035070007",
            },
        ]

    monkeypatch.setattr("job_agent.job_search._fetch_greenhouse_board_jobs", fake_fetch)
    resolved = asyncio.run(_resolve_greenhouse_board_job_url_from_lead(lead))
    assert resolved == "https://job-boards.greenhouse.io/smarterdx/jobs/5004750007"


def test_matches_filters_requires_remote_salary_and_recency() -> None:
    settings = build_settings()
    job = JobPosting(
        company_name="Acme AI",
        role_title="Senior Product Manager, AI",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="United States",
        is_fully_remote=True,
        posted_date_text="3 days ago",
        posted_date_iso=None,
        base_salary_min_usd=210000,
        base_salary_max_usd=260000,
        salary_text="$210,000 - $260,000",
        evidence_notes="Remote, salary listed, posted 3 days ago, direct Greenhouse URL.",
    )
    assert _matches_filters(job, settings)


def test_salary_is_base_salary_does_not_treat_remote_as_ote() -> None:
    job = JobPosting(
        company_name="Hopper",
        role_title="Principal Product Manager - AI Travel (100% Remote - USA)",
        direct_job_url="https://jobs.ashbyhq.com/hopper/123",
        ats_platform="Ashby",
        location_text="United States",
        is_fully_remote=True,
        posted_date_text="2026-03-20",
        posted_date_iso="2026-03-20",
        base_salary_min_usd=150000,
        base_salary_max_usd=350000,
        salary_text="$150,000 - $350,000",
        evidence_notes="Remote AI product manager role.",
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="Ashby",
        status_code=200,
        text_excerpt="Principal Product Manager - AI Travel (100% Remote - USA) salary $150,000 - $350,000.",
    )
    assert _salary_is_base_salary(job, snapshot) is True


def test_matches_filters_rejects_missing_salary() -> None:
    settings = build_settings()
    job = JobPosting(
        company_name="Acme AI",
        role_title="Senior Product Manager, AI",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="United States",
        is_fully_remote=True,
        posted_date_text="2 days ago",
        posted_date_iso=None,
        base_salary_min_usd=None,
        base_salary_max_usd=None,
        salary_text=None,
        evidence_notes="Remote and recent but salary not available.",
    )
    assert not _matches_filters(job, settings)


def test_matches_filters_accepts_inferred_salary_from_experience() -> None:
    settings = build_settings()
    job = JobPosting(
        company_name="Acme AI",
        role_title="Principal Product Manager, AI Platform",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote, United States",
        is_fully_remote=True,
        posted_date_text="2 days ago",
        posted_date_iso=None,
        base_salary_min_usd=None,
        base_salary_max_usd=None,
        salary_text="Inferred likely >= $200,000 base from 8+ years requirement.",
        salary_inferred=True,
        salary_inference_reason="Posting indicates at least 8 years of experience.",
        inferred_experience_years_min=8,
        evidence_notes="Remote and recent with clear seniority requirement.",
    )
    assert _matches_filters(job, settings)


def test_apply_salary_inference_rejects_non_us_role_without_explicit_compensation() -> None:
    settings = build_settings()
    job = JobPosting(
        company_name="HighLevel",
        role_title="Sr. Product Manager - AI Platform",
        direct_job_url="https://jobs.lever.co/gohighlevel/377b5e0d-c6f7-4635-bbb5-c6ac0971b351",
        ats_platform="Lever",
        location_text="India / Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-17",
        posted_date_iso="2026-03-17",
        base_salary_min_usd=None,
        base_salary_max_usd=None,
        salary_text=None,
        evidence_notes="AI platform role requiring 8+ years of experience.",
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="Lever",
        status_code=200,
        location_text="India / Remote",
        text_excerpt="Senior Product Manager - AI Platform. India / Remote. Requires 8+ years of product management experience.",
    )
    inferred = _apply_salary_inference(job, snapshot, settings)
    assert inferred.salary_inferred is False


def test_apply_salary_inference_accepts_us_remote_principal_role_without_explicit_salary(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 24))
    job = JobPosting(
        company_name="Webflow",
        role_title="Principal Product Manager, AI",
        direct_job_url="https://boards.greenhouse.io/webflow/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote, United States",
        is_fully_remote=True,
        posted_date_text="2 days ago",
        posted_date_iso="2026-03-22",
        base_salary_min_usd=None,
        base_salary_max_usd=None,
        salary_text=None,
        evidence_notes="US remote principal AI product role.",
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="Greenhouse",
        status_code=200,
        location_text="Remote, United States",
        text_excerpt="Principal Product Manager, AI. Remote - United States. Build AI product strategy across the platform.",
    )
    inferred = _apply_salary_inference(job, snapshot, settings)
    assert inferred.salary_inferred is True
    reason_code, detail = _evaluate_merged_job(inferred, snapshot, settings, expected_company_name="Webflow")
    assert reason_code is None, detail


def test_apply_salary_inference_does_not_use_search_snippet_title_noise_for_senior_role() -> None:
    settings = build_settings()
    job = JobPosting(
        company_name="National Debt Relief, LLC",
        role_title="Senior Product Manager, Applied AI",
        direct_job_url="https://careers-nationaldebtrelief.icims.com/jobs/6244/senior-product-manager%2c-applied-ai/job",
        ats_platform="iCIMS",
        location_text="United States",
        is_fully_remote=True,
        posted_date_text="2026-03-13",
        posted_date_iso=None,
        base_salary_min_usd=None,
        base_salary_max_usd=None,
        salary_text=None,
        evidence_notes="Discovered while searching for group product manager applied AI roles.",
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="iCIMS",
        status_code=200,
        location_text="United States",
        text_excerpt="Senior Product Manager, Applied AI. United States remote.",
    )
    inferred = _apply_salary_inference(job, snapshot, settings)
    assert inferred.salary_inferred is False


def test_merge_candidate_with_snapshot_keeps_remote_hint_for_js_shell_pages() -> None:
    candidate = JobPosting(
        company_name="January",
        role_title="Staff Product Manager, AI Agents",
        direct_job_url="https://jobs.ashbyhq.com/january/123",
        resolved_job_url="https://jobs.ashbyhq.com/january/123",
        ats_platform="Ashby",
        location_text="Remote, United States",
        is_fully_remote=True,
        posted_date_text="2026-03-18",
        posted_date_iso="2026-03-18",
        base_salary_min_usd=None,
        base_salary_max_usd=None,
        salary_text=None,
        evidence_notes="Built In marked the role as remote.",
    )
    snapshot = JobPageSnapshot(
        requested_url=candidate.direct_job_url,
        resolved_url=candidate.direct_job_url,
        ats_platform="Ashby",
        status_code=200,
        company_name="January",
        role_title="Staff Product Manager, AI Agents",
        location_text="New York, New York, United States",
        is_fully_remote=False,
        text_excerpt="Staff Product Manager, AI Agents @ January You need to enable JavaScript to run this app.",
        evidence_snippets=["AI context: Staff Product Manager, AI Agents @ January You need to enable JavaScript to run this app."],
    )
    merged = _merge_candidate_with_snapshot(candidate, snapshot)
    assert merged.is_fully_remote is True


def test_merge_candidate_with_generic_snapshot_title_preserves_candidate_role_for_ai_inference(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 24))
    candidate = JobPosting(
        company_name="Yelp",
        role_title="Principal Product Manager - Applied ML (Remote - United States)",
        direct_job_url="https://uscareers-yelp.icims.com/jobs/13442/principal-product-manager---applied-ml/job",
        resolved_job_url="https://uscareers-yelp.icims.com/jobs/13442/principal-product-manager---applied-ml/job",
        ats_platform="iCIMS",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-20",
        posted_date_iso="2026-03-20",
        base_salary_min_usd=None,
        base_salary_max_usd=None,
        salary_text=None,
        evidence_notes="Applied ML principal PM role at Yelp.",
    )
    snapshot = JobPageSnapshot(
        requested_url=candidate.direct_job_url,
        resolved_url=candidate.direct_job_url,
        ats_platform="iCIMS",
        status_code=200,
        company_name="Yelp",
        role_title="Careers",
        page_title="Careers at Yelp | Yelp Jobs",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-20",
        posted_date_iso="2026-03-20",
        text_excerpt="Careers at Yelp.",
        evidence_snippets=[],
    )
    merged = _apply_salary_inference(_merge_candidate_with_snapshot(candidate, snapshot), snapshot, settings)
    assert merged.role_title == candidate.role_title
    assert merged.salary_inferred is True
    reason_code, detail = _evaluate_merged_job(merged, snapshot, settings, expected_company_name="Yelp")
    assert reason_code is None, detail


def test_merge_candidate_with_misaligned_snapshot_role_preserves_discovered_ai_pm_title() -> None:
    candidate = JobPosting(
        company_name="Acme AI",
        role_title="Senior Product Manager, AI & Data Products",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        resolved_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-24",
        posted_date_iso="2026-03-24",
        base_salary_min_usd=210000,
        base_salary_max_usd=250000,
        salary_text="$210,000-$250,000",
        evidence_notes="Discovered from a direct ATS role page.",
    )
    snapshot = JobPageSnapshot(
        requested_url=candidate.direct_job_url,
        resolved_url=candidate.direct_job_url,
        ats_platform="Greenhouse",
        status_code=200,
        company_name="Acme AI",
        role_title="Client Partner",
        page_title="Client Partner at Acme AI",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-24",
        posted_date_iso="2026-03-24",
        base_salary_min_usd=210000,
        base_salary_max_usd=250000,
        salary_text="$210,000-$250,000",
        text_excerpt="Client Partner at Acme AI.",
        evidence_snippets=[],
    )

    merged = _merge_candidate_with_snapshot(candidate, snapshot)

    assert merged.role_title == candidate.role_title


def test_evaluate_merged_job_marks_specific_wrong_role_page_as_resolution_missing(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 24))
    job = JobPosting(
        company_name="Acme AI",
        role_title="Senior Product Manager, AI & Data Products",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        resolved_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-24",
        posted_date_iso="2026-03-24",
        base_salary_min_usd=210000,
        base_salary_max_usd=250000,
        salary_text="$210,000-$250,000",
        evidence_notes="AI product role discovered from the direct ATS page.",
        validation_evidence=[],
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="Greenhouse",
        status_code=200,
        company_name="Acme AI",
        role_title="Client Partner",
        page_title="Client Partner at Acme AI",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-24",
        posted_date_iso="2026-03-24",
        base_salary_min_usd=210000,
        base_salary_max_usd=250000,
        salary_text="$210,000-$250,000",
        text_excerpt="Client Partner at Acme AI.",
        evidence_snippets=[],
    )

    reason_code, detail = _evaluate_merged_job(
        job,
        snapshot,
        settings,
        expected_company_name="Acme AI",
        expected_role_title="Senior Product Manager, AI & Data Products",
    )

    assert reason_code == "resolution_missing"
    assert "did not line up with expected role" in detail


def test_evaluate_merged_job_rejects_geo_limited_remote_role(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 29))
    job = JobPosting(
        company_name="Linktree",
        role_title="Staff AI Product Manager",
        direct_job_url="https://jobs.gem.com/linktree/example",
        resolved_job_url="https://jobs.gem.com/linktree/example",
        ats_platform="Gem",
        location_text="Remote - California only",
        is_fully_remote=True,
        posted_date_text="2026-03-26",
        posted_date_iso="2026-03-26",
        base_salary_min_usd=220000,
        base_salary_max_usd=240000,
        salary_text="$220,000 - $240,000",
        evidence_notes="Remote restriction: California only. Staff AI Product Manager role.",
        validation_evidence=[],
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="Gem",
        status_code=200,
        company_name="Linktree",
        role_title="Linktree Careers",
        page_title="Linktree Careers",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-26",
        posted_date_iso="2026-03-26",
        text_excerpt="Candidates can choose fully remote or hybrid, but we are only considering candidates who reside in California.",
        evidence_snippets=[],
    )

    reason_code, detail = _evaluate_merged_job(
        job,
        snapshot,
        settings,
        expected_company_name="Linktree",
        expected_role_title="Staff AI Product Manager",
        allow_trusted_source_role_fallback=True,
    )

    assert reason_code == "not_remote"
    assert "geographically restricted" in detail


def test_evaluate_merged_job_rejects_blank_workday_page_with_specific_location_slug(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 29))
    job = JobPosting(
        company_name="Citi",
        role_title="Wealth - AI Product Manager - Senior Vice President",
        direct_job_url=(
            "https://citi.wd5.myworkdayjobs.com/2/job/New-York-New-York-United-States/"
            "Wealth---AI-Product-Manager---Senior-Vice-President_25923027"
        ),
        resolved_job_url=(
            "https://citi.wd5.myworkdayjobs.com/2/job/New-York-New-York-United-States/"
            "Wealth---AI-Product-Manager---Senior-Vice-President_25923027"
        ),
        ats_platform="Workday",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-20",
        posted_date_iso="2026-03-20",
        base_salary_min_usd=176720,
        base_salary_max_usd=265080,
        salary_text="$176,720.00 - $265,080.00",
        evidence_notes="Remote hint from discovery source.",
        validation_evidence=[],
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="Workday",
        status_code=200,
        company_name="Citi",
        role_title=None,
        page_title="",
        text_excerpt="",
        evidence_snippets=[],
    )

    reason_code, detail = _evaluate_merged_job(
        job,
        snapshot,
        settings,
        expected_company_name="Citi",
        expected_role_title="Wealth - AI Product Manager - Senior Vice President",
    )

    assert reason_code == "not_remote"
    assert "specific location" in detail


def test_merge_candidate_with_snapshot_keeps_host_specific_non_remote_signal(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 29))
    candidate = JobPosting(
        company_name="GSK",
        role_title="Senior Product Manager, GenAI Platform Products",
        direct_job_url=(
            "https://gsk.wd5.myworkdayjobs.com/GSKCareers/job/200-CambridgePark-Drive/"
            "Senior-Product-Manager--GenAI-Platform-Products_431265-1"
        ),
        resolved_job_url=(
            "https://gsk.wd5.myworkdayjobs.com/GSKCareers/job/200-CambridgePark-Drive/"
            "Senior-Product-Manager--GenAI-Platform-Products_431265-1"
        ),
        ats_platform="Workday",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-23",
        posted_date_iso="2026-03-23",
        base_salary_min_usd=147675,
        base_salary_max_usd=246125,
        salary_text="$147,675 to $246,125",
        evidence_notes="Remote hint from discovery source.",
    )
    snapshot = JobPageSnapshot(
        requested_url=candidate.direct_job_url,
        resolved_url=candidate.direct_job_url,
        ats_platform="Workday",
        status_code=200,
        company_name="GSK",
        role_title="Senior Product Manager, GenAI Platform Products",
        page_title="Senior Product Manager, GenAI Platform Products",
        location_text="Cambridge, MA",
        is_fully_remote=False,
        posted_date_text="2026-03-23",
        posted_date_iso="2026-03-23",
        text_excerpt="Workday application page for the Cambridge office.",
        evidence_snippets=[],
    )

    merged = _merge_candidate_with_snapshot(candidate, snapshot)

    assert merged.is_fully_remote is False


def test_merge_candidate_with_snapshot_keeps_high_quality_remote_candidate_when_snapshot_text_is_strongly_remote(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 29))
    candidate = JobPosting(
        company_name="Vercel",
        role_title="Product Manager - Agent Platform",
        direct_job_url="https://job-boards.greenhouse.io/vercel/jobs/5808590004",
        resolved_job_url="https://job-boards.greenhouse.io/vercel/jobs/5808590004",
        ats_platform="Greenhouse",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-28",
        posted_date_iso="2026-03-28",
        base_salary_min_usd=196000,
        base_salary_max_usd=294000,
        salary_text="$196,000-$294,000",
        evidence_notes="Built In marked the role as remote. Own the roadmap for AI agents and LLM platform experiences.",
        validation_evidence=[],
        source_quality_score=18,
    )
    snapshot = JobPageSnapshot(
        requested_url=candidate.direct_job_url,
        resolved_url=candidate.direct_job_url,
        ats_platform="Greenhouse",
        status_code=200,
        company_name="Vercel",
        role_title="Product Manager - Agent Platform",
        page_title="Product Manager - Agent Platform at Vercel",
        location_text="Remote - United States",
        is_fully_remote=False,
        posted_date_text="2026-03-28",
        posted_date_iso="2026-03-28",
        base_salary_min_usd=196000,
        base_salary_max_usd=294000,
        salary_text="$196,000-$294,000",
        text_excerpt="Product Manager - Agent Platform. Remote - United States. Work from anywhere across the U.S. Own the roadmap for AI agents and LLM platform experiences.",
        evidence_snippets=["Remote - United States", "Work from anywhere across the U.S.", "AI agents and LLM platform experiences."],
    )

    merged = _merge_candidate_with_snapshot(candidate, snapshot)

    assert merged.is_fully_remote is True
    reason_code, detail = _evaluate_merged_job(
        merged,
        snapshot,
        settings,
        expected_company_name="Vercel",
        expected_role_title="Product Manager - Agent Platform",
    )
    assert reason_code is None, detail


def test_evaluate_merged_job_rejects_icims_location_conflict_without_strong_remote_evidence(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 29))
    job = JobPosting(
        company_name="Yelp",
        role_title="Principal Product Manager - Applied ML",
        direct_job_url="https://uscareers-yelp.icims.com/jobs/13442/principal-product-manager---applied-ml/job",
        resolved_job_url="https://uscareers-yelp.icims.com/jobs/13442/principal-product-manager---applied-ml/job",
        ats_platform="iCIMS",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-25",
        posted_date_iso="2026-03-25",
        base_salary_min_usd=220000,
        base_salary_max_usd=260000,
        salary_text="$220,000 - $260,000",
        evidence_notes="Remote hint from discovery source.",
        validation_evidence=[],
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="iCIMS",
        status_code=200,
        company_name="Yelp",
        role_title="Principal Product Manager - Applied ML",
        page_title="Principal Product Manager - Applied ML",
        location_text="Chicago, IL",
        is_fully_remote=None,
        posted_date_text="2026-03-25",
        posted_date_iso="2026-03-25",
        text_excerpt="Join the Yelp Chicago team in our downtown office.",
        evidence_snippets=[],
    )

    reason_code, detail = _evaluate_merged_job(
        job,
        snapshot,
        settings,
        expected_company_name="Yelp",
        expected_role_title="Principal Product Manager - Applied ML",
    )

    assert reason_code == "not_remote"
    assert "host-specific remote confirmation" in detail.lower()


def test_evaluate_merged_job_allows_trusted_source_fallback_for_generic_brand_page(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 29))
    job = JobPosting(
        company_name="Acme AI",
        role_title="Principal Product Manager, AI Platform",
        direct_job_url="https://jobs.gem.com/acme/example",
        resolved_job_url="https://jobs.gem.com/acme/example",
        ats_platform="Gem",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-26",
        posted_date_iso="2026-03-26",
        base_salary_min_usd=220000,
        base_salary_max_usd=260000,
        salary_text="$220,000 - $260,000",
        evidence_notes="Built In source provided explicit salary and fully remote evidence.",
        validation_evidence=[],
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="Gem",
        status_code=200,
        company_name="Acme AI",
        role_title="Acme AI",
        page_title="Acme AI Careers",
        location_text=None,
        is_fully_remote=None,
        posted_date_text=None,
        posted_date_iso=None,
        text_excerpt="Apply to join Acme AI.",
        evidence_snippets=[],
    )

    reason_code, detail = _evaluate_merged_job(
        job,
        snapshot,
        settings,
        expected_company_name="Acme AI",
        expected_role_title="Principal Product Manager, AI Platform",
        allow_trusted_source_role_fallback=True,
    )

    assert reason_code is None, detail


def test_seed_lead_from_failure_parses_salary_text_into_numeric_hints() -> None:
    failure = SearchFailure(
        stage="validation",
        reason_code="missing_salary",
        detail="No salary range was available from the direct page.",
        company_name="Citi",
        role_title="Wealth - AI Product Manager - Senior Vice President",
        source_url="https://builtinnyc.com/job/example",
        direct_job_url="https://citi.wd5.myworkdayjobs.com/job/example",
        posted_date_text="2026-03-20",
        salary_text="$176,720.00 - $265,080.00",
        is_remote=True,
    )

    lead = _seed_lead_from_failure(failure)

    assert lead is not None
    assert lead.base_salary_min_usd_hint == 176720
    assert lead.base_salary_max_usd_hint == 265080


def test_build_candidate_job_hydrates_numeric_salary_from_hint() -> None:
    lead = JobLead(
        company_name="Citi",
        role_title="Wealth - AI Product Manager - Senior Vice President",
        source_url="https://builtinnyc.com/job/example",
        source_type="builtin",
        direct_job_url="https://citi.wd5.myworkdayjobs.com/job/example",
        location_hint="Remote",
        posted_date_hint="2026-03-20",
        is_remote_hint=True,
        salary_text_hint="$176,720.00 - $265,080.00",
        evidence_notes="Built In source carried explicit salary.",
    )

    candidate = _build_candidate_job(
        lead,
        DirectJobResolution(
            accepted=True,
            direct_job_url="https://citi.wd5.myworkdayjobs.com/job/example",
            ats_platform="Workday",
            evidence_notes="Resolved from Built In apply link.",
        ),
    )

    assert candidate.base_salary_min_usd == 176720
    assert candidate.base_salary_max_usd == 265080


def test_is_ai_related_product_manager_ignores_discovery_snippet_noise() -> None:
    job = JobPosting(
        company_name="Jobgether",
        role_title="Sr. Group Product Manager - US Derivatives",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2 days ago",
        posted_date_iso=None,
        base_salary_min_usd=220000,
        base_salary_max_usd=250000,
        salary_text="$220,000 - $250,000",
        evidence_notes="Discovery snippet mentioned AI and machine learning.",
        validation_evidence=[],
    )
    assert not _is_ai_related_product_manager(job)


def test_is_ai_related_product_manager_ignores_hiring_process_boilerplate() -> None:
    job = JobPosting(
        company_name="Jobgether",
        role_title="Sr. Group Product Manager - US Derivatives",
        direct_job_url="https://jobs.lever.co/jobgether/c2ad1e8a-aaeb-44f3-9e8b-32d1fad6d274",
        ats_platform="Lever",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-23",
        posted_date_iso="2026-03-23",
        base_salary_min_usd=220000,
        base_salary_max_usd=260000,
        salary_text="$220,000 - $260,000",
        evidence_notes="Validated from the direct page.",
        validation_evidence=[
            "AI context: We use an AI-powered matching process to ensure your application is reviewed quickly, objectively, and fairly against the role's core requirements.",
            "AI context: We may use artificial intelligence (AI) tools to support parts of the hiring process, such as reviewing applications, analyzing resumes, or assessing responses.",
        ],
    )
    assert not _is_ai_related_product_manager(job)


def test_is_ai_related_product_manager_accepts_generic_pm_title_with_ml_evidence() -> None:
    job = JobPosting(
        company_name="Acme",
        role_title="Senior Product Manager, Platform",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2 days ago",
        posted_date_iso="2026-03-22",
        base_salary_min_usd=210000,
        base_salary_max_usd=260000,
        salary_text="$210,000 - $260,000",
        evidence_notes="Role owns computer vision and model serving strategy for the ML platform.",
        validation_evidence=["Computer vision models and ML platform tooling are central to the role."],
    )
    assert _is_ai_related_product_manager(job)


def test_lead_ai_classifier_rejects_generic_search_snippet_noise() -> None:
    lead = JobLead(
        company_name="Alt",
        role_title="Senior Product Manager-Growth",
        source_url="https://www.linkedin.com/jobs/view/123456",
        source_type="linkedin",
        direct_job_url=None,
        evidence_notes="Search snippet mentioned AI tools and machine learning capabilities.",
    )
    assert not _lead_is_ai_related_product_manager(lead)


def test_lead_ai_classifier_allows_builtin_description_evidence_for_generic_title() -> None:
    lead = JobLead(
        company_name="Acme",
        role_title="Senior Product Manager",
        source_url="https://builtin.com/job/senior-product-manager/123456",
        source_type="builtin",
        direct_job_url=None,
        evidence_notes="Lead the roadmap for conversational AI assistants and LLM-powered workflows.",
    )
    assert _lead_is_ai_related_product_manager(lead)


def test_seed_lead_from_failure_skips_low_trust_replay_sources() -> None:
    failure = SearchFailure(
        stage="validation",
        reason_code="missing_salary",
        detail="Mirror page omitted salary details.",
        company_name="Acme",
        role_title="Senior Product Manager, AI",
        direct_job_url="https://jobs.lever.co/acme/123",
        source_url="https://www.mediabistro.com/jobs/acme/senior-product-manager-ai/",
        is_remote=True,
        attempt_number=1,
        round_number=1,
    )

    assert _seed_lead_from_failure(failure) is None


def test_is_ai_related_product_manager_accepts_direct_page_ai_context_even_if_title_is_generic() -> None:
    job = JobPosting(
        company_name="Zapier",
        role_title="Sr. Product Manager - Chat & Knowledge",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2 days ago",
        posted_date_iso=None,
        base_salary_min_usd=210000,
        base_salary_max_usd=240000,
        salary_text="$210,000 - $240,000",
        evidence_notes="Validated from the direct job page.",
        validation_evidence=["AI context: Build chatbot and conversational AI experiences powered by LLM workflows."],
    )
    assert _is_ai_related_product_manager(job)


def test_company_names_match_rejects_different_companies() -> None:
    assert _company_names_match("Coinbase", "Coinbase, Inc.")
    assert _company_names_match("GSK", "1925 GlaxoSmithKline LLC")
    assert not _company_names_match("Coinbase", "Spindl")


def test_extract_experience_years_floor_supports_range_and_plus_formats() -> None:
    assert _extract_experience_years_floor("Requires 7+ years of experience in product management.") == 7
    assert _extract_experience_years_floor("Minimum of 8 years experience with AI products.") == 8
    assert _extract_experience_years_floor("Experience: 6-10 years in product leadership.") == 6
    assert _extract_experience_years_floor("No explicit experience requirement listed.") is None


def test_ai_related_product_manager_text_supports_ai_ml_format() -> None:
    assert _is_ai_related_product_manager_text("Principal Product Manager - AI/ML")
    assert _is_ai_related_product_manager_text("Staff Product Manager, Gen-AI Platform")


def test_extract_salary_hint_ignores_experience_ranges() -> None:
    assert _extract_salary_hint("Requires 5-7 years of experience with AI products.") == (None, None, None)
    assert _extract_salary_hint("Compensation: $175,000 - $225,000 base salary.") == (175000, 225000, "$175,000 - $225,000")


def test_query_bank_includes_direct_and_aggregator_discovery_sources() -> None:
    settings = build_settings()
    query_bank = _build_search_query_bank(settings, SearchTuning(attempt_number=2))
    assert any("site:boards.greenhouse.io" in query for query in query_bank)
    assert any("site:linkedin.com/jobs/view" in query for query in query_bank)
    assert any("site:builtin.com/jobs" in query for query in query_bank)
    assert any("site:glassdoor.com/Job" in query for query in query_bank)
    assert len(query_bank) == len(set(query_bank))


def test_query_rounds_respect_configured_round_size() -> None:
    settings = build_settings()
    rounds = _build_query_rounds(settings)
    assert rounds
    assert all(1 <= len(query_round) <= settings.search_round_query_limit for query_round in rounds)
    assert len(rounds) <= settings.max_search_rounds


def test_chunk_queries_splits_queries_into_small_batches() -> None:
    assert _chunk_queries(["a", "b", "c", "d", "e"], 2) == [["a", "b"], ["c", "d"], ["e"]]


def test_local_query_rounds_use_simpler_builtin_friendly_terms() -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    rounds = _build_query_rounds(settings)
    flattened = [query for query_round in rounds for query in query_round]
    assert any("ai product manager" in query.lower() for query in flattened)
    assert any(query.startswith("site:") for query in flattened)
    assert all(query.count("site:") <= 1 for query in flattened)
    assert any("posted this week" in query.lower() for query in flattened)
    assert any("machine learning product manager" in query.lower() for query in flattened)


def test_local_query_rounds_prefer_focus_roles_before_company_drilldown() -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    rounds = _build_query_rounds(
        settings,
        SearchTuning(
            attempt_number=2,
            focus_companies=["Alt", "Coinbase"],
            focus_roles=["principal product manager AI", "staff product manager machine learning"],
        ),
    )
    flattened = [query for query_round in rounds for query in query_round]
    assert flattened[0] == "principal product manager AI"
    assert "staff product manager machine learning" in flattened[:4]
    focus_company_queries = [query for query in flattened[:12] if "Alt" in query or "Coinbase" in query]
    assert any("Alt" in query for query in focus_company_queries)
    assert any("Coinbase" in query for query in focus_company_queries)


def test_local_query_rounds_prune_cross_run_timeout_cooldown_families_before_execution() -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    history = {
        "structured_ats": {"consecutive_timeout_heavy_runs": 2},
        "startup_workatastartup": {"consecutive_timeout_heavy_runs": 2},
        "startup_generic": {"consecutive_timeout_heavy_runs": 2},
    }

    rounds = _build_query_rounds(
        settings,
        SearchTuning(attempt_number=1),
        query_family_history=history,
    )
    flattened = [query for query_round in rounds for query in query_round]

    assert flattened
    assert all(
        _query_family_key(query) not in {"structured_ats", "startup_workatastartup", "startup_generic"}
        for query in flattened
    )
    assert len(flattened) < settings.max_search_rounds * settings.search_round_query_limit


def test_builtin_search_terms_expand_brittle_queries() -> None:
    assert _builtin_search_terms_for_query("AI/ML product manager") == [
        "ai ml product manager",
        "machine learning product manager",
    ]
    assert _builtin_search_terms_for_query("AI agents product manager") == [
        "ai agents product manager",
        "agentic AI product manager",
    ]
    assert _builtin_search_terms_for_query("staff product manager machine learning") == [
        "staff product manager machine learning",
        "machine learning product manager",
    ]


def test_select_focus_roles_ignores_non_ai_failure_titles() -> None:
    settings = build_settings()
    diagnostics = SearchDiagnostics(
        minimum_qualifying_jobs=5,
        failures=[
            SearchFailure(
                stage="resolution",
                reason_code="resolution_missing",
                detail="Could not resolve.",
                company_name="Alt",
                role_title="Senior Product Manager-Growth",
                attempt_number=1,
                round_number=1,
            ),
            SearchFailure(
                stage="resolution",
                reason_code="resolution_missing",
                detail="Could not resolve.",
                company_name="Twilio",
                role_title="Principal, Product Manager - AI / LLM",
                attempt_number=1,
                round_number=1,
            ),
        ],
    )
    roles = _select_focus_roles(settings, diagnostics, 1)
    assert any("AI" in role or "LLM" in role for role in roles)
    assert all("growth" not in role.lower() for role in roles)


def test_builtin_category_urls_expand_ai_ml_topics() -> None:
    urls = _builtin_category_urls_for_query("principal product manager AI/ML")
    assert "https://builtin.com/jobs/remote/product/artificial-intelligence" in urls
    assert "https://builtin.com/jobs/remote/artificial-intelligence" in urls
    assert "https://builtin.com/jobs/remote/product/machine-learning" in urls
    assert any("builtinnyc.com/jobs/remote/product/artificial-intelligence" in url for url in urls)


def test_builtin_search_base_urls_include_all_supported_regions() -> None:
    urls = _builtin_search_base_urls("AI product manager")
    assert "https://builtin.com" in urls
    assert "https://www.builtinnyc.com" in urls
    assert "https://www.builtinsf.com" in urls
    assert "https://www.builtinseattle.com" in urls
    assert "https://www.builtinchicago.org" in urls
    assert "https://www.builtinla.com" in urls


def test_builtin_paginated_category_urls_expand_page_numbers() -> None:
    assert _builtin_paginated_category_urls("https://builtin.com/jobs/remote/product", 3) == [
        "https://builtin.com/jobs/remote/product",
        "https://builtin.com/jobs/remote/product?page=2",
        "https://builtin.com/jobs/remote/product?page=3",
    ]


def test_builtin_remote_hint_trusts_remote_listing_context_unless_explicitly_negative() -> None:
    assert _extract_builtin_remote_hint("United States", "Product Manager, AI", source_is_remote_listing=True) is True
    assert _extract_builtin_remote_hint("San Francisco", "Hybrid AI product role", source_is_remote_listing=True) is False


def test_local_search_engine_queries_cover_non_builtin_boards_and_direct_ats() -> None:
    queries = _build_local_search_engine_queries("AI product manager")
    board_domains = (
        "linkedin.com/jobs/view",
        "glassdoor.com/Job",
        "builtin.com/jobs",
        "wellfound.com/jobs",
        "workatastartup.com/jobs",
        "getro.com/companies",
        "ziprecruiter.com/jobs",
        "themuse.com/jobs",
        "monster.com/jobs",
        "startup.jobs",
        "jobgether.com",
        "welcometothejungle.com/en/companies",
        "joinhandshake.com/jobs",
        "ycombinator.com/companies",
        "dynamitejobs.com/company",
        "dailyremote.com/remote-job",
        "remote.io/remote-product-jobs",
        "flexhired.com/jobs",
        "remoteai.io/roles",
        "indeed.com/viewjob",
    )
    ats_domains = (
        "job-boards.greenhouse.io",
        "boards.greenhouse.io",
        "jobs.lever.co",
        "jobs.ashbyhq.com",
        "jobs.recruitee.com",
        "careers.tellent.com",
        "comeet.com/jobs",
        "jobscore.com",
        "myworkdayjobs.com",
        "jobs.smartrecruiters.com",
        "jobs.jobvite.com",
        "jobs.workable.com",
        "jobs.icims.com",
        "careers.bamboohr.com",
        "jobs.dayforcehcm.com",
        "recruiting.paylocity.com",
        "careers.adp.com",
        "careers.workday.com",
    )
    board_query_count = sum(1 for query in queries if any(f"site:{domain}" in query for domain in board_domains))
    ats_query_count = sum(1 for query in queries if any(f"site:{domain}" in query for domain in ats_domains))
    assert board_query_count >= 2
    assert ats_query_count >= 4
    assert ats_query_count >= board_query_count
    assert any('"$200,000"' in query for query in queries)
    assert queries[0].startswith("site:")
    assert any(f"site:{domain}" in queries[0] for domain in ats_domains)
    assert len(queries) <= 14
    assert any(query == "AI product manager remote" for query in queries)


def test_supported_discovery_sources_include_specific_job_board_pages() -> None:
    assert _is_supported_discovery_source_url("https://www.linkedin.com/jobs/view/123456789")
    assert _is_supported_discovery_source_url("https://www.glassdoor.com/Job/acme-ai-product-manager-job123.htm")
    assert _is_supported_discovery_source_url("https://wellfound.com/jobs/123456-product-manager-ai")
    assert _is_supported_discovery_source_url("https://www.workatastartup.com/jobs/76698")
    assert _is_supported_discovery_source_url("https://mercuryfund.getro.com/companies/span-2/jobs/62085282-senior-product-manager-ai")
    assert _is_supported_discovery_source_url("https://www.indeed.com/viewjob?jk=123456")
    assert _is_supported_discovery_source_url("https://dynamitejobs.com/company/rula/remote-job/sr-product-manager-ai-remote")
    assert _is_supported_discovery_source_url("https://dailyremote.com/remote-job/senior-product-manager-ai-1234")
    assert _is_supported_discovery_source_url("https://www.remote.io/remote-product-jobs/principal-product-manager-ai-agents-42964")
    assert _is_supported_discovery_source_url("https://flexhired.com/jobs/sr-product-manager-ai-remote-2689499")
    assert _is_supported_discovery_source_url("https://www.ycombinator.com/companies/dynamo-ai/jobs/tt5OVwf-product-manager-ai")
    assert _is_supported_discovery_source_url("https://remoteai.io/roles/AI-Product-Management/job-123")
    assert not _is_supported_discovery_source_url("https://example.com/blog/ai-product-manager-role")


def test_precheck_lead_hints_fast_rejects_stale_non_remote_and_low_salary() -> None:
    settings = build_settings()
    stale_lead = JobLead(
        company_name="Acme",
        role_title="Senior Product Manager, AI",
        source_url="https://workatastartup.com/jobs/123",
        source_type="other",
        posted_date_hint="2026-03-01",
        evidence_notes="Remote startup role.",
    )
    stale_failure = _precheck_lead_hints(stale_lead, settings, attempt_number=1, round_number=1)
    assert stale_failure is not None
    assert stale_failure.reason_code == "stale_posting"

    hybrid_lead = JobLead(
        company_name="Acme",
        role_title="Senior Product Manager, AI",
        source_url="https://workatastartup.com/jobs/123",
        source_type="other",
        is_remote_hint=False,
        evidence_notes="Hybrid in San Francisco office.",
    )
    hybrid_failure = _precheck_lead_hints(hybrid_lead, settings, attempt_number=1, round_number=1)
    assert hybrid_failure is not None
    assert hybrid_failure.reason_code == "not_remote"

    low_salary_lead = JobLead(
        company_name="Acme",
        role_title="Senior Product Manager, AI",
        source_url="https://workatastartup.com/jobs/123",
        source_type="other",
        salary_text_hint="$160,000 - $180,000",
        evidence_notes="Remote startup role with salary range.",
    )
    low_salary_failure = _precheck_lead_hints(low_salary_lead, settings, attempt_number=1, round_number=1)
    assert low_salary_failure is not None
    assert low_salary_failure.reason_code == "salary_below_min"

    location_specific_workday_lead = JobLead(
        company_name="Citi",
        role_title="Wealth - AI Product Manager - Senior Vice President",
        source_url="https://builtin.com/job/wealth-ai-product-manager/123",
        source_type="builtin",
        direct_job_url=(
            "https://citi.wd5.myworkdayjobs.com/2/job/New-York-New-York-United-States/"
            "Wealth---AI-Product-Manager---Senior-Vice-President_25923027"
        ),
        evidence_notes="Remote hint from discovery source.",
    )
    location_failure = _precheck_lead_hints(location_specific_workday_lead, settings, attempt_number=1, round_number=1)
    assert location_failure is not None
    assert location_failure.reason_code == "not_remote"


def test_precheck_lead_hints_rejects_mismatched_direct_job_url_company() -> None:
    settings = build_settings()
    lead = JobLead(
        company_name="Vouch Insurance",
        role_title="Senior Product Manager, Platform (AI)",
        source_url="https://builtin.com/job/senior-product-manager-platform-ai/123",
        source_type="builtin",
        direct_job_url="https://jobs.lever.co/finch/abc123",
        is_remote_hint=True,
        posted_date_hint="2026-03-24",
        salary_text_hint="$210,000 - $240,000",
        evidence_notes="Built In role with a mismatched apply URL.",
    )

    failure = _precheck_lead_hints(lead, settings, attempt_number=1, round_number=1)

    assert failure is not None
    assert failure.reason_code == "company_mismatch"
    assert "finch" in failure.detail.lower()


def test_precheck_lead_hints_rejects_company_hosted_direct_job_url_mismatch() -> None:
    settings = build_settings()
    lead = JobLead(
        company_name="Domino Data Lab",
        role_title="Principal Product Manager, AI Factory",
        source_url="https://www.workatastartup.com/jobs/123",
        source_type="company_site",
        direct_job_url="https://jobs.dominos.com/us/jobs/supply-chain",
        is_remote_hint=True,
        posted_date_hint="2026-03-29",
        salary_text_hint="$250,000",
        evidence_notes="Remote startup AI product role with salary disclosure.",
    )

    failure = _precheck_lead_hints(lead, settings, attempt_number=1, round_number=1)

    assert failure is not None
    assert failure.reason_code == "company_mismatch"
    assert "dominos" in failure.detail.lower()


def test_query_timeout_seconds_for_query_keeps_broad_queries_tighter_than_targeted_queries() -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"

    broad_timeout = _query_timeout_seconds_for_query(settings, "principal product manager AI")
    targeted_timeout = _query_timeout_seconds_for_query(
        settings,
        'site:workatastartup.com/jobs "principal product manager" "AI" remote "posted this week"',
    )

    assert targeted_timeout < broad_timeout
    assert broad_timeout <= 25
    assert targeted_timeout <= 22


def test_build_search_query_bank_defers_generic_discovery_domains_until_later_attempts() -> None:
    settings = build_settings()

    first_attempt_queries = _build_search_query_bank(settings, SearchTuning(attempt_number=1))
    second_attempt_queries = _build_search_query_bank(settings, SearchTuning(attempt_number=2))

    assert not any("site:linkedin.com/jobs/view" in query for query in first_attempt_queries)
    assert any("site:linkedin.com/jobs/view" in query for query in second_attempt_queries)


def test_build_local_targeted_attempt_queries_biases_first_attempt_to_ats_sites() -> None:
    settings = build_settings()

    first_attempt_queries = _build_local_targeted_attempt_queries(settings, SearchTuning(attempt_number=1))
    second_attempt_queries = _build_local_targeted_attempt_queries(settings, SearchTuning(attempt_number=2))

    assert first_attempt_queries
    assert all("site:" in query for query in first_attempt_queries)
    assert any("site:" not in query for query in second_attempt_queries)


def test_query_timeout_skip_reason_repeats_and_broad_circuit_breaker() -> None:
    diagnostics = SearchDiagnostics(
        minimum_qualifying_jobs=5,
        failures=[
            SearchFailure(
                stage="discovery",
                reason_code="query_timeout",
                detail="timed out",
                source_query="AI product manager remote",
                attempt_number=1,
                round_number=1,
            ),
            *[
                SearchFailure(
                    stage="discovery",
                    reason_code="query_timeout",
                    detail="timed out",
                    source_query=f"principal product manager AI remote {index}",
                    attempt_number=1,
                    round_number=1,
                )
                for index in range(1, 7)
            ],
        ],
    )

    repeated_reason = _query_timeout_skip_reason(
        diagnostics,
        "AI product manager remote",
        attempt_number=1,
    )
    circuit_breaker_reason = _query_timeout_skip_reason(
        diagnostics,
        "staff product manager AI",
        attempt_number=1,
    )
    targeted_reason = _query_timeout_skip_reason(
        diagnostics,
        'site:jobs.lever.co "staff product manager" "AI" remote',
        attempt_number=1,
    )

    assert repeated_reason is not None
    assert "already timed out" in repeated_reason.lower()
    assert circuit_breaker_reason is not None
    assert "circuit breaker" in circuit_breaker_reason.lower()
    assert targeted_reason is None


def test_query_timeout_skip_reason_opens_startup_board_family_circuit_breaker() -> None:
    diagnostics = SearchDiagnostics(
        minimum_qualifying_jobs=5,
        failures=[
            SearchFailure(
                stage="discovery",
                reason_code="query_timeout",
                detail="timed out",
                source_query='site:workatastartup.com/jobs "product manager" "AI" remote startup',
                attempt_number=1,
                round_number=1,
            ),
            SearchFailure(
                stage="discovery",
                reason_code="query_timeout",
                detail="timed out",
                source_query='site:workatastartup.com/jobs "senior product manager" "AI" remote startup',
                attempt_number=1,
                round_number=1,
            ),
        ],
    )

    skip_reason = _query_timeout_skip_reason(
        diagnostics,
        'site:workatastartup.com/jobs "staff product manager" "AI" remote startup',
        attempt_number=1,
    )

    assert skip_reason is not None
    assert "startup-board timeout circuit breaker" in skip_reason.lower()


def test_query_timeout_skip_reason_opens_company_focused_circuit_breaker() -> None:
    diagnostics = SearchDiagnostics(
        minimum_qualifying_jobs=5,
        failures=[
            SearchFailure(
                stage="discovery",
                reason_code="query_timeout",
                detail="timed out",
                source_query='"Chartahealth" "AI Product Manager" remote',
                attempt_number=1,
                round_number=1,
            ),
            SearchFailure(
                stage="discovery",
                reason_code="query_timeout",
                detail="timed out",
                source_query='"Chartahealth" careers "AI Product Manager" remote',
                attempt_number=1,
                round_number=1,
            ),
        ],
    )

    skip_reason = _query_timeout_skip_reason(
        diagnostics,
        '"Chartahealth" "AI Product Manager" remote "$200,000"',
        attempt_number=1,
    )

    assert skip_reason is not None
    assert "company-specific timeout circuit breaker" in skip_reason.lower()
    assert "chartahealth" in skip_reason.lower()


def test_query_timeout_skip_reason_skips_company_careers_variant_after_single_timeout() -> None:
    diagnostics = SearchDiagnostics(
        minimum_qualifying_jobs=5,
        failures=[
            SearchFailure(
                stage="discovery",
                reason_code="query_timeout",
                detail="timed out",
                source_query='"Vercel" "AI Product Manager" remote',
                attempt_number=1,
                round_number=1,
            ),
        ],
    )

    skip_reason = _query_timeout_skip_reason(
        diagnostics,
        '"Vercel" careers "AI Product Manager" remote',
        attempt_number=1,
    )

    assert skip_reason is not None
    assert "careers query variant" in skip_reason.lower()
    assert "vercel" in skip_reason.lower()


def test_query_timeout_skip_reason_suppresses_late_pass_getro_and_voice_ai_queries() -> None:
    diagnostics = SearchDiagnostics(minimum_qualifying_jobs=5)

    getro_reason = _query_timeout_skip_reason(
        diagnostics,
        'site:getro.com/companies "product manager" "AI agents" remote startup',
        attempt_number=3,
    )
    voice_reason = _query_timeout_skip_reason(
        diagnostics,
        '"product manager" "voice AI" remote "growth stage"',
        attempt_number=3,
    )

    assert getro_reason is not None
    assert "getro" in getro_reason.lower()
    assert voice_reason is not None
    assert "voice" in voice_reason.lower()


def test_query_timeout_skip_reason_respects_cross_run_family_cooldown() -> None:
    diagnostics = SearchDiagnostics(minimum_qualifying_jobs=5)
    history = {
        "broad_generic": {
            "consecutive_timeout_heavy_runs": 2,
            "consecutive_zero_yield_runs": 1,
        }
    }

    reason = _query_timeout_skip_reason(
        diagnostics,
        "AI product manager remote",
        attempt_number=1,
        query_family_history=history,
    )

    assert reason is not None
    assert "cooling down" in reason.lower()


def test_query_timeout_skip_reason_respects_structured_ats_family_cooldown() -> None:
    diagnostics = SearchDiagnostics(minimum_qualifying_jobs=5)
    history = {
        "structured_ats": {
            "consecutive_timeout_heavy_runs": 2,
            "consecutive_zero_yield_runs": 2,
        }
    }

    reason = _query_timeout_skip_reason(
        diagnostics,
        'site:jobs.lever.co "staff product manager" "AI" remote',
        attempt_number=1,
        query_family_history=history,
    )

    assert reason is not None
    assert "structured ats" in reason.lower()
    assert "cooling down" in reason.lower()


def test_should_refine_local_leads_with_ollama_uses_broad_borderline_queries() -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"

    assert _should_refine_local_leads_with_ollama(
        settings,
        query="principal product manager AI",
        candidate_pool_count=12,
        average_confidence=0.93,
        cleanup_signal_count=1,
        low_trust_source_count=1,
        trustworthy_direct_url_count=4,
    )
    assert not _should_refine_local_leads_with_ollama(
        settings,
        query='"senior product manager" "AI" remote "series a"',
        candidate_pool_count=12,
        average_confidence=0.97,
        cleanup_signal_count=0,
        low_trust_source_count=0,
        trustworthy_direct_url_count=5,
    )
    assert _should_refine_local_leads_with_ollama(
        settings,
        query='"senior product manager" "AI" remote',
        candidate_pool_count=4,
        average_confidence=0.965,
        cleanup_signal_count=1,
        low_trust_source_count=1,
        trustworthy_direct_url_count=2,
    )
    assert _should_refine_local_leads_with_ollama(
        settings,
        query='site:workatastartup.com/jobs "senior product manager" "AI" remote startup',
        candidate_pool_count=8,
        average_confidence=0.975,
        cleanup_signal_count=1,
        low_trust_source_count=1,
        trustworthy_direct_url_count=4,
    )


def test_should_force_ollama_refinement_sample_requires_actual_cleanup_signals() -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"

    assert not _should_force_ollama_refinement_sample(
        settings,
        sample_size=5,
        average_confidence=0.86,
        cleanup_signal_count=0,
        low_trust_source_count=0,
        trustworthy_direct_url_count=3,
    )


def test_should_accept_trusted_source_fallback_on_fetch_failure_for_strong_company_hosted_role(monkeypatch) -> None:
    settings = build_settings()
    settings.posted_within_days = 14
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 30))
    lead = JobLead(
        company_name="Caterpillar",
        role_title="Principal Digital GenAI Product Manager; In-Cab Experience",
        source_url="https://builtin.com/job/principal-digital-genai-product-manager-cab-experience/8233588",
        source_type="builtin",
        direct_job_url="https://careers.caterpillar.com/en/jobs/r0000341589/principal-digital-genai-product-manager-in-cab-experience",
        location_hint="Remote - United States",
        posted_date_hint="2026-03-19",
        is_remote_hint=True,
        salary_text_hint="$147,760.00 - $240,110.00",
        base_salary_min_usd_hint=147760,
        base_salary_max_usd_hint=240110,
        evidence_notes="Built In source disclosed fully remote and salary information for this principal GenAI PM role.",
        source_quality_score_hint=19,
    )
    candidate = JobPosting(
        company_name="Caterpillar",
        role_title="Principal Digital GenAI Product Manager; In-Cab Experience",
        direct_job_url=lead.direct_job_url,
        resolved_job_url=lead.direct_job_url,
        ats_platform="careers.caterpillar.com",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-19",
        posted_date_iso="2026-03-19",
        base_salary_min_usd=147760,
        base_salary_max_usd=240110,
        salary_text="$147,760.00 - $240,110.00",
        evidence_notes=lead.evidence_notes,
        validation_evidence=[],
    )

    assert _should_accept_trusted_source_fallback_on_fetch_failure(lead, candidate, settings)


def test_should_accept_trusted_source_fallback_on_fetch_failure_uses_candidate_evidence_when_lead_hints_are_sparse(
    monkeypatch,
) -> None:
    settings = build_settings()
    settings.posted_within_days = 14
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 30))
    lead = JobLead(
        company_name="Caterpillar",
        role_title="Principal Digital GenAI Product Manager; In-Cab Experience",
        source_url="https://builtin.com/job/principal-digital-genai-product-manager-cab-experience/8233588",
        source_type="company_site",
        direct_job_url="https://careers.caterpillar.com/en/jobs/r0000341589/principal-digital-genai-product-manager-in-cab-experience",
        evidence_notes="Built In source surfaced the role, but hint hydration was sparse.",
    )
    candidate = JobPosting(
        company_name="Caterpillar",
        role_title="Principal Digital GenAI Product Manager; In-Cab Experience",
        direct_job_url=lead.direct_job_url,
        resolved_job_url=lead.direct_job_url,
        ats_platform="careers.caterpillar.com",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-19",
        posted_date_iso="2026-03-19",
        base_salary_min_usd=147760,
        base_salary_max_usd=240110,
        salary_text="$147,760.00 - $240,110.00",
        evidence_notes="Built In source disclosed fully remote and salary information for this principal GenAI PM role.",
        validation_evidence=[],
        source_quality_score=19,
    )

    assert _should_accept_trusted_source_fallback_on_fetch_failure(lead, candidate, settings)


def test_should_accept_trusted_source_fallback_on_fetch_failure_accepts_medium_quality_company_hosted_role(
    monkeypatch,
) -> None:
    settings = build_settings()
    settings.posted_within_days = 14
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 30))
    lead = JobLead(
        company_name="Caterpillar",
        role_title="Principal Digital GenAI Product Manager; In-Cab Experience",
        source_url="https://builtin.com/job/principal-digital-genai-product-manager-cab-experience/8233588",
        source_type="company_site",
        direct_job_url="https://careers.caterpillar.com/en/jobs/r0000341589/principal-digital-genai-product-manager-in-cab-experience",
        location_hint="Remote - United States",
        posted_date_hint="2026-03-19",
        is_remote_hint=True,
        salary_text_hint="$147,760.00 - $240,110.00",
        base_salary_min_usd_hint=147760,
        base_salary_max_usd_hint=240110,
        evidence_notes="Built In source disclosed fully remote and salary information for this principal GenAI PM role.",
        source_quality_score_hint=9,
    )
    candidate = JobPosting(
        company_name="Caterpillar",
        role_title="Principal Digital GenAI Product Manager; In-Cab Experience",
        direct_job_url=lead.direct_job_url,
        resolved_job_url=lead.direct_job_url,
        ats_platform="careers.caterpillar.com",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-19",
        posted_date_iso="2026-03-19",
        base_salary_min_usd=147760,
        base_salary_max_usd=240110,
        salary_text="$147,760.00 - $240,110.00",
        evidence_notes=lead.evidence_notes,
        validation_evidence=[],
    )

    assert _should_accept_trusted_source_fallback_on_fetch_failure(lead, candidate, settings)


def test_should_accept_trusted_source_fallback_on_fetch_failure_accepts_low_quality_company_hosted_role_with_strong_evidence(
    monkeypatch,
) -> None:
    settings = build_settings()
    settings.posted_within_days = 14
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 30))
    lead = JobLead(
        company_name="Caterpillar",
        role_title="Principal GenAI Product Manager - Dealer and Customer Support",
        source_url="https://builtin.com/job/principal-genai-product-manager-dealer-and-customer-support/8796394",
        source_type="company_site",
        direct_job_url="https://careers.caterpillar.com/en/jobs/r0000350921/principal-genai-product-manager-dealer-and-customer-support",
        location_hint="Remote - United States",
        posted_date_hint="2026-03-19",
        is_remote_hint=True,
        salary_text_hint="$147,760.00 - $240,110.00",
        base_salary_min_usd_hint=147760,
        base_salary_max_usd_hint=240110,
        evidence_notes="Built In source disclosed fully remote and salary information for this principal GenAI PM role.",
        source_quality_score_hint=4,
    )
    candidate = JobPosting(
        company_name="Caterpillar",
        role_title="Principal GenAI Product Manager - Dealer and Customer Support",
        direct_job_url=lead.direct_job_url,
        resolved_job_url=lead.direct_job_url,
        ats_platform="careers.caterpillar.com",
        location_text="Remote - United States",
        is_fully_remote=True,
        posted_date_text="2026-03-19",
        posted_date_iso="2026-03-19",
        base_salary_min_usd=147760,
        base_salary_max_usd=240110,
        salary_text="$147,760.00 - $240,110.00",
        evidence_notes=lead.evidence_notes,
        validation_evidence=[],
    )

    assert _should_accept_trusted_source_fallback_on_fetch_failure(lead, candidate, settings)


def test_seed_lead_from_failure_only_replays_fixable_failures() -> None:
    replayable = SearchFailure(
        stage="validation",
        reason_code="missing_salary",
        detail="No salary",
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://builtin.com/job/tiny-ai",
        direct_job_url="https://jobs.ashbyhq.com/tinyai/123",
        posted_date_text="2026-03-28",
        is_remote=True,
    )
    terminal = SearchFailure(
        stage="validation",
        reason_code="stale_posting",
        detail="Too old",
        company_name="OldCo",
        role_title="Senior Product Manager, AI",
        source_url="https://builtin.com/job/oldco",
        direct_job_url="https://jobs.ashbyhq.com/oldco/123",
        posted_date_text="2026-03-01",
        is_remote=True,
    )

    assert _seed_lead_from_failure(replayable) is not None
    assert _seed_lead_from_failure(terminal) is None


def test_failed_lead_history_skip_reason_suppresses_repeat_failures() -> None:
    settings = build_settings()
    lead = JobLead(
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://builtin.com/job/tiny-ai",
        source_type="builtin",
        direct_job_url="https://jobs.ashbyhq.com/tinyai/123",
        is_remote_hint=None,
        posted_date_hint=None,
        evidence_notes="Repeated stale role without new override hints.",
    )
    history = {
        "url:ashby:tinyai:123": {
            "watch_count": 3,
            "recent_rejection_reasons": {"stale_posting": 3},
        }
    }

    skip_reason = _failed_lead_history_skip_reason(lead, settings, history)

    assert skip_reason is not None
    assert skip_reason[0] == "stale_posting"


def test_persist_validated_jobs_checkpoint_writes_current_jobs(tmp_path: Path) -> None:
    settings = build_settings()
    settings.data_dir = tmp_path
    diagnostics = SearchDiagnostics(minimum_qualifying_jobs=5, unique_leads_discovered=42)
    jobs_by_url = {
        "https://microsoft.ai/job/principal-product-manager": JobPosting(
            company_name="Microsoft AI",
            role_title="Principal Product Manager",
            direct_job_url="https://microsoft.ai/job/principal-product-manager",
            resolved_job_url="https://microsoft.ai/job/principal-product-manager",
            ats_platform="Microsoft AI",
            location_text="Remote - United States",
            is_fully_remote=True,
            posted_date_text="2026-03-29",
            posted_date_iso="2026-03-29",
            salary_text="$220,000 - $260,000",
            evidence_notes="Direct company role.",
        )
    }

    _persist_validated_jobs_checkpoint(settings, jobs_by_url, run_id="run-checkpoint", diagnostics=diagnostics)

    latest_payload = json.loads((tmp_path / "validated-jobs-checkpoint-latest.json").read_text(encoding="utf-8"))
    run_payload = json.loads((tmp_path / "validated-jobs-checkpoint-run-checkpoint.json").read_text(encoding="utf-8"))
    assert latest_payload["qualifying_job_count"] == 1
    assert latest_payload["unique_leads_discovered"] == 42
    assert latest_payload["jobs"][0]["company_name"] == "Microsoft AI"
    assert run_payload["run_id"] == "run-checkpoint"


def test_evaluate_merged_job_rejects_generic_company_home_or_benefits_page(monkeypatch) -> None:
    settings = build_settings()
    monkeypatch.setattr("job_agent.job_search._today_for_timezone", lambda timezone_name: date(2026, 3, 29))
    job = JobPosting(
        company_name="Applied Systems",
        role_title="Senior Product Manager, AI",
        direct_job_url="https://careers-appliedsystems.icims.com/jobs/intro",
        resolved_job_url="https://careers-appliedsystems.icims.com/jobs/intro",
        ats_platform="iCIMS",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-26",
        posted_date_iso="2026-03-26",
        base_salary_min_usd=210000,
        base_salary_max_usd=235000,
        salary_text="$210,000 - $235,000",
        evidence_notes="Discovery source had the full role and salary evidence.",
        validation_evidence=[],
    )
    snapshot = JobPageSnapshot(
        requested_url=job.direct_job_url,
        resolved_url=job.direct_job_url,
        ats_platform="iCIMS",
        status_code=200,
        company_name="Applied Systems",
        role_title=None,
        page_title="Careers | Applied Systems",
        location_text=None,
        is_fully_remote=None,
        posted_date_text=None,
        posted_date_iso=None,
        text_excerpt="Learn about our benefits and career opportunities at Applied Systems.",
        evidence_snippets=[],
    )

    reason_code, detail = _evaluate_merged_job(
        job,
        snapshot,
        settings,
        expected_company_name="Applied Systems",
        expected_role_title="Senior Product Manager, AI",
    )

    assert reason_code == "not_specific_job_page"
    assert "specific posting" in detail.lower()


def test_annotate_and_filter_resolution_leads_skips_repeat_stale_companies_without_override_hints() -> None:
    settings = build_settings()
    repeat_stale = JobLead(
        company_name="RepeatCo",
        role_title="Senior Product Manager, AI",
        source_url="https://www.linkedin.com/jobs/view/123",
        source_type="linkedin",
        is_remote_hint=True,
        evidence_notes="AI product manager role.",
    )
    fresh_override = JobLead(
        company_name="RepeatCo",
        role_title="Senior Product Manager, AI",
        source_url="https://builtin.com/job/repeatco-ai-pm/123",
        source_type="builtin",
        direct_job_url="https://jobs.lever.co/repeatco/abc123",
        posted_date_hint="today",
        is_remote_hint=True,
        salary_text_hint="$210,000 - $240,000",
        evidence_notes="Remote AI product manager role with published salary.",
    )
    watchlist = {
        "repeatco": {
            "company_name": "RepeatCo",
            "watch_count": 18,
            "recent_rejection_reasons": {"stale_posting": 8},
        }
    }

    weak_only = _annotate_and_filter_resolution_leads([repeat_stale], settings, watchlist)
    assert weak_only == []

    with_override = _annotate_and_filter_resolution_leads([fresh_override], settings, watchlist)
    assert len(with_override) == 1
    assert with_override[0].company_name == "RepeatCo"


def test_normalize_and_filter_discovery_leads_keeps_supported_job_board_pages_without_direct_urls() -> None:
    lead = JobLead(
        company_name="Acme",
        role_title="Senior Product Manager, AI",
        source_url="https://www.linkedin.com/jobs/view/123456789",
        source_type="linkedin",
        direct_job_url=None,
        evidence_notes="Remote AI product manager role posted 2 days ago.",
    )
    normalized = _normalize_and_filter_discovery_leads([lead], "AI product manager")
    assert len(normalized) == 1


def test_build_lead_from_search_result_uses_title_or_url_for_remote_hint() -> None:
    lead = _build_lead_from_search_result(
        "https://jobs.twilio.com/careers/job/1099549995199-senior-product-manager-enterprise-ai-remote-us",
        "Senior Product Manager - Enterprise AI | Twilio",
        "Apply today.",
        "twilio ai product manager",
    )
    assert lead is not None
    assert lead.is_remote_hint is True


def test_dedupe_round_leads_keeps_best_source_for_same_role() -> None:
    settings = build_settings()
    linkedin = JobLead(
        company_name="Hopper",
        role_title="Principal Product Manager - AI Travel",
        source_url="https://www.linkedin.com/jobs/view/1",
        source_type="linkedin",
        direct_job_url=None,
        is_remote_hint=True,
        evidence_notes="LinkedIn result.",
    )
    direct = JobLead(
        company_name="Hopper",
        role_title="Principal Product Manager - AI Travel",
        source_url="https://jobs.ashbyhq.com/hopper/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.ashbyhq.com/hopper/123",
        is_remote_hint=True,
        posted_date_hint="today",
        evidence_notes="Direct ATS result.",
    )

    deduped = _dedupe_round_leads([linkedin, direct], settings)
    assert deduped == [direct]


def test_extract_linkedin_guest_search_leads_parses_public_cards() -> None:
    html = """
    <div class="base-search-card">
      <a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/principal-product-manager-applied-ml-4376120174?position=1"></a>
      <div class="base-search-card__info">
        <h3 class="base-search-card__title">Principal Product Manager - Applied ML (Remote - United States)</h3>
        <h4 class="base-search-card__subtitle"><a href="https://www.linkedin.com/company/yelp-com">Yelp</a></h4>
        <div class="base-search-card__metadata">
          <span class="job-search-card__location">San Francisco, CA</span>
          <time class="job-search-card__listdate" datetime="2026-03-14">1 week ago</time>
        </div>
      </div>
    </div>
    """
    leads = _extract_linkedin_guest_search_leads(html, "applied AI product manager")
    assert len(leads) == 1
    lead = leads[0]
    assert lead.company_name == "Yelp"
    assert lead.role_title == "Principal Product Manager - Applied ML (Remote - United States)"
    assert lead.source_type == "linkedin"
    assert lead.source_url.startswith("https://www.linkedin.com/jobs/view/principal-product-manager-applied-ml-4376120174")
    assert lead.posted_date_hint == "2026-03-14"
    assert lead.is_remote_hint is True


def test_google_interstitial_detection_flags_enable_js_page() -> None:
    html = '<html><head><title>Google Search</title></head><body>Please click <a href="/httpservice/retry/enablejs?sei=abc">here</a> if you are not redirected.</body></html>'
    assert _is_google_interstitial_page(200, html)


def test_extract_startpage_search_results_parses_result_cards() -> None:
    html = """
    <div class="result">
      <a class="result-link" href="https://www.indeed.com/viewjob?jk=abc123">Senior Product Manager, AI</a>
      <p class="description">Remote role posted 2 days ago with salary $210k-$260k.</p>
    </div>
    """
    results = _extract_startpage_search_results(html, max_results=5)
    assert results == [("https://www.indeed.com/viewjob?jk=abc123", "Senior Product Manager, AI", "Remote role posted 2 days ago with salary $210k-$260k.")]


def test_extract_yahoo_search_results_parses_algo_cards() -> None:
    html = """
    <div class="algo">
      <div class="compTitle">
        <a href="https://r.search.yahoo.com/_ylt=test/RV=2/RE=1775570248/RO=10/RU=https%3a%2f%2fjobs.lever.co%2facme%2f123/RK=2/RS=test-">
          Senior Product Manager, AI
        </a>
      </div>
      <div class="compText"><p>Posted 3 days ago. Remote. $220k-$260k.</p></div>
    </div>
    """
    results = _extract_yahoo_search_results(html, max_results=5)
    assert results == [("https://jobs.lever.co/acme/123", "Senior Product Manager, AI", "Posted 3 days ago. Remote. $220k-$260k.")]


def test_extract_mojeek_search_results_parses_standard_results() -> None:
    html = """
    <div class="results-standard">
      <li class="r1">
        <h2><a class="title" href="https://dynamitejobs.com/company/rula/remote-job/sr-product-manager-ai-remote">Sr. Product Manager - AI</a></h2>
        <p class="s">Remote. $181k-$212.9k. Launch and scale AI products.</p>
      </li>
    </div>
    """
    results = _extract_mojeek_search_results(html, max_results=5)
    assert results == [("https://dynamitejobs.com/company/rula/remote-job/sr-product-manager-ai-remote", "Sr. Product Manager - AI", "Remote. $181k-$212.9k. Launch and scale AI products.")]


def test_build_lead_from_search_result_parses_dynamitejobs_company_and_role_from_url() -> None:
    lead = _build_lead_from_search_result(
        "https://dynamitejobs.com/company/figma/remote-job/product-manager-ai",
        "Dynamite Jobs dynamitejobs.com › remote-job › product-manager-ai Product Manager, AI, Remote Job, March 2026",
        "Updated March 12, 2026 Product Manager, AI Figma Full Time $164k - $294k per year Remote Job",
        "AI product manager",
    )
    assert lead is not None
    assert lead.company_name == "Figma"
    assert lead.role_title == "Product Manager Ai"


def test_looks_like_careers_hub_url_accepts_generic_company_careers_pages() -> None:
    assert _looks_like_careers_hub_url("https://careers.circle.com/us/en")
    assert _looks_like_careers_hub_url("https://www.versapay.com/careers")
    assert _looks_like_careers_hub_url("https://jobs.lever.co/versapay")
    assert not _looks_like_careers_hub_url("https://www.reddit.com/r/jobs")


def test_allowed_direct_job_url_keeps_specific_company_job_pages_but_not_generic_careers_hubs() -> None:
    assert _is_allowed_direct_job_url("https://navan.com/careers/openings?gh_jid=7660273&gh_src=5f7fcffe1")
    assert not _is_allowed_direct_job_url("https://webflow.com/made-in-webflow/careers")
    assert not _is_allowed_direct_job_url("https://www.coreweave.com/careers/eu")


def test_extract_followup_resolution_urls_keeps_careers_pages_and_generic_ats_boards() -> None:
    html = """
    <a href="/careers">Careers</a>
    <a href="https://jobs.lever.co/versapay">Open roles</a>
    <a href="https://jobs.lever.co/versapay/1305d3eb-5d36-4a7b-86d7-9c2ab53f83d4">Exact role</a>
    """
    followups = _extract_followup_resolution_urls("https://www.versapay.com/", html)
    assert "https://www.versapay.com/careers" in followups
    assert "https://jobs.lever.co/versapay" in followups
    assert "https://jobs.lever.co/versapay/1305d3eb-5d36-4a7b-86d7-9c2ab53f83d4" not in followups


def test_extract_followup_resolution_urls_sanitizes_escaped_malformed_urls() -> None:
    html = '<a href="https://www.hopper.com/careers/&quot;&gt;Click">Broken</a>'
    assert _extract_followup_resolution_urls("https://www.hopper.com/", html) == ["https://www.hopper.com/careers"]


def test_url_candidate_score_prefers_product_manager_titles() -> None:
    lead = JobLead(
        company_name="Airwallex",
        role_title="Staff Product Manager, AI Growth",
        source_url="https://www.linkedin.com/jobs/view/123",
        source_type="linkedin",
        evidence_notes="AI growth PM role.",
    )
    product_score = _url_candidate_score(
        "https://careers.airwallex.com/job/abc/staff-product-manager-ai-growth/",
        "Staff Product Manager, AI Growth",
        lead,
    )
    non_pm_score = _url_candidate_score(
        "https://careers.airwallex.com/job/xyz/staff-data-scientist-growth-analytics/",
        "Staff Data Scientist, Growth Analytics",
        lead,
    )
    assert product_score > non_pm_score


def test_url_candidate_score_uses_parent_card_context_for_generic_view_job_links() -> None:
    lead = JobLead(
        company_name="Viant Technology",
        role_title="Principal, Product Manager- Data/ML",
        source_url="https://www.linkedin.com/jobs/view/123",
        source_type="linkedin",
        evidence_notes="AI/ML PM role.",
    )
    contextual_score = _url_candidate_score(
        "https://job-boards.greenhouse.io/vianttechnology/jobs/4193444009",
        "View Job Principal, Product Manager- Data/ML Irvine, California",
        lead,
    )
    wrong_score = _url_candidate_score(
        "https://job-boards.greenhouse.io/vianttechnology/jobs/4157363009",
        "View Job Account Director Dallas, Texas",
        lead,
    )
    assert contextual_score > wrong_score


def test_url_candidate_score_rewards_company_job_query_ids() -> None:
    lead = JobLead(
        company_name="Invoca",
        role_title="Staff Product Manager - Voice AI",
        source_url="https://builtin.com/job/staff-product-manager-voice-ai/8220410",
        source_type="builtin",
        evidence_notes="Remote AI PM role.",
    )
    score = _url_candidate_score(
        "https://www.invoca.com/company/job-listings?gh_jid=8375510002",
        "Apply",
        lead,
    )
    assert score[0] > 0


def test_extract_direct_job_url_from_builtin_source_prefers_specific_company_job_url(monkeypatch) -> None:
    lead = JobLead(
        company_name="Invoca",
        role_title="Staff Product Manager - Voice AI",
        source_url="https://builtin.com/job/staff-product-manager-voice-ai/8220410",
        source_type="builtin",
        evidence_notes="Remote AI PM role.",
    )

    class FakeResponse:
        def __init__(self, url: str, text: str) -> None:
            self.url = url
            self.text = text

    class FakeAsyncClient:
        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            assert url == lead.source_url
            return FakeResponse(
                url,
                '<a href="https://www.invoca.com/company/job-listings?gh_jid=8375510002">Apply now</a>',
            )

    monkeypatch.setattr("job_agent.job_search.httpx.AsyncClient", lambda **kwargs: FakeAsyncClient())

    direct_url = asyncio.run(_extract_direct_job_url_from_source(lead))
    assert direct_url == "https://www.invoca.com/company/job-listings?gh_jid=8375510002"


def test_extract_direct_job_url_from_linkedin_source_rejects_wrong_company_ats_links(monkeypatch) -> None:
    lead = JobLead(
        company_name="Vouch Insurance",
        role_title="Senior Product Manager, Platform (AI)",
        source_url="https://www.linkedin.com/jobs/view/4383931358",
        source_type="linkedin",
        evidence_notes="Remote AI product manager role.",
    )

    class FakeResponse:
        def __init__(self, url: str, text: str) -> None:
            self.url = url
            self.text = text

    class FakeAsyncClient:
        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            assert url == lead.source_url
            return FakeResponse(
                url,
                """
                <a href="https://jobs.lever.co/finch/814e67f8-ea70-493c-84b3-f39d7e281e9a">Apply</a>
                <a href="https://jobs.lever.co/vouch/2b7e5ccd-f64b-41c1-b443-fc8187c466b6">Correct</a>
                """,
            )

    monkeypatch.setattr("job_agent.job_search.httpx.AsyncClient", lambda **kwargs: FakeAsyncClient())

    direct_url = asyncio.run(_extract_direct_job_url_from_source(lead))
    assert direct_url == "https://jobs.lever.co/vouch/2b7e5ccd-f64b-41c1-b443-fc8187c466b6"


def test_deterministic_trim_local_leads_collapses_duplicate_company_role_variants() -> None:
    settings = build_settings()
    duplicate_linkedin = JobLead(
        company_name="Zeta Global",
        role_title="Senior Product Manager - AI Agents",
        source_url="https://www.linkedin.com/jobs/view/1",
        source_type="linkedin",
        evidence_notes="Remote AI agents PM role.",
    )
    duplicate_direct = JobLead(
        company_name="Zeta Global",
        role_title="Senior Product Manager - AI Agents",
        source_url="https://jobs.lever.co/zeta/abc",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/zeta/abc",
        evidence_notes="Direct ATS role.",
    )
    trimmed = _deterministic_trim_local_leads(settings, "senior product manager AI", [duplicate_linkedin, duplicate_direct], limit=10)
    assert len(trimmed) == 1
    assert trimmed[0].direct_job_url == "https://jobs.lever.co/zeta/abc"


def test_job_posting_dedupe_key_normalizes_direct_url_variants() -> None:
    job = JobPosting(
        company_name="January",
        role_title="Senior/Staff Product Manager, AI Agents",
        direct_job_url="https://jobs.ashbyhq.com/january/837101a2-6bc5-44e8-8f93-110638dcaca3?utm_source=5zgqMql0dg",
        resolved_job_url="https://jobs.ashbyhq.com/january/837101a2-6bc5-44e8-8f93-110638dcaca3?utm_source=5zgqMql0dg",
        ats_platform="Ashby",
        location_text="Remote, United States",
        is_fully_remote=True,
        posted_date_text="2026-03-17",
        posted_date_iso="2026-03-17",
        base_salary_min_usd=185000,
        base_salary_max_usd=244000,
        salary_text="$185,000 - $244,000",
        evidence_notes="AI agents role.",
    )
    assert _job_posting_dedupe_key(job) == "ashby:january:837101a2-6bc5-44e8-8f93-110638dcaca3"


def test_is_ai_related_product_manager_respects_primary_ai_pm_titles_even_with_long_evidence() -> None:
    job = JobPosting(
        company_name="Surgo Health",
        role_title="[Hiring] Senior Product Manager, AI & Data Products @Surgo Health",
        direct_job_url="https://remotive.com/remote/jobs/product/senior-product-manager-ai-data-products-3780284",
        resolved_job_url="https://remotive.com/remote/jobs/product/senior-product-manager-ai-data-products-3780284",
        ats_platform="remotive.com",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2026-03-17",
        posted_date_iso="2026-03-17",
        evidence_notes="Strong AI product role.",
        job_page_title="[Hiring] Senior Product Manager, AI & Data Products @Surgo Health",
        validation_evidence=[
            " ".join(
                [
                    "This long evidence blob includes plenty of surrounding text",
                    "but should not erase the clear AI product manager signal from the title.",
                ]
                * 12
            )
        ],
    )

    assert _is_ai_related_product_manager(job) is True


def test_resolve_lead_via_company_careers_pages_walks_homepage_to_board(monkeypatch) -> None:
    lead = JobLead(
        company_name="Versapay",
        role_title="Principal Product Manager - AI/ML",
        source_url="https://www.linkedin.com/jobs/view/4389040590",
        source_type="linkedin",
        evidence_notes="Remote AI/ML PM role.",
    )

    async def fake_search_company_resolution_candidates(_lead: JobLead) -> list[str]:
        return ["https://www.versapay.com/"]

    async def fake_extract_direct_job_url_from_source(candidate_lead: JobLead) -> str | None:
        if candidate_lead.source_url == "https://jobs.lever.co/versapay":
            return "https://jobs.lever.co/versapay/1305d3eb-5d36-4a7b-86d7-9c2ab53f83d4"
        return None

    class FakeResponse:
        def __init__(self, url: str, text: str) -> None:
            self.url = url
            self.text = text
            self.status_code = 200

    class FakeAsyncClient:
        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            if url == "https://www.versapay.com/":
                return FakeResponse(url, '<a href="/careers">Careers</a>')
            if url == "https://www.versapay.com/careers":
                return FakeResponse(url, '<a href="https://jobs.lever.co/versapay">Open roles</a>')
            raise AssertionError(f"Unexpected URL fetched: {url}")

    monkeypatch.setattr(
        "job_agent.job_search._search_company_resolution_candidates",
        fake_search_company_resolution_candidates,
    )
    monkeypatch.setattr(
        "job_agent.job_search._extract_direct_job_url_from_source",
        fake_extract_direct_job_url_from_source,
    )
    monkeypatch.setattr("job_agent.job_search.httpx.AsyncClient", lambda **kwargs: FakeAsyncClient())

    resolution = asyncio.run(_resolve_lead_via_company_careers_pages(lead))
    assert resolution is not None
    assert resolution.direct_job_url == "https://jobs.lever.co/versapay/1305d3eb-5d36-4a7b-86d7-9c2ab53f83d4"


def test_repair_direct_job_url_uses_company_careers_resolution_before_agent(monkeypatch) -> None:
    lead = JobLead(
        company_name="Versapay",
        role_title="Principal Product Manager - AI/ML",
        source_url="https://www.linkedin.com/jobs/view/4389040590",
        source_type="linkedin",
        evidence_notes="Remote AI/ML PM role.",
    )

    async def fake_extract_direct_job_url_from_source(_lead: JobLead) -> str | None:
        return None

    async def fake_resolve_lead_via_company_careers_pages(_lead: JobLead) -> DirectJobResolution | None:
        return DirectJobResolution(
            accepted=True,
            direct_job_url="https://jobs.lever.co/versapay/1305d3eb-5d36-4a7b-86d7-9c2ab53f83d4",
            ats_platform="Lever",
            evidence_notes="Resolved via careers page.",
        )

    monkeypatch.setattr(
        "job_agent.job_search._extract_direct_job_url_from_source",
        fake_extract_direct_job_url_from_source,
    )
    monkeypatch.setattr(
        "job_agent.job_search._resolve_lead_via_company_careers_pages",
        fake_resolve_lead_via_company_careers_pages,
    )

    repaired = asyncio.run(
        _repair_direct_job_url(
            None,
            lead,
            "https://jobs.smartrecruiters.com/versapay/wrong-role",
            "Resolved page title did not line up with expected role.",
        )
    )

    assert repaired == "https://jobs.lever.co/versapay/1305d3eb-5d36-4a7b-86d7-9c2ab53f83d4"


def test_local_query_rounds_rotate_to_new_variants_across_attempts() -> None:
    settings = build_settings()
    attempt_one = _build_local_query_rounds(settings, SearchTuning(attempt_number=1))
    attempt_two = _build_local_query_rounds(settings, SearchTuning(attempt_number=2))
    flat_one = [query for query_round in attempt_one for query in query_round]
    flat_two = [query for query_round in attempt_two for query in query_round]
    assert len(flat_one) == settings.max_search_rounds * settings.search_round_query_limit
    assert 0 < len(flat_two) <= settings.max_search_rounds * settings.search_round_query_limit
    assert len(set(flat_one).intersection(flat_two)) <= settings.search_round_query_limit


def test_normalize_role_title_to_focus_queries_generalizes_titles() -> None:
    genai_queries = _normalize_role_title_to_focus_queries("Senior Product Manager, GenAI Platform Products")
    assert genai_queries[:2] == [
        "senior product manager generative AI",
        "senior product manager AI platform",
    ]
    assert "senior product manager AI" in genai_queries
    assert _normalize_role_title_to_focus_queries("Senior ML Product Manager") == [
        "senior product manager machine learning",
        "senior product manager AI",
    ]


def test_duckduckgo_anomaly_detection_flags_bot_challenges() -> None:
    assert _is_duckduckgo_anomaly_page(403, "anomaly-modal")
    assert _is_duckduckgo_anomaly_page(202, "Unfortunately, bots use DuckDuckGo too.")
    assert not _is_duckduckgo_anomaly_page(200, "<div class='result'>ok</div>")


def test_adaptive_query_bank_prioritizes_focus_companies() -> None:
    settings = build_settings()
    tuning = SearchTuning(
        attempt_number=2,
        prioritize_recency=True,
        prioritize_salary=True,
        focus_companies=["Highspot"],
        focus_roles=["\"Principal Product Manager, AI\""],
    )
    query_bank = _build_search_query_bank(settings, tuning=tuning)
    assert query_bank
    assert "\"Highspot\" \"Principal Product Manager, AI\" remote" in query_bank[:12]


def test_apply_company_novelty_quota_keeps_ninety_percent_novel_companies() -> None:
    settings = build_settings()
    novel_leads = [
        JobLead(
            company_name=f"Novel {index}",
            role_title="Senior Product Manager, AI",
            source_url=f"https://jobs.lever.co/novel{index}/123",
            source_type="direct_ats",
            direct_job_url=f"https://jobs.lever.co/novel{index}/123",
            evidence_notes="Novel company role.",
        )
        for index in range(1, 10)
    ]
    repeated_leads = [
        JobLead(
            company_name=f"Known {index}",
            role_title="Senior Product Manager, AI",
            source_url=f"https://jobs.lever.co/known{index}/123",
            source_type="direct_ats",
            direct_job_url=f"https://jobs.lever.co/known{index}/123",
            evidence_notes="Previously reported company role.",
        )
        for index in range(1, 3)
    ]
    known_company_keys = {_normalize_company_key(lead.company_name) for lead in repeated_leads}

    ordered = _apply_company_novelty_quota(
        [*novel_leads, *repeated_leads],
        known_company_keys,
        min_novelty_ratio=0.90,
        limit=10,
    )

    ordered_keys = [_normalize_company_key(lead.company_name) for lead in ordered]
    assert sum(1 for key in ordered_keys if key not in known_company_keys) >= 9


def test_select_watchlist_focus_companies_prefers_small_unknown_companies(tmp_path: Path) -> None:
    settings = build_settings()
    settings.data_dir = tmp_path
    (settings.data_dir / "company-watchlist.json").write_text(
        json.dumps(
            {
                "tinyai": {
                    "company_name": "Tiny AI",
                    "priority_score": 12,
                    "watch_count": 3,
                    "source_hosts": ["jobs.ashbyhq.com"],
                },
                "bigcorp": {
                    "company_name": "Big Corp",
                    "priority_score": 20,
                    "watch_count": 5,
                    "source_hosts": ["linkedin.com"],
                },
            }
        ),
        encoding="utf-8",
    )

    focus_companies = _select_watchlist_focus_companies(settings, {"bigcorp"})
    assert focus_companies[0] == "Tiny AI"


def test_select_watchlist_focus_companies_penalizes_stale_high_churn_companies(tmp_path: Path) -> None:
    settings = build_settings()
    settings.data_dir = tmp_path
    (settings.data_dir / "company-watchlist.json").write_text(
        json.dumps(
            {
                "twilio": {
                    "company_name": "Twilio",
                    "priority_score": 2960,
                    "watch_count": 440,
                    "source_hosts": ["job-boards.greenhouse.io"],
                    "recent_rejection_reasons": {"stale_posting": 440},
                },
                "tinyai": {
                    "company_name": "Tiny AI",
                    "priority_score": 120,
                    "watch_count": 8,
                    "source_hosts": ["jobs.ashbyhq.com"],
                    "recent_rejection_reasons": {"stale_posting": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    focus_companies = _select_watchlist_focus_companies(settings, set())
    assert focus_companies[0] == "Tiny AI"


def test_select_watchlist_focus_companies_skips_saturated_repeat_failures(tmp_path: Path) -> None:
    settings = build_settings()
    settings.data_dir = tmp_path
    (settings.data_dir / "company-watchlist.json").write_text(
        json.dumps(
            {
                "jerry": {
                    "company_name": "Jerry",
                    "priority_score": 72,
                    "watch_count": 12,
                    "source_hosts": ["jobs.ashbyhq.com"],
                    "last_reason_code": "fetch_non_200",
                    "recent_rejection_reasons": {"fetch_non_200": 12},
                },
                "fresh": {
                    "company_name": "Fresh Startup",
                    "priority_score": 30,
                    "watch_count": 2,
                    "source_hosts": ["jobs.ashbyhq.com"],
                    "last_reason_code": "remote_unclear",
                    "recent_rejection_reasons": {"remote_unclear": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    focus_companies = _select_watchlist_focus_companies(settings, set())
    assert focus_companies == ["Fresh Startup"]


def test_small_company_scout_queries_bias_toward_direct_ats_hosts() -> None:
    settings = build_settings()
    queries = _build_small_company_scout_queries(
        settings,
        SearchTuning(attempt_number=14, prioritize_recency=True, prioritize_remote=True),
    )
    assert queries
    assert any("site:" in query and "product manager" in query for query in queries)
    assert all("site:" in query for query in queries[:6])
    assert any("\"portfolio company\"" in query for query in queries)
    assert any("\"early stage\"" in query for query in queries)


def test_watchlist_board_focus_queries_use_known_board_identifiers(tmp_path: Path) -> None:
    settings = build_settings()
    settings.data_dir = tmp_path
    (settings.data_dir / "company-watchlist.json").write_text(
        json.dumps(
            {
                "tinyai": {
                    "company_name": "Tiny AI",
                    "watch_count": 2,
                    "priority_score": 20,
                    "last_reason_code": "missing_salary",
                    "recent_rejection_reasons": {"missing_salary": 2},
                    "board_identifiers": ["ashby:tinyai"],
                    "source_hosts": ["jobs.ashbyhq.com"],
                }
            }
        ),
        encoding="utf-8",
    )

    queries = _build_watchlist_board_focus_queries(
        settings,
        SearchTuning(
            attempt_number=2,
            prioritize_recency=True,
            prioritize_salary=True,
            focus_companies=["Tiny AI"],
            focus_roles=["principal product manager AI"],
        ),
    )

    assert queries
    assert any("site:jobs.ashbyhq.com/tinyai" in query for query in queries)


def test_portfolio_company_scout_queries_target_startup_network_boards() -> None:
    settings = build_settings()
    queries = _build_portfolio_company_scout_queries(
        settings,
        SearchTuning(attempt_number=1, prioritize_recency=True, prioritize_remote=True),
    )
    assert queries
    assert any("workatastartup.com/jobs" in query or "ycombinator.com/companies" in query for query in queries)
    assert not any("getro.com/companies" in query for query in queries)
    assert any("startup" in query or "\"portfolio company\"" in query for query in queries)


def test_portfolio_company_scout_queries_promote_yc_before_getro_on_second_attempt() -> None:
    settings = build_settings()
    queries = _build_portfolio_company_scout_queries(
        settings,
        SearchTuning(attempt_number=2, prioritize_recency=True, prioritize_remote=True),
    )

    assert any("ycombinator.com/companies" in query for query in queries[:6])
    assert not any("getro.com/companies" in query for query in queries[:6])


def test_local_query_rounds_prioritize_focus_companies_without_compound_site_queries() -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    rounds = _build_local_query_rounds(
        settings,
        SearchTuning(
            attempt_number=1,
            prioritize_recency=True,
            prioritize_salary=True,
            focus_companies=["Tiny AI"],
            focus_roles=["\"Senior Product Manager, AI\""],
        ),
    )
    flat_queries = [query for query_round in rounds for query in query_round]
    assert flat_queries
    assert any("Tiny AI" in query for query in flat_queries[:12])
    assert all(query.count("site:") <= 1 for query in flat_queries)


def test_local_query_rounds_use_site_scoped_focus_queries_on_late_attempts() -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    (settings.data_dir / "company-watchlist.json").parent.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "company-watchlist.json").write_text(
        json.dumps(
            {
                "tinyai": {
                    "company_name": "Tiny AI",
                    "watch_count": 2,
                    "priority_score": 20,
                    "last_reason_code": "missing_salary",
                    "recent_rejection_reasons": {"missing_salary": 2},
                    "board_identifiers": ["ashby:tinyai"],
                    "source_hosts": ["jobs.ashbyhq.com"],
                }
            }
        ),
        encoding="utf-8",
    )
    rounds = _build_local_query_rounds(
        settings,
        SearchTuning(
            attempt_number=2,
            prioritize_recency=True,
            prioritize_salary=True,
            focus_companies=["Tiny AI"],
            focus_roles=["principal product manager AI"],
        ),
    )

    flat_queries = [query for query_round in rounds for query in query_round]
    assert flat_queries
    assert any('"Tiny AI"' in query and "site:" in query for query in flat_queries)
    assert not any(query == "AI product manager remote" for query in flat_queries)


def test_lead_priority_prefers_small_company_hosts_over_enterprise_ats() -> None:
    settings = build_settings()
    small_company_lead = JobLead(
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://jobs.ashbyhq.com/tinyai/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.ashbyhq.com/tinyai/123",
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=220000,
        evidence_notes="Direct startup ATS role.",
    )
    enterprise_lead = JobLead(
        company_name="Big Enterprise",
        role_title="Senior Product Manager, AI",
        source_url="https://bigenterprise.myworkdayjobs.com/en-US/Careers/job/123",
        source_type="direct_ats",
        direct_job_url="https://bigenterprise.myworkdayjobs.com/en-US/Careers/job/123",
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=220000,
        evidence_notes="Direct enterprise ATS role.",
    )

    assert _lead_priority(small_company_lead, settings) < _lead_priority(enterprise_lead, settings)


def test_refine_local_leads_with_ollama_merges_cleaned_candidates(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    original = JobLead(
        company_name="Acme",
        role_title="Senior Product Manager, AI",
        source_url="https://www.linkedin.com/jobs/view/123",
        source_type="linkedin",
        direct_job_url=None,
        evidence_notes="LinkedIn discovery result.",
    )
    cleaned = JobLead(
        company_name="Acme",
        role_title="Senior Product Manager, AI",
        source_url="https://jobs.lever.co/acme/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/acme/123",
        evidence_notes="Direct ATS role.",
    )

    async def fake_cleanup(_settings: Settings, _query: str, _leads: list[JobLead]) -> list[JobLead]:
        return [cleaned]

    monkeypatch.setattr("job_agent.job_search._cleanup_local_leads_with_ollama", fake_cleanup)

    refined = asyncio.run(
        _refine_local_leads_with_ollama(
            settings,
            "senior product manager ai",
            [original],
            cleanup_limit=8,
        )
    )
    assert refined[0].direct_job_url == "https://jobs.lever.co/acme/123"
    assert any(lead.direct_job_url == "https://jobs.lever.co/acme/123" for lead in refined)


def test_refine_local_leads_with_ollama_skips_forced_modes_without_cleanup_signals(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    lead_one = JobLead(
        company_name="Acme",
        role_title="Senior Product Manager, AI",
        source_url="https://jobs.lever.co/acme/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/acme/123",
        evidence_notes="Direct ATS role.",
    )
    lead_two = JobLead(
        company_name="Beta",
        role_title="Staff Product Manager, AI",
        source_url="https://jobs.ashbyhq.com/beta/456",
        source_type="direct_ats",
        direct_job_url="https://jobs.ashbyhq.com/beta/456",
        evidence_notes="Direct ATS role.",
    )

    async def fail_cleanup(*args, **kwargs):
        raise AssertionError("Ollama cleanup should not run when forced refinement has no cleanup signals.")

    monkeypatch.setattr("job_agent.job_search._cleanup_local_leads_with_ollama", fail_cleanup)
    monkeypatch.setattr("job_agent.job_search.record_ollama_event", lambda *args, **kwargs: None)

    refined = asyncio.run(
        _refine_local_leads_with_ollama(
            settings,
            "seed replay triage",
            [lead_one, lead_two],
            cleanup_limit=2,
            refinement_mode="forced_seed_triage",
            pre_refinement_average_confidence=0.967,
            pre_refinement_cleanup_signal_count=0,
            pre_refinement_trustworthy_direct_url_count=2,
            run_id="run-guard",
        )
    )

    assert refined == [lead_one, lead_two]


def test_search_single_query_local_forces_one_ollama_sample_per_attempt(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    lead_one = JobLead(
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://builtin.com/job/123",
        source_type="builtin",
        direct_job_url=None,
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=220000,
        evidence_notes="Mirror board result that still needs cleanup.",
    )
    lead_two = JobLead(
        company_name="Acme AI",
        role_title="Staff Product Manager, AI",
        source_url="https://jobs.lever.co/acme/456",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/acme/456",
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=230000,
        evidence_notes="Direct ATS role.",
    )
    lead_map = {
        "https://builtin.com/job/123": lead_one,
        "https://jobs.lever.co/acme/456": lead_two,
    }
    forced_calls: list[tuple[int, str | None]] = []

    async def fake_builtin_search(_query: str, _settings: Settings) -> list[JobLead]:
        return []

    async def fake_linkedin_search(_query: str, _settings: Settings) -> list[JobLead]:
        return []

    async def fake_local_search(_query: str, *, max_results_per_query: int) -> list[tuple[str, str, str]]:
        return [
            ("https://builtin.com/job/123", "Tiny AI - Senior Product Manager, AI", ""),
            ("https://jobs.lever.co/acme/456", "Acme AI - Staff Product Manager, AI", ""),
        ]

    async def fake_refine(
        _settings: Settings,
        _query: str,
        candidate_pool: list[JobLead],
        *,
        cleanup_limit: int,
        refinement_mode: str | None = None,
        pre_refinement_average_confidence: float | None = None,
        pre_refinement_cleanup_signal_count: int | None = None,
        pre_refinement_trustworthy_direct_url_count: int | None = None,
        run_id: str | None = None,
    ) -> list[JobLead]:
        forced_calls.append((cleanup_limit, refinement_mode))
        return candidate_pool

    monkeypatch.setattr("job_agent.job_search._builtin_search", fake_builtin_search)
    monkeypatch.setattr("job_agent.job_search._linkedin_guest_search", fake_linkedin_search)
    monkeypatch.setattr("job_agent.job_search._run_local_search_engine_queries", fake_local_search)
    monkeypatch.setattr("job_agent.job_search._build_lead_from_search_result", lambda url, title, snippet, query: lead_map[url])
    monkeypatch.setattr("job_agent.job_search._ollama_refinement_mode_for_local_leads", lambda *args, **kwargs: None)
    monkeypatch.setattr("job_agent.job_search._refine_local_leads_with_ollama", fake_refine)
    monkeypatch.setattr("job_agent.job_search.record_ollama_event", lambda *args, **kwargs: None)
    import job_agent.job_search as job_search_module

    job_search_module.FORCED_OLLAMA_REFINEMENT_ATTEMPTS.clear()

    asyncio.run(
        _search_single_query_local(
            settings,
            'site:ycombinator.com/companies "product manager" "applied intelligence" remote startup',
            attempt_number=2,
            run_id="run-1",
        )
    )
    asyncio.run(
        _search_single_query_local(
            settings,
            'site:ycombinator.com/companies "senior product manager" "applied intelligence" remote',
            attempt_number=2,
            run_id="run-1",
        )
    )

    assert forced_calls == [(2, "forced_sample")]


def test_search_single_query_local_skips_forced_ollama_sample_for_clean_direct_ats_bundle(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    lead_one = JobLead(
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://jobs.ashbyhq.com/tinyai/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.ashbyhq.com/tinyai/123",
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=220000,
        evidence_notes="Direct ATS role.",
    )
    lead_two = JobLead(
        company_name="Acme AI",
        role_title="Staff Product Manager, AI",
        source_url="https://jobs.lever.co/acme/456",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/acme/456",
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=230000,
        evidence_notes="Direct ATS role.",
    )
    lead_map = {
        "https://jobs.ashbyhq.com/tinyai/123": lead_one,
        "https://jobs.lever.co/acme/456": lead_two,
    }
    forced_calls: list[tuple[int, str | None]] = []

    async def fake_builtin_search(_query: str, _settings: Settings) -> list[JobLead]:
        return []

    async def fake_linkedin_search(_query: str, _settings: Settings) -> list[JobLead]:
        return []

    async def fake_local_search(_query: str, *, max_results_per_query: int) -> list[tuple[str, str, str]]:
        return [
            ("https://jobs.ashbyhq.com/tinyai/123", "Tiny AI - Senior Product Manager, AI", ""),
            ("https://jobs.lever.co/acme/456", "Acme AI - Staff Product Manager, AI", ""),
        ]

    async def fake_refine(
        _settings: Settings,
        _query: str,
        candidate_pool: list[JobLead],
        *,
        cleanup_limit: int,
        refinement_mode: str | None = None,
        pre_refinement_average_confidence: float | None = None,
        pre_refinement_cleanup_signal_count: int | None = None,
        pre_refinement_trustworthy_direct_url_count: int | None = None,
        run_id: str | None = None,
    ) -> list[JobLead]:
        forced_calls.append((cleanup_limit, refinement_mode))
        return candidate_pool

    monkeypatch.setattr("job_agent.job_search._builtin_search", fake_builtin_search)
    monkeypatch.setattr("job_agent.job_search._linkedin_guest_search", fake_linkedin_search)
    monkeypatch.setattr("job_agent.job_search._run_local_search_engine_queries", fake_local_search)
    monkeypatch.setattr("job_agent.job_search._build_lead_from_search_result", lambda url, title, snippet, query: lead_map[url])
    monkeypatch.setattr("job_agent.job_search._ollama_refinement_mode_for_local_leads", lambda *args, **kwargs: None)
    monkeypatch.setattr("job_agent.job_search._refine_local_leads_with_ollama", fake_refine)
    monkeypatch.setattr("job_agent.job_search.record_ollama_event", lambda *args, **kwargs: None)
    import job_agent.job_search as job_search_module

    job_search_module.FORCED_OLLAMA_REFINEMENT_ATTEMPTS.clear()

    asyncio.run(
        _search_single_query_local(
            settings,
            'site:ycombinator.com/companies "product manager" "applied intelligence" remote startup',
            attempt_number=2,
            run_id="run-1",
        )
    )

    assert forced_calls == []


def test_maybe_force_round_lead_refinement_with_ollama_runs_once_per_attempt(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    lead_one = JobLead(
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://builtin.com/job/123",
        source_type="builtin",
        direct_job_url=None,
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=220000,
        evidence_notes="Mirror board result that still needs cleanup.",
    )
    lead_two = JobLead(
        company_name="Acme AI",
        role_title="Staff Product Manager, AI",
        source_url="https://jobs.lever.co/acme/456",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/acme/456",
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=230000,
        evidence_notes="Direct ATS role.",
    )
    forced_calls: list[tuple[int, str | None, str]] = []

    async def fake_refine(
        _settings: Settings,
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
        forced_calls.append((cleanup_limit, refinement_mode, query))
        return candidate_pool

    monkeypatch.setattr("job_agent.job_search._refine_local_leads_with_ollama", fake_refine)
    import job_agent.job_search as job_search_module

    job_search_module.FORCED_OLLAMA_ROUND_REFINEMENT_ATTEMPTS.clear()

    first = asyncio.run(
        _maybe_force_round_lead_refinement_with_ollama(
            settings,
            [lead_one, lead_two],
            attempt_number=2,
            round_number=1,
            run_id="run-1",
        )
    )
    second = asyncio.run(
        _maybe_force_round_lead_refinement_with_ollama(
            settings,
            [lead_one, lead_two],
            attempt_number=2,
            round_number=2,
            run_id="run-1",
        )
    )

    assert first == [lead_one, lead_two]
    assert second == [lead_one, lead_two]
    assert forced_calls == [(2, "forced_round_sample", "attempt 2 round 1 aggregate cleanup")]


def test_maybe_force_round_lead_refinement_with_ollama_skips_clean_high_confidence_bundle(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    lead_one = JobLead(
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://builtin.com/job/123",
        source_type="builtin",
        direct_job_url=None,
        evidence_notes="Mirror board result that still needs cleanup.",
    )
    lead_two = JobLead(
        company_name="Acme AI",
        role_title="Staff Product Manager, AI",
        source_url="https://jobs.lever.co/acme/456",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/acme/456",
        evidence_notes="Direct ATS role.",
    )
    forced_calls: list[str] = []

    async def fake_refine(*args, **kwargs):
        forced_calls.append("called")
        return [lead_one, lead_two]

    monkeypatch.setattr("job_agent.job_search._should_force_ollama_refinement_sample", lambda *args, **kwargs: True)
    monkeypatch.setattr("job_agent.job_search._lead_confidence", lambda lead: 0.9)
    monkeypatch.setattr("job_agent.job_search._refine_local_leads_with_ollama", fake_refine)
    import job_agent.job_search as job_search_module

    job_search_module.FORCED_OLLAMA_ROUND_REFINEMENT_ATTEMPTS.clear()

    refined = asyncio.run(
        _maybe_force_round_lead_refinement_with_ollama(
            settings,
            [lead_one, lead_two],
            attempt_number=1,
            round_number=1,
            run_id="run-clean-round-guard",
        )
    )

    assert refined == [lead_one, lead_two]
    assert forced_calls == []


def test_maybe_force_seed_lead_refinement_with_ollama_skips_clean_seed_bundle(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    lead_one = JobLead(
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://jobs.ashbyhq.com/tinyai/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.ashbyhq.com/tinyai/123",
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=220000,
        evidence_notes="Direct ATS role.",
    )
    lead_two = JobLead(
        company_name="Acme AI",
        role_title="Staff Product Manager, AI",
        source_url="https://jobs.lever.co/acme/456",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/acme/456",
        is_remote_hint=True,
        posted_date_hint="today",
        base_salary_min_usd_hint=230000,
        evidence_notes="Direct ATS role.",
    )
    forced_calls: list[tuple[int, str | None, str]] = []

    async def fake_refine(
        _settings: Settings,
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
        forced_calls.append((cleanup_limit, refinement_mode, query))
        return candidate_pool

    monkeypatch.setattr("job_agent.job_search._refine_local_leads_with_ollama", fake_refine)
    import job_agent.job_search as job_search_module

    job_search_module.FORCED_OLLAMA_SEED_REFINEMENT_RUNS.clear()

    refined = asyncio.run(
        _maybe_force_seed_lead_refinement_with_ollama(
            settings,
            [lead_one, lead_two],
            run_id="run-1",
        )
    )

    assert refined == [lead_one, lead_two]
    assert forced_calls == []


def test_maybe_force_seed_lead_refinement_with_ollama_requires_cleanup_signals_even_when_gate_opens(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    lead_one = JobLead(
        company_name="Tiny AI",
        role_title="Senior Product Manager, AI",
        source_url="https://jobs.ashbyhq.com/tinyai/123",
        source_type="direct_ats",
        direct_job_url="https://jobs.ashbyhq.com/tinyai/123",
        evidence_notes="Direct ATS role.",
    )
    lead_two = JobLead(
        company_name="Acme AI",
        role_title="Staff Product Manager, AI",
        source_url="https://jobs.lever.co/acme/456",
        source_type="direct_ats",
        direct_job_url="https://jobs.lever.co/acme/456",
        evidence_notes="Direct ATS role.",
    )
    forced_calls: list[str] = []

    async def fake_refine(*args, **kwargs):
        forced_calls.append("called")
        return [lead_one, lead_two]

    monkeypatch.setattr("job_agent.job_search._should_force_ollama_refinement_sample", lambda *args, **kwargs: True)
    monkeypatch.setattr("job_agent.job_search._refine_local_leads_with_ollama", fake_refine)
    import job_agent.job_search as job_search_module

    job_search_module.FORCED_OLLAMA_SEED_REFINEMENT_RUNS.clear()

    refined = asyncio.run(
        _maybe_force_seed_lead_refinement_with_ollama(
            settings,
            [lead_one, lead_two],
            run_id="run-clean-guard",
        )
    )

    assert refined == [lead_one, lead_two]
    assert forced_calls == []


def test_ensure_lazy_ollama_prewarm_only_runs_once_per_run(monkeypatch) -> None:
    settings = build_settings()
    settings.llm_provider = "ollama"
    prewarm_calls: list[str | None] = []

    async def fake_prewarm(_settings: Settings, *, run_id: str | None = None):
        prewarm_calls.append(run_id)
        return True, None, 1.0

    monkeypatch.setattr("job_agent.job_search.prewarm_ollama_model", fake_prewarm)
    import job_agent.job_search as job_search_module

    job_search_module.LAZY_OLLAMA_PREWARM_RUNS.clear()

    first = asyncio.run(_ensure_lazy_ollama_prewarm(settings, run_id="run-lazy"))
    second = asyncio.run(_ensure_lazy_ollama_prewarm(settings, run_id="run-lazy"))

    assert first is True
    assert second is True
    assert prewarm_calls == ["run-lazy"]


def test_dead_attempt_helpers_stop_timeout_heavy_zero_yield_passes() -> None:
    diagnostics = SearchDiagnostics(
        minimum_qualifying_jobs=5,
        failures=[
            SearchFailure(stage="discovery", reason_code="query_timeout", detail="timed out", attempt_number=2, round_number=1),
            SearchFailure(stage="discovery", reason_code="query_timeout", detail="timed out", attempt_number=2, round_number=1),
            SearchFailure(stage="discovery", reason_code="query_skipped_timeout_budget", detail="skip", attempt_number=2, round_number=2),
            SearchFailure(stage="discovery", reason_code="query_skipped_timeout_budget", detail="skip", attempt_number=2, round_number=2),
        ],
    )

    assert _should_abort_dead_attempt_round(
        diagnostics,
        attempt_number=2,
        consecutive_zero_yield_rounds=2,
        attempt_discovery_gain=0,
    )
    assert _should_stop_after_dead_attempt(
        diagnostics,
        attempt_number=2,
        attempt_discovery_gain=0,
        resolved_leads_this_attempt=0,
    )
