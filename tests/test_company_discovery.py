import json
from pathlib import Path

from job_agent.company_discovery import (
    board_identifier_from_url,
    extract_embedded_board_urls,
    extract_careers_page_urls,
    extract_directory_company_tasks,
    infer_careers_root,
    load_company_discovery_entries,
    load_company_discovery_frontier,
    make_frontier_task,
    save_company_discovery_entries,
    save_company_discovery_frontier,
    select_frontier_tasks,
    source_directory_seed_tasks,
    is_low_value_company_discovery_entry,
    upsert_frontier_task,
    upsert_company_discovery_entry,
    company_discovery_index_path,
    company_discovery_frontier_path,
    extract_company_homepage_urls,
)


def test_extract_embedded_board_urls_finds_embedded_ashby_and_greenhouse_links() -> None:
    html = """
    <html>
      <body>
        <a href="https://jobs.ashbyhq.com/butterflymx">Open roles</a>
        <script src="https://boards.greenhouse.io/embed/job_board/js?for=acme"></script>
      </body>
    </html>
    """

    urls = extract_embedded_board_urls("https://butterflymx.com/careers", html)

    assert "https://jobs.ashbyhq.com/butterflymx" in urls
    assert "https://boards.greenhouse.io/acme" in urls


def test_extract_embedded_board_urls_finds_smartrecruiters_urls_in_data_attributes_and_escaped_json() -> None:
    html = r"""
    <html>
      <body>
        <div data-board-url="https://careers.smartrecruiters.com/acme-ai"></div>
        <script>
          window.__JOBS__ = {"careersUrl":"https:\/\/jobs.smartrecruiters.com\/acme-ai"};
        </script>
      </body>
    </html>
    """

    urls = extract_embedded_board_urls("https://acme.example/careers", html)

    assert "https://careers.smartrecruiters.com/acme-ai" in urls
    assert "https://jobs.smartrecruiters.com/acme-ai" in urls


def test_smartrecruiters_board_urls_preserve_company_token() -> None:
    assert board_identifier_from_url("https://careers.smartrecruiters.com/acme-ai") == "smartrecruiters:acme-ai"
    assert (
        infer_careers_root("https://jobs.smartrecruiters.com/acme-ai/744000123456789-principal-product-manager-ai")
        == "https://jobs.smartrecruiters.com/acme-ai"
    )


def test_workday_job_urls_preserve_board_root_path() -> None:
    assert board_identifier_from_url(
        "https://capitalgroup.wd1.myworkdayjobs.com/en-US/CGCareers/job/Principal-Product-Manager-AI_R-123456"
    ) == "workday:capitalgroup.wd1"
    assert (
        infer_careers_root(
            "https://capitalgroup.wd1.myworkdayjobs.com/en-US/CGCareers/job/Principal-Product-Manager-AI_R-123456"
        )
        == "https://capitalgroup.wd1.myworkdayjobs.com/en-US/CGCareers"
    )
    assert (
        infer_careers_root(
            "https://hpe.wd5.myworkdayjobs.com/ACJobSite/job/Sunnyvale-California-United-States-of-America/Principal-Product-Hardware-Manager--Cloud-Infrastructure-and-AI-Networking_1201888-2"
        )
        == "https://hpe.wd5.myworkdayjobs.com/ACJobSite"
    )


def test_upsert_company_discovery_entry_merges_repeated_board_discoveries(tmp_path: Path) -> None:
    entries: dict[str, dict[str, object]] = {}

    was_new_company, new_board_count = upsert_company_discovery_entry(
        entries,
        company_name="ButterflyMX",
        source_url="https://butterflymx.com/careers",
        careers_root="https://butterflymx.com/careers",
        board_urls=["https://jobs.ashbyhq.com/butterflymx"],
        source_trust=8,
        run_id="run-1",
    )
    assert was_new_company is True
    assert new_board_count == 1

    was_new_company, new_board_count = upsert_company_discovery_entry(
        entries,
        company_name="ButterflyMX",
        source_url="https://butterflymx.com/careers",
        careers_root="https://butterflymx.com/careers",
        board_urls=[
            "https://jobs.ashbyhq.com/butterflymx",
            "https://job-boards.greenhouse.io/butterflymx",
        ],
        source_trust=9,
        run_id="run-2",
    )
    assert was_new_company is False
    assert new_board_count == 1
    assert sorted(entries["butterflymx"]["board_identifiers"]) == ["ashby:butterflymx", "greenhouse:butterflymx"]

    save_company_discovery_entries(tmp_path, entries)
    reloaded = load_company_discovery_entries(tmp_path)
    assert reloaded["butterflymx"]["source_trust"] == 9
    assert "https://butterflymx.com/careers" in reloaded["butterflymx"]["careers_roots"]


def test_select_frontier_tasks_only_returns_pending_ready_items() -> None:
    tasks = [
        make_frontier_task(task_type="directory_source", url="https://www.ycombinator.com/companies", priority=3),
        {
            **make_frontier_task(task_type="board_url", url="https://jobs.ashbyhq.com/butterflymx", priority=9),
            "status": "completed",
        },
        {
            **make_frontier_task(task_type="company_page", url="https://butterflymx.com", priority=5),
            "next_retry_at": "2999-01-01T00:00:00+00:00",
        },
    ]

    selected = select_frontier_tasks(tasks, budget=5)

    assert len(selected) == 1
    assert selected[0]["task_type"] == "directory_source"


def test_select_frontier_tasks_round_robins_across_companies() -> None:
    tasks = [
        make_frontier_task(task_type="company_page", url="https://alpha.example", company_name="Alpha", priority=10),
        make_frontier_task(task_type="careers_root", url="https://alpha.example/careers", company_name="Alpha", priority=9),
        make_frontier_task(task_type="company_page", url="https://beta.example", company_name="Beta", priority=8),
    ]

    selected = select_frontier_tasks(tasks, budget=3)

    assert [task["company_key"] for task in selected] == ["alpha", "beta", "alpha"]


def test_select_frontier_tasks_returns_live_task_references() -> None:
    tasks = [
        make_frontier_task(task_type="board_url", url="https://jobs.smartrecruiters.com/acme-ai", priority=10),
    ]

    selected = select_frontier_tasks(tasks, budget=1)
    selected[0]["status"] = "completed"

    assert tasks[0]["status"] == "completed"


def test_upsert_frontier_task_does_not_reactivate_completed_entries_by_default() -> None:
    tasks = [
        {
            **make_frontier_task(task_type="company_page", url="https://alpha.example", company_name="Alpha", priority=6),
            "status": "completed",
        }
    ]

    created = upsert_frontier_task(
        tasks,
        task_type="company_page",
        url="https://alpha.example",
        company_name="Alpha",
        priority=9,
    )

    assert created is False
    assert tasks[0]["status"] == "completed"


def test_extract_directory_company_tasks_preserves_company_identity_for_directory_pages() -> None:
    html = """
    <html>
      <body>
        <a href="/company/bilt-rewards/jobs">Bilt Rewards Jobs</a>
        <a href="https://www.ycombinator.com/companies/dynamo-ai">Dynamo AI</a>
      </body>
    </html>
    """

    tasks = extract_directory_company_tasks("https://www.builtin.com/jobs", html)

    assert {"url": "https://www.builtin.com/company/bilt-rewards/jobs", "company_name": "Bilt Rewards", "task_type": "careers_root"} in tasks
    assert {"url": "https://www.ycombinator.com/companies/dynamo-ai", "company_name": "Dynamo AI", "task_type": "company_page"} in tasks


def test_extract_directory_company_tasks_supports_welcome_to_the_jungle_company_routes() -> None:
    html = """
    <html>
      <body>
        <a href="/en/companies/figma/jobs">Figma jobs</a>
        <a href="/en/companies/linear">Linear</a>
      </body>
    </html>
    """

    tasks = extract_directory_company_tasks("https://www.welcometothejungle.com/en/companies", html)

    assert {"url": "https://www.welcometothejungle.com/en/companies/figma/jobs", "company_name": "Figma", "task_type": "careers_root"} in tasks
    assert {"url": "https://www.welcometothejungle.com/en/companies/linear", "company_name": "Linear", "task_type": "company_page"} in tasks


def test_extract_careers_page_urls_canonicalizes_directory_company_jobs_and_ignores_cross_company_links() -> None:
    html = """
    <html>
      <body>
        <a href="/company/freewheel/jobs">Jobs</a>
        <a href="/company/freewheel/offices">Offices</a>
        <a href="/company/freewheel/articles">Articles</a>
        <a href="/company/freewheel/benefits">Benefits</a>
        <a href="/company/comcast-3/jobs">Parent company jobs</a>
      </body>
    </html>
    """

    urls = extract_careers_page_urls("https://www.builtin.com/company/freewheel", html)

    assert urls == ["https://www.builtin.com/company/freewheel/jobs"]


def test_extract_careers_page_urls_collapses_non_directory_detail_filter_and_saved_jobs_links() -> None:
    html = """
    <html>
      <body>
        <a href="/jobs?page=2#results">Page 2</a>
        <a href="/jobs/saved-jobs">Saved jobs</a>
        <a href="/jobs/744000107435185/principal-inbound-product-manager-ai-assistant#content">Job detail</a>
      </body>
    </html>
    """

    urls = extract_careers_page_urls("https://careers.servicenow.com/jobs", html)

    assert urls == ["https://careers.servicenow.com/jobs"]


def test_extract_company_homepage_urls_skips_social_and_directory_hosts() -> None:
    html = """
    <html>
      <body>
        <a href="https://twitter.com/acme">Twitter</a>
        <a href="https://www.linkedin.com/company/acme">LinkedIn</a>
        <a href="https://www.builtinnyc.com/company/acme/jobs">Built In</a>
        <a href="https://acme.example">Acme</a>
      </body>
    </html>
    """

    urls = extract_company_homepage_urls("https://www.ycombinator.com/companies", html)

    assert urls == ["https://acme.example"]


def test_source_directory_seed_tasks_include_welcome_to_the_jungle() -> None:
    tasks = source_directory_seed_tasks()

    assert any(task["url"] == "https://www.welcometothejungle.com/en/companies" for task in tasks)


def test_load_company_discovery_frontier_repairs_directory_identity_and_filters_recursive_platforms(tmp_path: Path) -> None:
    frontier_path = company_discovery_frontier_path(tmp_path)
    frontier_path.write_text(
        """
        [
          {
            "task_key": "careers_root:https://www.builtin.com/company/bilt-rewards/jobs",
            "task_type": "careers_root",
            "url": "https://www.builtin.com/company/bilt-rewards/jobs",
            "company_name": "Builtin",
            "company_key": "builtin",
            "board_identifier": null,
            "source_kind": "careers_root",
            "source_trust": 5,
            "priority": 8,
            "attempts": 0,
            "status": "pending",
            "discovered_from": "https://www.builtin.com/jobs",
            "next_retry_at": null
          },
          {
            "task_key": "company_page:https://www.linkedin.com",
            "task_type": "company_page",
            "url": "https://www.linkedin.com",
            "company_name": null,
            "company_key": null,
            "board_identifier": null,
            "source_kind": "company_page",
            "source_trust": 7,
            "priority": 6,
            "attempts": 0,
            "status": "pending",
            "discovered_from": "https://www.builtin.com/jobs",
            "next_retry_at": null
          },
          {
            "task_key": "directory_source:https://wellfound.com/jobs",
            "task_type": "directory_source",
            "url": "https://wellfound.com/jobs",
            "company_name": null,
            "company_key": null,
            "board_identifier": null,
            "source_kind": "directory_source",
            "source_trust": 5,
            "priority": 3,
            "attempts": 0,
            "status": "pending",
            "discovered_from": null,
            "next_retry_at": null
          }
        ]
        """,
        encoding="utf-8",
    )

    tasks = load_company_discovery_frontier(tmp_path)

    assert len(tasks) == 2
    assert tasks[0]["company_name"] == "Bilt Rewards"
    assert tasks[0]["company_key"] == "biltrewards"
    assert {task["task_type"] for task in tasks} == {"careers_root", "directory_source"}


def test_load_company_discovery_frontier_collapses_non_directory_careers_root_variants(tmp_path: Path) -> None:
    save_company_discovery_frontier(
        tmp_path,
        [
            {
                "task_key": "careers_root:https://careers.servicenow.com/jobs?page=2#results",
                "task_type": "careers_root",
                "url": "https://careers.servicenow.com/jobs?page=2#results",
                "company_name": "ServiceNow",
                "company_key": "servicenow",
                "board_identifier": None,
                "source_kind": "careers_root",
                "source_trust": 7,
                "priority": 7,
                "attempts": 0,
                "status": "pending",
                "discovered_from": "https://careers.servicenow.com/jobs",
                "next_retry_at": None,
            },
            {
                "task_key": "careers_root:https://careers.servicenow.com/jobs/saved-jobs",
                "task_type": "careers_root",
                "url": "https://careers.servicenow.com/jobs/saved-jobs",
                "company_name": "ServiceNow",
                "company_key": "servicenow",
                "board_identifier": None,
                "source_kind": "careers_root",
                "source_trust": 7,
                "priority": 7,
                "attempts": 0,
                "status": "pending",
                "discovered_from": "https://careers.servicenow.com/jobs",
                "next_retry_at": None,
            },
            {
                "task_key": "careers_root:https://careers.servicenow.com/jobs/744000107435185/principal-inbound-product-manager-ai-assistant#content",
                "task_type": "careers_root",
                "url": "https://careers.servicenow.com/jobs/744000107435185/principal-inbound-product-manager-ai-assistant#content",
                "company_name": "ServiceNow",
                "company_key": "servicenow",
                "board_identifier": None,
                "source_kind": "careers_root",
                "source_trust": 7,
                "priority": 7,
                "attempts": 0,
                "status": "pending",
                "discovered_from": "https://careers.servicenow.com/jobs",
                "next_retry_at": None,
            },
        ],
    )

    tasks = load_company_discovery_frontier(tmp_path)

    assert len(tasks) == 1
    assert tasks[0]["task_type"] == "careers_root"
    assert tasks[0]["url"] == "https://careers.servicenow.com/jobs"


def test_load_company_discovery_frontier_repairs_board_identifier_strings_and_smartrecruiters_roots(
    tmp_path: Path,
) -> None:
    frontier_path = company_discovery_frontier_path(tmp_path)
    frontier_path.write_text(
        """
        [
          {
            "task_key": "board_url:https://jobs.smartrecruiters.com:smartrecruiters:acme-ai",
            "task_type": "board_url",
            "url": "https://jobs.smartrecruiters.com",
            "company_name": "Acme AI",
            "company_key": "acmeai",
            "board_identifier": "smartrecruiters:acme-ai",
            "source_kind": "board_url",
            "source_trust": 10,
            "priority": 10,
            "attempts": 1,
            "status": "pending",
            "discovered_from": "seed",
            "next_retry_at": "2999-01-01T00:00:00+00:00",
            "last_error": "unsupported_adapter"
          },
          {
            "task_key": "board_url:https://careers.example.com/jobs/123:None",
            "task_type": "board_url",
            "url": "https://careers.example.com/jobs/123",
            "company_name": "Example",
            "company_key": "example",
            "board_identifier": "None",
            "source_kind": "board_url",
            "source_trust": 7,
            "priority": 8,
            "attempts": 1,
            "status": "pending",
            "discovered_from": "seed",
            "next_retry_at": null,
            "last_error": "missing_board_identifier"
          }
        ]
        """,
        encoding="utf-8",
    )

    tasks = load_company_discovery_frontier(tmp_path)

    smartrecruiters_task = next(task for task in tasks if task["company_key"] == "acmeai")
    assert smartrecruiters_task["url"] == "https://jobs.smartrecruiters.com/acme-ai"
    assert smartrecruiters_task["board_identifier"] == "smartrecruiters:acme-ai"

    unknown_task = next(task for task in tasks if task["company_key"] == "example")
    assert unknown_task["board_identifier"] is None
    assert not str(unknown_task["task_key"]).endswith(":None")


def test_load_company_discovery_frontier_canonicalizes_directory_company_tasks_and_drops_cross_company_leaks(tmp_path: Path) -> None:
    frontier_path = company_discovery_frontier_path(tmp_path)
    frontier_path.write_text(
        """
        [
          {
            "task_key": "careers_root:https://www.builtin.com/company/boeing/articles/jobs",
            "task_type": "careers_root",
            "url": "https://www.builtin.com/company/boeing/articles/jobs",
            "company_name": "Boeing",
            "company_key": "boeing",
            "board_identifier": null,
            "source_kind": "careers_root",
            "source_trust": 5,
            "priority": 8,
            "attempts": 0,
            "status": "pending",
            "discovered_from": "https://www.builtin.com/company/boeing",
            "next_retry_at": null
          },
          {
            "task_key": "careers_root:https://www.builtin.com/company/boeing/jobs",
            "task_type": "careers_root",
            "url": "https://www.builtin.com/company/boeing/jobs",
            "company_name": "Boeing",
            "company_key": "boeing",
            "board_identifier": null,
            "source_kind": "careers_root",
            "source_trust": 5,
            "priority": 6,
            "attempts": 1,
            "status": "completed",
            "discovered_from": "https://www.builtin.com/company/boeing",
            "next_retry_at": null
          },
          {
            "task_key": "careers_root:https://www.builtin.com/company/comcast-3/jobs",
            "task_type": "careers_root",
            "url": "https://www.builtin.com/company/comcast-3/jobs",
            "company_name": "FreeWheel",
            "company_key": "freewheel",
            "board_identifier": null,
            "source_kind": "careers_root",
            "source_trust": 5,
            "priority": 8,
            "attempts": 0,
            "status": "pending",
            "discovered_from": "https://www.builtin.com/company/freewheel",
            "next_retry_at": null
          }
        ]
        """,
        encoding="utf-8",
    )

    tasks = load_company_discovery_frontier(tmp_path)

    assert len(tasks) == 1
    assert tasks[0]["url"] == "https://www.builtin.com/company/boeing/jobs"
    assert tasks[0]["status"] == "completed"


def test_load_company_discovery_frontier_dedupes_normalized_board_tasks(tmp_path: Path) -> None:
    frontier_path = company_discovery_frontier_path(tmp_path)
    frontier_path.write_text(
        """
        [
          {
            "task_key": "board_url:https://jobs.ashbyhq.com/butterflymx/jobs/123",
            "task_type": "board_url",
            "url": "https://jobs.ashbyhq.com/butterflymx/jobs/123",
            "company_name": "ButterflyMX",
            "company_key": "butterflymx",
            "board_identifier": null,
            "source_kind": "board_url",
            "source_trust": 10,
            "priority": 9,
            "attempts": 0,
            "status": "pending",
            "discovered_from": null,
            "next_retry_at": null
          },
          {
            "task_key": "board_url:https://jobs.ashbyhq.com/butterflymx",
            "task_type": "board_url",
            "url": "https://jobs.ashbyhq.com/butterflymx",
            "company_name": "ButterflyMX",
            "company_key": "butterflymx",
            "board_identifier": "ashby:butterflymx",
            "source_kind": "board_url",
            "source_trust": 10,
            "priority": 8,
            "attempts": 1,
            "status": "completed",
            "discovered_from": null,
            "next_retry_at": null
          }
        ]
        """,
        encoding="utf-8",
    )

    tasks = load_company_discovery_frontier(tmp_path)

    assert len(tasks) == 1
    assert tasks[0]["url"] == "https://jobs.ashbyhq.com/butterflymx"
    assert tasks[0]["board_identifier"] == "ashby:butterflymx"
    assert tasks[0]["status"] == "completed"


def test_save_company_discovery_frontier_sanitizes_directory_company_leaks(tmp_path: Path) -> None:
    save_company_discovery_frontier(
        tmp_path,
        [
            {
                "task_key": "careers_root:https://www.builtin.com/company/boeing/articles/jobs",
                "task_type": "careers_root",
                "url": "https://www.builtin.com/company/boeing/articles/jobs",
                "company_name": "Boeing",
                "company_key": "boeing",
                "board_identifier": None,
                "source_kind": "careers_root",
                "source_trust": 5,
                "priority": 8,
                "attempts": 0,
                "status": "pending",
                "discovered_from": "https://www.builtin.com/company/boeing",
                "next_retry_at": None,
            },
            {
                "task_key": "careers_root:https://www.builtin.com/company/boeing/jobs",
                "task_type": "careers_root",
                "url": "https://www.builtin.com/company/boeing/jobs",
                "company_name": "Boeing",
                "company_key": "boeing",
                "board_identifier": None,
                "source_kind": "careers_root",
                "source_trust": 5,
                "priority": 6,
                "attempts": 1,
                "status": "completed",
                "discovered_from": "https://www.builtin.com/company/boeing",
                "next_retry_at": None,
            },
            {
                "task_key": "careers_root:https://www.builtin.com/company/comcast-3/jobs",
                "task_type": "careers_root",
                "url": "https://www.builtin.com/company/comcast-3/jobs",
                "company_name": "FreeWheel",
                "company_key": "freewheel",
                "board_identifier": None,
                "source_kind": "careers_root",
                "source_trust": 5,
                "priority": 8,
                "attempts": 0,
                "status": "pending",
                "discovered_from": "https://www.builtin.com/company/freewheel",
                "next_retry_at": None,
            },
        ],
    )

    payload = json.loads(company_discovery_frontier_path(tmp_path).read_text(encoding="utf-8"))

    assert len(payload) == 1
    assert payload[0]["url"] == "https://www.builtin.com/company/boeing/jobs"
    assert payload[0]["status"] == "completed"


def test_load_company_discovery_entries_sanitizes_directory_careers_roots(tmp_path: Path) -> None:
    company_discovery_index_path(tmp_path).write_text(
        json.dumps(
            {
                "freewheel": {
                    "company_key": "freewheel",
                    "company_name": "FreeWheel",
                    "careers_roots": [
                        "https://www.builtin.com/company/freewheel/articles/jobs",
                        "https://www.builtin.com/company/comcast-3/jobs",
                    ],
                    "ats_types": [],
                    "board_identifiers": [],
                    "board_urls": [],
                    "source_hosts": ["www.builtin.com"],
                    "source_trust": 5,
                }
            }
        ),
        encoding="utf-8",
    )

    entries = load_company_discovery_entries(tmp_path)

    assert entries["freewheel"]["careers_roots"] == ["https://www.builtin.com/company/freewheel/jobs"]


def test_load_company_discovery_entries_drops_generic_portal_careers_roots(tmp_path: Path) -> None:
    company_discovery_index_path(tmp_path).write_text(
        json.dumps(
            {
                "capitalgroup": {
                    "company_key": "capitalgroup",
                    "company_name": "Capital Group",
                    "careers_roots": [
                        "https://builtin.com/careers",
                        "https://www.builtin.com/company/capital-group/jobs",
                    ],
                    "ats_types": [],
                    "board_identifiers": [],
                    "board_urls": [],
                    "source_hosts": ["builtin.com"],
                    "source_trust": 5,
                }
            }
        ),
        encoding="utf-8",
    )

    entries = load_company_discovery_entries(tmp_path)

    assert entries["capitalgroup"]["careers_roots"] == ["https://www.builtin.com/company/capital-group/jobs"]


def test_infer_careers_root_skips_generic_discovery_portal_detail_pages() -> None:
    assert infer_careers_root("https://builtin.com/job/principal-product-manager-ai/8679660") is None


def test_is_low_value_company_discovery_entry_flags_recursive_platform_only_entries() -> None:
    assert (
        is_low_value_company_discovery_entry(
            {
                "source_hosts": ["www.linkedin.com"],
                "board_identifiers": [],
                "board_urls": [],
                "official_board_lead_count": 0,
                "ai_pm_candidate_count": 0,
                "recent_fresh_role_count": 0,
            }
        )
        is True
    )
    assert (
        is_low_value_company_discovery_entry(
            {
                "source_hosts": ["www.ycombinator.com"],
                "board_identifiers": [],
                "board_urls": [],
                "official_board_lead_count": 0,
                "ai_pm_candidate_count": 0,
                "recent_fresh_role_count": 0,
            }
        )
        is False
    )
