from pathlib import Path

from job_agent.company_discovery import (
    extract_embedded_board_urls,
    extract_directory_company_tasks,
    load_company_discovery_entries,
    load_company_discovery_frontier,
    make_frontier_task,
    save_company_discovery_entries,
    select_frontier_tasks,
    is_low_value_company_discovery_entry,
    upsert_company_discovery_entry,
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
