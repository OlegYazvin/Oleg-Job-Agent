from pathlib import Path

from job_agent.company_discovery import (
    extract_embedded_board_urls,
    load_company_discovery_entries,
    make_frontier_task,
    save_company_discovery_entries,
    select_frontier_tasks,
    upsert_company_discovery_entry,
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
