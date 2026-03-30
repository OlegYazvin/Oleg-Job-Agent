from pathlib import Path
from types import SimpleNamespace
import asyncio

import job_agent.linkedin_extension_bridge as bridge_module
from job_agent.linkedin import LinkedInClient
from job_agent.linkedin_extension_bridge import (
    ExtensionCaptureSession,
    LinkedInExtensionBridge,
    build_extension_capture_urls,
)


def test_build_extension_capture_urls_include_session_metadata() -> None:
    urls = build_extension_capture_urls("Acme AI", "session-123", "Senior Product Manager, AI")
    assert len(urls["1st"]) >= 3
    assert all("job_agent_session=session-123" in url for url in urls["1st"])
    assert any("job_agent_degree=1st" in url for url in urls["1st"])
    assert any("job_agent_degree=2nd" in url for url in urls["2nd"])
    assert any("Acme%20AI" in url for url in urls["1st"])


def test_build_discovery_from_extension_session_merges_histories() -> None:
    client = LinkedInClient(SimpleNamespace(linkedin_capture_mode="firefox_extension"))
    session = ExtensionCaptureSession(
        session_id="session-123",
        company_name="Acme AI",
        search_urls={},
    )
    session.first_order_contacts = [
        {
            "name": "John Smith",
            "profile_url": "https://www.linkedin.com/in/john-smith/",
            "raw_text": "John Smith\n• 1st\nSenior PM at Acme AI",
            "headline": "Senior PM at Acme AI",
            "company_text": "Senior PM at Acme AI",
        }
    ]
    session.second_order_contacts = [
        {
            "name": "Priya Patel",
            "profile_url": "https://www.linkedin.com/in/priya-patel/",
            "raw_text": "Priya Patel\n• 2nd\nAI Product Lead at Acme AI",
            "headline": "AI Product Lead at Acme AI",
            "company_text": "AI Product Lead at Acme AI",
            "connected_first_order_names": ["John Smith"],
            "connected_first_order_profile_urls": {
                "John Smith": "https://www.linkedin.com/in/john-smith/"
            },
        }
    ]
    session.message_histories_by_name["johnsmith"] = ["Great to reconnect on AI products."]

    discovery = client._build_discovery_from_extension_session("Acme AI", session)

    assert len(discovery.first_order_contacts) == 1
    assert discovery.first_order_contacts[0].message_history == ["Great to reconnect on AI products."]
    assert len(discovery.second_order_contacts) == 1
    assert (
        discovery.second_order_contacts[0].connected_first_order_message_histories["John Smith"]
        == ["Great to reconnect on AI products."]
    )


def test_open_url_in_firefox_prefers_running_extension_host_queue(monkeypatch, tmp_path: Path) -> None:
    enqueued: list[str] = []

    monkeypatch.setattr(
        bridge_module,
        "load_settings",
        lambda project_root, require_openai=False: SimpleNamespace(firefox_extension_profile_dir=None),
    )
    monkeypatch.setattr(
        bridge_module,
        "read_firefox_extension_host_state",
        lambda project_root: {"pid": 4242, "profile_dir": str(tmp_path / "profile")},
    )
    monkeypatch.setattr(bridge_module, "enqueue_open_url", lambda project_root, url: enqueued.append(url))
    monkeypatch.setattr(bridge_module.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(
        bridge_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess fallback should not be used")),
    )
    monkeypatch.setattr(
        bridge_module.webbrowser,
        "open_new_tab",
        lambda url: (_ for _ in ()).throw(AssertionError("browser fallback should not be used")),
    )

    bridge_module._open_url_in_firefox("https://www.linkedin.com/search/results/people/", tmp_path)

    assert enqueued == ["https://www.linkedin.com/search/results/people/"]


def test_open_url_in_firefox_prefers_configured_real_profile(monkeypatch, tmp_path: Path) -> None:
    profile_dir = tmp_path / "real-profile"
    profile_dir.mkdir()
    opened: list[tuple[Path, str]] = []

    monkeypatch.setattr(
        bridge_module,
        "load_settings",
        lambda project_root, require_openai=False: SimpleNamespace(firefox_extension_profile_dir=profile_dir),
    )
    monkeypatch.setattr(
        bridge_module,
        "open_url_in_firefox_profile",
        lambda profile, url: opened.append((profile, url)) or True,
    )
    monkeypatch.setattr(
        bridge_module,
        "read_firefox_extension_host_state",
        lambda project_root: (_ for _ in ()).throw(AssertionError("host state fallback should not be used")),
    )

    bridge_module._open_url_in_firefox("https://www.linkedin.com/search/results/people/", tmp_path)

    assert opened == [(profile_dir, "https://www.linkedin.com/search/results/people/")]


def test_wait_for_search_results_returns_when_login_required() -> None:
    bridge = LinkedInExtensionBridge(
        SimpleNamespace(
            linkedin_extension_bridge_host="127.0.0.1",
            linkedin_extension_bridge_port=8765,
            linkedin_extension_capture_timeout_seconds=30,
            linkedin_extension_history_timeout_seconds=0,
            linkedin_extension_auto_open_search_tabs=False,
            data_dir=Path("."),
        )
    )
    session = ExtensionCaptureSession(session_id="session-123", company_name="Acme AI", search_urls={})
    session.login_required = True

    asyncio.run(bridge.wait_for_search_results(session))

    assert session.login_required is True


def test_open_search_tabs_fails_fast_when_configured_profile_is_missing_extension(
    monkeypatch,
    tmp_path: Path,
) -> None:
    bridge = LinkedInExtensionBridge(
        SimpleNamespace(
            linkedin_extension_bridge_host="127.0.0.1",
            linkedin_extension_bridge_port=8765,
            linkedin_extension_capture_timeout_seconds=30,
            linkedin_extension_history_timeout_seconds=0,
            linkedin_extension_auto_open_search_tabs=True,
            firefox_extension_profile_dir=tmp_path / "real-profile",
            data_dir=tmp_path,
            project_root=tmp_path,
        )
    )
    session = ExtensionCaptureSession(
        session_id="session-123",
        company_name="Acme AI",
        search_urls={"1st": ["https://example.com/1st"], "2nd": ["https://example.com/2nd"]},
    )
    monkeypatch.setattr(
        bridge_module,
        "inspect_configured_firefox_extension_profile",
        lambda settings: {
            "configured": True,
            "path": str(tmp_path / "real-profile"),
            "exists": True,
            "extension_installed": False,
            "linkedin_authenticated": True,
            "ready": False,
        },
    )

    try:
        asyncio.run(bridge.open_search_tabs(session))
    except RuntimeError as exc:
        assert "does not currently have the Job Agent LinkedIn Bridge add-on loaded" in str(exc)
    else:
        raise AssertionError("Expected open_search_tabs to fail fast for a missing extension")


def test_open_search_tabs_primes_feed_before_people_search(monkeypatch, tmp_path: Path) -> None:
    opened_urls: list[str] = []
    bridge = LinkedInExtensionBridge(
        SimpleNamespace(
            linkedin_extension_bridge_host="127.0.0.1",
            linkedin_extension_bridge_port=8765,
            linkedin_extension_capture_timeout_seconds=30,
            linkedin_extension_history_timeout_seconds=0,
            linkedin_extension_auto_open_search_tabs=True,
            firefox_extension_profile_dir=tmp_path / "real-profile",
            data_dir=tmp_path,
            project_root=tmp_path,
        )
    )
    session = ExtensionCaptureSession(
        session_id="session-123",
        company_name="Acme AI",
        search_urls={"1st": ["https://example.com/1st"], "2nd": ["https://example.com/2nd"]},
    )
    monkeypatch.setattr(
        bridge_module,
        "inspect_configured_firefox_extension_profile",
        lambda settings: {
            "configured": True,
            "path": str(tmp_path / "real-profile"),
            "exists": True,
            "extension_installed": True,
            "linkedin_authenticated": True,
            "ready": True,
        },
    )
    monkeypatch.setattr(bridge_module, "_open_url_in_firefox", lambda url, project_root: opened_urls.append(url))

    asyncio.run(bridge.open_search_tabs(session))

    assert opened_urls[0] == "https://www.linkedin.com/feed/"
    assert opened_urls[1:] == ["https://example.com/1st", "https://example.com/2nd"]


def test_discover_company_contacts_via_extension_retries_once_after_login_redirect() -> None:
    client = LinkedInClient(SimpleNamespace(linkedin_capture_mode="firefox_extension"))

    class FakeBridge:
        def __init__(self) -> None:
            self.created = 0
            self.opened_session_ids: list[str] = []
            self.retried = False

        async def create_session(self, company_name: str, *, role_title: str | None = None):
            self.created += 1
            session = ExtensionCaptureSession(
                session_id=f"session-{self.created}",
                company_name=company_name,
                role_title=role_title,
                search_urls={},
            )
            if self.created >= 2:
                session.first_order_contacts = [
                    {
                        "name": "Ryan Roche",
                        "profile_url": "https://www.linkedin.com/in/ryan-roche/",
                        "raw_text": "Ryan Roche\n• 1st\nSenior PM at Acme AI",
                        "headline": "Senior PM at Acme AI",
                        "company_text": "Senior PM at Acme AI",
                    }
                ]
                session.received_degrees = {"1st", "2nd"}
            return session

        async def open_search_tabs(self, session, *, prime_linkedin_feed: bool = True):
            self.opened_session_ids.append(session.session_id)

        async def wait_for_search_results(self, session) -> None:
            if session.session_id == "session-1":
                session.login_required = True

        async def retry_session_after_login_redirect(self, company_name: str, *, role_title: str | None = None):
            self.retried = True
            session = await self.create_session(company_name, role_title=role_title)
            await self.open_search_tabs(session, prime_linkedin_feed=True)
            return session

        async def wait_for_history_settle(self, session) -> None:
            return None

    discovery = asyncio.run(
        client._discover_company_contacts_via_extension(
            FakeBridge(),
            "Acme AI",
            role_title="Senior Product Manager, AI",
        )
    )

    assert len(discovery.first_order_contacts) == 1
    assert discovery.first_order_contacts[0].name == "Ryan Roche"
