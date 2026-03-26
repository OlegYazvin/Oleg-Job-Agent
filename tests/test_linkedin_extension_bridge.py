from pathlib import Path
from types import SimpleNamespace

import job_agent.linkedin_extension_bridge as bridge_module
from job_agent.linkedin import LinkedInClient
from job_agent.linkedin_extension_bridge import ExtensionCaptureSession, build_extension_capture_urls


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
