import sqlite3
import asyncio
from pathlib import Path
from types import SimpleNamespace

from job_agent.linkedin import (
    LinkedInClient,
    _company_matches,
    _default_firefox_profile_dir,
    _load_firefox_linkedin_cookies,
    _load_linkedin_auth_cookies,
    _load_linkedin_storage_state_cookies,
    _normalize_cookie_expiry,
    build_manual_review_links,
)
from job_agent.linkedin_extension_bridge import ExtensionCaptureSession


def test_normalize_profile_url_handles_relative_links() -> None:
    assert (
        LinkedInClient._normalize_profile_url("/in/jane-doe/?miniProfileUrn=abc")
        == "https://www.linkedin.com/in/jane-doe/"
    )


def test_extract_mutual_connection_names_cleans_text() -> None:
    text = "John Smith and Priya Patel are mutual connections"
    assert LinkedInClient._extract_mutual_connection_names(text) == ["John Smith", "Priya Patel"]


def test_extract_mutual_connection_names_skips_count_placeholders() -> None:
    text = "Ryan Roche and 10 other mutual connections"
    assert LinkedInClient._extract_mutual_connection_names(text) == ["Ryan Roche"]


def test_company_matches_ignores_corporate_suffixes() -> None:
    assert _company_matches("Acme AI, Inc.", "Senior PM at Acme AI")


def test_company_matches_requires_token_match_for_one_word_company_names() -> None:
    assert _company_matches("January", "Senior Product Manager at January")
    assert not _company_matches("January", "Founder @JanuarysAdvisoryGroup")


def test_manual_review_links_are_company_focused_and_include_degrees() -> None:
    links = build_manual_review_links("Acme AI", "Senior Product Manager, AI")
    assert len(links) >= 4
    assert all("linkedin.com/search/results/people" in link.url for link in links)
    assert any("network=%5B%22F%22%5D" in link.url for link in links)
    assert any("network=%5B%22S%22%5D" in link.url for link in links)
    assert any("Acme%20AI" in link.url for link in links)


def test_default_firefox_profile_dir_prefers_install_default(tmp_path: Path) -> None:
    firefox_root = tmp_path / "firefox"
    firefox_root.mkdir()
    install_profile = firefox_root / "install-default"
    install_profile.mkdir()
    fallback_profile = firefox_root / "fallback-default"
    fallback_profile.mkdir()
    (firefox_root / "profiles.ini").write_text(
        "\n".join(
            [
                "[Profile0]",
                "Name=default-release",
                "IsRelative=1",
                "Path=fallback-default",
                "Default=1",
                "",
                "[InstallABC]",
                "Default=install-default",
                "Locked=1",
            ]
        )
    )
    assert _default_firefox_profile_dir(firefox_root) == install_profile


def test_load_firefox_linkedin_cookies_reads_copied_cookie_db(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    db_path = profile_dir / "cookies.sqlite"
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute(
        """
        create table moz_cookies (
            id integer primary key,
            originAttributes text not null default '',
            name text,
            value text,
            host text,
            path text,
            expiry integer,
            lastAccessed integer default 0,
            creationTime integer default 0,
            isSecure integer,
            isHttpOnly integer,
            inBrowserElement integer default 0,
            sameSite integer default 0,
            rawSameSite integer default 0,
            schemeMap integer default 0
        )
        """
    )
    cursor.execute(
        """
        insert into moz_cookies (name, value, host, path, expiry, isSecure, isHttpOnly)
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        ("li_at", "abc123", ".linkedin.com", "/", 2000000000, 1, 1),
    )
    cursor.execute(
        """
        insert into moz_cookies (name, value, host, path, expiry, isSecure, isHttpOnly)
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        ("other", "ignore", ".example.com", "/", 2000000000, 0, 0),
    )
    connection.commit()
    connection.close()

    cookies = _load_firefox_linkedin_cookies(profile_dir)
    assert len(cookies) == 1
    assert cookies[0]["name"] == "li_at"
    assert cookies[0]["domain"] == ".linkedin.com"


def test_load_linkedin_storage_state_cookies_reads_linkedin_entries(tmp_path: Path) -> None:
    storage_state_path = tmp_path / "linkedin-state.json"
    storage_state_path.write_text(
        """
        {
          "cookies": [
            {
              "name": "li_at",
              "value": "abc123",
              "domain": ".linkedin.com",
              "path": "/",
              "expires": 2000000000,
              "httpOnly": true,
              "secure": true,
              "sameSite": "Lax"
            },
            {
              "name": "ignored",
              "value": "nope",
              "domain": ".example.com",
              "path": "/",
              "expires": 2000000000,
              "httpOnly": false,
              "secure": false
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    cookies = _load_linkedin_storage_state_cookies(storage_state_path)

    assert len(cookies) == 1
    assert cookies[0]["name"] == "li_at"
    assert cookies[0]["domain"] == ".linkedin.com"
    assert cookies[0]["sameSite"] == "Lax"


def test_load_linkedin_auth_cookies_prefers_env_then_storage_then_firefox(tmp_path: Path, monkeypatch) -> None:
    storage_state_path = tmp_path / "linkedin-state.json"
    storage_state_path.write_text(
        """
        {
          "cookies": [
            {
              "name": "li_at",
              "value": "from-storage",
              "domain": ".linkedin.com",
              "path": "/",
              "expires": 2000000000,
              "httpOnly": true,
              "secure": true
            },
            {
              "name": "storage-only",
              "value": "keep-me",
              "domain": ".www.linkedin.com",
              "path": "/",
              "expires": 2000000000,
              "httpOnly": false,
              "secure": true
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "job_agent.linkedin._load_firefox_linkedin_cookies",
        lambda profile_dir=None: [
            {
                "name": "li_at",
                "value": "from-firefox",
                "domain": ".linkedin.com",
                "path": "/",
                "expires": 2000000000,
                "httpOnly": True,
                "secure": True,
            },
            {
                "name": "firefox-only",
                "value": "keep-too",
                "domain": ".linkedin.com",
                "path": "/",
                "expires": 2000000000,
                "httpOnly": True,
                "secure": True,
            },
        ],
    )
    settings = SimpleNamespace(
        linkedin_li_at="from-env",
        linkedin_jsessionid=None,
        linkedin_storage_state=storage_state_path,
    )

    cookies = _load_linkedin_auth_cookies(settings)

    by_name = {cookie["name"]: cookie for cookie in cookies}
    assert by_name["li_at"]["value"] == "from-env"
    assert by_name["storage-only"]["value"] == "keep-me"
    assert by_name["firefox-only"]["value"] == "keep-too"


def test_normalize_cookie_expiry_handles_milliseconds() -> None:
    assert _normalize_cookie_expiry(1805342663000) == 1805342663
    assert _normalize_cookie_expiry(1805342663) == 1805342663
    assert _normalize_cookie_expiry(0) == -1


def test_finalize_parsed_search_result_extracts_mutual_connectors() -> None:
    client = LinkedInClient(SimpleNamespace(max_linkedin_results_per_company=10))
    parsed = client._finalize_parsed_search_result(
        {
            "href": "https://www.linkedin.com/in/heather-corbett-69763950/?trk=abc",
            "text": "\n".join(
                [
                    "Heather Corbett",
                    "• 2nd",
                    "Principal Talent Acquisition Advisor at Dynatrace",
                    "Detroit Metropolitan Area",
                    "Tatyana Smirnova and 14 other mutual connections",
                ]
            ),
            "mutuals": [("Tatyana Smirnova", "https://www.linkedin.com/in/tanyaspm/?trk=abc")],
        }
    )
    assert parsed is not None
    assert parsed.name == "Heather Corbett"
    assert parsed.profile_url == "https://www.linkedin.com/in/heather-corbett-69763950/"
    assert parsed.company_text == "Principal Talent Acquisition Advisor at Dynatrace"
    assert parsed.mutual_connection_profile_urls["Tatyana Smirnova"] == "https://www.linkedin.com/in/tanyaspm/"


class _FakePage:
    async def eval_on_selector_all(self, selector: str, script: str):
        assert selector == 'a[href*="/in/"]'
        return [
            {
                "href": "https://www.linkedin.com/in/heather-corbett-69763950/?trk=abc",
                "text": "\n".join(
                    [
                        "Heather Corbett",
                        "• 2nd",
                        "Principal Talent Acquisition Advisor at Dynatrace",
                        "Detroit Metropolitan Area",
                        "Tatyana Smirnova and 14 other mutual connections",
                    ]
                ),
            },
            {
                "href": "https://www.linkedin.com/in/heather-corbett-69763950/?trk=dup",
                "text": "Heather Corbett",
            },
            {
                "href": "https://www.linkedin.com/in/tanyaspm/?trk=dup",
                "text": "Tatyana Smirnova",
            },
        ]


def test_parse_search_result_anchors_groups_mutual_profile_links() -> None:
    client = LinkedInClient(SimpleNamespace(max_linkedin_results_per_company=10))
    results = asyncio.run(client._parse_search_result_anchors(_FakePage()))
    assert len(results) == 1
    assert results[0].name == "Heather Corbett"
    assert results[0].mutual_connection_profile_urls == {
        "Tatyana Smirnova": "https://www.linkedin.com/in/tanyaspm/"
    }


class _FakePageWithTrailingAnchorArtifacts:
    async def eval_on_selector_all(self, selector: str, script: str):
        assert selector == 'a[href*="/in/"]'
        return [
            {
                "href": "https://www.linkedin.com/in/tania-nemes/?trk=abc",
                "text": "\n".join(
                    [
                        "Tania Nemes",
                        "• 2nd",
                        "Principal Talent Acquisition Advisor at Dynatrace",
                        "Ryan Roche and 10 other mutual connections",
                    ]
                ),
            },
            {
                "href": "https://www.linkedin.com/in/tania-nemes/?trk=dup",
                "text": "Tania Nemes",
            },
            {
                "href": "https://www.linkedin.com/in/ryanroche17/?trk=dup",
                "text": "Ryan Roche  a",
            },
        ]


def test_parse_search_result_anchors_cleans_trailing_anchor_artifacts() -> None:
    client = LinkedInClient(SimpleNamespace(max_linkedin_results_per_company=10))
    results = asyncio.run(client._parse_search_result_anchors(_FakePageWithTrailingAnchorArtifacts()))
    assert len(results) == 1
    assert results[0].mutual_connection_names == ["Ryan Roche"]
    assert results[0].mutual_connection_profile_urls == {
        "Ryan Roche": "https://www.linkedin.com/in/ryanroche17/"
    }


class _FakeSearchPageWithNamedMutuals:
    async def eval_on_selector_all(self, selector: str, script: str):
        assert selector == 'a[href*="/in/"]'
        return [
            {
                "href": "https://www.linkedin.com/in/heather-corbett-69763950/?trk=abc",
                "text": "\n".join(
                    [
                        "Heather Corbett",
                        "• 2nd",
                        "Principal Talent Acquisition Advisor at Dynatrace",
                        "Detroit Metropolitan Area",
                        "Tatyana Smirnova and Priya Patel are mutual connections",
                    ]
                ),
            },
            {
                "href": "https://www.linkedin.com/in/heather-corbett-69763950/?trk=dup",
                "text": "Heather Corbett",
            },
            {
                "href": "https://www.linkedin.com/in/tanyaspm/?trk=dup",
                "text": "Tatyana Smirnova",
            },
        ]


def test_collect_contacts_from_search_results_keeps_named_mutuals_without_profile_links() -> None:
    client = LinkedInClient(SimpleNamespace(max_linkedin_results_per_company=10))
    contacts = asyncio.run(
        client._collect_contacts_from_search_results(
            _FakeSearchPageWithNamedMutuals(),
            "Dynatrace",
            "2nd",
        )
    )
    assert len(contacts) == 1
    assert contacts[0].connected_first_order_names == ["Tatyana Smirnova", "Priya Patel"]
    assert contacts[0].connected_first_order_profile_urls == {
        "Tatyana Smirnova": "https://www.linkedin.com/in/tanyaspm/"
    }


def test_build_discovery_from_extension_session_rejects_surname_only_company_match() -> None:
    client = LinkedInClient(SimpleNamespace(linkedin_capture_mode="firefox_extension"))
    session = ExtensionCaptureSession(
        session_id="session-123",
        company_name="Hopper",
        search_urls={},
    )
    session.second_order_contacts = [
        {
            "name": "Mark Hopper",
            "profile_url": "https://www.linkedin.com/in/mark-hopper/",
            "raw_text": "Mark Hopper\n• 2nd\nSr. Product Manager Architect @ Microsoft",
            "headline": "Sr. Product Manager Architect @ Microsoft",
            "company_text": "Sr. Product Manager Architect @ Microsoft",
            "connected_first_order_names": ["John Smith"],
            "connected_first_order_profile_urls": {
                "John Smith": "https://www.linkedin.com/in/john-smith/"
            },
        }
    ]

    discovery = client._build_discovery_from_extension_session("Hopper", session)

    assert discovery.second_order_contacts == []


def test_build_discovery_from_extension_session_keeps_real_one_word_company_match() -> None:
    client = LinkedInClient(SimpleNamespace(linkedin_capture_mode="firefox_extension"))
    session = ExtensionCaptureSession(
        session_id="session-123",
        company_name="January",
        search_urls={},
    )
    session.second_order_contacts = [
        {
            "name": "Alex Smith",
            "profile_url": "https://www.linkedin.com/in/alex-smith/",
            "raw_text": "Alex Smith\n• 2nd\nSenior Product Manager at January",
            "headline": "Senior Product Manager at January",
            "company_text": "Senior Product Manager at January",
            "connected_first_order_names": ["John Smith"],
            "connected_first_order_profile_urls": {
                "John Smith": "https://www.linkedin.com/in/john-smith/"
            },
        }
    ]

    discovery = client._build_discovery_from_extension_session("January", session)

    assert len(discovery.second_order_contacts) == 1
    assert discovery.second_order_contacts[0].name == "Alex Smith"


def test_build_discovery_from_extension_session_rejects_skill_only_company_mentions() -> None:
    client = LinkedInClient(SimpleNamespace(linkedin_capture_mode="firefox_extension"))
    session = ExtensionCaptureSession(
        session_id="session-123",
        company_name="Dynatrace",
        search_urls={},
    )
    session.second_order_contacts = [
        {
            "name": "Aman G.",
            "profile_url": "https://www.linkedin.com/in/aman-g/",
            "raw_text": "Aman G.\n• 2nd\nSoftware Engineer | AWS Certified | Python, AI/ML, Dynatrace",
            "headline": "Software Engineer | AWS Certified | Python, AI/ML, Dynatrace",
            "company_text": "Software Engineer | AWS Certified | Python, AI/ML, Dynatrace",
            "connected_first_order_names": ["John Smith"],
            "connected_first_order_profile_urls": {
                "John Smith": "https://www.linkedin.com/in/john-smith/"
            },
        },
        {
            "name": "Rob V.",
            "profile_url": "https://www.linkedin.com/in/rob-v/",
            "raw_text": "Rob V.\n• 2nd\nProduct @ Dynatrace",
            "headline": "Product @ Dynatrace",
            "company_text": "Product @ Dynatrace",
            "connected_first_order_names": ["John Smith"],
            "connected_first_order_profile_urls": {
                "John Smith": "https://www.linkedin.com/in/john-smith/"
            },
        },
    ]

    discovery = client._build_discovery_from_extension_session("Dynatrace", session)

    assert [contact.name for contact in discovery.second_order_contacts] == ["Rob V."]


def test_build_discovery_from_extension_session_uses_mutual_names_when_profile_urls_are_missing() -> None:
    client = LinkedInClient(SimpleNamespace(linkedin_capture_mode="firefox_extension"))
    session = ExtensionCaptureSession(
        session_id="session-123",
        company_name="Dynatrace",
        search_urls={},
    )
    session.second_order_contacts = [
        {
            "name": "Heather Corbett",
            "profile_url": "https://www.linkedin.com/in/heather-corbett-69763950/",
            "raw_text": "Heather Corbett\n• 2nd\nPrincipal Talent Acquisition Advisor at Dynatrace",
            "headline": "Principal Talent Acquisition Advisor at Dynatrace",
            "company_text": "Principal Talent Acquisition Advisor at Dynatrace",
            "mutual_connection_names": ["Tatyana Smirnova", "Priya Patel"],
            "connected_first_order_profile_urls": {
                "Tatyana Smirnova": "https://www.linkedin.com/in/tanyaspm/"
            },
        }
    ]

    discovery = client._build_discovery_from_extension_session("Dynatrace", session)

    assert len(discovery.second_order_contacts) == 1
    assert discovery.second_order_contacts[0].connected_first_order_names == [
        "Tatyana Smirnova",
        "Priya Patel",
    ]
