from job_agent.drafting import (
    _append_required_links,
    _finalize_second_order_messages,
    _template_first_order_messages,
    _template_second_order_messages,
)
from job_agent.models import JobPosting, LinkedInContact, SecondOrderIntroMessage


def build_job() -> JobPosting:
    return JobPosting(
        company_name="Acme AI",
        role_title="Senior Product Manager, AI",
        direct_job_url="https://boards.greenhouse.io/acme/jobs/123",
        ats_platform="Greenhouse",
        location_text="Remote",
        is_fully_remote=True,
        posted_date_text="2 days ago",
        posted_date_iso=None,
        base_salary_min_usd=210000,
        base_salary_max_usd=240000,
        salary_text="$210,000 - $240,000",
        evidence_notes="Direct ATS, remote, recent, salary listed.",
    )


def test_append_required_links_adds_missing_links() -> None:
    body = _append_required_links(
        "Happy to connect if helpful.",
        job_url="https://boards.greenhouse.io/acme/jobs/123",
        profile_url="https://www.linkedin.com/in/jane-doe/",
    )
    assert "Job link:" in body
    assert "Profile link:" in body


def test_finalize_second_order_messages_keeps_only_valid_connectors() -> None:
    job = build_job()
    first_order = []
    second_order = [
        LinkedInContact(
            name="Priya Patel",
            profile_url="https://www.linkedin.com/in/priya-patel/",
            connection_degree="2nd",
            connected_first_order_names=["John Smith", "Jane Doe"],
            connected_first_order_profile_urls={
                "John Smith": "https://www.linkedin.com/in/john-smith/",
                "Jane Doe": "https://www.linkedin.com/in/jane-doe/",
            },
        ),
        LinkedInContact(
            name="Alex Chen",
            profile_url="https://www.linkedin.com/in/alex-chen/",
            connection_degree="2nd",
            connected_first_order_names=["John Smith"],
            connected_first_order_profile_urls={"John Smith": "https://www.linkedin.com/in/john-smith/"},
        )
    ]
    messages = [
        SecondOrderIntroMessage(
            first_order_contact_name="John Smith",
            second_order_contact_names=["Priya Patel"],
            second_order_contact_profile_urls=["https://www.linkedin.com/in/priya-patel/"],
            message_body="Could you introduce me?",
        ),
        SecondOrderIntroMessage(
            first_order_contact_name="Someone Else",
            second_order_contact_names=["Priya Patel"],
            second_order_contact_profile_urls=["https://www.linkedin.com/in/priya-patel/"],
            message_body="Wrong connector",
        ),
    ]
    finalized = _finalize_second_order_messages(job, first_order, second_order, messages)
    assert len(finalized) == 2
    john_message = next(message for message in finalized if message.first_order_contact_name == "John Smith")
    jane_message = next(message for message in finalized if message.first_order_contact_name == "Jane Doe")
    assert "Job link:" in finalized[0].message_body
    assert john_message.first_order_contact_profile_url == "https://www.linkedin.com/in/john-smith/"
    assert john_message.second_order_contact_names == ["Priya Patel", "Alex Chen"]
    assert jane_message.second_order_contact_names == ["Priya Patel"]


def test_template_second_order_messages_uses_message_history_and_aggregates_targets() -> None:
    job = build_job()
    second_order = [
        LinkedInContact(
            name="Priya Patel",
            profile_url="https://www.linkedin.com/in/priya-patel/",
            connection_degree="2nd",
            connected_first_order_names=["John Smith"],
            connected_first_order_profile_urls={"John Smith": "https://www.linkedin.com/in/john-smith/"},
            connected_first_order_message_histories={"John Smith": ["Loved our chat about AI roadmaps last fall."]},
        ),
        LinkedInContact(
            name="Alex Chen",
            profile_url="https://www.linkedin.com/in/alex-chen/",
            connection_degree="2nd",
            connected_first_order_names=["John Smith"],
            connected_first_order_profile_urls={"John Smith": "https://www.linkedin.com/in/john-smith/"},
        )
    ]
    templated = _template_second_order_messages(job, [], second_order)
    assert len(templated) == 1
    assert "AI roadmaps" in templated[0].message_body
    assert templated[0].second_order_contact_names == ["Priya Patel", "Alex Chen"]
    assert "Alex Chen" in templated[0].message_body


def test_template_first_order_messages_asks_for_role_context_not_blunt_referral() -> None:
    job = build_job()
    contacts = [
        LinkedInContact(
            name="Jane Doe",
            profile_url="https://www.linkedin.com/in/jane-doe/",
            connection_degree="1st",
            message_history=["Loved catching up on AI hiring trends."],
        )
    ]
    templated = _template_first_order_messages(job, contacts)
    assert len(templated) == 1
    assert "sharing any perspective on the team" in templated[0].message_body
    assert "hiring team or recruiter" not in templated[0].message_body


def test_template_second_order_messages_prioritizes_long_message_history() -> None:
    job = build_job()
    second_order = [
        LinkedInContact(
            name="Priya Patel",
            profile_url="https://www.linkedin.com/in/priya-patel/",
            connection_degree="2nd",
            connected_first_order_names=["Long History"],
            connected_first_order_profile_urls={"Long History": "https://www.linkedin.com/in/long-history/"},
            connected_first_order_message_histories={
                "Long History": [f"message {index}" for index in range(12)]
            },
        ),
        LinkedInContact(
            name="Alex Chen",
            profile_url="https://www.linkedin.com/in/alex-chen/",
            connection_degree="2nd",
            connected_first_order_names=["Short History"],
            connected_first_order_profile_urls={"Short History": "https://www.linkedin.com/in/short-history/"},
            connected_first_order_message_histories={"Short History": ["message 1"]},
        ),
    ]
    templated = _template_second_order_messages(job, [], second_order)
    assert [message.first_order_contact_name for message in templated] == ["Long History", "Short History"]
