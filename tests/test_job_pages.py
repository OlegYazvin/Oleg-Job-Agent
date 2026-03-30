import asyncio

from bs4 import BeautifulSoup
import httpx

from job_agent.job_pages import (
    _extract_ashby_fields,
    _extract_ai_context_snippets,
    _extract_greenhouse_board_job_reference,
    _extract_greenhouse_fields,
    _extract_jobscore_fields,
    _extract_jsonld_location,
    _extract_lever_fields,
    _extract_recruitee_fields,
    _extract_salary_range,
    _infer_remote_status,
    fetch_job_page,
)


LEVER_HTML = """
<html>
  <head>
    <title>Agiloft - Senior Product Manager, AI Solutions</title>
    <script type="application/ld+json">
      {"@context":"https://schema.org","@type":"JobPosting","datePosted":"2026-01-26"}
    </script>
  </head>
  <body>
    <div class="posting-categories">
      <div class="sort-by-location">United States</div>
      <div class="workplaceTypes">Remote</div>
    </div>
    <div data-qa="salary-range">$210,000 - $260,000 a year</div>
  </body>
</html>
"""

LEVER_REMOTE_TEXT_HTML = """
<html>
  <head>
    <title>Versapay - Principal Product Manager - AI/ML</title>
    <script type="application/ld+json">
      {"@context":"https://schema.org","@type":"JobPosting","datePosted":"2026-03-24"}
    </script>
  </head>
  <body>
    <div class="posting-categories">
      <div class="sort-by-location">New York, NY</div>
    </div>
    <div>Product Management - Product Management / Regular Full-Time / Remote apply for this job</div>
  </body>
</html>
"""

GREENHOUSE_HTML = """
<html>
  <head><title>page_title</title></head>
  <body>
    <script>
      window.__remixContext = {"state":{"loaderData":{"routes/$url_token_.jobs_.$job_post_id":{"jobPost":{
        "title":"Senior Product Manager, AI Platform",
        "company_name":"Acme AI",
        "job_post_location":"Remote-US",
        "published_at":"2026-03-20T10:00:00-05:00",
        "pay_ranges":[{"min":"$210,000","max":"$260,000"}],
        "content":"<p>Fully remote role working on AI products.</p>"
      }}}}};
    </script>
  </body>
</html>
"""

ASHBY_HTML = """
<html>
  <body>
    <script>
      window.__appData = {
        "organization": {"name": "Jerry.ai"},
        "posting": {
          "title": "Senior Product Manager, AI Agents and Platform",
          "locationName": "Remote - US",
          "publishedAt": "2026-03-26T19:07:17.783Z",
          "descriptionHtml": "<p>Base salary range: $210,000 - $260,000. This is a fully remote role.</p>"
        }
      }
    </script>
  </body>
</html>
"""

RECRUITEE_HTML = """
<html>
  <body>
    <script>
      var state = &quot;offers&quot;:[{&quot;remote&quot;:true,&quot;guid&quot;:&quot;abc123&quot;,&quot;id&quot;:1804373,&quot;countryCode&quot;:&quot;US&quot;,&quot;hybrid&quot;:false,&quot;city&quot;:&quot;Remote&quot;,&quot;salary&quot;:{&quot;currency&quot;:&quot;USD&quot;,&quot;max&quot;:260000,&quot;min&quot;:210000,&quot;period&quot;:&quot;year&quot;},&quot;translations&quot;:{&quot;en&quot;:{&quot;name&quot;:&quot;Senior Product Manager, AI Platform&quot;,&quot;country&quot;:&quot;United States&quot;,&quot;descriptionHtml&quot;:&quot;&lt;p&gt;Remote role&lt;/p&gt;&quot;}}}],&quot;translations&quot;:{};
    </script>
  </body>
</html>
"""

JOBSCORE_HTML = """
<html>
  <head>
    <title>Share the Product Manager – AI Solutions open at Unified in Remote, United States., powered by JobScore</title>
    <meta name="title" content="Share the Product Manager – AI Solutions open at Unified in Remote, United States." />
    <meta name="description" content="Share the Product Manager – AI Solutions open at Unified in Remote, United States." />
  </head>
  <body>
    <main>This role is fully remote and focused on AI products.</main>
  </body>
</html>
"""


def test_extract_salary_range_parses_usd_ranges() -> None:
    assert _extract_salary_range("Compensation: $210,000 - $260,000 base salary") == (
        210000,
        260000,
        "$210,000 - $260,000",
    )


def test_extract_salary_range_parses_base_salary_up_to_value() -> None:
    assert _extract_salary_range(
        "The annual base salary for this position is anticipated to be up to $180,000."
    ) == (
        180000,
        180000,
        "$180,000",
    )


def test_extract_salary_range_ignores_demographic_form_codes() -> None:
    assert _extract_salary_range("OMB Control Number 1250-0005 Expires 04/30/2026") == (None, None, None)


def test_extract_salary_range_ignores_percent_ranges() -> None:
    assert _extract_salary_range("Travel requirement is 75-80% for field visits.") == (None, None, None)


def test_extract_lever_fields_reads_salary_posted_and_remote() -> None:
    values = _extract_lever_fields(BeautifulSoup(LEVER_HTML, "html.parser"), LEVER_HTML, [])
    assert values["base_salary_min_usd"] == 210000
    assert values["base_salary_max_usd"] == 260000
    assert values["posted_date_iso"] == "2026-01-26"
    assert values["is_fully_remote"] is True


def test_extract_lever_fields_uses_body_text_for_remote_signal() -> None:
    values = _extract_lever_fields(BeautifulSoup(LEVER_REMOTE_TEXT_HTML, "html.parser"), LEVER_REMOTE_TEXT_HTML, [])
    assert values["posted_date_iso"] == "2026-03-24"
    assert values["is_fully_remote"] is True


def test_extract_greenhouse_fields_reads_salary_posted_and_remote() -> None:
    values = _extract_greenhouse_fields(GREENHOUSE_HTML, [])
    assert values["company_name"] == "Acme AI"
    assert values["base_salary_min_usd"] == 210000
    assert values["base_salary_max_usd"] == 260000
    assert values["posted_date_iso"] == "2026-03-20"
    assert values["is_fully_remote"] is True


def test_extract_ashby_fields_reads_embedded_app_data() -> None:
    values = _extract_ashby_fields(ASHBY_HTML, [])
    assert values["company_name"] == "Jerry.ai"
    assert values["role_title"] == "Senior Product Manager, AI Agents and Platform"
    assert values["posted_date_iso"] == "2026-03-26"
    assert values["base_salary_min_usd"] == 210000
    assert values["base_salary_max_usd"] == 260000
    assert values["is_fully_remote"] is True


def test_extract_recruitee_fields_reads_embedded_offer_payload() -> None:
    values = _extract_recruitee_fields(RECRUITEE_HTML, [])
    assert values["role_title"] == "Senior Product Manager, AI Platform"
    assert values["location_text"] == "Remote, United States"
    assert values["base_salary_min_usd"] == 210000
    assert values["base_salary_max_usd"] == 260000
    assert values["is_fully_remote"] is True


def test_extract_jobscore_fields_uses_meta_title_and_description() -> None:
    values = _extract_jobscore_fields(BeautifulSoup(JOBSCORE_HTML, "html.parser"), JOBSCORE_HTML, [])
    assert values["company_name"] == "Unified"
    assert values["role_title"] == "Product Manager – AI Solutions"
    assert values["location_text"] == "Remote, United States"
    assert values["is_fully_remote"] is True


def test_infer_remote_status_treats_required_in_office_as_not_remote() -> None:
    text = "Work Persona: Required in Office. Work personas include flexible, remote, or required in office."
    assert _infer_remote_status("remote", text) is False


def test_infer_remote_status_leaves_plain_us_location_ambiguous() -> None:
    assert _infer_remote_status("United States", "Senior Product Manager, AI/ML") is None


def test_infer_remote_status_treats_specific_city_location_as_not_remote() -> None:
    assert _infer_remote_status("Irvine, California, United States", "Principal, Product Manager- Data/ML") is False


def test_infer_remote_status_treats_office_location_as_not_remote_even_with_generic_remote_footer() -> None:
    text = "Fictiv is continuing to expand our remote US workforce. Applicants from these states are eligible."
    assert _infer_remote_status("Oakland, CA Office", text) is False


def test_infer_remote_status_allows_strong_remote_title_to_override_city_location() -> None:
    text = "Principal Product Manager - AI Travel (100% Remote - USA)"
    assert _infer_remote_status("New York, New York, United States", text) is True
    assert _infer_remote_status("Los Angeles, California, United States", "Sr. Product Manager - AI (Remote)") is True
    assert (
        _infer_remote_status(
            "Boston, United States",
            "Senior AI Product Manager Full-time Remote Boston , United States Business Systems",
        )
        is True
    )


def test_infer_remote_status_treats_remote_if_local_hybrid_wording_as_remote() -> None:
    text = (
        "Senior Product Manager - AI Strategy (USA - Remote) "
        "Model of Work: Hybrid if located in Houston, TX or Dallas, TX or Remote with Travel if the work location is USA - Remote"
    )
    assert _infer_remote_status("United States", text) is True


def test_extract_jsonld_location_handles_address_country_objects() -> None:
    payload = {
        "jobLocation": {
            "address": {
                "addressLocality": "Boise",
                "addressRegion": "ID",
                "addressCountry": {"name": "United States"},
            }
        }
    }
    assert _extract_jsonld_location(payload) == "Boise, ID, United States"


def test_extract_greenhouse_board_job_reference_supports_custom_company_shell_urls() -> None:
    url = "https://coreweave.com/careers/job?4638816006&board=coreweave&gh_jid=4638816006"
    html = '<!-- <script src="https://boards.greenhouse.io/embed/job_board/js?for=coreweave"></script> -->'
    assert _extract_greenhouse_board_job_reference(url, html) == ("coreweave", "4638816006")


def test_extract_ai_context_snippets_finds_ai_terms_in_body_text() -> None:
    snippets = _extract_ai_context_snippets(
        "You will lead strategy for conversational AI assistants and chatbot workflows across enterprise support experiences."
    )
    assert snippets
    assert "conversational ai" in snippets[0].lower()


def test_fetch_job_page_returns_status_zero_on_request_errors(monkeypatch) -> None:
    class FakeAsyncClient:
        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str):
            raise httpx.RequestError("dns failure", request=httpx.Request("GET", url))

    monkeypatch.setattr("job_agent.job_pages.httpx.AsyncClient", lambda **kwargs: FakeAsyncClient())

    snapshot = asyncio.run(fetch_job_page("https://careers.example.com/jobs/123"))
    assert snapshot.status_code == 0
    assert snapshot.resolved_url == "https://careers.example.com/jobs/123"
    assert "dns failure" in snapshot.text_excerpt.lower()


def test_fetch_job_page_rechecks_remote_after_location_fallback(monkeypatch) -> None:
    html = """
    <html>
      <head>
        <title>Senior AI Product Manager | Dynatrace Careers</title>
        <script type="application/ld+json">
          {
            "@context": "https://schema.org",
            "@type": "JobPosting",
            "title": "Senior AI Product Manager",
            "hiringOrganization": {"name": "Dynatrace"},
            "datePosted": "2026-03-11T13:15:21+01:00"
          }
        </script>
      </head>
      <body>
        <main>
          Senior AI Product Manager
          Full-time Remote Boston, United States
          This AI product role may collaborate with hybrid teams.
        </main>
      </body>
    </html>
    """

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
            return FakeResponse(url, html)

    monkeypatch.setattr("job_agent.job_pages.httpx.AsyncClient", lambda **kwargs: FakeAsyncClient())

    snapshot = asyncio.run(fetch_job_page("https://careers.example.com/jobs/123"))
    assert snapshot.location_text == "Remote"
    assert snapshot.is_fully_remote is True
